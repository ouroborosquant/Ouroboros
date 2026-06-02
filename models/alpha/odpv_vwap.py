"""
FORTRESS v5 — models/alpha/odpv_vwap.py  [v1.0 — ODPV]

Dynamic VWAP Oscillator (ODPV)
==============================

Economic Thesis
---------------
Institutional portfolio managers building or reducing multi-billion-dollar
positions are legally constrained to VWAP execution to demonstrate best
execution under MiFID II / SEC Rule 10b-5 obligations. A 21-day rolling VWAP
therefore accumulates the weighted centroid of all institutional order flow
over the prior month.

If the current close rests materially *above* the 21-day VWAP, the math is
deterministic: heavier volume occurred on upticks than downticks over that
window. This is the unhideable footprint of institutional net-buying pressure
that cannot be disguised by TWAP slicing.

Critically, the spread is normalised to its own historical distribution via a
252-day trailing Z-score before being bounded. This converts the absolute
basis-point spread into a regime-relative signal: an identical +0.5% spread
in 2020 (volatile) versus 2024 (compressed) carries very different information.

Mathematical Formulation
------------------------
VWAP_21,t = Σ_{k=0}^{20} (C_{t-k} · V_{t-k}) / Σ_{k=0}^{20} V_{t-k}

Spread_t = (C_t / VWAP_21,t) − 1

Signal = tanh( (Spread_t − μ_252) / (σ_252 · 3.0) )

Implementation Notes
--------------------
• Volume arrays must be monotonic-non-negative. Adjusted-volume series from
  yfinance occasionally produce zeros on split-adjusted days; these are
  forward-filled (not dropped) to avoid denominator collapse.
• The rolling VWAP is computed via vectorised rolling sum of (price × volume)
  and rolling sum of volume — two O(T·N) passes, no Python loops.
• Cross-sectional behaviour: ODPV is a *time-series* signal (Z-scored against
  own history), not a cross-sectional rank. The GAT router will blend it with
  cross-sectionally normalised signals; this is intentional — the diversity
  in normalisation space improves portfolio-level IC stability.
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger("Ouroboros.ODPV")

_VWAP_WINDOW:    int   = 21
_ZSCORE_WINDOW:  int   = 252
_TANH_SCALE:     float = 3.0
_MIN_VOL_FLOOR:  float = 1.0       # shares; prevents divide-by-zero on zero-volume days
_MIN_SIGMA:      float = 1e-6


class ODPVEngine:
    """
    Dynamic VWAP Oscillator signal engine.

    Parameters
    ----------
    tickers : List[str]
        Universe in canonical order.
    vwap_window : int
        Rolling window for VWAP calculation (default 21 trading days ≈ 1 month).
    zscore_window : int
        Trailing window for cross-time Z-score (default 252 ≈ 1 year).
    """

    def __init__(
        self,
        tickers:      List[str],
        vwap_window:  int   = _VWAP_WINDOW,
        zscore_window: int  = _ZSCORE_WINDOW,
        tanh_scale:   float = _TANH_SCALE,
    ) -> None:
        self.tickers       = tickers
        self.vwap_window   = vwap_window
        self.zscore_window = zscore_window
        self.tanh_scale    = tanh_scale

    def compute_signal(
        self,
        close_df:  pd.DataFrame,
        volume_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        close_df  : (T, N) adjusted daily closes.
        volume_df : (T, N) daily consolidated volume.
                    Must align perfectly with close_df on the date index.

        Returns
        -------
        pd.DataFrame  (T, N)  ∈ (−1, 1)
        """
        close  = close_df.reindex(columns=self.tickers).astype(np.float64)
        volume = volume_df.reindex(columns=self.tickers).astype(np.float64)

        # ── Guard: zero-volume days produce undefined VWAP ───────────────────
        # Forward-fill rather than drop: preserves index alignment with close.
        # Replace genuine zeros with the floor so rolling sums stay bounded.
        volume = volume.replace(0.0, np.nan).ffill().fillna(_MIN_VOL_FLOOR)
        volume = volume.clip(lower=_MIN_VOL_FLOOR)

        # ── Rolling VWAP via cumulative numerator / denominator ──────────────
        # This is exact (not approximate) for a rectangular rolling window
        # because rolling().sum() is an O(1)-per-step sliding-window kernel.
        dv      = close * volume                          # dollar-volume, (T, N)
        min_p   = max(3, self.vwap_window // 7)          # warm-up floor: ~3 bars
        roll_dv = dv.rolling(self.vwap_window, min_periods=min_p).sum()
        roll_v  = volume.rolling(self.vwap_window, min_periods=min_p).sum()

        vwap = roll_dv / roll_v.clip(lower=_MIN_VOL_FLOOR)

        # ── Percentage spread: + means price > VWAP → institutional net buy ──
        spread = (close / vwap) - 1.0

        # ── Cross-time Z-score ───────────────────────────────────────────────
        roll = spread.rolling(self.zscore_window, min_periods=max(63, self.zscore_window // 4))
        mu   = roll.mean()
        sig  = roll.std().clip(lower=_MIN_SIGMA)

        # Smooth Hyperbolic Volatility Saturation (Fix 3)
        expanding_median = sig.expanding(min_periods=63).median()
        cap_limit = 3.0 * expanding_median.replace(0, np.nan).ffill()
        
        # Smoothly saturate using x_capped = cap * tanh(x / cap)
        valid_mask = cap_limit.notna() & (cap_limit > 0)
        sig = pd.DataFrame(
            np.where(valid_mask, cap_limit * np.tanh(sig / cap_limit), sig),
            index=sig.index,
            columns=sig.columns
        )

        z      = (spread - mu) / (sig * self.tanh_scale)
        signal = np.tanh(z).astype(np.float32)

        result = pd.DataFrame(signal, index=close.index, columns=self.tickers)

        logger.info(
            f"ODPV-VWAP | shape={result.shape} | "
            f"mean|α|={result.abs().mean().mean():.4f} | "
            f"non-zero={(result.abs() > 0.01).any(axis=1).mean()*100:.1f}%"
        )
        return result