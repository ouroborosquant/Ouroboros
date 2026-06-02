"""
FORTRESS v5 — models/alpha/dtfe_trend.py  [v1.0 — DTFE]

Directional Trend Fractal Efficiency (DTFE)
==========================================

Economic Thesis
---------------
Not all price trends carry the same information about institutional conviction.
A retail-speculation-driven rally is characterised by high-amplitude intraday
oscillations, frequent reversals within the trend, and large daily swings —
high "friction" in the price path. A sovereign wealth fund passively
rebalancing a $50B equity portfolio generates an almost perfectly linear price
trajectory because their size forces VWAP participation over weeks, ironing
out all but the most exogenous shocks.

Kaufman's Efficiency Ratio (KER) is the canonical measure of this fractal
geometry:

  KER = |Net Displacement over window| / Path Length (sum of |daily changes|)

  • KER → 1.0: perfectly straight-line trend (Euclidean-optimal path)
  • KER → 0.0: random walk / sideways chop (Brownian motion)

DTFE makes KER directional by multiplying by sign(return_21d), producing
a signal that is:
  • Large positive: fast, linear, upward trend   → maximum long conviction
  • Large negative: fast, linear, downward trend → maximum short conviction
  • Near zero:      choppy / sideways            → zero allocation (noise filter)

This asymmetry is precisely what differentiates institutional accumulation
from retail momentum: institutional flows are smooth, monotonic, and
directional; retail flows are high-KER on both sides simultaneously and wash
out cross-sectionally.

Mathematical Formulation
------------------------
Net_Change_t = |C_t − C_{t−21}|
Path_Length_t = Σ_{k=0}^{20} |C_{t−k} − C_{t−k−1}|
KER_t = Net_Change_t / Path_Length_t      ∈ [0, 1]
DTFE_t = KER_t × sign(C_t − C_{t−21})    ∈ [−1, +1]
Signal = tanh( (DTFE_t − μ_252) / (σ_252 · 3.0) )

Vectorised Path Length
----------------------
|C_{t-k} - C_{t-k-1}| for k=0..20 is equivalent to the rolling sum of
|R_t| over a 21-bar window:  path_length_t = Σ_{j=t-20}^{t} |C_j - C_{j-1}|
This is a single .rolling(21).sum() on abs(close.diff()) — O(T·N).

Edge Cases
----------
• Path length == 0 (all prices identical in window → synthetic/bad data):
  KER = 0.0; DTFE = 0.0; signal = 0.0. Guarded via clip(lower=ε).
• NaN propagation: first 21+252 = 273 bars will produce NaN; precompute
  script fillna(0.0) handles this consistently with other engines.
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger("Ouroboros.DTFE")

_KER_WINDOW:     int   = 21
_ZSCORE_WINDOW:  int   = 252
_TANH_SCALE:     float = 3.0
_MIN_PATH:       float = 1e-10   # floor on path length to prevent KER blow-up
_MIN_SIGMA:      float = 1e-6


class DTFETrendEngine:
    """
    Directional Trend Fractal Efficiency signal engine.

    Parameters
    ----------
    tickers : List[str]
        Universe in canonical order.
    ker_window : int
        Window for KER computation (21 trading days ≈ 1 month).
    zscore_window : int
        Trailing window for cross-time Z-score (252 ≈ 1 year).
    """

    def __init__(
        self,
        tickers:       List[str],
        ker_window:    int   = _KER_WINDOW,
        zscore_window: int   = _ZSCORE_WINDOW,
        tanh_scale:    float = _TANH_SCALE,
    ) -> None:
        self.tickers       = tickers
        self.ker_window    = ker_window
        self.zscore_window = zscore_window
        self.tanh_scale    = tanh_scale

    def compute_signal(
        self,
        close_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        close_df : (T, N) adjusted daily closes.

        Returns
        -------
        pd.DataFrame  (T, N)  ∈ (−1, 1)
        """
        close = close_df.reindex(columns=self.tickers).astype(np.float64)

        n = self.ker_window

        # ── Net displacement: |C_t − C_{t−n}| ───────────────────────────────
        net_change = close.diff(n).abs()   # (T, N); NaN for first n rows

        # ── Path length: Σ|ΔC| over n bars ──────────────────────────────────
        # abs(diff(1)) gives |C_t - C_{t-1}| for every t.
        # rolling(n).sum() sums n consecutive bars: exactly Σ_{k=0}^{n-1}|ΔC|.
        # min_periods guards partial windows at the head; using n//2 allows
        # warm-up to start earlier without distorting fully-formed windows.
        abs_daily = close.diff(1).abs()
        path_length = abs_daily.rolling(n, min_periods=max(5, n // 2)).sum()
        path_length = path_length.clip(lower=_MIN_PATH)

        # ── Kaufman Efficiency Ratio ∈ [0, 1] ───────────────────────────────
        ker = (net_change / path_length).clip(0.0, 1.0)

        # ── Directionalise: multiply KER by sign of n-day return ─────────────
        # sign(0) = 0 (flat over window → no signal)
        direction = np.sign(close.diff(n))
        dtfe = ker * direction                 # (T, N)  ∈ [−1, +1]

        # ── Cross-time Z-score + tanh ────────────────────────────────────────
        roll = dtfe.rolling(
            self.zscore_window, min_periods=max(63, self.zscore_window // 4)
        )
        mu  = roll.mean()
        sig = roll.std().clip(lower=_MIN_SIGMA)

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

        z      = (dtfe - mu) / (sig * self.tanh_scale)
        signal = np.tanh(z).astype(np.float32)

        result = pd.DataFrame(signal, index=close.index, columns=self.tickers)

        logger.info(
            f"DTFE-Trend| shape={result.shape} | "
            f"mean|α|={result.abs().mean().mean():.4f} | "
            f"non-zero={(result.abs() > 0.01).any(axis=1).mean()*100:.1f}%"
        )
        return result