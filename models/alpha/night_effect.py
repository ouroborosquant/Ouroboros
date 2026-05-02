"""
FORTRESS v5 — models/alpha/night_effect.py  [v1.0]

Night Effect Signal: Exploits the diurnal liquidity premium arising from the
structural divergence between overnight (institutional inventory repricing) and
intraday (VWAP-algo execution) return components.

Economic Basis
--------------
~100% of long-run S&P 500 gains accrued overnight (Cliff & Cooper, 2010;
Branch & Ma, 2012). The mechanism: retail order flow pools overnight → executes
at the open → dealers unwind inventory intraday via VWAP algos → mean reversion.
When the overnight gap is extreme, the subsequent intraday bleed is predictable.

Signal Construction
-------------------
1.  R_ON_{i,t}  = ln(O_{i,t} / C_{i,t-1})          — overnight return
2.  R_ID_{i,t}  = ln(C_{i,t} / O_{i,t})             — intraday return
3.  D_{i,t}     = Σ_{k=0}^{4} (R_ON - R_ID)_{i,t-k} — 5d discrepancy spread
4.  σ_GK_{i,t}  = √(½·ln(H/L)² − (2ln2−1)·ln(C/O)²) — Garman-Klass daily vol
5.  σ̄_GK_{i,t}  = rolling 21d mean of σ_GK           — vol regime denominator
6.  Z_{i,t}     = CS-ZScore_equity(-D_{i,t} / σ̄_GK_{i,t})
7.  α_{i,t}     = tanh(Z_{i,t} / 3.0)               — bounded ∈ (-1, +1)

Negative sign: large +D means stock gapped up AND bled intraday — bet on
further intraday mean-reversion of the accumulated imbalance.

Scope: equity_single universe ONLY. All macro ETF slots → 0.0.
       Cross-sectional Z-score is computed within the equity cross-section.

Bias Hazard
-----------
Requires split/dividend-ADJUSTED Open prices aligned identically to Close.
unadjusted open vs adjusted close creates ~0.1–2% phantom gaps on ex-dates
that will dominate the signal. yfinance auto_adjust=True handles this correctly.
"""
from __future__ import annotations

import logging
from typing import Dict, FrozenSet, List, Optional, Set

import numpy as np
import pandas as pd

logger = logging.getLogger("Ouroboros.NightEffect")

# Garman-Klass constant: (2·ln2 − 1) ≈ 0.3863
_GK_CONSTANT: float = 2.0 * np.log(2.0) - 1.0

# Warm-up periods
_DISCREPANCY_WINDOW: int = 1   # single-day divergence; 5d over-smooths the short-term reversion
                               # and captures multi-day momentum trends instead of microstructure noise
_GK_VOL_WINDOW:      int = 21  # vol regime lookback
_GK_MIN_PERIODS:     int = 10  # min valid days before GK estimate is trusted
_DISC_MIN_PERIODS:   int = 1   # min days in discrepancy window


class NightEffectEngine:
    """
    Computes the Night Effect alpha signal over the full price history.

    Parameters
    ----------
    equity_tickers : list[str]
        Tickers to compute the signal for. ETF/macro tickers receive 0.0.
    all_tickers : list[str]
        Full universe in column-order; output DataFrame columns match this.
    """

    def __init__(
        self,
        equity_tickers: List[str],
        all_tickers: List[str],
    ) -> None:
        self._equity_set: FrozenSet[str] = frozenset(equity_tickers)
        self._all_tickers: List[str] = all_tickers

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_signal(
        self,
        open_df:  pd.DataFrame,
        high_df:  pd.DataFrame,
        low_df:   pd.DataFrame,
        close_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute the Night Effect signal for every date in the input DataFrames.

        Parameters
        ----------
        open_df, high_df, low_df, close_df : pd.DataFrame
            Daily OHLC price DataFrames — **split/dividend adjusted** — with
            columns = all_tickers and DatetimeIndex rows.

        Returns
        -------
        pd.DataFrame
            Shape (T, N_all). Float32. Equity tickers: signal ∈ (-1, 1).
            Macro ETF tickers: 0.0 always.
        """
        equity_cols = [t for t in self._all_tickers if t in self._equity_set]

        o = open_df[equity_cols].values.astype(np.float64)    # (T, E)
        h = high_df[equity_cols].values.astype(np.float64)
        l = low_df[equity_cols].values.astype(np.float64)
        c = close_df[equity_cols].values.astype(np.float64)
        dates = open_df.index
        T, E = c.shape

        # ── Step 1: Diurnal return decomposition ──────────────────────────────
        # Shift close by 1 to get C_{t-1} aligned to O_{t}; no look-ahead.
        c_prev = np.full_like(c, np.nan)
        c_prev[1:] = c[:-1]

        with np.errstate(divide="ignore", invalid="ignore"):
            r_on = np.where(
                (o > 0) & (c_prev > 0),
                np.log(o / c_prev),       # overnight: Close(t-1) → Open(t)
                np.nan,
            )
            r_id = np.where(
                (c > 0) & (o > 0),
                np.log(c / o),             # intraday:  Open(t) → Close(t)
                np.nan,
            )

        divergence = r_on - r_id  # (+) means strong overnight, weak intraday

        # ── Step 2: 5-day rolling cumulative discrepancy ──────────────────────
        # Manual rolling sum to respect min_periods without pandas overhead.
        D = self._rolling_sum(divergence, window=_DISCREPANCY_WINDOW,
                               min_periods=_DISC_MIN_PERIODS)  # (T, E)

        # ── Step 3: Garman-Klass instantaneous vol ────────────────────────────
        # σ_GK = √(½·[ln(H/L)]² − (2ln2−1)·[ln(C/O)]²)
        # Floor at 0 before sqrt: theoretical minimum is 0, numerical noise can
        # produce tiny negatives when H≈L and C≈O.
        with np.errstate(divide="ignore", invalid="ignore"):
            hl_log = np.where((h > 0) & (l > 0), np.log(h / l), np.nan)
            co_log = np.where((c > 0) & (o > 0), np.log(c / o), np.nan)

        gk_daily = np.sqrt(
            np.maximum(0.5 * hl_log**2 - _GK_CONSTANT * co_log**2, 1e-12)
        )

        # ── Step 4: 21-day rolling mean of GK vol ─────────────────────────────
        gk_mean = self._rolling_mean(gk_daily, window=_GK_VOL_WINDOW,
                                      min_periods=_GK_MIN_PERIODS)  # (T, E)

        # ── Step 5: Raw signal = −D / σ̄_GK ──────────────────────────────────
        # Negative sign: large +D (overnight accumulation) → short signal
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = -D / np.where(gk_mean > 1e-8, gk_mean, np.nan)

        # ── Step 6: Cross-sectional Z-score within equity universe ────────────
        # Use warnings.catch_warnings to suppress nanmean/nanstd "Mean of empty
        # slice" and "Degrees of freedom <= 0" warnings that fire on warmup rows
        # (before GK vol window fills). nan_to_num handles the NaN output cleanly.
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice")
            warnings.filterwarnings("ignore", message="Degrees of freedom")
            cs_mean = np.nanmean(raw, axis=1, keepdims=True)    # (T, 1)
            cs_std  = np.nanstd(raw,  axis=1, keepdims=True)    # (T, 1)
        z = (raw - cs_mean) / np.where(cs_std > 1e-8, cs_std, 1.0)

        # ── Step 7: Bound via tanh ────────────────────────────────────────────
        signal_equity = np.tanh(z / 3.0).astype(np.float32)
        signal_equity = np.nan_to_num(signal_equity, nan=0.0)

        # ── Assemble full-universe output ─────────────────────────────────────
        result = pd.DataFrame(
            0.0, index=dates, columns=self._all_tickers, dtype=np.float32
        )
        result[equity_cols] = pd.DataFrame(
            signal_equity, index=dates, columns=equity_cols
        )

        _log_stats(result, equity_cols, "NightEffect")
        return result

    # ------------------------------------------------------------------
    # Vectorised rolling utilities (avoids pandas overhead for large arrays)
    # ------------------------------------------------------------------

    @staticmethod
    def _rolling_sum(
        arr: np.ndarray, window: int, min_periods: int
    ) -> np.ndarray:
        """Causal rolling window sum; NaN-safe via cumsum trick."""
        T, E = arr.shape
        out = np.full_like(arr, np.nan)
        # Replace NaN with 0 for cumsum then correct for missing counts
        valid = (~np.isnan(arr)).astype(np.float64)
        filled = np.where(np.isnan(arr), 0.0, arr)

        cum     = np.cumsum(filled, axis=0)
        cum_v   = np.cumsum(valid,  axis=0)

        for t in range(T):
            t0 = max(0, t - window + 1)
            s  = cum[t] - (cum[t0 - 1] if t0 > 0 else 0.0)
            sv = cum_v[t] - (cum_v[t0 - 1] if t0 > 0 else 0.0)
            out[t] = np.where(sv >= min_periods, s, np.nan)
        return out

    @staticmethod
    def _rolling_mean(
        arr: np.ndarray, window: int, min_periods: int
    ) -> np.ndarray:
        """Causal rolling window mean; NaN-safe."""
        T, E = arr.shape
        out = np.full_like(arr, np.nan)
        valid  = (~np.isnan(arr)).astype(np.float64)
        filled = np.where(np.isnan(arr), 0.0, arr)

        cum   = np.cumsum(filled, axis=0)
        cum_v = np.cumsum(valid,  axis=0)

        for t in range(T):
            t0 = max(0, t - window + 1)
            s  = cum[t] - (cum[t0 - 1] if t0 > 0 else 0.0)
            sv = cum_v[t] - (cum_v[t0 - 1] if t0 > 0 else 0.0)
            out[t] = np.where(sv >= min_periods, s / np.where(sv > 0, sv, 1.0), np.nan)
        return out


# ── Shared logging helper ─────────────────────────────────────────────────────

def _log_stats(df: pd.DataFrame, active_cols: List[str], name: str) -> None:
    active = df[active_cols]
    nonzero_pct = (active.abs() > 0.01).any(axis=1).mean() * 100
    logger.info(
        f"  ✓ {name}: {len(df)} days | "
        f"mean|α|={active.abs().mean().mean():.3f} | "
        f"active_days={nonzero_pct:.1f}%"
    )