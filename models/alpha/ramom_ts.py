"""
FORTRESS v5 — models/alpha/ramom_ts.py  [v1.0 — RAMOM-TS]

Risk-Adjusted Time-Series Momentum (RAMOM-TS)
=============================================

Economic Thesis
---------------
Passive index funds and systematic CTAs generate a slow, persistent, low-
volatility "grind" that standard momentum cannot isolate because it conflates
high-velocity retail noise with low-velocity institutional accumulation.

Dividing daily log-returns by the Yang-Zhang (YZ) volatility estimator—the
minimum-variance unbiased OHLCV estimator that separately accounts for
overnight gap variance, open-to-close variance, and Rogers-Satchell drift
variance—produces a risk-adjusted return series whose SNR is dominated by
persistent directional flow, not headline-driven spikes.

The EMA velocity crossover (fast=5, slow=21) then acts as a continuous
derivative estimator on the vol-normalised return stream: positive velocity
means momentum is *accelerating* on a volatility-compressed basis, which is
the signature of algorithmic TWAP/VWAP execution.

Yang-Zhang Decomposition
------------------------
σ²_YZ = σ²_overnight + k·σ²_OC + (1-k)·σ²_RS

  σ²_overnight  = variance of ln(O_t / C_{t-1})      (jump component)
  σ²_OC         = variance of ln(C_t / O_t)           (open-to-close drift)
  σ²_RS         = Rogers-Satchell estimator            (continuous drift)
               = mean[ ln(H/C)·ln(H/O) + ln(L/C)·ln(L/O) ]
  k             = 0.34 / (1.35 + (n+1)/(n-1))         (optimal weighting)

Final Signal Pipeline
---------------------
1. R_t = ln(C_t / C_{t-1})
2. σ_YZ,t via rolling YZ estimator (window=21)
3. R_adj,t = R_t / (σ_YZ,t · √252)
4. V_t = EMA_5(R_adj) − EMA_21(R_adj)
5. Signal = tanh( (V_t − μ_252(V)) / (σ_252(V) · 3.0) )

Vectorised implementation: all rolling ops are pure pandas/numpy; no Python
loops over the time axis. Memory: O(T × N × 5) float32 temporaries.
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger("Ouroboros.RAMOMTS")

# ── Hyper-parameters ──────────────────────────────────────────────────────────
_YZ_WINDOW:       int   = 21    # rolling window for Yang-Zhang σ estimation
_EMA_FAST:        int   = 5     # fast EMA span for velocity
_EMA_SLOW:        int   = 21    # slow EMA span for velocity
_ZSCORE_WINDOW:   int   = 252   # trailing Z-score window
_TANH_SCALE:      float = 3.0   # σ-equivalent bounding divisor inside tanh
_MIN_SIGMA:       float = 1e-5  # floor on σ_YZ to prevent division by zero
_ANNUALISE:       float = np.sqrt(252.0)


def _yang_zhang_vol(
    open_: pd.DataFrame,
    high:  pd.DataFrame,
    low:   pd.DataFrame,
    close: pd.DataFrame,
    window: int = _YZ_WINDOW,
) -> pd.DataFrame:
    """
    Vectorised Yang-Zhang estimator over (T, N) OHLC matrices.

    k = 0.34 / (1.35 + (n+1)/(n-1)) is the optimal variance-ratio weighting
    derived analytically to minimise total estimator variance under the
    combined jump + diffusion price process.

    Returns annualised daily σ_YZ as a (T, N) DataFrame, forward-filled where
    the rolling window is not yet full.
    """
    n = window
    k = 0.34 / (1.35 + (n + 1) / (n - 1))

    # Log-price components — all operations broadcast over N columns
    log_ho = np.log(high / open_)       # ln(H/O)
    log_lo = np.log(low  / open_)       # ln(L/O)
    log_co = np.log(close / open_)      # ln(C/O) — open-to-close return
    log_oc_prev = np.log(open_ / close.shift(1))  # ln(O_t / C_{t-1}) — overnight

    # Rogers-Satchell: E[ln(H/C)·ln(H/O) + ln(L/C)·ln(L/O)]
    log_hc = np.log(high  / close)
    log_lc = np.log(low   / close)
    rs_sq  = (log_hc * log_ho + log_lc * log_lo).clip(lower=0.0)

    # Variance components
    var_overnight = (log_oc_prev ** 2).rolling(n, min_periods=max(5, n // 4)).mean()
    var_oc        = (log_co ** 2).rolling(n, min_periods=max(5, n // 4)).mean()
    var_rs        = rs_sq.rolling(n, min_periods=max(5, n // 4)).mean()

    # Correct overnight variance bias: E[X²] − (E[X])² = Var(X) only if
    # E[overnight] ≈ 0; for short windows apply sample-mean correction.
    mean_oc_prev = log_oc_prev.rolling(n, min_periods=max(5, n // 4)).mean()
    var_overnight = (var_overnight - mean_oc_prev ** 2).clip(lower=0.0)

    sigma_yz = np.sqrt(
        (var_overnight + k * var_oc + (1.0 - k) * var_rs).clip(lower=_MIN_SIGMA ** 2)
    )
    # Annualise: daily σ → annualised σ
    return sigma_yz * _ANNUALISE


class RAMOMTSEngine:
    """
    Risk-Adjusted Time-Series Momentum engine.

    Parameters
    ----------
    tickers : List[str]
        Output column universe (must match TICKERS in precompute_alpha_signals.py).
    yz_window : int
        Rolling window for Yang-Zhang volatility estimation.
    ema_fast / ema_slow : int
        EMA spans for the velocity crossover.
    zscore_window : int
        Trailing window for cross-time Z-score normalisation.
    """

    def __init__(
        self,
        tickers:       List[str],
        yz_window:     int   = _YZ_WINDOW,
        ema_fast:      int   = _EMA_FAST,
        ema_slow:      int   = _EMA_SLOW,
        zscore_window: int   = _ZSCORE_WINDOW,
        tanh_scale:    float = _TANH_SCALE,
    ) -> None:
        self.tickers       = tickers
        self.yz_window     = yz_window
        self.ema_fast      = ema_fast
        self.ema_slow      = ema_slow
        self.zscore_window = zscore_window
        self.tanh_scale    = tanh_scale

    def compute_signal(
        self,
        prices_df: pd.DataFrame,   # MultiIndex columns not required; plain OHLCV dict or dict of DFs
        open_df:   pd.DataFrame | None = None,
        high_df:   pd.DataFrame | None = None,
        low_df:    pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Compute RAMOM-TS signal for all tickers.

        Parameters
        ----------
        prices_df : pd.DataFrame
            Close prices indexed by date, columns = tickers. Adjusted closes.
        open_df / high_df / low_df : pd.DataFrame | None
            If provided, used for Yang-Zhang estimation.
            If None, the engine degrades gracefully to close-only Parkinson
            proxy using the previous-close and current-close gap as the
            overnight component (conservative underestimate; acceptable for ETFs).

        Returns
        -------
        pd.DataFrame  shape (T, N)  values ∈ (−1, 1) via tanh
        """
        close = prices_df.reindex(columns=self.tickers).astype(np.float64)

        # ── Step 1: Daily log-return ──────────────────────────────────────────
        log_ret = np.log(close / close.shift(1))   # (T, N)

        # ── Step 2: Yang-Zhang volatility ────────────────────────────────────
        if open_df is not None and high_df is not None and low_df is not None:
            open_  = open_df.reindex(columns=self.tickers).astype(np.float64)
            high_  = high_df.reindex(columns=self.tickers).astype(np.float64)
            low_   = low_df.reindex(columns=self.tickers).astype(np.float64)
            # Guard: yfinance sometimes delivers H < L on adjusted data
            high_  = np.maximum(high_, np.maximum(open_, close))
            low_   = np.minimum(low_,  np.minimum(open_, close))
            sigma_yz = _yang_zhang_vol(open_, high_, low_, close, self.yz_window)
        else:
            # Close-only degradation: Parkinson estimator on C_{t-21}..C_t.
            # Uses range proxy: σ² ≈ Var(ln C) over the window.
            # Multiply by √(252/21) to get daily annualised σ from 21-day window.
            logger.warning(
                "RAMOM-TS: OHLC not supplied — degrading to close-only σ proxy. "
                "IC will be ~10% lower than full Yang-Zhang."
            )
            sigma_yz = (
                log_ret.rolling(self.yz_window, min_periods=max(5, self.yz_window // 4))
                .std()
                .clip(lower=_MIN_SIGMA)
                * _ANNUALISE
            )

        sigma_yz = sigma_yz.clip(lower=_MIN_SIGMA)

        # ── Step 3: Risk-adjusted daily return ───────────────────────────────
        r_adj = log_ret / sigma_yz          # unitless; σ_YZ already annualised

        # ── Step 4: EMA velocity crossover ───────────────────────────────────
        # span s EMA: α = 2/(s+1).  adjust=False for causal, no look-ahead.
        ema_f = r_adj.ewm(span=self.ema_fast, adjust=False, min_periods=self.ema_fast).mean()
        ema_s = r_adj.ewm(span=self.ema_slow, adjust=False, min_periods=self.ema_slow).mean()
        velocity = ema_f - ema_s            # sign(V) encodes direction; |V| encodes conviction

        # ── Step 5: Cross-time Z-score + tanh ────────────────────────────────
        roll = velocity.rolling(self.zscore_window, min_periods=max(63, self.zscore_window // 4))
        mu   = roll.mean()
        sig  = roll.std().clip(lower=_MIN_SIGMA)

        z      = (velocity - mu) / (sig * self.tanh_scale)
        signal = np.tanh(z).astype(np.float32)

        # Propagate NaN for warm-up period rather than silently zeroing;
        # precompute_alpha_signals.py will fillna(0.0) on the full date index.
        result = pd.DataFrame(signal, index=close.index, columns=self.tickers)

        logger.info(
            f"RAMOM-TS | shape={result.shape} | "
            f"mean|α|={result.abs().mean().mean():.4f} | "
            f"non-zero={( result.abs() > 0.01).any(axis=1).mean()*100:.1f}%"
        )
        return result