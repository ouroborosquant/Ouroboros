"""
FORTRESS v5 — models/alpha/vts_lead.py  [v1.0 — VTS Beta Lead-Lag]

VTS (Volatility Term Structure) Beta Lead-Lag Signal Engine
===========================================================

Theory
------
The VIX term structure slope (VIX3M / VIX) encodes the vol risk premium
distribution: backwardation (VIX > VIX3M) signals acute stress; contango
(VIX < VIX3M) signals complacency. Each equity's *sensitivity* to innovations
in this slope (β^VTS) is the lead-lag signal:

  • High β^VTS: equity amplifies VTS moves → richly bid when structure steepens
    (risk-on), but a contra-signal if the current environment is backwardated.
  • The *cross-sectional rank* of β^VTS embeds relative vol-regime positioning
    across the equity universe, independent of the direction of the VTS move.

Signal construction (end-of-day close, no look-ahead):
  1. ΔVTS_t = ln(VIX3M_t / VIX_t) - ln(VIX3M_{t-1} / VIX_{t-1})
     — first difference isolates the innovation shock; the level carries
     mean-reverting serial correlation that contaminates the OLS fit.

  2. For each equity i at day t, trailing 63-day bivariate OLS:
       E_{i,τ} = α_i + β^VTS_{i,t} · ΔVTS_τ + β^MKT_{i,t} · SPY_τ + ε
     The market factor controls for co-movement that would otherwise
     inflate/deflate β^VTS via omitted variable bias.

  3. β^VTS vector (cross-section of 75 equities) → Z-score → tanh(Z / 3.0)
     tanh bounds prevent fat-tail outlier betas from dominating the alpha
     vector during crisis regimes.

Vectorized OLS via batched normal equations (no Python inner loop over T):
  XᵀX : (n_windows, 3, 3) — assembled via np.einsum
  Xᵀy : (n_windows, 3, N_eq) — assembled via np.einsum
  Tikhonov λI (1e-6) on XᵀX diagonal prevents singular systems on
  low-variance VTS windows (e.g., volatility pinning episodes).
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

logger = logging.getLogger("Ouroboros.VTSLead")

# ── Hyper-parameters ──────────────────────────────────────────────────────────
_OLS_WINDOW:  int   = 63    # rolling window for β^VTS estimation
_TANH_SCALE:  float = 3.0   # divisor inside tanh — σ-equivalent bounding
_RIDGE_LAMBDA: float = 1e-6  # Tikhonov regularisation on (XᵀX)
_MIN_VIX_LEVEL: float = 5.0  # floor to prevent ln(0) on rare yfinance gaps


class VTSLeadEngine:
    """
    Rolling bivariate OLS VTS beta extractor for equity_single universe.

    Parameters
    ----------
    equity_tickers : List[str]
        75 single-name equities to compute signals for.
    all_tickers : List[str]
        Full universe in output-column order (100 assets).
    ols_window : int
        Trailing window length for β^VTS OLS estimation. Default 63.
    """

    def __init__(
        self,
        equity_tickers: List[str],
        all_tickers:    List[str],
        ols_window:     int = _OLS_WINDOW,
    ) -> None:
        self._equity_cols: List[str] = [t for t in all_tickers if t in set(equity_tickers)]
        self._all_tickers: List[str] = all_tickers
        self._W: int = ols_window

    # ──────────────────────────────────────────────────────────────────────────

    def compute_signal(
        self,
        prices_df:  pd.DataFrame,
        returns_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        prices_df : pd.DataFrame
            Adjusted close prices — must contain columns '^VIX', '^VIX3M', 'SPY',
            and all equity tickers. Shape (T, ≥100).
        returns_df : pd.DataFrame
            Simple daily returns aligned to prices_df.index.

        Returns
        -------
        pd.DataFrame
            Shape (T, N_all). Equity tickers: tanh-bounded β^VTS Z-score ∈ (-1, 1).
            Macro ETF tickers: 0.0.
        """
        dates   = prices_df.index
        T       = len(dates)
        W       = self._W
        N_eq    = len(self._equity_cols)

        # ── 1. VTS innovation series ──────────────────────────────────────────
        vix  = prices_df["^VIX"].ffill().clip(lower=_MIN_VIX_LEVEL).values.astype(np.float64)
        vix3m = prices_df["^VIX3M"].ffill().clip(lower=_MIN_VIX_LEVEL).values.astype(np.float64)

        log_slope = np.log(vix3m) - np.log(vix)   # ln(VIX3M/VIX), shape (T,)
        dVTS = np.diff(log_slope, prepend=log_slope[0])  # first difference; index-aligned

        # ── 2. Equity returns and SPY market factor ───────────────────────────
        spy_ret = returns_df["SPY"].fillna(0.0).values.astype(np.float64)   # (T,)
        E_arr   = (                                                           # (T, N_eq)
            returns_df[self._equity_cols].fillna(0.0).values.astype(np.float64)
        )

        # ── 3. Build design matrix [1, ΔVTS, SPY] ───────────────────────────
        # shape: (T, 3)
        design = np.column_stack([
            np.ones(T, dtype=np.float64),
            dVTS,
            spy_ret,
        ])

        # ── 4. Sliding window views — O(1) memory relative to naive copy ─────
        # X_wins: (n_win, 3, W) → transpose → (n_win, W, 3)
        # E_wins: (n_win, N_eq, W) → transpose → (n_win, W, N_eq)
        n_win = T - W          # number of valid windows (window ends at t=W..T-1)

        X_wins = sliding_window_view(design, W, axis=0)   # (n_win, 3, W)
        X_wins = X_wins.transpose(0, 2, 1).copy()         # (n_win, W, 3) — copy for contiguity

        E_wins = sliding_window_view(E_arr, W, axis=0)    # (n_win, N_eq, W)
        E_wins = E_wins.transpose(0, 2, 1).copy()         # (n_win, W, N_eq)

        logger.info(
            f"  VTSLead: {T} dates, {n_win} windows (W={W}), "
            f"{N_eq} equities — vectorised batched OLS"
        )

        # ── 5. Batched normal equations: β = (XᵀX + λI)⁻¹ Xᵀy ──────────────
        # XᵀX : (n_win, 3, 3)  — einsum 'wij,wik->wjk' [batch outer products]
        XTX = np.einsum("wij,wik->wjk", X_wins, X_wins)           # (n_win, 3, 3)
        XTX[:, 0, 0] += _RIDGE_LAMBDA                              # ridge only on intercept
        XTX[:, 1, 1] += _RIDGE_LAMBDA
        XTX[:, 2, 2] += _RIDGE_LAMBDA

        # XᵀY : (n_win, 3, N_eq) — einsum 'wij,wik->wjk'
        XTY = np.einsum("wij,wik->wjk", X_wins, E_wins)           # (n_win, 3, N_eq)

        # np.linalg.solve supports batch: (n_win,3,3) \ (n_win,3,N_eq) → (n_win,3,N_eq)
        # β[w, 1, :] = β^VTS coefficient for all equities at window w
        betas_all = np.linalg.solve(XTX, XTY)                     # (n_win, 3, N_eq)
        beta_vts  = betas_all[:, 1, :]                             # (n_win, N_eq)

        # ── 6. Align to full date index (first W rows are warm-up → 0.0) ─────
        beta_full = np.zeros((T, N_eq), dtype=np.float64)
        # sliding_window_view returns T-W+1 windows; window[i] covers dates[i:i+W],
        # signal belongs at index i+W-1. Fill start = W-1, not W.
        beta_full[W - 1:] = beta_vts

        # ── 7. Cross-sectional Z-score → tanh bound ───────────────────────────
        # Z-score computed with ddof=1; NaN-safe via nanmean/nanstd
        mu  = np.nanmean(beta_full, axis=1, keepdims=True)         # (T, 1)
        sig = np.nanstd(beta_full,  axis=1, keepdims=True, ddof=1) # (T, 1)
        sig = np.where(sig < 1e-8, 1.0, sig)                       # prevent /0 on flat cross-sections

        z_scores = (beta_full - mu) / sig
        signal_eq = np.tanh(z_scores / _TANH_SCALE).astype(np.float32)

        # ── 8. Embed into full N_all output DataFrame (ETFs = 0.0) ────────────
        out = pd.DataFrame(0.0, index=dates, columns=self._all_tickers, dtype=np.float32)
        for j, ticker in enumerate(self._equity_cols):
            out[ticker] = signal_eq[:, j]

        # Zero the warm-up period explicitly
        out.iloc[:W] = 0.0

        non_zero_frac = (out.abs() > 0.01).any(axis=1).mean() * 100
        mean_abs = out[self._equity_cols].abs().mean().mean()
        logger.info(
            f"  VTSLead ✓ | equity mean|α|={mean_abs:.4f} | "
            f"active_dates={non_zero_frac:.1f}%"
        )
        return out