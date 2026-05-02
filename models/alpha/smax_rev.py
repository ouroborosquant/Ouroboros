"""
FORTRESS v5 — models/alpha/smax_rev.py  [v1.0 — SMAX Reversal]

Idiosyncratic Maximum Return (SMAX) Reversal Signal Engine
==========================================================

Theory
------
The MAX anomaly (Bali et al. 2011): stocks exhibiting the highest recent
single-day return are subsequently subject to a *lottery premium reversal*.
Retail investors exhibit overweighting of low-probability extreme gains
(Tversky–Kahneman probability weighting), bidding up high-MAX stocks,
which then underperform as the lottery premium decays.

The key methodological upgrade here: naive MAX confounds *idiosyncratic*
lottery exposure with systemic high-beta co-movement. A high-beta stock
(e.g., NVDA) will naturally have high MAX during risk-on rallies purely
because of market exposure, not lottery premium. Cross-sectional OLS
orthogonalisation against β_sys removes this contamination:

    naive_max_{i,t} = γ₀ᵗ + γ₁ᵗ · β_sys_{i,t} + SMAX_{i,t}

The residual SMAX_{i,t} is, by construction, orthogonal to the cross-
sectional variation in systemic beta — it captures pure idiosyncratic jump
risk, which is the cleanest predictor of the lottery reversal.

Signal sign: HIGH SMAX → short (multiply by -1) because lottery premium
reversal produces *negative* forward returns for high-SMAX stocks.

Implementation details
----------------------
• β_sys: 252-day rolling window OLS beta against SPY.
  - Computed via rolling cov/var, numerically equivalent to OLS slope.
  - 252d chosen for stationarity vs 63d (avoids regime-switching noise
    in beta estimates for individual equities).

• naive_max: mean of top 5 single-day returns in trailing 21-day window.
  - 5 days chosen to smooth single-event noise while maintaining
    recency (Bali et al. use 1-day; we use 5-day average for robustness).
  - np.partition is O(N) vs sort O(N log N); for N=21, ~2× faster.
  - Fully vectorised via sliding_window_view; zero Python loops.

• Cross-sectional OLS: batched normal equations across T timesteps.
  - System is (T, N_eq, 2) — small enough for full vectorisation.
  - Same Tikhonov ridge as vts_lead for numerical stability.

Warm-up: max(252, 21) = 252 days before signals are non-zero.
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

logger = logging.getLogger("Ouroboros.SMAXRev")

# ── Hyper-parameters ──────────────────────────────────────────────────────────
_BETA_WINDOW:   int   = 252   # rolling window for systemic beta estimation
_MAX_WINDOW:    int   = 21    # trailing window for naive MAX computation
_N_TOP:         int   = 5     # number of top returns to average for naive_max
_TANH_SCALE:    float = 3.0   # divisor inside tanh
_RIDGE_LAMBDA:  float = 1e-8  # Tikhonov ridge for cross-sectional OLS
_WARMUP:        int   = max(_BETA_WINDOW, _MAX_WINDOW)


class SMAXReversalEngine:
    """
    Systemic-beta-orthogonalised idiosyncratic maximum return reversal.

    Parameters
    ----------
    equity_tickers : List[str]
        75 single-name equities to compute signals for.
    all_tickers : List[str]
        Full universe in output-column order.
    """

    def __init__(
        self,
        equity_tickers: List[str],
        all_tickers:    List[str],
    ) -> None:
        self._equity_cols: List[str] = [t for t in all_tickers if t in set(equity_tickers)]
        self._all_tickers: List[str] = all_tickers

    # ──────────────────────────────────────────────────────────────────────────

    def compute_signal(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        returns_df : pd.DataFrame
            Simple daily returns. Must contain 'SPY' + all equity tickers.
            Shape (T, N_all).

        Returns
        -------
        pd.DataFrame
            Shape (T, N_all). Equity tickers: inverted tanh-bounded SMAX Z-score.
            Macro ETF tickers: 0.0.
        """
        dates   = returns_df.index
        T       = len(dates)
        N_eq    = len(self._equity_cols)

        spy_ret = returns_df["SPY"].fillna(0.0).values.astype(np.float64)  # (T,)
        E_arr   = (
            returns_df[self._equity_cols].fillna(0.0).values.astype(np.float64)
        )  # (T, N_eq)

        logger.info(
            f"  SMAXRev: {T} dates, {N_eq} equities | "
            f"β_window={_BETA_WINDOW}, max_window={_MAX_WINDOW}, top={_N_TOP}"
        )

        # ── Step 1: Systemic beta via rolling cov/var ─────────────────────────
        # β_sys_{i,t} = cov(E_i, SPY; 252d) / var(SPY; 252d)
        # Numerically equivalent to OLS slope; vectorised via pandas
        E_df    = pd.DataFrame(E_arr, index=dates, columns=self._equity_cols)
        spy_s   = pd.Series(spy_ret, index=dates, name="SPY")

        cov_252 = E_df.rolling(_BETA_WINDOW, min_periods=63).cov(spy_s)  # (T, N_eq)
        var_spy = spy_s.rolling(_BETA_WINDOW, min_periods=63).var()       # (T,)
        var_spy = var_spy.clip(lower=1e-10)                               # floor before division

        beta_sys = cov_252.div(var_spy, axis=0).values.astype(np.float64)  # (T, N_eq)
        # NaN during warm-up → fill with cross-sectional median for OLS stability
        beta_sys = np.where(
            np.isfinite(beta_sys),
            beta_sys,
            np.nanmedian(beta_sys, axis=1, keepdims=True),
        )

        # ── Step 2: Vectorised naive_max via sliding window + np.partition ────
        # E_wins: sliding_window_view(E_arr, 21, axis=0) → (T-20, N_eq, 21)
        # np.partition along last axis: O(N) partial sort to find top _N_TOP
        E_wins = sliding_window_view(E_arr, _MAX_WINDOW, axis=0)  # (T-20, N_eq, 21)
        # Partition in-place: re-orders last axis so the last _N_TOP entries are the largest
        partitioned = np.partition(E_wins, -_N_TOP, axis=-1)       # (T-20, N_eq, 21)
        naive_max_partial = partitioned[..., -_N_TOP:].mean(axis=-1)  # (T-20, N_eq)

        # Align to full T: warm-up rows → NaN (handled later by filling with 0)
        n_valid = T - _MAX_WINDOW
        naive_max_full = np.full((T, N_eq), np.nan, dtype=np.float64)
        # sliding_window_view(arr, W, axis=0).shape == (T - W + 1, ...)
        # Window [i] covers arr[i : i+W], signal belongs at index i+W-1.
        # Starting fill index = W-1 (not W), so shape (T-W+1) fills (T-W+1) rows exactly.
        naive_max_full[_MAX_WINDOW - 1:] = naive_max_partial
        # naive_max_full[t] uses returns[t-20:t+1] — end-of-day signal, no look-ahead

        # ── Step 3: Cross-sectional OLS orthogonalisation ─────────────────────
        # At each t: project naive_max across N_eq equities onto β_sys
        #   naive_max_{:,t} = γ₀ᵗ + γ₁ᵗ · β_sys_{:,t} + SMAX_{:,t}
        # SMAX = residuals, orthogonal to cross-sectional β_sys variation.
        #
        # Batched normal equations:
        #   X_cs: (T, N_eq, 2)  — [1, β_sys]
        #   XᵀX:  (T, 2, 2)
        #   Xᵀy:  (T, 2)
        #   Solved via np.linalg.solve broadcast over T

        # Identify valid rows (both naive_max and beta_sys finite)
        valid_mask = np.isfinite(naive_max_full).all(axis=1) & \
                     np.isfinite(beta_sys).all(axis=1)       # (T,)

        X_cs = np.stack(
            [np.ones((T, N_eq)), beta_sys], axis=-1
        )  # (T, N_eq, 2)

        # XᵀX: (T, 2, 2)
        XTX = np.einsum("tni,tnj->tij", X_cs, X_cs)
        XTX[:, 0, 0] += _RIDGE_LAMBDA
        XTX[:, 1, 1] += _RIDGE_LAMBDA

        # Replace NaN in naive_max with 0 for the linear system (rows later masked)
        y_cs = np.where(np.isfinite(naive_max_full), naive_max_full, 0.0)  # (T, N_eq)

        # Xᵀy: reshape to (T, 2, 1) — explicit 3-D so numpy's gufunc
        # signature (m,m),(m,n)->(m,n) can unambiguously treat T as the
        # batch dim. With shape (T, 2), numpy maps core dims (m=T, n=2),
        # conflicting with m=2 from XTX — the "size 2093 ≠ 2" error.
        XTy = np.einsum("tni,tn->ti", X_cs, y_cs)[:, :, np.newaxis]  # (T, 2, 1)

        # Solve: (T, 2, 2) \ (T, 2, 1) → (T, 2, 1); squeeze → (T, 2)
        gamma = np.linalg.solve(XTX, XTy)[:, :, 0]  # (T, 2) = [γ₀, γ₁]

        # Fitted values: (T, N_eq)
        fitted = np.einsum("tni,ti->tn", X_cs, gamma)

        # SMAX residuals: idiosyncratic jump risk
        smax_raw = y_cs - fitted   # (T, N_eq)

        # Zero out warm-up and invalid rows
        smax_raw[~valid_mask] = 0.0

        # ── Step 4: Invert → Z-score → tanh bound ─────────────────────────────
        # -1.0 * SMAX: high idiosyncratic jump → short signal (lottery reversal)
        smax_inv = -1.0 * smax_raw  # (T, N_eq)

        mu  = np.nanmean(smax_inv, axis=1, keepdims=True)
        sig = np.nanstd(smax_inv,  axis=1, keepdims=True, ddof=1)
        sig = np.where(sig < 1e-8, 1.0, sig)

        z_scores = (smax_inv - mu) / sig
        signal_eq = np.tanh(z_scores / _TANH_SCALE).astype(np.float32)

        # Zero out warm-up rows
        signal_eq[:_WARMUP] = 0.0

        # ── Step 5: Embed into full N_all output DataFrame ────────────────────
        out = pd.DataFrame(0.0, index=dates, columns=self._all_tickers, dtype=np.float32)
        for j, ticker in enumerate(self._equity_cols):
            out[ticker] = signal_eq[:, j]

        non_zero_frac = (out.abs() > 0.01).any(axis=1).mean() * 100
        mean_abs = out[self._equity_cols].abs().mean().mean()
        logger.info(
            f"  SMAXRev ✓ | equity mean|α|={mean_abs:.4f} | "
            f"active_dates={non_zero_frac:.1f}%"
        )
        return out