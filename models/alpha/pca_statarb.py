"""
FORTRESS v5 — models/alpha/pca_statarb.py  [v1.0]

PCA Statistical Arbitrage Signal: Extracts idiosyncratic Ornstein-Uhlenbeck
mean-reversion residuals from single-name equities after orthogonalising
against the 25 macro ETF principal components.

Economic Basis
--------------
Single-name equity moves decompose as:
    E_i = Σ_k β_{ik} F_k  +  ε_i
where F_k are latent macro risk factors and ε_i is the idiosyncratic residual.
ε_i is stationary and mean-reverting (empirically approximated as OU process)
because: passive indexing noise, uncoordinated institutional flows, and ETF
arbitrage revert single-stock mispricing relative to its macro factor basket.

Signal Construction (rolling 60-day window)
--------------------------------------------
1.  PCA on 25 ETF returns → top-K eigenvectors explaining ≥ 85% variance
    (typically K ≈ 3–6 across regimes)
2.  Factor returns: F_{k,t} = V_{:,k}ᵀ M_t
3.  OLS regression (with intercept) for each equity:
        E_{i,t} = α_i + Σ_k β_{ik} F_{k,t} + ε_{i,t}
    Solved as a single batched lstsq: β_all = lstsq(F_aug, E_all)  — (K+1, N_eq)
4.  Cumulative residual path: X_{i,t} = Σ_{s=t-59}^{t} ε_{i,s}
5.  AR(1) on cumulative path to estimate OU params:
        X_{i,t} = a_i + b_i X_{i,t-1} + ζ_i
        m_i = a_i / (1 − b_i)               — equilibrium mean
        σ²_{eq,i} = Var(ζ_i) / (1 − b_i²)  — equilibrium variance
6.  S-score: S_{i,t} = (X_{i,t} − m_i) / σ_{eq,i}
7.  α_{i,t} = tanh(−S_{i,t} / 2.0)         — bounded ∈ (-1, +1)

Negative sign: S > 0 means idiosyncratically overpriced → short signal.

AR(1) Validity Guard
--------------------
Mean-reversion only holds when 0 < b < 1 (stationary OU).
If |b| ≥ 0.99 (unit root / explosive), signal is zeroed for that asset/date.
If σ_eq < ε_floor (degenerate fit), signal is zeroed.

Scope: equity_single tickers ONLY. Macro ETF slots → 0.0.
       PCA uses ETF-only return submatrix as factor basis.
"""
from __future__ import annotations

import logging
from typing import FrozenSet, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.linalg import lstsq, eigh

logger = logging.getLogger("Ouroboros.PCAStatArb")

_PCA_WINDOW:     int   = 60     # rolling window for PCA + OLS (trading days)
_PCA_MIN_DAYS:   int   = 45     # min valid days before signal is trusted
_VARIANCE_THRESH: float = 0.85  # cumulative variance to retain (K selection)
_B_STATIONARITY: float = 0.99   # |b| >= this → non-stationary → zero signal
_SIGMA_EQ_FLOOR: float = 1e-6   # degenerate OU fit guard
_TANH_SCALE:     float = 2.0    # S-score → tanh denominator


class PCAStatArbEngine:
    """
    Rolling PCA-based statistical arbitrage engine.

    Parameters
    ----------
    etf_tickers : list[str]
        25 macro ETF tickers used as the factor basis.
    equity_tickers : list[str]
        75 single-name equities to compute S-scores for.
    all_tickers : list[str]
        Full universe in output-column order.
    variance_threshold : float
        Cumulative PCA variance required to select K; default 0.85.
    """

    def __init__(
        self,
        etf_tickers:     List[str],
        equity_tickers:  List[str],
        all_tickers:     List[str],
        variance_threshold: float = _VARIANCE_THRESH,
    ) -> None:
        self._etf_set:      FrozenSet[str] = frozenset(etf_tickers)
        self._equity_set:   FrozenSet[str] = frozenset(equity_tickers)
        self._all_tickers:  List[str]      = all_tickers
        self._var_thresh:   float          = variance_threshold

        # Ordered sub-lists for indexing
        self._etf_cols:    List[str] = [t for t in all_tickers if t in self._etf_set]
        self._equity_cols: List[str] = [t for t in all_tickers if t in self._equity_set]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_signal(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute PCA StatArb signal over the full return history.

        Parameters
        ----------
        returns_df : pd.DataFrame
            Daily log or simple returns, columns = all_tickers, DatetimeIndex.
            Log returns preferred for PCA orthogonality.

        Returns
        -------
        pd.DataFrame
            Shape (T, N_all). Float32. Equity tickers: signal ∈ (-1, 1).
            Macro ETF tickers: 0.0 always.
        """
        dates = returns_df.index
        T = len(dates)

        M = returns_df[self._etf_cols].values.astype(np.float64)    # (T, 25)
        E = returns_df[self._equity_cols].values.astype(np.float64)  # (T, N_eq)
        N_eq = E.shape[1]

        signal_arr = np.zeros((T, N_eq), dtype=np.float64)

        logger.info(
            f"  PCAStatArb: {T} days × {len(self._etf_cols)} ETFs → "
            f"{N_eq} equities | window={_PCA_WINDOW}d"
        )

        for t in range(_PCA_WINDOW, T):
            t0 = t - _PCA_WINDOW  # window start (exclusive); window = [t0, t)
            # Note: we use [t0, t) → t days at indices t0..t-1; day t is NOT included.
            # This ensures no look-ahead: the signal on date[t] is computed from
            # returns[t0:t] = returns known at close of date[t-1].
            # We then assign signal_arr[t] to be USED for trading on date[t+1].

            M_win = M[t0:t]   # (60, 25)
            E_win = E[t0:t]   # (60, N_eq)

            valid_rows = np.all(np.isfinite(M_win), axis=1) & \
                         np.all(np.isfinite(E_win), axis=1)
            if valid_rows.sum() < _PCA_MIN_DAYS:
                continue

            M_valid = M_win[valid_rows]
            E_valid = E_win[valid_rows]

            s_scores = self._compute_s_scores(M_valid, E_valid)
            signal_arr[t] = np.tanh(-s_scores / _TANH_SCALE)

        signal_arr = np.nan_to_num(signal_arr, nan=0.0).astype(np.float32)

        # ── Assemble full-universe output ─────────────────────────────────────
        result = pd.DataFrame(
            0.0, index=dates, columns=self._all_tickers, dtype=np.float32
        )
        result[self._equity_cols] = pd.DataFrame(
            signal_arr, index=dates, columns=self._equity_cols
        )

        _log_stats(result, self._equity_cols, "PCAStatArb")
        return result

    # ------------------------------------------------------------------
    # Core computation (single time-step window)
    # ------------------------------------------------------------------

    def _compute_s_scores(
        self,
        M_win: np.ndarray,  # (n, 25) — ETF returns in window
        E_win: np.ndarray,  # (n, N_eq) — equity returns in window
    ) -> np.ndarray:        # (N_eq,) — S-scores
        """
        Full pipeline for one rolling window:
          PCA → factor returns → batched OLS → cumulative residual → AR(1) → S-score.
        """
        n = M_win.shape[0]

        # ── Step 1: PCA on ETF correlation matrix ─────────────────────────────
        # eigh returns eigenvalues ascending; flip for descending order.
        # Using correlation matrix (not covariance) normalises for vol differences
        # between ETFs — prevents TLT or GLD from dominating due to scale.
        M_centred = M_win - M_win.mean(axis=0, keepdims=True)
        std = M_centred.std(axis=0)
        std = np.where(std > 1e-10, std, 1.0)
        M_norm = M_centred / std                         # (n, 25)

        cov = (M_norm.T @ M_norm) / (n - 1)             # correlation matrix (25, 25)

        eigenvalues, eigenvectors = eigh(cov)            # ascending order
        eigenvalues = eigenvalues[::-1]                  # descending
        eigenvectors = eigenvectors[:, ::-1]             # (25, 25) matching order

        # Select top-K explaining >= variance_threshold cumulative variance
        total_var = eigenvalues.sum()
        cum_var   = np.cumsum(eigenvalues) / (total_var + 1e-10)
        K = int(np.searchsorted(cum_var, self._var_thresh)) + 1
        K = max(1, min(K, len(eigenvalues)))

        V_k = eigenvectors[:, :K]                        # (25, K)

        # ── Step 2: Factor returns ────────────────────────────────────────────
        F = M_norm @ V_k                                 # (n, K)

        # ── Step 3: Batched OLS across all equities ───────────────────────────
        # Augment F with intercept column → shape (n, K+1)
        F_aug = np.column_stack([np.ones(n), F])         # (n, K+1)

        # lstsq solves: E_win ≈ F_aug @ Beta; Beta has shape (K+1, N_eq)
        Beta, _, _, _ = lstsq(F_aug, E_win, rcond=None)  # (K+1, N_eq)

        residuals = E_win - F_aug @ Beta                 # (n, N_eq)

        # ── Step 4: Cumulative residual path ──────────────────────────────────
        X = np.cumsum(residuals, axis=0)                 # (n, N_eq)

        # ── Step 5: Vectorised AR(1) fit across all equities ──────────────────
        # Closed-form OLS on X_curr = a + b * X_prev for each equity simultaneously.
        X_prev = X[:-1]   # (n-1, N_eq)
        X_curr = X[1:]    # (n-1, N_eq)
        m_ar   = n - 1

        sum_xp   = X_prev.sum(axis=0)           # (N_eq,)
        sum_xc   = X_curr.sum(axis=0)
        sum_xp2  = (X_prev ** 2).sum(axis=0)
        sum_xpxc = (X_prev * X_curr).sum(axis=0)

        denom = m_ar * sum_xp2 - sum_xp ** 2
        # Avoid division by zero for assets with zero-variance residual path
        safe_denom = np.where(np.abs(denom) > 1e-12, denom, 1.0)

        b = (m_ar * sum_xpxc - sum_xp * sum_xc) / safe_denom   # (N_eq,)
        a = (sum_xc - b * sum_xp) / m_ar                        # (N_eq,)

        # ── Step 6: OU equilibrium parameters ────────────────────────────────
        # m = a/(1-b);  σ²_eq = Var(ζ)/(1-b²)
        one_minus_b = 1.0 - b
        # Guard: if b≈1 (unit root), set m to current path value (no mean reversion)
        m = np.where(np.abs(one_minus_b) > 1e-4, a / one_minus_b, X[-1])

        zeta       = X_curr - (a[None, :] + b[None, :] * X_prev)  # (n-1, N_eq)
        var_zeta   = (zeta ** 2).sum(axis=0) / max(m_ar - 2, 1)   # (N_eq,)

        one_minus_b2 = np.maximum(1.0 - b ** 2, 1e-8)
        sigma_eq2    = var_zeta / one_minus_b2                      # (N_eq,)
        sigma_eq     = np.sqrt(np.maximum(sigma_eq2, _SIGMA_EQ_FLOOR**2))

        # ── Step 7: S-score ───────────────────────────────────────────────────
        X_last   = X[-1]                                           # (N_eq,)
        s_scores = (X_last - m) / sigma_eq                        # (N_eq,)

        # Zero-out non-stationary fits (|b| ≥ threshold → unit root or explosive)
        non_stationary = np.abs(b) >= _B_STATIONARITY
        degenerate     = sigma_eq < _SIGMA_EQ_FLOOR
        s_scores = np.where(non_stationary | degenerate, 0.0, s_scores)

        return s_scores


# ── Shared logging helper ─────────────────────────────────────────────────────

def _log_stats(df: pd.DataFrame, active_cols: List[str], name: str) -> None:
    active      = df[active_cols]
    nonzero_pct = (active.abs() > 0.01).any(axis=1).mean() * 100
    logger.info(
        f"  ✓ {name}: {len(df)} days | "
        f"mean|α|={active.abs().mean().mean():.3f} | "
        f"active_days={nonzero_pct:.1f}%"
    )