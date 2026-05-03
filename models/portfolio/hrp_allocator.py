"""
FORTRESS v6 — models/portfolio/hrp_allocator.py
═══════════════════════════════════════════════════════════════════════════════
Hierarchical Risk Parity Allocator with Ledoit-Wolf Shrinkage.

WHY HRP OVER CVaR-MVO
─────────────────────
Classical MVO requires inverting Σ. The inverse is the bottleneck:

    w* = Σ⁻¹μ / (1ᵀΣ⁻¹μ)      [unconstrained MV]

When Σ is ill-conditioned (κ(Σ) >> 1, typical during regime shifts), the
pseudo-inverse amplifies estimation noise by a factor of κ(Σ). A ±1%
perturbation in μ translates to ±κ(Σ)% weight oscillation. At κ(Σ) = 1e4
and 25 assets, that means turnover spikes of 100x — exactly the overtrading
pathology observed in v5 OOS.

HRP (López de Prado, 2016) never inverts Σ. It uses a tree of recursive
variance-minimizing bisections. The weight of each asset is determined by
the inverse variance of its cluster branch — structurally bounded.

LEDOIT-WOLF SHRINKAGE
─────────────────────
The sample covariance Σ_S converges to the population Σ at rate O(1/√T).
For T=252, N=25, the estimation error in each off-diagonal entry is O(5%).
These errors are the primary source of ill-conditioning.

Ledoit-Wolf shrinkage applies a convex combination:

    Σ̂ = (1-δ)·Σ_S + δ·F

where F is a structured target (here: constant-correlation model) and δ* is
the analytically optimal shrinkage intensity minimising the Frobenius-norm
expected loss E[‖Σ̂ - Σ‖²_F]. The analytical formula for δ* avoids
cross-validation and is O(N²T) — fast enough for daily re-estimation.

Constant-Correlation shrinkage target (Ledoit-Wolf 2004):
    F_ii = Σ_S_ii     (diagonal preserved)
    F_ij = r̄·√(Σ_S_ii·Σ_S_jj)  where r̄ = mean off-diagonal correlation

CORRELATION DISTANCE
────────────────────
For the hierarchical clustering we use:

    d_ij = √(0.5·(1 - ρ_ij))

This is the ONLY distance metric that guarantees d ∈ [0,1] and satisfies
the triangle inequality for correlation matrices (Mantegna, 1999).
Standard |1-ρ| or (1-ρ)² do NOT satisfy the triangle inequality; they
produce inconsistent cluster merges that under-diversify.

SIGNAL TILT
───────────
Post-HRP, the raw weights are risk-balanced but alpha-blind. We tilt by:

    w̃_i = w_HRP_i · (1 + β · ŝ_i)     [unnormalised]
    w_final = clip(w̃, lb, ub) / Σ clip(w̃, lb, ub)

where ŝ_i = (s_i - μ_s) / (σ_s + ε) is the z-scored signal. This preserves
the HRP diversification structure while allowing the alpha signal to shift
weight from low-IC to high-IC assets. β controls the tilt intensity;
β = 0 recovers pure HRP, β >> 1 approaches pure alpha (dangerous).

ASYNC INTERFACE
───────────────
The allocator is synchronous internally (all scipy, no I/O). The async
wrapper `async_allocate` is a thin executor wrapper that keeps the event
loop free during the O(N² log N) clustering and recursive bisection.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
from sklearn.covariance import LedoitWolf

log = logging.getLogger("HRPAllocator")

# ── Universe constants ─────────────────────────────────────────────────────────
TICKERS: List[str] = [
    "SPY","QQQ","IWM","TLT","HYG","LQD","GLD","SLV","GDX","XLE",
    "XLF","XLK","XLV","XLU","XLI","XLP","XLY","XLB","XLC","VIXY",
    "BIL","SHV","USO","PDBC","COWZ",
]
N: int = len(TICKERS)
CASH_IDXS: Tuple[int, ...] = (TICKERS.index("BIL"), TICKERS.index("SHV"))


# ══════════════════════════════════════════════════════════════════════════════
# §1  LEDOIT-WOLF SHRINKAGE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ShrinkageResult:
    """Output of LedoitWolfShrinkage.fit()."""
    cov_shrunk:   np.ndarray    # (N, N) shrunk covariance
    cov_sample:   np.ndarray    # (N, N) raw sample covariance
    corr_shrunk:  np.ndarray    # (N, N) derived correlation matrix
    shrinkage_coeff: float      # δ* ∈ [0, 1] — diagnostic only
    condition_number: float     # κ(Σ̂) — should be << κ(Σ_S)


class LedoitWolfShrinkage:
    r"""
    Constant-Correlation Ledoit-Wolf shrinkage estimator.

    Shrinkage target F (constant-correlation model):
        F_ii = Σ_S_ii                           (diagonal entries preserved)
        F_ij = r̄·√(Σ_S_ii · Σ_S_jj)             (off-diagonal: avg corr × vol product)

    where r̄ = (2 / (N·(N-1))) · Σ_{i<j} ρ_ij   (mean pairwise correlation).

    This target is the most common choice in practice because it is:
        1. Semi-definite positive by construction.
        2. Calibrated to the cross-sectional average correlation level.
        3. Shrinks extreme off-diagonal entries toward the market average —
           the most common source of ill-conditioning at N < 100.

    We delegate the optimal δ* computation to sklearn's LedoitWolf, which
    uses the Oracle Approximating Shrinkage (OAS) analytical formula — O(N²T)
    with no simulation required.

    Parameters
    ----------
    use_constant_corr_target : bool
        If True, use the constant-correlation target F as described above.
        If False, fall back to sklearn's default identity-like target.
        Default: True.
    min_periods : int
        Minimum number of observations required to fit.  Below this,
        returns an identity-scaled covariance.  Default: 30.
    """

    def __init__(
        self,
        use_constant_corr_target: bool = True,
        min_periods:              int   = 30,
    ) -> None:
        self.use_cc_target = use_constant_corr_target
        self.min_periods   = min_periods

    def fit(self, returns: np.ndarray) -> ShrinkageResult:
        r"""
        Compute the shrunk covariance from a (T, N) return matrix.

        Algorithm
        ─────────
        1.  Compute Σ_S = (1/T) · Xᵀ·X   (de-meaned returns).
        2.  Compute r̄ = mean off-diagonal correlation of Σ_S.
        3.  Build constant-correlation target F.
        4.  Compute δ* via sklearn OAS.
        5.  Return Σ̂ = (1-δ*)·Σ_S + δ*·F.

        Args:
            returns: (T, N) array of daily log-returns. Rows are time steps.

        Returns:
            ShrinkageResult with shrunk covariance and diagnostics.
        """
        T, N = returns.shape

        if T < self.min_periods:
            log.warning(
                "LedoitWolf: T=%d < min_periods=%d — returning identity covariance.",
                T, self.min_periods
            )
            cov_eye = np.eye(N) * float(np.var(returns, axis=0).mean())
            return ShrinkageResult(
                cov_shrunk       = cov_eye,
                cov_sample       = cov_eye,
                corr_shrunk      = np.eye(N),
                shrinkage_coeff  = 1.0,
                condition_number = 1.0,
            )

        # ── 1. Sample covariance ────────────────────────────────────────────
        # sklearn LedoitWolf expects (N_samples, N_features) = (T, N)
        lw = LedoitWolf(assume_centered=False)
        lw.fit(returns)

        cov_sample  = np.cov(returns.T, ddof=1)   # unbiased
        delta_star  = float(lw.shrinkage_)

        # ── 2. Constant-correlation target ───────────────────────────────────
        if self.use_cc_target:
            cov_shrunk = self._constant_corr_shrink(cov_sample, delta_star)
        else:
            # sklearn's default oracle target
            cov_shrunk = lw.covariance_

        # ── 3. Derived correlation ───────────────────────────────────────────
        D_inv = np.diag(1.0 / np.sqrt(np.diag(cov_shrunk).clip(min=1e-10)))
        corr  = D_inv @ cov_shrunk @ D_inv
        corr  = np.clip(corr, -1.0 + 1e-6, 1.0 - 1e-6)
        np.fill_diagonal(corr, 1.0)

        kappa = float(np.linalg.cond(cov_shrunk + 1e-10 * np.eye(N)))

        log.debug(
            "LedoitWolf: δ*=%.4f  κ(Σ_S)=%.1f → κ(Σ̂)=%.1f",
            delta_star,
            float(np.linalg.cond(cov_sample + 1e-10 * np.eye(N))),
            kappa,
        )

        return ShrinkageResult(
            cov_shrunk       = cov_shrunk,
            cov_sample       = cov_sample,
            corr_shrunk      = corr,
            shrinkage_coeff  = delta_star,
            condition_number = kappa,
        )

    @staticmethod
    def _constant_corr_shrink(cov_S: np.ndarray, delta: float) -> np.ndarray:
        r"""
        Build Σ̂ = (1-δ)·Σ_S + δ·F where F is the constant-correlation target.

        F_ij = r̄·√(Σ_S_ii · Σ_S_jj)   for i ≠ j
        F_ii = Σ_S_ii

        This preserves the sample variances on the diagonal (critical for
        correct portfolio volatility scaling) and shrinks cross-asset
        correlations toward the market average.
        """
        N   = cov_S.shape[0]
        std = np.sqrt(np.diag(cov_S).clip(min=1e-10))

        # Correlation matrix from sample covariance
        D_inv  = np.diag(1.0 / std)
        corr_S = D_inv @ cov_S @ D_inv
        np.clip(corr_S, -1.0, 1.0, out=corr_S)
        np.fill_diagonal(corr_S, 1.0)

        # Mean off-diagonal correlation r̄
        upper_mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        r_bar      = float(corr_S[upper_mask].mean())

        # Constant-correlation target F
        F = r_bar * np.outer(std, std)
        np.fill_diagonal(F, np.diag(cov_S))   # preserve sample variances

        return (1.0 - delta) * cov_S + delta * F


# ══════════════════════════════════════════════════════════════════════════════
# §2  HRP CORE ALGORITHM
# ══════════════════════════════════════════════════════════════════════════════

def _corr_to_distance(corr: np.ndarray) -> np.ndarray:
    r"""
    Convert a correlation matrix to a distance matrix using the metric:

        d_ij = √(0.5·(1 - ρ_ij))

    Properties of this metric that make it correct for hierarchical clustering:
        1.  d_ij ∈ [0, 1]       (bounded, interpretable)
        2.  d_ii = 0            (reflexivity)
        3.  d_ij = d_ji         (symmetry)
        4.  Triangle inequality satisfied (Mantegna 1999)

    Competing metrics |1-ρ| and (1-ρ)² satisfy (1-3) but NOT (4), leading
    to spurious cluster merges and artificially concentrated portfolios.

    Args:
        corr: (N, N) symmetric correlation matrix with diagonal 1s.

    Returns:
        dist: (N, N) distance matrix.
    """
    # Guard: clip correlations to avoid sqrt of negative under floating noise
    dist = np.sqrt(0.5 * (1.0 - np.clip(corr, -1.0, 1.0)))
    np.fill_diagonal(dist, 0.0)
    return dist


def _get_quasi_diag(link: np.ndarray) -> List[int]:
    """
    Quasi-diagonalisation: extract the leaf order from the linkage matrix
    such that similar assets are adjacent.

    López de Prado Algorithm 2 (Machine Learning for Asset Managers, 2020).

    The linkage matrix has shape (N-1, 4):
      [left_cluster, right_cluster, distance, n_leaves]
    Leaf indices are 0 … N-1; internal node i is referenced as N+i.

    Returns:
        Ordered list of original asset indices (length N).
    """
    N_leaves = link.shape[0] + 1

    def _recurse(node_id: int) -> List[int]:
        # Leaf node: base case
        if node_id < N_leaves:
            return [int(node_id)]
        # Internal node: look up in linkage matrix
        row   = link[int(node_id) - N_leaves]
        left  = int(row[0])
        right = int(row[1])
        return _recurse(left) + _recurse(right)

    root_id = 2 * N_leaves - 2   # Root is always the last merge
    return _recurse(root_id)


def _cluster_var(
    cov:     np.ndarray,
    indices: List[int],
) -> float:
    """
    Minimum-variance portfolio variance within a cluster.

    For a set of assets S, the minimum-variance cluster portfolio is:
        w_S = (Σ_S⁻¹·1) / (1ᵀ·Σ_S⁻¹·1)
        σ²_S = 1 / (1ᵀ·Σ_S⁻¹·1)

    This is analytically equivalent to the reciprocal of the sum of entries
    of the inverse sub-covariance matrix.

    Args:
        cov:     Full (N, N) covariance matrix.
        indices: List of asset indices in this cluster.

    Returns:
        Scalar cluster variance σ²_S.
    """
    cov_slice = cov[np.ix_(indices, indices)]

    # Regularise for potential near-singularity in small clusters
    cov_slice = cov_slice + 1e-10 * np.eye(len(indices))

    try:
        cov_inv = np.linalg.inv(cov_slice)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov_slice)

    ones  = np.ones(len(indices))
    denom = float(ones @ cov_inv @ ones)
    return 1.0 / max(denom, 1e-10)


def _rec_bisect(
    cov:     np.ndarray,
    sorted_idx: List[int],
) -> np.ndarray:
    r"""
    Recursive bisection algorithm for HRP weight allocation.

    Starting with the full sorted index list (quasi-diagonalised order),
    at each level the list is split into two halves. The weight allocated
    to each half is proportional to the INVERSE of its minimum-variance
    cluster variance:

        α = σ²_R / (σ²_L + σ²_R)    [weight assigned to left cluster]
        1 - α                         [weight assigned to right cluster]

    This top-down recursion naturally allocates more weight to lower-risk
    clusters while maintaining full investment (weights sum to 1).

    The recursion terminates when each cluster has exactly one asset,
    at which point the allocated weight is the asset's final portfolio weight.

    Args:
        cov:        (N, N) shrunk covariance matrix.
        sorted_idx: Quasi-diagonalised list of asset indices.

    Returns:
        (N,) weight vector in the ORIGINAL (un-sorted) asset order.
    """
    weights = np.zeros(cov.shape[0])
    _rec(cov, sorted_idx, weights, allocation=1.0)
    return weights


def _rec(
    cov:         np.ndarray,
    items:       List[int],
    weights:     np.ndarray,
    allocation:  float,
) -> None:
    """In-place recursive bisection helper."""
    if len(items) == 1:
        weights[items[0]] += allocation
        return

    # Split into two halves at the midpoint
    mid   = len(items) // 2
    left  = items[:mid]
    right = items[mid:]

    # Variance of each half under its own minimum-variance portfolio
    var_L = _cluster_var(cov, left)
    var_R = _cluster_var(cov, right)

    # Weight split: inversely proportional to variance
    # α_L = var_R / (var_L + var_R)  — more weight to LOWER-variance cluster
    denom = var_L + var_R
    if denom < 1e-12:
        alpha_L = 0.5
    else:
        alpha_L = var_R / denom

    _rec(cov, left,  weights, allocation * alpha_L)
    _rec(cov, right, weights, allocation * (1.0 - alpha_L))


# ══════════════════════════════════════════════════════════════════════════════
# §3  SIGNAL TILT
# ══════════════════════════════════════════════════════════════════════════════

def _apply_signal_tilt(
    w_hrp:          np.ndarray,    # (N,) pure HRP weights
    signal:         np.ndarray,    # (N,) raw alpha signal
    beta:           float,         # tilt strength
    weight_lb:      float,
    weight_ub:      float,
) -> np.ndarray:
    r"""
    Tilt HRP weights in the direction of the alpha signal.

    Formula:
        w̃_i = w_HRP_i · (1 + β · ŝ_i)
        w_final = clip(w̃, lb, ub) / Σ_i clip(w̃, lb, ub)

    where ŝ_i = (s_i - μ_s) / (σ_s + ε) is the cross-sectionally normalised
    signal. This normalisation ensures:
        1. Σ_i ŝ_i ≈ 0: tilt does not bias total invested exposure.
        2. std(ŝ) = 1: β has a consistent, interpretable magnitude.
           β=0.2 means a 1-σ signal advantage → 20% extra weight.

    Box constraints are applied AFTER tilt to enforce universe limits,
    then the result is renormalised. This ensures the sum-to-1 constraint
    is always satisfied regardless of β or extreme signal values.

    Args:
        w_hrp:     (N,) HRP weights, sum = 1.
        signal:    (N,) alpha signal (any scale).
        beta:      Tilt intensity. 0 = pure HRP, 0.3 is a sensible default.
        weight_lb: Per-asset lower bound.
        weight_ub: Per-asset upper bound.

    Returns:
        (N,) tilted, clipped, renormalised weight vector.
    """
    if beta == 0.0 or np.all(signal == 0.0):
        return w_hrp.copy()

    # Cross-sectional z-score of signal
    mu_s    = signal.mean()
    sigma_s = signal.std() + 1e-8
    s_hat   = (signal - mu_s) / sigma_s

    # Tilt: multiplicative adjustment preserves HRP proportionality
    w_tilt = w_hrp * (1.0 + beta * s_hat)

    # Box constraints + renormalisation
    w_clipped = np.clip(w_tilt, weight_lb, weight_ub)
    total     = w_clipped.sum()
    if total < 1e-10:
        log.warning("Signal tilt produced near-zero weight sum — returning pure HRP.")
        return w_hrp.copy()

    return w_clipped / total


# ══════════════════════════════════════════════════════════════════════════════
# §4  DRAWDOWN GUARD
# ══════════════════════════════════════════════════════════════════════════════

def _apply_drawdown_guard(
    weights:       np.ndarray,    # (N,) proposed weights
    current_nav:   float,
    peak_nav:      float,
    dd_threshold:  float = 0.05,  # 5% daily DD → full cash
    cash_idx:      int   = 20,    # BIL index
) -> np.ndarray:
    """
    Hard drawdown circuit breaker.

    If (peak_nav - current_nav) / peak_nav >= dd_threshold, override the
    proposed allocation with 100% BIL regardless of signal or HRP output.

    This is the Supreme Law §1 enforcer: capital preservation overrides
    all alpha optimisation.

    Args:
        weights:      (N,) proposed portfolio weights.
        current_nav:  Current portfolio NAV.
        peak_nav:     High-water mark NAV.
        dd_threshold: Drawdown fraction that triggers the circuit.
        cash_idx:     Index of the cash proxy (BIL).

    Returns:
        (N,) weights — possibly overridden to 100% cash.
    """
    dd = (peak_nav - current_nav) / (peak_nav + 1e-10)
    if dd >= dd_threshold:
        log.warning(
            "Drawdown circuit breaker triggered: DD=%.2f%% >= %.2f%% — "
            "forcing 100%% BIL.",
            dd * 100, dd_threshold * 100,
        )
        w_cash = np.zeros_like(weights)
        w_cash[cash_idx] = 1.0
        return w_cash
    return weights


# ══════════════════════════════════════════════════════════════════════════════
# §5  MAIN ALLOCATOR CLASS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class HRPResult:
    """Output of HRPAllocator.allocate()."""
    weights:          np.ndarray    # (N,) final portfolio weights
    weights_hrp_pure: np.ndarray    # (N,) HRP weights pre-tilt (diagnostic)
    cov_shrunk:       np.ndarray    # (N, N) shrunk covariance used
    corr_shrunk:      np.ndarray    # (N, N) shrunk correlation
    sorted_order:     List[int]     # quasi-diagonalised asset order
    shrinkage_coeff:  float         # δ* diagnostic
    condition_number: float         # κ(Σ̂) diagnostic
    cluster_variances: Dict[str, float]  # {ticker: cluster_var} for logging


class HRPAllocator:
    """
    Hierarchical Risk Parity portfolio allocator.

    Signal flow:
        returns_window (T, N)
            → LedoitWolfShrinkage.fit()          (Σ̂, ρ̂)
            → _corr_to_distance(ρ̂)               (D)
            → scipy Ward linkage + quasi-diag    (sorted_idx)
            → _rec_bisect(Σ̂, sorted_idx)         (w_HRP)
            → _apply_signal_tilt(w_HRP, s, β)    (w_tilt)
            → _apply_drawdown_guard(...)          (w_final)

    Parameters
    ----------
    n_assets:      Universe size N.
    lookback:      Number of trading days for covariance estimation.
    beta:          Signal tilt intensity. 0 = pure HRP.
    weight_lb:     Per-asset lower bound.
    weight_ub:     Per-asset upper bound.
    dd_threshold:  Max-drawdown circuit breaker threshold.
    linkage_method: Hierarchical clustering linkage. 'single' follows López
                    de Prado's original paper; 'ward' is sometimes more stable
                    for large universes.
    """

    def __init__(
        self,
        n_assets:       int   = N,
        lookback:       int   = 252,
        beta:           float = 0.25,
        weight_lb:      float = 0.0,
        weight_ub:      float = 0.25,
        dd_threshold:   float = 0.05,
        linkage_method: str   = "single",
    ) -> None:
        self.n_assets       = n_assets
        self.lookback       = lookback
        self.beta           = beta
        self.weight_lb      = weight_lb
        self.weight_ub      = weight_ub
        self.dd_threshold   = dd_threshold
        self.linkage_method = linkage_method

        self._shrinkage = LedoitWolfShrinkage(use_constant_corr_target=True)

        # State for drawdown guard
        self._nav:      float = 1.0
        self._peak_nav: float = 1.0

    def update_nav(self, nav: float) -> None:
        """Call once per day after marking to market."""
        self._nav      = nav
        self._peak_nav = max(self._peak_nav, nav)

    def allocate(
        self,
        returns_window: np.ndarray,    # (T, N) daily log-returns
        signal:         Optional[np.ndarray] = None,  # (N,) alpha signal
        w_prev:         Optional[np.ndarray] = None,  # (N,) for logging only
    ) -> HRPResult:
        """
        Compute HRP + signal-tilted portfolio weights.

        Args:
            returns_window: (T, N) return matrix, strictly causal.
            signal:         (N,) alpha signal, or None for pure HRP.
            w_prev:         (N,) previous weights (used only for turnover log).

        Returns:
            HRPResult dataclass with weights and diagnostics.
        """
        T, N = returns_window.shape
        assert N == self.n_assets, f"Universe mismatch: got {N}, expected {self.n_assets}"

        # ── Step 1: Ledoit-Wolf shrinkage ─────────────────────────────────
        shrink = self._shrinkage.fit(returns_window[-self.lookback:])

        # ── Step 2: Correlation → distance matrix ─────────────────────────
        dist_mat = _corr_to_distance(shrink.corr_shrunk)   # (N, N)

        # Convert to condensed 1D form required by scipy linkage
        # np.triu flattened with k=1 gives the upper triangle in row order
        dist_cond = ssd.squareform(dist_mat, checks=False)  # (N*(N-1)/2,)

        # ── Step 3: Hierarchical clustering ───────────────────────────────
        link = sch.linkage(dist_cond, method=self.linkage_method)

        # ── Step 4: Quasi-diagonalisation ────────────────────────────────
        sorted_idx = _get_quasi_diag(link)

        # ── Step 5: Recursive bisection ───────────────────────────────────
        w_hrp = _rec_bisect(shrink.cov_shrunk, sorted_idx)
        w_hrp = np.clip(w_hrp, self.weight_lb, self.weight_ub)
        w_hrp /= w_hrp.sum()

        # ── Step 6: Signal tilt ────────────────────────────────────────────
        s = signal if signal is not None else np.zeros(N)
        w_tilt = _apply_signal_tilt(
            w_hrp, s, self.beta, self.weight_lb, self.weight_ub
        )

        # ── Step 7: Drawdown circuit breaker ──────────────────────────────
        w_final = _apply_drawdown_guard(
            w_tilt, self._nav, self._peak_nav,
            dd_threshold = self.dd_threshold,
            cash_idx     = CASH_IDXS[0],
        )

        # ── Diagnostics ───────────────────────────────────────────────────
        cluster_vars: Dict[str, float] = {}
        for i in range(N):
            cv = float(shrink.cov_shrunk[i, i])
            if i < len(TICKERS):
                cluster_vars[TICKERS[i]] = cv

        if w_prev is not None:
            turnover = float(np.abs(w_final - w_prev).sum())
            log.info(
                "HRPAllocator: δ*=%.3f  κ(Σ̂)=%.1f  turnover=%.4f  "
                "max_w=%.3f  min_w=%.3f",
                shrink.shrinkage_coeff,
                shrink.condition_number,
                turnover,
                float(w_final.max()),
                float(w_final[w_final > 0].min()) if (w_final > 0).any() else 0.0,
            )

        return HRPResult(
            weights           = w_final,
            weights_hrp_pure  = w_hrp,
            cov_shrunk        = shrink.cov_shrunk,
            corr_shrunk       = shrink.corr_shrunk,
            sorted_order      = sorted_idx,
            shrinkage_coeff   = shrink.shrinkage_coeff,
            condition_number  = shrink.condition_number,
            cluster_variances = cluster_vars,
        )

    # ── Async interface ────────────────────────────────────────────────────────

    async def async_allocate(
        self,
        returns_window: np.ndarray,
        signal:         Optional[np.ndarray] = None,
        w_prev:         Optional[np.ndarray] = None,
    ) -> HRPResult:
        """
        Async executor wrapper.

        All HRP computation is CPU-bound (scipy clustering, matrix ops).
        Running it in the default executor prevents blocking the event loop
        during the 5-60ms clustering step.

        Usage:
            result = await allocator.async_allocate(returns, signal, w_prev)
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.allocate(returns_window, signal, w_prev)
        )

    def rebalance_needed(
        self,
        w_current:     np.ndarray,
        w_proposed:    np.ndarray,
        min_turnover:  float = 0.02,
    ) -> bool:
        """
        Returns True if proposed rebalance exceeds the minimum turnover
        threshold — avoids trivial rebalances that generate commissions
        without meaningfully changing the portfolio.

        min_turnover = 0.02 means "only rebalance if L1 weight change > 2%".
        This is the complement of the γ turnover penalty in CVaR-MVO:
        instead of penalising turnover in the objective, we gate rebalancing
        at the execution layer.
        """
        turnover = float(np.abs(w_proposed - w_current).sum())
        return turnover >= min_turnover


# ══════════════════════════════════════════════════════════════════════════════
# §6  STANDALONE SMOKE TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.DEBUG)

    rng = np.random.default_rng(42)
    T, N_ = 252, 25

    # Synthetic factor returns: common factor + idiosyncratic
    factor = rng.normal(0, 0.01, (T, 1))
    betas  = rng.uniform(0.3, 1.2, (1, N_))
    idio   = rng.normal(0, 0.008, (T, N_))
    rets   = factor @ betas + idio

    signal = rng.normal(0, 1, N_)

    allocator = HRPAllocator(n_assets=N_, beta=0.25)

    async def run():
        result = await allocator.async_allocate(rets, signal)
        print(f"\nHRP Weights (top 5):")
        sorted_w = sorted(enumerate(result.weights), key=lambda x: -x[1])
        for i, w in sorted_w[:5]:
            ticker = TICKERS[i] if i < len(TICKERS) else f"Asset_{i}"
            print(f"  {ticker:6s}: {w:.4f}")
        print(f"\nShrinkage δ*:  {result.shrinkage_coeff:.4f}")
        print(f"Condition κ:   {result.condition_number:.1f}")
        print(f"Sum of weights: {result.weights.sum():.6f}")
        print(f"Max weight:     {result.weights.max():.4f}")

    asyncio.run(run())