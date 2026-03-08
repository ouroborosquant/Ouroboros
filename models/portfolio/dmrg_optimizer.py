"""
FORTRESS v5 — dmrg_optimizer.py  [P1 UPGRADE — Full OSQP Path]
Path: models/portfolio/dmrg_optimizer.py

Quantum-Inspired Tensor Network Portfolio Optimizer.

Architecture:
  Two solver tiers gated by covariance condition number:

  TIER 1 — OSQP fast path (normal market):
    Condition number cond(Σ) ≤ cond_threshold (default 1e4, aligned with
    the hardened _cov() in run_standalone_backtest.py post-ridge).
    OSQP (Operator Splitting QP) solves the Markowitz QP directly:
      min   0.5 λ w^T Σ w  −  μ^T w
      s.t.  Σ_i w_i = 1
            0 ≤ w_i ≤ ub_i  ∀ i

    OSQP is an ADMM-based first-order solver. Advantages over SLSQP here:
      - O(N²) per iteration vs O(N³) for SLSQP's sequential QP factorisation.
      - Warm-start: subsequent calls reuse the previous solution as x0,
        exploiting temporal autocorrelation in portfolio weights (~0.85 daily).
      - Infeasibility certificate: OSQP returns a structured status code
        rather than an ambiguous convergence flag, enabling targeted fallback.
      - No gradient computation required — pure primal-dual for bound-constrained QP.

  TIER 2 — DMRG-regularised OSQP (crisis/ill-conditioned market):
    Condition number cond(Σ) > cond_threshold.
    Applies quantum-inspired eigenvalue truncation before solving:
      1. Eigen-decompose Σ = V D V^T.
      2. Truncate noisy small eigenvalues (< truncation_tol × max_eigenvalue).
         This mimics DMRG bond-dimension truncation in tensor networks —
         retaining only the K dominant eigenvalue components captures the
         K largest principal risk factors, discarding noise.
      3. Reconstruct Σ_reg = V D_trunc V^T + ε I  (ε = reg_floor for PSD).
      4. Solve regularised QP via OSQP.

    Why eigenvalue truncation rather than ridge?
      Ridge adds uniform inflation ε·I, inflating ALL eigenvalues equally.
      Truncation zeros the genuinely-noisy small eigenvalues and leaves the
      dominant factors untouched. For portfolio optimisation this matters:
      ridge inflates small-eigenvalue directions into equal-risk contributors,
      smearing the portfolio toward 1/N. Truncation preserves the risk
      factor structure of dominant eigenvalues.

  FALLBACK — inverse-volatility weights:
    If both OSQP solvers fail (status not in {'solved', 'solved_inaccurate'}),
    returns inverse-vol weights clipped to per-asset bounds. Identical logic
    to the fallback in _mvo_weights() in run_standalone_backtest.py for
    internal consistency.

OSQP QP formulation:
  Variables:     w ∈ R^N
  Objective:     min  0.5 w^T P w  +  q^T w
                   P = λ Σ_ann  (annualised covariance, already scaled)
                   q = -μ        (negative expected returns / alpha signal)
  Constraints:   l ≤ A w ≤ u
                   Row 0: Σw = 1  → A[0,:] = 1^T, l[0]=u[0]=1.0
                   Rows 1..N: 0 ≤ w_i ≤ ub_i
                              A[1:,:] = I_N, l[1:]=0, u[1:]=ub

  All matrices passed as sparse CSC for OSQP's native format.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp

logger = logging.getLogger("DMRGOptimizer")

# ── OSQP import (required) ────────────────────────────────────────────────────
try:
    import osqp
    _OSQP_AVAILABLE = True
except ImportError:
    _OSQP_AVAILABLE = False
    logger.warning(
        "osqp not installed. Install via: pip install osqp. "
        "DMRGOptimizer will use scipy SLSQP fallback for ALL calls."
    )

# ── scipy fallback import ────────────────────────────────────────────────────
import scipy.optimize as sco


class TensorNetworkOptimizer:
    """
    Regime-gated portfolio optimizer combining DMRG-inspired covariance
    regularization with a high-performance OSQP QP solver.

    Replaces the original SLSQP-only implementation. OSQP is the primary
    solver at both tiers; SLSQP activates only if osqp is not installed.

    State:
      _warm_weights: last successful solution, used as OSQP warm-start.
                     Reset to None on regime transition (condition number
                     crossing cond_threshold) to avoid stale primal seeding.
    """

    def __init__(self, config: Dict) -> None:
        # Aligned with _cov() ridge threshold in run_standalone_backtest.py P1 fix
        self.cond_threshold:   float = config.get("condition_number_threshold", 1e4)
        self.bond_dim:         int   = config.get("bond_dim", 16)
        self.n_sweeps:         int   = config.get("n_sweeps", 10)
        # Truncation: fraction of max eigenvalue below which we zero
        self.truncation_tol:   float = config.get("truncation_threshold", 1e-4)
        # PSD regularization floor applied after truncation
        self.reg_floor:        float = config.get("reg_floor", 1e-8)
        self.default_risk_aversion: float = config.get("risk_aversion", 2.0)

        # OSQP solver settings
        self._osqp_settings: Dict = {
            "warm_starting":   True,
            "eps_abs":         1e-8,
            "eps_rel":         1e-8,
            "max_iter":        10_000,
            "verbose":         False,
            "polish":          True,     # Final LAPACK refinement — near machine precision
            "polish_refine_iter": 3,
        }

        # Warm-start state: persists between sequential calls on the same date series
        self._warm_weights: Optional[np.ndarray] = None
        self._prev_was_dmrg: bool = False  # track tier transitions

    # ── Public API ────────────────────────────────────────────────────────────

    def optimize(
        self,
        expected_returns: np.ndarray,   # (N,) alpha signal
        cov_matrix:       np.ndarray,   # (N,N) daily covariance
        bounds:           Tuple,        # sequence of (lb, ub) per asset
        risk_aversion:    Optional[float] = None,
    ) -> np.ndarray:
        """
        Main entry point.

        Dispatches to:
          _osqp_optimize(mu, Σ_reg, bounds)    — TIER 1 or TIER 2
          _scipy_fallback(mu, Σ_reg, bounds)   — if OSQP unavailable

        Args:
            expected_returns: alpha signal vector μ ∈ R^N.
            cov_matrix:       daily covariance Σ ∈ R^{N×N}. Will be
                              annualised internally (×252).
            bounds:           list/tuple of (lb, ub) per asset.
            risk_aversion:    overrides config default for this call.

        Returns:
            w ∈ R^N  s.t. Σw=1, 0≤w_i≤ub_i.
        """
        lam = risk_aversion if risk_aversion is not None else self.default_risk_aversion
        N   = len(expected_returns)

        cond_num  = self._condition_number(cov_matrix)
        use_dmrg  = cond_num > self.cond_threshold

        # Reset warm-start on tier transition to prevent stale primal pollution
        if use_dmrg != self._prev_was_dmrg:
            logger.info(
                f"Solver tier transition: "
                f"{'OSQP→DMRG-OSQP' if use_dmrg else 'DMRG-OSQP→OSQP'} "
                f"(cond={cond_num:.1f})"
            )
            self._warm_weights = None
        self._prev_was_dmrg = use_dmrg

        if use_dmrg:
            logger.warning(
                f"Covariance cond={cond_num:.1f} > {self.cond_threshold:.0f}. "
                f"Activating DMRG eigenvalue truncation (bond_dim={self.bond_dim})."
            )
            cov_reg = self._dmrg_regularize(cov_matrix)
        else:
            cov_reg = cov_matrix

        # Annualise
        Σ_ann = cov_reg * 252.0

        if _OSQP_AVAILABLE:
            return self._osqp_optimize(expected_returns, Σ_ann, bounds, lam, N)
        else:
            return self._scipy_fallback(expected_returns, Σ_ann, bounds, lam, N)

    # ── TIER-2: DMRG eigenvalue truncation ───────────────────────────────────

    def _dmrg_regularize(self, cov: np.ndarray) -> np.ndarray:
        """
        Quantum-inspired rank truncation of the covariance matrix.

        Mimics DMRG bond-dimension truncation in 1D tensor networks:
          - Interpret the covariance as a 1D MPO (matrix product operator)
            representing the risk Hamiltonian H = λΣ.
          - The singular value decomposition of Σ gives the "entanglement
            spectrum" of the risk structure.
          - Retaining bond_dim largest singular values is equivalent to
            keeping the bond_dim dominant risk factors.

        Implementation:
          eigen-decompose Σ = V D V^T  (symmetric → real eigenvalues)
          D_trunc[i] = D[i] if D[i] ≥ truncation_tol × max(D) else 0
          Σ_reg = V D_trunc V^T + reg_floor × I

        The reg_floor ε ensures strict positive definiteness. Its value
        (1e-8) is below the noise floor of any real asset covariance
        (daily variance of a 1bps/day asset ≈ 1e-8), so it is analytically
        invisible to the optimizer yet guarantees Cholesky existence.
        """
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # eigh returns ascending order — max is last
        max_eigenval = float(eigenvalues[-1])
        abs_threshold = self.truncation_tol * max(max_eigenval, 1e-10)

        # Zero subthreshold eigenvalues (DMRG bond truncation)
        eigenvalues_trunc = np.where(eigenvalues >= abs_threshold, eigenvalues, 0.0)

        # Retain at most bond_dim non-zero eigenvalues
        nonzero_mask = eigenvalues_trunc > 0.0
        n_nonzero    = int(nonzero_mask.sum())
        if n_nonzero > self.bond_dim:
            # Keep the bond_dim largest — set the rest to zero
            sorted_nonzero_idxs = np.argsort(eigenvalues_trunc)
            cutoff_idx           = len(eigenvalues_trunc) - self.bond_dim
            eigenvalues_trunc[:cutoff_idx] = 0.0

        n_retained = int((eigenvalues_trunc > 0.0).sum())
        logger.info(
            f"DMRG truncation: {n_retained}/{len(eigenvalues)} eigenvalues retained "
            f"(threshold={abs_threshold:.2e}, bond_dim={self.bond_dim})"
        )

        cov_reg = (
            eigenvectors @ np.diag(eigenvalues_trunc) @ eigenvectors.T
            + self.reg_floor * np.eye(len(cov))
        )
        return cov_reg

    # ── TIER-1/2: OSQP QP solver ─────────────────────────────────────────────

    def _osqp_optimize(
        self,
        mu:     np.ndarray,
        Σ_ann:  np.ndarray,
        bounds: Tuple,
        lam:    float,
        N:      int,
    ) -> np.ndarray:
        """
        Solves the MVO QP via OSQP (ADMM-based, first-order).

        QP formulation (see module docstring):
          P = λ Σ_ann   (symmetric positive semidefinite)
          q = -μ
          A = [1^T ; I_N]  as CSC sparse
          l = [1.0 ; 0_{N}]
          u = [1.0 ; ub]

        Warm-start: initialises primal from the previous solution.
        On OSQP status 'solved_inaccurate', logs a warning but accepts
        the result — in practice polish=True renders this indistinguishable
        from 'solved' for well-conditioned problems.
        """
        lb_arr  = np.array([b[0] for b in bounds], dtype=np.float64)
        ub_arr  = np.array([b[1] for b in bounds], dtype=np.float64)

        # P matrix (upper triangular CSC — OSQP convention)
        P_dense = lam * Σ_ann
        P_upper = np.triu(P_dense)
        P_csc   = sp.csc_matrix(P_upper)

        q       = -mu.astype(np.float64)

        # Constraint matrix A: stack [1^T ; I_N]
        ones_row = sp.csc_matrix(np.ones((1, N), dtype=np.float64))
        eye_N    = sp.eye(N, format="csc", dtype=np.float64)
        A_csc    = sp.vstack([ones_row, eye_N], format="csc")

        l = np.concatenate([[1.0], lb_arr])
        u = np.concatenate([[1.0], ub_arr])

        solver = osqp.OSQP()
        solver.setup(P_csc, q, A_csc, l, u, **self._osqp_settings)

        # Warm-start from previous solution
        if self._warm_weights is not None and len(self._warm_weights) == N:
            # Dual variable warm-start: zeros is safe when primal is warm-started
            solver.warm_start(x=self._warm_weights, y=np.zeros(N + 1))

        result = solver.solve()
        status = result.info.status

        if status in ("solved", "solved_inaccurate"):
            if status == "solved_inaccurate":
                logger.warning(
                    f"OSQP: solved_inaccurate (obj={result.info.obj_val:.6f}). "
                    f"Accepting polished result."
                )
            w = np.clip(result.x, lb_arr, ub_arr)
            w_sum = w.sum()
            if w_sum > 1e-10:
                w /= w_sum
            self._warm_weights = w.copy()
            return w
        else:
            logger.error(
                f"OSQP failed: status='{status}'. "
                f"Falling back to inverse-vol weights."
            )
            self._warm_weights = None
            return self._inverse_vol_fallback(Σ_ann, ub_arr, N)

    # ── scipy SLSQP fallback (when OSQP not installed) ───────────────────────

    def _scipy_fallback(
        self,
        mu:     np.ndarray,
        Σ_ann:  np.ndarray,
        bounds: Tuple,
        lam:    float,
        N:      int,
    ) -> np.ndarray:
        """
        SLSQP fallback — only activated when osqp is not installed.
        Uses the feasibility-guaranteed x0 from the P1 fix.
        """
        ub_arr = np.array([b[1] for b in bounds], dtype=np.float64)
        lb_arr = np.array([b[0] for b in bounds], dtype=np.float64)

        # Feasibility-guaranteed x0: clip 1/N to bounds, renormalise onto simplex
        x0 = np.clip(np.full(N, 1.0 / N), lb_arr, ub_arr)
        x0_sum = x0.sum()
        x0 = x0 / x0_sum if x0_sum > 1e-10 else (
            np.where(ub_arr > 0, ub_arr / (ub_arr.sum() + 1e-10), 0.0)
        )

        res = sco.minimize(
            fun=lambda w: 0.5 * lam * w @ Σ_ann @ w - mu @ w,
            jac=lambda w: lam * Σ_ann @ w - mu,
            x0=x0,
            method="SLSQP",
            bounds=list(zip(lb_arr.tolist(), ub_arr.tolist())),
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
            options={"ftol": 1e-9, "maxiter": 500},
        )
        if res.success:
            w = np.clip(res.x, lb_arr, ub_arr)
            w /= (w.sum() + 1e-10)
            self._warm_weights = w.copy()
            return w
        else:
            logger.error(
                f"SLSQP fallback failed: {res.message}. "
                f"Returning inverse-vol weights."
            )
            return self._inverse_vol_fallback(Σ_ann, ub_arr, N)

    # ── Shared utilities ──────────────────────────────────────────────────────

    def _condition_number(self, cov: np.ndarray) -> float:
        """
        Condition number with a tiny ridge to avoid divide-by-zero on
        singular matrices (e.g. initial warmup with fewer returns than assets).
        """
        ridge_cov = cov + np.eye(cov.shape[0]) * 1e-10
        return float(np.linalg.cond(ridge_cov))

    @staticmethod
    def _inverse_vol_fallback(
        Σ_ann:  np.ndarray,
        ub_arr: np.ndarray,
        N:      int,
    ) -> np.ndarray:
        """
        Inverse-vol weights clipped to per-asset bounds.
        Consistent with the fallback in _mvo_weights() of run_standalone_backtest.py.
        """
        vols = np.sqrt(np.maximum(np.diag(Σ_ann), 1e-10))
        iv   = np.minimum(1.0 / vols, ub_arr * 3.0)
        iv   = np.clip(iv, 0.0, ub_arr)
        iv_s = iv.sum()
        return iv / iv_s if iv_s > 1e-10 else np.where(
            ub_arr > 0, 1.0 / max((ub_arr > 0).sum(), 1), 0.0
        )

    def is_activated(self, cov_matrix: np.ndarray) -> bool:
        """
        Public predicate: True if DMRG regularisation is active.
        Kept for backward compatibility with any callers that check
        the regime tier before calling optimize().
        """
        return self._condition_number(cov_matrix) > self.cond_threshold