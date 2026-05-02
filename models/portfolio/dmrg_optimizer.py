"""
FORTRESS v5 — dmrg_optimizer.py  [P1 UPGRADE — Full OSQP Path + Constraints]
Path: models/portfolio/dmrg_optimizer.py

Quantum-Inspired Tensor Network Portfolio Optimizer.

Updates:
  - Added native support for linear inequality group bounds in `optimize()`.
  - Maps transparently into OSQP's sparse (CSC) representation.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple, List

import numpy as np
import scipy.sparse as sp

logger = logging.getLogger("DMRGOptimizer")

try:
    import osqp
    _OSQP_AVAILABLE = True
except ImportError:
    _OSQP_AVAILABLE = False
    logger.warning(
        "osqp not installed. Install via: pip install osqp. "
        "DMRGOptimizer will use scipy SLSQP fallback for ALL calls."
    )

import scipy.optimize as sco

class TensorNetworkOptimizer:
    def __init__(self, config: Dict) -> None:
        self.cond_threshold:   float = config.get("condition_number_threshold", 1e4)
        self.bond_dim:         int   = config.get("bond_dim", 16)
        self.n_sweeps:         int   = config.get("n_sweeps", 10)
        self.truncation_tol:   float = config.get("truncation_threshold", 1e-4)
        self.reg_floor:        float = config.get("reg_floor", 1e-8)
        self.default_risk_aversion: float = config.get("risk_aversion", 2.0)

        self._osqp_settings: Dict = {
            "warm_starting":   True,
            "eps_abs":         1e-8,
            "eps_rel":         1e-8,
            "max_iter":        10_000,
            "verbose":         False,
            "polish":          True,
            "polish_refine_iter": 3,
        }

        self._warm_weights: Optional[np.ndarray] = None
        self._prev_was_dmrg: bool = False

    def optimize(
        self,
        expected_returns: np.ndarray,
        cov_matrix:       np.ndarray,
        bounds:           Tuple,
        risk_aversion:    Optional[float] = None,
        cash_indices:     Optional[List[int]] = None,
        max_cash:         float = 0.10
    ) -> np.ndarray:
        """
        Args:
            expected_returns: alpha signal vector μ ∈ R^N.
            cov_matrix:       daily covariance Σ ∈ R^{N×N}. 
            bounds:           list/tuple of (lb, ub) per asset.
            risk_aversion:    overrides config default for this call.
            cash_indices:     Indices of cash proxy assets (e.g. BIL, SHV).
            max_cash:         Upper bound for total allocation across cash proxies.
        """
        lam = risk_aversion if risk_aversion is not None else self.default_risk_aversion
        N   = len(expected_returns)

        cond_num  = self._condition_number(cov_matrix)
        use_dmrg  = cond_num > self.cond_threshold

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

        Σ_ann = cov_reg * 252.0

        if _OSQP_AVAILABLE:
            return self._osqp_optimize(expected_returns, Σ_ann, bounds, lam, N, cash_indices, max_cash)
        else:
            return self._scipy_fallback(expected_returns, Σ_ann, bounds, lam, N, cash_indices, max_cash)

    def _dmrg_regularize(self, cov: np.ndarray) -> np.ndarray:
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        max_eigenval = float(eigenvalues[-1])
        abs_threshold = self.truncation_tol * max(max_eigenval, 1e-10)

        eigenvalues_trunc = np.where(eigenvalues >= abs_threshold, eigenvalues, 0.0)
        nonzero_mask = eigenvalues_trunc > 0.0
        n_nonzero    = int(nonzero_mask.sum())
        if n_nonzero > self.bond_dim:
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

    def _osqp_optimize(
        self,
        mu:           np.ndarray,
        Σ_ann:        np.ndarray,
        bounds:       Tuple,
        lam:          float,
        N:            int,
        cash_indices: Optional[List[int]],
        max_cash:     float
    ) -> np.ndarray:

        lb_arr  = np.array([b[0] for b in bounds], dtype=np.float64)
        ub_arr  = np.array([b[1] for b in bounds], dtype=np.float64)

        P_dense = lam * Σ_ann
        P_upper = np.triu(P_dense)
        P_csc   = sp.csc_matrix(P_upper)

        q       = -mu.astype(np.float64)

        ones_row = sp.csc_matrix(np.ones((1, N), dtype=np.float64))
        eye_N    = sp.eye(N, format="csc", dtype=np.float64)
        A_csc    = sp.vstack([ones_row, eye_N], format="csc")

        l = np.concatenate([[1.0], lb_arr])
        u = np.concatenate([[1.0], ub_arr])

        # Enforce Cash constraint
        if cash_indices:
            cash_row = sp.csc_matrix((np.ones(len(cash_indices)), 
                                      (np.zeros(len(cash_indices)), cash_indices)), 
                                     shape=(1, N), dtype=np.float64)
            A_csc = sp.vstack([A_csc, cash_row], format="csc")
            l = np.concatenate([l, [0.0]])
            u = np.concatenate([u, [max_cash]])

        solver = osqp.OSQP()
        solver.setup(P_csc, q, A_csc, l, u, **self._osqp_settings)

        if self._warm_weights is not None and len(self._warm_weights) == N:
            # Match the new constraint dimensions for dual initialization
            y_dim = N + 1 + (1 if cash_indices else 0)
            solver.warm_start(x=self._warm_weights, y=np.zeros(y_dim))

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

    def _scipy_fallback(
        self,
        mu:           np.ndarray,
        Σ_ann:        np.ndarray,
        bounds:       Tuple,
        lam:          float,
        N:            int,
        cash_indices: Optional[List[int]],
        max_cash:     float
    ) -> np.ndarray:

        ub_arr = np.array([b[1] for b in bounds], dtype=np.float64)
        lb_arr = np.array([b[0] for b in bounds], dtype=np.float64)

        x0 = np.clip(np.full(N, 1.0 / N), lb_arr, ub_arr)
        x0_sum = x0.sum()
        x0 = x0 / x0_sum if x0_sum > 1e-10 else (
            np.where(ub_arr > 0, ub_arr / (ub_arr.sum() + 1e-10), 0.0)
        )

        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
        if cash_indices:
            constraints.append({"type": "ineq", "fun": lambda w: max_cash - w[cash_indices].sum()})

        res = sco.minimize(
            fun=lambda w: 0.5 * lam * w @ Σ_ann @ w - mu @ w,
            jac=lambda w: lam * Σ_ann @ w - mu,
            x0=x0,
            method="SLSQP",
            bounds=list(zip(lb_arr.tolist(), ub_arr.tolist())),
            constraints=constraints,
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

    def _condition_number(self, cov: np.ndarray) -> float:
        ridge_cov = cov + np.eye(cov.shape[0]) * 1e-10
        return float(np.linalg.cond(ridge_cov))

    @staticmethod
    def _inverse_vol_fallback(
        Σ_ann:  np.ndarray,
        ub_arr: np.ndarray,
        N:      int,
    ) -> np.ndarray:
        vols = np.sqrt(np.maximum(np.diag(Σ_ann), 1e-10))
        iv   = np.minimum(1.0 / vols, ub_arr * 3.0)
        iv   = np.clip(iv, 0.0, ub_arr)
        iv_s = iv.sum()
        return iv / iv_s if iv_s > 1e-10 else np.where(
            ub_arr > 0, 1.0 / max((ub_arr > 0).sum(), 1), 0.0
        )

    def is_activated(self, cov_matrix: np.ndarray) -> bool:
        return self._condition_number(cov_matrix) > self.cond_threshold