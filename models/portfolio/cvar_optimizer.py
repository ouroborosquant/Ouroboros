"""
fortress_v5/portfolio/cvar_optimizer.py
────────────────────────────────────────
CVaR Mean-Variance Optimizer with L1 Turnover Friction.

Objective (Rockafellar-Uryasev, 2000):

    min  -μᵀw  +  λ_cvar · CVaR_α(w)  +  γ · ‖w - w_{t-1}‖₁
    s.t. 1ᵀw = 1,  w ≥ 0,  (optional sector constraints)

CVaR LP reformulation:
    CVaR_α(w) = ξ + 1/((1-α)·T) · Σ_t z_t
    z_t ≥ -Rₜᵀw - ξ,   z_t ≥ 0

L1 linearization (auxiliary variables u⁺, u⁻ ≥ 0):
    ‖w - w_prev‖₁  =  Σᵢ (uᵢ⁺ + uᵢ⁻)
    w - w_prev      =  u⁺ - u⁻

The friction threshold γ must satisfy:
    γ > |∂μᵀw/∂wᵢ|  for rotation to trigger — i.e. net alpha improvement
    per unit turnover must exceed γ.  Calibrate γ ≈ 2× round-trip cost.

Solver: Clarabel (default cvxpy ≥ 1.3) — faster than ECOS for conic programs
        with warm-start support via solver_cache.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cvxpy as cp
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class OptimResult:
    weights: np.ndarray               # (N,) optimal weights
    expected_return: float
    cvar: float                        # CVaR at confidence α
    turnover: float                    # L1 turnover vs w_prev
    solve_status: str
    solver_stats: Dict


class CVaRMVOOptimizer:
    """
    CVaR-regularized MVO with L1 turnover penalty.

    The portfolio only rotates when net alpha improvement exceeds the friction
    threshold γ — this eliminates the low-conviction oscillation ("cash trap")
    observed in pure MVO with small alpha differentials.

    Parameters
    ----------
    n_assets    : N — universe size (fixed at construction for warm-start caching)
    alpha       : CVaR tail probability (0.95 = ES at 95th percentile)
    lambda_cvar : CVaR aversion weight
    gamma       : L1 turnover penalty (≈ 2 × round-trip slippage + commission)
    weight_lb   : per-asset lower bound (0 = long-only)
    weight_ub   : per-asset upper bound
    solver      : cvxpy solver string ("CLARABEL" preferred)
    max_iter    : solver iteration cap
    """

    def __init__(
        self,
        n_assets: int,
        alpha: float = 0.95,
        lambda_cvar: float = 0.5,
        gamma: float = 0.003,          # 30bps per unit turnover → ~15bps 1-way
        weight_lb: float = 0.0,
        weight_ub: float = 0.25,       # max 25% single-name concentration
        solver: str = "CLARABEL",
        max_iter: int = 10_000,
    ) -> None:
        self.N = n_assets
        self.alpha = alpha
        self.lambda_cvar = lambda_cvar
        self.gamma = gamma
        self.weight_lb = weight_lb
        self.weight_ub = weight_ub
        self.solver = solver
        self.max_iter = max_iter

        # Pre-allocate cvxpy variable graph — rebuilt only when N changes
        self._build_problem()
        self._prev_weights: np.ndarray = np.ones(n_assets) / n_assets

    # ── Problem construction ───────────────────────────────────────────────

    def _build_problem(self) -> None:
        """
        Construct the parametric cvxpy problem once. Parameters (μ, R, w_prev)
        are injected at solve-time via cp.Parameter — avoids recompilation.
        """
        N = self.N

        # ── Decision variables ──────────────────────────────────────────
        self.w      = cp.Variable(N, nonneg=True, name="w")          # weights
        self.xi     = cp.Variable(name="xi")                          # VaR auxiliary
        self.z      = cp.Variable(name="z_cvar")                      # will be reshaped
        self.u_plus = cp.Variable(N, nonneg=True, name="u_plus")      # L1 auxiliary +
        self.u_minus= cp.Variable(N, nonneg=True, name="u_minus")     # L1 auxiliary -

        # ── Parameters (injected per solve call) ────────────────────────
        self.p_mu     = cp.Parameter(N, name="mu")                    # expected returns
        self.p_R      = cp.Parameter(shape=(1, N), name="R")          # scenario matrix (T,N) — resized at solve time
        self.p_w_prev = cp.Parameter(N, nonneg=True, name="w_prev")   # prior weights
        self.p_T      = cp.Parameter(pos=True, name="T_scen")         # scenario count

        # NOTE: The scenario matrix size (T) varies — we rebuild constraints
        # per solve call when T changes. Cache last T for efficiency.
        self._last_T: int = -1
        self._problem: Optional[cp.Problem] = None

    def _rebuild_for_T(self, T: int) -> None:
        """Rebuild problem with correct scenario dimension T."""
        N = self.N
        w, xi, u_plus, u_minus = self.w, self.xi, self.u_plus, self.u_minus

        z_t = cp.Variable(T, nonneg=True, name="z_scenarios")        # CVaR slacks
        self._z_t = z_t

        # ── CVaR objective term: ξ + 1/((1-α)·T) · 1ᵀz ────────────────
        cvar_term = xi + cp.sum(z_t) / ((1 - self.alpha) * T)

        # ── Return term ─────────────────────────────────────────────────
        ret_term = self.p_mu @ w

        # ── L1 turnover term ────────────────────────────────────────────
        turnover_term = cp.sum(u_plus + u_minus)

        objective = cp.Minimize(
            -ret_term
            + self.lambda_cvar * cvar_term
            + self.gamma * turnover_term
        )

        # ── Constraints ─────────────────────────────────────────────────
        # Reuse p_R parameter — will be assigned (T,N) ndarray at solve time
        self.p_R = cp.Parameter((T, N), name="R")

        constraints = [
            # CVaR scenario constraints: z_t ≥ -Rₜᵀw - ξ
            z_t >= -self.p_R @ w - xi,

            # L1 linearization: w - w_prev = u⁺ - u⁻
            w - self.p_w_prev == u_plus - u_minus,

            # Simplex: fully invested
            cp.sum(w) == 1.0,

            # Box constraints
            w >= self.weight_lb,
            w <= self.weight_ub,
        ]

        self._problem = cp.Problem(objective, constraints)
        self._last_T = T

    # ── Public API ─────────────────────────────────────────────────────────

    def solve(
        self,
        mu: np.ndarray,                    # (N,) expected returns
        scenario_returns: np.ndarray,      # (T, N) historical/simulated returns
        w_prev: Optional[np.ndarray] = None,
        sector_constraints: Optional[Dict[str, Tuple[np.ndarray, float, float]]] = None,
        exposure_multiplier: float = 1.0,   # from WassersteinHMM
    ) -> OptimResult:
        """
        Solve the CVaR-MVO problem.

        Parameters
        ----------
        mu                  : (N,) alpha signal (annualized expected return)
        scenario_returns    : (T, N) scenario matrix — e.g. rolling 252-day window
        w_prev              : (N,) previous weights for turnover penalty
        sector_constraints  : dict of {name: (membership_vector, lb, ub)}
        exposure_multiplier : regime-gated scalar ∈ [0,1]. If < 1, cash buffer
                              (1 - multiplier) is forced into BIL (index 0 assumed)
        """
        mu               = np.asarray(mu,               dtype=float)
        scenario_returns = np.asarray(scenario_returns, dtype=float)
        T, N             = scenario_returns.shape

        assert N == self.N, f"Universe mismatch: got {N}, expected {self.N}"

        if w_prev is None:
            w_prev = self._prev_weights
        w_prev = np.asarray(w_prev, dtype=float)
        w_prev = np.clip(w_prev / w_prev.sum(), 0.0, 1.0)

        # Rebuild problem graph only when scenario count changes
        if T != self._last_T:
            self._rebuild_for_T(T)

        # ── Inject parameter values ──────────────────────────────────────
        self.p_mu.value     = mu
        self.p_R.value      = scenario_returns
        self.p_w_prev.value = w_prev

        # ── Exposure multiplier: cap total risky weight ─────────────────
        extra_constraints: list[cp.Constraint] = []
        if exposure_multiplier < 0.999:
            cash_floor = 1.0 - exposure_multiplier
            # BIL (index=-1 convention; caller must ensure cash asset ordering)
            extra_constraints.append(self.w[-1] >= cash_floor)

        # ── Optional sector constraints ──────────────────────────────────
        if sector_constraints:
            for name, (mask, lb, ub) in sector_constraints.items():
                sector_w = self.w @ np.asarray(mask, dtype=float)
                extra_constraints += [sector_w >= lb, sector_w <= ub]

        # Rebuild with extra constraints if needed
        if extra_constraints:
            prob = cp.Problem(
                self._problem.objective,
                self._problem.constraints + extra_constraints
            )
        else:
            prob = self._problem

        # ── Solve ────────────────────────────────────────────────────────
        solver_opts = {
            "max_iter": self.max_iter,
            "eps_abs":  1e-7,
            "eps_rel":  1e-7,
        }
        try:
            prob.solve(
                solver   = self.solver,
                warm_start = True,
                **solver_opts,
            )
        except cp.SolverError as exc:
            log.warning("Clarabel failed (%s), falling back to SCS", exc)
            prob.solve(solver="SCS", warm_start=True)

        status = prob.status
        if status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            log.error("Optimizer failed: status=%s — returning prior weights", status)
            return OptimResult(
                weights=w_prev, expected_return=float(mu @ w_prev),
                cvar=np.nan, turnover=0.0,
                solve_status=status, solver_stats={},
            )

        w_opt     = np.clip(self.w.value, 0.0, None)
        w_opt    /= w_opt.sum()                                       # re-normalize numerical noise
        turnover  = float(np.abs(w_opt - w_prev).sum())
        er        = float(mu @ w_opt)
        cvar_val  = float(self.xi.value + np.sum(self._z_t.value) / ((1 - self.alpha) * T))

        self._prev_weights = w_opt.copy()

        log.info(
            "CVaR-MVO solved: ER=%.4f CVaR=%.4f turnover=%.4f γ=%.4f status=%s",
            er, cvar_val, turnover, self.gamma, status
        )

        return OptimResult(
            weights         = w_opt,
            expected_return = er,
            cvar            = cvar_val,
            turnover        = turnover,
            solve_status    = status,
            solver_stats    = prob.solver_stats.__dict__ if prob.solver_stats else {},
        )

    def calibrate_gamma(
        self,
        round_trip_cost_bps: float = 15.0,
        safety_multiple: float = 2.0,
    ) -> float:
        """
        Compute γ from round-trip transaction cost in basis points.
        γ = safety_multiple × (round_trip_cost / 10000).
        Rotation only fires when net alpha improvement > γ per unit turnover.
        """
        gamma = safety_multiple * round_trip_cost_bps / 10_000.0
        self.gamma = gamma
        log.info("γ calibrated to %.5f (%.1f bps × %.1fx)", gamma, round_trip_cost_bps, safety_multiple)
        return gamma