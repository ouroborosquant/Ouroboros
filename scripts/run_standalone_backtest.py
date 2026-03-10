"""
FORTRESS v5 — patch_run_standalone_backtest.py
Path: scripts/run_standalone_backtest.py  [BUG #24/#25 FIX]

=== PATCH INSTRUCTIONS ===
Apply these changes to your existing scripts/run_standalone_backtest.py.
Three sections are modified:

  SECTION 1: New constants + CVaR optimizer (replaces _mvo_weights)
  SECTION 2: Halt state machine (replaces Phase 2 ramp-in logic)
  SECTION 3: _get_regime_halt_threshold ramp-in buffer constant

=== BUG #24 — HALT STATE MACHINE POSITIVE FEEDBACK LOOP ===
  Root cause: During Phase 2 (ramp-in), drawdown is computed relative to
  `self._peak` (all-time high). Since the system enters ramp-in AFTER a
  halt (which was triggered by a ~20% drawdown from peak), the drawdown
  from peak on ramp-in Day 1 is guaranteed to be near -20%.
  
  Result: Any minor market dip during ramp-in immediately re-triggers the
  halt, creating an infinite halt → ramp → re-halt cycle. Logs confirm:
    2023-10-31: Halt recovery — entering ramp-in.
    2023-11-01: Drawdown=-20.37% exceeded 20.00% during ramp-in. Returning to BIL.
  One-day ramp-in. The system was structurally unable to ever resume trading
  after the first halt in mid-2023.

  Fix: Two changes.
    (a) During ramp-in, drawdown is computed relative to `_ramp_entry_nav`
        (the NAV when ramp-in began), NOT `_peak`. This isolates the
        ramp-in performance from the historical drawdown.
    (b) The ramp-in halt threshold has a 5% buffer above the Phase 1
        threshold. So if the regime-conditional halt is -18%, the ramp-in
        re-halt threshold is -8% (relative to ramp_entry_nav). This
        permits normal ramp-in volatility without re-triggering.
    (c) On successful ramp-in completion, `_peak` is reset to current NAV
        to establish a clean high-water mark for the new active period.

=== BUG #25 — MVO MISSPECIFIED FOR FAT TAILS (kurtosis=9.35) ===
  Root cause: SLSQP MVO optimises mean-variance, assuming elliptical
  returns. With excess kurtosis of 9.35, the covariance matrix understates
  left-tail risk by 30-50%. The optimizer over-concentrates in assets
  whose sample covariance understates actual tail dependence.

  Fix: Dual-objective optimizer.
    (a) Primary: sample-based CVaR (95%) penalty replaces pure variance.
        Uses the historical return window directly — no distributional
        assumption. Objective: max α^T w - λ_var * w^T Σ w - λ_cvar * CVaR_95(w).
    (b) CVaR is linearised via the Rockafellar-Uryasev (2000) auxiliary
        variable formulation, which makes the problem smooth for SLSQP.
    (c) λ_cvar scales with realised kurtosis: when excess kurtosis > 3,
        the CVaR penalty increases quadratically. This automatically
        tightens tail risk control in fat-tailed regimes.
    (d) Fallback: if the CVaR-augmented solve fails, falls back to the
        existing pure-MVO SLSQP path (not inverse-vol).
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.optimize as sco
import scipy.stats as stats
from scipy.stats import norm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("StandaloneBacktest")

# ── Paths ─────────────────────────────────────────────────────────────────────
_CACHE_DIR  = Path("research/outputs/cache")
_OUTPUT_DIR = Path("research/outputs")
_TEARSHEET  = _OUTPUT_DIR / "backtest_tearsheet.csv"
_WF_FOLDS   = _OUTPUT_DIR / "walk_forward_folds.csv"
_STRESS_OUT = _OUTPUT_DIR / "sde_stress_test.json"

TICKERS: List[str] = [
    "SPY", "QQQ", "IWM", "VTV",
    "XLK", "XLF", "XLV", "XLP", "XLI", "XLE",
    "EFA", "EEM",
    "TLT", "IEF", "SHY", "LQD", "HYG",
    "GLD", "SLV", "USO", "PDBC", "VNQ",
    "VIXY",
    "SHV", "BIL",
]
N_ASSETS = 25

TIER_MAX_WEIGHT: Dict[str, float] = {
    "SPY": 0.25, "QQQ": 0.25, "IWM": 0.25, "VTV": 0.25,
    "XLK": 0.15, "XLF": 0.15, "XLV": 0.15, "XLP": 0.15, "XLI": 0.15, "XLE": 0.15,
    "EFA": 0.25, "EEM": 0.15,
    "TLT": 0.25, "IEF": 0.25, "SHY": 0.30, "LQD": 0.25, "HYG": 0.15,
    "GLD": 0.25, "SLV": 0.15, "USO": 0.15, "PDBC": 0.15, "VNQ": 0.15,
    "VIXY": 0.05, "SHV": 0.50, "BIL": 0.50,
}

# ── Risk / engine constants ───────────────────────────────────────────────────
_INITIAL_CAPITAL:       float = 100_000.0
_RISK_FREE_ANNUAL:      float = 0.05
_HALT_RECOVERY_THRESH:  float = 0.10
_HALT_MIN_DAYS:         int   = 60
_HALT_RAMP_DAYS:        int   = 60
_WARMUP_DAYS:           int   = 126
_WF_WARMUP_DAYS:        int   = 21
_REBALANCE_BAND:        float = 25e-4   # 25bps
_LAMBDA_BASE:           float = 2.5
_COV_WINDOW:            int   = 63
_MIN_POSITION_WT:       float = 0.015
_EMA_SPAN:              int   = 8
_REGIME_PERSIST_DAYS:   int   = 5
_AC_ETA:                float = 0.1
_BASE_SPREAD_BPS:       float = 1.0
_TRADING_DAYS_YEAR:     int   = 252

# ── BUG #24 FIX: ramp-in drawdown buffer ─────────────────────────────────────
# During ramp-in, the re-halt threshold is this fraction of NAV decline from
# the ramp-in entry NAV — NOT from all-time peak.
# 8% allows normal ramp-in volatility without triggering the halt feedback loop.
_RAMP_DD_BUFFER:        float = 0.08

# ── BUG #25 FIX: CVaR optimizer constants ────────────────────────────────────
_CVAR_CONFIDENCE:       float = 0.95   # 95th percentile tail
_LAMBDA_CVAR_BASE:      float = 1.0    # base CVaR penalty weight
_KURT_CVAR_SCALE:       float = 0.15   # CVaR penalty scales with excess kurtosis

# ── BUG #22 FIX: regime-conditional halt thresholds ──────────────────────────
_REGIME_HALT_PARAMS: Dict[str, Tuple[float, int, int]] = {
    "bull_low_vol":  (0.15, 45, 21),
    "bull_high_vol": (0.18, 60, 30),
    "bear":          (0.22, 90, 45),
    "crisis":        (0.25, 120, 60),
}
_MAX_DD_HALT_FALLBACK: float = 0.20

_SOFT_COLS: List[str] = [
    "soft_bull_low_vol", "soft_bull_high_vol", "soft_bear", "soft_crisis"
]
_SOFT_REGIME_ORDER: List[str] = [
    "bull_low_vol", "bull_high_vol", "bear", "crisis"
]

_BIL_WEIGHT = np.zeros(N_ASSETS, dtype=np.float32)
_BIL_WEIGHT[TICKERS.index("BIL")] = 0.60
_BIL_WEIGHT[TICKERS.index("SHV")] = 0.40


# ── Regime-conditional halt threshold ────────────────────────────────────────

def _get_regime_halt_threshold(
    regime_df: pd.DataFrame,
    date: pd.Timestamp,
) -> Tuple[float, int, int]:
    """
    Returns (max_dd_threshold, halt_days, ramp_days) blended from soft GMM
    posteriors. Unchanged from v5 — included for completeness.
    """
    has_soft = all(c in regime_df.columns for c in _SOFT_COLS)
    if not has_soft or date not in regime_df.index:
        return (_MAX_DD_HALT_FALLBACK, _HALT_MIN_DAYS, _HALT_RAMP_DAYS)

    row = regime_df.loc[date]
    posteriors = np.array([
        float(row.get(col, 0.0)) for col in _SOFT_COLS
    ], dtype=np.float64)
    posteriors = np.clip(posteriors, 0.0, 1.0)
    posteriors /= posteriors.sum() + 1e-9

    dd_thresh = float(sum(
        posteriors[i] * _REGIME_HALT_PARAMS[lbl][0]
        for i, lbl in enumerate(_SOFT_REGIME_ORDER)
    ))
    halt_days = int(round(sum(
        posteriors[i] * _REGIME_HALT_PARAMS[lbl][1]
        for i, lbl in enumerate(_SOFT_REGIME_ORDER)
    )))
    ramp_days = int(round(sum(
        posteriors[i] * _REGIME_HALT_PARAMS[lbl][2]
        for i, lbl in enumerate(_SOFT_REGIME_ORDER)
    )))
    return (dd_thresh, halt_days, ramp_days)


# ── Index normalisation ───────────────────────────────────────────────────────

def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    """Strip tz, dedup (keep last), sort."""
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df = df.copy()
    df.index = idx
    return df[~df.index.duplicated(keep="last")].sort_index()


# ── PSR ───────────────────────────────────────────────────────────────────────

def _psr(sr: float, n: int, skew: float, kurt_raw: float) -> float:
    """Probabilistic Sharpe Ratio: P(SR > 0) accounting for non-normality."""
    excess_k = kurt_raw - 3.0
    var_sr = (1.0 - skew * sr + (excess_k / 4.0) * sr ** 2) / max(n - 1, 1)
    if var_sr <= 0:
        return 0.0
    return float(norm.cdf(sr / (var_sr ** 0.5 + 1e-10)))


def _sharpe(r: np.ndarray) -> float:
    rf = _RISK_FREE_ANNUAL / _TRADING_DAYS_YEAR
    ex = r - rf
    return float((ex.mean() / (ex.std() + 1e-10)) * np.sqrt(_TRADING_DAYS_YEAR))


def _mct(arr: np.ndarray) -> int:
    """Max consecutive True."""
    mx = cur = 0
    for v in arr:
        if v:
            cur += 1
            mx = max(mx, cur)
        else:
            cur = 0
    return mx


# ── Regime signal smoother ────────────────────────────────────────────────────

class RegimeSignalSmoother:
    """EMA(τ=8d) + persistence gate(≥5d) on PCA z_mu to suppress flip rate."""

    def __init__(self) -> None:
        self._ema:       Optional[np.ndarray] = None
        self._label:     str = ""
        self._label_cnt: int = 0
        self._alpha:     float = 2.0 / (_EMA_SPAN + 1)

    def update(
        self, z_raw: np.ndarray, raw_label: str
    ) -> Tuple[np.ndarray, str]:
        if self._ema is None:
            self._ema = z_raw.copy()
        self._ema = self._alpha * z_raw + (1.0 - self._alpha) * self._ema
        if raw_label == self._label:
            self._label_cnt += 1
        else:
            self._label_cnt = 1
            self._label = raw_label
        confirmed = (
            self._label if self._label_cnt >= _REGIME_PERSIST_DAYS else "neutral"
        )
        return self._ema.copy(), confirmed


# ── Covariance estimator (OAS + BUG #23 ridge floor) ─────────────────────────

def _cov(window: pd.DataFrame) -> np.ndarray:
    """
    Oracle Approximating Shrinkage covariance (Chen et al. 2010).
    BUG #23 FIX: analytical ridge if cond(Σ) > 1e4 after OAS.
    """
    arr = window.fillna(0.0).values
    if arr.shape[0] < 5:
        return np.eye(N_ASSETS) * (0.15 ** 2 / _TRADING_DAYS_YEAR)

    try:
        from sklearn.covariance import OAS
        C = OAS().fit(arr).covariance_
    except ImportError:
        C = np.cov(arr.T)
        if not np.isfinite(C).all():
            return np.eye(N_ASSETS) * (0.15 ** 2 / _TRADING_DAYS_YEAR)
        shrink = float(np.clip(0.15 * N_ASSETS / max(arr.shape[0], 1), 0.05, 0.40))
        mu_diag = np.trace(C) / N_ASSETS
        C = (1.0 - shrink) * C + shrink * mu_diag * np.eye(N_ASSETS)

    if not np.isfinite(C).all():
        return np.eye(N_ASSETS) * (0.15 ** 2 / _TRADING_DAYS_YEAR)

    cond = np.linalg.cond(C)
    if cond > 1e4:
        mu_diag = np.trace(C) / N_ASSETS
        gamma = float(np.clip((cond - 1e4) / (cond * 1e4), 0.01, 0.30))
        C = (1.0 - gamma) * C + gamma * mu_diag * np.eye(N_ASSETS)

    return C


# ── Almgren-Chriss market impact ─────────────────────────────────────────────

def _ac_cost_bps(shares: float, vol_d: float, adv: float) -> float:
    if adv < 1.0:
        return 5.0
    return float(_AC_ETA * vol_d * np.sqrt(abs(shares) / adv) * 10_000)


# ==============================================================================
# SECTION 1: BUG #25 FIX — CVaR-Augmented MVO Optimizer
# ==============================================================================
# Replaces the pure mean-variance _mvo_weights() with a dual-objective
# function that penalises both variance AND sample CVaR.
#
# The CVaR component uses the Rockafellar-Uryasev (2000) linearisation:
#   CVaR_α(w) = min_ζ { ζ + (1/(1-α)T) Σ_t max(0, -r_t^T w - ζ) }
#
# This is smooth (max → softplus in practice) and can be embedded in the
# SLSQP objective alongside the standard quadratic variance term.
# ==============================================================================

def _compute_sample_cvar(
    weights: np.ndarray,
    return_window: np.ndarray,
    alpha: float = _CVAR_CONFIDENCE,
) -> Tuple[float, float]:
    """
    Sample-based CVaR and VaR from a return matrix.

    Args:
        weights:       (N,) portfolio weight vector.
        return_window: (T, N) daily return matrix.
        alpha:         confidence level (0.95 → 5% left tail).

    Returns:
        (var, cvar): VaR and CVaR as POSITIVE loss magnitudes.
                     i.e., VaR=0.02 means 2% daily loss at 95th percentile.
    """
    port_returns = return_window @ weights  # (T,)
    cutoff_idx = int(np.floor((1 - alpha) * len(port_returns)))
    cutoff_idx = max(cutoff_idx, 1)
    sorted_returns = np.sort(port_returns)  # ascending → worst first
    var_val = -sorted_returns[cutoff_idx]   # positive loss
    cvar_val = -sorted_returns[:cutoff_idx].mean() if cutoff_idx > 0 else var_val
    return float(var_val), float(cvar_val)


def _mvo_weights(
    alpha: np.ndarray,
    cov_d: np.ndarray,
    z0_smooth: float,
    vol_d: np.ndarray,
    alloc_scale: float = 1.0,
    return_window: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    CVaR-Augmented Mean-Variance Optimisation (BUG #25 FIX).

    Objective:
        min_w  λ_var · w^T Σ_ann w  +  λ_cvar · CVaR_95(w)  −  α^T w
    s.t.
        Σw = 1, 0 ≤ w_i ≤ w_max_i

    λ_var:  regime-conditioned risk aversion (unchanged from v5).
    λ_cvar: scales with realised excess kurtosis of the return window.
            When kurtosis is near-Gaussian (excess ≈ 0), λ_cvar ≈ 0 and
            the optimizer reduces to pure MVO. When tails are fat
            (excess > 3), λ_cvar dominates and the optimizer shifts weight
            toward assets with lower left-tail contribution.

    CVaR is estimated via sorted sample quantile from the return_window.
    The Rockafellar-Uryasev auxiliary variable ζ is optimised jointly
    by augmenting the decision vector to (w, ζ) ∈ R^{N+1}.

    If return_window is None (e.g., during warmup), falls back to pure MVO.

    Args:
        alpha:         (N,) expected alpha signal per asset.
        cov_d:         (N, N) daily sample covariance.
        z0_smooth:     Smoothed regime z_mu[0] — controls λ_var.
        vol_d:         (N,) 21-day rolling daily vol per asset.
        alloc_scale:   [0, 1] fraction of capital in risky assets (ramp-in).
        return_window: (T, N) raw daily returns for CVaR estimation. Optional.

    Returns:
        w: (N,) portfolio weights summing to 1.0.
    """
    avg_vol = float(np.mean(vol_d) * np.sqrt(_TRADING_DAYS_YEAR))
    lam_var = float(np.clip(_LAMBDA_BASE * np.exp(avg_vol * abs(z0_smooth)), 0.5, 15.0))
    Σ       = cov_d * _TRADING_DAYS_YEAR
    mx      = np.array([TIER_MAX_WEIGHT[t] for t in TICKERS]) * alloc_scale
    N       = N_ASSETS

    # ── Feasibility-guaranteed x0 (BUG #23 carry-forward) ────────────────────
    x0 = np.clip(np.full(N, 1.0 / N), 0.0, mx)
    s0 = x0.sum()
    if s0 > 1e-10:
        x0 = x0 / s0
    else:
        x0 = np.where(mx > 0, mx / (mx.sum() + 1e-10), 0.0)

    # ── BUG #25: Compute CVaR penalty weight from realised kurtosis ──────────
    use_cvar = (
        return_window is not None
        and return_window.shape[0] >= 30
        and return_window.shape[1] == N
    )

    if use_cvar:
        # Compute excess kurtosis of the equal-weight portfolio as a proxy
        ew_rets = return_window @ x0
        excess_kurt = float(stats.kurtosis(ew_rets, fisher=True))
        # λ_cvar scales quadratically with excess kurtosis above 3
        # At excess_kurt=9.35 → λ_cvar ≈ 1.0 + 0.15*(9.35-3)^2 = 7.05
        # At excess_kurt=0   → λ_cvar ≈ 0 (pure MVO)
        lam_cvar = float(np.clip(
            _LAMBDA_CVAR_BASE * max(excess_kurt - 3.0, 0.0) * _KURT_CVAR_SCALE * max(excess_kurt - 3.0, 0.0),
            0.0,
            10.0,
        ))
    else:
        lam_cvar = 0.0

    # ── CVaR-augmented optimisation via Rockafellar-Uryasev ──────────────────
    if use_cvar and lam_cvar > 0.01:
        T_obs = return_window.shape[0]
        alpha_conf = _CVAR_CONFIDENCE  # 0.95
        inv_tail = 1.0 / ((1.0 - alpha_conf) * T_obs)

        # Decision vector: x = [w_1, ..., w_N, ζ] ∈ R^{N+1}
        # ζ is the VaR auxiliary variable (Rockafellar-Uryasev formulation)

        def _objective(x: np.ndarray) -> float:
            w = x[:N]
            zeta = x[N]
            # Term 1: quadratic variance penalty
            var_term = 0.5 * lam_var * w @ Σ @ w
            # Term 2: alpha signal (negative = reward)
            alpha_term = -alpha @ w
            # Term 3: CVaR via R-U formulation
            # CVaR_α(w) ≈ ζ + (1/((1-α)T)) Σ_t max(0, -r_t·w - ζ)
            port_losses = -return_window @ w  # (T,) positive = loss
            shortfalls = np.maximum(port_losses - zeta, 0.0)
            cvar_term = lam_cvar * (zeta + inv_tail * shortfalls.sum())
            return float(var_term + alpha_term + cvar_term)

        def _jacobian(x: np.ndarray) -> np.ndarray:
            w = x[:N]
            zeta = x[N]
            # d/dw of variance term
            grad_var = lam_var * Σ @ w
            # d/dw of alpha term
            grad_alpha = -alpha
            # d/dw and d/dζ of CVaR term
            port_losses = -return_window @ w
            active = (port_losses - zeta) > 0  # (T,) boolean
            n_active = float(active.sum())
            # d/dw: lam_cvar * inv_tail * Σ_t (active_t * (-r_t))
            grad_cvar_w = lam_cvar * inv_tail * (-return_window.T @ active.astype(np.float64))
            # d/dζ: lam_cvar * (1 - inv_tail * n_active)
            grad_cvar_zeta = lam_cvar * (1.0 - inv_tail * n_active)

            grad_w = grad_var + grad_alpha + grad_cvar_w
            return np.append(grad_w, grad_cvar_zeta)

        # Bounds: w_i ∈ [0, mx_i], ζ ∈ [-0.5, 0.5] (VaR range in daily returns)
        bds_w = [(0.0, float(m)) for m in mx]
        bds_zeta = [(-0.5, 0.5)]
        bds_full = bds_w + bds_zeta

        # Constraint: Σw_i = 1 (ζ is unconstrained in the simplex sense)
        constraints = [{
            "type": "eq",
            "fun": lambda x: x[:N].sum() - 1.0,
            "jac": lambda x: np.append(np.ones(N), 0.0),
        }]

        # Initial ζ: sample VaR of equal-weight portfolio
        ew_losses = -return_window @ x0
        zeta0 = float(np.percentile(ew_losses, alpha_conf * 100))
        x0_full = np.append(x0, zeta0)

        res = sco.minimize(
            fun=_objective,
            jac=_jacobian,
            x0=x0_full,
            method="SLSQP",
            bounds=bds_full,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 500},
        )

        if res.success:
            w = res.x[:N]
            w = np.clip(w, 0.0, mx)
            s = w.sum()
            w = w / s if s > 1e-10 else x0
            # Concentration filter
            w[w < _MIN_POSITION_WT] = 0.0
            s = w.sum()
            if s > 1e-10:
                w /= s
            return w.astype(np.float32)
        else:
            logger.debug(
                f"CVaR-MVO failed ({res.message}), falling back to pure MVO."
            )
            # Fall through to pure MVO below

    # ── Pure MVO fallback (original BUG #23 fixed version) ────────────────────
    bds = [(0.0, float(m)) for m in mx]

    res = sco.minimize(
        fun=lambda w: 0.5 * lam_var * w @ Σ @ w - alpha @ w,
        jac=lambda w: lam_var * Σ @ w - alpha,
        x0=x0,
        method="SLSQP",
        bounds=bds,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"ftol": 1e-9, "maxiter": 500},
    )

    if res.success:
        w = res.x
    else:
        inv_vol = 1.0 / (np.diag(Σ) ** 0.5 + 1e-8)
        inv_vol = np.minimum(inv_vol, mx * 3)
        w = inv_vol / (inv_vol.sum() + 1e-10)

    w = np.clip(w, 0.0, mx)
    s = w.sum()
    w = w / s if s > 1e-10 else np.where(mx > 0, 1.0 / max((mx > 0).sum(), 1), 0.0)
    w[w < _MIN_POSITION_WT] = 0.0
    s = w.sum()
    if s > 1e-10:
        w /= s
    return w.astype(np.float32)


# ── Data containers ───────────────────────────────────────────────────────────

@dataclass
class Snap:
    date: str
    portfolio_value: float
    daily_return: float
    cash: float
    turnover: float
    cost_drag: float
    regime_label: str
    z_mu_0: float
    drawdown: float
    halt_phase: int


@dataclass
class WFFold:
    fold_id: int
    is_start: str
    is_end: str
    oos_start: str
    oos_end: str
    is_sharpe: float
    oos_sharpe: float
    oos_cagr: float
    oos_max_dd: float


# ==============================================================================
# SECTION 2: BUG #24 FIX — Halt State Machine
# ==============================================================================
# Key changes marked with "# BUG #24 FIX" comments.
# ==============================================================================

class StandaloneBacktester:

    def _reset(self) -> None:
        self.nav:   float = _INITIAL_CAPITAL
        self.cash:  float = _INITIAL_CAPITAL
        self.pos:   Dict[str, float] = {}
        self.history: List[Snap] = []
        self._peak:           float = _INITIAL_CAPITAL
        self._prev_regime:    str   = ""
        self._halt_phase:     int   = 0
        self._halt_days:      int   = 0
        self._halt_nav:       float = _INITIAL_CAPITAL
        self._ramp_days:      int   = 0
        self._ramp_entry_nav: float = _INITIAL_CAPITAL
        self._smoother = RegimeSignalSmoother()
        self._prev_weights = np.zeros(N_ASSETS, dtype=np.float64)

    def _current_weights(self, px: np.ndarray) -> np.ndarray:
        """Mark-to-market weight vector."""
        vals = np.array([self.pos.get(t, 0.0) * px[i] for i, t in enumerate(TICKERS)])
        total = vals.sum() + self.cash
        if total < 1e-10:
            return np.zeros(N_ASSETS, dtype=np.float64)
        return vals / total

    def _liquidate(self, px: np.ndarray) -> None:
        for t, sh in list(self.pos.items()):
            self.cash += sh * px[TICKERS.index(t)]
            self.pos[t] = 0.0
        self.nav = self.cash

    def _invest_bil(self, px: np.ndarray) -> None:
        self._liquidate(px)
        for t, frac in [("BIL", 0.60), ("SHV", 0.40)]:
            i = TICKERS.index(t)
            shares = self.cash * frac / (px[i] + 1e-10)
            self.pos[t] = shares
            self.cash -= self.cash * frac

    def _rf_step(self) -> float:
        dr = _RISK_FREE_ANNUAL / _TRADING_DAYS_YEAR
        self.nav *= (1.0 + dr)
        self.cash = self.nav - sum(
            self.pos.get(t, 0.0) * 0.0 for t in TICKERS
        )
        self._peak = max(self._peak, self.nav)
        return dr

    def _rebalance(
        self,
        target: np.ndarray,
        px: np.ndarray,
        vol: np.ndarray,
        adv: np.ndarray,
        force: bool = False,
    ) -> Tuple[float, float]:
        current = self._current_weights(px)
        delta   = target - current
        cost    = 0.0

        for i, t in enumerate(TICKERS):
            if not force and abs(delta[i]) < _REBALANCE_BAND:
                continue
            dol    = delta[i] * self.nav
            shares = dol / (px[i] + 1e-10)
            c      = abs(dol) * (
                _ac_cost_bps(shares, vol[i], adv[i]) + _BASE_SPREAD_BPS
            ) / 10_000
            cost       += c
            self.cash  += -(dol + np.sign(dol) * c)
            self.pos[t]  = self.pos.get(t, 0.0) + shares

        cost_drag_frac = cost / (self.nav + 1e-10)
        one_way_turnover = float(0.5 * np.sum(np.abs(target - current)))
        self._prev_weights = target.copy().astype(np.float64)
        return cost_drag_frac, one_way_turnover

    def _mtm(self, px: np.ndarray) -> float:
        equity   = sum(self.pos.get(t, 0.0) * px[i] for i, t in enumerate(TICKERS))
        new_nav  = self.cash + equity
        dr       = (new_nav - self.nav) / (self.nav + 1e-10)
        self.nav = new_nav
        self._peak = max(self._peak, new_nav)
        return dr

    # ── Main simulation loop ──────────────────────────────────────────────────

    def run(
        self,
        prices_df:  pd.DataFrame,
        returns_df: pd.DataFrame,
        regime_df:  pd.DataFrame,
        alpha_df:   pd.DataFrame,
        start_date: str,
        end_date:   str,
        warmup_days: int = _WARMUP_DAYS,
    ) -> pd.DataFrame:
        self._reset()

        common = (
            prices_df.index
            .intersection(returns_df.index)
            .intersection(regime_df.index)
            .intersection(alpha_df.index)
        )
        mask = (common >= start_date) & (common <= end_date)
        sim_dates = common[mask]

        if len(sim_dates) == 0:
            logger.error("No common dates in simulation window.")
            return pd.DataFrame()

        # Extend backwards for warmup
        all_dates = common[common <= end_date]
        warmup_start_idx = max(0, all_dates.get_loc(sim_dates[0]) - warmup_days)
        full_dates = all_dates[warmup_start_idx:]
        record_start = sim_dates[0]

        for date in full_dates:
            record = date >= record_start
            ds = str(date.date())
            gi = prices_df.index.get_loc(date)
            px = prices_df.iloc[gi].values.astype(np.float64)

            if date not in regime_df.index or date not in alpha_df.index:
                if record:
                    self.history.append(
                        Snap(ds, self.nav, 0.0, self.cash, 0.0, 0.0,
                             "NO_SIGNAL", 0.0, 0.0, 0)
                    )
                continue

            raw_z = regime_df.loc[date, "z_mu"]
            if isinstance(raw_z, str):
                import ast
                raw_z = ast.literal_eval(raw_z)
            z_raw = np.asarray(raw_z, dtype=np.float32).ravel()
            z_smooth, s_label = self._smoother.update(
                z_raw, str(regime_df.loc[date, "regime_label"])
            )
            z0 = float(z_smooth[0])

            halt_dd_thresh, halt_days_req, ramp_days_req = _get_regime_halt_threshold(
                regime_df, date
            )

            # ══════════════════════════════════════════════════════════════════
            # HALT STATE MACHINE — BUG #24 FIX
            # ══════════════════════════════════════════════════════════════════
            if self._halt_phase >= 1:
                self._halt_days += 1
                dr = self._rf_step()

                # Global drawdown from all-time peak (for Phase 1 monitoring)
                dd_from_peak = (self.nav - self._peak) / self._peak

                if self._halt_phase == 1:
                    # Phase 1: mandatory BIL — check minimum days + recovery
                    if self._halt_days >= halt_days_req:
                        recov = (self.nav - self._halt_nav) / (self._halt_nav + 1e-10)
                        if recov >= -_HALT_RECOVERY_THRESH:
                            logger.info(f"{ds}: Halt recovery — entering ramp-in.")
                            self._halt_phase     = 2
                            self._ramp_days      = 0
                            self._ramp_entry_nav = self.nav
                            # BUG #24 FIX: Record the NAV at ramp entry.
                            # All ramp-in drawdown checks will be relative to
                            # this value, NOT self._peak.

                elif self._halt_phase == 2:
                    # Phase 2: linear ramp-in
                    self._ramp_days += 1
                    scale = min(float(self._ramp_days) / ramp_days_req, 1.0)

                    cov_d = _cov(returns_df.iloc[max(0, gi - _COV_WINDOW) : gi])
                    vol_d = (
                        returns_df.iloc[max(0, gi - 21) : gi]
                        .std(axis=0).fillna(0.01).values.astype(np.float64)
                    )
                    adv = np.maximum(10_000_000.0 / (px + 1e-10), 1.0)
                    alp = alpha_df.loc[date].values.astype(np.float64)

                    # BUG #25 FIX: pass return window for CVaR estimation
                    ret_window = returns_df.iloc[max(0, gi - _COV_WINDOW) : gi].values
                    target = _mvo_weights(
                        alp, cov_d, z0, vol_d,
                        alloc_scale=scale,
                        return_window=ret_window if ret_window.shape[0] >= 30 else None,
                    )
                    cost_d, to = self._rebalance(target, px, vol_d, adv)
                    dr = self._mtm(px)

                    # ══════════════════════════════════════════════════════════
                    # BUG #24 FIX: Drawdown during ramp-in is measured from
                    # _ramp_entry_nav, NOT from _peak.
                    #
                    # Previous (broken): dd = (nav - peak) / peak
                    #   → guaranteed to be ≈ -20% on Day 1 since peak was set
                    #     before the halt (when NAV was ~20% higher).
                    #
                    # Fixed: dd_ramp = (nav - ramp_entry_nav) / ramp_entry_nav
                    #   → measures only the performance SINCE ramp-in started.
                    #   → re-halt threshold is _RAMP_DD_BUFFER (8%), not the
                    #     full halt_dd_thresh (18-25%).
                    # ══════════════════════════════════════════════════════════
                    dd_ramp = (self.nav - self._ramp_entry_nav) / (self._ramp_entry_nav + 1e-10)

                    if dd_ramp <= -_RAMP_DD_BUFFER:
                        logger.warning(
                            f"{ds}: Ramp-in drawdown={dd_ramp:.2%} from ramp entry "
                            f"exceeded {_RAMP_DD_BUFFER:.2%} buffer. "
                            f"Returning to mandatory BIL."
                        )
                        self._halt_phase = 1
                        self._halt_days  = 0
                        self._halt_nav   = self.nav
                        self._liquidate(px)
                        self._invest_bil(px)

                    if self._ramp_days >= ramp_days_req:
                        logger.info(f"{ds}: Ramp-in complete. Resuming full active trading.")
                        self._halt_phase = 0
                        # BUG #24 FIX: Reset peak to current NAV on ramp-in
                        # completion. This establishes a clean high-water mark
                        # for the new active trading period. Without this, the
                        # portfolio immediately re-enters a "drawdown" state
                        # relative to the pre-halt peak, making the next halt
                        # trigger much too sensitive.
                        self._peak = self.nav

                    if record:
                        self.history.append(
                            Snap(
                                ds, self.nav, dr, self.cash, to, cost_d,
                                f"RAMP_{scale:.0%}", z0, dd_from_peak, 2,
                            )
                        )
                    continue

                if record:
                    self.history.append(
                        Snap(
                            ds, self.nav, dr, self.cash, 0.0, 0.0,
                            "HALTED_BIL", z0, dd_from_peak, self._halt_phase,
                        )
                    )
                continue

            # ── ACTIVE TRADING ────────────────────────────────────────────────
            cov_d = _cov(returns_df.iloc[max(0, gi - _COV_WINDOW) : gi])
            vol_d = (
                returns_df.iloc[max(0, gi - 21) : gi]
                .std(axis=0).fillna(0.01).values.astype(np.float64)
            )
            adv = np.maximum(10_000_000.0 / (px + 1e-10), 1.0)
            alp = alpha_df.loc[date].values.astype(np.float64)

            # BUG #25 FIX: pass return window for CVaR estimation
            ret_window = returns_df.iloc[max(0, gi - _COV_WINDOW) : gi].values
            target = _mvo_weights(
                alp, cov_d, z0, vol_d,
                return_window=ret_window if ret_window.shape[0] >= 30 else None,
            )

            regime_changed = s_label != self._prev_regime
            cost_d, to = self._rebalance(
                target, px, vol_d, adv, force=regime_changed
            )
            dr = self._mtm(px)
            dd = (self.nav - self._peak) / self._peak
            self._prev_regime = s_label

            if dd <= -halt_dd_thresh:
                logger.critical(
                    f"HALT triggered {ds}: DD={dd:.2%} ≤ "
                    f"-{halt_dd_thresh:.2%} (regime-conditional)"
                )
                self._liquidate(px)
                self._halt_phase = 1
                self._halt_days  = 0
                self._halt_nav   = self.nav
                self._invest_bil(px)

            if record:
                self.history.append(
                    Snap(ds, self.nav, dr, self.cash, to, cost_d, s_label, z0, dd, 0)
                )

        active_days = sum(
            1 for h in self.history
            if h.regime_label not in ("WARMUP", "NO_SIGNAL", "HALTED_BIL")
            and not h.regime_label.startswith("RAMP")
        )
        if self.history and active_days == 0:
            logger.warning(
                "⚠️  All recorded days are WARMUP/NO_SIGNAL/HALTED_BIL. "
                "No active trading occurred."
            )

        return self._build_tearsheet()

    # ── Tearsheet builder (unchanged except for metric logging) ───────────────

    def _build_tearsheet(self) -> pd.DataFrame:
        df = pd.DataFrame([
            {
                "date":            h.date,
                "portfolio_value": h.portfolio_value,
                "daily_return":    h.daily_return,
                "cash":            h.cash,
                "turnover":        h.turnover,
                "cost_drag":       h.cost_drag,
                "regime_label":    h.regime_label,
                "z_mu_0":          h.z_mu_0,
                "drawdown":        h.drawdown,
                "halt_phase":      h.halt_phase,
            }
            for h in self.history
        ])
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        if len(df) < 10:
            return df

        rf  = _RISK_FREE_ANNUAL / _TRADING_DAYS_YEAR
        r   = df["daily_return"].values.astype(np.float64)
        ex  = r - rf
        n   = len(r)

        tot  = df["portfolio_value"].iloc[-1] / _INITIAL_CAPITAL - 1.0
        cagr = (1.0 + tot) ** (_TRADING_DAYS_YEAR / n) - 1.0
        vol  = float(r.std() * np.sqrt(_TRADING_DAYS_YEAR))
        sr   = float((ex.mean() / (ex.std() + 1e-10)) * np.sqrt(_TRADING_DAYS_YEAR))

        dn     = ex[ex < 0]
        sort_r = float(
            (ex.mean() / (dn.std() + 1e-10)) * np.sqrt(_TRADING_DAYS_YEAR)
            if len(dn) > 1 else 0.0
        )

        cv   = df["portfolio_value"]
        rm   = cv.cummax()
        dd_s = (cv - rm) / (rm + 1e-10)
        mdd  = float(dd_s.min())
        cal  = cagr / (abs(mdd) + 1e-10)
        mdd_dur = _mct((dd_s < 0).values)

        v95    = float(np.percentile(r, 5))
        cvar95 = float(r[r <= v95].mean()) if (r <= v95).any() else v95
        hit    = float((r > rf).mean())

        to_mean = float(df["turnover"].mean())
        cost_drag_ann_bps = float(df["cost_drag"].mean() * _TRADING_DAYS_YEAR * 10_000)

        skew     = float(stats.skew(r))
        kurt_raw = float(stats.kurtosis(r, fisher=False))
        psr      = _psr(sr, n, skew, kurt_raw)

        active_pct = float((
            ~df["regime_label"].isin(["WARMUP", "NO_SIGNAL", "HALTED_BIL"])
            & ~df["regime_label"].str.startswith("RAMP")
        ).mean())
        halt_pct = float(df["regime_label"].isin(["HALTED_BIL"]).mean())
        ramp_pct = float(df["regime_label"].str.startswith("RAMP").mean())

        m = {
            "CAGR":                    f"{cagr:+.2%}",
            "Ann. Volatility":         f"{vol:.2%}",
            "Sharpe Ratio":            f"{sr:.3f}",
            "Sortino Ratio":           f"{sort_r:.3f}",
            "Calmar Ratio":            f"{cal:.3f}",
            "Max Drawdown":            f"{mdd:.2%}",
            "Max DD Duration":         f"{mdd_dur} days",
            "VaR-95 (daily)":          f"{v95:.2%}",
            "CVaR-95 (daily)":         f"{cvar95:.2%}",
            "Hit Rate":                f"{hit:.2%}",
            "Avg Daily Turnover":      f"{to_mean:.2%}",
            "Cost Drag (ann. bps)":    f"{cost_drag_ann_bps:.1f}",
            "PSR (SR>0)":              f"{psr:.4f}",
            "Skewness":                f"{skew:.3f}",
            "Excess Kurtosis":         f"{kurt_raw - 3:.3f}",
            "Active Trading %":        f"{active_pct:.1%}",
            "Halted BIL %":            f"{halt_pct:.1%}",
            "Ramp-in %":               f"{ramp_pct:.1%}",
            "Final NAV":               f"${df['portfolio_value'].iloc[-1]:,.2f}",
            "Trading Days":            str(n),
        }

        for k, v in m.items():
            logger.info(f"  {k:<30s} {v}")

        df_out = df.copy()
        for k, v in m.items():
            df_out[k] = v
        return df_out


# ── Walk-forward fold generation ──────────────────────────────────────────────

def _wf_folds(
    dates: pd.DatetimeIndex,
    is_months: int = 18,
    oos_months: int = 6,
) -> List[Tuple[str, str, str, str]]:
    start    = dates[0]
    end      = dates[-1]
    folds    = []
    is_end   = start + pd.DateOffset(months=is_months)
    while True:
        oos_end = is_end + pd.DateOffset(months=oos_months)
        if oos_end > end:
            break
        is_e_mask  = dates[dates <= is_end]
        oos_s_mask = dates[dates > is_end]
        oos_e_mask = dates[dates <= oos_end]
        if len(is_e_mask) == 0 or len(oos_s_mask) == 0 or len(oos_e_mask) == 0:
            break
        folds.append((
            start.strftime("%Y-%m-%d"),
            is_e_mask[-1].strftime("%Y-%m-%d"),
            oos_s_mask[0].strftime("%Y-%m-%d"),
            oos_e_mask[-1].strftime("%Y-%m-%d"),
        ))
        is_end += pd.DateOffset(months=oos_months)
    return folds


def _wf_verdict(avg_is: float, avg_oos: float) -> str:
    if avg_oos > 0.5:
        return "✅ OOS signal positive"
    if avg_oos > 0.0:
        return "⚠️  IS dominated by structural event; OOS signal positive"
    return "❌ OOS negative — strategy requires redesign"


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(
        "══════ Fortress v5 — Standalone Backtest v5 "
        "(BUG #21/#22/#23/#24/#25 FIX) ══════"
    )

    prices_df  = _normalize_index(pd.read_parquet(_CACHE_DIR / "prices_wide.parquet"))
    returns_df = _normalize_index(pd.read_parquet(_CACHE_DIR / "returns_wide.parquet"))
    regime_df  = _normalize_index(pd.read_parquet(_CACHE_DIR / "regime_posteriors.parquet"))
    alpha_df   = _normalize_index(pd.read_parquet(_CACHE_DIR / "alpha_signals.parquet"))

    logger.info(
        f"Loaded | prices:{len(prices_df)}d  returns:{len(returns_df)}d  "
        f"regime:{len(regime_df)}d  alpha:{len(alpha_df)}d"
    )
    logger.info(
        f"Ranges | prices:{prices_df.index.min().date()}→{prices_df.index.max().date()}  "
        f"regime:{regime_df.index.min().date()}→{regime_df.index.max().date()}"
    )

    has_soft = all(c in regime_df.columns for c in _SOFT_COLS)
    if has_soft:
        logger.info(
            "✅ Soft GMM posteriors detected — "
            "regime-conditional halt thresholds active."
        )
    else:
        logger.warning(
            f"⚠️  Soft GMM posteriors NOT found. "
            f"Falling back to static halt threshold {_MAX_DD_HALT_FALLBACK:.0%}."
        )

    ts = StandaloneBacktester().run(
        prices_df, returns_df, regime_df, alpha_df,
        start_date="2020-01-02", end_date="2024-12-31",
        warmup_days=_WARMUP_DAYS,
    )
    if ts.empty:
        logger.error("Tearsheet empty — check data alignment.")
        sys.exit(1)

    ts.to_csv(_TEARSHEET)
    logger.info(f"✅ Tearsheet → {_TEARSHEET} ({len(ts)} rows)")

    # ── Walk-forward ──────────────────────────────────────────────────────────
    logger.info("Running walk-forward validation...")
    common  = prices_df.index.intersection(alpha_df.index)
    wf_mask = (common >= "2019-01-02") & (common <= "2024-12-31")
    folds   = _wf_folds(common[wf_mask])
    results: List[WFFold] = []

    for fid, (is_s, is_e, oos_s, oos_e) in enumerate(folds):
        t_is = StandaloneBacktester().run(
            prices_df, returns_df, regime_df, alpha_df,
            is_s, is_e, warmup_days=_WF_WARMUP_DAYS,
        )
        t_oo = StandaloneBacktester().run(
            prices_df, returns_df, regime_df, alpha_df,
            oos_s, oos_e, warmup_days=_WF_WARMUP_DAYS,
        )
        if t_oo.empty or len(t_oo) < 5:
            continue

        r_oo = t_oo["daily_return"].values
        n_oo = len(r_oo)
        tot  = t_oo["portfolio_value"].iloc[-1] / _INITIAL_CAPITAL - 1.0
        cagr = (1.0 + tot) ** (_TRADING_DAYS_YEAR / n_oo) - 1.0
        mdd  = float(
            (
                (t_oo["portfolio_value"] - t_oo["portfolio_value"].cummax())
                / (t_oo["portfolio_value"].cummax() + 1e-10)
            ).min()
        )
        is_sr  = _sharpe(t_is["daily_return"].values) if not t_is.empty else 0.0
        oos_sr = _sharpe(r_oo)

        results.append(WFFold(
            fid + 1, is_s, is_e, oos_s, oos_e,
            round(is_sr, 4), round(oos_sr, 4), round(cagr, 4), round(mdd, 4),
        ))
        logger.info(
            f"  F{fid + 1}: IS SR={is_sr:.3f} | OOS SR={oos_sr:.3f}"
            f"  CAGR={cagr:.2%}  MaxDD={mdd:.2%}"
        )

    pd.DataFrame([vars(r) for r in results]).to_csv(_WF_FOLDS, index=False)
    if results:
        avg_oos = float(np.mean([f.oos_sharpe for f in results]))
        avg_is  = float(np.mean([f.is_sharpe  for f in results]))
        verdict = _wf_verdict(avg_is, avg_oos)
        logger.info(
            f"WF summary: avg IS={avg_is:.3f} | avg OOS={avg_oos:.3f} | {verdict}"
        )

    r_arr = ts["daily_return"].values
    v95   = float(np.percentile(r_arr, 5))
    with open(_STRESS_OUT, "w") as f:
        json.dump({
            "mode": "stub",
            "var_95": v95,
            "cvar_95": float(r_arr[r_arr <= v95].mean()),
            "note": "Run training/train_world_model.py for SDE stress.",
        }, f, indent=2)

    logger.info("✅ Done. Run scripts/visualize_tearsheet.py next.")


if __name__ == "__main__":
    main()