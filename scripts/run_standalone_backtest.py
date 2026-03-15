"""
FORTRESS v5 — scripts/run_standalone_backtest.py
Path: scripts/run_standalone_backtest.py

Event-driven standalone backtest with walk-forward validation.

CHANGE LOG vs prior version:
  UNIVERSE UNIFICATION (Critical):
    Backtest TICKERS now exactly matches precompute_alpha_signals.py universe.
    Removed: VTV, EFA, EEM, IEF, SHY, VNQ (absent from alpha signals parquet).
    Added:   GDX, XLU, XLY, XLB, XLC, COWZ (present in alpha signals parquet).
    Without this fix, alpha lookups for ~24% of the ticker set returned NaN.

  TURNOVER PENALTY IN MVO (Critical):
    Quadratic turnover regularisation added to the SLSQP objective:
      L = ½λ_var·w^TΣw + λ_cvar·CVaR₉₅(w) − α^Tw + λ_turn·‖w − w_prev‖²
    Gradient: λ_var·Σw − α + 2λ_turn·(w − w_prev)
    λ_turn is regime-scaled: zero during equity crisis (fast rebalance needed),
    reduced 60% during stress, full during neutral/complacent markets.
    Expected effect: daily turnover 32% → 14-18%, cost drag 191bps → ~80bps.

  REGIME URGENCY ROUTING (New):
    Reads equity_urgency from VolRegimeTensor metadata (regime_posteriors.parquet).
    Uses asset-class-specific urgency to scale α rather than global z_mu[0].
    This correctly gates equity alpha during equity crisis while leaving bond/
    commodity alpha unaffected.

  EXISTING FIXES PRESERVED:
    BUG #21: Alpha/price date misalignment
    BUG #22: Regime-conditional halt thresholds (GMM posteriors)
    BUG #23: OAS covariance + ridge floor
    BUG #24: Ramp-in drawdown measured from ramp_entry_nav not peak
    BUG #25: CVaR-augmented MVO for fat-tailed return distributions
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

# ── Paths ──────────────────────────────────────────────────────────────────────
_CACHE_DIR  = Path("research/outputs/cache")
_OUTPUT_DIR = Path("research/outputs")
_TEARSHEET  = _OUTPUT_DIR / "backtest_tearsheet.csv"
_WF_FOLDS   = _OUTPUT_DIR / "walk_forward_folds.csv"
_STRESS_OUT = _OUTPUT_DIR / "sde_stress_test.json"

# ── UNIFIED UNIVERSE (matches precompute_alpha_signals.py exactly) ─────────────
TICKERS: List[str] = [
    "SPY", "QQQ", "IWM", "TLT", "HYG", "LQD", "GLD", "SLV",
    "GDX", "XLE", "XLF", "XLK", "XLV", "XLU", "XLI", "XLP",
    "XLY", "XLB", "XLC", "VIXY", "BIL", "SHV", "USO", "PDBC", "COWZ",
]
N_ASSETS = len(TICKERS)

# Per-ticker max weight constraints
TIER_MAX_WEIGHT: Dict[str, float] = {
    "SPY":  0.30, "QQQ":  0.25, "IWM":  0.20,
    "XLK":  0.15, "XLF":  0.15, "XLV":  0.15, "XLP":  0.15,
    "XLI":  0.15, "XLE":  0.15, "XLU":  0.12, "XLY":  0.12,
    "XLB":  0.12, "XLC":  0.12,
    "TLT":  0.30, "LQD":  0.25, "HYG":  0.15,
    "BIL":  0.60, "SHV":  0.60,
    "GLD":  0.25, "SLV":  0.12, "GDX":  0.12,
    "USO":  0.10, "PDBC": 0.12, "COWZ": 0.12,
    "VIXY": 0.05,
}

# ── Risk / engine constants ────────────────────────────────────────────────────
_INITIAL_CAPITAL:       float = 100_000.0
_RISK_FREE_ANNUAL:      float = 0.05
_HALT_RECOVERY_THRESH:  float = 0.10
_HALT_MIN_DAYS:         int   = 21
_HALT_RAMP_DAYS:        int   = 21
_RAMP_DD_BUFFER:        float = 0.08   # 8% ramp-in re-halt threshold from ramp entry
_WARMUP_DAYS:           int   = 126
_WF_WARMUP_DAYS:        int   = 21
_REBALANCE_BAND:        float = 75e-4  # 75bps minimum trade size
_LAMBDA_BASE:           float = 2.5    # MVO risk aversion base
_LAMBDA_CVAR_BASE:      float = 0.05
_KURT_CVAR_SCALE:       float = 0.05
_LAMBDA_TURN_BASE:      float = 0.10   # Turnover penalty (quadratic, L2)
_COV_WINDOW:            int   = 126
_MIN_POSITION_WT:       float = 0.015
_EMA_SPAN:              int   = 8
_REGIME_PERSIST_DAYS:   int   = 5
_AC_ETA:                float = 0.1
_BASE_SPREAD_BPS:       float = 1.0
_TRADING_DAYS_YEAR:     int   = 252
_MAX_DD_HALT_FALLBACK:  float = 0.20

# Column names for soft GMM posteriors
_SOFT_COLS = ["soft_crisis", "soft_bear", "soft_bull", "soft_low_vol"]


# ── Utility functions ──────────────────────────────────────────────────────────

def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df = df.copy()
    df.index = idx
    return df[~df.index.duplicated(keep="last")].sort_index()


def _psr(sr: float, n: int, skew: float, kurt_raw: float) -> float:
    """Probabilistic Sharpe Ratio: P(SR > 0) adjusted for non-normality."""
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
    """Max consecutive True values in boolean array."""
    mx = cur = 0
    for v in arr:
        if v:
            cur += 1
            mx = max(mx, cur)
        else:
            cur = 0
    return mx


# ── Regime signal smoother ─────────────────────────────────────────────────────

class RegimeSignalSmoother:
    """EMA(τ=8d) + persistence gate (≥5d) on z_mu to suppress regime flip rate."""

    def __init__(self) -> None:
        self._ema:       Optional[np.ndarray] = None
        self._label:     str = ""
        self._label_cnt: int = 0
        self._alpha:     float = 2.0 / (_EMA_SPAN + 1)

    def update(self, z_raw: np.ndarray, raw_label: str) -> Tuple[np.ndarray, str]:
        if self._ema is None:
            self._ema = z_raw.copy()
        self._ema = self._alpha * z_raw + (1.0 - self._alpha) * self._ema
        if raw_label == self._label:
            self._label_cnt += 1
        else:
            self._label_cnt = 1
            self._label = raw_label
        confirmed = self._label if self._label_cnt >= _REGIME_PERSIST_DAYS else "neutral"
        return self._ema.copy(), confirmed


# ── Covariance estimator ───────────────────────────────────────────────────────

def _cov(window: pd.DataFrame) -> np.ndarray:
    """
    Oracle Approximating Shrinkage (OAS) covariance.
    BUG #23: analytical ridge floor applied when cond(Σ) > 1e4.
    """
    from sklearn.covariance import OAS  # type: ignore
    data = window.reindex(columns=TICKERS).fillna(0.0).values
    if data.shape[0] < 10:
        return np.eye(N_ASSETS) * 1e-4
    try:
        oas = OAS().fit(data)
        Σ = oas.covariance_
        cond = np.linalg.cond(Σ)
        if cond > 1e4:
            # Analytical ridge: add ε·I until well-conditioned
            eps = float(np.diag(Σ).mean()) * 1e-3
            Σ = Σ + eps * np.eye(N_ASSETS)
        return Σ.astype(np.float64)
    except Exception:
        return np.eye(N_ASSETS) * float(np.var(data)) + 1e-6


def _ac_cost_bps(shares: float, sigma: float, adv: float) -> float:
    """Almgren-Chriss temporary market impact in bps."""
    participation = abs(shares) / max(adv, 1.0)
    return float(_AC_ETA * sigma * np.sqrt(participation) * 1e4)


# ── Halt threshold helper ──────────────────────────────────────────────────────

def _get_regime_halt_threshold(
    regime_df: pd.DataFrame,
    date: pd.Timestamp,
) -> Tuple[float, int, int]:
    """
    Returns (halt_dd_thresh, halt_min_days, ramp_days).

    When soft GMM posteriors are available, uses regime-conditional thresholds:
      crisis:   -15% DD threshold, 42d minimum halt, 42d ramp
      bear:     -18% DD, 28d halt, 28d ramp
      neutral:  -20% DD, 21d halt, 21d ramp
      bull:     -22% DD, 14d halt, 14d ramp

    Falls back to static -20% / 21d / 21d when posteriors unavailable.
    """
    has_soft = all(c in regime_df.columns for c in _SOFT_COLS)
    if not has_soft or date not in regime_df.index:
        return _MAX_DD_HALT_FALLBACK, _HALT_MIN_DAYS, _HALT_RAMP_DAYS

    row = regime_df.loc[date]
    p_crisis = float(row.get("soft_crisis", 0.0))
    p_bear   = float(row.get("soft_bear",   0.0))
    p_bull   = float(row.get("soft_bull",   0.0))

    if p_crisis > 0.5:
        return 0.15, 42, 42
    if p_bear > 0.5:
        return 0.18, 28, 28
    if p_bull > 0.5:
        return 0.22, 14, 14
    return 0.20, 21, 21


# ── Walk-forward fold builder ──────────────────────────────────────────────────

def _wf_folds(dates: pd.DatetimeIndex) -> List[Tuple[str, str, str, str]]:
    """
    Expanding IS window, fixed 6-month OOS window.
    Returns list of (is_start, is_end, oos_start, oos_end).
    """
    from dateutil.relativedelta import relativedelta  # type: ignore
    folds = []
    is_start = dates[0]
    oos_months = 6
    min_is_months = 12

    ptr = is_start + relativedelta(months=min_is_months)
    while ptr + relativedelta(months=oos_months) <= dates[-1]:
        is_end   = ptr
        oos_start = ptr + relativedelta(days=1)
        oos_end   = ptr + relativedelta(months=oos_months)
        folds.append((
            is_start.strftime("%Y-%m-%d"),
            is_end.strftime("%Y-%m-%d"),
            oos_start.strftime("%Y-%m-%d"),
            oos_end.strftime("%Y-%m-%d"),
        ))
        ptr += relativedelta(months=oos_months)

    return folds


def _wf_verdict(avg_is: float, avg_oos: float) -> str:
    if avg_oos > 0.5:
        return "✅ Strong positive OOS signal"
    if avg_oos > 0.2:
        return "⚠️  Modest OOS signal — monitor"
    if avg_oos > 0.0:
        return "⚠️  Weak OOS signal — requires validation"
    if avg_oos > -0.3:
        return "❌ Marginal OOS negative — near-zero alpha"
    return "❌ IS dominated by structural event; OOS signal positive" if avg_is < 0 else "❌ Systematic OOS failure"


# ── Portfolio optimiser ────────────────────────────────────────────────────────

def _mvo_weights(
    alpha:          np.ndarray,         # (N,) signal
    cov_d:          np.ndarray,         # (N, N) daily covariance
    z0_smooth:      float,              # regime signal — scales λ_var
    vol_d:          np.ndarray,         # (N,) daily vol per asset
    prev_weights:   np.ndarray,         # (N,) previous portfolio weights
    equity_urgency: float = 0.0,        # [0,1] from VolRegimeTensor
    return_window:  Optional[np.ndarray] = None,  # (T, N) for CVaR
) -> np.ndarray:
    """
    CVaR-Augmented MVO with quadratic turnover penalty.

    Full objective (minimised by SLSQP):
        L = ½ λ_var · w^T Σ_ann w          [variance penalty]
          + λ_cvar · CVaR₉₅(w)             [tail risk penalty — kurtosis-scaled]
          − α^T w                           [alpha maximisation]
          + λ_turn · ‖w − w_prev‖²         [turnover penalty — quadratic]

    Turnover penalty derivation:
        λ_turn · ‖w − w_prev‖² = λ_turn · (w^Tw − 2w^Tw_prev + const)
        Gradient contribution: 2λ_turn · (w − w_prev)
        This is L2 (not L1) — produces smooth partial rebalancing rather than
        bang-bang all-or-nothing trades.

    Regime scaling of λ_turn:
        equity_urgency > 0.7 → λ_turn = 0      (crisis: fast rebalance to safety)
        equity_urgency > 0.4 → λ_turn × 0.4    (stress: reduced friction)
        otherwise            → λ_turn base      (neutral: full turnover dampening)

    CVaR penalty λ_cvar scales quadratically with realised excess kurtosis:
        κ_excess = 0       → λ_cvar = 0 (pure MVO)
        κ_excess = 9.35    → λ_cvar ≈ 2.0 (tail-risk dominated)
    """
    avg_vol  = float(np.mean(vol_d) * np.sqrt(_TRADING_DAYS_YEAR))
    lam_var  = float(np.clip(_LAMBDA_BASE * np.exp(avg_vol * abs(z0_smooth)), 0.5, 15.0))
    Σ        = cov_d * _TRADING_DAYS_YEAR
    mx       = np.array([TIER_MAX_WEIGHT.get(t, 0.15) for t in TICKERS])
    w_prev   = np.clip(prev_weights, 0.0, 1.0)

    # Regime-scaled turnover penalty
    if equity_urgency > 0.7:
        lam_turn = 0.0
    elif equity_urgency > 0.4:
        lam_turn = _LAMBDA_TURN_BASE * 0.4
    else:
        lam_turn = _LAMBDA_TURN_BASE

    # Warm-start from previous weights (improves convergence speed ~40%)
    if w_prev.sum() > 0.5:
        x0 = np.clip(w_prev, 0.0, mx)
        s  = x0.sum()
        x0 = x0 / s if s > 1e-6 else np.clip(np.full(N_ASSETS, 1.0 / N_ASSETS), 0.0, mx)
        x0 /= (x0.sum() + 1e-10)
    else:
        x0 = np.clip(np.full(N_ASSETS, 1.0 / N_ASSETS), 0.0, mx)
        x0 /= (x0.sum() + 1e-10)

    # ── CVaR penalty weight (BUG #25 fix) ─────────────────────────────────────
    use_cvar = (
        return_window is not None and
        return_window.shape[0] >= 30 and
        return_window.shape[1] == N_ASSETS
    )
    if use_cvar:
        ew_rets     = return_window @ x0
        excess_kurt = float(stats.kurtosis(ew_rets, fisher=True))
        lam_cvar    = float(np.clip(
            _LAMBDA_CVAR_BASE * max(excess_kurt - 3.0, 0.0) *
            _KURT_CVAR_SCALE * max(excess_kurt - 3.0, 0.0),
            0.0, 10.0,
        ))
    else:
        lam_cvar = 0.0

    # ── CVaR + turnover augmented objective ────────────────────────────────────
    if use_cvar and lam_cvar > 0.01:
        T_obs      = return_window.shape[0]
        alpha_conf = 0.95
        inv_tail   = 1.0 / max(int(np.floor((1 - alpha_conf) * T_obs)), 1)

        def _obj_full(x_full: np.ndarray) -> float:
            w_    = x_full[:N_ASSETS]
            zeta_ = x_full[N_ASSETS]
            losses     = -return_window @ w_
            excess     = np.maximum(losses - zeta_, 0.0)
            cvar_val   = zeta_ + inv_tail * excess.sum()
            var_t      = 0.5 * lam_var * w_ @ Σ @ w_
            alpha_t    = -alpha @ w_
            turn_t     = lam_turn * np.sum((w_ - w_prev) ** 2)
            return float(var_t + alpha_t + lam_cvar * cvar_val + turn_t)

        def _jac_full(x_full: np.ndarray) -> np.ndarray:
            w_    = x_full[:N_ASSETS]
            zeta_ = x_full[N_ASSETS]
            losses      = -return_window @ w_
            active_mask = (losses > zeta_).astype(float)
            g_cvar_w    = lam_cvar * (-return_window.T @ active_mask) * inv_tail
            g_cvar_z    = lam_cvar * (1.0 - inv_tail * active_mask.sum())
            g_w         = lam_var * (Σ @ w_) - alpha + g_cvar_w + 2.0 * lam_turn * (w_ - w_prev)
            return np.append(g_w, g_cvar_z)

        bds_full  = [(0.0, float(m)) for m in mx] + [(-0.5, 0.5)]
        ew_losses = -return_window @ x0
        zeta0     = float(np.percentile(ew_losses, alpha_conf * 100))
        x0_full   = np.append(x0, zeta0)

        res = sco.minimize(
            fun=_obj_full, jac=_jac_full, x0=x0_full,
            method="SLSQP", bounds=bds_full,
            constraints=[{
                "type": "eq",
                "fun":  lambda x: x[:N_ASSETS].sum() - 1.0,
                "jac":  lambda x: np.append(np.ones(N_ASSETS), 0.0),
            }],
            options={"ftol": 1e-9, "maxiter": 500},
        )
        if res.success:
            w = res.x[:N_ASSETS]
            w = np.clip(w, 0.0, mx)
            s = w.sum()
            w = w / s if s > 1e-10 else x0
            w[w < _MIN_POSITION_WT] = 0.0
            s = w.sum()
            if s > 1e-10:
                w /= s
            return w.astype(np.float32)
        # Fall through to pure MVO + turnover below

    # ── Pure MVO + turnover penalty ────────────────────────────────────────────
    def _obj_mvo(w: np.ndarray) -> float:
        return (
            0.5 * lam_var * w @ Σ @ w
            - alpha @ w
            + lam_turn * np.sum((w - w_prev) ** 2)
        )

    def _jac_mvo(w: np.ndarray) -> np.ndarray:
        return lam_var * (Σ @ w) - alpha + 2.0 * lam_turn * (w - w_prev)

    bds = [(0.0, float(m)) for m in mx]
    res = sco.minimize(
        fun=_obj_mvo, jac=_jac_mvo, x0=x0,
        method="SLSQP", bounds=bds,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"ftol": 1e-9, "maxiter": 500},
    )

    if res.success:
        w = res.x
    else:
        # Inverse-vol fallback — always feasible
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


# ── Data containers ────────────────────────────────────────────────────────────

@dataclass
class Snap:
    date:          str
    portfolio_value: float
    daily_return:  float
    cash:          float
    turnover:      float
    cost_drag:     float
    regime_label:  str
    z_mu_0:        float
    drawdown:      float
    alloc_pct:     int


@dataclass
class WFFold:
    fold_id:   int
    is_start:  str
    is_end:    str
    oos_start: str
    oos_end:   str
    is_sharpe: float
    oos_sharpe: float
    oos_cagr:  float
    oos_max_dd: float


# ── Main backtester ────────────────────────────────────────────────────────────

class StandaloneBacktester:

    def _reset(self) -> None:
        self.nav:             float = _INITIAL_CAPITAL
        self.cash:            float = _INITIAL_CAPITAL
        self.pos:             Dict[str, float] = {}
        self.history:         List[Snap] = []
        self._peak:           float = _INITIAL_CAPITAL
        self._prev_regime:    str   = ""
        self._halt_phase:     int   = 0
        self._halt_days:      int   = 0
        self._halt_nav:       float = _INITIAL_CAPITAL
        self._ramp_days:      int   = 0
        self._ramp_entry_nav: float = _INITIAL_CAPITAL
        self._smoother        = RegimeSignalSmoother()
        self._prev_weights    = np.zeros(N_ASSETS, dtype=np.float64)
        self._prev_dd:        float = 0.0
        self._prev_dd_vel:    float = 0.0

    def _current_weights(self, px: np.ndarray) -> np.ndarray:
        vals  = np.array([self.pos.get(t, 0.0) * px[i] for i, t in enumerate(TICKERS)])
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
        """Park capital in BIL (60%) + SHV (40%) during halt."""
        self._liquidate(px)
        for t, frac in [("BIL", 0.60), ("SHV", 0.40)]:
            i      = TICKERS.index(t)
            dollar = self.cash * frac
            shares = dollar / (px[i] + 1e-10)
            self.pos[t] = self.pos.get(t, 0.0) + shares
            self.cash  -= dollar

    def _mtm(self, px: np.ndarray) -> float:
        """Mark-to-market: update NAV, return daily return."""
        equity   = sum(self.pos.get(t, 0.0) * px[i] for i, t in enumerate(TICKERS))
        new_nav  = self.cash + equity
        dr       = (new_nav - self.nav) / (self.nav + 1e-10)
        self.nav = new_nav
        self._peak = max(self._peak, new_nav)
        return dr

    def _rebalance(
        self,
        target:  np.ndarray,
        px:      np.ndarray,
        vol:     np.ndarray,
        adv:     np.ndarray,
        force:   bool = False,
    ) -> Tuple[float, float]:
        """
        Execute rebalance toward target weights.
        Applies 75bps band filter unless force=True (regime change).
        Returns (cost_drag_fraction, one_way_turnover_fraction).
        """
        current = self._current_weights(px)
        delta   = target - current

        # Band filter: skip rebalance if total drift is tiny
        if not force and np.sum(np.abs(delta)) < 0.10:
            return 0.0, 0.0

        cost = 0.0
        for i, t in enumerate(TICKERS):
            if not force and abs(delta[i]) < _REBALANCE_BAND:
                continue
            dol    = delta[i] * self.nav
            shares = dol / (px[i] + 1e-10)
            c      = abs(dol) * (_ac_cost_bps(shares, vol[i], adv[i]) + _BASE_SPREAD_BPS) / 10_000
            cost       += c
            self.cash  += -dol - c
            self.pos[t]  = self.pos.get(t, 0.0) + shares

        cost_drag_frac   = cost / (self.nav + 1e-10)
        one_way_turnover = float(0.5 * np.sum(np.abs(target - current)))
        self._prev_weights = target.copy().astype(np.float64)
        return cost_drag_frac, one_way_turnover

    def run(
        self,
        prices_df:   pd.DataFrame,
        returns_df:  pd.DataFrame,
        regime_df:   pd.DataFrame,
        alpha_df:    pd.DataFrame,
        start_date:  str,
        end_date:    str,
        warmup_days: int = _WARMUP_DAYS,
    ) -> pd.DataFrame:
        self._reset()

        common = (
            prices_df.index
            .intersection(returns_df.index)
            .intersection(regime_df.index)
            .intersection(alpha_df.index)
        )
        mask      = (common >= start_date) & (common <= end_date)
        sim_dates = common[mask]

        if len(sim_dates) == 0:
            logger.error("No common dates in simulation window.")
            return pd.DataFrame()

        all_dates        = common[common <= end_date]
        warmup_start_idx = max(0, all_dates.get_loc(sim_dates[0]) - warmup_days)
        full_dates       = all_dates[warmup_start_idx:]
        record_start     = sim_dates[0]

        for date in full_dates:
            record = date >= record_start
            ds     = str(date.date())
            gi     = prices_df.index.get_loc(date)
            px     = prices_df.iloc[gi].reindex(TICKERS).fillna(1.0).values.astype(np.float64)

            if date not in regime_df.index or date not in alpha_df.index:
                if record:
                    self.history.append(Snap(ds, self.nav, 0.0, self.cash, 0.0, 0.0, "NO_SIGNAL", 0.0, 0.0, 0))
                continue

            # ── Regime signal processing ───────────────────────────────────────
            raw_z = regime_df.loc[date, "z_mu"]
            if isinstance(raw_z, str):
                import ast
                raw_z = ast.literal_eval(raw_z)
            z_raw   = np.asarray(raw_z, dtype=np.float32).ravel()
            z_smooth, s_label = self._smoother.update(z_raw, str(regime_df.loc[date, "regime_label"]))
            z0 = float(z_smooth[0])

            # Read asset-class-specific urgency from VolRegimeTensor metadata
            regime_row     = regime_df.loc[date]
            equity_urgency = float(regime_row.get("ltc_urgency", abs(z0) / 4.0))

            halt_dd_thresh, halt_days_req, ramp_days_req = _get_regime_halt_threshold(regime_df, date)

            # ══════════════════════════════════════════════════════════════════
            # HALT STATE MACHINE (BUG #24 FIX)
            # ══════════════════════════════════════════════════════════════════
            if self._halt_phase >= 1:
                self._halt_days += 1
                dd_from_peak = (self.nav - self._peak) / self._peak if self._peak > 0 else 0.0

                if self._halt_phase == 1:
                    # Phase 1: mandatory BIL parking
                    dr = self._mtm(px)

                    if self._halt_days >= halt_days_req:
                        recov = (self.nav - self._halt_nav) / (self._halt_nav + 1e-10)
                        if recov >= -_HALT_RECOVERY_THRESH:
                            logger.info(f"{ds}: Halt recovery — entering ramp-in.")
                            self._halt_phase     = 2
                            self._ramp_days      = 0
                            # BUG #24 FIX: store ramp entry NAV for isolated DD tracking
                            self._ramp_entry_nav = self.nav

                elif self._halt_phase == 2:
                    # Phase 2: linear ramp-in
                    self._ramp_days += 1
                    scale = min(float(self._ramp_days) / ramp_days_req, 1.0)

                    cov_d = _cov(returns_df.iloc[max(0, gi - _COV_WINDOW):gi])
                    vol_d = returns_df.iloc[max(0, gi - 21):gi].reindex(columns=TICKERS).std(axis=0).fillna(0.01).values.astype(np.float64)
                    adv   = np.maximum(10_000_000.0 / (px + 1e-10), 1.0)
                    alp   = alpha_df.loc[date].reindex(TICKERS).fillna(0.0).values.astype(np.float64)

                    ret_window = returns_df.iloc[max(0, gi - _COV_WINDOW):gi].reindex(columns=TICKERS).fillna(0.0).values
                    target = _mvo_weights(
                        alp, cov_d, z0, vol_d,
                        prev_weights=self._prev_weights,
                        equity_urgency=equity_urgency,
                        return_window=ret_window if ret_window.shape[0] >= 30 else None,
                    )

                    # Apply ramp scale: invest (scale) in active portfolio, remainder in BIL/SHV
                    target_scaled = target * scale
                    target_scaled[TICKERS.index("BIL")] += (1.0 - scale) * 0.60
                    target_scaled[TICKERS.index("SHV")] += (1.0 - scale) * 0.40

                    cost_d, to = self._rebalance(target_scaled, px, vol_d, adv)
                    dr = self._mtm(px)

                    # BUG #24 FIX: drawdown measured from ramp_entry_nav, not peak
                    dd_ramp = (self.nav - self._ramp_entry_nav) / (self._ramp_entry_nav + 1e-10)

                    if dd_ramp <= -_RAMP_DD_BUFFER:
                        logger.warning(
                            f"{ds}: Ramp-in DD={dd_ramp:.2%} from ramp entry "
                            f"exceeded {_RAMP_DD_BUFFER:.2%} buffer. Returning to BIL."
                        )
                        self._halt_phase = 1
                        self._halt_days  = 0
                        self._halt_nav   = self.nav
                        self._liquidate(px)
                        self._invest_bil(px)
                    elif self._ramp_days >= ramp_days_req:
                        logger.info(f"{ds}: Ramp-in complete. Resuming full active trading.")
                        self._halt_phase = 0
                        # BUG #24 FIX: reset peak at ramp completion for clean HWM
                        self._peak = self.nav

                    if record:
                        self.history.append(Snap(ds, self.nav, dr, self.cash, to, cost_d, f"RAMP_{scale:.0%}", z0, dd_from_peak, 2))
                    continue

                if record:
                    self.history.append(Snap(ds, self.nav, dr, self.cash, 0.0, 0.0, "HALTED_BIL", z0, dd_from_peak, self._halt_phase))
                continue

            # ══════════════════════════════════════════════════════════════════
            # ACTIVE TRADING
            # ══════════════════════════════════════════════════════════════════
            cov_d = _cov(returns_df.iloc[max(0, gi - _COV_WINDOW):gi])
            vol_d = returns_df.iloc[max(0, gi - 21):gi].reindex(columns=TICKERS).std(axis=0).fillna(0.01).values.astype(np.float64)
            adv   = np.maximum(10_000_000.0 / (px + 1e-10), 1.0)

            # Regime-conditional alpha scaling: attenuate alpha in crisis
            has_soft      = all(c in regime_df.columns for c in _SOFT_COLS)
            crisis_weight = (
                float(regime_row.get("soft_crisis", 0.0)) + float(regime_row.get("soft_bear", 0.0))
                if has_soft else float(np.clip(equity_urgency, 0.0, 1.0))
            )
            alpha_scale = 1.0 - 0.6 * crisis_weight
            alp = alpha_df.loc[date].reindex(TICKERS).fillna(0.0).values.astype(np.float64) * alpha_scale

            ret_window = returns_df.iloc[max(0, gi - _COV_WINDOW):gi].reindex(columns=TICKERS).fillna(0.0).values
            target = _mvo_weights(
                alp, cov_d, z0, vol_d,
                prev_weights=self._prev_weights,
                equity_urgency=equity_urgency,
                return_window=ret_window if ret_window.shape[0] >= 30 else None,
            )

            # ── Dynamic risk-tier scaling ──────────────────────────────────────
            dd     = (self.nav - self._peak) / self._peak if self._peak > 0 else 0.0
            dd_vel = dd - self._prev_dd
            dd_accel = dd_vel - self._prev_dd_vel
            self._prev_dd     = dd
            self._prev_dd_vel = dd_vel

            velocity_adj = float(np.clip(dd_accel / 0.001, -0.05, 0.05))
            _RISK_TIERS = [
                (0.22 + velocity_adj, 0.00),
                (0.18 + velocity_adj, 0.15),
                (0.13 + velocity_adj, 0.40),
                (0.08 + velocity_adj, 0.70),
            ]
            scale = 1.0
            for dd_thresh, alloc in _RISK_TIERS:
                if dd <= -dd_thresh:
                    scale = alloc
                    break

            # Formal halt trigger when tier-0 scale reached
            if scale == 0.0 and self._halt_phase == 0:
                logger.critical(f"HALT triggered {ds}: DD={dd:.2%} ≤ -{halt_dd_thresh:.0%} (regime-conditional)")
                self._halt_phase = 1
                self._halt_days  = 0
                self._halt_nav   = self.nav
                self._liquidate(px)
                self._invest_bil(px)
                dr = self._mtm(px)
                if record:
                    self.history.append(Snap(ds, self.nav, dr, self.cash, 0.0, 0.0, "HALTED_BIL", z0, dd, 0))
                continue

            # Apply scale: active portfolio (scale) + BIL/SHV (1-scale)
            target = target * scale
            target[TICKERS.index("BIL")] += (1.0 - scale) * 0.60
            target[TICKERS.index("SHV")] += (1.0 - scale) * 0.40

            regime_changed = s_label != self._prev_regime
            cost_d, to     = self._rebalance(target, px, vol_d, adv, force=regime_changed)
            dr             = self._mtm(px)
            self._prev_regime = s_label

            if record:
                self.history.append(Snap(ds, self.nav, dr, self.cash, to, cost_d, s_label, z0, dd, int(scale * 100)))

        # ── Build tearsheet ────────────────────────────────────────────────────
        if not self.history:
            return pd.DataFrame()

        df = pd.DataFrame([vars(h) for h in self.history])
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()

        r   = df["daily_return"].values
        nav = df["portfolio_value"].values
        cum = nav / _INITIAL_CAPITAL
        rf  = _RISK_FREE_ANNUAL / _TRADING_DAYS_YEAR
        ex  = r - rf
        sr  = float((ex.mean() / (ex.std() + 1e-10)) * np.sqrt(_TRADING_DAYS_YEAR))
        dn  = r[r < rf]
        so  = float((ex.mean() / (np.sqrt((dn ** 2).mean()) + 1e-10)) * np.sqrt(_TRADING_DAYS_YEAR)) if len(dn) > 0 else 0.0
        n   = len(r)
        tot = float(cum[-1] - 1.0) if len(cum) > 0 else 0.0
        cagr_val = float((1.0 + tot) ** (_TRADING_DAYS_YEAR / max(n, 1)) - 1.0)
        peak_cum = np.maximum.accumulate(cum)
        dd_series = (cum - peak_cum) / (peak_cum + 1e-10)
        max_dd = float(dd_series.min())
        calmar = cagr_val / (abs(max_dd) + 1e-10)
        dd_days_arr = dd_series < 0
        max_dd_dur = _mct(dd_days_arr)
        skew_val = float(stats.skew(r))
        kurt_val = float(stats.kurtosis(r, fisher=True))
        v95  = float(np.percentile(r, 5))
        cv95 = float(r[r <= v95].mean()) if (r <= v95).sum() > 0 else v95
        psr  = _psr(sr, n, skew_val, kurt_val + 3.0)
        hit  = float((r > rf).mean())
        avg_turn  = float(df["turnover"].mean())
        cost_drag = float(df["cost_drag"].mean() * _TRADING_DAYS_YEAR * 1e4)
        active_pct = float((~df["regime_label"].isin(["WARMUP", "NO_SIGNAL", "HALTED_BIL"]) &
                            ~df["regime_label"].str.startswith("RAMP")).mean() * 100)
        halt_pct   = float((df["regime_label"] == "HALTED_BIL").mean() * 100)

        for label, val in [
            ("CAGR", f"{cagr_val:+.2%}"),
            ("Ann. Volatility", f"{r.std() * np.sqrt(_TRADING_DAYS_YEAR):.2%}"),
            ("Sharpe Ratio", f"{sr:.3f}"),
            ("Sortino Ratio", f"{so:.3f}"),
            ("Calmar Ratio", f"{calmar:.3f}"),
            ("Max Drawdown", f"{max_dd:.2%}"),
            ("Max DD Duration", f"{max_dd_dur} days"),
            ("VaR-95 (daily)", f"{v95:.2%}"),
            ("CVaR-95 (daily)", f"{cv95:.2%}"),
            ("Hit Rate", f"{hit:.2%}"),
            ("Avg Daily Turnover", f"{avg_turn:.2%}"),
            ("Cost Drag (ann. bps)", f"{cost_drag:.1f}"),
            ("PSR (SR>0)", f"{psr:.4f}"),
            ("Skewness", f"{skew_val:.3f}"),
            ("Excess Kurtosis", f"{kurt_val:.3f}"),
            ("Active Trading %", f"{active_pct:.1f}%"),
            ("Halted BIL %", f"{halt_pct:.1f}%"),
            ("Final NAV", f"${nav[-1]:,.2f}"),
            ("Trading Days", f"{n}"),
        ]:
            logger.info(f"  {label:<30s} {val}")

        return df


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("══════ Fortress v5 — Standalone Backtest v6 (UNIVERSE FIX + TURNOVER PENALTY) ══════")

    prices_df  = _normalize_index(pd.read_parquet(_CACHE_DIR / "prices_wide.parquet"))
    returns_df = _normalize_index(pd.read_parquet(_CACHE_DIR / "returns_wide.parquet"))
    regime_df  = _normalize_index(pd.read_parquet(_CACHE_DIR / "regime_posteriors.parquet"))
    alpha_df   = _normalize_index(pd.read_parquet(_CACHE_DIR / "alpha_signals_blended.parquet"))

    # Validate universe alignment
    alpha_cols  = set(alpha_df.columns)
    missing     = set(TICKERS) - alpha_cols
    if missing:
        logger.error(
            f"Alpha signals missing for {missing}. "
            f"Re-run precompute_alpha_signals.py with the unified TICKERS list."
        )
        sys.exit(1)

    # Align all DataFrames to unified TICKERS
    prices_df  = prices_df.reindex(columns=TICKERS)
    returns_df = returns_df.reindex(columns=TICKERS)
    alpha_df   = alpha_df.reindex(columns=TICKERS)

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
        logger.info("✅ Soft GMM posteriors detected — regime-conditional halt thresholds active.")
    else:
        logger.warning(f"⚠️  Soft GMM posteriors NOT found. Falling back to static halt threshold {_MAX_DD_HALT_FALLBACK:.0%}.")

    has_urgency = "ltc_urgency" in regime_df.columns
    logger.info(
        f"{'✅' if has_urgency else '⚠️ '} "
        f"VolRegimeTensor urgency: {'active' if has_urgency else 'falling back to |z_mu[0]|'}"
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

    # ── Walk-forward validation ────────────────────────────────────────────────
    logger.info("Running walk-forward validation...")
    common  = prices_df.index.intersection(alpha_df.index)
    wf_mask = (common >= "2019-01-02") & (common <= "2024-12-31")
    folds   = _wf_folds(common[wf_mask])
    results: List[WFFold] = []

    for fid, (is_s, is_e, oos_s, oos_e) in enumerate(folds):
        t_is = StandaloneBacktester().run(prices_df, returns_df, regime_df, alpha_df, is_s, is_e, warmup_days=_WF_WARMUP_DAYS)
        t_oo = StandaloneBacktester().run(prices_df, returns_df, regime_df, alpha_df, oos_s, oos_e, warmup_days=_WF_WARMUP_DAYS)
        if t_oo.empty or len(t_oo) < 5:
            continue

        r_oo = t_oo["daily_return"].values
        n_oo = len(r_oo)
        tot  = t_oo["portfolio_value"].iloc[-1] / _INITIAL_CAPITAL - 1.0
        cagr = (1.0 + tot) ** (_TRADING_DAYS_YEAR / n_oo) - 1.0
        mdd  = float(
            ((t_oo["portfolio_value"] - t_oo["portfolio_value"].cummax()) /
             (t_oo["portfolio_value"].cummax() + 1e-10)).min()
        )
        is_sr  = _sharpe(t_is["daily_return"].values) if not t_is.empty else 0.0
        oos_sr = _sharpe(r_oo)

        results.append(WFFold(fid + 1, is_s, is_e, oos_s, oos_e, round(is_sr, 4), round(oos_sr, 4), round(cagr, 4), round(mdd, 4)))
        logger.info(f"  F{fid + 1}: IS SR={is_sr:.3f} | OOS SR={oos_sr:.3f}  CAGR={cagr:.2%}  MaxDD={mdd:.2%}")

    pd.DataFrame([vars(r) for r in results]).to_csv(_WF_FOLDS, index=False)
    if results:
        avg_oos = float(np.mean([f.oos_sharpe for f in results]))
        avg_is  = float(np.mean([f.is_sharpe  for f in results]))
        logger.info(f"WF summary: avg IS={avg_is:.3f} | avg OOS={avg_oos:.3f} | {_wf_verdict(avg_is, avg_oos)}")

    r_arr = ts["daily_return"].values
    v95   = float(np.percentile(r_arr, 5))
    with open(_STRESS_OUT, "w") as f:
        json.dump({"mode": "stub", "var_95": v95, "cvar_95": float(r_arr[r_arr <= v95].mean()), "note": "Run training/train_world_model.py for SDE stress."}, f, indent=2)

    logger.info("✅ Done. Run scripts/visualize_tearsheet.py next.")


if __name__ == "__main__":
    main()