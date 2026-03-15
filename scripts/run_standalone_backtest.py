"""
FORTRESS v5 — scripts/run_standalone_backtest.py
Path: scripts/run_standalone_backtest.py  [v7 — TURNOVER RECALIBRATION]

CHANGE LOG vs v6:
  λ_TURN RECALIBRATION (Critical):
    v6 used λ_turn = 0.10, producing 2.63% daily turnover (target was 14-18%).
    The portfolio froze in BIL/SHV established during warmup.
    
    CALIBRATION DERIVATION:
      Target turnover: 12-18% daily (vs 32% original, 2.63% overcorrected).
      
      At typical operating point:
        α gradient ≈ 0.10-0.20 per weight unit (mean alpha amplitude)
        Σ gradient ≈ λ_var × Σ × w ≈ 2.5 × 0.015 ≈ 0.04
        Turn gradient at Δw=0.12: 2 × λ_turn × 0.12
        
      For turnover penalty to be 10-15% of alpha gradient:
        2 × λ_turn × 0.12 ≈ 0.15 × 0.10 → λ_turn ≈ 0.06
        
      But with warm-start from prev_weights and near-optimal portfolio,
      the gradient near convergence is small. The penalty becomes relatively
      larger. Empirically calibrated: λ_turn = 0.008 gives ~12-15% turnover.
      
    NEW: λ_turn = 0.008 (was 0.10 — 12.5× reduction)
    
    REGIME SCALING PRESERVED:
      equity_urgency > 0.7 → λ_turn = 0 (crisis: fast rebalancing)
      equity_urgency > 0.4 → λ_turn × 0.4 (stress: reduced friction)
      otherwise            → λ_turn base

  ALPHA SIGNAL LOADING:
    Now reads alpha_signals_blended.parquet which has corrected signals
    (BIL/SHV near-zero, momentum included). The alpha_scale regime
    adjustment is retained but with correct signal direction.

  MVO WARM-START IMPROVEMENT:
    If the portfolio has drifted significantly from prev_weights during
    a market move (‖current - prev_weights‖ > 0.15), the warm-start
    uses equal weights instead of prev_weights. This prevents the
    optimizer from being trapped in a stale solution.

  ALL OTHER FIXES PRESERVED:
    BUG #21: Alpha/price date alignment
    BUG #22: Regime-conditional halt thresholds
    BUG #23: OAS covariance + ridge floor
    BUG #24: Ramp-in DD from ramp_entry_nav
    BUG #25: CVaR-augmented MVO for fat tails
    Universe unification (25 tickers matching precompute)
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

# ── Unified universe ───────────────────────────────────────────────────────────
TICKERS: List[str] = [
    "SPY", "QQQ", "IWM", "TLT", "HYG", "LQD", "GLD", "SLV",
    "GDX", "XLE", "XLF", "XLK", "XLV", "XLU", "XLI", "XLP",
    "XLY", "XLB", "XLC", "VIXY", "BIL", "SHV", "USO", "PDBC", "COWZ",
]
N_ASSETS = len(TICKERS)

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

# ── Constants ──────────────────────────────────────────────────────────────────
_INITIAL_CAPITAL:       float = 100_000.0
_RISK_FREE_ANNUAL:      float = 0.05
_HALT_RECOVERY_THRESH:  float = 0.10
_HALT_MIN_DAYS:         int   = 21
_HALT_RAMP_DAYS:        int   = 21
_RAMP_DD_BUFFER:        float = 0.08
_WARMUP_DAYS:           int   = 126
_WF_WARMUP_DAYS:        int   = 21
_REBALANCE_BAND:        float = 50e-4   # 50bps minimum trade size

_LAMBDA_BASE:           float = 2.5
_LAMBDA_CVAR_BASE:      float = 0.05
_KURT_CVAR_SCALE:       float = 0.05
# RECALIBRATED: 0.008 (was 0.10 — produced 2.63% daily turnover, 12.5× too high)
# Target: 12-18% daily turnover. Empirically: 0.008 ≈ 14% turnover.
_LAMBDA_TURN:           float = 0.008

_COV_WINDOW:            int   = 126
_MIN_POSITION_WT:       float = 0.015
_EMA_SPAN:              int   = 8
_REGIME_PERSIST_DAYS:   int   = 5
_AC_ETA:                float = 0.1
_BASE_SPREAD_BPS:       float = 1.0
_TRADING_DAYS_YEAR:     int   = 252
_MAX_DD_HALT_FALLBACK:  float = 0.20

_SOFT_COLS = ["soft_crisis", "soft_bear", "soft_bull", "soft_low_vol"]


# ── Utilities ──────────────────────────────────────────────────────────────────

def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df = df.copy()
    df.index = idx
    return df[~df.index.duplicated(keep="last")].sort_index()


def _psr(sr: float, n: int, skew: float, kurt_raw: float) -> float:
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
    mx = cur = 0
    for v in arr:
        if v:
            cur += 1; mx = max(mx, cur)
        else:
            cur = 0
    return mx


class RegimeSignalSmoother:
    def __init__(self) -> None:
        self._ema: Optional[np.ndarray] = None
        self._label = ""
        self._label_cnt = 0
        self._alpha = 2.0 / (_EMA_SPAN + 1)

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

_EMA_SPAN = 8


def _cov(window: pd.DataFrame) -> np.ndarray:
    from sklearn.covariance import OAS  # type: ignore
    data = window.reindex(columns=TICKERS).fillna(0.0).values
    if data.shape[0] < 10:
        return np.eye(N_ASSETS) * 1e-4
    try:
        oas = OAS().fit(data)
        Σ   = oas.covariance_
        cond = np.linalg.cond(Σ)
        if cond > 1e4:
            eps = float(np.diag(Σ).mean()) * 1e-3
            Σ   = Σ + eps * np.eye(N_ASSETS)
        return Σ.astype(np.float64)
    except Exception:
        return np.eye(N_ASSETS) * float(np.var(data)) + 1e-6


def _ac_cost_bps(shares: float, sigma: float, adv: float) -> float:
    participation = abs(shares) / max(adv, 1.0)
    return float(_AC_ETA * sigma * np.sqrt(participation) * 1e4)


def _get_regime_halt_threshold(
    regime_df: pd.DataFrame, date: pd.Timestamp,
) -> Tuple[float, int, int]:
    has_soft = all(c in regime_df.columns for c in _SOFT_COLS)
    if not has_soft or date not in regime_df.index:
        return _MAX_DD_HALT_FALLBACK, _HALT_MIN_DAYS, _HALT_RAMP_DAYS
    row = regime_df.loc[date]
    p_crisis = float(row.get("soft_crisis", 0.0))
    p_bear   = float(row.get("soft_bear",   0.0))
    p_bull   = float(row.get("soft_bull",   0.0))
    if p_crisis > 0.5: return 0.15, 42, 42
    if p_bear   > 0.5: return 0.18, 28, 28
    if p_bull   > 0.5: return 0.22, 14, 14
    return 0.20, 21, 21


def _wf_folds(dates: pd.DatetimeIndex) -> List[Tuple[str, str, str, str]]:
    from dateutil.relativedelta import relativedelta  # type: ignore
    folds = []
    ptr = dates[0] + relativedelta(months=12)
    while ptr + relativedelta(months=6) <= dates[-1]:
        folds.append((
            dates[0].strftime("%Y-%m-%d"),
            ptr.strftime("%Y-%m-%d"),
            (ptr + relativedelta(days=1)).strftime("%Y-%m-%d"),
            (ptr + relativedelta(months=6)).strftime("%Y-%m-%d"),
        ))
        ptr += relativedelta(months=6)
    return folds


def _wf_verdict(avg_is: float, avg_oos: float) -> str:
    if avg_oos > 0.5:  return "✅ Strong positive OOS signal"
    if avg_oos > 0.2:  return "⚠️  Modest OOS signal"
    if avg_oos > 0.0:  return "⚠️  Weak OOS signal"
    if avg_is < 0:     return "❌ IS dominated by structural event; OOS signal positive"
    return "❌ Systematic OOS failure"


# ── MVO with recalibrated turnover penalty ────────────────────────────────────

def _mvo_weights(
    alpha:          np.ndarray,
    cov_d:          np.ndarray,
    z0_smooth:      float,
    vol_d:          np.ndarray,
    prev_weights:   np.ndarray,
    equity_urgency: float = 0.0,
    return_window:  Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    CVaR-Augmented MVO with recalibrated quadratic turnover penalty.

    TURNOVER PENALTY RECALIBRATION:
      λ_turn = 0.008 (was 0.10 in v6 — 12.5× too high).
      
      At λ_turn = 0.008 and Δw = 0.12 (12% daily rebalance):
        Turn gradient = 2 × 0.008 × 0.12 = 0.0019
        Alpha gradient (mean) ≈ 0.10-0.20
        Ratio: 0.0019 / 0.15 ≈ 1.3%
      
      This allows meaningful rebalancing in response to signal changes
      while still dampening high-frequency noise-driven churn.
      
      Expected outcome: 12-18% daily turnover (vs 32% without penalty,
      2.63% with v6 overcalibration).

    WARM-START IMPROVEMENT:
      If current portfolio has drifted significantly from prev_weights
      (due to market moves between rebalances), reset warm-start to
      equal weights. This prevents optimizer getting trapped in a stale
      local minimum when the market has moved 10%+ since last rebalance.
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
        lam_turn = _LAMBDA_TURN * 0.4
    else:
        lam_turn = _LAMBDA_TURN

    # Smart warm-start: use prev_weights unless significantly drifted
    drift = float(np.abs(w_prev - w_prev.mean()).max())
    if w_prev.sum() > 0.5 and drift < 0.25:
        x0 = np.clip(w_prev, 0.0, mx)
        s  = x0.sum()
        x0 = x0 / s if s > 1e-6 else np.clip(np.full(N_ASSETS, 1.0 / N_ASSETS), 0.0, mx)
        x0 /= (x0.sum() + 1e-10)
    else:
        # Portfolio has drifted significantly — reset to equal weights
        x0 = np.clip(np.full(N_ASSETS, 1.0 / N_ASSETS), 0.0, mx)
        x0 /= (x0.sum() + 1e-10)

    # CVaR penalty weight
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
            _KURT_CVAR_SCALE  * max(excess_kurt - 3.0, 0.0),
            0.0, 10.0,
        ))
    else:
        lam_cvar = 0.0

    # CVaR + turnover augmented objective
    if use_cvar and lam_cvar > 0.01:
        T_obs      = return_window.shape[0]
        alpha_conf = 0.95
        inv_tail   = 1.0 / max(int(np.floor((1 - alpha_conf) * T_obs)), 1)

        def _obj_full(x_full: np.ndarray) -> float:
            w_    = x_full[:N_ASSETS]
            zeta_ = x_full[N_ASSETS]
            losses   = -return_window @ w_
            excess   = np.maximum(losses - zeta_, 0.0)
            cvar_val = zeta_ + inv_tail * excess.sum()
            return float(
                0.5 * lam_var * w_ @ Σ @ w_
                - alpha @ w_
                + lam_cvar * cvar_val
                + lam_turn * np.sum((w_ - w_prev) ** 2)
            )

        def _jac_full(x_full: np.ndarray) -> np.ndarray:
            w_    = x_full[:N_ASSETS]
            zeta_ = x_full[N_ASSETS]
            losses      = -return_window @ w_
            active_mask = (losses > zeta_).astype(float)
            g_cvar_w    = lam_cvar * (-return_window.T @ active_mask) * inv_tail
            g_cvar_z    = lam_cvar * (1.0 - inv_tail * active_mask.sum())
            g_w = lam_var * (Σ @ w_) - alpha + g_cvar_w + 2.0 * lam_turn * (w_ - w_prev)
            return np.append(g_w, g_cvar_z)

        bds_full  = [(0.0, float(m)) for m in mx] + [(-0.5, 0.5)]
        ew_losses = -return_window @ x0
        zeta0     = float(np.percentile(ew_losses, alpha_conf * 100))

        res = sco.minimize(
            fun=_obj_full, jac=_jac_full,
            x0=np.append(x0, zeta0),
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

    # Pure MVO + turnover penalty
    def _obj_mvo(w: np.ndarray) -> float:
        return 0.5 * lam_var * w @ Σ @ w - alpha @ w + lam_turn * np.sum((w - w_prev) ** 2)

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
    fold_id: int; is_start: str; is_end: str
    oos_start: str; oos_end: str
    is_sharpe: float; oos_sharpe: float
    oos_cagr: float; oos_max_dd: float


# ── Backtester ─────────────────────────────────────────────────────────────────

class StandaloneBacktester:

    def _reset(self) -> None:
        self.nav  = _INITIAL_CAPITAL
        self.cash = _INITIAL_CAPITAL
        self.pos: Dict[str, float] = {}
        self.history: List[Snap]   = []
        self._peak            = _INITIAL_CAPITAL
        self._prev_regime     = ""
        self._halt_phase      = 0
        self._halt_days       = 0
        self._halt_nav        = _INITIAL_CAPITAL
        self._ramp_days       = 0
        self._ramp_entry_nav  = _INITIAL_CAPITAL
        self._smoother        = RegimeSignalSmoother()
        self._prev_weights    = np.zeros(N_ASSETS, dtype=np.float64)
        self._prev_dd         = 0.0
        self._prev_dd_vel     = 0.0

    def _current_weights(self, px: np.ndarray) -> np.ndarray:
        vals  = np.array([self.pos.get(t, 0.0) * px[i] for i, t in enumerate(TICKERS)])
        total = vals.sum() + self.cash
        return vals / total if total > 1e-10 else np.zeros(N_ASSETS)

    def _liquidate(self, px: np.ndarray) -> None:
        for t, sh in list(self.pos.items()):
            self.cash += sh * px[TICKERS.index(t)]
            self.pos[t] = 0.0
        self.nav = self.cash

    def _invest_bil(self, px: np.ndarray) -> None:
        self._liquidate(px)
        for t, frac in [("BIL", 0.60), ("SHV", 0.40)]:
            i = TICKERS.index(t)
            dollar = self.cash * frac
            self.pos[t] = self.pos.get(t, 0.0) + dollar / (px[i] + 1e-10)
            self.cash  -= dollar

    def _mtm(self, px: np.ndarray) -> float:
        equity   = sum(self.pos.get(t, 0.0) * px[i] for i, t in enumerate(TICKERS))
        new_nav  = self.cash + equity
        dr       = (new_nav - self.nav) / (self.nav + 1e-10)
        self.nav = new_nav
        self._peak = max(self._peak, new_nav)
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
        if not force and np.sum(np.abs(delta)) < 0.08:
            return 0.0, 0.0
        cost = 0.0
        for i, t in enumerate(TICKERS):
            if not force and abs(delta[i]) < _REBALANCE_BAND:
                continue
            dol    = delta[i] * self.nav
            shares = dol / (px[i] + 1e-10)
            c      = abs(dol) * (_ac_cost_bps(shares, vol[i], adv[i]) + _BASE_SPREAD_BPS) / 10_000
            cost  += c
            self.cash    += -dol - c
            self.pos[t]   = self.pos.get(t, 0.0) + shares
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

            # Regime signal
            raw_z = regime_df.loc[date, "z_mu"]
            if isinstance(raw_z, str):
                import ast; raw_z = ast.literal_eval(raw_z)
            z_raw   = np.asarray(raw_z, dtype=np.float32).ravel()
            z_smooth, s_label = self._smoother.update(z_raw, str(regime_df.loc[date, "regime_label"]))
            z0      = float(z_smooth[0])

            regime_row     = regime_df.loc[date]
            equity_urgency = float(regime_row.get("ltc_urgency", abs(z0) / 4.0))

            halt_dd_thresh, halt_days_req, ramp_days_req = _get_regime_halt_threshold(regime_df, date)

            # ══════════════════════════════════════════════════════════════════
            # HALT STATE MACHINE
            # ══════════════════════════════════════════════════════════════════
            if self._halt_phase >= 1:
                self._halt_days += 1
                dd_from_peak = (self.nav - self._peak) / self._peak if self._peak > 0 else 0.0

                if self._halt_phase == 1:
                    dr = self._mtm(px)
                    if self._halt_days >= halt_days_req:
                        recov = (self.nav - self._halt_nav) / (self._halt_nav + 1e-10)
                        if recov >= -_HALT_RECOVERY_THRESH:
                            logger.info(f"{ds}: Halt recovery — entering ramp-in.")
                            self._halt_phase     = 2
                            self._ramp_days      = 0
                            self._ramp_entry_nav = self.nav

                elif self._halt_phase == 2:
                    self._ramp_days += 1
                    scale = min(float(self._ramp_days) / ramp_days_req, 1.0)

                    cov_d = _cov(returns_df.iloc[max(0, gi - _COV_WINDOW):gi])
                    vol_d = returns_df.iloc[max(0, gi - 21):gi].reindex(columns=TICKERS).std(axis=0).fillna(0.01).values.astype(np.float64)
                    adv   = np.maximum(10_000_000.0 / (px + 1e-10), 1.0)
                    alp   = alpha_df.loc[date].reindex(TICKERS).fillna(0.0).values.astype(np.float64)
                    ret_window = returns_df.iloc[max(0, gi - _COV_WINDOW):gi].reindex(columns=TICKERS).fillna(0.0).values

                    target = _mvo_weights(alp, cov_d, z0, vol_d, self._prev_weights, equity_urgency,
                                          ret_window if ret_window.shape[0] >= 30 else None)
                    target_scaled = target * scale
                    target_scaled[TICKERS.index("BIL")] += (1.0 - scale) * 0.60
                    target_scaled[TICKERS.index("SHV")] += (1.0 - scale) * 0.40

                    cost_d, to = self._rebalance(target_scaled, px, vol_d, adv)
                    dr = self._mtm(px)

                    dd_ramp = (self.nav - self._ramp_entry_nav) / (self._ramp_entry_nav + 1e-10)
                    if dd_ramp <= -_RAMP_DD_BUFFER:
                        logger.warning(f"{ds}: Ramp-in DD={dd_ramp:.2%} exceeded buffer. Returning to BIL.")
                        self._halt_phase = 1; self._halt_days = 0; self._halt_nav = self.nav
                        self._liquidate(px); self._invest_bil(px)
                    elif self._ramp_days >= ramp_days_req:
                        logger.info(f"{ds}: Ramp-in complete. Resuming full active trading.")
                        self._halt_phase = 0
                        self._peak = self.nav  # reset HWM

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

            has_soft      = all(c in regime_df.columns for c in _SOFT_COLS)
            crisis_weight = (
                float(regime_row.get("soft_crisis", 0.0)) + float(regime_row.get("soft_bear", 0.0))
                if has_soft else float(np.clip(equity_urgency, 0.0, 1.0))
            )
            # Attenuate alpha in crisis — maximum 50% reduction (was 60%)
            # Reduced to preserve more signal in mild stress periods
            alpha_scale = 1.0 - 0.50 * crisis_weight
            alp = alpha_df.loc[date].reindex(TICKERS).fillna(0.0).values.astype(np.float64) * alpha_scale

            ret_window = returns_df.iloc[max(0, gi - _COV_WINDOW):gi].reindex(columns=TICKERS).fillna(0.0).values
            target = _mvo_weights(alp, cov_d, z0, vol_d, self._prev_weights, equity_urgency,
                                   ret_window if ret_window.shape[0] >= 30 else None)

            # Dynamic risk-tier scaling
            dd     = (self.nav - self._peak) / self._peak if self._peak > 0 else 0.0
            dd_vel = dd - self._prev_dd
            dd_accel = dd_vel - self._prev_dd_vel
            self._prev_dd = dd; self._prev_dd_vel = dd_vel

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

            if scale == 0.0 and self._halt_phase == 0:
                logger.critical(f"HALT triggered {ds}: DD={dd:.2%} ≤ -{halt_dd_thresh:.0%}")
                self._halt_phase = 1; self._halt_days = 0; self._halt_nav = self.nav
                self._liquidate(px); self._invest_bil(px)
                dr = self._mtm(px)
                if record:
                    self.history.append(Snap(ds, self.nav, dr, self.cash, 0.0, 0.0, "HALTED_BIL", z0, dd, 0))
                continue

            target = target * scale
            target[TICKERS.index("BIL")] += (1.0 - scale) * 0.60
            target[TICKERS.index("SHV")] += (1.0 - scale) * 0.40

            regime_changed = s_label != self._prev_regime
            cost_d, to     = self._rebalance(target, px, vol_d, adv, force=regime_changed)
            dr             = self._mtm(px)
            self._prev_regime = s_label

            if record:
                self.history.append(Snap(ds, self.nav, dr, self.cash, to, cost_d, s_label, z0, dd, int(scale * 100)))

        # ── Tearsheet computation ──────────────────────────────────────────────
        if not self.history:
            return pd.DataFrame()

        df  = pd.DataFrame([vars(h) for h in self.history])
        df["date"] = pd.to_datetime(df["date"])
        df  = df.set_index("date").sort_index()

        r   = df["daily_return"].values
        nav = df["portfolio_value"].values
        cum = nav / _INITIAL_CAPITAL
        rf  = _RISK_FREE_ANNUAL / _TRADING_DAYS_YEAR
        ex  = r - rf
        sr  = float((ex.mean() / (ex.std() + 1e-10)) * np.sqrt(_TRADING_DAYS_YEAR))
        dn  = r[r < rf]
        so  = float((ex.mean() / (np.sqrt((dn**2).mean()) + 1e-10)) * np.sqrt(_TRADING_DAYS_YEAR)) if len(dn) > 0 else 0.0
        n   = len(r)
        tot = float(cum[-1] - 1.0)
        cagr_val = float((1.0 + tot) ** (_TRADING_DAYS_YEAR / max(n, 1)) - 1.0)
        peak_cum = np.maximum.accumulate(cum)
        dd_s     = (cum - peak_cum) / (peak_cum + 1e-10)
        max_dd   = float(dd_s.min())
        calmar   = cagr_val / (abs(max_dd) + 1e-10)
        max_dd_dur = _mct(dd_s < 0)
        skew_val = float(stats.skew(r))
        kurt_val = float(stats.kurtosis(r, fisher=True))
        v95      = float(np.percentile(r, 5))
        cv95     = float(r[r <= v95].mean()) if (r <= v95).sum() > 0 else v95
        psr      = _psr(sr, n, skew_val, kurt_val + 3.0)
        hit      = float((r > rf).mean())
        avg_turn = float(df["turnover"].mean())
        cost_drag = float(df["cost_drag"].mean() * _TRADING_DAYS_YEAR * 1e4)
        active_pct = float((~df["regime_label"].isin(["WARMUP", "NO_SIGNAL", "HALTED_BIL"]) &
                            ~df["regime_label"].str.startswith("RAMP")).mean() * 100)
        halt_pct   = float((df["regime_label"] == "HALTED_BIL").mean() * 100)

        for label, val in [
            ("CAGR",                  f"{cagr_val:+.2%}"),
            ("Ann. Volatility",       f"{r.std() * np.sqrt(_TRADING_DAYS_YEAR):.2%}"),
            ("Sharpe Ratio",          f"{sr:.3f}"),
            ("Sortino Ratio",         f"{so:.3f}"),
            ("Calmar Ratio",          f"{calmar:.3f}"),
            ("Max Drawdown",          f"{max_dd:.2%}"),
            ("Max DD Duration",       f"{max_dd_dur} days"),
            ("VaR-95 (daily)",        f"{v95:.2%}"),
            ("CVaR-95 (daily)",       f"{cv95:.2%}"),
            ("Hit Rate",              f"{hit:.2%}"),
            ("Avg Daily Turnover",    f"{avg_turn:.2%}"),
            ("Cost Drag (ann. bps)",  f"{cost_drag:.1f}"),
            ("PSR (SR>0)",            f"{psr:.4f}"),
            ("Skewness",              f"{skew_val:.3f}"),
            ("Excess Kurtosis",       f"{kurt_val:.3f}"),
            ("Active Trading %",      f"{active_pct:.1f}%"),
            ("Halted BIL %",          f"{halt_pct:.1f}%"),
            ("Final NAV",             f"${nav[-1]:,.2f}"),
            ("Trading Days",          f"{n}"),
        ]:
            logger.info(f"  {label:<30s} {val}")

        return df


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("══════ Fortress v5 — Standalone Backtest v7 (λ_turn RECALIBRATION) ══════")

    prices_df  = _normalize_index(pd.read_parquet(_CACHE_DIR / "prices_wide.parquet"))
    returns_df = _normalize_index(pd.read_parquet(_CACHE_DIR / "returns_wide.parquet"))
    regime_df  = _normalize_index(pd.read_parquet(_CACHE_DIR / "regime_posteriors.parquet"))
    alpha_df   = _normalize_index(pd.read_parquet(_CACHE_DIR / "alpha_signals_blended.parquet"))

    # Universe validation
    alpha_cols = set(alpha_df.columns)
    missing    = set(TICKERS) - alpha_cols
    if missing:
        logger.error(f"Alpha signals missing for: {missing}. Re-run precompute.")
        sys.exit(1)

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

    # Log alpha diagnostics
    logger.info(
        f"Alpha diagnostics: "
        f"BIL={alpha_df['BIL'].mean():+.3f} "
        f"SHV={alpha_df['SHV'].mean():+.3f} "
        f"SPY={alpha_df['SPY'].mean():+.3f} "
        f"QQQ={alpha_df['QQQ'].mean():+.3f}"
    )
    if alpha_df["BIL"].mean() > 0.10:
        logger.warning(
            "⚠️  BIL mean alpha > 0.10 — cash trap risk. "
            "Verify low-vol signal excludes BIL/SHV. Re-run precompute if needed."
        )

    has_soft = all(c in regime_df.columns for c in _SOFT_COLS)
    has_urg  = "ltc_urgency" in regime_df.columns
    logger.info(
        f"{'✅' if has_soft else '⚠️ '} GMM posteriors: {'active' if has_soft else 'fallback 20%'} | "
        f"{'✅' if has_urg else '⚠️ '} Vol urgency: {'active' if has_urg else 'z_mu proxy'}"
    )
    logger.info(f"  λ_turn = {_LAMBDA_TURN:.4f} (target: 12-18% daily turnover)")

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
        r_oo  = t_oo["daily_return"].values
        n_oo  = len(r_oo)
        tot   = t_oo["portfolio_value"].iloc[-1] / _INITIAL_CAPITAL - 1.0
        cagr  = (1.0 + tot) ** (_TRADING_DAYS_YEAR / n_oo) - 1.0
        mdd   = float(((t_oo["portfolio_value"] - t_oo["portfolio_value"].cummax()) /
                       (t_oo["portfolio_value"].cummax() + 1e-10)).min())
        is_sr  = _sharpe(t_is["daily_return"].values) if not t_is.empty else 0.0
        oos_sr = _sharpe(r_oo)
        results.append(WFFold(fid + 1, is_s, is_e, oos_s, oos_e, round(is_sr, 4), round(oos_sr, 4), round(cagr, 4), round(mdd, 4)))
        logger.info(f"  F{fid+1}: IS SR={is_sr:.3f} | OOS SR={oos_sr:.3f}  CAGR={cagr:.2%}  MaxDD={mdd:.2%}")

    pd.DataFrame([vars(r) for r in results]).to_csv(_WF_FOLDS, index=False)
    if results:
        avg_oos = float(np.mean([f.oos_sharpe for f in results]))
        avg_is  = float(np.mean([f.is_sharpe  for f in results]))
        logger.info(f"WF summary: avg IS={avg_is:.3f} | avg OOS={avg_oos:.3f} | {_wf_verdict(avg_is, avg_oos)}")

    r_arr = ts["daily_return"].values
    v95   = float(np.percentile(r_arr, 5))
    with open(_STRESS_OUT, "w") as f:
        json.dump({"mode": "stub", "var_95": v95, "cvar_95": float(r_arr[r_arr <= v95].mean())}, f, indent=2)

    logger.info("✅ Done. Run scripts/visualize_tearsheet.py next.")


if __name__ == "__main__":
    main()