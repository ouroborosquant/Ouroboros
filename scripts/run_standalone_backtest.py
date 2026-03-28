"""
FORTRESS v5 — scripts/run_standalone_backtest.py  [v11.0 — ALPHA UNLOCKED]

v11.0 FIXES (addresses the "Alpha Paralysis" diagnostic):

  BUG A FIX — IC COMPUTATION OVERHAUL:
    1. Spearman → Pearson IC on equity domain. Spearman rank correlation is
       broken on sparse, zero-inflated alpha arrays (CONC signal). Spearman
       assigns ranks to zero-alpha assets, creating noise that systematically
       biases IC negative when 8/14 equity assets have near-zero alpha.
       Pearson naturally handles sparsity: zero-alpha assets contribute zero
       covariance. The IC now reflects whether concentrated bets are paying off.

    2. _ROLLING_IC_WIN 21 → 42 days. CONC operates at 63d frequency. Measuring
       IC at 21d catches 1/3 of the holding period → noise. At 42d we capture
       2/3 and the IC reflects the signal's intended horizon.

    3. _IC_HALT_MIN_HOLD = 10 days (NEW). Prevents the costly halt/resume/halt
       churn cycle. Once halted, the system stays halted for at least 10 trading
       days before checking resume conditions. Each false resume costs ~20-40 bps
       in liquidation/re-entry friction.

    4. Thresholds recalibrated for Pearson scale:
       _IC_HALT_THRESHOLD: -0.025 → -0.020 (Pearson is less noisy on sparse)
       _IC_RESUME_THRESHOLD: 0.030 → 0.025

  BUG B FIX — CONVICTION-ADJUSTED λ_var:
    5. When the alpha vector has high cross-sectional spread (CONC is active,
       strongly favoring QQQ/XLK over XLU/XLP), the optimizer gets a λ_var
       reduction of up to 25%. This is the mechanism that lets the optimizer
       FOLLOW the CONC signal into high-vol names instead of mathematically
       rejecting them because Σ(QQQ) > Σ(XLP).

       conviction_adj = 1.0 - 0.25 * clip((alpha_spread - 0.05) / 0.12, 0, 1)
       At spread=0.05: adj=1.0 (no change)
       At spread=0.12: adj=0.85 (15% λ_var reduction)
       At spread=0.17: adj=0.75 (25% λ_var reduction — max)

  COST DRAG FIXES:
    6. _LAMBDA_TURN: 0.15 → 0.25. The CONC signal at 63d frequency + MOM at
       252d creates conflicting turnover impulses. Higher λ_turn dampens the
       day-to-day noise without preventing regime-driven rotations.

    7. _LAMBDA_TURN_LOW: 0.05 → 0.08. Even in high-conviction, 5% turnover
       penalty was too loose.

    8. _REBALANCE_BAND: 50bps → 80bps. Wider dead zone prevents micro-trades
       that consume bps without improving allocation.

    9. Alpha EMA smoothing (5-day span). Applied to the alpha vector BEFORE
       the MVO, dampening daily rank noise from CONC's 63d return lookback.
       A 63d signal shouldn't change the portfolio daily — the EMA enforces
       this without losing the directional content.

    10. Crisis alpha scaling cap: 0.50 → 0.35. The old 50% reduction during
        moderate stress killed the CONC signal when the market was merely
        choppy (not crashing). 35% preserves 65% of signal in stress.
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
import scipy.stats as stats
from scipy.stats import norm
import cvxpy as cp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("Ouroboros.StandaloneBacktest")

_CACHE_DIR   = Path("research/outputs/cache")
_OUTPUT_DIR  = Path("research/outputs")
_TEARSHEET   = _OUTPUT_DIR / "backtest_tearsheet.csv"
_WF_FOLDS    = _OUTPUT_DIR / "walk_forward_folds.csv"
_STRESS_OUT  = _OUTPUT_DIR / "sde_stress_test.json"
_GEX_CACHE   = _CACHE_DIR  / "gex_alpha.parquet"

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

_INITIAL_CAPITAL:       float = 100_000.0
_RISK_FREE_ANNUAL:      float = 0.05
_HALT_RECOVERY_THRESH:  float = 0.10
_HALT_MIN_DAYS:         int   = 21
_HALT_RAMP_DAYS:        int   = 21
_RAMP_DD_BUFFER:        float = 0.08
_WARMUP_DAYS:           int   = 126
_WF_WARMUP_DAYS:        int   = 21
_REBALANCE_BAND:        float = 80e-4         # v11: was 50e-4

_LAMBDA_BASE:           float = 2.5
_LAMBDA_CVAR_BASE:      float = 0.05
_KURT_CVAR_SCALE:       float = 0.05

_LAMBDA_TURN:           float = 0.25           # v11: was 0.15
_LAMBDA_TURN_LOW:       float = 0.08           # v11: was 0.05
_SIGNAL_STRENGTH_FLOOR: float = 0.05
_SIGNAL_STRENGTH_CEIL:  float = 0.15
_SIGNAL_STRENGTH_WIN:   int   = 21

_ROLLING_IC_WIN:         int   = 42            # v11: was 21
_IC_NEGATIVE_THRESHOLD:  float = -0.01

_IC_HALT_THRESHOLD:      float = -0.020        # v11: was -0.025
_IC_HALT_MIN_DAYS:       int   = 15
_IC_RESUME_THRESHOLD:    float = 0.025         # v11: was 0.030
_IC_HALT_MIN_HOLD:       int   = 10            # v11: NEW

_COV_WINDOW:            int   = 126
_MIN_POSITION_WT:       float = 0.015
_EMA_SPAN:              int   = 8
_REGIME_PERSIST_DAYS:   int   = 5
_AC_ETA:                float = 0.1
_BASE_SPREAD_BPS:       float = 1.0
_TRADING_DAYS_YEAR:     int   = 252
_MAX_DD_HALT_FALLBACK:  float = 0.20
_ALPHA_EMA_SPAN:        int   = 5              # v11: NEW — smooths alpha before MVO
_CRISIS_ALPHA_CAP:      float = 0.35           # v11: was 0.50 — max alpha reduction in stress

_TECH_TICKERS:     List[str] = ["QQQ", "XLK", "XLC"]
_EQUITY_TICKERS:   List[str] = [
    "SPY", "QQQ", "IWM", "XLK", "XLF", "XLV", "XLU", "XLI", "XLP",
    "XLY", "XLB", "XLC", "COWZ", "XLE",
]
_MAX_TECH_WEIGHT:   float = 0.30
_MAX_EQUITY_WEIGHT: float = 0.60

_TECH_IDX   = [TICKERS.index(t) for t in _TECH_TICKERS   if t in TICKERS]
_EQUITY_IDX = [TICKERS.index(t) for t in _EQUITY_TICKERS if t in TICKERS]

_SOFT_COLS = ["soft_crisis", "soft_bear", "soft_bull", "soft_low_vol"]


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


def _cov(window: pd.DataFrame) -> np.ndarray:
    from sklearn.covariance import OAS
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
    from dateutil.relativedelta import relativedelta
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
    if avg_is < 0 and avg_oos > 0:
        return "❌ IS dominated by structural event; OOS signal positive"
    return "❌ Systematic OOS failure"


def _adaptive_lambda_turn(signal_strength: float, rolling_ic: float = 0.0) -> float:
    if np.isnan(signal_strength): signal_strength = _SIGNAL_STRENGTH_CEIL
    if np.isnan(rolling_ic): rolling_ic = 0.0
    if rolling_ic < 0.0:
        return _LAMBDA_TURN * 2.0
    if signal_strength <= _SIGNAL_STRENGTH_FLOOR:
        return _LAMBDA_TURN * 1.5
    if signal_strength >= _SIGNAL_STRENGTH_CEIL:
        return _LAMBDA_TURN_LOW
    t = (signal_strength - _SIGNAL_STRENGTH_FLOOR) / (
        _SIGNAL_STRENGTH_CEIL - _SIGNAL_STRENGTH_FLOOR
    )
    return (_LAMBDA_TURN * 1.5) - t * ((_LAMBDA_TURN * 1.5) - _LAMBDA_TURN_LOW)


def _gex_equity_cap(
    gex_zscore: float, base_cap: float, equity_urgency: float = 0.0,
) -> float:
    if np.isnan(gex_zscore):     gex_zscore = 0.0
    if np.isnan(equity_urgency): equity_urgency = 0.0
    if gex_zscore >= 0.0:
        delta = 0.25 * float(np.tanh(gex_zscore * 0.65))
        if equity_urgency > 0.40:
            dampener = 1.0 - 0.60 * float(np.clip(
                (equity_urgency - 0.40) / 0.30, 0.0, 1.0
            ))
            delta *= dampener
    else:
        delta = -0.18 * float(np.tanh(-gex_zscore * 0.50))
    return float(np.clip(base_cap + delta, 0.25, 0.85))


def _gex_risk_aversion_adj(gex_zscore: float) -> float:
    if np.isnan(gex_zscore):
        return 1.0
    adj = 1.0 - 0.30 * float(np.tanh(gex_zscore * 0.55))
    return float(np.clip(adj, 0.70, 1.30))


def _mvo_weights(
    alpha:           np.ndarray,
    cov_d:           np.ndarray,
    z0_smooth:       float,
    vol_d:           np.ndarray,
    prev_weights:    np.ndarray,
    equity_urgency:  float = 0.0,
    return_window:   Optional[np.ndarray] = None,
    signal_strength: float = 0.10,
    rolling_ic:      float = 0.0,
    gex_alpha:       Optional[np.ndarray] = None,
) -> np.ndarray:
    """CVaR-MVO v11.0 — conviction-adjusted λ_var + GEX decoupled."""

    z0_smooth = 0.0 if np.isnan(z0_smooth) else z0_smooth
    avg_vol = float(np.nanmean(vol_d) * np.sqrt(_TRADING_DAYS_YEAR))
    if np.isnan(avg_vol): avg_vol = 0.15

    lam_var = float(np.clip(_LAMBDA_BASE * np.exp(avg_vol * abs(z0_smooth)), 0.5, 15.0))
    if np.isnan(lam_var): lam_var = _LAMBDA_BASE

    # GEX → λ_var modulation (v10.0)
    if gex_alpha is not None and len(gex_alpha) == N_ASSETS:
        equity_gex_vals = gex_alpha[_EQUITY_IDX]
        gex_zscore = float(np.nanmean(equity_gex_vals))
    else:
        gex_zscore = 0.0
    lam_var *= _gex_risk_aversion_adj(gex_zscore)

    # v11.0: Conviction-adjusted λ_var — when alpha has high cross-sectional
    # spread (CONC is active, strongly favoring leaders), reduce λ_var to let
    # the optimizer follow the signal into high-vol names.
    eq_alpha_spread = float(np.std(alpha[_EQUITY_IDX]))
    conviction_adj = 1.0 - 0.25 * float(np.clip(
        (eq_alpha_spread - 0.05) / 0.12, 0.0, 1.0
    ))
    lam_var *= conviction_adj

    Σ  = cov_d * _TRADING_DAYS_YEAR
    mx = np.array([TIER_MAX_WEIGHT.get(t, 0.15) for t in TICKERS])

    w_prev = np.clip(prev_weights, 0.0, 1.0)
    w_prev = np.where(w_prev < _MIN_POSITION_WT, 0.0, w_prev)
    s_prev = w_prev.sum()
    if s_prev > 1e-10:
        w_prev /= s_prev

    base_lam_turn = _adaptive_lambda_turn(signal_strength, rolling_ic)
    if np.isnan(base_lam_turn): base_lam_turn = _LAMBDA_TURN

    if np.isnan(equity_urgency): equity_urgency = 0.0
    if equity_urgency > 0.7:
        lam_turn = 0.0
    elif equity_urgency > 0.4:
        lam_turn = base_lam_turn * 0.4
    else:
        lam_turn = base_lam_turn

    eigenvalues, eigenvectors = np.linalg.eigh(Σ)
    eigenvalues = np.maximum(eigenvalues, 1e-8)
    Σ_psd = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    tech_alpha = np.mean(alpha[_TECH_IDX])
    univ_alpha = np.mean(alpha)
    dynamic_tech_cap = _MAX_TECH_WEIGHT
    if tech_alpha > univ_alpha * 2.0 and tech_alpha > 0.02:
        dynamic_tech_cap = 0.45

    dynamic_eq_cap = _gex_equity_cap(gex_zscore, _MAX_EQUITY_WEIGHT, equity_urgency)

    use_cvar = (
        return_window is not None and
        return_window.shape[0] >= 30 and
        return_window.shape[1] == N_ASSETS
    )

    lam_cvar = 0.0
    if use_cvar:
        drift = float(np.abs(w_prev - w_prev.mean()).max())
        if w_prev.sum() > 0.5 and drift < 0.25:
            x0 = np.clip(w_prev, 0.0, mx)
        else:
            x0 = np.clip(np.full(N_ASSETS, 1.0 / N_ASSETS), 0.0, mx)
        x0 /= (x0.sum() + 1e-10)
        ew_rets = return_window @ x0
        with np.errstate(invalid="ignore", divide="ignore"):
            excess_kurt = float(stats.kurtosis(ew_rets, fisher=True))
        if np.isnan(excess_kurt):
            excess_kurt = 0.0
        lam_cvar = float(np.clip(
            _LAMBDA_CVAR_BASE * _KURT_CVAR_SCALE * max(excess_kurt, 0.0),
            0.0, 1.0,
        ))
        if np.isnan(lam_cvar): lam_cvar = 0.0

    w = cp.Variable(N_ASSETS)
    constraints = [
        cp.sum(w) == 1.0,
        w >= 0.0,
        w <= mx,
        cp.sum(w[_TECH_IDX])   <= dynamic_tech_cap,
        cp.sum(w[_EQUITY_IDX]) <= dynamic_eq_cap,
    ]

    safe_alpha = np.nan_to_num(alpha, nan=0.0)
    expected_return    = safe_alpha.T @ w
    portfolio_variance = cp.quad_form(w, cp.psd_wrap(Σ_psd))
    turnover_penalty   = cp.sum_squares(w - w_prev)

    if use_cvar and lam_cvar > 0.01:
        T_obs      = return_window.shape[0]
        alpha_conf = 0.95
        inv_tail   = 1.0 / max(int(np.floor((1 - alpha_conf) * T_obs)), 1)
        zeta = cp.Variable()
        z    = cp.Variable(T_obs)
        safe_returns = np.nan_to_num(return_window, nan=0.0)
        constraints.extend([z >= 0, z >= -safe_returns @ w - zeta])
        cvar_loss = zeta + inv_tail * cp.sum(z)
        objective = cp.Maximize(
            expected_return
            - (0.5 * lam_var * portfolio_variance)
            - (lam_cvar * cvar_loss)
            - (lam_turn * turnover_penalty)
        )
    else:
        objective = cp.Maximize(
            expected_return
            - (0.5 * lam_var * portfolio_variance)
            - (lam_turn * turnover_penalty)
        )

    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.ECOS, warm_start=True)
        if prob.status in ["optimal", "optimal_inaccurate"] and w.value is not None:
            w_val = w.value
        else:
            raise cp.SolverError(f"Status: {prob.status}")
    except Exception as e:
        logger.debug(f"CVXPY fallback: {e}")
        inv_vol = 1.0 / (np.diag(Σ_psd) ** 0.5 + 1e-8)
        inv_vol = np.minimum(inv_vol, mx * 3)
        w_val   = inv_vol / (inv_vol.sum() + 1e-10)

    w_val = np.clip(w_val, 0.0, mx)
    w_val = np.where(w_val < _MIN_POSITION_WT, 0.0, w_val)
    s = w_val.sum()
    if s > 1e-10:
        w_val /= s
    else:
        w_val = np.where(mx > 0, 1.0 / max((mx > 0).sum(), 1), 0.0)
    return w_val.astype(np.float32)


@dataclass
class Snap:
    date: str; portfolio_value: float; daily_return: float
    cash: float; turnover: float; cost_drag: float
    regime_label: str; z_mu_0: float; drawdown: float; alloc_pct: int

@dataclass
class WFFold:
    fold_id: int; is_start: str; is_end: str
    oos_start: str; oos_end: str
    is_sharpe: float; oos_sharpe: float
    oos_cagr: float; oos_max_dd: float


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
        self._ic_below_days:  int  = 0
        self._ic_halted:      bool = False
        self._ic_halt_day:    int  = 0    # v11: tracks days since halt started
        self._alpha_ema: Optional[np.ndarray] = None  # v11: alpha smoothing

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
        equity  = sum(self.pos.get(t, 0.0) * px[i] for i, t in enumerate(TICKERS))
        new_nav = self.cash + equity
        dr      = (new_nav - self.nav) / (self.nav + 1e-10)
        self.nav = new_nav
        self._peak = max(self._peak, new_nav)
        return dr

    def _rebalance(self, target, px, vol, adv, force=False):
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
            self.cash   += -dol - c
            self.pos[t]  = self.pos.get(t, 0.0) + shares
        self._prev_weights = target.copy().astype(np.float64)
        return cost / (self.nav + 1e-10), float(0.5 * np.sum(np.abs(target - current)))

    def run(self, prices_df, returns_df, regime_df, alpha_df, gex_df,
            start_date, end_date, warmup_days=_WARMUP_DAYS) -> pd.DataFrame:
        self._reset()
        common = (
            prices_df.index.intersection(returns_df.index)
            .intersection(regime_df.index).intersection(alpha_df.index)
        )
        mask = (common >= start_date) & (common <= end_date)
        sim_dates = common[mask]
        if len(sim_dates) == 0:
            return pd.DataFrame()

        all_dates = common[common <= end_date]
        warmup_start_idx = max(0, all_dates.get_loc(sim_dates[0]) - warmup_days)
        full_dates = all_dates[warmup_start_idx:]
        record_start = sim_dates[0]

        alpha_abs_mean = (
            alpha_df.reindex(full_dates).abs().mean(axis=1)
            .rolling(_SIGNAL_STRENGTH_WIN, min_periods=5).mean()
            .fillna(_SIGNAL_STRENGTH_CEIL)
        )

        _alpha_arr = alpha_df.reindex(full_dates).reindex(columns=TICKERS).fillna(0.0)
        _ret_arr   = returns_df.reindex(full_dates).reindex(columns=TICKERS).fillna(0.0)
        _ic_series = pd.Series(np.nan, index=full_dates)

        equity_mask = np.array([t in set(_EQUITY_TICKERS) for t in TICKERS])

        # v11: Pearson IC on equity domain — handles sparse CONC signals correctly
        for _i in range(1, len(full_dates)):
            _alpha_t  = _alpha_arr.iloc[_i - 1].values
            _return_t = _ret_arr.iloc[_i].values

            eq_alpha  = _alpha_t[equity_mask]
            eq_return = _return_t[equity_mask]

            alpha_std = float(np.std(eq_alpha))
            if alpha_std > 0.005 and len(eq_alpha) >= 5:
                from scipy.stats import pearsonr as _pearsonr
                _ic, _ = _pearsonr(eq_alpha, eq_return)
                if np.isfinite(_ic):
                    _ic_series.iloc[_i] = _ic

        rolling_ic_series = (
            _ic_series
            .rolling(_ROLLING_IC_WIN, min_periods=10)
            .mean()
            .fillna(0.0)
        )

        # v11: alpha EMA decay constant
        _alpha_decay = 2.0 / (_ALPHA_EMA_SPAN + 1)

        for date in full_dates:
            record = date >= record_start
            ds     = str(date.date())
            gi     = prices_df.index.get_loc(date)
            px     = prices_df.iloc[gi].reindex(TICKERS).fillna(1.0).values.astype(np.float64)

            sig_strength = float(alpha_abs_mean.get(date, _SIGNAL_STRENGTH_CEIL))
            roll_ic      = float(rolling_ic_series.get(date, 0.0))

            if date not in regime_df.index or date not in alpha_df.index:
                if record:
                    self.history.append(Snap(ds, self.nav, 0.0, self.cash, 0.0, 0.0, "NO_SIGNAL", 0.0, 0.0, 0))
                continue

            raw_z = regime_df.loc[date, "z_mu"]
            if isinstance(raw_z, str):
                import ast; raw_z = ast.literal_eval(raw_z)
            z_raw   = np.asarray(raw_z, dtype=np.float32).ravel()
            z_smooth, s_label = self._smoother.update(z_raw, str(regime_df.loc[date, "regime_label"]))
            z0      = float(z_smooth[0])

            regime_row     = regime_df.loc[date]
            equity_urgency = float(regime_row.get("ltc_urgency", abs(z0) / 4.0))
            if np.isnan(equity_urgency): equity_urgency = 0.0

            gex_alp = (
                gex_df.loc[date].reindex(TICKERS).fillna(0.0).values.astype(np.float64)
                if date in gex_df.index else np.zeros(N_ASSETS, dtype=np.float64)
            )

            halt_dd_thresh, halt_days_req, ramp_days_req = _get_regime_halt_threshold(regime_df, date)
            eff_cov_window = max(int(_COV_WINDOW * (1.0 - 0.5 * equity_urgency)), 42)

            if self._halt_phase >= 1:
                self._halt_days += 1
                dd_from_peak = (self.nav - self._peak) / self._peak if self._peak > 0 else 0.0
                if self._halt_phase == 1:
                    dr = self._mtm(px)
                    if self._halt_days >= halt_days_req:
                        recov = (self.nav - self._halt_nav) / (self._halt_nav + 1e-10)
                        if recov >= -_HALT_RECOVERY_THRESH:
                            logger.info(f"{ds}: Halt recovery — entering ramp-in.")
                            self._halt_phase = 2; self._ramp_days = 0; self._ramp_entry_nav = self.nav
                elif self._halt_phase == 2:
                    self._ramp_days += 1
                    scale = min(float(self._ramp_days) / ramp_days_req, 1.0)
                    cov_d = _cov(returns_df.iloc[max(0, gi - eff_cov_window):gi])
                    vol_d = returns_df.iloc[max(0, gi - 21):gi].reindex(columns=TICKERS).std(axis=0).fillna(0.01).values.astype(np.float64)
                    adv   = np.maximum(10_000_000.0 / (px + 1e-10), 1.0)
                    alp   = alpha_df.loc[date].reindex(TICKERS).fillna(0.0).values.astype(np.float64)
                    ret_window = returns_df.iloc[max(0, gi - _COV_WINDOW):gi].reindex(columns=TICKERS).fillna(0.0).values
                    target = _mvo_weights(
                        alp, cov_d, z0, vol_d, self._prev_weights, equity_urgency,
                        ret_window if ret_window.shape[0] >= 30 else None,
                        signal_strength=sig_strength, rolling_ic=roll_ic, gex_alpha=gex_alp,
                    )
                    target_scaled = target * scale
                    target_scaled[TICKERS.index("BIL")] += (1.0 - scale) * 0.60
                    target_scaled[TICKERS.index("SHV")] += (1.0 - scale) * 0.40
                    cost_d, to = self._rebalance(target_scaled, px, vol_d, adv)
                    dr = self._mtm(px)
                    dd_ramp = (self.nav - self._ramp_entry_nav) / (self._ramp_entry_nav + 1e-10)
                    if dd_ramp <= -_RAMP_DD_BUFFER:
                        self._halt_phase = 1; self._halt_days = 0; self._halt_nav = self.nav
                        self._liquidate(px); self._invest_bil(px)
                    elif self._ramp_days >= ramp_days_req:
                        logger.info(f"{ds}: Ramp-in complete.")
                        self._halt_phase = 0; self._peak = self.nav
                    if record:
                        self.history.append(Snap(ds, self.nav, dr, self.cash, to, cost_d, f"RAMP_{scale:.0%}", z0, dd_from_peak, 2))
                    continue
                if record:
                    self.history.append(Snap(ds, self.nav, dr, self.cash, 0.0, 0.0, "HALTED_BIL", z0, dd_from_peak, self._halt_phase))
                continue

            cov_d = _cov(returns_df.iloc[max(0, gi - eff_cov_window):gi])
            vol_d = returns_df.iloc[max(0, gi - 21):gi].reindex(columns=TICKERS).std(axis=0).fillna(0.01).values.astype(np.float64)
            adv   = np.maximum(10_000_000.0 / (px + 1e-10), 1.0)

            if roll_ic < _IC_HALT_THRESHOLD:
                self._ic_below_days += 1
            else:
                self._ic_below_days = 0

            if not self._ic_halted and self._ic_below_days >= _IC_HALT_MIN_DAYS:
                self._ic_halted = True
                self._ic_halt_day = 0  # v11: track halt duration
                logger.info(
                    f"{ds}: IC-DECAY HALT — rolling_IC={roll_ic:+.4f} below "
                    f"{_IC_HALT_THRESHOLD} for {self._ic_below_days}d. Moving to BIL."
                )
                self._liquidate(px)
                self._invest_bil(px)

            if self._ic_halted:
                self._ic_halt_day += 1  # v11: increment halt counter

                # v11: Minimum hold period — don't check resume until _IC_HALT_MIN_HOLD
                can_resume = (
                    self._ic_halt_day >= _IC_HALT_MIN_HOLD
                    and roll_ic >= _IC_RESUME_THRESHOLD
                )
                if can_resume:
                    self._ic_halted = False
                    self._ic_below_days = 0
                    self._ic_halt_day = 0
                    logger.info(
                        f"{ds}: IC-DECAY RESUME — rolling_IC={roll_ic:+.4f} ≥ "
                        f"{_IC_RESUME_THRESHOLD:.3f}. Resuming active trading."
                    )
                else:
                    dr = self._mtm(px)
                    if record:
                        self.history.append(Snap(ds, self.nav, dr, self.cash, 0.0, 0.0, "IC_HALTED_BIL", z0, (self.nav - self._peak) / self._peak, 0))
                    continue

            has_soft      = all(c in regime_df.columns for c in _SOFT_COLS)
            crisis_weight = (
                float(regime_row.get("soft_crisis", 0.0)) + float(regime_row.get("soft_bear", 0.0))
                if has_soft else float(np.clip(equity_urgency, 0.0, 1.0))
            )
            # v11: Reduced crisis alpha cap (0.50 → 0.35) — old value killed CONC in choppy markets
            alpha_scale = 1.0 - _CRISIS_ALPHA_CAP * crisis_weight
            alp = alpha_df.loc[date].reindex(TICKERS).fillna(0.0).values.astype(np.float64) * alpha_scale

            # v11: Alpha EMA smoothing — dampens day-to-day noise from CONC's 63d lookback
            if self._alpha_ema is None:
                self._alpha_ema = alp.copy()
            self._alpha_ema = _alpha_decay * alp + (1.0 - _alpha_decay) * self._alpha_ema
            alp_smooth = self._alpha_ema.copy()

            ret_window = returns_df.iloc[max(0, gi - _COV_WINDOW):gi].reindex(columns=TICKERS).fillna(0.0).values
            target = _mvo_weights(
                alp_smooth, cov_d, z0, vol_d, self._prev_weights, equity_urgency,
                ret_window if ret_window.shape[0] >= 30 else None,
                signal_strength=sig_strength, rolling_ic=roll_ic, gex_alpha=gex_alp,
            )

            dd      = (self.nav - self._peak) / self._peak if self._peak > 0 else 0.0
            dd_vel  = dd - self._prev_dd
            dd_accel = dd_vel - self._prev_dd_vel
            self._prev_dd = dd; self._prev_dd_vel = dd_vel

            velocity_adj = float(np.clip(dd_accel / 0.001, -0.05, 0.05))
            _RISK_TIERS  = [
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
        active_pct = float((~df["regime_label"].isin(["WARMUP", "NO_SIGNAL", "HALTED_BIL", "IC_HALTED_BIL"]) &
                            ~df["regime_label"].str.startswith("RAMP")).mean() * 100)
        halt_pct    = float((df["regime_label"] == "HALTED_BIL").mean() * 100)
        ic_halt_pct = float((df["regime_label"] == "IC_HALTED_BIL").mean() * 100)

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
            ("IC Decay Halt %",       f"{ic_halt_pct:.1f}%"),
            ("Final NAV",             f"${nav[-1]:,.2f}"),
            ("Trading Days",          f"{n}"),
        ]:
            logger.info(f"  {label:<30s} {val}")
        return df


def main() -> None:
    logger.info("══════ Fortress v5 — Standalone Backtest v11.0 (ALPHA UNLOCKED) ══════")

    prices_df  = _normalize_index(pd.read_parquet(_CACHE_DIR / "prices_wide.parquet"))
    returns_df = _normalize_index(pd.read_parquet(_CACHE_DIR / "returns_wide.parquet"))
    regime_df  = _normalize_index(pd.read_parquet(_CACHE_DIR / "regime_posteriors.parquet"))
    alpha_df   = _normalize_index(pd.read_parquet(_CACHE_DIR / "alpha_signals_blended.parquet"))

    if not _GEX_CACHE.exists():
        logger.error(f"GEX alpha cache missing: {_GEX_CACHE}")
        sys.exit(1)
    gex_df = _normalize_index(pd.read_parquet(_GEX_CACHE))

    missing = set(TICKERS) - set(alpha_df.columns)
    if missing:
        logger.error(f"Alpha signals missing for: {missing}")
        sys.exit(1)

    prices_df  = prices_df.reindex(columns=TICKERS)
    returns_df = returns_df.reindex(columns=TICKERS)
    alpha_df   = alpha_df.reindex(columns=TICKERS)
    gex_df     = gex_df.reindex(columns=TICKERS).fillna(0.0)

    logger.info(
        f"Loaded | prices:{len(prices_df)}d  returns:{len(returns_df)}d  "
        f"regime:{len(regime_df)}d  alpha:{len(alpha_df)}d  gex:{len(gex_df)}d"
    )

    def _run(start, end, warmup):
        return StandaloneBacktester().run(
            prices_df, returns_df, regime_df, alpha_df, gex_df,
            start_date=start, end_date=end, warmup_days=warmup,
        )

    ts = _run("2020-01-02", "2024-12-31", _WARMUP_DAYS)
    if ts.empty:
        logger.error("Tearsheet empty.")
        sys.exit(1)

    ts.to_csv(_TEARSHEET)
    logger.info(f"✅ Tearsheet → {_TEARSHEET} ({len(ts)} rows)")

    logger.info("Running walk-forward validation...")
    common  = prices_df.index.intersection(alpha_df.index)
    wf_mask = (common >= "2019-01-02") & (common <= "2024-12-31")
    folds   = _wf_folds(common[wf_mask])
    results: List[WFFold] = []

    for fid, (is_s, is_e, oos_s, oos_e) in enumerate(folds):
        t_is = _run(is_s, is_e, _WF_WARMUP_DAYS)
        t_oo = _run(oos_s, oos_e, _WF_WARMUP_DAYS)
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
        results.append(WFFold(fid+1, is_s, is_e, oos_s, oos_e, round(is_sr, 4), round(oos_sr, 4), round(cagr, 4), round(mdd, 4)))
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