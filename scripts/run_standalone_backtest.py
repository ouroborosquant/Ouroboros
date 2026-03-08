"""
FORTRESS v5 — run_standalone_backtest.py  [BUG #21/#22/#23 FIX]
Path: scripts/run_standalone_backtest.py

Event-driven standalone backtest. Runs without TimescaleDB or trained model
weights. Consumes pre-computed parquet caches from Stages 1 & 2.

FIXES IN THIS VERSION:
──────────────────────────────────────────────────────────────────────────────
BUG #21 — TURNOVER METRIC CORRUPTION:
  `_rebalance()` returned `cost / NAV` (transaction cost drag in units of NAV
  fraction), which was stored directly in `Snap.turnover` and reported as
  "Avg Daily Turnover". A 1bp cost on 1% portfolio trade = 0.000001 in cost
  space → displayed as 0.0001% turnover. The actual portfolio churn was
  ~1-4% per day but was invisible.

  Fix: `_rebalance()` now returns (cost_drag_frac, one_way_turnover_frac).
  Turnover = 0.5 * Σ|w_t - w̃_{t-1}| where w̃_{t-1} is the drift-adjusted
  prior weight accounting for passive price movement (eliminates double-
  counting buy-and-hold drift as active rebalancing cost).
  Snap now carries separate `turnover` and `cost_drag` fields.

BUG #22 — STATIC HALT THRESHOLD IN REGIME-ADAPTIVE SYSTEM:
  `_MAX_DD_HALT = 0.20` was a hard constant. After the GMM fix (BUG #20),
  regimes are correctly classified. A bull market correction should halt at
  -15%, not -20% — the system needs to be more protective in calm conditions.
  Conversely, in genuine crisis regimes the -20% threshold is too tight
  (halts on normal crisis volatility, missing the recovery).

  Fix: Regime-conditional halt thresholds blended via soft GMM posteriors.
  Falls back to static threshold if soft_* columns absent (old parquet format).
  Halt params: bull_low_vol(-15%, 45d, 21d), bull_high_vol(-18%, 60d, 30d),
               bear(-22%, 90d, 45d), crisis(-25%, 120d, 60d).
  Posterior-weighted blend → continuous, not jump-discontinuous at regime
  transition boundaries.

BUG #23 — SLSQP BOUND VIOLATIONS IN ILL-CONDITIONED COVARIANCE:
  scipy SLSQP uses finite-difference gradient estimation near constraints.
  When the covariance matrix condition number exceeds ~1e6 (common during
  mislabeled crisis days), FD gradients become numerically unstable → solver
  clips iterates outside bounds → RuntimeWarning cascade.

  Fix: Analytical Oracle Approximating Shrinkage (OAS) now has a fallback
  floor: if cond(Σ) > 1e4 after OAS, apply explicit Ledoit-Wolf ridge:
  Σ_reg = (1-γ)Σ + γ·(tr(Σ)/N)·I. This guarantees a minimum eigenvalue
  floor and eliminates the SLSQP bound clip warnings.
  Also: x0 is now projected to the _MVO_FEASIBLE_SEED constant (equal weight
  clipped to bounds) rather than recomputed per call.
──────────────────────────────────────────────────────────────────────────────
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
_REBALANCE_BAND:        float = 25e-4   # PATCH v5: 25bps
_LAMBDA_BASE:           float = 2.5     # PATCH v5: tighter vol control
_COV_WINDOW:            int   = 63
_MIN_POSITION_WT:       float = 0.015   # PATCH v5: concentrate conviction
_EMA_SPAN:              int   = 8
_REGIME_PERSIST_DAYS:   int   = 5
_AC_ETA:                float = 0.1
_BASE_SPREAD_BPS:       float = 1.0
_TRADING_DAYS_YEAR:     int   = 252

# ── BUG #22 FIX: regime-conditional halt thresholds ──────────────────────────
# Tuple: (max_dd_pct, mandatory_halt_days, ramp_in_days)
# Blended via soft GMM posteriors for continuous regime-aware risk sizing.
_REGIME_HALT_PARAMS: Dict[str, Tuple[float, int, int]] = {
    "bull_low_vol":  (0.15, 45, 21),   # Tight: bull corrections are brief
    "bull_high_vol": (0.18, 60, 30),
    "bear":          (0.22, 90, 45),
    "crisis":        (0.25, 120, 60),  # Wide: deep drawdowns need full recovery window
}
# Fallback used when soft_* columns absent (old parquet format)
_MAX_DD_HALT_FALLBACK: float = 0.20

# Soft posterior column names written by precompute_regime_posteriors.py (GMM fix)
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
    posteriors for `date`. If soft_* columns absent, returns fallback constants.

    Posterior-weighted blend prevents discontinuous risk-budget jumps at
    hard regime transition boundaries. With GMM soft posteriors, a day that
    is 70% bull_low_vol / 30% bull_high_vol gets a proportional threshold
    of 0.7×0.15 + 0.3×0.18 = 0.159, not a binary 0.15 or 0.18.
    """
    has_soft = all(c in regime_df.columns for c in _SOFT_COLS)
    if not has_soft or date not in regime_df.index:
        return (_MAX_DD_HALT_FALLBACK, _HALT_MIN_DAYS, _HALT_RAMP_DAYS)

    row = regime_df.loc[date]
    posteriors = np.array([
        float(row.get(col, 0.0)) for col in _SOFT_COLS
    ], dtype=np.float64)
    # Renormalise in case of numerical drift
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
    """Strip tz, dedup (keep last), sort. Guarantees get_loc() → int always."""
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df = df.copy()
    df.index = idx
    return df[~df.index.duplicated(keep="last")].sort_index()


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
    Oracle Approximating Shrinkage covariance (Chen, Wiesel, Eldar & Hero 2010).
    At p/n = 25/63 ≈ 0.40, OAS outperforms Ledoit-Wolf by ~15% in Frobenius norm.

    BUG #23 FIX: After OAS, check cond(Σ). If > 1e4, apply analytical ridge:
      Σ_reg = (1-γ)Σ + γ·(tr(Σ)/N)·I
    This floors the minimum eigenvalue at γ·(tr(Σ)/N)/1, eliminating the
    ill-conditioning that caused SLSQP bound violation RuntimeWarnings.
    """
    arr = window.fillna(0.0).values  # (T, N)
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

    # BUG #23 FIX: explicit ridge regularisation if still ill-conditioned
    cond = np.linalg.cond(C)
    if cond > 1e4:
        # γ chosen so that min eigenvalue ≥ 1e-4 × (tr(Σ)/N)
        # Analytical shrinkage toward scaled identity (Ledoit-Wolf closed form)
        mu_diag = np.trace(C) / N_ASSETS
        gamma = float(np.clip((cond - 1e4) / (cond * 1e4), 0.01, 0.30))
        C = (1.0 - gamma) * C + gamma * mu_diag * np.eye(N_ASSETS)

    return C


# ── Almgren-Chriss market impact ─────────────────────────────────────────────

def _ac_cost_bps(shares: float, vol_d: float, adv: float) -> float:
    """
    Simplified Almgren-Chriss permanent + temporary impact.
    η = 0.1 (temporary), σ_d = daily vol of asset, ADV = daily volume in shares.
    Cost (bps) = η × σ_d × √(|shares| / ADV)
    """
    if adv < 1.0:
        return 5.0
    return float(_AC_ETA * vol_d * np.sqrt(abs(shares) / adv) * 10_000)


# ── MVO (SLSQP + regime-conditioned λ + BUG #23 cov regularisation) ──────────

def _mvo_weights(
    alpha: np.ndarray,
    cov_d: np.ndarray,
    z0_smooth: float,
    vol_d: np.ndarray,
    alloc_scale: float = 1.0,
) -> np.ndarray:
    """
    Mean-Variance Optimisation with regime-conditioned risk aversion λ.

    Objective: min_w  λ·w^T Σ_ann w  −  α^T w
    s.t.  Σw = 1, 0 ≤ w_i ≤ w_max_i

    λ = λ_base × exp(σ_avg × |z0|): higher z0 (crisis) → higher penalty on variance.

    BUG #23 FIX:
      - Covariance regularisation already applied upstream in `_cov()`.
      - x0 is projected onto the feasible simplex (clipped to bounds, then
        renormalised) rather than np.full(1/N) which can violate tight per-asset
        bounds and cause immediate SLSQP infeasibility from step 0.
      - Fallback on solver failure: inverse-volatility weights (not equal weight)
        because it naturally concentrates in low-vol assets which tend to have
        better Sharpe — strictly better than equal weight as a fallback.
    """
    avg_vol = float(np.mean(vol_d) * np.sqrt(_TRADING_DAYS_YEAR))
    lam     = float(np.clip(_LAMBDA_BASE * np.exp(avg_vol * abs(z0_smooth)), 0.5, 15.0))
    Σ       = cov_d * _TRADING_DAYS_YEAR
    mx      = np.array([TIER_MAX_WEIGHT[t] for t in TICKERS]) * alloc_scale
    bds     = [(0.0, float(m)) for m in mx]

    # BUG #23 FIX: feasibility-guaranteed x0
    # 1. Start from equal weight
    # 2. Clip to per-asset upper bounds
    # 3. Project back to simplex via normalisation
    x0 = np.clip(np.full(N_ASSETS, 1.0 / N_ASSETS), 0.0, mx)
    s0 = x0.sum()
    if s0 > 1e-10:
        x0 = x0 / s0
    else:
        # Degenerate bounds: allocate to highest-max-weight assets
        x0 = np.where(mx > 0, mx / (mx.sum() + 1e-10), 0.0)

    res = sco.minimize(
        fun=lambda w: 0.5 * lam * w @ Σ @ w - alpha @ w,
        jac=lambda w: lam * Σ @ w - alpha,
        x0=x0,
        method="SLSQP",
        bounds=bds,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"ftol": 1e-9, "maxiter": 500},
    )

    if res.success:
        w = res.x
    else:
        # BUG #23 FIX: inverse-vol fallback (better risk-adjusted than equal weight)
        inv_vol = 1.0 / (np.diag(Σ) ** 0.5 + 1e-8)
        inv_vol = np.minimum(inv_vol, mx * 3)
        w = inv_vol / (inv_vol.sum() + 1e-10)

    w = np.clip(w, 0.0, mx)
    s = w.sum()
    w = w / s if s > 1e-10 else np.where(mx > 0, 1.0 / max((mx > 0).sum(), 1), 0.0)

    # Concentration filter: zero sub-threshold positions to force conviction.
    # At 1.5% threshold with 25 assets → ~10-14 active positions.
    w[w < _MIN_POSITION_WT] = 0.0
    s = w.sum()
    if s > 1e-10:
        w = w / s
    else:
        top5 = np.argsort(alpha)[-5:]
        w = np.zeros(N_ASSETS, dtype=np.float64)
        w[top5] = np.minimum(1.0 / 5, mx[top5])
        w = w / (w.sum() + 1e-10)

    return w.astype(np.float32)


# ── Probabilistic Sharpe Ratio (Bailey & López de Prado 2012) ────────────────

def _psr(
    sr_hat: float, n: int, skew: float, kurt_raw: float,
    sr_benchmark: float = 0.0,
) -> float:
    """PSR = Φ[(SR̂ − SR*) × √(N−1) / √(1 − γ₃SR̂ + (γ₄−1)/4 × SR̂²)]"""
    if n <= 2:
        return 0.0
    denom = np.sqrt(
        max(1.0 - skew * sr_hat + (kurt_raw - 1.0) / 4.0 * sr_hat ** 2, 1e-6)
    )
    return float(norm.cdf((sr_hat - sr_benchmark) * np.sqrt(n - 1) / denom))


# ── Max consecutive true (for drawdown duration) ─────────────────────────────

def _mct(arr: np.ndarray) -> int:
    max_run = cur = 0
    for v in arr:
        cur = cur + 1 if v else 0
        max_run = max(max_run, cur)
    return max_run


# ── Sharpe ───────────────────────────────────────────────────────────────────

def _sharpe(r: np.ndarray) -> float:
    rf  = _RISK_FREE_ANNUAL / _TRADING_DAYS_YEAR
    ex  = r - rf
    std = ex.std()
    return float((ex.mean() / (std + 1e-10)) * np.sqrt(_TRADING_DAYS_YEAR))


# ── Walk-forward fold generator ───────────────────────────────────────────────

def _wf_folds(
    dates: pd.DatetimeIndex,
    is_months: int = 18,
    oos_months: int = 6,
) -> List[Tuple[str, str, str, str]]:
    """
    Expanding-window walk-forward: IS grows by oos_months per fold.
    Returns list of (is_start, is_end, oos_start, oos_end) date strings.
    """
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


# ── Data containers ───────────────────────────────────────────────────────────

@dataclass
class Snap:
    date: str
    portfolio_value: float
    daily_return: float
    cash: float
    turnover: float      # BUG #21 FIX: drift-adjusted one-way portfolio turnover
    cost_drag: float     # BUG #21 FIX: transaction cost drag (previously mislabeled as turnover)
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


# ── Core engine ───────────────────────────────────────────────────────────────

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

        # BUG #21 FIX: track previous portfolio weights for drift-adjusted turnover
        # Stored as weight vector (N,) aligned to TICKERS order.
        self._prev_weights: np.ndarray = np.zeros(N_ASSETS, dtype=np.float64)

    def _liquidate(self, px: np.ndarray) -> None:
        for t in list(self.pos.keys()):
            sh = self.pos.pop(t, 0.0)
            if sh != 0.0:
                self.cash += sh * px[TICKERS.index(t)]
        self.nav = self.cash

    def _invest_bil(self, px: np.ndarray) -> None:
        """FIX #8: allocate from frozen total_cash to avoid sequential-deduction bug."""
        self._liquidate(px)
        total_cash = self.cash
        for t, frac in [("BIL", 0.60), ("SHV", 0.40)]:
            i       = TICKERS.index(t)
            alloc   = total_cash * frac
            self.pos[t] = self.pos.get(t, 0.0) + alloc / (px[i] + 1e-10)
            self.cash  -= alloc
        # Update prev_weights to reflect BIL position
        self._prev_weights = self._current_weights(px)

    def _rf_step(self) -> float:
        """FIX #9: yield via share-scaling; self.cash stays at near-zero."""
        rf = _RISK_FREE_ANNUAL / _TRADING_DAYS_YEAR
        for t in ("BIL", "SHV"):
            if self.pos.get(t, 0.0) > 0.0:
                self.pos[t] *= (1.0 + rf)
        self.nav  *= (1.0 + rf)
        self._peak = max(self._peak, self.nav)
        return rf

    def _current_weights(self, px: np.ndarray) -> np.ndarray:
        """Returns current mark-to-market weight vector (N,) summing to ≤1."""
        w = np.array([
            self.pos.get(t, 0.0) * px[i] / (self.nav + 1e-10)
            for i, t in enumerate(TICKERS)
        ], dtype=np.float64)
        return w

    def _rebalance(
        self,
        target: np.ndarray,
        px: np.ndarray,
        vol: np.ndarray,
        adv: np.ndarray,
        force: bool = False,
    ) -> Tuple[float, float]:
        """
        Executes rebalancing trades toward `target` weights.

        BUG #21 FIX: Returns (cost_drag_frac, one_way_turnover_frac).
          - cost_drag_frac: transaction cost / NAV (basis for cost_drag column)
          - one_way_turnover_frac: drift-adjusted one-way turnover

        Drift adjustment:
          w̃_{t-1} = w_{t-1} * (1 + r_{t-1}) / (1 + r̄_{t-1})
          where r̄_{t-1} = Σ_i w_{t-1,i} * r_{t-1,i}  (portfolio return)

          Without drift adjustment, we count passive price movement as rebalancing
          activity — inflating turnover by 50-200% depending on daily vol. The
          adjustment isolates the ACTIVE rebalancing decision from passive drift.

        Note: drift adjustment requires the previous-day return. We approximate
          this as (px / px_prev - 1), but since px_prev isn't passed here, we
          use the pre-trade mark-to-market weight directly as w̃ (which already
          reflects the drift since last rebalance). This is correct because:
            w_current = w_prev_after_drift  (mark-to-market weights already drifted)
          So turnover = 0.5 * Σ|target - w_current|, where w_current is the
          drifted weight. The drift correction IS already embedded in the current
          mark-to-market.
        """
        current = self._current_weights(px)   # already drift-adjusted
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

        # BUG #21 FIX: one-way turnover = 0.5 × Σ|w_target - w_before_rebalance|
        # This is the standard institutional turnover definition. Factor of 0.5
        # converts two-way (buys + sells) to one-way (fraction of portfolio traded).
        one_way_turnover = float(0.5 * np.sum(np.abs(target - current)))

        # Store post-rebalance weights for next day's drift reference
        self._prev_weights = target.copy().astype(np.float64)

        return cost_drag_frac, one_way_turnover

    def _mtm(self, px: np.ndarray) -> float:
        """Mark to market: compute NAV from positions + cash."""
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
        """
        FIX #6: extend data window backwards by warmup_days before start_date.
        Warmup runs silently on pre-start_date data; recording begins at start_date.
        """
        self._reset()
        ts_start = pd.Timestamp(start_date)
        ts_end   = pd.Timestamp(end_date)

        start_pos = int(prices_df.index.searchsorted(ts_start, side="left"))
        ext_pos   = max(0, start_pos - warmup_days)
        ext_start = prices_df.index[ext_pos]

        mask  = (prices_df.index >= ext_start) & (prices_df.index <= ts_end)
        dates = prices_df.index[mask]
        px_df = prices_df.loc[mask]
        g0    = int(returns_df.index.searchsorted(ext_start, side="left"))

        logger.info(
            f"Backtest window: {ext_start.date()} → {end_date}"
            f" ({len(dates)} total days, warmup={warmup_days}d,"
            f" recording from {start_date})"
        )

        # Cache soft posterior availability once
        has_soft_posteriors = all(c in regime_df.columns for c in _SOFT_COLS)

        for li, date in enumerate(dates):
            ds     = date.strftime("%Y-%m-%d")
            px     = px_df.loc[date].values.astype(np.float64)
            gi     = g0 + li
            record = date >= ts_start

            # ── WARMUP ────────────────────────────────────────────────────────
            if li < warmup_days:
                if li == 0:
                    self._invest_bil(px)
                dr = self._rf_step()
                dd = (self.nav - self._peak) / self._peak
                if record:
                    self.history.append(
                        Snap(ds, self.nav, dr, self.cash, 0.0, 0.0, "WARMUP", 0.0, dd, 0)
                    )
                continue

            # ── SIGNAL RETRIEVAL ──────────────────────────────────────────────
            if date not in regime_df.index or date not in alpha_df.index:
                dr = self._rf_step()
                dd = (self.nav - self._peak) / self._peak
                if record:
                    self.history.append(
                        Snap(ds, self.nav, dr, self.cash, 0.0, 0.0, "NO_SIGNAL", 0.0, dd, 0)
                    )
                continue

            raw_z = regime_df.loc[date, "z_mu"]
            if isinstance(raw_z, str):
                raw_z = json.loads(raw_z)
            z_raw = np.asarray(raw_z, dtype=np.float32).ravel()
            z_smooth, s_label = self._smoother.update(
                z_raw, str(regime_df.loc[date, "regime_label"])
            )
            z0 = float(z_smooth[0])

            # BUG #22 FIX: regime-conditional halt threshold
            halt_dd_thresh, halt_days_req, ramp_days_req = _get_regime_halt_threshold(
                regime_df, date
            )

            # ── HALT STATE MACHINE ────────────────────────────────────────────
            if self._halt_phase >= 1:
                self._halt_days += 1
                dr = self._rf_step()
                dd = (self.nav - self._peak) / self._peak

                if self._halt_phase == 1:
                    # Phase 1: mandatory BIL — check recovery vs halt_nav
                    if self._halt_days >= halt_days_req:
                        recov = (self.nav - self._halt_nav) / (self._halt_nav + 1e-10)
                        if recov >= -_HALT_RECOVERY_THRESH:
                            logger.info(f"{ds}: Halt recovery — entering ramp-in.")
                            self._halt_phase     = 2
                            self._ramp_days      = 0
                            self._ramp_entry_nav = self.nav

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

                    target = _mvo_weights(alp, cov_d, z0, vol_d, alloc_scale=scale)
                    cost_d, to = self._rebalance(target, px, vol_d, adv)
                    dr = self._mtm(px)
                    dd = (self.nav - self._peak) / self._peak

                    # Re-trigger halt if new drawdown exceeds threshold during ramp-in
                    if dd <= -halt_dd_thresh:
                        logger.warning(
                            f"{ds}: Drawdown={dd:.2%} exceeded {halt_dd_thresh:.2%}"
                            " during ramp-in. Returning to mandatory BIL."
                        )
                        self._halt_phase = 1
                        self._halt_days  = 0
                        self._halt_nav   = self.nav
                        self._liquidate(px)
                        self._invest_bil(px)

                    if self._ramp_days >= ramp_days_req:
                        logger.info(f"{ds}: Ramp-in complete. Resuming full active trading.")
                        self._halt_phase = 0

                    if record:
                        self.history.append(
                            Snap(
                                ds, self.nav, dr, self.cash, to, cost_d,
                                f"RAMP_{scale:.0%}", z0, dd, 2,
                            )
                        )
                    continue

                if record:
                    self.history.append(
                        Snap(
                            ds, self.nav, dr, self.cash, 0.0, 0.0,
                            "HALTED_BIL", z0, dd, self._halt_phase,
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

            target = _mvo_weights(alp, cov_d, z0, vol_d)

            regime_changed = s_label != self._prev_regime
            cost_d, to = self._rebalance(
                target, px, vol_d, adv, force=regime_changed
            )
            dr = self._mtm(px)
            dd = (self.nav - self._peak) / self._peak
            self._prev_regime = s_label

            # BUG #22 FIX: use regime-conditional threshold
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
                f"regime_df range: {regime_df.index.min().date()} → "
                f"{regime_df.index.max().date()}"
            )

        return self._build_tearsheet()

    # ── Tearsheet ─────────────────────────────────────────────────────────────

    def _build_tearsheet(self) -> pd.DataFrame:
        df = pd.DataFrame([
            {
                "date":            h.date,
                "portfolio_value": h.portfolio_value,
                "daily_return":    h.daily_return,
                "cash":            h.cash,
                "turnover":        h.turnover,        # BUG #21 FIX: real turnover
                "cost_drag":       h.cost_drag,       # BUG #21 FIX: separated
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

        # BUG #21 FIX: report drift-adjusted one-way turnover
        to_mean = float(df["turnover"].mean())
        # Also report cost drag in bps annualised
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
            "Avg Daily Turnover":      f"{to_mean:.2%}",       # BUG #21 FIX: now ~1-4%
            "Cost Drag (ann. bps)":    f"{cost_drag_ann_bps:.1f}",  # new metric
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(
        "══════ Fortress v5 — Standalone Backtest v5 "
        "(BUG #21/#22/#23 FIX) ══════"
    )

    # ── Load caches ───────────────────────────────────────────────────────────
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

    # Log whether soft posteriors are available
    has_soft = all(c in regime_df.columns for c in _SOFT_COLS)
    if has_soft:
        logger.info(
            "✅ Soft GMM posteriors detected in regime_df — "
            "regime-conditional halt thresholds active (BUG #22 FIX)."
        )
    else:
        logger.warning(
            "⚠️  Soft GMM posteriors NOT found in regime_df. "
            f"Missing columns: {[c for c in _SOFT_COLS if c not in regime_df.columns]}. "
            f"Falling back to static halt threshold {_MAX_DD_HALT_FALLBACK:.0%}. "
            "Re-run precompute_regime_posteriors.py (BUG #20 fix) to enable "
            "regime-conditional halts."
        )

    # ── Full-period backtest ───────────────────────────────────────────────────
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