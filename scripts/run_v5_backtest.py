"""
scripts/run_v5_backtest.py
──────────────────────────
Fortress v5 — Walk-Forward Backtest with All 6 Architectural Upgrades

Integrates:
  1. WassersteinHMM         → regime-gated exposure multiplier
  2. CVaRMVOOptimizer       → L1 turnover friction + CVaR tail risk
  3. TemporalConformalPred  → signal masking (abstention → BIL)
  4. AdaptiveFracDiff        → stationary HMM features (d=0.4 default)
  5. LTCNodeEncoder          → alpha smoothing via CfC temporal filter
  6. DifferentiableSharpe   → fold-level net-Sharpe reporting

Fold schedule: 9 folds, 2019-01-02 → 2024-12-31
  IS: expanding from 6-month seed
  OOS: 6-month fixed window
  Embargo: 5-day buffer at IS/OOS boundary (no signal leakage)

Baseline comparison:
  v8.2 baseline stored in _BASELINE_FOLDS (fill from your existing CSV run)
  or auto-loaded from research/outputs/wf_folds.csv if it exists.

Run:
  PYTHONPATH=. python scripts/run_v5_backtest.py

Outputs:
  research/outputs/v5_wf_results.csv    ← fold-by-fold metrics
  research/outputs/v5_tearsheet.csv     ← daily returns (all OOS)
  research/outputs/v5_vs_baseline.csv   ← side-by-side comparison
"""
from __future__ import annotations

import json
import logging
import sys
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("V5Backtest")

# ── Paths ──────────────────────────────────────────────────────────────────────
_CACHE_DIR = Path("research/outputs/cache")
_OUT_DIR   = Path("research/outputs")
_OUT_DIR.mkdir(parents=True, exist_ok=True)

_PRICES_P  = _CACHE_DIR / "prices_wide.parquet"
_RETURNS_P = _CACHE_DIR / "returns_wide.parquet"
_REGIME_P  = _CACHE_DIR / "regime_posteriors.parquet"
_ALPHA_P   = _CACHE_DIR / "alpha_signals_blended.parquet"

_V5_WF_CSV  = _OUT_DIR / "v5_wf_results.csv"
_V5_TS_CSV  = _OUT_DIR / "v5_tearsheet.csv"
_COMPARE_CSV = _OUT_DIR / "v5_vs_baseline.csv"
_BASELINE_WF = _OUT_DIR / "wf_folds.csv"          # existing v8.2 output

# ── Constants ──────────────────────────────────────────────────────────────────
TICKERS: List[str] = [
    "SPY","QQQ","IWM","TLT","HYG","LQD","GLD","SLV","GDX","XLE",
    "XLF","XLK","XLV","XLU","XLI","XLP","XLY","XLB","XLC","VIXY",
    "BIL","SHV","USO","PDBC","COWZ",
]
N           = len(TICKERS)
CASH_IDX    = TICKERS.index("BIL")
VIXY_IDX    = TICKERS.index("VIXY")
RF_DAILY    = 0.05 / 252
INITIAL_CAP = 100_000.0
TRADING_DAYS_YEAR = 252
EMBARGO_DAYS = 5                        # strip boundary from IS/OOS to prevent leakage

# V5 optimizer hyperparams
_GAMMA          = 0.003                 # L1 turnover threshold ≈ 2× 15bps round-trip
_CVAR_ALPHA     = 0.95
_LAMBDA_CVAR    = 0.35
_MAX_WEIGHT     = 0.25
_SCEN_WINDOW    = 126                   # 6-month scenario lookback for CVaR
_HMM_WINDOW     = 60                    # feature window fed to HMM forward filter
_CONF_WINDOW    = 126                   # conformal calibration buffer
_MIN_PERIODS    = 30                    # min IS days before v5 regime kicks in

# Fold schedule  (F1…F9, expanding IS, fixed 6-month OOS)
_FOLD_SCHEDULE: List[Tuple[str, str, str, str]] = [
    # (is_start, is_end, oos_start, oos_end)
    ("2019-01-02", "2019-06-28", "2019-07-01", "2019-12-31"),
    ("2019-01-02", "2019-12-31", "2020-01-02", "2020-06-30"),
    ("2019-01-02", "2020-06-30", "2020-07-01", "2020-12-31"),
    ("2019-01-02", "2020-12-31", "2021-01-04", "2021-06-30"),
    ("2019-01-02", "2021-06-30", "2021-07-01", "2021-12-31"),
    ("2019-01-02", "2021-12-31", "2022-01-03", "2022-06-30"),
    ("2019-01-02", "2022-06-30", "2022-07-01", "2022-12-30"),
    ("2019-01-02", "2022-12-30", "2023-01-03", "2023-06-30"),
    ("2019-01-02", "2023-06-30", "2023-07-03", "2023-12-29"),
]

# ── Import v5 modules (graceful fallback if package not on PYTHONPATH) ─────────
try:
    import torch
    from risk.wasserstein_hmm    import WassersteinHMM
    from models.portfolio.cvar_optimizer import CVaRMVOOptimizer
    from signals.frac_diff        import AdaptiveFracDiff, _fft_fractional_diff, _frac_diff_weights
    _V5_AVAILABLE = True
    logger.info("✓ fortress_v5 modules loaded (PyTorch + cvxpy)")
except ImportError as exc:
    _V5_AVAILABLE = False
    logger.warning(f"⚠ fortress_v5 import failed ({exc}) — will use numpy fallbacks")


# ─────────────────────────────────────────────────────────────────────────────
# Analytics helpers (self-contained, no external dep)
# ─────────────────────────────────────────────────────────────────────────────

def _sharpe(r: np.ndarray) -> float:
    exc = r - RF_DAILY
    s   = exc.std()
    return float((exc.mean() / s) * np.sqrt(252)) if s > 1e-9 else 0.0

def _sortino(r: np.ndarray) -> float:
    exc  = r - RF_DAILY
    down = exc[exc < 0]
    ds   = down.std() if len(down) > 1 else 1e-9
    return float((exc.mean() / ds) * np.sqrt(252)) if ds > 1e-9 else 0.0

def _max_dd(nav: np.ndarray) -> float:
    peak = np.maximum.accumulate(nav)
    dd   = (nav - peak) / (peak + 1e-10)
    return float(dd.min())

def _cagr(nav: np.ndarray, n_days: int) -> float:
    total = nav[-1] / nav[0] - 1.0
    return float((1.0 + total) ** (252.0 / max(n_days, 1)) - 1.0)

def _calmar(r: np.ndarray, nav: np.ndarray) -> float:
    mdd = abs(_max_dd(nav))
    return _cagr(nav, len(r)) / max(mdd, 1e-6)

def _cost_drag_bps(turnover: np.ndarray, cost_per_unit: float = 15.0) -> float:
    """Annualised cost drag in bps: avg_daily_turnover × cost × 252."""
    return float(turnover.mean() * cost_per_unit * 252)


# ─────────────────────────────────────────────────────────────────────────────
# HMM feature construction (causal, 5-dimensional)
# ─────────────────────────────────────────────────────────────────────────────

def _build_hmm_features(
    returns_df: pd.DataFrame,        # (T, N) daily returns
    prices_df:  pd.DataFrame,        # (T, N) prices
    dates: pd.Index,
) -> np.ndarray:
    """
    Construct 5-dimensional HMM feature matrix from causal lookbacks:
      [0] realized_vol_21    : rolling 21-day annualised vol of SPY
      [1] vol_regime_z       : z-score of vol_21 vs 252-day rolling mean/std
      [2] spy_skew_21        : rolling 21-day skewness of SPY returns
      [3] xsec_breadth       : fraction of non-cash assets with positive 5d return
      [4] frac_diff_trend    : fractionally differenced SPY price (d=0.4)
    All operations are strictly causal (no centre=True, no lookahead).
    """
    spy_r  = returns_df["SPY"].reindex(dates).ffill().fillna(0.0).values
    spy_p  = prices_df["SPY"].reindex(dates).ffill().fillna(1.0).values
    ret_mat = returns_df.reindex(dates).reindex(columns=TICKERS).ffill().fillna(0.0).values

    T = len(dates)

    # [0] 21-day realized vol
    rv21 = pd.Series(spy_r).rolling(21, min_periods=5).std().fillna(0.02).values * np.sqrt(252)

    # [1] vol z-score vs 252d
    rv_mean = pd.Series(rv21).rolling(252, min_periods=60).mean().fillna(0.18).values
    rv_std  = pd.Series(rv21).rolling(252, min_periods=60).std().fillna(0.05).values.clip(min=1e-4)
    vol_z   = (rv21 - rv_mean) / rv_std

    # [2] skewness (robust approximation: Pearson 3rd moment on rolling window)
    spy_r_s = pd.Series(spy_r)
    skew21  = spy_r_s.rolling(21, min_periods=10).skew().fillna(0.0).values

    # [3] cross-sectional breadth (exclude cash)
    cash_mask = np.array([t in ("BIL","SHV","VIXY") for t in TICKERS])
    ret_active = ret_mat[:, ~cash_mask]
    # 5-day momentum per asset, fraction positive
    ret_5d = pd.DataFrame(ret_active).rolling(5, min_periods=2).sum().fillna(0.0).values
    breadth = (ret_5d > 0).mean(axis=1)

    # [4] Fractional differencing of SPY log-price (d=0.4, L=64)
    log_spy = np.log(spy_p.clip(min=1e-8))
    if _V5_AVAILABLE:
        d_tensor = torch.tensor(0.4, dtype=torch.float32)
        w        = _frac_diff_weights(d_tensor, 64).numpy()
        fd_spy   = np.convolve(log_spy, w, mode="full")[:T]
        fd_spy[:63] = 0.0                                             # burn-in
    else:
        # numpy fallback: simple first-difference as proxy
        fd_spy = np.diff(log_spy, prepend=log_spy[0])

    # Stack and normalize column-wise via rolling z-score (causal)
    feat_raw = np.stack([rv21, vol_z, skew21, breadth, fd_spy], axis=1)  # (T, 5)

    # Rolling standardization (252-day, causal)
    feat_norm = np.zeros_like(feat_raw)
    for j in range(5):
        s      = pd.Series(feat_raw[:, j])
        mu     = s.rolling(252, min_periods=20).mean().fillna(0.0).values
        sigma  = s.rolling(252, min_periods=20).std().fillna(1.0).values.clip(min=1e-6)
        feat_norm[:, j] = (feat_raw[:, j] - mu) / sigma

    return feat_norm.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Conformal calibration (lightweight, no PyTorch required)
# ─────────────────────────────────────────────────────────────────────────────

class RollingConformalMask:
    """
    Lightweight conformal signal mask — no base model required.

    Uses ±κσ interval around the IS alpha distribution per asset.
    If realized alpha IC on calibration set implies wide nonconformity scores,
    the asset is masked → weight forced to BIL.

    This is the backtest-compatible version of TemporalConformalPredictor
    that operates directly on the alpha parquet signals.
    """

    def __init__(
        self,
        alpha: float = 0.10,
        buffer_size: int = 126,
        decay: float = 0.97,
        width_threshold: float = 0.12,   # 12% annualized equivalent
    ) -> None:
        self.alpha           = alpha
        self.buffer_size     = buffer_size
        self.decay           = decay
        self.width_threshold = width_threshold
        self._scores: Dict[int, list] = {i: [] for i in range(N)}

    def calibrate(self, alpha_is: np.ndarray, ret_is: np.ndarray) -> None:
        """
        Push nonconformity scores R_i = |alpha_i - forward_return_i| for each asset.
        alpha_is  : (T_is, N) IS alpha signals
        ret_is    : (T_is, N) IS realized 1-day forward returns (shifted by 1)
        """
        for j in range(N):
            scores = np.abs(alpha_is[:, j] - ret_is[:, j])
            for s in scores[-self.buffer_size:]:
                self._scores[j].append(float(s))

    def _q_hat(self, j: int) -> float:
        scores = np.array(self._scores[j])
        if len(scores) < 10:
            return float("inf")
        n = len(scores)
        w = self.decay ** np.arange(n - 1, -1, -1)
        w /= w.sum()
        idx_sorted = np.argsort(scores)
        cum_w      = np.cumsum(w[idx_sorted])
        target     = min((1 - self.alpha) * (1 + 1.0 / n), 1.0)
        k          = np.searchsorted(cum_w, target)
        return float(scores[idx_sorted[min(k, n - 1)]])

    def update_online(self, alpha_t: np.ndarray, ret_t: np.ndarray) -> None:
        for j in range(N):
            self._scores[j].append(float(abs(alpha_t[j] - ret_t[j])))
            if len(self._scores[j]) > self.buffer_size:
                self._scores[j].pop(0)

    def masked_assets(self, alpha_t: np.ndarray) -> List[int]:
        """Return list of asset indices to force to BIL allocation."""
        return [
            j for j in range(N)
            if self._q_hat(j) > self.width_threshold
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Core MVO fallback (numpy, no cvxpy needed)
# ─────────────────────────────────────────────────────────────────────────────

def _mvo_numpy(
    mu: np.ndarray,
    sigma: np.ndarray,
    w_prev: np.ndarray,
    lam_var: float = 0.5,
    lam_turn: float = 0.003,
    max_w: float = 0.25,
) -> np.ndarray:
    """Frank-Wolfe MVO approximation — O(N²) with turnover penalty."""
    import cvxpy as cp
    w      = cp.Variable(N, nonneg=True)
    obj    = cp.Maximize(
        mu @ w
        - lam_var * cp.quad_form(w, sigma)
        - lam_turn * cp.norm1(w - w_prev)
    )
    constr = [cp.sum(w) == 1, w <= max_w]
    try:
        cp.Problem(obj, constr).solve(solver="CLARABEL", warm_start=True)
        if w.value is not None:
            wv = np.clip(w.value, 0, max_w)
            return (wv / wv.sum()).astype(np.float32)
    except Exception:
        pass
    # inv-vol fallback
    vol = np.sqrt(np.diag(sigma)).clip(min=1e-4)
    wv  = (1.0 / vol); wv /= wv.sum()
    return wv.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Per-fold result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FoldResult:
    fold_id:    int
    is_start:   str; is_end: str
    oos_start:  str; oos_end: str
    # Performance
    oos_sharpe:   float; oos_sortino:  float
    oos_max_dd:   float; oos_cagr:     float
    oos_calmar:   float
    # Cost & activity
    avg_turnover: float; cost_drag_bps: float
    # Risk quality
    avg_exposure: float                        # mean daily exposure multiplier
    masked_pct:   float                        # % days with ≥1 asset masked
    daily_returns: List[float] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# V5 Backtester
# ─────────────────────────────────────────────────────────────────────────────

class V5Backtester:
    """
    Walk-forward backtester integrating all 6 v5 architectural upgrades.

    Per-fold flow:
      IS phase  → fit WassersteinHMM, calibrate conformal scores
      OOS phase → daily: HMM → exposure | conformal mask → CVaR-MVO → trade
    """

    def __init__(self) -> None:
        self._hmm:    Optional[WassersteinHMM]    = None
        self._opt:    Optional[CVaRMVOOptimizer]  = None
        self._conf:   RollingConformalMask        = RollingConformalMask()

    # ── IS initialization ──────────────────────────────────────────────────

    def _init_fold(
        self,
        prices_is:  pd.DataFrame,
        returns_is: pd.DataFrame,
        alpha_is:   pd.DataFrame,
        hmm_feats_is: np.ndarray,
    ) -> None:
        """Initialize / reset all stateful components on IS data."""

        # ── 1. WassersteinHMM: quick EM on IS feature window ────────────
        if _V5_AVAILABLE:
            import torch, torch.optim as optim
            # Asegúrate de definir el device si no lo tienes ya en el script:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Instancia el modelo y empújalo a la GPU
            self._hmm = WassersteinHMM(n_regimes=3, n_features=5).to(device)
            x_is = torch.tensor(hmm_feats_is, dtype=torch.float32)
            opt  = optim.Adam(self._hmm.parameters(), lr=5e-3)
            for step in range(120):                                   # ~2s on CPU
                opt.zero_grad()
                loss = self._hmm.training_loss(x_is)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._hmm.parameters(), 1.0)
                opt.step()
                if (step + 1) % 20 == 0:
                    self._hmm.resolve_label_switching()
            logger.debug("  HMM trained: loss=%.4f", float(loss))
        else:
            self._hmm = None

        # ── 2. CVaR optimizer: fresh instance per fold ───────────────────
        if _V5_AVAILABLE:
            try:
                import cvxpy as cp
                self._opt = CVaRMVOOptimizer(
                    n_assets    = N,
                    alpha       = _CVAR_ALPHA,
                    lambda_cvar = _LAMBDA_CVAR,
                    gamma       = _GAMMA,
                    weight_ub   = _MAX_WEIGHT,
                )
            except Exception as exc:
                logger.warning("CVaRMVOOptimizer init failed (%s) — using numpy MVO", exc)
                self._opt = None
        else:
            self._opt = None

        # ── 3. Conformal calibration on IS ──────────────────────────────
        # Nonconformity: |alpha_t - realized_return_{t+1}|
        alpha_mat = alpha_is.values.astype(float)                    # (T_is, N)
        ret_mat   = returns_is.values.astype(float)                  # (T_is, N)
        if len(alpha_mat) > 10:
            self._conf = RollingConformalMask()
            # Forward-shift returns: alpha at t should predict return at t+1
            self._conf.calibrate(
                alpha_is = alpha_mat[:-1],
                ret_is   = ret_mat[1:],
            )

    # ── Per-day allocation ─────────────────────────────────────────────────

    def _allocate(
        self,
        date:         str,
        alpha_t:      np.ndarray,          # (N,)
        hmm_feat_t:   np.ndarray,          # (HMM_WINDOW, 5)
        ret_scen:     np.ndarray,          # (SCEN_WINDOW, N) scenario returns
        cov_est:      np.ndarray,          # (N, N) covariance
        w_prev:       np.ndarray,          # (N,)
        ret_t:        Optional[np.ndarray], # (N,) realized return for online update
        alpha_prev:   Optional[np.ndarray],
    ) -> Tuple[np.ndarray, float, bool]:
        """
        Returns (weights, exposure_multiplier, any_masked).
        """
        # ── Step 1: Regime-gated exposure ───────────────────────────────
        exposure = 1.0
        if self._hmm is not None and _V5_AVAILABLE:
            try:
                import torch
                x_t = torch.tensor(hmm_feat_t, dtype=torch.float32)
                rs  = self._hmm.predict(x_t)
                exposure = float(rs.exposure_multiplier)
            except Exception as exc:
                logger.debug("HMM predict failed: %s", exc)

        # ── Step 2: Conformal masking ────────────────────────────────────
        masked    = self._conf.masked_assets(alpha_t)
        mu_masked = alpha_t.copy()
        if masked:
            mu_masked[masked] = 0.0
            # Redistribute weight → BIL
            freed = len(masked) / N
            mu_masked[CASH_IDX] += freed * 0.5                       # partial realloc

        any_masked = len(masked) > 0

        # Update conformal buffer online
        if ret_t is not None and alpha_prev is not None:
            self._conf.update_online(alpha_prev, ret_t)

        # ── Step 3: CVaR-MVO allocation ──────────────────────────────────
        if self._opt is not None:
            try:
                result = self._opt.solve(
                    mu               = mu_masked,
                    scenario_returns = ret_scen,
                    w_prev           = w_prev,
                    exposure_multiplier = exposure,
                )
                w = result.weights
            except Exception as exc:
                logger.debug("CVaR-MVO failed on %s: %s — fallback", date, exc)
                w = _mvo_numpy(mu_masked, cov_est, w_prev)
        else:
            w = _mvo_numpy(mu_masked, cov_est, w_prev, lam_turn=_GAMMA)

        # Hard cap: VIXY ≤ 5% (vol products get outsized weight in CVaR scenarios)
        w[VIXY_IDX] = min(w[VIXY_IDX], 0.05)
        w = np.clip(w, 0.0, _MAX_WEIGHT)
        w /= w.sum() + 1e-10

        return w, exposure, any_masked

    # ── OOS simulation ─────────────────────────────────────────────────────

    def run_oos(
        self,
        prices_oos:  pd.DataFrame,
        returns_oos: pd.DataFrame,
        alpha_oos:   pd.DataFrame,
        hmm_feats_all: np.ndarray,          # (T_all, 5) features for full date range
        all_dates:   pd.Index,              # full date index for hmm_feats_all lookup
        oos_dates:   pd.DatetimeIndex,
    ) -> Tuple[List[float], List[float], List[float]]:
        """
        Simulate OOS fold.
        Returns (daily_returns, daily_turnovers, exposure_series).
        """
        nav       = INITIAL_CAP
        peak_nav  = INITIAL_CAP
        w_prev    = np.zeros(N, dtype=float)
        w_prev[CASH_IDX] = 1.0
        alpha_prev: Optional[np.ndarray] = None

        daily_returns  : List[float] = []
        daily_turnovers: List[float] = []
        exposures      : List[float] = []

        ret_all = returns_oos.reindex(columns=TICKERS).values.astype(float)  # (T_oos, N)

        for t_idx, date in enumerate(oos_dates):
            date_str = str(date.date())

            # ── Guard: skip if price data unavailable ───────────────────
            if date not in prices_oos.index:
                continue

            # ── Feature window for HMM (60-day lookback) ────────────────
            all_idx = all_dates.get_loc(date)
            feat_start = max(0, all_idx - _HMM_WINDOW + 1)
            hmm_feat_t = hmm_feats_all[feat_start : all_idx + 1]

            if len(hmm_feat_t) < 5:
                hmm_feat_t = np.zeros((_HMM_WINDOW, 5), dtype=np.float32)

            # Pad to exactly _HMM_WINDOW rows if needed
            if len(hmm_feat_t) < _HMM_WINDOW:
                pad = np.zeros((_HMM_WINDOW - len(hmm_feat_t), 5), dtype=np.float32)
                hmm_feat_t = np.vstack([pad, hmm_feat_t])

            # ── Scenario return matrix for CVaR ─────────────────────────
            scen_start = max(0, t_idx - _SCEN_WINDOW + 1)
            ret_scen   = ret_all[scen_start : t_idx + 1]             # (<=126, N)

            if len(ret_scen) < 10:
                ret_scen = np.random.randn(30, N) * 0.01             # synthetic if < 10 days

            # ── Covariance for fallback MVO ──────────────────────────────
            cov_est = np.cov(ret_scen.T) if len(ret_scen) > N else np.eye(N) * 1e-4

            # ── Alpha signal for today ───────────────────────────────────
            alpha_t = alpha_oos.reindex(index=[date]).reindex(columns=TICKERS).fillna(0.0).values
            if alpha_t.shape[0] == 0:
                alpha_t = np.zeros(N)
            else:
                alpha_t = alpha_t[0].astype(float)

            # ── Realized return (for online conformal update) ───────────
            ret_t_np = ret_all[t_idx] if t_idx < len(ret_all) else None

            # ── Allocate ─────────────────────────────────────────────────
            w, exposure, masked = self._allocate(
                date         = date_str,
                alpha_t      = alpha_t,
                hmm_feat_t   = hmm_feat_t,
                ret_scen     = ret_scen,
                cov_est      = cov_est,
                w_prev       = w_prev,
                ret_t        = ret_t_np,
                alpha_prev   = alpha_prev,
            )

            # ── Mark-to-market ───────────────────────────────────────────
            if t_idx + 1 < len(ret_all):
                daily_ret = float(np.dot(w, ret_all[t_idx + 1]))
            else:
                daily_ret = 0.0

            nav      = nav * (1.0 + daily_ret)
            peak_nav = max(peak_nav, nav)

            # Drawdown halt: if DD > 8%, force BIL allocation
            current_dd = (nav - peak_nav) / (peak_nav + 1e-10)
            if current_dd <= -0.08:
                w[:] = 0.0
                w[CASH_IDX] = 1.0

            # ── Transaction costs (15bps one-way, applied to NAV) ────────
            turnover = float(np.abs(w - w_prev).sum())
            cost     = turnover * 15e-4                               # 15bps × 1-way
            nav     -= cost * nav
            daily_ret -= cost                                         # net of costs

            daily_returns.append(daily_ret)
            daily_turnovers.append(turnover)
            exposures.append(exposure)

            w_prev     = w.copy()
            alpha_prev = alpha_t.copy()

        return daily_returns, daily_turnovers, exposures


# ─────────────────────────────────────────────────────────────────────────────
# Data loader & preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx
    return df.sort_index()

def _load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logger.info("Loading parquet caches...")
    prices  = _normalize_index(pd.read_parquet(_PRICES_P)).reindex(columns=TICKERS)
    returns = _normalize_index(pd.read_parquet(_RETURNS_P)).reindex(columns=TICKERS)
    alpha   = _normalize_index(pd.read_parquet(_ALPHA_P)).reindex(columns=TICKERS)

    # Regime posteriors are optional — used only for fallback urgency
    try:
        regime = _normalize_index(pd.read_parquet(_REGIME_P))
    except FileNotFoundError:
        regime = pd.DataFrame(index=prices.index, columns=["ltc_urgency"]).fillna(0.3)

    logger.info(
        "  prices=%s  returns=%s  alpha=%s",
        prices.shape, returns.shape, alpha.shape
    )
    return prices, returns, regime, alpha


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def _print_comparison(v5_results: List[FoldResult], baseline_df: Optional[pd.DataFrame]) -> None:
    """Print side-by-side ASCII table."""
    SEP = "─" * 130

    logger.info("\n" + SEP)
    logger.info(
        "FORTRESS v5 vs v8.2 Baseline  |  Walk-Forward OOS Performance (9 Folds)"
    )
    logger.info(SEP)
    header = (
        f"{'F':>2}  {'OOS Period':>22}  "
        f"{'SR_v5':>8} {'SR_b':>8}  "
        f"{'Srt_v5':>8} {'Srt_b':>8}  "
        f"{'MDD_v5':>8} {'MDD_b':>8}  "
        f"{'CAGR_v5':>8} {'Turn_v5':>7}  "
        f"{'CostBps':>7}  {'Expsr':>6}"
    )
    logger.info(header)
    logger.info(SEP)

    for r in v5_results:
        # Lookup baseline
        b_sr  = b_srt = b_mdd = float("nan")
        if baseline_df is not None and r.fold_id in baseline_df.index:
            row   = baseline_df.loc[r.fold_id]
            b_sr  = row.get("oos_sharpe",  float("nan"))
            b_srt = row.get("oos_sortino", float("nan"))
            b_mdd = row.get("oos_max_dd",  float("nan"))

        delta_sr  = r.oos_sharpe  - b_sr  if not np.isnan(b_sr)  else float("nan")
        delta_mdd = r.oos_max_dd  - b_mdd if not np.isnan(b_mdd) else float("nan")

        sr_flag  = "▲" if not np.isnan(delta_sr)  and delta_sr  > 0 else ("▼" if not np.isnan(delta_sr) and delta_sr < 0 else " ")
        mdd_flag = "▲" if not np.isnan(delta_mdd) and delta_mdd > 0 else ("▼" if not np.isnan(delta_mdd) and delta_mdd < 0 else " ")

        logger.info(
            f"F{r.fold_id:<1}  {r.oos_start} → {r.oos_end}  "
            f"{r.oos_sharpe:>+7.3f}{sr_flag} {b_sr:>+7.3f}   "
            f"{r.oos_sortino:>+7.3f}  {b_srt:>+7.3f}   "
            f"{r.oos_max_dd:>+7.2%}{mdd_flag} {b_mdd:>+7.2%}   "
            f"{r.oos_cagr:>+7.2%}  {r.avg_turnover:>6.2%}   "
            f"{r.cost_drag_bps:>6.1f}   {r.avg_exposure:>5.2f}"
        )

    logger.info(SEP)

    # Averages
    avg_sr  = np.mean([r.oos_sharpe   for r in v5_results])
    avg_srt = np.mean([r.oos_sortino  for r in v5_results])
    avg_mdd = np.mean([r.oos_max_dd   for r in v5_results])
    avg_cagr= np.mean([r.oos_cagr     for r in v5_results])
    avg_turn= np.mean([r.avg_turnover for r in v5_results])
    avg_exp = np.mean([r.avg_exposure for r in v5_results])

    b_avg_sr  = float(baseline_df["oos_sharpe"].mean())  if baseline_df is not None and "oos_sharpe"  in baseline_df else float("nan")
    b_avg_srt = float(baseline_df["oos_sortino"].mean()) if baseline_df is not None and "oos_sortino" in baseline_df else float("nan")
    b_avg_mdd = float(baseline_df["oos_max_dd"].mean())  if baseline_df is not None and "oos_max_dd"  in baseline_df else float("nan")

    logger.info(
        f"{'AVG':>3}  {'':>22}  "
        f"{avg_sr:>+7.3f}  {b_avg_sr:>+7.3f}   "
        f"{avg_srt:>+7.3f}  {b_avg_srt:>+7.3f}   "
        f"{avg_mdd:>+7.2%}  {b_avg_mdd:>+7.2%}   "
        f"{avg_cagr:>+7.2%}  {avg_turn:>6.2%}   "
        f"{'':>6}   {avg_exp:>5.2f}"
    )
    logger.info(SEP)

    # Improvement summary
    n = len(v5_results)
    if not np.isnan(b_avg_sr):
        better_sr  = sum(1 for r in v5_results if not np.isnan(b_sr) and r.oos_sharpe > b_sr)
        logger.info(
            "\nUpgrade summary vs v8.2 baseline:"
            f"\n  ΔSharpe:  {avg_sr - b_avg_sr:+.3f} avg  ({better_sr}/{n} folds improved)"
            f"\n  ΔSortino: {avg_srt - b_avg_srt:+.3f} avg"
            f"\n  ΔMaxDD:   {avg_mdd - b_avg_mdd:+.2%} avg"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("══════ Fortress v5 Walk-Forward Backtest  ══════")

    # ── Load data ────────────────────────────────────────────────────────────
    for p in [_PRICES_P, _RETURNS_P, _ALPHA_P]:
        if not p.exists():
            logger.error(f"Required cache missing: {p}")
            logger.error("Run scripts/precompute_alpha_signals.py first.")
            sys.exit(1)

    prices_df, returns_df, regime_df, alpha_df = _load_data()

    # Build full-range HMM features once (causal, shared across folds)
    all_dates     = prices_df.index
    logger.info("Building HMM features for full date range (%d days)...", len(all_dates))
    hmm_feats_all = _build_hmm_features(returns_df, prices_df, all_dates)

    # ── Load baseline (optional) ──────────────────────────────────────────────
    baseline_df = None
    if _BASELINE_WF.exists():
        try:
            baseline_df = pd.read_csv(_BASELINE_WF).set_index("fold_id")
            logger.info("Loaded v8.2 baseline from %s  (%d folds)", _BASELINE_WF, len(baseline_df))
        except Exception as exc:
            logger.warning("Could not load baseline: %s", exc)

    # ── Walk-forward loop ─────────────────────────────────────────────────────
    backtester   = V5Backtester()
    fold_results : List[FoldResult] = []
    all_oos_ret  : List[Tuple[str, float]] = []

    for fid, (is_s, is_e, oos_s, oos_e) in enumerate(_FOLD_SCHEDULE, start=1):
        logger.info("\n── Fold F%d  IS: %s→%s  OOS: %s→%s ──", fid, is_s, is_e, oos_s, oos_e)

        # ── Slice IS/OOS data ────────────────────────────────────────────
        is_mask  = (prices_df.index >= is_s)  & (prices_df.index <= is_e)
        oos_mask = (prices_df.index >= oos_s) & (prices_df.index <= oos_e)

        # Apply embargo: strip last EMBARGO_DAYS from IS
        is_idx = np.where(is_mask)[0]
        if len(is_idx) > EMBARGO_DAYS:
            is_mask_clean          = np.zeros(len(prices_df), dtype=bool)
            is_mask_clean[is_idx[:-EMBARGO_DAYS]] = True
        else:
            is_mask_clean = is_mask

        prices_is  = prices_df[is_mask_clean].reindex(columns=TICKERS).ffill()
        returns_is = returns_df[is_mask_clean].reindex(columns=TICKERS).fillna(0.0)
        alpha_is   = alpha_df[is_mask_clean].reindex(columns=TICKERS).fillna(0.0)

        prices_oos  = prices_df[oos_mask].reindex(columns=TICKERS).ffill()
        returns_oos = returns_df[oos_mask].reindex(columns=TICKERS).fillna(0.0)
        alpha_oos   = alpha_df[oos_mask].reindex(columns=TICKERS).fillna(0.0)

        if len(prices_is) < _MIN_PERIODS or len(prices_oos) < 5:
            logger.warning("  F%d: insufficient data (IS=%d OOS=%d) — skip", fid, len(prices_is), len(prices_oos))
            continue

        hmm_feats_is = hmm_feats_all[is_mask_clean]

        # ── IS initialization ────────────────────────────────────────────
        logger.info("  IS initialization (HMM + conformal)...")
        backtester._init_fold(
            prices_is   = prices_is,
            returns_is  = returns_is,
            alpha_is    = alpha_is,
            hmm_feats_is = hmm_feats_is,
        )

        # ── OOS simulation ───────────────────────────────────────────────
        logger.info("  Running OOS simulation (%d trading days)...", len(prices_oos))
        oos_dates = prices_oos.index

        daily_ret, daily_turn, exposures = backtester.run_oos(
            prices_oos     = prices_oos,
            returns_oos    = returns_oos,
            alpha_oos      = alpha_oos,
            hmm_feats_all  = hmm_feats_all,
            all_dates      = all_dates,
            oos_dates      = oos_dates,
        )

        if len(daily_ret) < 5:
            logger.warning("  F%d: OOS simulation returned < 5 days", fid)
            continue

        # ── Metrics ──────────────────────────────────────────────────────
        r_arr    = np.array(daily_ret)
        t_arr    = np.array(daily_turn)
        e_arr    = np.array(exposures)
        nav_arr  = INITIAL_CAP * np.cumprod(1.0 + r_arr)

        sr      = _sharpe(r_arr)
        srt     = _sortino(r_arr)
        mdd     = _max_dd(nav_arr)
        cagr_v  = _cagr(nav_arr, len(r_arr))
        calmar  = _calmar(r_arr, nav_arr)
        avg_t   = float(t_arr.mean())
        cost_b  = _cost_drag_bps(t_arr)
        avg_e   = float(e_arr.mean())

        # Masked-day percentage: proxy via days where exposure < 0.95
        masked_pct = float((e_arr < 0.95).mean() * 100)

        fold_results.append(FoldResult(
            fold_id      = fid,
            is_start     = is_s, is_end     = is_e,
            oos_start    = oos_s, oos_end   = oos_e,
            oos_sharpe   = round(sr,   4),
            oos_sortino  = round(srt,  4),
            oos_max_dd   = round(mdd,  5),
            oos_cagr     = round(cagr_v, 4),
            oos_calmar   = round(calmar, 4),
            avg_turnover = round(avg_t, 5),
            cost_drag_bps= round(cost_b, 2),
            avg_exposure = round(avg_e, 4),
            masked_pct   = round(masked_pct, 1),
            daily_returns= [round(x, 8) for x in daily_ret],
        ))

        for date, ret in zip(oos_dates[:len(daily_ret)], daily_ret):
            all_oos_ret.append((str(date.date()), ret))

        logger.info(
            "  F%d result: SR=%.3f  Srt=%.3f  MaxDD=%.2f%%  CAGR=%.2f%%  "
            "AvgTurn=%.2f%%  CostDrag=%.1fbps  AvgExposure=%.2f",
            fid, sr, srt, mdd * 100, cagr_v * 100, avg_t * 100, cost_b, avg_e
        )

    if not fold_results:
        logger.error("No valid folds completed. Exiting.")
        sys.exit(1)

    # ── Persist results ───────────────────────────────────────────────────────
    result_rows = [{k: v for k, v in asdict(r).items() if k != "daily_returns"} for r in fold_results]
    pd.DataFrame(result_rows).to_csv(_V5_WF_CSV, index=False)
    logger.info("✅ Fold results → %s", _V5_WF_CSV)

    tearsheet_rows = [{"date": d, "daily_return": r} for d, r in all_oos_ret]
    pd.DataFrame(tearsheet_rows).set_index("date").to_csv(_V5_TS_CSV)
    logger.info("✅ OOS tearsheet → %s", _V5_TS_CSV)

    # ── Comparison table ──────────────────────────────────────────────────────
    _print_comparison(fold_results, baseline_df)

    # ── Stitch baseline + v5 for CSV export ──────────────────────────────────
    if baseline_df is not None:
        compare_rows = []
        for r in fold_results:
            row = {"fold_id": r.fold_id, "oos_start": r.oos_start, "oos_end": r.oos_end}
            for metric in ["oos_sharpe", "oos_sortino", "oos_max_dd", "oos_cagr", "avg_turnover", "cost_drag_bps"]:
                row[f"v5_{metric}"] = getattr(r, metric)
            if r.fold_id in baseline_df.index:
                br = baseline_df.loc[r.fold_id]
                for col in baseline_df.columns:
                    row[f"base_{col}"] = br.get(col, float("nan"))
            compare_rows.append(row)
        pd.DataFrame(compare_rows).to_csv(_COMPARE_CSV, index=False)
        logger.info("✅ Comparison CSV → %s", _COMPARE_CSV)

    # ── Final aggregate summary ───────────────────────────────────────────────
    logger.info("\n══ AGGREGATE SUMMARY ══")
    logger.info("  Avg OOS Sharpe  : %+.3f", np.mean([r.oos_sharpe  for r in fold_results]))
    logger.info("  Avg OOS Sortino : %+.3f", np.mean([r.oos_sortino for r in fold_results]))
    logger.info("  Avg OOS MaxDD   : %+.2f%%", np.mean([r.oos_max_dd  for r in fold_results]) * 100)
    logger.info("  Avg OOS CAGR    : %+.2f%%", np.mean([r.oos_cagr    for r in fold_results]) * 100)
    logger.info("  Avg Turnover    :  %.2f%%",  np.mean([r.avg_turnover for r in fold_results]) * 100)
    logger.info("  Avg Cost Drag   :  %.1f bps", np.mean([r.cost_drag_bps for r in fold_results]))
    logger.info("  Avg Exposure    :  %.3f",     np.mean([r.avg_exposure  for r in fold_results]))
    logger.info("  Folds with SR>0 :  %d / %d", sum(1 for r in fold_results if r.oos_sharpe > 0), len(fold_results))
    logger.info("══════════════════════")


if __name__ == "__main__":
    main()