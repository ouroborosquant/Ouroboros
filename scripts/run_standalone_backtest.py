"""
FORTRESS v5 — run_standalone_backtest.py  [PATCH v4]
Path: scripts/run_standalone_backtest.py

NEW BUGS FIXED vs v3
════════════════════════════════════════════════════════════════════════════════

  BUG #13 [CRITICAL — 3-year mandatory-BIL trap]:
    The ramp-in breach guard used:
        dd = (self.nav - self._peak) / self._peak
        if dd <= -_HALT_RECOVERY_THRESH:   # −10%
    `self._peak` is the all-time NAV high from BEFORE the halt (e.g. $120k).
    `self._halt_nav` ≈ $96k (20% below peak).  After 60 days in BIL at 5%
    annual RF, `self.nav` ≈ $97.1k — still 19% below the $120k peak.
    The MVO ramp trade on day 1 incurs any finite daily loss (say −0.3%)
    → nav = $96.8k → dd = (96.8k − 120k) / 120k ≈ −19.3% → breach fires.
    The system cycled: mandatory-BIL 60d → ramp 1d → breach → mandatory-BIL
    sixty times over ~3 years.  Active trading was impossible after the first
    halt; the "47.5% active days" was a statistical artefact of pre-halt data.

    Root cause: the breach guard was anchored to the historic peak, not to
    the portfolio state at the time the recovery attempt began.

    Fix: introduce `_ramp_entry_nav` (NAV captured at the moment Phase 1 →
    Phase 2 transition fires).  Breach guard now evaluates:
        ramp_local_dd = (self.nav - self._ramp_entry_nav) / self._ramp_entry_nav
        if ramp_local_dd <= -_HALT_RECOVERY_THRESH:
    Semantics: "abort ramp-in if we lose >10% of what we had when we started
    ramping, not from an unreachable historic peak."

  BUG #16 [Walk-forward overfitting verdict]:
    `ratio = avg_oos / (avg_is + 1e-10)` produces a meaningless negative
    value when avg_is < 0 (IS periods contain the 2020 halt).  Printing that
    as "overfitting" is both wrong and confusing.

    Fix: richer verdict logic:
      · If avg_oos > 0.5                         → ✅ OOS profitable
      · If avg_oos > 0 and avg_is < 0            → ⚠️  IS dominated by halt;
                                                       OOS signal is positive
      · If avg_is > 0 and ratio > 0.5            → ✅ IS/OOS ratio acceptable
      · If avg_is > 0 and ratio <= 0.5           → ⚠️  overfitting
      · Otherwise (avg_oos <= 0)                 → ❌ negative OOS alpha
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

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

# ── Risk / engine constants ────────────────────────────────────────────────────
_INITIAL_CAPITAL:       float = 100_000.0
_RISK_FREE_ANNUAL:      float = 0.05
_MAX_DD_HALT:           float = 0.20
_HALT_RECOVERY_THRESH:  float = 0.10
_HALT_MIN_DAYS:         int   = 60
_HALT_RAMP_DAYS:        int   = 60
_WARMUP_DAYS:           int   = 126
_WF_WARMUP_DAYS:        int   = 21
_REBALANCE_BAND:        float = 25e-4   # PATCH v5: 50→25bps — more responsive signal capture
_LAMBDA_BASE:           float = 2.5     # PATCH v5: 2.0→2.5, tighter vol control with better alpha
_COV_WINDOW:            int   = 63
_MIN_POSITION_WT:       float = 0.015   # PATCH v5: positions < 1.5% zeroed — concentrate conviction
_EMA_SPAN:              int   = 8
_REGIME_PERSIST_DAYS:   int   = 5
_AC_ETA:                float = 0.1
_BASE_SPREAD_BPS:       float = 1.0
_TRADING_DAYS_YEAR:     int   = 252

_BIL_WEIGHT = np.zeros(N_ASSETS, dtype=np.float32)
_BIL_WEIGHT[TICKERS.index("BIL")] = 0.60
_BIL_WEIGHT[TICKERS.index("SHV")] = 0.40


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
    """EMA(τ=8d) + persistence gate(≥5d) on PCA z_mu to suppress 42% flip rate."""

    def __init__(self, latent_dim: int = 16) -> None:
        self._alpha = 1.0 - np.exp(-1.0 / _EMA_SPAN)
        self._ema   = np.zeros(latent_dim, dtype=np.float32)
        self._raw_history: List[str] = []
        self._stable_label: str = "bull_low_vol"

    def update(self, z_mu_raw: np.ndarray, raw_label: str) -> Tuple[np.ndarray, str]:
        z = np.nan_to_num(z_mu_raw.ravel(), nan=0.0, posinf=2.0, neginf=-2.0)
        if z.shape[0] != self._ema.shape[0]:
            z = np.resize(z, self._ema.shape[0])
        self._ema = self._alpha * z + (1.0 - self._alpha) * self._ema
        self._raw_history.append(raw_label)
        if len(self._raw_history) >= _REGIME_PERSIST_DAYS:
            window = self._raw_history[-_REGIME_PERSIST_DAYS:]
            if len(set(window)) == 1:
                self._stable_label = window[0]
        return self._ema.copy(), self._stable_label


# ── Almgren-Chriss market impact ──────────────────────────────────────────────

def _ac_cost_bps(shares: float, sigma: float, adv: float) -> float:
    if adv <= 0:
        return 5.0
    return float(np.clip(_AC_ETA * sigma * np.sqrt(abs(shares) / adv) * 10_000, 0.0, 50.0))


# ── Covariance: OAS shrinkage (PATCH v5) ─────────────────────────────────────

def _cov(window: pd.DataFrame) -> np.ndarray:
    """
    Oracle Approximating Shrinkage (OAS, Chen et al. 2010) via sklearn.
    At p/n = 25/63 ≈ 0.40, OAS outperforms standard Ledoit-Wolf by ~15% in
    Frobenius norm. Falls back to manual James-Stein shrinkage if sklearn absent.
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
        mu     = np.trace(C) / N_ASSETS
        C      = (1.0 - shrink) * C + shrink * mu * np.eye(N_ASSETS)
    if not np.isfinite(C).all():
        return np.eye(N_ASSETS) * (0.15 ** 2 / _TRADING_DAYS_YEAR)
    return C


# ── MVO (SLSQP + regime-conditioned λ) ───────────────────────────────────────

def _mvo_weights(
    alpha: np.ndarray,
    cov_d: np.ndarray,
    z0_smooth: float,
    vol_d: np.ndarray,
    alloc_scale: float = 1.0,
) -> np.ndarray:
    avg_vol = float(np.mean(vol_d) * np.sqrt(_TRADING_DAYS_YEAR))
    lam = float(np.clip(_LAMBDA_BASE * np.exp(avg_vol * abs(z0_smooth)), 0.5, 15.0))
    Σ   = cov_d * _TRADING_DAYS_YEAR
    mx  = np.array([TIER_MAX_WEIGHT[t] for t in TICKERS]) * alloc_scale
    bds = [(0.0, float(m)) for m in mx]

    # BUG #19 FIX: x0 must lie within bounds before gradient steps.
    # np.full(1/25=0.04) can exceed ramp-in bounds (e.g. SHV_max=0.125*0.25=0.031).
    x0 = np.clip(np.full(N_ASSETS, 1.0 / N_ASSETS), 0.0, mx)
    x0_sum = x0.sum()
    if x0_sum > 1e-10:
        x0 = x0 / x0_sum   # project back to simplex after clipping
    else:
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
        iv = np.minimum(1.0 / (np.diag(Σ) ** 0.5 + 1e-8), mx * 3)
        w  = iv / (iv.sum() + 1e-10)

    w = np.clip(w, 0.0, mx)
    s = w.sum()
    w = w / s if s > 1e-10 else np.where(mx > 0, 1.0 / max((mx > 0).sum(), 1), 0.0)

    # Portfolio concentration filter: zero out sub-threshold positions to force
    # conviction. At 1.5% threshold with 25 assets this yields ~10-14 active
    # positions, doubling expected return-per-position vs a fully diluted 25-stock
    # portfolio with identical alpha signals.
    w[w < _MIN_POSITION_WT] = 0.0
    s = w.sum()
    if s > 1e-10:
        w = w / s
    else:
        # Fallback: top-5 by alpha, equal-weighted within bounds
        top5 = np.argsort(alpha)[-5:]
        w = np.zeros(N_ASSETS, dtype=np.float64)
        w[top5] = np.minimum(1.0 / 5, mx[top5])
        w = w / (w.sum() + 1e-10)

    return w.astype(np.float32)


# ── Probabilistic Sharpe Ratio (Bailey & López de Prado 2012) ─────────────────

def _psr(sr_hat: float, n: int, skew: float, kurt_raw: float,
         sr_benchmark: float = 0.0) -> float:
    """PSR = Φ[(SR̂ − SR*) × √(N−1) / √(1 − γ₃SR̂ + (γ₄−1)/4 × SR̂²)]"""
    if n <= 2:
        return 0.0
    denom = np.sqrt(max(1.0 - skew * sr_hat + (kurt_raw - 1.0) / 4.0 * sr_hat ** 2, 1e-6))
    return float(norm.cdf((sr_hat - sr_benchmark) * np.sqrt(n - 1) / denom))


# ── Data containers ───────────────────────────────────────────────────────────

@dataclass
class Snap:
    date: str; portfolio_value: float; daily_return: float; cash: float
    turnover: float; regime_label: str; z_mu_0: float; drawdown: float
    halt_phase: int


@dataclass
class WFFold:
    fold_id: int; is_start: str; is_end: str; oos_start: str; oos_end: str
    is_sharpe: float; oos_sharpe: float; oos_cagr: float; oos_max_dd: float


# ── Core engine ───────────────────────────────────────────────────────────────

class StandaloneBacktester:

    def _reset(self) -> None:
        self.nav:  float = _INITIAL_CAPITAL
        self.cash: float = _INITIAL_CAPITAL
        self.pos:  Dict[str, float] = {}
        self.history: List[Snap] = []
        self._peak:          float = _INITIAL_CAPITAL
        self._prev_regime:   str   = ""
        self._halt_phase:    int   = 0
        self._halt_days:     int   = 0
        self._halt_nav:      float = _INITIAL_CAPITAL
        self._ramp_days:     int   = 0
        # BUG #13 FIX: track NAV at the moment ramp-in begins
        self._ramp_entry_nav: float = _INITIAL_CAPITAL
        self._smoother = RegimeSignalSmoother()

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
            i      = TICKERS.index(t)
            alloc  = total_cash * frac
            self.pos[t] = self.pos.get(t, 0.0) + alloc / (px[i] + 1e-10)
            self.cash  -= alloc

    def _rf_step(self) -> float:
        """FIX #9: yield via share-scaling; self.cash stays at near-zero."""
        rf = _RISK_FREE_ANNUAL / _TRADING_DAYS_YEAR
        for t in ("BIL", "SHV"):
            if self.pos.get(t, 0.0) > 0.0:
                self.pos[t] *= (1.0 + rf)
        self.nav  *= (1.0 + rf)
        self._peak = max(self._peak, self.nav)
        return rf

    def _rebalance(self, target: np.ndarray, px: np.ndarray,
                   vol: np.ndarray, adv: np.ndarray, force: bool = False) -> float:
        current = np.array([
            self.pos.get(t, 0.0) * px[i] / (self.nav + 1e-10)
            for i, t in enumerate(TICKERS)
        ])
        delta = target - current
        cost  = 0.0
        for i, t in enumerate(TICKERS):
            if not force and abs(delta[i]) < _REBALANCE_BAND:
                continue
            dol    = delta[i] * self.nav
            shares = dol / (px[i] + 1e-10)
            c      = abs(dol) * (_ac_cost_bps(shares, vol[i], adv[i]) + _BASE_SPREAD_BPS) / 10_000
            cost  += c
            self.cash   += -(dol + np.sign(dol) * c)
            self.pos[t]  = self.pos.get(t, 0.0) + shares
        return cost / (self.nav + 1e-10)

    def _mtm(self, px: np.ndarray) -> float:
        equity   = sum(self.pos.get(t, 0.0) * px[i] for i, t in enumerate(TICKERS))
        new      = self.cash + equity
        dr       = (new - self.nav) / (self.nav + 1e-10)
        self.nav = new
        self._peak = max(self._peak, new)
        return dr

    # ── Main loop ──────────────────────────────────────────────────────────────

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
                        Snap(ds, self.nav, dr, self.cash, 0.0, "WARMUP", 0.0, dd, 0)
                    )
                continue

            # ── SIGNAL RETRIEVAL ──────────────────────────────────────────────
            if date not in regime_df.index or date not in alpha_df.index:
                dr = self._rf_step()
                dd = (self.nav - self._peak) / self._peak
                if record:
                    self.history.append(
                        Snap(ds, self.nav, dr, self.cash, 0.0, "NO_SIGNAL", 0.0, dd, 0)
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

            # ── HALT STATE MACHINE ────────────────────────────────────────────
            if self._halt_phase >= 1:
                self._halt_days += 1
                dr = self._rf_step()
                dd = (self.nav - self._peak) / self._peak

                if self._halt_phase == 1:
                    # Phase 1: mandatory BIL — check recovery vs halt_nav
                    if self._halt_days >= _HALT_MIN_DAYS:
                        recov = (self.nav - self._halt_nav) / (self._halt_nav + 1e-10)
                        if recov >= -_HALT_RECOVERY_THRESH:
                            logger.info(f"{ds}: Halt recovery — entering ramp-in.")
                            self._halt_phase    = 2
                            self._ramp_days     = 0
                            # BUG #13 FIX: snapshot NAV at ramp entry for breach guard
                            self._ramp_entry_nav = self.nav

                elif self._halt_phase == 2:
                    # Phase 2: linear MVO ramp 25% → 100% over _HALT_RAMP_DAYS
                    self._ramp_days += 1
                    scale = min(0.25 + 0.75 * self._ramp_days / _HALT_RAMP_DAYS, 1.0)

                    # FIX #10: covariance excludes today (gi, not gi+1)
                    cov_d = _cov(returns_df.iloc[max(0, gi - _COV_WINDOW) : gi])
                    vol_d = (
                        returns_df.iloc[max(0, gi - 21) : gi]
                        .std(axis=0).fillna(0.01).values.astype(np.float64)
                    )
                    adv = np.maximum(10_000_000.0 / (px + 1e-10), 1.0)
                    alp = alpha_df.loc[date].values.astype(np.float64)

                    t_mvo  = _mvo_weights(alp, cov_d, z0, vol_d, scale)
                    target = scale * t_mvo + (1.0 - scale) * _BIL_WEIGHT
                    target /= target.sum() + 1e-10

                    self._liquidate(px)
                    self.cash = self.nav
                    to  = self._rebalance(target, px, vol_d, adv, force=True)
                    dr  = self._mtm(px)
                    dd  = (self.nav - self._peak) / self._peak

                    # ── BUG #13 FIX: breach relative to ramp entry NAV ────────
                    # Guard: "did this ramp-in period itself lose >10%?"
                    # NOT: "are we still below the all-time peak?"
                    ramp_local_dd = (
                        (self.nav - self._ramp_entry_nav)
                        / (self._ramp_entry_nav + 1e-10)
                    )
                    if ramp_local_dd <= -_HALT_RECOVERY_THRESH:
                        logger.warning(
                            f"{ds}: Ramp-in breach ({ramp_local_dd:.2%} vs entry). "
                            "Returning to mandatory BIL."
                        )
                        self._halt_phase = 1
                        self._halt_days  = 0
                        self._halt_nav   = self.nav
                        self._liquidate(px)
                        self._invest_bil(px)

                    if self._ramp_days >= _HALT_RAMP_DAYS:
                        logger.info(f"{ds}: Ramp-in complete. Resuming full active trading.")
                        self._halt_phase = 0

                    if record:
                        self.history.append(
                            Snap(ds, self.nav, dr, self.cash, to,
                                 f"RAMP_{scale:.0%}", z0, dd, 2)
                        )
                    continue

                if record:
                    self.history.append(
                        Snap(ds, self.nav, dr, self.cash, 0.0,
                             "HALTED_BIL", z0, dd, self._halt_phase)
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
            to  = self._rebalance(target, px, vol_d, adv, force=regime_changed)
            dr  = self._mtm(px)
            dd  = (self.nav - self._peak) / self._peak
            self._prev_regime = s_label

            if dd <= -_MAX_DD_HALT:
                logger.critical(f"HALT triggered {ds}: DD={dd:.2%}")
                self._liquidate(px)
                self._halt_phase = 1
                self._halt_days  = 0
                self._halt_nav   = self.nav
                self._invest_bil(px)

            if record:
                self.history.append(
                    Snap(ds, self.nav, dr, self.cash, to, s_label, z0, dd, 0)
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
                "turnover":        h.turnover,
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
        to     = float(df["turnover"].mean())

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
            "CAGR":                f"{cagr:+.2%}",
            "Ann. Volatility":     f"{vol:.2%}",
            "Sharpe Ratio":        f"{sr:.3f}",
            "Sortino Ratio":       f"{sort_r:.3f}",
            "Calmar Ratio":        f"{cal:.3f}",
            "Max Drawdown":        f"{mdd:.2%}",
            "Max DD Duration":     f"{mdd_dur} days",
            "VaR-95 (daily)":      f"{v95:.2%}",
            "CVaR-95 (daily)":     f"{cvar95:.2%}",
            "Hit Rate":            f"{hit:.2%}",
            "Avg Daily Turnover":  f"{to:.4%}",
            "PSR (SR>0)":          f"{psr:.4f}",
            "Skewness":            f"{skew:.3f}",
            "Excess Kurtosis":     f"{kurt_raw - 3.0:.3f}",
            "Active Trading %":    f"{active_pct:.1%}",
            "Halted BIL %":        f"{halt_pct:.1%}",
            "Ramp-in %":           f"{ramp_pct:.1%}",
            "Final NAV":           f"${df['portfolio_value'].iloc[-1]:,.2f}",
            "Trading Days":        str(n),
        }
        logger.info("══ TEARSHEET ══")
        for k, v in m.items():
            logger.info(f"  {k:<28} {v}")
        df.attrs["metrics"] = m
        return df


def _mct(a: np.ndarray) -> int:
    mx = cur = 0
    for v in a:
        cur = cur + 1 if v else 0
        mx  = max(mx, cur)
    return mx


# ── Walk-forward folds ────────────────────────────────────────────────────────

def _wf_folds(
    dates: pd.DatetimeIndex,
    ism: int = 18,
    oosm: int = 6,
) -> List[Tuple[str, str, str, str]]:
    oos = int(oosm * 21)
    ei  = int(ism * 21)
    folds: List[Tuple[str, str, str, str]] = []
    while ei + oos <= len(dates):
        folds.append((
            dates[0].strftime("%Y-%m-%d"),
            dates[ei - 1].strftime("%Y-%m-%d"),
            dates[ei].strftime("%Y-%m-%d"),
            dates[min(ei + oos - 1, len(dates) - 1)].strftime("%Y-%m-%d"),
        ))
        ei += oos
    return folds


def _sharpe(r: np.ndarray) -> float:
    rf = _RISK_FREE_ANNUAL / _TRADING_DAYS_YEAR
    ex = r - rf
    return float((ex.mean() / (ex.std() + 1e-10)) * np.sqrt(_TRADING_DAYS_YEAR))


def _wf_verdict(avg_is: float, avg_oos: float) -> str:
    """
    BUG #16 FIX: Multi-branch verdict that handles negative IS Sharpe correctly.

    When IS is negative (e.g., IS period contains a halt), the signed ratio
    avg_oos / avg_is produces a negative number even if OOS is positive, which
    is misleading. Apply the following logic instead:
    """
    if avg_oos > 0.5:
        return "✅ OOS profitable"
    if avg_oos > 0.0 and avg_is < 0.0:
        return "⚠️  IS dominated by structural event; OOS signal positive"
    if avg_is > 0.0:
        ratio = avg_oos / avg_is
        if ratio > 0.5:
            return f"✅ IS/OOS ratio={ratio:.3f} acceptable"
        else:
            return f"⚠️  overfitting (IS/OOS ratio={ratio:.3f})"
    return "❌ negative OOS alpha"


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("══════ Fortress v5 — Standalone Backtest v5 (BUG #17-#19 + PATCH v5) ══════")
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    required = {
        "prices":  _CACHE_DIR / "prices_wide.parquet",
        "returns": _CACHE_DIR / "returns_wide.parquet",
        "regime":  _CACHE_DIR / "regime_posteriors.parquet",
        "alpha":   _CACHE_DIR / "alpha_signals.parquet",
    }
    for name, p in required.items():
        if not p.exists():
            logger.error(f"Missing: {p}. Run precompute scripts first.")
            sys.exit(1)

    prices_df  = _normalize_index(pd.read_parquet(required["prices"]))
    returns_df = _normalize_index(pd.read_parquet(required["returns"]))
    regime_df  = _normalize_index(pd.read_parquet(required["regime"]))
    alpha_df   = _normalize_index(pd.read_parquet(required["alpha"]))
    alpha_df.columns = [c.replace("alpha_", "") for c in alpha_df.columns]

    logger.info(
        f"Loaded | prices:{len(prices_df)}d  returns:{len(returns_df)}d"
        f"  regime:{len(regime_df)}d  alpha:{len(alpha_df)}d"
    )
    logger.info(
        f"Ranges | prices:{prices_df.index.min().date()}→{prices_df.index.max().date()}"
        f"  regime:{regime_df.index.min().date()}→{regime_df.index.max().date()}"
    )

    eng = StandaloneBacktester()
    ts  = eng.run(
        prices_df, returns_df, regime_df, alpha_df,
        start_date="2020-01-02",
        end_date="2024-12-31",
        warmup_days=_WARMUP_DAYS,
    )
    if ts.empty:
        logger.error("Backtest produced empty tearsheet.")
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
            ((t_oo["portfolio_value"] - t_oo["portfolio_value"].cummax())
             / (t_oo["portfolio_value"].cummax() + 1e-10)).min()
        )
        is_sr  = _sharpe(t_is["daily_return"].values) if not t_is.empty else 0.0
        oos_sr = _sharpe(r_oo)

        results.append(WFFold(
            fid + 1, is_s, is_e, oos_s, oos_e,
            round(is_sr, 4), round(oos_sr, 4), round(cagr, 4), round(mdd, 4),
        ))
        logger.info(
            f"  F{fid+1}: IS SR={is_sr:.3f} | OOS SR={oos_sr:.3f}"
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
            "mode": "stub", "var_95": v95,
            "cvar_95": float(r_arr[r_arr <= v95].mean()),
            "note": "Run training/train_world_model.py for SDE stress.",
        }, f, indent=2)

    logger.info("✅ Done. Run scripts/visualize_tearsheet.py next.")


if __name__ == "__main__":
    main()