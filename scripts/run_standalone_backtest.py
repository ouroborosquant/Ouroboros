"""
FORTRESS v5 — run_standalone_backtest.py  [PATCH v2]
Path: scripts/run_standalone_backtest.py

BUGS FIXED vs v1:
  BUG #1: Halted portfolio earned zero return for 1,029 days.
          portfolio_value was never updated during halt — _mark_to_market on an
          all-cash position returns zero equity delta. Cash must compound at RF.
          Fix: self.portfolio_value *= (1 + rf_daily) each halted day.

  BUG #2: Regime signal (z_mu[0]) is binary ±2.0 with 42.4% 2-day reversals.
          PCA normalises z_mu to ‖z_mu‖=2 → any single return reversal flips sign.
          Fix: EMA smoothing (τ=8 days) on the raw z_mu vector + persistence gate
               (new regime label accepted only if it persists ≥5 consecutive days).

  BUG #3: _halted = True was permanent — no re-entry logic.
          Fix: Three-phase recovery:
            Phase 1 (mandatory BIL, ≥60 days):   100% BIL/SHV, earn RF.
            Phase 2 (ramp-in, 60 days):           MVO at 25%→100% allocation.
            Phase 3 (active):                      normal operation resumes.
          Guard: if drawdown breaches -10% during ramp, return to Phase 1.

  BUG #4: No warmup period — covariance estimated on 5 rows, strategy trades day 1.
          Fix: _WARMUP_DAYS = 126. Portfolio sits 60% BIL / 40% SHV for the first
               126 trading days earning RF, building covariance/regime history.

  BUG #5: Rebalancing fired on only 18.2% of days — 25bps band suppressed
          regime-driven rotations while allowing noise-driven micro-trades.
          Fix: Dual-threshold rebalancing:
               - BAND: 50bps absolute weight delta (replaces 25bps)
               - REGIME: unconditionally rebalance when the stable regime label changes.
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

# ── Risk parameters ────────────────────────────────────────────────────────────
_INITIAL_CAPITAL:       float = 100_000.0
_RISK_FREE_ANNUAL:      float = 0.05
_MAX_DD_HALT:           float = 0.20    # trigger halt at -20% from peak NAV
_HALT_RECOVERY_THRESH:  float = 0.10   # re-entry allowed when DD < -10% from halt NAV
_HALT_MIN_DAYS:         int   = 60      # mandatory BIL period before re-entry assessment
_HALT_RAMP_DAYS:        int   = 60      # linear scale-in from 25% → 100% over N days
_WARMUP_DAYS:           int   = 126     # FIX #4: no live trading for first 126 days
_REBALANCE_BAND:        float = 50e-4   # FIX #5: 50bps dead-band (was 25bps)
_LAMBDA_BASE:           float = 2.0
_COV_WINDOW:            int   = 63
_EMA_SPAN:              int   = 8       # FIX #2: EMA smoothing span for z_mu
_REGIME_PERSIST_DAYS:   int   = 5       # FIX #2: min days before regime label changes
_AC_ETA:                float = 0.1
_BASE_SPREAD_BPS:       float = 1.0
_TRADING_DAYS_YEAR:     int   = 252

# Default safe-haven allocation (60% BIL + 40% SHV)
_BIL_WEIGHT = np.zeros(N_ASSETS, dtype=np.float32)
_BIL_WEIGHT[TICKERS.index("BIL")] = 0.60
_BIL_WEIGHT[TICKERS.index("SHV")] = 0.40


# ── FIX #2: Regime signal smoother ────────────────────────────────────────────

class RegimeSignalSmoother:
    """
    Two-stage de-noiser for the PCA-derived z_mu signal.

    The PCA surrogate normalises each z_mu to unit ball of radius 2, meaning a
    single-day return flip causes z_mu[0] to switch from +2 to -2. The raw
    regime_posteriors.parquet contains this binary noise.

    Stage 1 — EMA on the continuous z_mu vector (span=_EMA_SPAN days).
              Smooths the ±2 binary signal into a continuous value in [-2, 2]
              that varies at the pace of actual market regime changes (~weeks).

    Stage 2 — Persistence gate on the discrete label.
              Accepts a new regime label only if it has been the raw label for
              _REGIME_PERSIST_DAYS consecutive days. Single-day label reversals
              (42.4% of v1 active days) are suppressed.
    """

    def __init__(self, latent_dim: int = 16) -> None:
        self._alpha = 1.0 - np.exp(-1.0 / _EMA_SPAN)
        self._ema   = np.zeros(latent_dim, dtype=np.float32)
        self._raw_history: List[str] = []
        self._stable_label: str = "bull_low_vol"

    def update(self, z_mu_raw: np.ndarray, raw_label: str) -> Tuple[np.ndarray, str]:
        # Stage 1: EMA update
        self._ema = self._alpha * z_mu_raw + (1.0 - self._alpha) * self._ema

        # Stage 2: Persistence gate
        self._raw_history.append(raw_label)
        if len(self._raw_history) >= _REGIME_PERSIST_DAYS:
            window = self._raw_history[-_REGIME_PERSIST_DAYS:]
            if len(set(window)) == 1:
                self._stable_label = window[0]

        return self._ema.copy(), self._stable_label


# ── Almgren-Chriss ────────────────────────────────────────────────────────────

def _ac_cost_bps(shares: float, sigma: float, adv: float) -> float:
    if adv <= 0:
        return 5.0
    return float(np.clip(_AC_ETA * sigma * np.sqrt(abs(shares) / adv) * 10_000, 0.0, 50.0))


# ── Covariance (Ledoit-Wolf shrinkage) ────────────────────────────────────────

def _cov(window: pd.DataFrame, shrink: float = 0.1) -> np.ndarray:
    C = window.fillna(0.0).cov().values
    if not np.isfinite(C).all():
        return np.eye(N_ASSETS) * (0.15 ** 2 / _TRADING_DAYS_YEAR)
    I = np.trace(C) / N_ASSETS * np.eye(N_ASSETS)
    return (1 - shrink) * C + shrink * I


# ── MVO solver ────────────────────────────────────────────────────────────────

def _mvo_weights(alpha: np.ndarray, cov_d: np.ndarray,
                 z0_smooth: float, vol_d: np.ndarray,
                 alloc_scale: float = 1.0) -> np.ndarray:
    """
    Long-only MVO with regime-conditioned risk aversion.
    alloc_scale ∈ [0, 1]: fraction of capital in risky assets (rest goes BIL).
    Solved by SLSQP with Ledoit-Wolf shrinkage on Σ.
    """
    avg_vol = float(np.mean(vol_d) * np.sqrt(_TRADING_DAYS_YEAR))
    lam = float(np.clip(_LAMBDA_BASE * np.exp(avg_vol * abs(z0_smooth)), 0.5, 15.0))
    Σ   = cov_d * _TRADING_DAYS_YEAR

    mx  = np.array([TIER_MAX_WEIGHT[t] for t in TICKERS]) * alloc_scale
    bds = [(0.0, float(m)) for m in mx]

    res = sco.minimize(
        fun=lambda w: 0.5 * lam * w @ Σ @ w - alpha @ w,
        jac=lambda w: lam * Σ @ w - alpha,
        x0=np.full(N_ASSETS, 1.0 / N_ASSETS),
        method="SLSQP",
        bounds=bds,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"ftol": 1e-9, "maxiter": 500},
    )
    w = res.x if res.success else (
        # Inverse-vol fallback
        lambda iv: iv / iv.sum())(np.minimum(1.0 / (np.diag(Σ) ** 0.5 + 1e-8), mx * 3))
    w = np.clip(w, 0.0, mx)
    w /= w.sum() + 1e-10
    return w.astype(np.float32)


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
        self.nav: float  = _INITIAL_CAPITAL
        self.cash: float = _INITIAL_CAPITAL
        self.pos: Dict[str, float] = {}
        self.history: List[Snap] = []
        self._peak: float = _INITIAL_CAPITAL
        self._prev_regime: str = ""
        # Halt state machine
        self._halt_phase:    int   = 0   # 0=active, 1=mandatory BIL, 2=ramp, 3=re-entered
        self._halt_days:     int   = 0
        self._halt_nav:      float = _INITIAL_CAPITAL
        self._ramp_days:     int   = 0
        self._smoother = RegimeSignalSmoother()

    def _liquidate(self, px: np.ndarray) -> None:
        for t, sh in list(self.pos.items()):
            self.cash += sh * px[TICKERS.index(t)]
            self.pos[t] = 0.0
        self.nav = self.cash

    def _invest_bil(self, px: np.ndarray) -> None:
        """FIX #1/#3: Move all capital into 60% BIL + 40% SHV to earn RF going forward."""
        self._liquidate(px)
        for t, frac in [("BIL", 0.60), ("SHV", 0.40)]:
            i = TICKERS.index(t)
            shares = self.cash * frac / (px[i] + 1e-10)
            self.pos[t] = shares
            self.cash  -= self.cash * frac

    def _rf_step(self) -> float:
        """
        FIX #1: Compound portfolio at daily risk-free rate.
        Called every day the portfolio is in safe-haven mode.
        The BIL/SHV holdings are re-valued at the compounded rate rather than
        mark-to-market (since synthetic prices don't have the right yield built in).
        """
        rf = _RISK_FREE_ANNUAL / _TRADING_DAYS_YEAR
        delta = self.nav * rf
        self.nav  += delta
        self.cash += delta
        self._peak = max(self._peak, self.nav)
        return rf

    def _rebalance(self, target: np.ndarray, px: np.ndarray,
                   vol: np.ndarray, adv: np.ndarray,
                   force: bool = False) -> float:
        """FIX #5: Dual-threshold. force=True bypasses the dead-band (regime change)."""
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
            self.cash += -(dol + np.sign(dol) * c)
            self.pos[t] = self.pos.get(t, 0.0) + shares
        return cost / (self.nav + 1e-10)

    def _mtm(self, px: np.ndarray) -> float:
        equity = sum(self.pos.get(t, 0.0) * px[i] for i, t in enumerate(TICKERS))
        new    = self.cash + equity
        dr     = (new - self.nav) / (self.nav + 1e-10)
        self.nav   = new
        self._peak = max(self._peak, new)
        return dr

    # ── Main loop ──────────────────────────────────────────────────────────────

    def run(self, prices_df: pd.DataFrame, returns_df: pd.DataFrame,
            regime_df: pd.DataFrame, alpha_df: pd.DataFrame,
            start_date: str, end_date: str) -> pd.DataFrame:

        self._reset()
        mask   = (prices_df.index >= start_date) & (prices_df.index <= end_date)
        dates  = prices_df.index[mask]
        px_df  = prices_df.loc[mask]
        n      = len(dates)
        g0     = returns_df.index.get_loc(dates[0])  # global start idx in returns_df

        logger.info(f"Backtest: {start_date} → {end_date} ({n} days)")

        for li, date in enumerate(dates):
            ds   = date.strftime("%Y-%m-%d")
            px   = px_df.loc[date].values.astype(np.float64)
            gi   = g0 + li  # global index

            # ── WARMUP (FIX #4) ───────────────────────────────────────────────
            if li < _WARMUP_DAYS:
                if li == 0:
                    self._invest_bil(px)
                dr = self._rf_step()
                dd = (self.nav - self._peak) / self._peak
                self.history.append(Snap(ds, self.nav, dr, self.cash, 0.0,
                                         "WARMUP", 0.0, dd, 0))
                continue

            # ── SIGNAL RETRIEVAL ──────────────────────────────────────────────
            if date not in regime_df.index or date not in alpha_df.index:
                dr = self._rf_step()
                dd = (self.nav - self._peak) / self._peak
                self.history.append(Snap(ds, self.nav, dr, self.cash, 0.0,
                                         "NO_SIGNAL", 0.0, dd, 0))
                continue

            # Parse z_mu — stored as Python list or JSON string
            raw_z = regime_df.loc[date, "z_mu"]
            if isinstance(raw_z, str):
                raw_z = json.loads(raw_z)
            z_raw = np.array(raw_z, dtype=np.float32)
            z_smooth, s_label = self._smoother.update(z_raw, str(regime_df.loc[date, "regime_label"]))
            z0 = float(z_smooth[0])

            # ── HALT PHASES (FIX #1 + #3) ────────────────────────────────────
            if self._halt_phase >= 1:
                self._halt_days += 1
                dr = self._rf_step()  # FIX #1: always earn RF in halt
                dd = (self.nav - self._peak) / self._peak

                if self._halt_phase == 1:
                    # Check for transition to ramp
                    if self._halt_days >= _HALT_MIN_DAYS:
                        recov = (self.nav - self._halt_nav) / self._halt_nav
                        if recov >= -_HALT_RECOVERY_THRESH:
                            logger.info(f"{ds}: Halt recovery — entering ramp-in phase.")
                            self._halt_phase = 2
                            self._ramp_days  = 0

                elif self._halt_phase == 2:
                    # Ramp-in: partial MVO allocation
                    self._ramp_days += 1
                    scale = min(0.25 + 0.75 * self._ramp_days / _HALT_RAMP_DAYS, 1.0)

                    cov_d = _cov(returns_df.iloc[max(0, gi - _COV_WINDOW + 1): gi + 1])
                    vol_d = returns_df.iloc[max(0, gi - 21 + 1): gi + 1]\
                                      .std(axis=0).fillna(0.01).values.astype(np.float64)
                    adv   = np.maximum(10_000_000 / (px + 1e-10), 1.0)
                    alp   = alpha_df.loc[date].values.astype(np.float64)

                    t_mvo = _mvo_weights(alp, cov_d, z0, vol_d, scale)
                    # Blend: scale × MVO + (1-scale) × BIL
                    target = scale * t_mvo + (1.0 - scale) * _BIL_WEIGHT
                    target /= target.sum() + 1e-10

                    self._liquidate(px)
                    self.cash = self.nav
                    to  = self._rebalance(target, px, vol_d, adv, force=True)
                    dr  = self._mtm(px)
                    dd  = (self.nav - self._peak) / self._peak

                    # Guard: re-breach during ramp → back to mandatory BIL
                    if dd <= -_HALT_RECOVERY_THRESH:
                        logger.warning(f"{ds}: Ramp-in drawdown breach. Returning to mandatory BIL.")
                        self._halt_phase = 1
                        self._halt_days  = 0
                        self._halt_nav   = self.nav
                        self._liquidate(px)
                        self._invest_bil(px)
                        self.cash = 0.0

                    if self._ramp_days >= _HALT_RAMP_DAYS:
                        logger.info(f"{ds}: Ramp-in complete. Resuming full active trading.")
                        self._halt_phase = 0

                    self.history.append(Snap(ds, self.nav, dr, self.cash, to,
                                             f"RAMP_{scale:.0%}", z0, dd, 2))
                    continue

                self.history.append(Snap(ds, self.nav, dr, self.cash, 0.0,
                                         "HALTED_BIL", z0, dd, self._halt_phase))
                continue

            # ── ACTIVE TRADING ────────────────────────────────────────────────
            cov_d = _cov(returns_df.iloc[max(0, gi - _COV_WINDOW + 1): gi + 1])
            vol_d = returns_df.iloc[max(0, gi - 21 + 1): gi + 1]\
                              .std(axis=0).fillna(0.01).values.astype(np.float64)
            adv   = np.maximum(10_000_000 / (px + 1e-10), 1.0)
            alp   = alpha_df.loc[date].values.astype(np.float64)

            target = _mvo_weights(alp, cov_d, z0, vol_d)

            # FIX #5: regime change → unconditional rebalance
            regime_changed = (s_label != self._prev_regime)
            to  = self._rebalance(target, px, vol_d, adv, force=regime_changed)
            dr  = self._mtm(px)
            dd  = (self.nav - self._peak) / self._peak
            self._prev_regime = s_label

            # ── Drawdown halt trigger ──────────────────────────────────────────
            if dd <= -_MAX_DD_HALT:
                logger.critical(f"HALT triggered {ds}: DD={dd:.2%}")
                self._liquidate(px)
                self._halt_phase = 1
                self._halt_days  = 0
                self._halt_nav   = self.nav
                self._invest_bil(px)  # FIX #1: goes into BIL immediately
                self.cash = 0.0

            self.history.append(Snap(ds, self.nav, dr, self.cash, to,
                                     s_label, z0, dd, 0))

        return self._build_tearsheet()

    # ── Tearsheet ─────────────────────────────────────────────────────────────

    def _build_tearsheet(self) -> pd.DataFrame:
        df = pd.DataFrame([{
            "date":            h.date,
            "portfolio_value": h.portfolio_value,
            "daily_return":    h.daily_return,
            "cash":            h.cash,
            "turnover":        h.turnover,
            "regime_label":    h.regime_label,
            "z_mu_0":          h.z_mu_0,
            "drawdown":        h.drawdown,
            "halt_phase":      h.halt_phase,
        } for h in self.history])

        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        if len(df) < 10:
            return df

        rf   = _RISK_FREE_ANNUAL / _TRADING_DAYS_YEAR
        r    = df["daily_return"].values
        ex   = r - rf
        n    = len(r)

        tot  = df["portfolio_value"].iloc[-1] / _INITIAL_CAPITAL - 1.0
        cagr = (1 + tot) ** (_TRADING_DAYS_YEAR / n) - 1.0
        vol  = r.std() * np.sqrt(_TRADING_DAYS_YEAR)
        sr   = float((ex.mean() / (ex.std() + 1e-10)) * np.sqrt(_TRADING_DAYS_YEAR))
        dn   = ex[ex < 0]
        sort = float((ex.mean() / (dn.std() + 1e-10)) * np.sqrt(_TRADING_DAYS_YEAR))
        cv   = df["portfolio_value"]
        rm   = cv.cummax()
        dd_s = (cv - rm) / rm
        mdd  = float(dd_s.min())
        cal  = cagr / (abs(mdd) + 1e-10)
        mdd_dur = _mct((dd_s < 0).values)
        v95  = float(np.percentile(r, 5))
        cv95 = float(r[r <= v95].mean())
        hit  = float((r > rf).mean())
        to   = float(df["turnover"].mean())
        dsr  = float((sr - stats.norm.ppf(1 - 1.0 / n) *
                      np.sqrt((1 + 0.5 * sr**2) / (n - 1))) /
                     (np.sqrt((1 + 0.5 * sr**2) / (n - 1)) + 1e-10))

        m = {
            "CAGR":               f"{cagr:+.2%}",
            "Ann. Volatility":    f"{vol:.2%}",
            "Sharpe Ratio":       f"{sr:.3f}",
            "Sortino Ratio":      f"{sort:.3f}",
            "Calmar Ratio":       f"{cal:.3f}",
            "Max Drawdown":       f"{mdd:.2%}",
            "Max DD Duration":    f"{mdd_dur} days",
            "VaR-95 (daily)":     f"{v95:.2%}",
            "CVaR-95 (daily)":    f"{cv95:.2%}",
            "Hit Rate":           f"{hit:.2%}",
            "Avg Daily Turnover": f"{to:.4%}",
            "DSR":                f"{dsr:.4f}",
            "Skewness":           f"{stats.skew(r):.3f}",
            "Excess Kurtosis":    f"{stats.kurtosis(r):.3f}",
            "Final NAV":          f"${df['portfolio_value'].iloc[-1]:,.2f}",
            "Trading Days":       str(n),
        }
        logger.info("══ TEARSHEET ══")
        for k, v in m.items():
            logger.info(f"  {k:<26} {v}")
        df.attrs["metrics"] = m
        return df


def _mct(a: np.ndarray) -> int:
    mx = cur = 0
    for v in a:
        cur = cur + 1 if v else 0
        mx = max(mx, cur)
    return mx


# ── Walk-forward ──────────────────────────────────────────────────────────────

def _wf_folds(dates: pd.DatetimeIndex,
              ism: int = 18, oosm: int = 6) -> List[Tuple]:
    mi, oos, ei, folds = int(ism * 21), int(oosm * 21), int(ism * 21), []
    while ei + oos <= len(dates):
        folds.append((dates[0].strftime("%Y-%m-%d"), dates[ei-1].strftime("%Y-%m-%d"),
                      dates[ei].strftime("%Y-%m-%d"), dates[ei+oos-1].strftime("%Y-%m-%d")))
        ei += oos
    return folds


def _sharpe(r: np.ndarray) -> float:
    rf = _RISK_FREE_ANNUAL / _TRADING_DAYS_YEAR
    ex = r - rf
    return float((ex.mean() / (ex.std() + 1e-10)) * np.sqrt(_TRADING_DAYS_YEAR))


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("══════ Fortress v5 — Standalone Backtest v2 (5 bugs patched) ══════")
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for n, p in {"prices":  _CACHE_DIR/"prices_wide.parquet",
                  "returns": _CACHE_DIR/"returns_wide.parquet",
                  "regime":  _CACHE_DIR/"regime_posteriors.parquet",
                  "alpha":   _CACHE_DIR/"alpha_signals.parquet"}.items():
        if not p.exists():
            logger.error(f"Missing: {p}. Run precompute scripts first."); sys.exit(1)

    prices_df  = pd.read_parquet(_CACHE_DIR / "prices_wide.parquet")
    returns_df = pd.read_parquet(_CACHE_DIR / "returns_wide.parquet")
    regime_df  = pd.read_parquet(_CACHE_DIR / "regime_posteriors.parquet")
    alpha_df   = pd.read_parquet(_CACHE_DIR / "alpha_signals.parquet")

    for df in [prices_df, returns_df, regime_df, alpha_df]:
        df.index = pd.to_datetime(df.index)
    alpha_df.columns = [c.replace("alpha_", "") for c in alpha_df.columns]

    eng = StandaloneBacktester()
    ts  = eng.run(prices_df, returns_df, regime_df, alpha_df,
                  start_date="2020-01-02", end_date="2024-12-31")
    ts.to_csv(_TEARSHEET)
    logger.info(f"✅ Tearsheet → {_TEARSHEET}")

    # Walk-forward
    logger.info("Running walk-forward validation...")
    common  = prices_df.index.intersection(alpha_df.index)
    wf_mask = (common >= "2019-01-02") & (common <= "2024-12-31")
    folds   = _wf_folds(common[wf_mask])
    results = []
    for fid, (is_s, is_e, oos_s, oos_e) in enumerate(folds):
        t_is = StandaloneBacktester().run(prices_df, returns_df, regime_df, alpha_df, is_s, is_e)
        t_oo = StandaloneBacktester().run(prices_df, returns_df, regime_df, alpha_df, oos_s, oos_e)
        if t_oo.empty or len(t_oo) < 5:
            continue
        r_oo = t_oo["daily_return"].values
        n_oo = len(r_oo)
        tot  = t_oo["portfolio_value"].iloc[-1] / _INITIAL_CAPITAL - 1.0
        cagr = (1 + tot) ** (_TRADING_DAYS_YEAR / n_oo) - 1.0
        mdd  = float(((t_oo["portfolio_value"] - t_oo["portfolio_value"].cummax())
                       / t_oo["portfolio_value"].cummax()).min())
        results.append(WFFold(fid+1, is_s, is_e, oos_s, oos_e,
                               round(_sharpe(t_is["daily_return"].values), 4),
                               round(_sharpe(r_oo), 4), round(cagr, 4), round(mdd, 4)))
        logger.info(f"  F{fid+1}: IS SR={results[-1].is_sharpe:.3f} | "
                    f"OOS SR={results[-1].oos_sharpe:.3f} CAGR={cagr:.2%} MaxDD={mdd:.2%}")

    pd.DataFrame([vars(r) for r in results]).to_csv(_WF_FOLDS, index=False)
    if results:
        avg_oos = np.mean([f.oos_sharpe for f in results])
        avg_is  = np.mean([f.is_sharpe  for f in results])
        logger.info(f"WF: avg IS={avg_is:.3f} | avg OOS={avg_oos:.3f} | "
                    f"ratio={avg_oos/(avg_is+1e-10):.3f} "
                    f"({'✅ acceptable' if avg_oos/max(avg_is,0.01) > 0.5 else '⚠️ overfitting'})")

    # Stress stub
    r_arr = ts["daily_return"].values
    v95   = float(np.percentile(r_arr, 5))
    with open(_STRESS_OUT, "w") as f:
        json.dump({"mode": "stub", "var_95": v95,
                   "cvar_95": float(r_arr[r_arr <= v95].mean()),
                   "note": "Run training/train_world_model.py for real SDE stress."}, f, indent=2)

    logger.info(f"✅ Done. Run scripts/visualize_tearsheet.py next.")


if __name__ == "__main__":
    main()