"""
FORTRESS v5 — run_standalone_backtest.py  [PATCH v3]
Path: scripts/run_standalone_backtest.py

═══════════════════════════════════════════════════════════════════════════════
BUGS FIXED vs v2
═══════════════════════════════════════════════════════════════════════════════

  BUG #6 [CRITICAL — ROOT CAUSE of "Trading Days: 126"]:
    start_date="2020-01-02" was used as the first date loaded into `dates`.
    Warmup consumed li < 126 → only 126 warmup snaps entered `history`.
    The active trading code never ran because the entire recorded window was
    inside the warmup guard.

    Fix: `run()` now extends the data window *backwards* by `warmup_days`
    trading days before `start_date`. Warmup runs silently on pre-start_date
    data (no snaps recorded). Active trading and tearsheet recording begin
    exactly at `start_date`. The DataFrame index used for the extension is
    computed via `searchsorted` — zero string-comparison fragility.

  BUG #7 [CRITICAL — silent TypeError at li=126]:
    `returns_df.index.get_loc(date)` returns a `slice` object when the
    DatetimeIndex contains duplicates (the audit detected 1510 such duplicates
    in the stale cache). `gi = g0 + li` where `g0 = slice(...)` raises
    TypeError at the first post-warmup use of `gi` (_cov, vol_d slicing).
    The exception propagated out of `run()`, aborting the loop with only
    warmup snaps in `self.history`. `_build_tearsheet` then produced the
    126-row artifact.

    Fix: `_normalize_index()` strips timezone, deduplicates (keep='last'),
    and sorts all four DataFrames immediately after loading. After this,
    `get_loc()` always returns an int. Additionally switched to
    `searchsorted(side='left')` for `g0` which is O(log N) and slice-safe.

  BUG #8 [_invest_bil cash arithmetic]:
    Sequential cash deduction: first loop allocated 60% from full cash,
    reducing it to 40%. Second loop allocated 40% of the *remaining* 40%
    (= 16% of original). Net deployment: 76%. 24% left as uninvested cash.

    Fix: capture `total_cash` before the loop; allocate each tranche from
    the frozen total.

  BUG #9 [_rf_step double-counts cash]:
    `self.cash += delta` was called with `self.cash ≈ 0` (after _invest_bil
    deployed capital). RF delta was added to a near-zero cash balance,
    inflating it to ~2.6% of NAV over 126 warmup days. On day 127,
    `_mtm` computed `new = inflated_cash + market_valued_equity`, and
    `new ≠ self.nav` (RF-compounded) produced a phantom daily return on the
    first active day, corrupting all downstream metrics.

    Fix: scale BIL/SHV *position share counts* by (1+rf) each day, which
    embeds the yield directly into position size. `self.nav` is mirrored from
    `nav *= (1+rf)`. `self.cash` is not touched. The first `_mtm` call now
    returns a genuinely observed market return.

  BUG #10 [covariance look-ahead bias]:
    `returns_df.iloc[gi - W + 1 : gi + 1]` included today's realised return
    (index gi) in the covariance used to size today's trade. This is a subtle
    but genuine point-in-time violation — at trade construction time (open of
    day gi), today's return is unknowable.

    Fix: `returns_df.iloc[max(0, gi - W) : gi]` — W days through yesterday,
    exclusive of today.

  BUG #11 [walk-forward OOS warmup = OOS period]:
    Each 6-month OOS fold is ≈126 trading days. A fresh StandaloneBacktester
    with `_WARMUP_DAYS=126` spent *all* OOS days in warmup. Every OOS fold
    returned SR=0.000, CAGR=5.13%, MaxDD=0.00% — pure RF compounding.
    IS/OOS comparison was measuring warmup vs. warmup.

    Fix: `_WF_WARMUP_DAYS = 21` (one calendar month) for WF calls. The OOS
    runner now receives `warmup_days=_WF_WARMUP_DAYS`. Combined with the BUG
    #6 fix, warmup runs on the 21 trading days *before* oos_start, leaving
    ≈105 active-trading days per OOS period.

  BUG #12 [DSR / Probabilistic Sharpe Ratio formula]:
    v2 formula: `(SR - ppf) / sqrt(...)` was dimensionally wrong and omitted
    the higher-moment corrections (skewness, excess kurtosis) that are
    essential for non-Gaussian return distributions.

    Fix: Probabilistic Sharpe Ratio from Bailey & López de Prado (2012):
      Z = (SR̂ - SR*) × √(N-1) / √(1 - γ₃×SR̂ + (γ₄-1)/4 × SR̂²)
      PSR(SR*) = Φ[Z]
    γ₃ = skewness of daily returns, γ₄ = raw (non-excess) kurtosis.
    SR* = 0.0 (test whether strategy SR > zero).
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

# ── Risk / engine constants ────────────────────────────────────────────────────
_INITIAL_CAPITAL:       float = 100_000.0
_RISK_FREE_ANNUAL:      float = 0.05
_MAX_DD_HALT:           float = 0.20
_HALT_RECOVERY_THRESH:  float = 0.10
_HALT_MIN_DAYS:         int   = 60
_HALT_RAMP_DAYS:        int   = 60
_WARMUP_DAYS:           int   = 126    # main backtest warmup (pre-start_date)
_WF_WARMUP_DAYS:        int   = 21    # BUG #11 FIX: WF folds use 1-month warmup
_REBALANCE_BAND:        float = 50e-4
_LAMBDA_BASE:           float = 2.0
_COV_WINDOW:            int   = 63
_EMA_SPAN:              int   = 8
_REGIME_PERSIST_DAYS:   int   = 5
_AC_ETA:                float = 0.1
_BASE_SPREAD_BPS:       float = 1.0
_TRADING_DAYS_YEAR:     int   = 252

# Default safe-haven weight vector (60% BIL + 40% SHV)
_BIL_WEIGHT = np.zeros(N_ASSETS, dtype=np.float32)
_BIL_WEIGHT[TICKERS.index("BIL")] = 0.60
_BIL_WEIGHT[TICKERS.index("SHV")] = 0.40


# ── Index normalisation (BUG #7 FIX) ─────────────────────────────────────────

def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip timezone, deduplicate (keep last), sort ascending.

    Critical invariant: get_loc() on a unique DatetimeIndex always returns int,
    never a slice or boolean array. searchsorted() is also safe only on sorted
    unique indices.
    """
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df = df.copy()
    df.index = idx
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


# ── FIX #2 (retained): Regime signal smoother ────────────────────────────────

class RegimeSignalSmoother:
    """
    EMA (τ=8d) + persistence gate (≥5 consecutive days) on PCA z_mu.

    Prevents the 42.4% single-day label reversal rate observed in v1, where
    any one-day return flip drives z_mu[0] across the ±2 boundary.
    """

    def __init__(self, latent_dim: int = 16) -> None:
        self._alpha = 1.0 - np.exp(-1.0 / _EMA_SPAN)
        self._ema   = np.zeros(latent_dim, dtype=np.float32)
        self._raw_history: List[str] = []
        self._stable_label: str = "bull_low_vol"

    def update(self, z_mu_raw: np.ndarray, raw_label: str) -> Tuple[np.ndarray, str]:
        z_safe = np.nan_to_num(z_mu_raw.reshape(-1), nan=0.0, posinf=2.0, neginf=-2.0)
        # Resize to match internal EMA dimension if latent_dim differs
        if z_safe.shape[0] != self._ema.shape[0]:
            z_safe = np.resize(z_safe, self._ema.shape[0])
        self._ema = self._alpha * z_safe + (1.0 - self._alpha) * self._ema

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


# ── Ledoit-Wolf shrinkage covariance ──────────────────────────────────────────

def _cov(window: pd.DataFrame, shrink: float = 0.1) -> np.ndarray:
    C = window.fillna(0.0).cov().values
    if not np.isfinite(C).all():
        return np.eye(N_ASSETS) * (0.15 ** 2 / _TRADING_DAYS_YEAR)
    I = np.trace(C) / N_ASSETS * np.eye(N_ASSETS)
    return (1 - shrink) * C + shrink * I


# ── MVO (SLSQP, regime-conditioned risk aversion) ─────────────────────────────

def _mvo_weights(
    alpha: np.ndarray,
    cov_d: np.ndarray,
    z0_smooth: float,
    vol_d: np.ndarray,
    alloc_scale: float = 1.0,
) -> np.ndarray:
    """
    Long-only MVO with regime-conditioned λ.
    alloc_scale ∈ [0,1] blends result toward safe-haven during ramp-in.
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
    if res.success:
        w = res.x
    else:
        # Inverse-vol fallback: closed-form, always feasible
        iv = np.minimum(1.0 / (np.diag(Σ) ** 0.5 + 1e-8), mx * 3)
        w  = iv / (iv.sum() + 1e-10)

    w = np.clip(w, 0.0, mx)
    s = w.sum()
    if s < 1e-10:
        # Degenerate: all weights clipped to zero — fall back to equal weight
        w = np.where(mx > 0, 1.0 / max((mx > 0).sum(), 1), 0.0)
    else:
        w /= s
    return w.astype(np.float32)


# ── Probabilistic Sharpe Ratio (BUG #12 FIX) ─────────────────────────────────

def _psr(
    sr_hat: float,
    n: int,
    skew: float,
    kurt_raw: float,
    sr_benchmark: float = 0.0,
) -> float:
    """
    PSR = Φ[ (SR̂ − SR*) × √(N−1) / √(1 − γ₃×SR̂ + (γ₄−1)/4 × SR̂²) ]

    Bailey & López de Prado (2012), "The Sharpe Ratio Efficient Frontier".
    γ₃ = skewness of daily returns.
    γ₄ = raw (non-excess) kurtosis (kurt_raw = excess_kurtosis + 3).

    Denominator clipped at 1e-6 to prevent division by zero for degenerate
    return distributions (e.g. constant returns during warmup-only periods).
    """
    if n <= 2:
        return 0.0
    denom_sq = 1.0 - skew * sr_hat + (kurt_raw - 1.0) / 4.0 * sr_hat ** 2
    denom    = np.sqrt(max(denom_sq, 1e-6))
    z        = (sr_hat - sr_benchmark) * np.sqrt(n - 1) / denom
    return float(norm.cdf(z))


# ── Data containers ───────────────────────────────────────────────────────────

@dataclass
class Snap:
    date: str
    portfolio_value: float
    daily_return: float
    cash: float
    turnover: float
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


# ── Core backtesting engine ───────────────────────────────────────────────────

class StandaloneBacktester:

    def _reset(self) -> None:
        self.nav:  float = _INITIAL_CAPITAL
        self.cash: float = _INITIAL_CAPITAL
        self.pos:  Dict[str, float] = {}
        self.history: List[Snap] = []
        self._peak:        float = _INITIAL_CAPITAL
        self._prev_regime: str   = ""
        self._halt_phase:  int   = 0  # 0=active, 1=mandatory BIL, 2=ramp
        self._halt_days:   int   = 0
        self._halt_nav:    float = _INITIAL_CAPITAL
        self._ramp_days:   int   = 0
        self._smoother = RegimeSignalSmoother()

    def _liquidate(self, px: np.ndarray) -> None:
        """Convert all positions to cash at current prices."""
        for t in list(self.pos.keys()):
            sh = self.pos.pop(t, 0.0)
            if sh != 0.0:
                self.cash += sh * px[TICKERS.index(t)]
        self.nav = self.cash

    def _invest_bil(self, px: np.ndarray) -> None:
        """
        BUG #8 FIX: Allocate from captured total_cash.

        v2 bug: sequential deduction from self.cash meant:
          BIL loop: cash -= cash*0.60  → cash = 0.40×C
          SHV loop: cash -= cash*0.40  → cash = 0.24×C
        Result: 24% capital stranded as uninvested cash.

        Fix: freeze total_cash before the loop; deduct fixed fractions of that.
        """
        self._liquidate(px)
        total_cash = self.cash
        for t, frac in [("BIL", 0.60), ("SHV", 0.40)]:
            i      = TICKERS.index(t)
            alloc  = total_cash * frac
            shares = alloc / (px[i] + 1e-10)
            self.pos[t] = self.pos.get(t, 0.0) + shares
            self.cash  -= alloc
        # self.cash ≈ 0 (floating-point epsilon only)

    def _rf_step(self) -> float:
        """
        BUG #9 FIX: Yield-via-share-scaling instead of cash injection.

        v2 bug: `self.cash += self.nav * rf` added RF to a near-zero cash
        balance, inflating it to ~2.6% of NAV over 126 warmup days.
        First _mtm call saw new = inflated_cash + market_equity ≠ nav,
        generating a phantom return that corrupted all downstream metrics.

        Fix: scale BIL/SHV share counts by (1+rf). nav is mirrored via
        nav *= (1+rf). self.cash stays at its true near-zero value.
        """
        rf = _RISK_FREE_ANNUAL / _TRADING_DAYS_YEAR
        for t in ("BIL", "SHV"):
            if self.pos.get(t, 0.0) > 0.0:
                self.pos[t] *= (1.0 + rf)
        self.nav  *= (1.0 + rf)
        self._peak = max(self._peak, self.nav)
        return rf

    def _rebalance(
        self,
        target: np.ndarray,
        px: np.ndarray,
        vol: np.ndarray,
        adv: np.ndarray,
        force: bool = False,
    ) -> float:
        """
        FIX #5 (retained): Dual-threshold rebalancing.
        force=True: bypass dead-band (regime-change event).
        force=False: 50bps absolute weight delta required to trade.
        """
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
            c      = (
                abs(dol)
                * (_ac_cost_bps(shares, vol[i], adv[i]) + _BASE_SPREAD_BPS)
                / 10_000
            )
            cost += c
            self.cash   += -(dol + np.sign(dol) * c)
            self.pos[t]  = self.pos.get(t, 0.0) + shares
        return cost / (self.nav + 1e-10)

    def _mtm(self, px: np.ndarray) -> float:
        """Mark portfolio to market; return daily P&L rate."""
        equity = sum(
            self.pos.get(t, 0.0) * px[i] for i, t in enumerate(TICKERS)
        )
        new         = self.cash + equity
        dr          = (new - self.nav) / (self.nav + 1e-10)
        self.nav    = new
        self._peak  = max(self._peak, new)
        return dr

    # ── Main simulation loop ───────────────────────────────────────────────────

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
        BUG #6 FIX: Extend data window backwards by `warmup_days` trading days
        before `start_date`. Warmup runs on pre-start_date data (no history
        recorded). Recording and active trading begin exactly at `start_date`.

        This guarantees `Trading Days = (end_date − start_date)` regardless
        of `warmup_days`, as long as prices_df contains sufficient pre-history.
        """
        self._reset()

        ts_start = pd.Timestamp(start_date)
        ts_end   = pd.Timestamp(end_date)

        # ── BUG #6 FIX: resolve pre-start warmup window ───────────────────────
        # searchsorted is safe on sorted, deduplicated (normalised) index.
        start_pos  = int(prices_df.index.searchsorted(ts_start, side="left"))
        ext_pos    = max(0, start_pos - warmup_days)
        ext_start  = prices_df.index[ext_pos]
        # Count how many days in [ext_start, start_date) we have for warmup.
        # If pre-history is shorter than warmup_days, extra warmup occurs inside
        # the recording window and those snaps are tagged "WARMUP".
        pre_warmup_days = start_pos - ext_pos  # may be < warmup_days

        mask   = (prices_df.index >= ext_start) & (prices_df.index <= ts_end)
        dates  = prices_df.index[mask]
        px_df  = prices_df.loc[mask]
        n_ext  = len(dates)

        # BUG #7 FIX: positional anchor in returns_df (safe after normalisation)
        g0 = int(returns_df.index.searchsorted(ext_start, side="left"))

        logger.info(
            f"Backtest window: {ext_start.date()} → {end_date}"
            f" ({n_ext} total days, warmup={warmup_days}d,"
            f" recording from {start_date})"
        )

        for li, date in enumerate(dates):
            ds      = date.strftime("%Y-%m-%d")
            px      = px_df.loc[date].values.astype(np.float64)
            gi      = g0 + li  # positional index in returns_df (int, always safe)
            record  = date >= ts_start  # only write history post-start_date

            # ── WARMUP PHASE ──────────────────────────────────────────────────
            # Guard: li < warmup_days covers the pre-history window + any
            # spillover if pre-history was shorter than warmup_days.
            if li < warmup_days:
                if li == 0:
                    self._invest_bil(px)
                dr = self._rf_step()
                dd = (self.nav - self._peak) / self._peak
                if record:
                    # Rare: pre-history insufficient; warmup bleeds into recording
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

            # Parse z_mu with NaN guard (early PCA window may produce NaN rows)
            raw_z = regime_df.loc[date, "z_mu"]
            if isinstance(raw_z, str):
                raw_z = json.loads(raw_z)
            z_raw = np.asarray(raw_z, dtype=np.float32).ravel()
            z_smooth, s_label = self._smoother.update(
                z_raw, str(regime_df.loc[date, "regime_label"])
            )
            z0 = float(z_smooth[0])

            # ── HALT STATE MACHINE (BUG #1 + #3 retained) ────────────────────
            if self._halt_phase >= 1:
                self._halt_days += 1
                dr = self._rf_step()
                dd = (self.nav - self._peak) / self._peak

                if self._halt_phase == 1:
                    if self._halt_days >= _HALT_MIN_DAYS:
                        recov = (self.nav - self._halt_nav) / (self._halt_nav + 1e-10)
                        if recov >= -_HALT_RECOVERY_THRESH:
                            logger.info(f"{ds}: Halt recovery — entering ramp-in.")
                            self._halt_phase = 2
                            self._ramp_days  = 0

                elif self._halt_phase == 2:
                    self._ramp_days += 1
                    scale = min(0.25 + 0.75 * self._ramp_days / _HALT_RAMP_DAYS, 1.0)

                    # BUG #10 FIX: covariance excludes today (gi, not gi+1)
                    cov_d = _cov(returns_df.iloc[max(0, gi - _COV_WINDOW) : gi])
                    vol_d = (
                        returns_df.iloc[max(0, gi - 21) : gi]
                        .std(axis=0)
                        .fillna(0.01)
                        .values.astype(np.float64)
                    )
                    adv   = np.maximum(10_000_000.0 / (px + 1e-10), 1.0)
                    alp   = alpha_df.loc[date].values.astype(np.float64)

                    t_mvo  = _mvo_weights(alp, cov_d, z0, vol_d, scale)
                    target = scale * t_mvo + (1.0 - scale) * _BIL_WEIGHT
                    target /= target.sum() + 1e-10

                    self._liquidate(px)
                    self.cash = self.nav
                    to  = self._rebalance(target, px, vol_d, adv, force=True)
                    dr  = self._mtm(px)
                    dd  = (self.nav - self._peak) / self._peak

                    if dd <= -_HALT_RECOVERY_THRESH:
                        logger.warning(
                            f"{ds}: Ramp-in breach DD={dd:.2%}. Back to mandatory BIL."
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
            # BUG #10 FIX: W-day window ending at yesterday's close (gi exclusive).
            cov_d = _cov(returns_df.iloc[max(0, gi - _COV_WINDOW) : gi])
            vol_d = (
                returns_df.iloc[max(0, gi - 21) : gi]
                .std(axis=0)
                .fillna(0.01)
                .values.astype(np.float64)
            )
            adv = np.maximum(10_000_000.0 / (px + 1e-10), 1.0)
            alp = alpha_df.loc[date].values.astype(np.float64)

            target = _mvo_weights(alp, cov_d, z0, vol_d)

            # FIX #5 (retained): unconditional rebalance on regime change
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

        # Sanity check: if all history is WARMUP we likely have a data problem
        active_days = sum(
            1 for h in self.history if h.regime_label not in ("WARMUP", "NO_SIGNAL")
        )
        if self.history and active_days == 0:
            logger.warning(
                "⚠️  All recorded days are WARMUP/NO_SIGNAL. "
                "Check that regime_df and alpha_df cover the recording window. "
                f"Recording window: {start_date} → {end_date}. "
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

        dn   = ex[ex < 0]
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
        v95     = float(np.percentile(r, 5))
        cvar95  = float(r[r <= v95].mean()) if (r <= v95).any() else v95
        hit     = float((r > rf).mean())
        to      = float(df["turnover"].mean())

        # BUG #12 FIX: Bailey & López de Prado (2012) PSR
        skew = float(stats.skew(r))
        kurt_raw = float(stats.kurtosis(r, fisher=False))   # raw kurtosis (excess+3)
        psr  = _psr(sr, n, skew, kurt_raw)

        active_pct = float((
            ~df["regime_label"].isin(["WARMUP", "NO_SIGNAL", "HALTED_BIL"])
        ).mean())

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
            "Active Trading Days": f"{active_pct:.1%}",
            "Final NAV":           f"${df['portfolio_value'].iloc[-1]:,.2f}",
            "Trading Days":        str(n),
        }
        logger.info("══ TEARSHEET ══")
        for k, v in m.items():
            logger.info(f"  {k:<28} {v}")
        df.attrs["metrics"] = m
        return df


# ── Max consecutive True (drawdown duration) ─────────────────────────────────

def _mct(a: np.ndarray) -> int:
    mx = cur = 0
    for v in a:
        cur = cur + 1 if v else 0
        mx  = max(mx, cur)
    return mx


# ── Walk-forward fold generator ───────────────────────────────────────────────

def _wf_folds(
    dates: pd.DatetimeIndex,
    ism: int = 18,
    oosm: int = 6,
) -> List[Tuple[str, str, str, str]]:
    """
    Expanding IS / rolling OOS walk-forward.
    IS always anchored to dates[0] (prevents IS Sharpe shrinkage artefacts).
    OOS steps forward by oosm months each fold.
    """
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


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("══════ Fortress v5 — Standalone Backtest v3 (7 bugs patched) ══════")
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    required = {
        "prices":  _CACHE_DIR / "prices_wide.parquet",
        "returns": _CACHE_DIR / "returns_wide.parquet",
        "regime":  _CACHE_DIR / "regime_posteriors.parquet",
        "alpha":   _CACHE_DIR / "alpha_signals.parquet",
    }
    for name, p in required.items():
        if not p.exists():
            logger.error(f"Missing cache file '{name}': {p}. Run precompute scripts first.")
            sys.exit(1)

    # ── Load & normalise (BUG #7 FIX applied here) ───────────────────────────
    prices_df  = _normalize_index(pd.read_parquet(required["prices"]))
    returns_df = _normalize_index(pd.read_parquet(required["returns"]))
    regime_df  = _normalize_index(pd.read_parquet(required["regime"]))
    alpha_df   = _normalize_index(pd.read_parquet(required["alpha"]))

    # Strip alpha_ prefix from column names
    alpha_df.columns = [c.replace("alpha_", "") for c in alpha_df.columns]

    logger.info(
        f"Loaded data | prices: {len(prices_df)}d | returns: {len(returns_df)}d"
        f" | regime: {len(regime_df)}d | alpha: {len(alpha_df)}d"
    )
    logger.info(
        f"Date ranges | prices: {prices_df.index.min().date()} → {prices_df.index.max().date()}"
        f" | regime: {regime_df.index.min().date()} → {regime_df.index.max().date()}"
    )

    # ── Main backtest (2020-01-02 → 2024-12-31, warmup on 2019 data) ─────────
    eng = StandaloneBacktester()
    ts  = eng.run(
        prices_df, returns_df, regime_df, alpha_df,
        start_date="2020-01-02",
        end_date="2024-12-31",
        warmup_days=_WARMUP_DAYS,
    )

    if ts.empty:
        logger.error("Backtest produced empty tearsheet. Aborting.")
        sys.exit(1)

    ts.to_csv(_TEARSHEET)
    logger.info(f"✅ Tearsheet → {_TEARSHEET} ({len(ts)} rows)")

    # ── Walk-forward validation ───────────────────────────────────────────────
    logger.info("Running walk-forward validation...")
    common   = prices_df.index.intersection(alpha_df.index)
    wf_mask  = (common >= "2019-01-02") & (common <= "2024-12-31")
    folds    = _wf_folds(common[wf_mask])
    results: List[WFFold] = []

    for fid, (is_s, is_e, oos_s, oos_e) in enumerate(folds):
        # BUG #11 FIX: use _WF_WARMUP_DAYS (21d) so OOS folds have active trading
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

        results.append(
            WFFold(
                fid + 1, is_s, is_e, oos_s, oos_e,
                round(is_sr, 4), round(oos_sr, 4),
                round(cagr, 4), round(mdd, 4),
            )
        )
        logger.info(
            f"  F{fid+1}: IS SR={is_sr:.3f} | "
            f"OOS SR={oos_sr:.3f}  CAGR={cagr:.2%}  MaxDD={mdd:.2%}"
        )

    pd.DataFrame([vars(r) for r in results]).to_csv(_WF_FOLDS, index=False)
    if results:
        avg_oos = np.mean([f.oos_sharpe for f in results])
        avg_is  = np.mean([f.is_sharpe  for f in results])
        ratio   = avg_oos / (avg_is + 1e-10)
        verdict = "✅ acceptable" if ratio > 0.5 else "⚠️  overfitting"
        logger.info(
            f"WF summary: avg IS={avg_is:.3f} | avg OOS={avg_oos:.3f}"
            f" | IS/OOS ratio={ratio:.3f} ({verdict})"
        )

    # ── Stress-test stub ─────────────────────────────────────────────────────
    r_arr = ts["daily_return"].values
    v95   = float(np.percentile(r_arr, 5))
    with open(_STRESS_OUT, "w") as f:
        json.dump(
            {
                "mode":    "stub",
                "var_95":  v95,
                "cvar_95": float(r_arr[r_arr <= v95].mean()),
                "note":    "Run training/train_world_model.py for SDE stress.",
            },
            f,
            indent=2,
        )

    logger.info("✅ Done. Run scripts/visualize_tearsheet.py next.")


if __name__ == "__main__":
    main()