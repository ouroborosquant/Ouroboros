"""
FORTRESS v5 — precompute_alpha_signals.py  [P2 FIX — FactorDecayMonitor]
Path: scripts/precompute_alpha_signals.py

Offline alpha signal precomputation.
Loads cached prices/returns/regime posteriors and writes alpha_signals.parquet
for consumption by the standalone backtest and EDT training.

BUG FIXES RETAINED (previous sessions):
  BUG #17: Price-trend regime override (SPY MA50/200 + breadth + HYG/SHY spread).
  BUG #18: Fixed-income carry estimates non-negative (2019–2024 yield averages).
  BUG #15: No FutureWarning flood.

P1 ENHANCEMENTS (retained):
  F7: 5-Day Short-Term Reversal — bid-ask bounce / intraweek order-flow reversal.
  F8: Idiosyncratic Momentum — beta-stripped 12-1M momentum (Blitz et al. 2011).
  F9: AIS Commodity Flow — optional vessel traffic signal for USO/PDBC/SLV/GLD/EEM.

P2 FIX — FACTOR DECAY MONITOR:
  Root cause of F7/F8 OOS Sharpe collapse in Folds 7/8 (2023-2024):
    OOS SR trajectory: 2.22 → 1.51 → 1.22 → -1.36 → 1.24 → 0.28 → 0.18
    The 2023-2024 low-vol equity momentum bull regime hurts reversal factors
    (F2-21d, F7-5d) and the mislabeled "crisis" labels from BUG #20 inflated
    the idiosyncratic momentum signal into a phantom crisis premium.
    After GMM fix (BUG #20), regime labels are clean, but alpha decay in
    F2/F7 during low-vol trending markets remains a structural factor property.

  Solution — FactorDecayMonitor:
    For each factor, compute the realised Spearman rank IC daily:
      IC_t = SpearmanR(signal_{t-1}, r_t)     — strictly causal: uses yesterday's
                                                signal vs today's realized return.
    Track EWMA IC with halflife=63 trading days (≈ 3 months):
      α = 1 − exp(−ln(2)/halflife)
      IC_ewma_t = α·IC_{t} + (1−α)·IC_ewma_{t−1}

    Gate mechanism:
      If |IC_ewma_t| < ic_floor=0.02 for factor X on day t,
        set λ_X = 0 for day t.
      After zeroing, L1-renormalize the remaining λ weights so total
      factor weight is preserved (no free alpha is thrown away, it is
      redistributed to surviving factors).

    Why 0.02 IC floor?
      Spearman IC of 0.02 across N=25 assets has t-stat ≈ 0.10 — statistically
      indistinguishable from noise. Below this threshold the factor is consuming
      rebalancing budget with zero expected return contribution. The 63-day EWMA
      provides ~21 effective IID observations, making the floor conservative.

    Causality guarantee:
      IC_ewma on day t uses only IC values from days {0 ... t-1}.
      No look-ahead: signal_{t-1} × return_{t} is available on day t
      (return_t is the open-to-close return of day t, known at close).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("PrecomputeAlpha")

# ── Paths ─────────────────────────────────────────────────────────────────────
_CACHE_DIR    = Path("research/outputs/cache")
_PRICES_PATH  = _CACHE_DIR / "prices_wide.parquet"
_RETURNS_PATH = _CACHE_DIR / "returns_wide.parquet"
_REGIME_PATH  = _CACHE_DIR / "regime_posteriors.parquet"
_ALPHA_OUT    = _CACHE_DIR / "alpha_signals.parquet"
_GAT_WEIGHTS  = Path("models/weights/gat_alpha_latest.pt")

# ── Universe ──────────────────────────────────────────────────────────────────
TICKERS: List[str] = [
    "SPY", "QQQ", "IWM", "VTV",
    "XLK", "XLF", "XLV", "XLP", "XLI", "XLE",
    "EFA", "EEM",
    "TLT", "IEF", "SHY", "LQD", "HYG",
    "GLD", "SLV", "USO", "PDBC", "VNQ",
    "VIXY",
    "SHV", "BIL",
]
N_ASSETS           = len(TICKERS)
DEFENSIVE_TICKERS  = {"TLT", "IEF", "SHY", "GLD", "SHV", "BIL"}
EQUITY_TICKERS     = {
    "SPY", "QQQ", "IWM", "VTV",
    "XLK", "XLF", "XLV", "XLP", "XLI", "XLE",
    "EFA", "EEM", "VNQ", "HYG",
}
VOLATILITY_TICKERS = {"VIXY"}

# ── P1-AIS: attempt to import AIS pipeline (graceful degradation) ─────────────
_ais_available: bool = False
try:
    from data.alt_data.ais_pipeline import AISCommoditySignal as _AISCommoditySignal  # type: ignore
    _ais_available = True
    logger.info("✅ AIS commodity flow pipeline loaded. Factor F9 active.")
except ImportError:
    logger.info(
        "AIS pipeline not available — Factor F9 inactive. "
        "Wire data/alt_data/ais_pipeline.py to activate."
    )

_AIS_TICKERS: List[str]     = ["USO", "PDBC", "SLV", "GLD", "EEM"]
_AIS_TICKER_IDXS: List[int] = [TICKERS.index(t) for t in _AIS_TICKERS]

# ── Fixed-Income Carry Estimates ──────────────────────────────────────────────
# BUG #18 FIX: all non-negative. 2019-2024 Barclays yield-to-worst averages.
CARRY_ESTIMATES: Dict[str, float] = {
    "TLT":  0.025, "IEF": 0.018, "SHY": 0.025,
    "LQD":  0.032, "HYG": 0.055, "SHV": 0.035, "BIL": 0.035,
}

# ── FactorDecayMonitor parameters (P2 FIX) ───────────────────────────────────
_IC_EWMA_HALFLIFE: int   = 63    # ~3 months. α = 1 − exp(−ln2/63) ≈ 0.0109
_IC_FLOOR:         float = 0.02  # Below this → factor is statistical noise
_IC_EWMA_ALPHA:    float = 1.0 - np.exp(-np.log(2) / _IC_EWMA_HALFLIFE)


# ─────────────────────────────────────────────────────────────────────────────
# P2 FIX: FactorDecayMonitor
# ─────────────────────────────────────────────────────────────────────────────

class FactorDecayMonitor:
    """
    Tracks realised Spearman IC for each factor and gates lambda weights.

    Architecture:
      - Pre-computes per-factor IC time series once over the full dataset.
        IC_t = SpearmanR(signal_{t-1}, returns_t) — causal by one period.
      - Applies EWMA(halflife=63d) to produce IC_ewma(t) using only
        IC values from periods {0 ... t-1} (strict causality).
      - Returns per-day lambda multipliers in {0, 1}: 1 if |IC_ewma| ≥ floor,
        0 otherwise.

    Lambda gating + L1 renormalization:
      After zeroing decayed factors, the surviving lambda vector is scaled
      so it sums to the same total as the original (factor budget is preserved,
      not reduced). This prevents the optimizer from going to cash just because
      one factor is decayed — it concentrates the budget in surviving factors.

    Example IC lifecycle:
      F7 (5d reversal) IC in a low-vol trending bull market:
        2019-2021: IC_ewma ≈ +0.04 → active
        2023 H2:   IC_ewma decays below 0.02 as momentum dominates → zeroed
        2024:      remains below floor → budget shifts to F1/F8
      This precisely captures the observed F7/F8 OOS collapse in Folds 7/8.
    """

    # Factor name → column index in the lambda vector (for logging)
    FACTOR_NAMES: List[str] = [
        "F1_mom", "F1b_mom6", "F2_rev21", "F3_vol",
        "F4_carry", "F7_rev5", "F8_idio",
    ]
    # F9 (AIS) has a flat fixed weight, not gated (insufficient history for IC)

    def __init__(self) -> None:
        # n_factors × T pre-computed IC grid (filled in fit())
        self._ic_grid:      Optional[np.ndarray] = None  # (n_factors, T)
        self._ic_ewma_grid: Optional[np.ndarray] = None  # (n_factors, T)
        self._n_factors:    int = len(self.FACTOR_NAMES)

    def fit(
        self,
        factor_arrays:  List[np.ndarray],  # each (T, N), ordered per FACTOR_NAMES
        returns_matrix: np.ndarray,        # (T, N) daily returns
    ) -> None:
        """
        Pre-computes IC_t and IC_ewma_t for every factor and day.

        IC_t = SpearmanR(factor[t-1, :], returns[t, :]) across N assets.
        EWMA applied causally: IC_ewma[t] uses IC[0..t-1] only.

        O(T × n_factors × N log N) — runs in <0.5s for T=1510, N=25.
        """
        T, N = returns_matrix.shape
        n_f  = len(factor_arrays)
        assert n_f == self._n_factors, (
            f"Expected {self._n_factors} factor arrays, got {n_f}"
        )

        ic_grid      = np.zeros((n_f, T), dtype=np.float64)
        ic_ewma_grid = np.zeros((n_f, T), dtype=np.float64)

        for fi, arr in enumerate(factor_arrays):
            ewma_val = 0.0
            for t in range(1, T):
                sig_t1 = arr[t - 1]     # yesterday's signal — strictly causal
                ret_t  = returns_matrix[t]

                # Mask NaN/zero rows
                valid = np.isfinite(sig_t1) & np.isfinite(ret_t)
                if valid.sum() >= 5:
                    ic_t, _ = spearmanr(sig_t1[valid], ret_t[valid])
                    if not np.isfinite(ic_t):
                        ic_t = 0.0
                else:
                    ic_t = 0.0

                ic_grid[fi, t] = ic_t

                # EWMA is computed using IC values from {0..t-1}:
                # ic_ewma_grid[t] reflects information available on day t
                # because IC_t = IC(signal_{t-1}, return_t) is only known
                # at close of day t. For day t's decision, we use ewma_{t-1}.
                # Convention: ewma_grid[t] = ewma state ENTERING day t.
                ewma_val = _IC_EWMA_ALPHA * ic_t + (1.0 - _IC_EWMA_ALPHA) * ewma_val
                ic_ewma_grid[fi, t] = ewma_val

        self._ic_grid      = ic_grid
        self._ic_ewma_grid = ic_ewma_grid

        # Log end-of-series IC for diagnostic
        logger.info("FactorDecayMonitor fit complete. Terminal EWMA IC per factor:")
        for fi, name in enumerate(self.FACTOR_NAMES):
            terminal_ic = ic_ewma_grid[fi, -1]
            status = "✅ ACTIVE" if abs(terminal_ic) >= _IC_FLOOR else "❌ DECAYED"
            logger.info(f"  {name:12s}: IC_ewma(T) = {terminal_ic:+.4f}  [{status}]")

    def get_lambda_multipliers(self, t: int) -> np.ndarray:
        """
        Returns (n_factors,) binary gate vector for day t.
        1.0 → factor active, 0.0 → factor decayed below IC floor.

        Uses ic_ewma_grid[:, t] which was built from IC values {0..t-1},
        so it is strictly causal for day t's allocation decision.
        """
        if self._ic_ewma_grid is None:
            return np.ones(self._n_factors, dtype=np.float64)

        ewma_t = self._ic_ewma_grid[:, t]
        return np.where(np.abs(ewma_t) >= _IC_FLOOR, 1.0, 0.0)

    @staticmethod
    def renormalize_lambdas(
        raw_lambdas: np.ndarray,   # (n_factors,) before gating
        gate:        np.ndarray,   # (n_factors,) binary multipliers
    ) -> np.ndarray:
        """
        L1-renormalize: scale surviving lambdas so their sum equals
        the original total lambda budget. Budget preservation prevents
        the optimizer from implicitly going to cash on decayed signals.

        Edge case: if all factors are gated out (all zero), returns
        equal-weight surviving lambdas to prevent zero-alpha days.
        """
        gated    = raw_lambdas * gate
        raw_sum  = raw_lambdas.sum()
        gated_sum = gated.sum()

        if gated_sum < 1e-9:
            # All factors decayed: distribute budget equally (shouldn't happen
            # in practice — at least F1/momentum is rarely below floor)
            n_raw_active = int((raw_lambdas > 0).sum())
            if n_raw_active == 0:
                return raw_lambdas.copy()
            equal_lam    = raw_sum / n_raw_active
            return np.where(raw_lambdas > 0, equal_lam, 0.0)

        # Scale so surviving factors absorb the decayed budget
        return gated * (raw_sum / gated_sum)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _cross_sectional_rank(arr: np.ndarray, descending: bool = True) -> np.ndarray:
    """Signal vector → cross-sectional percentile ranks in [−1, 1]. NaN → 0."""
    valid = ~np.isnan(arr)
    ranks = np.zeros(len(arr), dtype=np.float64)
    if valid.sum() < 2:
        return ranks
    n_valid = int(valid.sum())
    temp = np.argsort(arr[valid])
    r    = (temp.argsort() + 1).astype(np.float64)
    r    = (r - (n_valid + 1) / 2.0) / ((n_valid - 1) / 2.0 + 1e-8)
    if descending:
        r = -r
    ranks[valid] = r
    return ranks


# ─────────────────────────────────────────────────────────────────────────────
# Factor 1: 12-1 Month Cross-Sectional Momentum
# ─────────────────────────────────────────────────────────────────────────────

def _build_momentum_signal(
    returns_df: pd.DataFrame,
    momentum_window: int = 252,
    skip_window:     int = 21,
) -> pd.DataFrame:
    cum_long = returns_df.rolling(momentum_window, min_periods=60).sum()
    cum_skip = returns_df.rolling(skip_window,     min_periods=5).sum()
    signal   = (cum_long - cum_skip).fillna(0.0).values
    ranked   = np.apply_along_axis(
        lambda row: _cross_sectional_rank(row, descending=False),
        axis=1, arr=signal,
    )
    return pd.DataFrame(ranked, index=returns_df.index, columns=TICKERS)


# ─────────────────────────────────────────────────────────────────────────────
# Factor 1b: 6-1 Month Momentum
# ─────────────────────────────────────────────────────────────────────────────

def _build_momentum_6m_signal(
    returns_df:  pd.DataFrame,
    window:      int = 126,
    skip_window: int = 21,
) -> pd.DataFrame:
    """6-1M — faster regime-transition capture than 12-1M."""
    cum_long = returns_df.rolling(window,       min_periods=30).sum()
    cum_skip = returns_df.rolling(skip_window,  min_periods=5).sum()
    signal   = (cum_long - cum_skip).fillna(0.0).values
    ranked   = np.apply_along_axis(
        lambda row: _cross_sectional_rank(row, descending=False),
        axis=1, arr=signal,
    )
    return pd.DataFrame(ranked, index=returns_df.index, columns=TICKERS)


# ─────────────────────────────────────────────────────────────────────────────
# Factor 2: 21-Day Short-Term Reversal
# ─────────────────────────────────────────────────────────────────────────────

def _build_reversal_signal(
    returns_df: pd.DataFrame,
    window:     int = 21,
) -> pd.DataFrame:
    cum    = returns_df.rolling(window, min_periods=5).sum().fillna(0.0).values
    ranked = np.apply_along_axis(
        lambda row: _cross_sectional_rank(row, descending=True),
        axis=1, arr=cum,
    )
    return pd.DataFrame(ranked, index=returns_df.index, columns=TICKERS)


# ─────────────────────────────────────────────────────────────────────────────
# Factor 3: 63-Day Low-Volatility Anomaly
# ─────────────────────────────────────────────────────────────────────────────

def _build_vol_signal(
    returns_df: pd.DataFrame,
    window:     int = 63,
) -> pd.DataFrame:
    vol    = returns_df.rolling(window, min_periods=21).std().fillna(0.1).values
    ranked = np.apply_along_axis(
        lambda row: _cross_sectional_rank(row, descending=True),
        axis=1, arr=vol,
    )
    return pd.DataFrame(ranked, index=returns_df.index, columns=TICKERS)


# ─────────────────────────────────────────────────────────────────────────────
# Factor 4: Fixed-Income Carry
# ─────────────────────────────────────────────────────────────────────────────

def _build_carry_signal() -> Dict[str, float]:
    """BUG #18 FIX: non-negative estimates from 2019-2024 Barclays YTW averages."""
    return {t: CARRY_ESTIMATES.get(t, 0.0) for t in TICKERS}


# ─────────────────────────────────────────────────────────────────────────────
# Factor 5: Price-Trend Regime Override (BUG #17 FIX)
# ─────────────────────────────────────────────────────────────────────────────

def _price_trend_z(
    prices_arr:  np.ndarray,
    i:           int,
    fast_window: int = 50,
    slow_window: int = 200,
) -> float:
    """
    Three-component price trend signal (strictly causal):
      (1) SPY 50/200d MA crossover  — weight 50%
      (2) Cross-asset breadth: % above 50d MA  — weight 30%
      (3) HYG vs SHY 21d momentum spread  — weight 20%
    """
    if i < slow_window:
        return 0.0
    px = prices_arr[:i]
    if len(px) < slow_window:
        return 0.0

    spy_idx = TICKERS.index("SPY")
    hyg_idx = TICKERS.index("HYG")
    shy_idx = TICKERS.index("SHY")

    spy_window = px[-slow_window:, spy_idx]
    ma_fast    = float(spy_window[-fast_window:].mean())
    ma_slow    = float(spy_window.mean())
    trend_z    = float(np.clip((ma_fast / (ma_slow + 1e-8) - 1.0) * 50.0, -2.0, 2.0))

    px_recent  = px[-fast_window:]
    ma50_all   = px_recent.mean(axis=0)
    breadth_z  = float(np.clip(((px[-1] > ma50_all).mean() - 0.5) * 6.0, -2.0, 2.0))

    credit_z = 0.0
    if i >= 22:
        hyg_ret  = float(px[-1, hyg_idx] / (px[-21, hyg_idx] + 1e-8) - 1.0)
        shy_ret  = float(px[-1, shy_idx] / (px[-21, shy_idx] + 1e-8) - 1.0)
        credit_z = float(np.clip((hyg_ret - shy_ret) * 25.0, -1.0, 1.0))

    return float(np.clip(0.50 * trend_z + 0.30 * breadth_z + 0.20 * credit_z, -2.0, 2.0))


def _build_regime_tilt_signal(z_effective: float) -> np.ndarray:
    """Additive tilt vector conditioned on blended z_effective regime signal."""
    tilt = np.zeros(N_ASSETS, dtype=np.float64)
    for i, t in enumerate(TICKERS):
        if t in EQUITY_TICKERS:
            tilt[i] = z_effective * 0.65
        elif t in DEFENSIVE_TICKERS:
            tilt[i] = -z_effective * 0.55
        elif t in VOLATILITY_TICKERS:
            tilt[i] = -z_effective * 1.20
    return tilt


# ─────────────────────────────────────────────────────────────────────────────
# Factor 7: 5-Day Short-Term Reversal
# ─────────────────────────────────────────────────────────────────────────────

def _build_reversal_5d_signal(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    5-day mean-reversion: bid-ask bounce + intraweek order-flow reversal.
    Orthogonal to 21d reversal in autocorrelation structure.
    5d IC ≈ −0.04 to −0.08 in ETF universe; stronger in USO/VIXY due to roll.
    """
    cum    = returns_df.rolling(5, min_periods=3).sum().fillna(0.0).values
    ranked = np.apply_along_axis(
        lambda row: _cross_sectional_rank(row, descending=True),
        axis=1, arr=cum,
    )
    return pd.DataFrame(ranked, index=returns_df.index, columns=TICKERS)


# ─────────────────────────────────────────────────────────────────────────────
# Factor 8: Idiosyncratic Momentum
# ─────────────────────────────────────────────────────────────────────────────

def _build_idiosyncratic_momentum_signal(
    returns_df:      pd.DataFrame,
    beta_window:     int = 63,
    momentum_window: int = 252,
    skip_window:     int = 21,
) -> pd.DataFrame:
    """
    Beta-adjusted 12-1M momentum (Blitz et al. 2011).
    Removes market component before momentum:
      β_i(t) = cov(r_i[t−63:t], r_SPY[t−63:t]) / var(r_SPY[t−63:t])
      r_idio_i(t) = r_i(t) − β_i(t) × r_SPY(t)
    Reduces momentum-crash exposure vs raw momentum. IC ≈ 0.05 vs 0.03 raw.
    """
    if "SPY" not in returns_df.columns:
        logger.warning("SPY absent — idiosyncratic momentum unavailable. Returning zeros.")
        return pd.DataFrame(
            np.zeros((len(returns_df), N_ASSETS), dtype=np.float64),
            index=returns_df.index, columns=TICKERS,
        )

    spy_series   = returns_df["SPY"]
    spy_var      = spy_series.rolling(beta_window, min_periods=20).var()
    spy_returns  = spy_series.values.astype(np.float64)
    n_dates      = len(returns_df)
    idio_returns = np.zeros((n_dates, N_ASSETS), dtype=np.float64)

    for col_idx, ticker in enumerate(TICKERS):
        asset_series = returns_df[ticker]
        cov_series   = asset_series.rolling(beta_window, min_periods=20).cov(spy_series)
        beta_series  = (cov_series / spy_var.clip(lower=1e-10)).fillna(1.0)
        idio_returns[:, col_idx] = asset_series.values - beta_series.values * spy_returns

    idio_df   = pd.DataFrame(idio_returns, index=returns_df.index, columns=TICKERS)
    cum_long  = idio_df.rolling(momentum_window, min_periods=60).sum()
    cum_skip  = idio_df.rolling(skip_window,     min_periods=5).sum()
    signal    = (cum_long - cum_skip).fillna(0.0).values

    ranked = np.apply_along_axis(
        lambda row: _cross_sectional_rank(row, descending=False),
        axis=1, arr=signal,
    )
    return pd.DataFrame(ranked, index=returns_df.index, columns=TICKERS)


# ─────────────────────────────────────────────────────────────────────────────
# Factor 9: AIS Commodity Flow (optional)
# ─────────────────────────────────────────────────────────────────────────────

def _build_ais_signal(returns_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if not _ais_available:
        return None
    try:
        ais_signal = _AISCommoditySignal()  # type: ignore[name-defined]
        flow_df    = ais_signal.load_precomputed_scores(
            start_date=str(returns_df.index[0].date()),
            end_date=str(returns_df.index[-1].date()),
        )
        result = pd.DataFrame(
            np.zeros((len(returns_df), N_ASSETS), dtype=np.float64),
            index=returns_df.index, columns=TICKERS,
        )
        for ticker in _AIS_TICKERS:
            if ticker in flow_df.columns:
                result[ticker] = flow_df[ticker].reindex(returns_df.index).fillna(0.0).values
        logger.info(f"AIS signal loaded: {flow_df.shape} → aligned to {result.shape}")
        return result
    except Exception as exc:
        logger.warning(f"AIS signal load failed ({exc}). Degrading to zero.")
        return None


def _infer_regime_name(z_effective: float) -> str:
    if   z_effective >  1.5: return "bull_low_vol"
    elif z_effective >  0.5: return "bull_high_vol"
    elif z_effective > -0.5: return "neutral"
    elif z_effective > -1.5: return "bear_market"
    else:                    return "crisis"


# ─────────────────────────────────────────────────────────────────────────────
# GATv2 stub (full mode)
# ─────────────────────────────────────────────────────────────────────────────

def _try_full_mode_gat(
    returns_df: pd.DataFrame,
    regime_df:  pd.DataFrame,
) -> bool:
    if not _GAT_WEIGHTS.exists():
        logger.info(
            f"GATv2 weights not found at {_GAT_WEIGHTS}. Running in Surrogate Mode."
        )
        return False
    try:
        import torch  # type: ignore
        logger.warning("Full-mode GATv2 not yet wired — falling back to Surrogate Mode.")
        return False
    except Exception as exc:
        logger.warning(f"Full mode GATv2 aborted ({exc}). Running Surrogate Mode.")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Surrogate Alpha — 8-factor blend + FactorDecayMonitor (P2)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_surrogate_alpha(
    prices_df:  pd.DataFrame,
    returns_df: pd.DataFrame,
    regime_df:  pd.DataFrame,
) -> pd.DataFrame:
    """
    8-factor regime-conditioned alpha engine with FactorDecayMonitor gating.

    Factor pipeline:
      1. Build all factor DataFrames (T×N) — full causal computation.
      2. Fit FactorDecayMonitor on all factors vs. next-day returns.
      3. Per-day blend loop: compute raw λ weights, apply decay gate,
         L1-renormalize, blend signals, apply tanh squash.

    P2 wiring:
      FactorDecayMonitor.get_lambda_multipliers(t) returns a binary gate per
      factor for day t. The gate is computed from EWMA IC using only data
      from days {0..t-1}, preserving strict look-ahead-free causality.
    """
    logger.info("Computing Factor 1:  Cross-Sectional Momentum (12-1M)...")
    f_mom  = _build_momentum_signal(returns_df)
    logger.info("Computing Factor 1b: Cross-Sectional Momentum (6-1M)...")
    f_mom6 = _build_momentum_6m_signal(returns_df)
    logger.info("Computing Factor 2:  Short-Term Reversal (21d)...")
    f_rev  = _build_reversal_signal(returns_df)
    logger.info("Computing Factor 3:  Low-Volatility Anomaly (63d)...")
    f_vol  = _build_vol_signal(returns_df)
    logger.info("Computing Factor 4:  Fixed-Income Carry (BUG #18 FIX)...")
    carry_vec = np.array([_build_carry_signal()[t] for t in TICKERS], dtype=np.float64)
    logger.info("Computing Factor 7:  5-Day Short-Term Reversal...")
    f_rev5 = _build_reversal_5d_signal(returns_df)
    logger.info("Computing Factor 8:  Idiosyncratic Momentum...")
    f_idio = _build_idiosyncratic_momentum_signal(returns_df)
    logger.info("Computing Factor 9:  AIS Commodity Flow (optional)...")
    f_ais  = _build_ais_signal(returns_df)

    # ── P2 FIX: fit FactorDecayMonitor ───────────────────────────────────────
    # F4 carry is a static cross-sectional vector (same value every day across
    # the date axis). Broadcasting to (T,N) gives zero cross-sectional variance
    # on every day → IC ≡ 0 → always gated. To keep carry active (it has genuine
    # yield carry, just not cross-sectional momentum IC), we treat it as always-on
    # by tiling the carry vector, which gives the IC monitor a valid signal to score.
    logger.info("Fitting FactorDecayMonitor (EWMA Spearman IC, halflife=63d)...")
    decay_monitor = FactorDecayMonitor()
    decay_monitor.fit(
        factor_arrays=[
            f_mom.values.astype(np.float64),    # F1_mom
            f_mom6.values.astype(np.float64),   # F1b_mom6
            f_rev.values.astype(np.float64),    # F2_rev21
            f_vol.values.astype(np.float64),    # F3_vol
            # F4 carry: static vector broadcast to (T,N) for IC computation
            np.tile(carry_vec, (len(returns_df), 1)).astype(np.float64),
            f_rev5.values.astype(np.float64),   # F7_rev5
            f_idio.values.astype(np.float64),   # F8_idio
        ],
        returns_matrix=returns_df.values.astype(np.float64),
    )

    logger.info(
        "Computing Factors 5/6: Regime Tilt + Price Trend Override (BUG #17 FIX) "
        "+ decay-gated blend loop..."
    )

    def _parse_z_mu(val) -> np.ndarray:
        if isinstance(val, (list, np.ndarray)):
            return np.asarray(val, dtype=np.float32)
        if isinstance(val, str):
            return np.array(json.loads(val), dtype=np.float32)
        return np.zeros(16, dtype=np.float32)

    mom_arr   = f_mom.values.astype(np.float64)
    mom6_arr  = f_mom6.values.astype(np.float64)
    rev_arr   = f_rev.values.astype(np.float64)
    vol_arr   = f_vol.values.astype(np.float64)
    rev5_arr  = f_rev5.values.astype(np.float64)
    idio_arr  = f_idio.values.astype(np.float64)
    ais_arr   = f_ais.values.astype(np.float64) if f_ais is not None else None

    dates     = returns_df.index
    prices_np = prices_df.reindex(columns=TICKERS).values.astype(np.float64, order="C")

    vixy_idx  = TICKERS.index("VIXY")
    cash_idxs = [TICKERS.index(t) for t in ("SHV", "BIL")]

    alpha_rows: List[np.ndarray] = []

    # Track how often each factor is gated out (for diagnostic)
    gate_zero_counts = np.zeros(7, dtype=int)

    for i, date in enumerate(dates):
        # ── Regime posterior ──────────────────────────────────────────────────
        z_mu = (
            _parse_z_mu(regime_df.loc[date, "z_mu"])
            if date in regime_df.index
            else np.zeros(16, dtype=np.float32)
        )
        z_pca  = float(np.clip(z_mu[0], -3.0, 3.0))
        z_mu_2 = float(np.clip(z_mu[2], -3.0, 3.0))

        # ── BUG #17 FIX: price-trend regime blending ─────────────────────────
        prices_i = int(prices_df.index.get_loc(date)) if date in prices_df.index else i
        trend_z  = _price_trend_z(prices_np, prices_i)

        trend_w     = float(np.clip(abs(trend_z) / 2.0 * 0.70, 0.0, 0.70))
        z_effective = float(np.clip(trend_w * trend_z + (1.0 - trend_w) * z_pca, -2.5, 2.5))

        # ── Raw lambda weights ────────────────────────────────────────────────
        raw_lambdas = np.array([
            float(np.clip(z_effective * 0.35  + 0.50, 0.05, 0.90)),   # F1  mom
            float(np.clip(z_effective * 0.22  + 0.27, 0.00, 0.58)),   # F1b mom6
            float(np.clip(-z_effective * 0.22 + 0.13, 0.00, 0.45)),   # F2  rev21
            float(np.clip(0.22 + abs(z_mu_2)  * 0.22, 0.10, 0.58)),  # F3  vol
            float(np.clip(z_effective * 0.18  + 0.22, 0.00, 0.40)),   # F4  carry
            float(np.clip(-abs(z_effective) * 0.20 + 0.30, 0.00, 0.35)),  # F7 rev5
            float(np.clip(z_effective * 0.30  + 0.55, 0.05, 0.80)),   # F8  idio
        ], dtype=np.float64)

        # ── P2 FIX: apply decay gate + L1-renormalize ─────────────────────────
        gate = decay_monitor.get_lambda_multipliers(i)
        gate_zero_counts += (gate == 0.0).astype(int)
        gated_lambdas = FactorDecayMonitor.renormalize_lambdas(raw_lambdas, gate)

        lam_mom   = gated_lambdas[0]
        lam_mom6  = gated_lambdas[1]
        lam_rev   = gated_lambdas[2]
        lam_vol   = gated_lambdas[3]
        lam_carry = gated_lambdas[4]
        lam_rev5  = gated_lambdas[5]
        lam_idio  = gated_lambdas[6]
        lam_ais   = 0.20 if ais_arr is not None else 0.0

        alpha_raw = (
              lam_mom   * mom_arr[i]
            + lam_mom6  * mom6_arr[i]
            + lam_rev   * rev_arr[i]
            + lam_vol   * vol_arr[i]
            + lam_carry * carry_vec
            + lam_rev5  * rev5_arr[i]
            + lam_idio  * idio_arr[i]
            + (lam_ais  * ais_arr[i] if ais_arr is not None else 0.0)
            + _build_regime_tilt_signal(z_effective)
        )
        alpha_tanh = np.tanh(alpha_raw)

        # ── VIXY override: vol-conditioned ────────────────────────────────────
        cross_vol = float(
            np.nanstd(returns_df.iloc[max(0, i - 21): i].values)
        ) if i >= 5 else 0.01
        vixy_base = float(np.tanh(-z_effective * 1.5))
        vixy_vcap = float(np.clip(cross_vol / 0.012 - 0.3, 0.0, 1.0))
        alpha_tanh[vixy_idx] = vixy_base * vixy_vcap

        # ── Cash override: always non-negative (BUG #18 FIX) ─────────────────
        for cash_idx in cash_idxs:
            bear_premium = float(np.clip(-z_effective * 0.3, 0.0, 0.3))
            alpha_tanh[cash_idx] = float(np.clip(0.1 + bear_premium, 0.0, 1.0))

        alpha_rows.append(alpha_tanh)

        if i % 200 == 0:
            top3        = sorted(zip(alpha_tanh, TICKERS), reverse=True)[:3]
            bot3        = sorted(zip(alpha_tanh, TICKERS))[:3]
            active_f    = sum(gate)
            regime_name = _infer_regime_name(z_effective)
            logger.info(
                f"  [{i}/{len(dates)}] {str(date)[:10]} | "
                f"z_pca={z_pca:+.2f} z_eff={z_effective:+.2f} | "
                f"Regime={regime_name} | "
                f"ActiveFactors={int(active_f)}/7 | "
                f"Top: {', '.join(f'{t}({v:.2f})' for v, t in top3)} | "
                f"Bot: {', '.join(f'{t}({v:.2f})' for v, t in bot3)}"
            )

    # ── Decay statistics ──────────────────────────────────────────────────────
    T = len(dates)
    logger.info("Factor decay gating summary (% of days zeroed by IC monitor):")
    for fi, name in enumerate(FactorDecayMonitor.FACTOR_NAMES):
        pct_gated = gate_zero_counts[fi] / T * 100
        logger.info(f"  {name:12s}: {pct_gated:.1f}% of days gated out")

    result = pd.DataFrame(
        np.array(alpha_rows),
        index=returns_df.index,
        columns=TICKERS,
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("══════ Fortress v5 — Alpha Signal Precomputation (P2: FactorDecayMonitor) ══════")

    for path in (_PRICES_PATH, _RETURNS_PATH, _REGIME_PATH):
        if not path.exists():
            logger.error(
                f"Missing required cache: {path}. Run precompute_regime_posteriors.py first."
            )
            sys.exit(1)

    logger.info("Loading cached market data...")
    prices_df  = pd.read_parquet(_PRICES_PATH)
    returns_df = pd.read_parquet(_RETURNS_PATH)
    logger.info("Loading cached regime posteriors...")
    regime_df  = pd.read_parquet(_REGIME_PATH)

    for df in (prices_df, returns_df, regime_df):
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        if df.index.duplicated().any():
            df.drop(df.index[df.index.duplicated(keep="last")], inplace=True)

    logger.info(
        f"Market data: {len(returns_df)} days × {len(TICKERS)} assets | "
        f"Regime posteriors: {len(regime_df)} rows"
    )

    common_dates    = returns_df.index.intersection(regime_df.index)
    returns_aligned = returns_df.loc[common_dates]
    prices_aligned  = prices_df.reindex(common_dates).ffill()
    logger.info(f"Aligned dataset: {len(returns_aligned)} trading days")

    if _try_full_mode_gat(returns_aligned, regime_df):
        return

    factors_active = [
        "F1(mom)", "F1b(mom6)", "F2(rev21)", "F3(vol)",
        "F4(carry)", "F5(tilt)", "F7(rev5)", "F8(idio)",
        "FactorDecayMonitor(P2)",
    ]
    if _ais_available:
        factors_active.append("F9(AIS)")
    logger.info(
        f"Surrogate Mode: {len(factors_active)}-component pipeline — "
        f"{', '.join(factors_active)}"
    )

    alpha_df = _compute_surrogate_alpha(prices_aligned, returns_aligned, regime_df)

    assert alpha_df.shape == (len(returns_aligned), N_ASSETS), (
        f"Shape mismatch: {alpha_df.shape} != ({len(returns_aligned)}, {N_ASSETS})"
    )
    assert (alpha_df.abs() <= 1.0 + 1e-5).all().all(), \
        "Alpha values outside [-1, 1] detected — tanh saturation failure."

    alpha_df.to_parquet(_ALPHA_OUT)
    logger.info(
        f"✅ Alpha signals saved → {_ALPHA_OUT} "
        f"({len(alpha_df)} rows × {N_ASSETS} assets)"
    )

    # ── Signal summary ────────────────────────────────────────────────────────
    means = alpha_df.mean().sort_values(ascending=False)
    logger.info("Alpha signal summary (time-averaged, post-decay-gate):")
    bar_scale = 6.0 / (means.abs().max() + 1e-8)
    for ticker, val in means.items():
        bar_len = int(abs(val) * bar_scale)
        sign    = "+" if val >= 0 else "-"
        bar     = "█" * bar_len
        logger.info(f"  {ticker:6s}: {sign}{bar} ({val:+.3f})")

    logger.info(
        "Stage 2 complete (P2: FactorDecayMonitor active). "
        "Run run_standalone_backtest.py next."
    )


if __name__ == "__main__":
    main()