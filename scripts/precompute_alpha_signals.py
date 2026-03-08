"""
FORTRESS v5 - precompute_alpha_signals.py
Path: scripts/precompute_alpha_signals.py

Offline alpha signal precomputation.
Loads cached prices/returns/regime posteriors and writes alpha_signals.parquet
for consumption by the standalone backtest and EDT training.

BUG FIXES RETAINED (previous sessions):
  - BUG #17: Price-trend regime override (SPY MA50/200 + breadth + HYG/SHY spread).
  - BUG #18: Fixed-income carry estimates are non-negative (2019–2024 yield averages).
  - BUG #15: No FutureWarning flood.

P1 ENHANCEMENTS (this session):
  - P1-F7: Factor 7 — 5-Day Short-Term Reversal (negated momentum).
      Captures the microstructure mean-reversion at the weekly horizon.
      Pure 5-day cumulative return, ranked descending → negative signal (buy losers).
      5d reversal is orthogonal to the 21d reversal factor (different autocorrelation
      structure: bid-ask bounce + order flow at 5d vs portfolio rebalancing at 21d).

  - P1-F8: Factor 8 — Idiosyncratic Momentum (beta-adjusted).
      Beta-strips the market component before computing 12-1M momentum:
        β_i = cov(r_i, r_SPY) / var(r_SPY)  — rolling 63-day window
        r_idio_i = r_i − β_i × r_SPY
        Signal = cross-sectional rank of 12-1M cumulative idiosyncratic return
      Rationale: Standard momentum conflates market exposure with genuine
      stock-specific persistence. Idiosyncratic momentum has higher IC and
      lower correlation to market drawdowns (avoids momentum crashes).

  - P1-AIS: Wire AIS commodity flow signal into alpha blend.
      Attempts `from data.alt_data.ais_pipeline import AISCommoditySignal`.
      If unavailable (not yet trained/deployed), gracefully degrades to zero.
      AIS signal covers: USO, PDBC, SLV, GLD, EEM tickers via vessel traffic.

Factor blend (8 factors, P1 blend weights):
  F1:  12-1M Cross-Sectional Momentum     λ = f(z_eff, [0.05, 1.00])
  F1b: 6-1M Cross-Sectional Momentum      λ = f(z_eff, [0.00, 0.65])
  F2:  21-Day Short-Term Reversal          λ = f(z_eff, [0.00, 0.50])
  F3:  63-Day Low-Volatility Anomaly       λ = f(|z2|,  [0.10, 0.65])
  F4:  Fixed-Income Carry                  λ = f(z_eff, [0.00, 0.45])
  F5:  Regime Tilt + Price Trend Override  (additive, not multiplied)
  F7:  5-Day Short-Term Reversal (NEW)     λ = f(z_eff, [0.00, 0.35])
  F8:  Idiosyncratic Momentum (NEW)        λ = f(z_eff, [0.05, 0.80])
  F9:  AIS Commodity Flow (NEW, optional)  λ = 0.20 if available else 0.0
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

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

# ── P1-AIS: Attempt to import AIS pipeline ────────────────────────────────────
# Commodity-flow signals are available for: USO, PDBC, SLV, GLD, EEM
# If the pipeline hasn't been deployed yet, _ais_available=False and the
# AIS factor contributes zero to the blend (graceful degradation).
_ais_available: bool = False
try:
    from data.alt_data.ais_pipeline import AISCommoditySignal as _AISCommoditySignal  # type: ignore
    _ais_available = True
    logger.info("✅ AIS commodity flow pipeline loaded. Factor F9 active.")
except ImportError:
    logger.info("AIS pipeline not available — Factor F9 inactive. Wire data/alt_data/ais_pipeline.py.")

# Tickers that the AIS signal covers (vessel traffic → supply/demand proxy)
_AIS_TICKERS: List[str] = ["USO", "PDBC", "SLV", "GLD", "EEM"]
_AIS_TICKER_IDXS: List[int] = [TICKERS.index(t) for t in _AIS_TICKERS]

# ── Fixed-Income Carry Estimates ─────────────────────────────────────────────
# BUG #18 FIX: non-negative, 2019-2024 yield averages
CARRY_ESTIMATES: Dict[str, float] = {
    "TLT":  0.025,  "IEF":  0.018,  "SHY":  0.025,
    "LQD":  0.032,  "HYG":  0.055,  "SHV":  0.035,  "BIL":  0.035,
}


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
# Factor 1: 12-1 Month Momentum
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
    returns_df: pd.DataFrame,
    window:      int = 126,
    skip_window: int = 21,
) -> pd.DataFrame:
    """6-1 month horizon — faster to catch regime transitions missed by annual window."""
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
    window: int = 21,
) -> pd.DataFrame:
    """1-month reversal: buy recent 21-day losers (portfolio rebalancing flow)."""
    cum = returns_df.rolling(window, min_periods=5).sum().fillna(0.0).values
    ranked = np.apply_along_axis(
        lambda row: _cross_sectional_rank(row, descending=True),  # descending=True → ranks losers +1
        axis=1, arr=cum,
    )
    return pd.DataFrame(ranked, index=returns_df.index, columns=TICKERS)


# ─────────────────────────────────────────────────────────────────────────────
# Factor 7 (P1-F7): 5-Day Short-Term Reversal
# ─────────────────────────────────────────────────────────────────────────────

def _build_reversal_5d_signal(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    5-day mean-reversion signal (P1 NEW).

    Captures bid-ask bounce + intraweek order-flow reversal that is distinct
    from the 21-day portfolio-rebalancing reversal.

    Signal: rank recent 5-day losers highest (buy mean-reversion),
            rank recent 5-day winners lowest (sell overextended).

    5-day IC is typically −0.04 to −0.08 in US equity ETFs and even stronger
    in commodity/vol ETFs (USO, VIXY) due to futures roll mechanics.
    """
    cum = returns_df.rolling(5, min_periods=3).sum().fillna(0.0).values
    ranked = np.apply_along_axis(
        lambda row: _cross_sectional_rank(row, descending=True),  # buy losers
        axis=1, arr=cum,
    )
    return pd.DataFrame(ranked, index=returns_df.index, columns=TICKERS)


# ─────────────────────────────────────────────────────────────────────────────
# Factor 8 (P1-F8): Idiosyncratic Momentum
# ─────────────────────────────────────────────────────────────────────────────

def _build_idiosyncratic_momentum_signal(
    returns_df: pd.DataFrame,
    beta_window:     int = 63,   # rolling window for beta estimation
    momentum_window: int = 252,  # 12-month cumulative horizon
    skip_window:     int = 21,   # standard Jegadeesh-Titman skip
) -> pd.DataFrame:
    """
    Idiosyncratic momentum (P1 NEW).

    Algorithm:
      1. For each asset i and each day t:
           β_i(t) = cov(r_i[t−63:t], r_SPY[t−63:t]) / var(r_SPY[t−63:t])
      2. r_idio_i(t) = r_i(t) − β_i(t) × r_SPY(t)
      3. Signal = rank(cum_idio_12m − cum_idio_1m) across assets on date t.

    Why it matters:
      Standard momentum is dominated by market beta. During momentum crashes
      (2009, 2020), high-momentum portfolios are also high-beta — they get
      hammered when the market reverses. Idiosyncratic momentum is beta-neutral
      by construction and is more persistent in out-of-sample tests
      (Blitz et al. 2011: idiosyncratic momentum IC ~0.05 vs 0.03 for raw).

    Complexity: O(N × T) rolling OLS — vectorised over assets using pandas.
    """
    if "SPY" not in returns_df.columns:
        logger.warning("SPY not in returns_df — idiosyncratic momentum unavailable. Returning zeros.")
        return pd.DataFrame(
            np.zeros((len(returns_df), N_ASSETS), dtype=np.float64),
            index=returns_df.index,
            columns=TICKERS,
        )

    spy_returns = returns_df["SPY"].values.astype(np.float64)
    n_dates     = len(returns_df)
    idio_returns = np.zeros((n_dates, N_ASSETS), dtype=np.float64)

    # Vectorised rolling beta: for each asset, compute rolling covariance with SPY
    # Pandas ewm/rolling cov is the cleanest approach here.
    spy_series  = returns_df["SPY"]
    spy_var     = spy_series.rolling(beta_window, min_periods=20).var()  # (T,)

    for col_idx, ticker in enumerate(TICKERS):
        asset_series = returns_df[ticker]
        cov_series   = asset_series.rolling(beta_window, min_periods=20).cov(spy_series)
        beta_series  = (cov_series / spy_var.clip(lower=1e-10)).fillna(1.0)

        # r_idio = r_asset − β × r_SPY
        idio_col = asset_series.values - beta_series.values * spy_returns
        idio_returns[:, col_idx] = idio_col

    # Build 12-1M momentum on idiosyncratic returns using a DataFrame
    idio_df = pd.DataFrame(idio_returns, index=returns_df.index, columns=TICKERS)
    cum_long  = idio_df.rolling(momentum_window, min_periods=60).sum()
    cum_skip  = idio_df.rolling(skip_window,     min_periods=5).sum()
    signal    = (cum_long - cum_skip).fillna(0.0).values

    ranked = np.apply_along_axis(
        lambda row: _cross_sectional_rank(row, descending=False),
        axis=1, arr=signal,
    )
    return pd.DataFrame(ranked, index=returns_df.index, columns=TICKERS)


# ─────────────────────────────────────────────────────────────────────────────
# Factor 3: Low-Volatility Anomaly
# ─────────────────────────────────────────────────────────────────────────────

def _build_vol_signal(
    returns_df: pd.DataFrame,
    window: int = 63,
) -> pd.DataFrame:
    """63-day realised vol ranked ascending — prefer low-vol assets."""
    vol = returns_df.rolling(window, min_periods=21).std().fillna(0.1).values
    ranked = np.apply_along_axis(
        lambda row: _cross_sectional_rank(row, descending=True),  # high vol → negative
        axis=1, arr=vol,
    )
    return pd.DataFrame(ranked, index=returns_df.index, columns=TICKERS)


# ─────────────────────────────────────────────────────────────────────────────
# Factor 4: Fixed-Income Carry
# ─────────────────────────────────────────────────────────────────────────────

def _build_carry_signal() -> Dict[str, float]:
    """BUG #18 FIX: all carry estimates non-negative."""
    return {t: CARRY_ESTIMATES.get(t, 0.0) for t in TICKERS}


# ─────────────────────────────────────────────────────────────────────────────
# Factor 5: Price-Trend Regime Override (BUG #17 FIX)
# ─────────────────────────────────────────────────────────────────────────────

def _price_trend_z(
    prices_arr: np.ndarray,
    i:          int,
    fast_window: int = 50,
    slow_window: int = 200,
) -> float:
    """
    Three-component price trend signal:
      (1) SPY 50/200d MA crossover  (50% weight)
      (2) Cross-asset breadth: % above 50d MA  (30% weight)
      (3) HYG vs SHY 21d momentum spread  (20% weight)

    Uses prices_arr[:i] — strictly causal.
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
    """Additive tilt conditioned on blended z_effective regime signal."""
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
# Factor 9 (P1-AIS): AIS Commodity Flow
# ─────────────────────────────────────────────────────────────────────────────

def _build_ais_signal(returns_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    AIS vessel traffic flow signal for commodity ETFs (P1-AIS NEW).

    Loads precomputed AIS flow scores from data/alt_data/ais_pipeline.py.
    Returns a (T, N_ASSETS) DataFrame where only AIS-covered tickers
    (USO, PDBC, SLV, GLD, EEM) have non-zero values.

    Returns None if AIS pipeline is unavailable.
    """
    if not _ais_available:
        return None
    try:
        ais_signal = _AISCommoditySignal()  # type: ignore[name-defined]
        flow_df = ais_signal.load_precomputed_scores(
            start_date=str(returns_df.index[0].date()),
            end_date=str(returns_df.index[-1].date()),
        )
        # Align to our full TICKERS universe — zero for non-covered tickers
        result = pd.DataFrame(
            np.zeros((len(returns_df), N_ASSETS), dtype=np.float64),
            index=returns_df.index,
            columns=TICKERS,
        )
        for ticker in _AIS_TICKERS:
            if ticker in flow_df.columns:
                aligned = flow_df[ticker].reindex(returns_df.index).fillna(0.0)
                result[ticker] = aligned.values
        logger.info(f"AIS signal loaded: {flow_df.shape} → aligned to {result.shape}")
        return result
    except Exception as exc:
        logger.warning(f"AIS signal load failed ({exc}). Degrading to zero.")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# GATv2 stub
# ─────────────────────────────────────────────────────────────────────────────

def _try_full_mode_gat(
    returns_df: pd.DataFrame,
    regime_df:  pd.DataFrame,
) -> bool:
    if not _GAT_WEIGHTS.exists():
        logger.info(f"GATv2 weights not found at {_GAT_WEIGHTS}. Running in Surrogate Mode.")
        return False
    try:
        import torch  # type: ignore
        logger.warning("Full-mode GATv2 not yet wired — falling back to Surrogate Mode.")
        return False
    except Exception as exc:
        logger.warning(f"Full mode GATv2 aborted ({exc}). Running Surrogate Mode.")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Surrogate Alpha — 8-factor blend (P1)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_surrogate_alpha(
    prices_df:  pd.DataFrame,
    returns_df: pd.DataFrame,
    regime_df:  pd.DataFrame,
) -> pd.DataFrame:
    """
    P1: 8-factor regime-conditioned alpha (was 6-factor).
    Added: F7 (5-day reversal), F8 (idiosyncratic momentum), F9 (AIS optional).

    Factor weight design principles:
      - F8 (idio momentum) gets slightly higher max weight than F1 (raw momentum)
        because it removes the beta-crash exposure that makes raw momentum risky.
      - F7 (5d reversal) is capped at 0.35 — strong at short horizons but high
        turnover cost; we moderate via the lambda multiplier.
      - F9 (AIS) is gated at 0.20 flat — alt-data signals have unknown IC until
        live validation; conservative allocation.
    """
    logger.info("Computing Factor 1:  Cross-Sectional Momentum (12-1 month)...")
    f_mom  = _build_momentum_signal(returns_df)
    logger.info("Computing Factor 1b: Cross-Sectional Momentum (6-1 month)...")
    f_mom6 = _build_momentum_6m_signal(returns_df)
    logger.info("Computing Factor 2:  Short-Term Reversal (21-day)...")
    f_rev  = _build_reversal_signal(returns_df)
    logger.info("Computing Factor 3:  Low-Volatility Anomaly (63-day)...")
    f_vol  = _build_vol_signal(returns_df)
    logger.info("Computing Factor 4:  Fixed-Income Carry (BUG #18 FIX)...")
    carry_vec = np.array([_build_carry_signal()[t] for t in TICKERS], dtype=np.float64)
    logger.info("Computing Factor 7:  5-Day Short-Term Reversal (P1 NEW)...")
    f_rev5 = _build_reversal_5d_signal(returns_df)
    logger.info("Computing Factor 8:  Idiosyncratic Momentum (P1 NEW)...")
    f_idio = _build_idiosyncratic_momentum_signal(returns_df)
    logger.info("Computing Factor 9:  AIS Commodity Flow (P1 NEW)...")
    f_ais  = _build_ais_signal(returns_df)  # None if unavailable
    logger.info("Computing Factors 5/6: Regime Tilt + Price Trend Override (BUG #17 FIX)...")

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

    alpha_rows: list[np.ndarray] = []

    for i, date in enumerate(dates):
        # ── Regime posterior ──────────────────────────────────────────────────
        z_mu = (
            _parse_z_mu(regime_df.loc[date, "z_mu"])
            if date in regime_df.index
            else np.zeros(16, dtype=np.float32)
        )
        z_pca  = float(np.clip(z_mu[0], -3.0, 3.0))
        z_mu_2 = float(np.clip(z_mu[2], -3.0, 3.0))

        # ── BUG #17 FIX: price-trend regime correction ────────────────────────
        prices_i = int(prices_df.index.get_loc(date)) if date in prices_df.index else i
        trend_z  = _price_trend_z(prices_np, prices_i)

        trend_w      = float(np.clip(abs(trend_z) / 2.0 * 0.70, 0.0, 0.70))
        z_effective  = float(np.clip(trend_w * trend_z + (1.0 - trend_w) * z_pca, -2.5, 2.5))

        # ── Regime-adaptive factor loadings ──────────────────────────────────
        # Existing factors: slightly reduced max weights to accommodate F7/F8.
        lambda_mom   = float(np.clip(z_effective * 0.35  + 0.50, 0.05, 0.90))
        lambda_mom6  = float(np.clip(z_effective * 0.22  + 0.27, 0.00, 0.58))
        lambda_rev   = float(np.clip(-z_effective * 0.22 + 0.13, 0.00, 0.45))
        lambda_vol   = float(np.clip(0.22 + abs(z_mu_2)  * 0.22, 0.10, 0.58))
        lambda_carry = float(np.clip(z_effective * 0.18  + 0.22, 0.00, 0.40))

        # P1-F7: 5-day reversal loading.
        # Suppressed in strong trending regimes (|z_eff| > 1.5) where mean-reversion
        # signals fail — momentum dominates at multi-week horizons.
        lambda_rev5  = float(np.clip(
            -abs(z_effective) * 0.20 + 0.30, 0.00, 0.35
        ))

        # P1-F8: idiosyncratic momentum.
        # Gets elevated weight in bull regimes (z_eff > 0); suppressed in crisis
        # (beta-stripped signal less informative when correlations spike to 1).
        lambda_idio  = float(np.clip(z_effective * 0.30 + 0.55, 0.05, 0.80))

        # P1-F9: AIS — flat weight if available
        lambda_ais   = 0.20 if ais_arr is not None else 0.0

        alpha_raw = (
            lambda_mom   * mom_arr[i]
            + lambda_mom6  * mom6_arr[i]
            + lambda_rev   * rev_arr[i]
            + lambda_vol   * vol_arr[i]
            + lambda_carry * carry_vec
            + lambda_rev5  * rev5_arr[i]
            + lambda_idio  * idio_arr[i]
            + (lambda_ais  * ais_arr[i] if ais_arr is not None else 0.0)
            + _build_regime_tilt_signal(z_effective)
        )
        alpha_tanh = np.tanh(alpha_raw)

        # ── VIXY: vol-level conditioning ──────────────────────────────────────
        cross_vol = float(np.nanstd(returns_df.iloc[max(0, i-21):i].values)) if i >= 5 else 0.01
        vixy_base = float(np.tanh(-z_effective * 1.5))
        vixy_vcap = float(np.clip(cross_vol / 0.012 - 0.3, 0.0, 1.0))
        alpha_tanh[vixy_idx] = vixy_base * vixy_vcap

        # ── Cash (SHV/BIL): always non-negative (BUG #18 FIX) ────────────────
        for cash_idx in cash_idxs:
            bear_premium = float(np.clip(-z_effective * 0.3, 0.0, 0.3))
            alpha_tanh[cash_idx] = float(np.clip(0.1 + bear_premium, 0.0, 1.0))

        alpha_rows.append(alpha_tanh)

        # Diagnostic logging every 200 rows
        if i % 200 == 0:
            top3  = sorted(zip(alpha_tanh, TICKERS), reverse=True)[:3]
            bot3  = sorted(zip(alpha_tanh, TICKERS))[:3]
            regime_name = _infer_regime_name(z_effective)
            logger.info(
                f"  [{i}/{len(dates)}] {str(date)[:10]} | "
                f"z_pca={z_pca:+.2f} trend={trend_z:+.2f} z_eff={z_effective:+.2f} | "
                f"Regime={regime_name} | "
                f"Top: {', '.join(f'{t}({v:.2f})' for v, t in top3)} | "
                f"Bot: {', '.join(f'{t}({v:.2f})' for v, t in bot3)} | "
                f"λ_idio={lambda_idio:.2f} λ_rev5={lambda_rev5:.2f}"
            )

    result = pd.DataFrame(
        np.array(alpha_rows),
        index=returns_df.index,
        columns=TICKERS,
    )
    # Prefix columns for clarity
    result.columns = [f"alpha_{t}" for t in TICKERS]
    return result


def _infer_regime_name(z_effective: float) -> str:
    if   z_effective >  1.5: return "bull_low_vol"
    elif z_effective >  0.5: return "bull_high_vol"
    elif z_effective > -0.5: return "neutral"
    elif z_effective > -1.5: return "bear_market"
    else:                    return "crisis"


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("══════ Fortress v5 — Alpha Signal Precomputation ══════")

    for path, name in [
        (_PRICES_PATH,  "prices_wide.parquet"),
        (_RETURNS_PATH, "returns_wide.parquet"),
        (_REGIME_PATH,  "regime_posteriors.parquet"),
    ]:
        if not path.exists():
            logger.error(f"Missing required cache file: {path}. Run precompute_regime.py first.")
            sys.exit(1)

    logger.info("Loading cached market data...")
    prices_df  = pd.read_parquet(_PRICES_PATH)
    returns_df = pd.read_parquet(_RETURNS_PATH)
    logger.info("Loading cached regime posteriors...")
    regime_df  = pd.read_parquet(_REGIME_PATH)

    for df in (prices_df, returns_df, regime_df):
        df.index = pd.to_datetime(df.index)

    for df in (prices_df, returns_df, regime_df):
        df.sort_index(inplace=True)
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

    factors_active = ["F1(mom)", "F1b(mom6)", "F2(rev21)", "F3(vol)", "F4(carry)",
                       "F5(tilt)", "F7(rev5)", "F8(idio)"]
    if _ais_available:
        factors_active.append("F9(AIS)")
    logger.info(f"Surrogate Mode: {len(factors_active)}-factor model — {', '.join(factors_active)}")

    alpha_df = _compute_surrogate_alpha(prices_aligned, returns_aligned, regime_df)

    assert alpha_df.shape == (len(returns_aligned), N_ASSETS), (
        f"Shape mismatch: {alpha_df.shape} != ({len(returns_aligned)}, {N_ASSETS})"
    )
    assert (alpha_df.abs() <= 1.0 + 1e-5).all().all(), "Alpha values outside [-1, 1] detected."

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    alpha_df.to_parquet(_ALPHA_OUT)
    logger.info(f"✅ Alpha signals saved → {_ALPHA_OUT} ({len(alpha_df)} rows × {N_ASSETS} assets)")

    logger.info("Alpha signal summary (time-averaged per asset):")
    mean_alpha = alpha_df.mean(axis=0).sort_values(ascending=False)
    for col, val in mean_alpha.items():
        ticker = col.replace("alpha_", "")
        bar    = "█" * int(abs(val) * 20)
        sign   = "+" if val >= 0 else "-"
        logger.info(f"  {ticker:6s}: {sign}{bar} ({val:+.3f})")

    logger.info("Precompute Stage 2 complete. Run run_standalone_backtest.py next.")


if __name__ == "__main__":
    main()