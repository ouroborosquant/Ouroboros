"""
FORTRESS v5 — precompute_alpha_signals.py  [PATCH v3]
Path: scripts/precompute_alpha_signals.py

BUG #17 FIX (Regime Semantic Inversion):
    PCA k-means labels 45.4% of days as 'crisis' when the true GBM simulation
    has 94.1% bull_low_vol days. Dates 2020-08-04 → 2021-05-20 show z_mu[0]
    = -1.94/-2.00 (crisis) → alpha loads VIXY/GLD/SHY on actual bull days.

    Fix: _price_trend_z() computes a strictly causal, model-free trend score:
        (1) SPY 50d/200d dual-MA crossover (primary trend)
        (2) Cross-asset market breadth (% of 25 assets above 50d MA)
        (3) HYG/SHY 21d return spread (credit risk-appetite proxy)
    Blended with PCA z_mu[0] via confidence-weighted merge:
        z_effective = w * trend_z + (1-w) * z_pca
        w = clip(|trend_z| / 2.0 * 0.70, 0.0, 0.70)
    When trend is decisive (|trend_z| > 1.4), price signal dominates.
    When ambiguous (|trend_z| < 0.3), PCA posterior dominates.
    Uses prices[:i] (strictly causal — excludes today).

BUG #18 FIX (Carry Sign Error):
    Previous: SHY=-0.01, SHV=-0.005, BIL=-0.01 (all negative).
    T-bill instruments earn the highest carry in a 5% risk-free environment.
    Fix: All carry estimates are non-negative, using 2019-2024 yield averages.

BUG #15 FIX (retained from v2):
    Zero FutureWarning flood via numpy array construction.

ENHANCEMENT — Factor 1b: 6-month momentum for faster trend inflection detection.
ENHANCEMENT — Increased regime tilt coefficients: equity 0.5→0.65, defensive 0.4→0.55.
ENHANCEMENT — VIXY conditioned on cross-asset realised vol level (low-vol suppression).
ENHANCEMENT — Trend override diagnostic logged (bull/neutral/bear% from trend signal).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

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
N_ASSETS = len(TICKERS)

DEFENSIVE_TICKERS  = {"TLT", "IEF", "SHY", "GLD", "SHV", "BIL"}
EQUITY_TICKERS     = {
    "SPY", "QQQ", "IWM", "VTV",
    "XLK", "XLF", "XLV", "XLP", "XLI", "XLE",
    "EFA", "EEM", "VNQ", "HYG",
}
VOLATILITY_TICKERS = {"VIXY"}

# BUG #18 FIX: all carry estimates now non-negative.
# Approximate 2019-2024 annualised yield averages.
CARRY_ESTIMATES: Dict[str, float] = {
    "TLT":  0.025,   # 20yr Treasury  ~2.5% avg
    "IEF":  0.018,   # 10yr Treasury  ~1.8% avg
    "SHY":  0.025,   # 2yr Treasury   ~2.5% avg
    "LQD":  0.032,   # IG corp        ~3.2% avg
    "HYG":  0.055,   # High yield     ~5.5% avg
    "SHV":  0.035,   # 1-3mo T-bills  ~3.5% avg
    "BIL":  0.035,   # T-bills        ~3.5% avg
}


# ── Cross-sectional rank helper ───────────────────────────────────────────────

def _cross_sectional_rank(arr: np.ndarray, descending: bool = True) -> np.ndarray:
    """Signal vector → cross-sectional percentile ranks in [−1, 1]. NaN → 0."""
    valid = ~np.isnan(arr)
    ranks = np.zeros(len(arr), dtype=np.float64)
    if valid.sum() < 2:
        return ranks
    n_valid = int(valid.sum())
    temp    = np.argsort(arr[valid])
    r       = (temp.argsort() + 1).astype(np.float64)
    r       = (r - (n_valid + 1) / 2.0) / ((n_valid - 1) / 2.0 + 1e-8)
    if descending:
        r = -r
    ranks[valid] = r
    return ranks


# ── Factor 1: 12-1 Month Momentum ────────────────────────────────────────────

def _build_momentum_signal(
    returns_df: pd.DataFrame,
    momentum_window: int = 252,
    skip_window:     int = 21,
) -> pd.DataFrame:
    cum_long = returns_df.rolling(momentum_window, min_periods=60).sum()
    cum_skip = returns_df.rolling(skip_window,     min_periods=5).sum()
    signal   = (cum_long - cum_skip).fillna(0.0).values
    ranked_arr = np.apply_along_axis(
        lambda row: _cross_sectional_rank(row, descending=False),
        axis=1, arr=signal,
    )
    return pd.DataFrame(ranked_arr, index=returns_df.index, columns=TICKERS)


# ── Factor 1b: 6-1 Month Momentum (ENHANCEMENT) ──────────────────────────────

def _build_momentum_6m_signal(
    returns_df: pd.DataFrame,
    window:      int = 126,
    skip_window: int = 21,
) -> pd.DataFrame:
    """Faster 6-1 month horizon. Catches regime transitions missed by annual window."""
    cum_long = returns_df.rolling(window,      min_periods=30).sum()
    cum_skip = returns_df.rolling(skip_window, min_periods=5).sum()
    signal   = (cum_long - cum_skip).fillna(0.0).values
    ranked_arr = np.apply_along_axis(
        lambda row: _cross_sectional_rank(row, descending=False),
        axis=1, arr=signal,
    )
    return pd.DataFrame(ranked_arr, index=returns_df.index, columns=TICKERS)


# ── Factor 2: Short-Term Reversal ─────────────────────────────────────────────

def _build_reversal_signal(
    returns_df: pd.DataFrame,
    reversal_window: int = 21,
) -> pd.DataFrame:
    signal = returns_df.rolling(reversal_window, min_periods=5).sum().fillna(0.0).values
    ranked_arr = np.apply_along_axis(
        lambda row: _cross_sectional_rank(row, descending=True),
        axis=1, arr=signal,
    )
    return pd.DataFrame(ranked_arr, index=returns_df.index, columns=TICKERS)


# ── Factor 3: Low-Volatility Anomaly ─────────────────────────────────────────

def _build_vol_signal(
    returns_df: pd.DataFrame,
    vol_window: int = 63,
) -> pd.DataFrame:
    cash_mask = np.array([t in {"SHV", "BIL"} for t in TICKERS])
    vol_arr   = returns_df.rolling(vol_window, min_periods=20).std().fillna(0.15).values

    def _rank_row(row: np.ndarray) -> np.ndarray:
        r = _cross_sectional_rank(row, descending=True)
        r[cash_mask] = 0.0
        return r

    ranked_arr = np.apply_along_axis(_rank_row, axis=1, arr=vol_arr)
    return pd.DataFrame(ranked_arr, index=returns_df.index, columns=TICKERS)


# ── Factor 4: Fixed-Income Carry (BUG #18 FIX) ───────────────────────────────

def _build_carry_signal() -> Dict[str, float]:
    max_carry = max(abs(v) for v in CARRY_ESTIMATES.values()) + 1e-8
    return {t: CARRY_ESTIMATES.get(t, 0.0) / max_carry for t in TICKERS}


# ── BUG #17 FIX: Causal Price-Based Trend Regime Signal ──────────────────────

def _price_trend_z(
    prices_arr: np.ndarray,
    i: int,
    fast_window: int = 50,
    slow_window: int = 200,
) -> float:
    """
    Returns z in [-2, +2]: positive → bull trend, negative → bear/crisis.
    Uses prices[:i] — strictly excludes current day (causal).

    Three sub-signals blended 50/30/20:
      (1) SPY 50/200d MA crossover: trend_pct * 50, clipped to [-2, +2]
      (2) Cross-asset breadth: (pct_above_50d_MA - 0.5) * 6.0
      (3) HYG vs SHY 21d momentum spread (credit risk-appetite)
    """
    if i < slow_window:
        return 0.0
    px = prices_arr[:i]   # (i, N_ASSETS) — strictly causal
    if len(px) < slow_window:
        return 0.0

    spy_idx = TICKERS.index("SPY")
    hyg_idx = TICKERS.index("HYG")
    shy_idx = TICKERS.index("SHY")

    # (1) SPY dual-MA crossover
    spy_window = px[-slow_window:, spy_idx]
    ma_fast    = float(spy_window[-fast_window:].mean())
    ma_slow    = float(spy_window.mean())
    trend_pct  = ma_fast / (ma_slow + 1e-8) - 1.0
    trend_z    = float(np.clip(trend_pct * 50.0, -2.0, 2.0))

    # (2) Cross-asset breadth
    px_recent  = px[-fast_window:]        # (50, N_ASSETS)
    px_current = px[-1]                   # (N_ASSETS,)
    ma50_all   = px_recent.mean(axis=0)   # (N_ASSETS,)
    breadth    = float((px_current > ma50_all).mean())
    breadth_z  = float(np.clip((breadth - 0.5) * 6.0, -2.0, 2.0))

    # (3) HYG/SHY momentum spread
    credit_z = 0.0
    if i >= 22:
        hyg_ret  = float(px[-1, hyg_idx] / (px[-21, hyg_idx] + 1e-8) - 1.0)
        shy_ret  = float(px[-1, shy_idx] / (px[-21, shy_idx] + 1e-8) - 1.0)
        credit_z = float(np.clip((hyg_ret - shy_ret) * 25.0, -1.0, 1.0))

    composite = 0.50 * trend_z + 0.30 * breadth_z + 0.20 * credit_z
    return float(np.clip(composite, -2.0, 2.0))


# ── Factor 5: Regime Tilt (uses z_effective) ─────────────────────────────────

def _build_regime_tilt_signal(z_effective: float) -> np.ndarray:
    """
    ENHANCEMENT: uses blended z_effective (PCA + trend), not raw z_pca.
    Increased tilt coefficients for sharper regime-conditional rotation.
    """
    tilt = np.zeros(N_ASSETS, dtype=np.float64)
    for i, t in enumerate(TICKERS):
        if t in EQUITY_TICKERS:
            tilt[i] = z_effective * 0.65
        elif t in DEFENSIVE_TICKERS:
            tilt[i] = -z_effective * 0.55
        elif t in VOLATILITY_TICKERS:
            tilt[i] = -z_effective * 1.20
    return tilt


# ── Full-mode GATv2 stub ──────────────────────────────────────────────────────

def _try_full_mode_gat(returns_df: pd.DataFrame, regime_df: pd.DataFrame) -> bool:
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


# ── Surrogate Alpha (PATCH v3) ────────────────────────────────────────────────

def _compute_surrogate_alpha(
    prices_df:  pd.DataFrame,
    returns_df: pd.DataFrame,
    regime_df:  pd.DataFrame,
) -> pd.DataFrame:
    """
    6-factor regime-conditioned alpha with BUG #17 + #18 fixes.
    Regime signal: z_effective = confidence_weight * trend_z + (1-w) * z_pca
    where confidence_weight = clip(|trend_z| / 2.0 * 0.70, 0.0, 0.70).
    """
    logger.info("Computing Factor 1:  Cross-Sectional Momentum (12-1 month)...")
    f_mom  = _build_momentum_signal(returns_df)
    logger.info("Computing Factor 1b: Cross-Sectional Momentum (6-1 month)...")
    f_mom6 = _build_momentum_6m_signal(returns_df)
    logger.info("Computing Factor 2:  Short-Term Reversal (1 month)...")
    f_rev  = _build_reversal_signal(returns_df)
    logger.info("Computing Factor 3:  Low-Volatility Anomaly (63-day realised vol)...")
    f_vol  = _build_vol_signal(returns_df)
    logger.info("Computing Factor 4:  Fixed-Income Carry (BUG #18 FIX)...")
    carry_dict = _build_carry_signal()
    carry_vec  = np.array([carry_dict[t] for t in TICKERS], dtype=np.float64)
    logger.info("Computing Factors 5/6: Regime Tilt + Price Trend Override (BUG #17 FIX)...")

    def _parse_z_mu(val) -> np.ndarray:
        if isinstance(val, (list, np.ndarray)):
            return np.asarray(val, dtype=np.float32)
        if isinstance(val, str):
            return np.array(json.loads(val), dtype=np.float32)
        return np.zeros(16, dtype=np.float32)

    mom_arr  = f_mom.values.astype(np.float64)
    mom6_arr = f_mom6.values.astype(np.float64)
    rev_arr  = f_rev.values.astype(np.float64)
    vol_arr  = f_vol.values.astype(np.float64)
    dates    = returns_df.index

    # Materialise prices as C-contiguous numpy array for fast row slicing
    # Align to TICKERS column order before converting
    prices_np = prices_df.reindex(columns=TICKERS).values.astype(np.float64, order="C")

    vixy_idx  = TICKERS.index("VIXY")
    cash_idxs = [TICKERS.index(t) for t in ("SHV", "BIL")]

    alpha_rows:      list[np.ndarray] = []
    trend_log:       list[float]      = []

    for i, date in enumerate(dates):
        # ── PCA posterior ──────────────────────────────────────────────────────
        z_mu = (
            _parse_z_mu(regime_df.loc[date, "z_mu"])
            if date in regime_df.index
            else np.zeros(16, dtype=np.float32)
        )
        z_pca  = float(np.clip(z_mu[0], -3.0, 3.0))
        z_mu_2 = float(np.clip(z_mu[2], -3.0, 3.0))

        # ── BUG #17 FIX: price-trend regime correction ────────────────────────
        # prices_i: causal index into prices_np (may differ from returns index)
        prices_i = int(prices_df.index.get_loc(date)) if date in prices_df.index else i
        trend_z  = _price_trend_z(prices_np, prices_i, fast_window=50, slow_window=200)
        trend_log.append(trend_z)

        # Confidence-weighted blend (max 70% trend override, 30% floor for PCA)
        trend_w   = float(np.clip(abs(trend_z) / 2.0 * 0.70, 0.0, 0.70))
        z_effective = float(np.clip(trend_w * trend_z + (1.0 - trend_w) * z_pca, -2.5, 2.5))

        # ── Regime-conditioned factor loadings ────────────────────────────────
        lambda_mom  = float(np.clip(z_effective * 0.40  + 0.55, 0.05, 1.00))
        lambda_mom6 = float(np.clip(z_effective * 0.25  + 0.30, 0.00, 0.65))
        lambda_rev  = float(np.clip(-z_effective * 0.25 + 0.15, 0.00, 0.50))
        lambda_vol  = float(np.clip(0.25 + abs(z_mu_2) * 0.25,  0.10, 0.65))
        lambda_carry = float(np.clip(z_effective * 0.20 + 0.25,  0.00, 0.45))

        alpha_raw = (
            lambda_mom   * mom_arr[i]
            + lambda_mom6 * mom6_arr[i]
            + lambda_rev  * rev_arr[i]
            + lambda_vol  * vol_arr[i]
            + lambda_carry * carry_vec
            + _build_regime_tilt_signal(z_effective)
        )
        alpha_tanh = np.tanh(alpha_raw)

        # ── VIXY: vol-level conditioning on top of regime tilt ────────────────
        # Cross-asset realised vol in last 21d; suppress VIXY in genuinely low-vol envs
        cross_vol = float(np.nanstd(returns_df.iloc[max(0, i-21):i].values)) if i >= 5 else 0.01
        vixy_base  = float(np.tanh(-z_effective * 1.5))
        vixy_vcap  = float(np.clip(cross_vol / 0.012 - 0.3, 0.0, 1.0))
        alpha_tanh[vixy_idx] = vixy_base * vixy_vcap

        # ── Cash (SHV/BIL): BUG #18 FIX — always non-negative carry alpha ────
        for cash_idx in cash_idxs:
            # Bear/crisis: carry + defensive premium. Bull: mild carry only.
            alpha_tanh[cash_idx] = float(np.tanh(max(-z_effective * 0.25, 0.0) + 0.10))

        alpha_rows.append(alpha_tanh.astype(np.float32))

        if i % 200 == 0:
            top3 = sorted(zip(TICKERS, alpha_tanh), key=lambda x: x[1], reverse=True)[:3]
            bot3 = sorted(zip(TICKERS, alpha_tanh), key=lambda x: x[1])[:3]
            label = regime_df.loc[date, "regime_label"] if date in regime_df.index else "N/A"
            logger.info(
                f"  [{i}/{len(dates)}] {date.date()} | "
                f"z_pca={z_pca:+.2f} trend={trend_z:+.2f} z_eff={z_effective:+.2f} | "
                f"Regime={label} | "
                f"Top: {', '.join(f'{t}({s:.2f})' for t,s in top3)} | "
                f"Bot: {', '.join(f'{t}({s:.2f})' for t,s in bot3)}"
            )

    alpha_arr  = np.array(alpha_rows, dtype=np.float32)
    trend_arr  = np.array(trend_log)
    pct_bull   = float((trend_arr > 0.5).mean())
    pct_bear   = float((trend_arr < -0.5).mean())
    logger.info(
        f"Price-trend override: bull={pct_bull:.1%}  "
        f"neutral={1-pct_bull-pct_bear:.1%}  bear={pct_bear:.1%} | "
        f"mean={trend_arr.mean():+.3f}  std={trend_arr.std():.3f}"
    )

    return pd.DataFrame(alpha_arr, index=dates, columns=[f"alpha_{t}" for t in TICKERS])


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("══════ Fortress v5 — Alpha Signal Precomputation ══════")

    for req_path in (_PRICES_PATH, _RETURNS_PATH, _REGIME_PATH):
        if not req_path.exists():
            logger.error(f"Required cache file missing: {req_path}.")
            sys.exit(1)

    logger.info("Loading cached market data...")
    prices_df  = pd.read_parquet(_PRICES_PATH)
    returns_df = pd.read_parquet(_RETURNS_PATH)
    logger.info("Loading cached regime posteriors...")
    regime_df  = pd.read_parquet(_REGIME_PATH)

    for df in (prices_df, returns_df, regime_df):
        df.index = pd.to_datetime(df.index)

    returns_df = returns_df[~returns_df.index.duplicated(keep="last")].sort_index()
    prices_df  = prices_df[~prices_df.index.duplicated(keep="last")].sort_index()
    regime_df  = regime_df[~regime_df.index.duplicated(keep="last")].sort_index()

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

    logger.info("Surrogate Mode: computing 6-factor regime-conditioned alpha model...")
    alpha_df = _compute_surrogate_alpha(prices_aligned, returns_aligned, regime_df)

    assert alpha_df.shape == (len(returns_aligned), N_ASSETS)
    assert (alpha_df.abs() <= 1.0 + 1e-5).all().all()

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    alpha_df.to_parquet(_ALPHA_OUT)
    logger.info(f"✅ Alpha signals saved → {_ALPHA_OUT} ({len(alpha_df)} rows × {N_ASSETS} assets)")

    logger.info("Alpha signal summary (time-averaged per asset):")
    mean_alpha = alpha_df.mean(axis=0).sort_values(ascending=False)
    for ticker_col, val in mean_alpha.items():
        ticker = ticker_col.replace("alpha_", "")
        bar    = "█" * int(abs(val) * 20)
        sign   = "+" if val >= 0 else "-"
        logger.info(f"  {ticker:6s}: {sign}{bar} ({val:+.3f})")

    logger.info("Precompute Stage 2 complete. Run run_standalone_backtest.py next.")


if __name__ == "__main__":
    main()