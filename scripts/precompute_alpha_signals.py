"""
FORTRESS v5 — scripts/precompute_alpha_signals.py
Path: scripts/precompute_alpha_signals.py  [v7 — CASH TRAP FIX + MOMENTUM]

ROOT CAUSE ANALYSIS OF v6 FAILURE:
  The backtest produced Sharpe -2.843, CAGR +0.69%, hit rate 21.94%.
  The portfolio was a money market fund, not a diversified strategy.

  Three compounding bugs caused this:

  BUG A — Low-vol signal gave BIL/SHV maximum positive alpha:
    BIL vol ≈ 0.1%/year → ranked #1 by -rank(vol) → alpha(BIL) = +0.45
    SHV vol ≈ 0.1%/year → ranked #2 → alpha(SHV) = +0.43
    SPY vol ≈ 15%/year  → ranked near-last → alpha(SPY) = -0.15
    Since low-vol has 100% active rate and 0.435 mean amplitude, it
    dominated the blend. Every day: long BIL/SHV, short SPY/QQQ.
    2019-2024: SPY +15%/year, BIL +5%/year → persistent losses.
    FIX: Zero out BIL/SHV/VIXY from low-vol ranking. These are
    parking/hedging instruments, not cross-sectional alpha assets.

  BUG B — GATv2 signal router trained on broken (zero-VRP) signals:
    signal_router_latest.pt was trained BEFORE VRP/VTS were working.
    Training data: VRP=0, VTS=0, insider=noisy 100%, low_vol=dominant.
    Router learned: assign all weight to low_vol → compounded BUG A.
    FIX: Bypass GATv2 router. Use calibrated static signal weights
    until the router can be retrained on clean signals.

  BUG C — λ_turn=0.10 overcorrected turnover (32% → 2.63%):
    Once portfolio settled into BIL/SHV during warmup, λ_turn frozen
    it there. Turnover 2.63% → portfolio effectively static.
    FIX: λ_turn is a backtest parameter, not precompute. See backtest.

  SIGNAL ARCHITECTURE CHANGES:
    - BIL/SHV/VIXY zeroed in low-vol signal (parking assets excluded)
    - Added cross-sectional momentum signal (F6: 12-1M, skip 1M)
    - Static signal weights replace GATv2 routing (awaiting retrain)
    - Stage 5 regime gate reduced: crisis_scale max 0.30 (was 0.70)
    - Safe-haven override removed (MVO handles allocation, not alpha)

SIGNAL STACK (6 signals):
  [0] VRP  — Variance risk premium CS z-score         weight 0.30
  [1] VTS  — VIX term structure delta z-score          weight 0.20
  [2] MOM  — Cross-sectional 12-1M momentum            weight 0.30
  [3] NAV  — ETF NAV/AP stress (sparse)                weight 0.05
  [4] INS  — SEC insider cluster (IWM + sectors)       weight 0.05
  [5] LOV  — Low-vol anomaly (BIL/SHV/VIXY excluded)  weight 0.10

Total weight: 1.00. Static until GATv2 retrained on clean signals.

RUN:
  PYTHONPATH=. python scripts/precompute_alpha_signals.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("PrecomputeAlpha")

# ── Paths ──────────────────────────────────────────────────────────────────────
_BASE_DIR   = Path(".")
_CACHE_DIR  = _BASE_DIR / "research" / "outputs" / "cache"
_WEIGHTS_DIR = _BASE_DIR / "models" / "weights"

_PRICES_PATH  = _CACHE_DIR / "prices_wide.parquet"
_RETURNS_PATH = _CACHE_DIR / "returns_wide.parquet"
_REGIME_OUT   = _CACHE_DIR / "regime_posteriors.parquet"
_SIGNALS_OUT  = _CACHE_DIR / "alpha_signals.parquet"
_ALPHA_OUT    = _CACHE_DIR / "alpha_signals_blended.parquet"

_ROUTER_WEIGHTS = _WEIGHTS_DIR / "signal_router_latest.pt"

_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Universe ───────────────────────────────────────────────────────────────────
TICKERS: List[str] = [
    "SPY", "QQQ", "IWM", "TLT", "HYG", "LQD", "GLD", "SLV",
    "GDX", "XLE", "XLF", "XLK", "XLV", "XLU", "XLI", "XLP",
    "XLY", "XLB", "XLC", "VIXY", "BIL", "SHV", "USO", "PDBC", "COWZ",
]
N_ASSETS = len(TICKERS)

# Parking assets — excluded from ALL cross-sectional alpha ranking
# These are held by the MVO only as cash substitutes, never as alpha targets
_PARKING_ASSETS = {"BIL", "SHV"}

# Hedging/vol assets — excluded from momentum and low-vol signals
_HEDGE_ASSETS = {"BIL", "SHV", "VIXY"}

# Signal names
SIGNAL_NAMES: List[str] = ["vrp", "vts", "mom", "nav_arb", "insider", "low_vol"]
N_SIGNALS = len(SIGNAL_NAMES)

# ── Static signal weights (bypassing GATv2 until retrained on clean signals) ──
# GATv2 was trained on pre-fix data (VRP=0, VTS=0) → routing is wrong.
# Calibrated weights based on expected IC from literature:
#   VRP IC ≈ 0.06 (CBOE vol premium, well-documented)
#   MOM IC ≈ 0.07 (cross-sectional momentum in ETFs, 30y evidence)
#   VTS IC ≈ 0.05 (term structure direction change)
#   LOV IC ≈ 0.03 (low-vol anomaly, weak in ETF universe)
#   NAV IC ≈ 0.04 (sparse but high when active)
#   INS IC ≈ 0.03 (very sparse — IWM only)
_STATIC_SIGNAL_WEIGHTS: Dict[str, float] = {
    "vrp":     0.30,
    "vts":     0.20,
    "mom":     0.30,
    "nav_arb": 0.05,
    "insider": 0.05,
    "low_vol": 0.10,
}


# ─────────────────────────────────────────────────────────────────────────────
# Stage 0: Multi-asset vol regime
# ─────────────────────────────────────────────────────────────────────────────

async def stage0_vol_regime(start: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute VolRegimeTensor replacing Mamba-KAN. See signals/vol_regime.py."""
    from signals.vol_regime import MultiAssetVolRegime
    logger.info("Stage 0: Multi-asset vol regime construction...")
    engine = MultiAssetVolRegime()
    await engine.load_history(start=start)
    regime_df, meta_df = engine.get_tensor_series(tickers=TICKERS)
    regime_df.to_parquet(_REGIME_OUT)
    logger.info(f"  ✓ Regime posteriors → {_REGIME_OUT} ({len(regime_df)} rows)")
    if "equity_label" in meta_df.columns:
        total = len(meta_df)
        for label in ["crisis", "stress", "neutral", "complacent"]:
            n = (meta_df["equity_label"] == label).sum()
            logger.info(f"  Equity regime [{label}]: {n} days ({n/total*100:.1f}%)")
    return regime_df, meta_df


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Options surface alpha (VRP + VTS)
# ─────────────────────────────────────────────────────────────────────────────

async def stage1_options_alpha(start: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """VRP and VTS signals from CBOE vol indices. Fully vectorised."""
    from signals.options_alpha import OptionsAlphaEngine
    logger.info("Stage 1: Options surface alpha (CBOE vol indices)...")
    engine = OptionsAlphaEngine()
    await engine.load_data(start=start)
    vrp_df = engine.compute_vrp_history()
    vts_df = engine.compute_vts_history()
    stats  = engine.get_signal_summary()
    logger.info(
        f"  ✓ VRP: mean|α|={stats['vrp_mean_abs']:.3f} active={stats['vrp_active_pct']:.1f}% | "
        f"VTS: mean|α|={stats['vts_mean_abs']:.3f} active={stats['vts_active_pct']:.1f}%"
    )
    return vrp_df, vts_df


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1b: Cross-sectional momentum (NEW — primary alpha source)
# ─────────────────────────────────────────────────────────────────────────────

def stage1b_momentum(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional 12-1M momentum signal.

    ALPHA THESIS:
      Cross-sectional momentum is the most robust documented anomaly in
      equity and multi-asset ETF universes (Jegadeesh & Titman 1993,
      Asness et al. 2013 across 5 asset classes). The mechanism is
      underreaction to fundamental information that takes 6-12 months to
      fully diffuse into prices.

    SIGNAL CONSTRUCTION:
      MOM_i(t) = cumulative return of asset i from t-252 to t-21
                 (12 months total, skipping the most recent month)
      The 1-month skip avoids the reversal effect — the most recent
      month exhibits reversal (not continuation) due to liquidity effects.

      Cross-sectional z-score applied daily:
        mom_z_i(t) = (MOM_i(t) - mean_j(MOM_j(t))) / std_j(MOM_j(t))

    EXCLUSIONS:
      BIL, SHV, VIXY excluded from ranking (parking/hedge assets).
      These are set to 0 — the MVO handles their allocation.
      Including them distorts the cross-section: BIL and SHV momentum
      is always near-zero regardless of market conditions, creating a
      gravity toward zero that dampens the signal.

    REGIME INTERACTION:
      Momentum is most reliable in trending markets (bull/neutral).
      In crisis regimes, momentum frequently fails (crash risk).
      We rely on the MVO's crisis_weight alpha_scale to handle this,
      not signal-level gating (cleaner separation of concerns).

    Returns:
      DataFrame (T, N) — cross-sectional momentum z-scores in tanh([-1,1])
    """
    logger.info("Stage 1b: Cross-sectional momentum (12-1M)...")
    ret = returns_df.reindex(columns=TICKERS)

    # Cumulative return from t-252 to t-21 (12 months, skip 1 month)
    # Using log returns for compounding accuracy
    log_ret = np.log1p(ret.fillna(0.0))
    # Rolling sum of log returns from t-252 to t-21
    cum_long  = log_ret.rolling(252, min_periods=126).sum()
    cum_short = log_ret.rolling(21,  min_periods=10).sum()
    mom_raw   = cum_long - cum_short  # 12M-1M momentum

    # Zero out parking/hedging assets
    for t in _HEDGE_ASSETS:
        if t in mom_raw.columns:
            mom_raw[t] = 0.0

    # Cross-sectional z-score: rank assets by momentum on each date
    # Only use non-zeroed assets for mean/std computation
    signal_mask = pd.Series({t: t not in _HEDGE_ASSETS for t in TICKERS})
    active_cols = [t for t in TICKERS if signal_mask[t]]

    cs_mean = mom_raw[active_cols].mean(axis=1)
    cs_std  = mom_raw[active_cols].std(axis=1).clip(lower=1e-6)

    # Broadcast z-score only to active tickers
    mom_z = mom_raw.copy()
    for col in active_cols:
        mom_z[col] = (mom_raw[col] - cs_mean) / cs_std

    mom_z[list(_HEDGE_ASSETS)] = 0.0

    result = np.tanh(mom_z * 0.5)  # dampen extremes
    result = result.fillna(0.0)

    logger.info(
        f"  ✓ Momentum signal: {len(result)} days | "
        f"Mean |signal| (active): {result[active_cols].abs().mean().mean():.3f}"
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: ETF NAV / AP stress
# ─────────────────────────────────────────────────────────────────────────────

async def stage2_nav_arb(start: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """AP capacity stress signal for bond/commodity ETFs. Sparse by design."""
    from signals.etf_nav_arb import ETFNavArbSignal
    logger.info("Stage 2: ETF NAV / AP stress signal...")
    engine = ETFNavArbSignal()
    await engine.load_data(start=start)
    signal_df, stress_meta = engine.compute_full_history()
    logger.info(
        f"  ✓ NAV arb: {len(signal_df)} days | "
        f"Active (AP stress): {(stress_meta['n_active'] > 0).mean()*100:.1f}%"
    )
    return signal_df, stress_meta


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: SEC filing intelligence
# ─────────────────────────────────────────────────────────────────────────────

async def stage3_sec_insider(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """IWM insider cluster buying + activist sector signals. Narrow, high-conviction."""
    from signals.sec_insider import SECFilingIntelligence
    logger.info("Stage 3: SEC filing intelligence (IWM insider + activist)...")
    engine = SECFilingIntelligence(lookback_days=30)

    weekly_dates   = dates[::5]
    weekly_signals: Dict[str, pd.Series] = {}
    n_computed     = 0

    for date in weekly_dates:
        date_str = str(date.date())
        try:
            alpha = await engine.get_combined_alpha_vector(date_str)
            weekly_signals[date_str] = alpha
            n_computed += 1
            if n_computed % 10 == 0:
                logger.info(f"  SEC signals: {n_computed}/{len(weekly_dates)} weeks")
        except Exception as e:
            logger.debug(f"SEC signal failed {date_str}: {e}")
            weekly_signals[date_str] = pd.Series(0.0, index=TICKERS)
        await asyncio.sleep(0.5)

    weekly_df = pd.DataFrame(weekly_signals).T
    weekly_df.index = pd.to_datetime(weekly_df.index)
    insider_df = weekly_df.reindex(dates).ffill().fillna(0.0)

    n_nonzero = (insider_df.abs() > 0.01).any(axis=1).mean() * 100
    n_iwm_active = (insider_df["IWM"].abs() > 0.01).mean() * 100 if "IWM" in insider_df else 0.0
    logger.info(
        f"  ✓ SEC insider: {len(insider_df)} days | "
        f"Any signal: {n_nonzero:.1f}% | IWM active: {n_iwm_active:.1f}%"
    )
    return insider_df


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4: Low-volatility anomaly (FIXED — parking assets excluded)
# ─────────────────────────────────────────────────────────────────────────────

def stage4_low_vol(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Low-volatility anomaly — FIXED to exclude parking/hedge assets.

    BUG FIX:
      Previous version ranked ALL 25 assets by realized vol. BIL/SHV
      have vol ≈ 0.1%/year → always ranked #1 and #2 → alpha(BIL) = +0.45.
      This created a permanent "go long cash" signal that dominated the blend.

    FIX:
      1. BIL, SHV, VIXY are zeroed before cross-sectional ranking.
         They're parking/hedging instruments, not return-generating assets.
         Their vol ranking carries no predictive information for the returns
         of the remaining 22 investable assets.

      2. The remaining 22 assets are ranked within their own cross-section.
         This preserves the within-universe low-vol anomaly: among investable
         ETFs, lower-vol assets (staples, utilities, bonds) have historically
         earned higher RISK-ADJUSTED returns than high-vol assets (energy, metals).

      3. Signal weight remains low (0.10) — it's a tiebreaker, not the
         primary alpha source. MOM and VRP carry the directional content.

    ECONOMIC RATIONALE:
      The low-vol anomaly works within homogeneous asset classes (equity-vs-equity,
      bond-vs-bond). Across heterogeneous asset classes (equity vs cash vs vol
      products), realized vol differences reflect risk structure, not mispricing.
      Cross-sectional z-scoring across BIL + VIXY + SPY is economically meaningless.
    """
    logger.info("Stage 4: Low-vol anomaly (parking assets excluded)...")
    rv_63 = (
        returns_df
        .reindex(columns=TICKERS)
        .rolling(63, min_periods=20)
        .std() * np.sqrt(252)
    )
    rv_63 = rv_63.ffill()

    # Active investable universe (excludes parking + hedging assets)
    active_cols = [t for t in TICKERS if t not in _HEDGE_ASSETS]

    # Rank within active universe only
    rank_active = rv_63[active_cols].rank(axis=1, pct=True)
    signal_active = -(rank_active - 0.5) * 2.0  # centered at 0, range [-1, 1]

    # Assemble full (T, N) signal matrix — parking assets get exactly 0
    result = pd.DataFrame(0.0, index=returns_df.index, columns=TICKERS)
    result[active_cols] = np.tanh(signal_active)
    result = result.fillna(0.0)

    mean_abs_active = result[active_cols].abs().mean().mean()
    logger.info(
        f"  ✓ Low-vol signal: {len(result)} days | "
        f"Mean |signal| (active 22): {mean_abs_active:.3f} | "
        f"BIL/SHV/VIXY: zeroed"
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5: Signal blending (static weights — GATv2 bypassed until retrained)
# ─────────────────────────────────────────────────────────────────────────────

def stage5_blend_signals(
    signal_dfs:  Dict[str, pd.DataFrame],
    regime_df:   pd.DataFrame,
    returns_df:  pd.DataFrame,
    dates:       pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Blend 6 signals into final alpha using static calibrated weights.

    GATv2 BYPASS RATIONALE:
      The signal_router_latest.pt was trained before VRP/VTS signals were
      working. It learned to weight everything toward low_vol (the only
      real signal during training). Applied to the fixed signal stack, it
      amplifies the cash-trap bias instead of routing correctly.

      The router MUST be retrained from scratch on the current clean
      signal stack before being re-enabled. Until then, static weights
      calibrated from expected IC estimates give better OOS performance
      than a miscalibrated neural router.

      Even the best static weights will underperform a properly trained
      router; however, they are guaranteed not to INVERT the signal
      direction, which the current router does.

    REGIME GATE (REDUCED):
      Previous version used crisis_scale = urgency × 0.70.
      This was too aggressive — even 30% urgency caused 21% signal
      compression, systematically biasing toward safe havens.

      New version: crisis_scale = max(urgency - 0.60, 0) × 0.40
      Gate only fires when urgency > 0.60 (genuine crisis, not stress).
      Maximum compression is 40% (was 70%).
      The MVO risk aversion (λ_var scaling via z0) handles the rest.

    SAFE HAVEN OVERRIDE REMOVED:
      Previous version added safe_haven_allocation × crisis_scale.
      This created alpha(BIL) = +0.8 during any stress period.
      The MVO already handles crisis allocation via:
        - λ_var scaling (reduces risk budget in crisis)
        - equity_urgency routing (reduces λ_turn in crisis for fast rebalancing)
      Double-handling via signal manipulation creates instability.
    """
    logger.info("Stage 5: Signal blending (static weights — GATv2 bypassed)...")

    T = len(dates)
    N = N_ASSETS

    # Build signal stack (T, N, S)
    S = N_SIGNALS
    signal_stack = np.zeros((T, N, S), dtype=np.float32)

    for s_idx, sig_name in enumerate(SIGNAL_NAMES):
        if sig_name in signal_dfs:
            df = (
                signal_dfs[sig_name]
                .reindex(dates)
                .ffill()
                .fillna(0.0)
                .reindex(columns=TICKERS)
                .fillna(0.0)
            )
            signal_stack[:, :, s_idx] = df.values.astype(np.float32)
        else:
            logger.warning(f"Signal '{sig_name}' missing from signal_dfs — using zeros.")

    # ── Static signal weights ──────────────────────────────────────────────────
    weights_vec = np.array([
        _STATIC_SIGNAL_WEIGHTS.get(n, 0.0) for n in SIGNAL_NAMES
    ], dtype=np.float32)
    weights_vec /= weights_vec.sum()  # ensure exact sum = 1.0

    # Broadcast weights to (T, N, S) — same weight for all assets and dates
    blending_weights = np.broadcast_to(
        weights_vec[np.newaxis, np.newaxis, :],
        (T, N, S),
    ).copy()

    logger.info(
        f"  Static weights: "
        + " | ".join(f"{n}={w:.2f}" for n, w in zip(SIGNAL_NAMES, weights_vec))
    )

    # ── Attempt GATv2 router if weights exist (log warning if found — needs retrain) ──
    if _ROUTER_WEIGHTS.exists():
        logger.warning(
            f"  ⚠️  GATv2 weights found at {_ROUTER_WEIGHTS} but BYPASSED. "
            f"Router was trained on pre-fix signals (VRP=0, VTS=0). "
            f"Delete the file and run training/train_signal_router.py on the "
            f"current signal stack before re-enabling. Using static weights."
        )

    # ── Blended alpha: weighted sum over signals ───────────────────────────────
    alpha_raw = (blending_weights * signal_stack).sum(axis=-1)  # (T, N)

    # ── Reduced regime gate ────────────────────────────────────────────────────
    # Gate only fires in genuine crisis (urgency > 0.60), max 40% compression.
    # Safe-haven override REMOVED — MVO handles this via risk aversion scaling.
    if "ltc_urgency" in regime_df.columns:
        regime_reindexed = regime_df.reindex(dates).ffill()
        base_urgency     = regime_reindexed["ltc_urgency"].fillna(0.0).values

        for i, ticker in enumerate(TICKERS):
            # Route urgency to correct asset-class axis
            if ticker in ("TLT", "LQD", "BIL", "SHV"):
                asset_urgency = base_urgency  # bond assets use full urgency
            elif ticker in ("GLD", "SLV", "GDX", "USO", "PDBC"):
                asset_urgency = base_urgency * 0.5  # commodity assets: partial
            else:
                asset_urgency = base_urgency  # equity assets

            # Compress only in genuine crisis (urgency > 0.60)
            crisis_scale = np.clip((asset_urgency - 0.60) / 0.40, 0.0, 1.0) * 0.40
            alpha_raw[:, i] = alpha_raw[:, i] * (1.0 - crisis_scale)
    else:
        logger.info("  ltc_urgency not in regime_df — regime gate inactive.")

    # ── Final tanh squash ──────────────────────────────────────────────────────
    alpha_final = np.tanh(alpha_raw).astype(np.float32)

    result_df = pd.DataFrame(alpha_final, index=dates, columns=TICKERS)

    # Diagnostic: log top and bottom 5 assets by mean alpha
    means = result_df.mean().sort_values(ascending=False)
    logger.info(f"  ✓ Blended alpha: {len(result_df)} days × {N} assets | "
                f"Mean |alpha|: {result_df.abs().mean().mean():.3f} | "
                f"Max: {result_df.max().max():.3f}")
    logger.info(f"  Top 5: {dict(means.head().round(3))}")
    logger.info(f"  Bot 5: {dict(means.tail().round(3))}")

    # Sanity check: parking assets should have near-zero alpha
    for t in _PARKING_ASSETS:
        mean_alpha = float(result_df[t].mean())
        if abs(mean_alpha) > 0.15:
            logger.warning(
                f"  ⚠️  {t} mean alpha={mean_alpha:.3f} — expected ~0. "
                f"Check if signal zeroing is working."
            )
        else:
            logger.info(f"  ✓ {t} mean alpha={mean_alpha:.3f} (correctly near-zero)")

    return result_df


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    logger.info("══════ Fortress v5 — Alpha Precompute v7 (CASH TRAP FIX + MOMENTUM) ══════")

    for path in [_PRICES_PATH, _RETURNS_PATH]:
        if not path.exists():
            logger.error(f"Missing: {path}. Run data ingestion first.")
            sys.exit(1)

    logger.info("Loading market data...")
    prices_df  = pd.read_parquet(_PRICES_PATH)
    returns_df = pd.read_parquet(_RETURNS_PATH)

    for df in [prices_df, returns_df]:
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        if df.index.duplicated().any():
            df.drop(df.index[df.index.duplicated(keep="last")], inplace=True)

    start_date = str(returns_df.index[0].date())
    dates      = returns_df.index

    logger.info(
        f"Market data: {len(returns_df)} days × {N_ASSETS} assets | "
        f"Start: {start_date} | End: {str(dates[-1].date())}"
    )

    # ── Stage 0 ────────────────────────────────────────────────────────────────
    regime_df, regime_meta = await stage0_vol_regime(start=start_date)
    regime_df.index = pd.to_datetime(regime_df.index)

    # ── Stage 1: VRP + VTS ────────────────────────────────────────────────────
    vrp_df, vts_df = await stage1_options_alpha(start=start_date)
    vrp_df.index   = pd.to_datetime(vrp_df.index)
    vts_df.index   = pd.to_datetime(vts_df.index)

    # ── Stage 1b: Momentum ────────────────────────────────────────────────────
    mom_df = stage1b_momentum(returns_df)
    mom_df.index = pd.to_datetime(mom_df.index)

    # ── Stage 2: ETF NAV arb ──────────────────────────────────────────────────
    nav_df, stress_meta = await stage2_nav_arb(start=start_date)
    nav_df.index = pd.to_datetime(nav_df.index)

    # ── Stage 3: SEC insider ──────────────────────────────────────────────────
    insider_df = await stage3_sec_insider(dates=dates)
    insider_df.index = pd.to_datetime(insider_df.index)

    # ── Stage 4: Low-vol (FIXED) ──────────────────────────────────────────────
    lowvol_df = stage4_low_vol(returns_df)
    lowvol_df.index = pd.to_datetime(lowvol_df.index)

    # ── Save per-signal matrix ────────────────────────────────────────────────
    signal_dfs: Dict[str, pd.DataFrame] = {
        "vrp":     vrp_df,
        "vts":     vts_df,
        "mom":     mom_df,
        "nav_arb": nav_df,
        "insider": insider_df,
        "low_vol": lowvol_df,
    }

    long_frames = []
    for sig_name, df in signal_dfs.items():
        df_aligned = df.reindex(dates).ffill().fillna(0.0).reindex(columns=TICKERS).fillna(0.0)
        df_long    = df_aligned.copy()
        df_long.columns = [f"{sig_name}_{t}" for t in df_long.columns]
        long_frames.append(df_long)

    signals_long_df = pd.concat(long_frames, axis=1)
    signals_long_df.to_parquet(_SIGNALS_OUT)
    logger.info(f"✓ Per-signal matrix → {_SIGNALS_OUT} ({signals_long_df.shape})")

    # ── Stage 5: Blend ────────────────────────────────────────────────────────
    alpha_df = stage5_blend_signals(
        signal_dfs=signal_dfs,
        regime_df=regime_df,
        returns_df=returns_df,
        dates=dates,
    )

    # Validate
    n_nans = alpha_df.isnull().sum().sum()
    if n_nans > 0:
        logger.warning(f"Caught {n_nans} NaN values (warmup). Filling with 0.")
        alpha_df = alpha_df.fillna(0.0)

    assert alpha_df.shape == (len(dates), N_ASSETS), f"Shape: {alpha_df.shape}"
    assert (alpha_df.abs() <= 1.0 + 1e-5).all().all(), "Values outside [-1,1]"

    alpha_df.to_parquet(_ALPHA_OUT)
    logger.info(f"✓ Blended alpha → {_ALPHA_OUT} ({alpha_df.shape})")

    # ── Per-signal summary ────────────────────────────────────────────────────
    logger.info("Per-signal statistics:")
    for sig_name, df in signal_dfs.items():
        df_a    = df.reindex(dates).ffill().fillna(0.0).reindex(columns=TICKERS).fillna(0.0)
        nonzero = (df_a.abs() > 0.01).any(axis=1).mean() * 100
        logger.info(
            f"  {sig_name:8s}: mean|α|={df_a.abs().mean().mean():.3f} | "
            f"active={nonzero:.1f}% | "
            f"BIL={df_a['BIL'].mean():+.3f} SHV={df_a['SHV'].mean():+.3f} "
            f"SPY={df_a['SPY'].mean():+.3f} QQQ={df_a['QQQ'].mean():+.3f}"
        )

    logger.info(
        "\n══════ Stage 2 COMPLETE ══════\n"
        "Next steps:\n"
        "  1. python scripts/run_standalone_backtest.py\n"
        "  2. Check: alpha(BIL) ≈ 0, alpha(SPY) > -0.05\n"
        "  3. Check: daily turnover 10-20%, hit rate > 48%\n"
        "  4. If signals stable: python training/train_signal_router.py\n"
        "     (retrain GATv2 on clean signal stack, then re-enable)\n"
    )


if __name__ == "__main__":
    asyncio.run(main())