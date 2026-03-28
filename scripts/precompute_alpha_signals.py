"""
FORTRESS v5 - scripts/precompute_alpha_signals.py  [v14 — GEX RESTORED + SECTOR BETAS]

CHANGES FROM v13 (the broken run):
  CRITICAL FIX: stage5_blend_signals was zeroing GEX for ALL assets, discarding the
  entire +0.0629 IC edge and leaving the optimizer running on pure Momentum+LowVol.
  That signal is anti-predictive in the F8/F9 Mag7 regime.

v14 FIXES:
  1. GEX restored to equity alpha blend at weight 0.60 (sector-attenuated via GEX_BETA)
  2. Attenuated GEX also exported separately as gex_alpha.parquet for the dynamic
     equity envelope constraint in run_standalone_backtest.py
  3. Breadth-adaptive momentum suppression retained from v13.1
  4. Unified equity_set with backtest _EQUITY_TICKERS (14 assets)

SIGNAL STACK (restored):
  EQUITY BUCKET:
    [0] GEX_FLOW — Sector-attenuated dealer gamma     weight 0.60
    [1] MOM      — 12-1M CS momentum, breadth-gated   weight 0.25
    [2] LOV      — Low-vol anomaly proxy               weight 0.15
  NON-EQUITY BUCKET:
    [0] GEX_FLOW — zeroed (no options chain signal)   weight 0.00
    [1] MOM      — cross-sectional momentum            weight 0.60
    [2] LOV      — low-vol proxy                      weight 0.40

RUN SEQUENCE:
  1. PYTHONPATH=. python signals/options_flow.py
  2. PYTHONPATH=. python scripts/precompute_alpha_signals.py
  3. PYTHONPATH=. python scripts/run_standalone_backtest.py
"""
from __future__ import annotations

import asyncio
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
logger = logging.getLogger("Ouroboros.AlphaPrecompute")

_BASE_DIR    = Path(".")
_CACHE_DIR   = _BASE_DIR / "research" / "outputs" / "cache"
_WEIGHTS_DIR = _BASE_DIR / "models" / "weights"

_PRICES_PATH    = _CACHE_DIR / "prices_wide.parquet"
_RETURNS_PATH   = _CACHE_DIR / "returns_wide.parquet"
_REGIME_OUT     = _CACHE_DIR / "regime_posteriors.parquet"
_SIGNALS_OUT    = _CACHE_DIR / "alpha_signals.parquet"
_ALPHA_OUT      = _CACHE_DIR / "alpha_signals_blended.parquet"
_GEX_ALPHA_OUT  = _CACHE_DIR / "gex_alpha.parquet"   # raw attenuated GEX for equity envelope
_PC_FLOW_CACHE  = _CACHE_DIR / "options_flow_pc.parquet"

_CACHE_DIR.mkdir(parents=True, exist_ok=True)

TICKERS: List[str] = [
    "SPY", "QQQ", "IWM", "TLT", "HYG", "LQD", "GLD", "SLV",
    "GDX", "XLE", "XLF", "XLK", "XLV", "XLU", "XLI", "XLP",
    "XLY", "XLB", "XLC", "VIXY", "BIL", "SHV", "USO", "PDBC", "COWZ",
]
N_ASSETS = len(TICKERS)

_PARKING_ASSETS: frozenset[str] = frozenset({"BIL", "SHV"})
_HEDGE_ASSETS:   frozenset[str] = frozenset({"BIL", "SHV", "VIXY"})

# UNIFIED with backtest _EQUITY_TICKERS — 14 assets
_EQUITY_SET: frozenset[str] = frozenset({
    "SPY", "QQQ", "IWM", "XLK", "XLF", "XLV", "XLU", "XLI", "XLP",
    "XLY", "XLB", "XLC", "COWZ", "XLE",
})

_EQUITY_TICKERS_FOR_MOM_GATE: frozenset[str] = frozenset({
    "SPY", "QQQ", "IWM", "XLK", "XLF", "XLV", "XLU",
    "XLI", "XLP", "XLY", "XLB", "XLC", "GDX", "XLE", "COWZ",
})
_MOM_CRASH_URGENCY_GATE:     float = 0.65
_MOM_DISPERSION_LOOKBACK:    int   = 252
_MOM_DISPERSION_QUANTILE:    float = 0.90
_MOM_DISPERSION_MEAN_THRESH: float = 0.25

SIGNAL_NAMES: List[str] = ["gex_flow", "mom", "low_vol"]
N_SIGNALS = len(SIGNAL_NAMES)

# Sector GEX beta map — attenuates the aggregate SPX GEX scalar
# to approximate implied dealer gamma for each sector ETF
GEX_BETA: Dict[str, float] = {
    "SPY":  1.00, "QQQ":  0.95, "IWM":  0.40,
    "XLK":  0.85, "XLC":  0.75, "XLY":  0.65,
    "XLF":  0.55, "XLV":  0.50, "XLI":  0.50,
    "XLU":  0.30, "XLP":  0.30, "XLB":  0.40,
    "XLE":  0.35, "COWZ": 0.45, "GDX":  0.15,
}


async def stage0_vol_regime(start: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    from signals.vol_regime import MultiAssetVolRegime
    logger.info("Stage 0: Multi-asset vol regime construction...")
    engine = MultiAssetVolRegime()
    await engine.load_history(start=start)
    regime_df, meta_df = engine.get_tensor_series(tickers=TICKERS)
    regime_df.to_parquet(_REGIME_OUT)
    logger.info(f"  Regime posteriors -> {_REGIME_OUT} ({len(regime_df)} rows)")
    if "equity_label" in meta_df.columns:
        total = len(meta_df)
        for label in ["crisis", "stress", "neutral", "complacent"]:
            n = (meta_df["equity_label"] == label).sum()
            logger.info(f"  Equity regime [{label}]: {n} days ({n/total*100:.1f}%)")
    return regime_df, meta_df


async def stage1a_dealer_gamma(
    start: str,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Institutional DIX/GEX Flow — attenuated by GEX_BETA.

    Returns a DataFrame with sector-specific GEX signals. The aggregate SPX GEX
    scalar is multiplied by each sector's beta to create a non-uniform cross-
    sectional gradient that prevents flat-alpha Markowitz Ping-Pong.

    Non-equity assets (TLT, GLD, etc.) remain zeroed — GEX has no explanatory
    power there.
    """
    logger.info("Stage 1a: Institutional GEX/DIX flow signal (Sector-Attenuated)...")

    if _PC_FLOW_CACHE.exists():
        try:
            cached_df = pd.read_parquet(_PC_FLOW_CACHE)
            cached_df.index = pd.to_datetime(cached_df.index)
            if len(cached_df) > 100:
                df = (
                    cached_df
                    .reindex(dates).ffill().fillna(0.0)
                    .reindex(columns=TICKERS).fillna(0.0)
                )
                # Apply sector GEX betas — creates non-uniform alpha gradient
                for ticker, beta in GEX_BETA.items():
                    if ticker in df.columns:
                        df[ticker] *= beta
                # Zero out non-equity assets explicitly
                for ticker in TICKERS:
                    if ticker not in _EQUITY_SET and ticker in df.columns:
                        df[ticker] = 0.0

                logger.info(
                    f"  GEX flow: loaded, sector-attenuated, non-equity zeroed | "
                    f"mean|a| equity={df[list(_EQUITY_SET & set(TICKERS))].abs().mean().mean():.4f}"
                )
                return df
        except Exception as e:
            logger.warning(f"  GEX flow cache load failed ({e}), regenerating...")

    logger.error(
        "  GEX flow cache missing. "
        "Run: PYTHONPATH=. python signals/options_flow.py first.\n"
        "  Falling back to zero signal — F8/F9 will fail without this."
    )
    return pd.DataFrame(0.0, index=dates, columns=TICKERS)


def stage1b_momentum(
    returns_df: pd.DataFrame,
    regime_df:  Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Cross-sectional 12-1M momentum with dual crash gate and breadth-adaptive
    self-suppression.

    When cross-sectional dispersion (cs_std) drops below its historical mean
    (a low-breadth, concentrated rally), the momentum signal scales itself down.
    This naturally attenuates momentum in Mag7-style regimes without any
    hard-coded regime gate or lookahead.
    """
    logger.info("Stage 1b: Cross-sectional momentum (12-1M)...")
    ret = returns_df.reindex(columns=TICKERS)

    log_ret   = np.log1p(ret.fillna(0.0))
    cum_long  = log_ret.rolling(252, min_periods=126).sum()
    cum_short = log_ret.rolling(21,  min_periods=10).sum()
    mom_raw   = cum_long - cum_short

    for t in _HEDGE_ASSETS:
        if t in mom_raw.columns:
            mom_raw[t] = 0.0

    active_cols      = [t for t in TICKERS if t not in _HEDGE_ASSETS]
    equity_gate_cols = [t for t in TICKERS if t in _EQUITY_TICKERS_FOR_MOM_GATE]

    cs_mean = mom_raw[active_cols].mean(axis=1)
    cs_std  = mom_raw[active_cols].std(axis=1).clip(lower=1e-6)

    mom_z = mom_raw.copy()
    for col in active_cols:
        mom_z[col] = (mom_raw[col] - cs_mean) / cs_std
    mom_z[list(_HEDGE_ASSETS)] = 0.0

    result = np.tanh(mom_z * 0.5).fillna(0.0)

    # Breadth-adaptive self-suppression:
    # When cs_std < historical mean (narrow rally), breadth_score < 1.0
    # and momentum scales down organically — no hard gate needed.
    rolling_mean_std = cs_std.rolling(252, min_periods=63).mean().clip(lower=1e-6)
    breadth_score = (cs_std / rolling_mean_std).clip(0.2, 2.0)
    result[active_cols] = result[active_cols].multiply(breadth_score, axis=0)

    n_combined = 0
    if regime_df is not None and "ltc_urgency" in regime_df.columns:
        urgency = regime_df["ltc_urgency"].reindex(result.index).ffill().fillna(0.0)
        urgency_gate: pd.Series = urgency > _MOM_CRASH_URGENCY_GATE

        eq_cols  = [c for c in equity_gate_cols if c in result.columns]
        mom_std  = result[eq_cols].std(axis=1)
        mom_mean = result[eq_cols].mean(axis=1)
        roll_90  = mom_std.rolling(_MOM_DISPERSION_LOOKBACK, min_periods=63).quantile(
            _MOM_DISPERSION_QUANTILE
        )
        dispersion_gate: pd.Series = (
            (mom_std > roll_90) & (mom_mean > _MOM_DISPERSION_MEAN_THRESH)
        )
        combined_gate = urgency_gate | dispersion_gate
        n_combined    = int(combined_gate.sum())

        for col in eq_cols:
            positive_on_gate = combined_gate & (result[col] > 0.0)
            result.loc[positive_on_gate, col] = 0.0

        logger.info(
            f"  Momentum crash gate: urgency={int(urgency_gate.sum())}d | "
            f"dispersion={int(dispersion_gate.sum())}d | "
            f"combined={n_combined}d ({n_combined/len(result)*100:.1f}%)"
        )
    else:
        logger.info("  Momentum crash gate: disabled (no regime_df or ltc_urgency)")

    logger.info(
        f"  Momentum: {len(result)} days | "
        f"Mean |signal| (active): {result[active_cols].abs().mean().mean():.3f}"
    )
    return result


def stage4_low_vol(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Low-vol anomaly proxy. Rank-based over 63d realized vol."""
    logger.info("Stage 4: Low-vol proxy...")
    rv_63 = (
        returns_df.reindex(columns=TICKERS)
        .rolling(63, min_periods=20).std() * np.sqrt(252)
    ).ffill()

    active_cols   = [t for t in TICKERS if t not in _HEDGE_ASSETS]
    rank_active   = rv_63[active_cols].rank(axis=1, pct=True)
    signal_active = (rank_active - 0.5) * 1.0

    result = pd.DataFrame(0.0, index=returns_df.index, columns=TICKERS)
    result[active_cols] = np.tanh(signal_active)
    result = result.fillna(0.0)

    qqq_mean = float(result["QQQ"].mean()) if "QQQ" in result.columns else 0.0
    logger.info(
        f"  Low-vol: {len(result)} days | "
        f"Mean |signal|: {result[active_cols].abs().mean().mean():.3f} | "
        f"QQQ mean: {qqq_mean:+.3f}"
    )
    return result


def stage5_blend_signals(
    signal_dfs: Dict[str, pd.DataFrame],
    regime_df:  pd.DataFrame,
    returns_df: pd.DataFrame,
    dates:      pd.DatetimeIndex,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Blend signals with Cross-Asset Domain Masking.

    Returns (alpha_blended_df, gex_alpha_df):
      - alpha_blended_df : the full blended alpha tensor fed to the MVO
      - gex_alpha_df     : the raw sector-attenuated GEX signal (equity assets
                           only, non-equity zeroed) for use in the dynamic equity
                           envelope constraint in the backtester

    EQUITY bucket  : GEX=0.60 + MOM=0.25 + LowVol=0.15
    NON-EQUITY bucket: GEX=0.00 + MOM=0.60 + LowVol=0.40

    The sector beta attenuation applied in stage1a ensures GEX is NOT uniform
    across the 14 equity ETFs — QQQ/SPY get beta≈1.0 while XLU/XLP get beta≈0.3,
    creating a genuine cross-sectional gradient for the optimizer to exploit.
    """
    logger.info("Stage 5: Blending signals with GEX restored to equity alpha (v14)...")
    N = N_ASSETS
    S = N_SIGNALS

    signal_arrays: Dict[str, np.ndarray] = {}
    for name in SIGNAL_NAMES:
        df = signal_dfs.get(name, pd.DataFrame(0.0, index=dates, columns=TICKERS))
        aligned = (
            df.reindex(dates).ffill().fillna(0.0)
            .reindex(columns=TICKERS).fillna(0.0)
        )
        signal_arrays[name] = aligned.values.astype(np.float32)

    T = len(dates)
    signal_stack = np.stack([signal_arrays[n] for n in SIGNAL_NAMES], axis=-1)

    blending_weights = np.zeros((T, N, S), dtype=np.float32)

    for i, ticker in enumerate(TICKERS):
        if ticker in _EQUITY_SET:
            # GEX is the primary structural edge for equities.
            # Its sector-attenuation (SPY=1.0, XLU=0.3, etc.) creates the
            # non-uniform cross-sectional gradient the MVO needs.
            blending_weights[:, i, 0] = 0.60  # gex_flow (sector-attenuated)
            blending_weights[:, i, 1] = 0.25  # mom (breadth-suppressed)
            blending_weights[:, i, 2] = 0.15  # low_vol
        else:
            # Non-equity: GEX has no edge here — Momentum dominates
            blending_weights[:, i, 0] = 0.00
            blending_weights[:, i, 1] = 0.60
            blending_weights[:, i, 2] = 0.40

    alpha_raw = (blending_weights * signal_stack).sum(axis=-1)

    # Regime-conditional alpha attenuation (unchanged from v13)
    if "ltc_urgency" in regime_df.columns:
        regime_reindexed = regime_df.reindex(dates).ffill()
        base_urgency     = regime_reindexed["ltc_urgency"].fillna(0.0).values
        for i, ticker in enumerate(TICKERS):
            if ticker in ("TLT", "LQD", "BIL", "SHV"):
                asset_urgency = base_urgency
            elif ticker in ("GLD", "SLV", "GDX", "USO", "PDBC"):
                asset_urgency = base_urgency * 0.5
            else:
                asset_urgency = base_urgency
            crisis_scale    = np.clip((asset_urgency - 0.60) / 0.40, 0.0, 1.0) * 0.40
            alpha_raw[:, i] *= (1.0 - crisis_scale)

    alpha_final = np.tanh(alpha_raw).astype(np.float32)
    result_df   = pd.DataFrame(alpha_final, index=dates, columns=TICKERS)

    # Build the separate raw GEX alpha output for the equity envelope constraint.
    # This is the sector-attenuated GEX BEFORE momentum mixing — it carries the
    # pure dealer-positioning signal for use in _gex_equity_cap().
    gex_raw_df = pd.DataFrame(
        signal_arrays["gex_flow"], index=dates, columns=TICKERS
    )

    eq_mean  = result_df[list(_EQUITY_SET & set(TICKERS))].abs().mean().mean()
    neq_mean = result_df[[t for t in TICKERS if t not in _EQUITY_SET]].abs().mean().mean()
    logger.info(
        f"  Blended alpha: {len(result_df)} days x {N} assets | "
        f"Mean |alpha| equity={eq_mean:.3f} | non-equity={neq_mean:.3f}"
    )

    return result_df, gex_raw_df


async def main() -> None:
    logger.info(
        "====== Ouroboros Alpha Precompute v14 "
        "(GEX RESTORED: equity gex=0.60|mom=0.25|lov=0.15, non-eq mom=0.60|lov=0.40) ======"
    )

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
        f"Market data: {len(returns_df)} days x {N_ASSETS} assets | "
        f"Start: {start_date} | End: {str(dates[-1].date())}"
    )

    regime_df, _ = await stage0_vol_regime(start=start_date)
    regime_df.index = pd.to_datetime(regime_df.index)

    gex_flow_df = await stage1a_dealer_gamma(start=start_date, dates=dates)
    gex_flow_df.index = pd.to_datetime(gex_flow_df.index)

    mom_df    = stage1b_momentum(returns_df, regime_df=regime_df)
    lowvol_df = stage4_low_vol(returns_df)
    mom_df.index    = pd.to_datetime(mom_df.index)
    lowvol_df.index = pd.to_datetime(lowvol_df.index)

    signal_dfs: Dict[str, pd.DataFrame] = {
        "gex_flow": gex_flow_df,
        "mom":      mom_df,
        "low_vol":  lowvol_df,
    }

    # Save per-signal matrix for analysis
    long_frames = []
    for sig_name, df in signal_dfs.items():
        df_aligned = (
            df.reindex(dates).ffill().fillna(0.0)
            .reindex(columns=TICKERS).fillna(0.0)
        )
        df_long = df_aligned.copy()
        df_long.columns = [f"{sig_name}_{t}" for t in df_long.columns]
        long_frames.append(df_long)

    signals_long_df = pd.concat(long_frames, axis=1)
    signals_long_df.to_parquet(_SIGNALS_OUT)
    logger.info(f"Per-signal matrix -> {_SIGNALS_OUT} ({signals_long_df.shape})")

    alpha_df, gex_alpha_df = stage5_blend_signals(
        signal_dfs=signal_dfs,
        regime_df=regime_df,
        returns_df=returns_df,
        dates=dates,
    )

    n_nans = alpha_df.isnull().sum().sum()
    if n_nans > 0:
        logger.warning(f"  {n_nans} NaN values in blended alpha. Filling 0.")
        alpha_df = alpha_df.fillna(0.0)

    assert alpha_df.shape == (len(dates), N_ASSETS), \
        f"Shape mismatch: {alpha_df.shape} vs expected ({len(dates)}, {N_ASSETS})"
    assert (alpha_df.abs() <= 1.0 + 1e-5).all().all(), \
        "Alpha values outside [-1, 1] — check tanh application"

    alpha_df.to_parquet(_ALPHA_OUT)
    logger.info(f"Blended alpha -> {_ALPHA_OUT} ({alpha_df.shape})")

    gex_alpha_df.to_parquet(_GEX_ALPHA_OUT)
    logger.info(f"Raw GEX alpha  -> {_GEX_ALPHA_OUT} ({gex_alpha_df.shape})")

    logger.info(
        "\n====== Stage 2 COMPLETE ======\n"
        "v14 changes vs v13 broken run:\n"
        "  GEX RESTORED to equity alpha at weight 0.60 (sector-attenuated)\n"
        "  Separate gex_alpha.parquet exported for dynamic equity envelope\n"
        "  Equity set unified with backtest (14 assets)\n"
        "\nThen run:\n"
        "  PYTHONPATH=. python scripts/run_standalone_backtest.py\n"
    )


if __name__ == "__main__":
    asyncio.run(main())