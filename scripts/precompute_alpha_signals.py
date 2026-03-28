"""
FORTRESS v5 — scripts/precompute_alpha_signals.py  [v16 — CONC LEADERSHIP + GEX MODIFIER]

CHANGES FROM v15:
  v15 correctly decoupled GEX from the alpha vector but exposed the root problem:
  MOM alone cannot discriminate in concentrated regimes. The 12-1M cross-sectional
  rank flattens when breadth collapses because all assets are chopping while QQQ
  drifts upward — not enough separation for rank-based signals.

  v16 adds two mechanisms:

  TIER 1 — GEX × ALPHA MULTIPLICATIVE INTERACTION:
    Instead of GEX as an additive signal (v14, broken) or fully decoupled (v15, weak),
    GEX is applied as a MULTIPLICATIVE modifier AFTER the cross-sectional blend:
      alpha_equity *= (1.0 + gamma * gex_sector_attenuated)

    When GEX is bullish and MOM identifies a leader → amplified (1.3× at GEX=+0.5)
    When GEX is bullish but MOM is zero → amplifying zero = zero (no flat gradient)
    When GEX is bearish → dampened (0.7× at GEX=-0.5)

    This preserves the cross-sectional gradient (only assets WITH existing alpha
    get boosted) while giving GEX a legitimate role in asset SELECTION. The sector
    attenuation ensures QQQ (beta=0.95) gets amplified more than XLU (beta=0.30).

  TIER 2 — CONCENTRATION LEADERSHIP SIGNAL:
    Detects when the market is being driven by narrow leadership and tilts toward
    the leaders. Orthogonal to MOM because:
      - Different lookback (63d vs 252-21d)
      - Gated by market structure (only fires when concentration is elevated)
      - Captures MEDIUM-TERM leadership, not LONG-TERM trend

    Implementation:
      1. concentration_spread = QQQ 63d return - equal-weight equity 63d return
      2. concentration_zscore = z-score of spread (252d lookback)
      3. For each equity asset: 63d return rank (shorter-term leadership score)
      4. conc_signal = concentration_zscore × rank → positive when concentrated
         AND asset is a leader; negative when concentrated AND asset is lagging.
      5. When concentration_zscore ≈ 0 (normal breadth): signal is near zero.

    In F8/F9 (Mag7): spread highly positive → QQQ/XLK get strong positive signal.
    In F1-F5 (dispersed): spread ≈ 0 → signal is near zero → no interference.

SIGNAL STACK (v16):
  ALPHA VECTOR (3 signals, breadth-adaptive):
    EQUITY BUCKET:
      High breadth:  MOM=0.45  LowVol=0.30  CONC=0.25
      Low breadth:   MOM=0.25  LowVol=0.10  CONC=0.65
    NON-EQUITY BUCKET:
      MOM=0.65  LowVol=0.35  CONC=0.00

  POST-BLEND MODIFIER (equity only):
    alpha *= (1.0 + 0.50 * gex_sector_attenuated)

  ENVELOPE SIGNAL (separate, unchanged from v15):
    gex_alpha.parquet → backtester uses for equity cap + λ_var modulation

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
_GEX_ALPHA_OUT  = _CACHE_DIR / "gex_alpha.parquet"
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

# v16: 3 signals in the blend stack
SIGNAL_NAMES: List[str] = ["mom", "low_vol", "conc_lead"]
N_SIGNALS = len(SIGNAL_NAMES)

GEX_BETA: Dict[str, float] = {
    "SPY":  1.00, "QQQ":  0.95, "IWM":  0.40,
    "XLK":  0.85, "XLC":  0.75, "XLY":  0.65,
    "XLF":  0.55, "XLV":  0.50, "XLI":  0.50,
    "XLU":  0.30, "XLP":  0.30, "XLB":  0.40,
    "XLE":  0.35, "COWZ": 0.45, "GDX":  0.15,
}

# ── Breadth-adaptive blending parameters (v16: 3-signal) ──────────────────────
_BREADTH_LOW:  float = 0.70
_BREADTH_HIGH: float = 1.20

# [mom, low_vol, conc_lead] weights at each extreme
_W_HIGH_BREADTH: List[float] = [0.45, 0.30, 0.25]   # dispersed: standard signals
_W_LOW_BREADTH:  List[float] = [0.25, 0.10, 0.65]   # concentrated: CONC dominates
_W_NON_EQUITY:   List[float] = [0.65, 0.35, 0.00]   # non-equity: no conc signal

# GEX × alpha multiplicative interaction strength (Tier 1)
_GEX_INTERACTION_GAMMA: float = 0.50

# Concentration signal parameters (Tier 2)
_CONC_LOOKBACK:     int   = 63    # medium-term leadership window
_CONC_ZSCORE_WIN:   int   = 252   # z-score normalization window
_CONC_SIGNAL_SCALE: float = 1.5   # pre-tanh scaling


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
    start: str, dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """GEX/DIX flow — sector-attenuated. Exported for envelope + modifier."""
    logger.info("Stage 1a: Institutional GEX/DIX flow (ENVELOPE + MODIFIER)...")

    if _PC_FLOW_CACHE.exists():
        try:
            cached_df = pd.read_parquet(_PC_FLOW_CACHE)
            cached_df.index = pd.to_datetime(cached_df.index)
            if len(cached_df) > 100:
                df = (
                    cached_df.reindex(dates).ffill().fillna(0.0)
                    .reindex(columns=TICKERS).fillna(0.0)
                )
                for ticker, beta in GEX_BETA.items():
                    if ticker in df.columns:
                        df[ticker] *= beta
                for ticker in TICKERS:
                    if ticker not in _EQUITY_SET and ticker in df.columns:
                        df[ticker] = 0.0
                logger.info(
                    f"  GEX flow: loaded, sector-attenuated | "
                    f"mean|a| equity={df[list(_EQUITY_SET & set(TICKERS))].abs().mean().mean():.4f}"
                )
                return df
        except Exception as e:
            logger.warning(f"  GEX flow cache load failed ({e}), regenerating...")

    logger.error("  GEX flow cache missing. Falling back to zero signal.")
    return pd.DataFrame(0.0, index=dates, columns=TICKERS)


def stage1b_momentum(
    returns_df: pd.DataFrame,
    regime_df:  Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Cross-sectional 12-1M momentum with breadth suppression + crash gate."""
    logger.info("Stage 1b: Cross-sectional momentum (12-1M, breadth-gated)...")

    active_cols      = [t for t in TICKERS if t not in _HEDGE_ASSETS]
    equity_gate_cols = list(_EQUITY_TICKERS_FOR_MOM_GATE)

    r = returns_df.reindex(columns=TICKERS).fillna(0.0)

    cum_12m = r[active_cols].rolling(252, min_periods=126).sum()
    cum_1m  = r[active_cols].rolling(21, min_periods=10).sum()
    raw_mom = cum_12m - cum_1m

    cs_rank = raw_mom.rank(axis=1, pct=True) - 0.5
    result  = pd.DataFrame(0.0, index=returns_df.index, columns=TICKERS)
    result[active_cols] = np.tanh(cs_rank * 2.0)

    cs_std = r[active_cols].std(axis=1).clip(lower=1e-6)
    rolling_mean_std = cs_std.rolling(252, min_periods=63).mean().clip(lower=1e-6)
    breadth_score = (cs_std / rolling_mean_std).clip(0.2, 2.0)
    result[active_cols] = result[active_cols].multiply(breadth_score, axis=0)

    n_combined = 0
    if regime_df is not None and "ltc_urgency" in regime_df.columns:
        urgency = regime_df["ltc_urgency"].reindex(result.index).ffill().fillna(0.0)
        urgency_gate = urgency > _MOM_CRASH_URGENCY_GATE

        eq_cols  = [c for c in equity_gate_cols if c in result.columns]
        mom_std  = result[eq_cols].std(axis=1)
        mom_mean = result[eq_cols].mean(axis=1)
        roll_90  = mom_std.rolling(_MOM_DISPERSION_LOOKBACK, min_periods=63).quantile(
            _MOM_DISPERSION_QUANTILE
        )
        dispersion_gate = (mom_std > roll_90) & (mom_mean > _MOM_DISPERSION_MEAN_THRESH)
        combined_gate   = urgency_gate | dispersion_gate
        n_combined      = int(combined_gate.sum())

        for col in eq_cols:
            result.loc[combined_gate & (result[col] > 0.0), col] = 0.0

        logger.info(
            f"  Momentum crash gate: urgency={int(urgency_gate.sum())}d | "
            f"dispersion={int(dispersion_gate.sum())}d | "
            f"combined={n_combined}d ({n_combined/len(result)*100:.1f}%)"
        )

    logger.info(
        f"  Momentum: {len(result)} days | "
        f"Mean |signal| (active): {result[active_cols].abs().mean().mean():.3f}"
    )
    return result


def stage4_low_vol(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Volatility rank proxy over 63d realized vol."""
    logger.info("Stage 4: Vol-rank proxy...")
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

    logger.info(
        f"  Vol-rank: {len(result)} days | "
        f"Mean |signal|: {result[active_cols].abs().mean().mean():.3f} | "
        f"QQQ mean: {result.get('QQQ', pd.Series(0.0)).mean():+.3f}"
    )
    return result


def stage_conc_leadership(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Concentration leadership signal — Tier 2.

    Detects when the market is driven by narrow mega-cap leadership and
    tilts toward the assets capturing that concentration.

    Orthogonal to 12-1M momentum because:
      - 63d lookback captures MEDIUM-TERM leadership (vs MOM's 252-21d)
      - GATED by concentration strength: only fires when QQQ >> equal-weight
      - When breadth is normal (spread ≈ 0), signal is near zero
      - Captures the RECENT month MOM deliberately skips (the -1M gap)

    Implementation:
      1. concentration_spread = QQQ 63d return - EW equity 63d return
      2. concentration_zscore = z-score(spread, 252d lookback)
      3. Per-asset leadership = 63d return rank (medium-term winners vs losers)
      4. signal = zscore × rank → fires ONLY when concentrated AND directional

    F8/F9 behavior: spread_z >> 0 (QQQ >> EW), QQQ rank ≈ 1.0 → strong positive.
    F1-F5 behavior: spread_z ≈ 0 → signal near zero → doesn't interfere.
    """
    logger.info("Stage CONC: Concentration leadership signal...")

    equity_cols = sorted(_EQUITY_SET & set(TICKERS) & set(returns_df.columns))
    active_cols = [t for t in TICKERS if t not in _HEDGE_ASSETS]

    r = returns_df.reindex(columns=TICKERS).fillna(0.0)

    # Equal-weight equity basket return (63d cumulative)
    ew_ret_63 = r[equity_cols].mean(axis=1).rolling(
        _CONC_LOOKBACK, min_periods=21
    ).sum()

    # QQQ excess return over EW basket
    qqq_ret_63 = r["QQQ"].rolling(_CONC_LOOKBACK, min_periods=21).sum()
    conc_spread = qqq_ret_63 - ew_ret_63

    # Z-score the concentration spread (252d lookback for stability)
    spread_mean = conc_spread.rolling(_CONC_ZSCORE_WIN, min_periods=63).mean()
    spread_std  = conc_spread.rolling(_CONC_ZSCORE_WIN, min_periods=63).std().clip(lower=1e-4)
    conc_zscore = ((conc_spread - spread_mean) / spread_std).clip(-3.0, 3.0).fillna(0.0)

    # Per-asset 63d return rank → medium-term leadership score
    ret_63_active = r[active_cols].rolling(_CONC_LOOKBACK, min_periods=21).sum()
    leadership_rank = ret_63_active.rank(axis=1, pct=True) - 0.5  # centered [-0.5, +0.5]

    # Concentration signal = zscore × rank
    # When concentrated (zscore > 0): leaders get positive, laggards get negative
    # When dispersed (zscore ≈ 0): signal is near zero for everyone
    # When anti-concentrated (zscore < 0): mean-reversion tilt
    result = pd.DataFrame(0.0, index=returns_df.index, columns=TICKERS)
    conc_product = leadership_rank.multiply(conc_zscore, axis=0)
    result[active_cols] = np.tanh(conc_product * _CONC_SIGNAL_SCALE)
    result = result.fillna(0.0)

    # Zero out non-equity — concentration is an equity phenomenon
    for t in TICKERS:
        if t not in _EQUITY_SET:
            result[t] = 0.0

    pct_active = float((conc_zscore.abs() > 0.5).mean() * 100)
    qqq_mean   = float(result["QQQ"].mean()) if "QQQ" in result.columns else 0.0
    iwm_mean   = float(result["IWM"].mean()) if "IWM" in result.columns else 0.0
    logger.info(
        f"  Concentration: {len(result)} days | "
        f"zscore active (|z|>0.5): {pct_active:.1f}% | "
        f"QQQ mean: {qqq_mean:+.3f} | IWM mean: {iwm_mean:+.3f}"
    )
    return result


def stage5_blend_signals(
    signal_dfs:  Dict[str, pd.DataFrame],
    regime_df:   pd.DataFrame,
    returns_df:  pd.DataFrame,
    dates:       pd.DatetimeIndex,
    gex_flow_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    v16: Blend 3 signals with breadth-adaptive weights, then apply GEX modifier.

    Two-step process:
      Step 1: Weighted sum of [mom, low_vol, conc_lead] with breadth-adaptive weights
      Step 2: Multiply equity alpha by (1 + gamma * GEX) — the Tier 1 interaction
    """
    logger.info("Stage 5: Blending 3 signals + GEX modifier (v16)...")
    N = N_ASSETS
    S = N_SIGNALS  # 3

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

    # ── Breadth ratio ──────────────────────────────────────────────────────
    equity_cols = sorted(_EQUITY_SET & set(TICKERS) & set(returns_df.columns))
    equity_rets = returns_df[equity_cols].reindex(dates).fillna(0.0)

    cs_std_series  = equity_rets.std(axis=1).clip(lower=1e-6)
    rolling_cs_63  = cs_std_series.rolling(63, min_periods=21).mean()
    rolling_cs_252 = cs_std_series.rolling(252, min_periods=63).mean().clip(lower=1e-6)
    breadth_ratio  = (rolling_cs_63 / rolling_cs_252).clip(0.3, 2.0).fillna(1.0)

    t_breadth = ((breadth_ratio - _BREADTH_LOW) / (_BREADTH_HIGH - _BREADTH_LOW)).clip(0.0, 1.0)

    # Interpolate 3-signal weights: low_breadth → high_breadth
    w_low  = np.array(_W_LOW_BREADTH, dtype=np.float32)   # [mom, lov, conc]
    w_high = np.array(_W_HIGH_BREADTH, dtype=np.float32)
    w_neq  = np.array(_W_NON_EQUITY, dtype=np.float32)

    t_arr = t_breadth.values.astype(np.float32)  # (T,)

    logger.info(
        f"  Breadth ratio: mean={breadth_ratio.mean():.3f} | "
        f"<0.70: {(breadth_ratio < _BREADTH_LOW).mean()*100:.1f}% | "
        f">1.20: {(breadth_ratio > _BREADTH_HIGH).mean()*100:.1f}%"
    )

    # ── Build time-varying blending weights (T, N, S) ──────────────────────
    blending_weights = np.zeros((T, N, S), dtype=np.float32)

    for i, ticker in enumerate(TICKERS):
        if ticker in _EQUITY_SET:
            for s_idx in range(S):
                blending_weights[:, i, s_idx] = (
                    w_low[s_idx] + t_arr * (w_high[s_idx] - w_low[s_idx])
                )
        else:
            for s_idx in range(S):
                blending_weights[:, i, s_idx] = w_neq[s_idx]

    alpha_raw = (blending_weights * signal_stack).sum(axis=-1)  # (T, N)

    # ── Regime-conditional attenuation (unchanged) ─────────────────────────
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

    # ── TIER 1: GEX × alpha multiplicative interaction (equity only) ───────
    gex_aligned = (
        gex_flow_df.reindex(dates).ffill().fillna(0.0)
        .reindex(columns=TICKERS).fillna(0.0)
    )
    gex_arr = gex_aligned.values.astype(np.float32)  # already tanh-bounded [-1,+1]

    for i, ticker in enumerate(TICKERS):
        if ticker in _EQUITY_SET:
            # Multiplicative: amplify existing alpha, don't create new gradient
            # GEX=+0.5: multiplier = 1.25 → 25% boost to MOM+CONC winners
            # GEX=-0.5: multiplier = 0.75 → 25% dampening
            # GEX=0:    multiplier = 1.00 → no effect
            alpha_raw[:, i] *= (1.0 + _GEX_INTERACTION_GAMMA * gex_arr[:, i])

    alpha_final = np.tanh(alpha_raw).astype(np.float32)
    result_df   = pd.DataFrame(alpha_final, index=dates, columns=TICKERS)

    # ── GEX envelope export (unchanged from v15) ──────────────────────────
    gex_raw_df = gex_aligned.copy()

    eq_cols_set = list(_EQUITY_SET & set(TICKERS))
    eq_mean  = result_df[eq_cols_set].abs().mean().mean()
    neq_mean = result_df[[t for t in TICKERS if t not in _EQUITY_SET]].abs().mean().mean()
    logger.info(
        f"  Blended alpha (3-sig + GEX modifier): {len(result_df)}d x {N} assets | "
        f"|alpha| equity={eq_mean:.3f} | non-equity={neq_mean:.3f}"
    )

    return result_df, gex_raw_df


async def main() -> None:
    logger.info(
        "====== Ouroboros Alpha Precompute v16 "
        "(CONC LEADERSHIP + GEX MODIFIER) ======"
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

    mom_df = stage1b_momentum(returns_df, regime_df=regime_df)
    mom_df.index = pd.to_datetime(mom_df.index)

    lowvol_df = stage4_low_vol(returns_df)
    lowvol_df.index = pd.to_datetime(lowvol_df.index)

    conc_df = stage_conc_leadership(returns_df)
    conc_df.index = pd.to_datetime(conc_df.index)

    signal_dfs: Dict[str, pd.DataFrame] = {
        "mom":       mom_df,
        "low_vol":   lowvol_df,
        "conc_lead": conc_df,
    }

    all_signals: Dict[str, pd.DataFrame] = {
        "gex_flow": gex_flow_df,
        **signal_dfs,
    }
    long_frames = []
    for sig_name, df in all_signals.items():
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
        gex_flow_df=gex_flow_df,
    )

    n_nans = alpha_df.isnull().sum().sum()
    if n_nans > 0:
        logger.warning(f"  {n_nans} NaN values in blended alpha. Filling 0.")
        alpha_df = alpha_df.fillna(0.0)

    assert alpha_df.shape == (len(dates), N_ASSETS), \
        f"Shape mismatch: {alpha_df.shape} vs expected ({len(dates)}, {N_ASSETS})"
    assert (alpha_df.abs() <= 1.0 + 1e-5).all().all(), \
        "Alpha values outside [-1, 1]"

    alpha_df.to_parquet(_ALPHA_OUT)
    logger.info(f"Blended alpha -> {_ALPHA_OUT} ({alpha_df.shape})")

    gex_alpha_df.to_parquet(_GEX_ALPHA_OUT)
    logger.info(f"Raw GEX alpha  -> {_GEX_ALPHA_OUT} ({gex_alpha_df.shape})")

    logger.info("Per-signal statistics:")
    for sig_name, df in all_signals.items():
        df_a    = df.reindex(dates).ffill().fillna(0.0).reindex(columns=TICKERS).fillna(0.0)
        nonzero = (df_a.abs() > 0.01).any(axis=1).mean() * 100
        role    = "IN ALPHA" if sig_name in signal_dfs else "MODIFIER+ENV"
        logger.info(
            f"  {sig_name:10s} [{role:12s}]: mean|α|={df_a.abs().mean().mean():.3f} | "
            f"active={nonzero:.1f}% | "
            f"SPY={df_a['SPY'].mean():+.3f} QQQ={df_a['QQQ'].mean():+.3f}"
        )

    logger.info(
        "\n====== COMPLETE ======\n"
        "v16: 3 signals (MOM + VolRank + ConcLeadership) + GEX multiplicative modifier\n"
        "\nNext:\n"
        "  PYTHONPATH=. python scripts/run_standalone_backtest.py\n"
    )


if __name__ == "__main__":
    asyncio.run(main())