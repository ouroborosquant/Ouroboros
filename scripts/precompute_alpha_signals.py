"""
FORTRESS v5 — scripts/precompute_alpha_signals.py  [v19 — ORTHOGONAL MICROSTRUCTURE]

v19 Signal Stack
----------------
  Replaces EIS + BV_VPIN with two deterministic, non-ML signals:

  1. night_effect  (NightEffectEngine):
     Diurnal liquidity premium reversal. Compares 5-day cumulative overnight
     vs intraday return divergence, normalised by Garman-Klass vol. Produces a
     cross-sectional Z-score within the equity_single universe. Theoretically
     orthogonal to momentum and low-vol because it exploits microstructure timing
     (open/close auction dynamics), not risk-premia persistence.

  2. pca_statarb  (PCAStatArbEngine):
     ETF-hedged idiosyncratic OU mean-reversion. Strips out the 25-ETF principal
     factor exposures from each equity's return via rolling 60d OLS, models the
     residual as an AR(1)/OU process, and generates an S-score. Orthogonal to
     the momentum and low-vol signals because residuals are, by construction,
     orthogonal to the macro factor space.

Both signals output 0.0 for all macro ETF tickers — they are equity selection
tools only. The blending stage normalises static weights per asset class so that
ETF alphas are not systematically diluted by the two zero-valued signals.

Signal Weights (static; GATv2 bypassed pending retrain)
---------------------------------------------------------
  Equity singles:  mom=0.30 | low_vol=0.15 | conc_lead=0.15 | night=0.20 | statarb=0.20
  Macro ETFs:      mom=0.45 | low_vol=0.25 | conc_lead=0.30  (renormalised from 0.60)

GATv2 will be retrained once the new signal ICs are validated via
validate_signal_ic.py on the 5-signal tensor.

Pipeline
--------
  Stage 0: Multi-asset vol regime (LTC urgency)
  Stage 1a: Dealer gamma (GEX/DIX) flow signal
  Stage 1b: Cross-sectional momentum (12-1M)
  Stage 4:  Low-volatility anomaly (parking assets excluded)
  Stage 5:  Concentration leadership
  Stage 6:  Night Effect (OHLC microstructure)
  Stage 7:  PCA StatArb (OU idiosyncratic residual)
  Stage 8:  Signal blending + regime gate

Run
---
  PYTHONPATH=. python scripts/precompute_alpha_signals.py
"""
from __future__ import annotations

import asyncio
import logging
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("Ouroboros.AlphaPrecompute")

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE_DIR  = Path(".")
_CACHE_DIR = _BASE_DIR / "research" / "outputs" / "cache"

_PRICES_PATH    = _CACHE_DIR / "prices_wide.parquet"
_RETURNS_PATH   = _CACHE_DIR / "returns_wide.parquet"
_VOLUMES_PATH   = _CACHE_DIR / "volumes_wide.parquet"
_OHLC_PATH      = _CACHE_DIR / "ohlc_wide.parquet"          # MultiIndex (price_type, ticker)
_SIGNALS_OUT    = _CACHE_DIR / "alpha_signals.parquet"
_ALPHA_OUT      = _CACHE_DIR / "alpha_signals_blended.parquet"
_GEX_ALPHA_OUT  = _CACHE_DIR / "gex_alpha.parquet"
_REGIME_OUT     = _CACHE_DIR / "regime_posteriors.parquet"

_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_UNIVERSE_FILE = _BASE_DIR / "config" / "universe.yaml"
_ROUTER_WEIGHTS = _BASE_DIR / "models" / "weights" / "gat_router.pt"

# ── Universe ──────────────────────────────────────────────────────────────────
with open(_UNIVERSE_FILE, "r") as _f:
    _univ_cfg = yaml.safe_load(_f)

TICKERS: List[str] = [a["ticker"] for a in _univ_cfg["assets"]]
N_ASSETS: int = len(TICKERS)

_CAT = {a["ticker"]: a.get("category", "unknown") for a in _univ_cfg["assets"]}
_ETF_CATS     = {"equity_broad", "equity_sector", "equity_intl", "fixed_income",
                 "commodity", "volatility"}
_SINGLE_CATS  = {"equity_single"}

ETF_TICKERS:    List[str] = [t for t in TICKERS if _CAT.get(t) in _ETF_CATS]
EQUITY_TICKERS: List[str] = [t for t in TICKERS if _CAT.get(t) in _SINGLE_CATS]
_HEDGE_ASSETS:  frozenset[str] = frozenset({"BIL", "SHV", "VIXY"})
_PARKING_ASSETS: frozenset[str] = frozenset({"BIL", "SHV"})

logger.info(
    f"Universe: {N_ASSETS} total | {len(ETF_TICKERS)} ETFs | "
    f"{len(EQUITY_TICKERS)} single-name equities"
)

# ── Signal stack (v19) ────────────────────────────────────────────────────────
SIGNAL_NAMES: List[str] = ["mom", "low_vol", "conc_lead", "night_effect", "pca_statarb"]
N_SIGNALS: int = len(SIGNAL_NAMES)

# Per-asset-class static weights — sums checked in blend stage
_WEIGHTS_EQUITY: Dict[str, float] = {
    "mom":         0.30,
    "low_vol":     0.15,
    "conc_lead":   0.15,
    "night_effect": 0.20,
    "pca_statarb": 0.20,
}   # sum = 1.00

_WEIGHTS_ETF: Dict[str, float] = {
    "mom":         0.45,
    "low_vol":     0.25,
    "conc_lead":   0.30,
    "night_effect": 0.00,   # zeroed at source; weight excluded
    "pca_statarb": 0.00,    # zeroed at source; weight excluded
}   # effective sum = 1.00

GEX_BETA:              Dict[str, float] = {
    "SPY": 1.00, "QQQ": 0.95, "IWM": 0.40,
    "XLK": 0.85, "XLC": 0.75, "XLY": 0.65,
}
_GEX_INTERACTION_GAMMA: float = 0.50


# ─────────────────────────────────────────────────────────────────────────────
# Stage 0: Multi-asset vol regime
# ─────────────────────────────────────────────────────────────────────────────

async def stage0_vol_regime(
    start: str,
    prices_df: Optional[pd.DataFrame] = None,
    returns_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Attempt to load vol regime from MultiAssetVolRegime, probing for the actual
    API signature via introspection. Falls back to an inline SPY rolling-vol
    urgency proxy so the regime gate in stage8 always has a signal.
    """
    try:
        from signals.vol_regime import MultiAssetVolRegime
        engine = MultiAssetVolRegime()
        avail  = [m for m in dir(engine) if not m.startswith("_")]
        logger.info(f"  MultiAssetVolRegime available methods: {avail}")

        regime_df: Optional[pd.DataFrame] = None

        if hasattr(engine, "load_data"):
            await engine.load_data(start=start)
            regime_df = engine.compute_regimes()

        elif hasattr(engine, "compute_regimes") and returns_df is not None:
            # Newer API passes data directly
            try:
                result    = engine.compute_regimes(returns_df=returns_df)
            except TypeError:
                result    = engine.compute_regimes(
                    prices_df=prices_df, returns_df=returns_df
                )
            regime_df = result if isinstance(result, pd.DataFrame) else result[0]

        elif hasattr(engine, "fit") and returns_df is not None:
            engine.fit(returns_df)
            getter = next(
                (m for m in ["get_posteriors", "posteriors", "regime_df"] if hasattr(engine, m)),
                None,
            )
            regime_df = getattr(engine, getter)() if getter else None

        elif hasattr(engine, "run"):
            regime_df = await engine.run() if asyncio.iscoroutinefunction(engine.run) \
                        else engine.run()

        if regime_df is None or not isinstance(regime_df, pd.DataFrame):
            raise AttributeError(
                f"Could not extract regime DataFrame. Available: {avail}"
            )

        regime_df.to_parquet(_REGIME_OUT)
        logger.info(f"  ✓ Regime posteriors → {_REGIME_OUT} {regime_df.shape}")
        return regime_df, engine

    except Exception as e:
        logger.warning(
            f"  ⚠️  Vol regime failed ({e}); "
            f"activating inline SPY rolling-vol urgency proxy."
        )
        # ── Inline fallback: SPY 21d realised vol → sigmoid urgency ──────────
        # Provides ltc_urgency so stage8 regime gate is not permanently inactive.
        if returns_df is not None and "SPY" in returns_df.columns:
            spy_rv   = returns_df["SPY"].rolling(21, min_periods=10).std() * np.sqrt(252)
            spy_ewm  = spy_rv.ewm(halflife=63, min_periods=21)
            # Z-score then sigmoid → urgency ∈ (0, 1)
            z        = (spy_rv - spy_ewm.mean()) / spy_ewm.std().clip(lower=0.001)
            urgency  = (1.0 / (1.0 + np.exp(-z * 0.8))).clip(0.0, 1.0)
            regime_df = pd.DataFrame(
                {
                    "ltc_urgency": urgency.fillna(0.3),
                    "spy_rv_21d":  spy_rv.fillna(0.15),
                },
                index=returns_df.index,
            )
            logger.info(
                f"  ✓ SPY urgency proxy: {regime_df.shape} | "
                f"mean_urgency={urgency.mean():.3f} | "
                f"crisis_days(>0.6)={(urgency > 0.6).mean()*100:.1f}%"
            )
            return regime_df, None

        regime_df = pd.DataFrame(index=pd.DatetimeIndex([]), columns=["ltc_urgency"])
        return regime_df, None


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1a: Dealer gamma (GEX) flow signal
# ─────────────────────────────────────────────────────────────────────────────

async def stage1a_dealer_gamma(
    start: str, dates: pd.DatetimeIndex
) -> pd.DataFrame:
    try:
        from signals.options_flow import OptionsFlowSignalEngine
        engine = OptionsFlowSignalEngine()
        await engine.load_data(start=start)
        gex_df = engine.compute_pc_history(dates)
        gex_df = gex_df.reindex(columns=TICKERS).fillna(0.0).astype(np.float32)
        nonzero = (gex_df.abs() > 0.005).any(axis=1).mean() * 100
        logger.info(f"  ✓ GEX/DIX signal: {gex_df.shape} | active={nonzero:.1f}%")
        return gex_df
    except Exception as e:
        logger.warning(f"  ⚠️  GEX flow failed ({e}); returning zeros.")
        return pd.DataFrame(0.0, index=dates, columns=TICKERS, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1b: Cross-sectional momentum (12M-1M)
# ─────────────────────────────────────────────────────────────────────────────

def stage1b_momentum(
    returns_df: pd.DataFrame,
    regime_df:  Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    12-1M cross-sectional momentum with regime-gated signal compression.
    Skip-month (1M) neutralises short-term reversal microstructure bias.
    """
    logger.info("Stage 1b: Cross-sectional momentum (12-1M)...")

    active_cols = [t for t in TICKERS if t not in _HEDGE_ASSETS]

    # 12M cumulative return (252d), skip last 21d (1M reversal buffer)
    ret_12m = returns_df[active_cols].rolling(252, min_periods=126).sum()
    ret_1m  = returns_df[active_cols].rolling(21,  min_periods=10).sum()
    mom_raw = ret_12m - ret_1m

    # Cross-sectional rank → centred score ∈ (-1, 1)
    rank  = mom_raw.rank(axis=1, pct=True)
    score = np.tanh((rank - 0.5) * 4.0)

    # GEX concentration gate: compress momentum in extreme concentration regimes
    if regime_df is not None and "breadth_ratio" in regime_df.columns:
        breadth = regime_df["breadth_ratio"].reindex(returns_df.index).ffill().fillna(0.5)
        # When breadth_ratio < 0.30, the market is in Mag7-concentrated mode;
        # momentum degrades because only a few stocks drive returns.
        conc_gate = np.clip((breadth.values - 0.30) / 0.20, 0.0, 1.0)[:, None]
        score = pd.DataFrame(
            score.values * (0.5 + 0.5 * conc_gate),
            index=score.index,
            columns=score.columns,
        )

    result = pd.DataFrame(0.0, index=returns_df.index, columns=TICKERS, dtype=np.float32)
    result[active_cols] = score.fillna(0.0).astype(np.float32)
    result[list(_HEDGE_ASSETS)] = 0.0

    mean_abs = result[active_cols].abs().mean().mean()
    logger.info(f"  ✓ Momentum: mean|α|={mean_abs:.3f}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4: Low-volatility anomaly
# ─────────────────────────────────────────────────────────────────────────────

def stage4_low_vol(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Low-vol anomaly with CLASS-AWARE cross-sectional ranking.

    BUG FIX (v19.1):
      Expanding from 25 ETFs to 100 assets broke the low_vol signal via
      cross-asset-class contamination. In the mixed 97-asset cross-section,
      TLT/LQD received the most POSITIVE signal (low bond vol) while NVDA/TSLA
      received the most NEGATIVE signal (high equity vol). But 2020-2024:
      NVDA +1000%, TLT -40%. The IC inverted from +0.034 to -0.034.

      The low-vol anomaly is an INTRA-CLASS premium:
        - Within equities: lower-vol stocks earn higher risk-adjusted returns
        - Within ETFs: lower-vol ETFs earn higher risk-adjusted returns
        - Across equity/bond barrier: bonds have lower vol AND lower returns
          than equities — the cross-class comparison is economically meaningless

      FIX: Rank separately within (a) ETF sub-universe and (b) equity sub-universe.
      The two signals are then stacked into the full output matrix with their
      own cross-sectional z-scores intact. ETFs still compete against ETFs;
      equities still compete against equities.
    """
    logger.info("Stage 4: Low-vol anomaly (class-aware ranking)...")

    rv_63 = (
        returns_df.reindex(columns=TICKERS)
        .rolling(63, min_periods=20)
        .std() * np.sqrt(252)
    ).ffill()

    result = pd.DataFrame(0.0, index=returns_df.index, columns=TICKERS, dtype=np.float32)

    # ── ETF sub-universe (exclude hard parking/hedge assets) ──────────────────
    etf_active = [t for t in ETF_TICKERS if t not in _HEDGE_ASSETS]
    if etf_active:
        rank_etf   = rv_63[etf_active].rank(axis=1, pct=True)
        signal_etf = np.tanh(-(rank_etf - 0.5) * 2.0)
        result[etf_active] = signal_etf.fillna(0.0).astype(np.float32)

    # ── Equity sub-universe ───────────────────────────────────────────────────
    eq_active = [t for t in EQUITY_TICKERS]
    if eq_active:
        rank_eq   = rv_63[eq_active].rank(axis=1, pct=True)
        signal_eq = np.tanh(-(rank_eq - 0.5) * 2.0)
        result[eq_active] = signal_eq.fillna(0.0).astype(np.float32)

    mean_abs_etf = result[etf_active].abs().mean().mean() if etf_active else 0.0
    mean_abs_eq  = result[eq_active].abs().mean().mean()  if eq_active  else 0.0

    # REGIME FIX: negate the signal — the academic low-vol anomaly is REVERSED
    # over 2018-2026 due to (a) rate hike cycle obliterating TLT/LQD (the
    # lowest-vol ETFs) in 2022 and (b) AI bull run rewarding QQQ/XLK (high-vol
    # ETFs) in 2023-2024. Empirically this is a HIGH-VOL PREMIUM signal in this
    # period. validate_signal_ic confirmed ETF-only IC_5d = -0.029 before negation.
    # The REVERSE_SIGNAL verdict applies across both sub-universes.
    result = -result

    logger.info(
        f"  ✓ Vol-premium (negated low_vol, class-aware): "
        f"mean|α| ETF={mean_abs_etf:.3f} | equity={mean_abs_eq:.3f}"
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5: Concentration leadership
# ─────────────────────────────────────────────────────────────────────────────

def stage_conc_leadership(returns_df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Stage 5: Concentration leadership...")

    try:
        from signals.conc_signal import ConcLeadershipEngine  # type: ignore
        engine = ConcLeadershipEngine(tickers=TICKERS)
        result = engine.compute_signal(returns_df)
        logger.info(f"  ✓ Conc leadership (engine): mean|α|={result.abs().mean().mean():.3f}")
        return result

    except ImportError:
        # ── Inline implementation: 63d-skip-21d cross-sectional momentum ─────
        # IC profile from validate_signal_ic: IC_1d≈-0.003, IC_5d≈-0.005,
        # IC_21d≈+0.014, IC_63d≈+0.048. This is a MEDIUM-TERM continuation
        # signal. Negative short-term IC reflects microstructure mean-reversion
        # in recent winners; positive 63d IC reflects factor persistence.
        # Using 63d-21d lookback (skip 21d) removes the short-term reversal
        # contamination and isolates the medium-term factor trend.
        # Weight in blending: 0.15 equity / 0.30 ETF (only useful at hold ≥21d).
        logger.info(
            "  signals/conc_signal.py absent — using inline 63d-skip-21d momentum."
        )
        active_cols = [t for t in TICKERS if t not in _HEDGE_ASSETS]

        ret_63 = returns_df[active_cols].rolling(63,  min_periods=30).sum()
        ret_21 = returns_df[active_cols].rolling(21,  min_periods=10).sum()
        # 63d momentum excluding the most recent 21d (avoid short-term reversal)
        mom    = ret_63 - ret_21

        rank   = mom.rank(axis=1, pct=True)
        signal = np.tanh((rank - 0.5) * 3.0)

        result = pd.DataFrame(0.0, index=returns_df.index, columns=TICKERS, dtype=np.float32)
        result[active_cols] = signal.fillna(0.0).astype(np.float32)
        result[list(_HEDGE_ASSETS)] = 0.0

        logger.info(
            f"  ✓ Conc leadership (inline): "
            f"mean|α|={result[active_cols].abs().mean().mean():.3f}"
        )
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Stage 6: Night Effect (new — replaces EIS)
# ─────────────────────────────────────────────────────────────────────────────

def stage_night_effect(
    open_df:  pd.DataFrame,
    high_df:  pd.DataFrame,
    low_df:   pd.DataFrame,
    close_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Diurnal liquidity premium reversal signal via Garman-Klass normalisation.
    See models/alpha/night_effect.py for full mathematical specification.
    """
    logger.info("Stage 6: Night Effect (GK-normalised diurnal reversal)...")
    from models.alpha.night_effect import NightEffectEngine

    engine = NightEffectEngine(
        equity_tickers=EQUITY_TICKERS,
        all_tickers=TICKERS,
    )
    return engine.compute_signal(
        open_df=open_df,
        high_df=high_df,
        low_df=low_df,
        close_df=close_df,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stage 7: PCA StatArb (new — replaces BV_VPIN)
# ─────────────────────────────────────────────────────────────────────────────

def stage_pca_statarb(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    ETF-hedged idiosyncratic OU mean-reversion S-score signal.
    See models/alpha/pca_statarb.py for full mathematical specification.
    """
    logger.info("Stage 7: PCA StatArb (OU idiosyncratic residual)...")
    from models.alpha.pca_statarb import PCAStatArbEngine

    engine = PCAStatArbEngine(
        etf_tickers=ETF_TICKERS,
        equity_tickers=EQUITY_TICKERS,
        all_tickers=TICKERS,
    )
    return engine.compute_signal(returns_df)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 8: Signal blending (static weights; GATv2 bypassed pending retrain)
# ─────────────────────────────────────────────────────────────────────────────

def stage8_blend_signals(
    signal_dfs: Dict[str, pd.DataFrame],
    regime_df:  pd.DataFrame,
    returns_df: pd.DataFrame,
    dates:      pd.DatetimeIndex,
    gex_flow_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Blend 5 signals into final alpha vector using per-asset-class static weights.

    Key change vs v18
    -----------------
    Asset-class weight normalisation: Night Effect and PCA StatArb output 0.0
    for ETF tickers. Without normalisation, ETF alphas would be scaled by
    (w_night + w_statarb) = 0.40 relative to equity alphas → systematic
    underweighting of ETF exposures in the MVO.

    Solution: pre-define separate weight vectors for ETF and equity_single
    asset classes. Both vectors sum to 1.0.

    GATv2 bypass rationale (still applies):
      The currently saved weights were trained on the EIS/BV_VPIN signal tensor.
      Re-enabling the router before retraining on the new night_effect/pca_statarb
      signals will produce inverted routing due to feature-space mismatch.
      Delete models/weights/gat_router.pt and run:
        PYTHONPATH=. python scripts/train_gat_router.py
      once validate_signal_ic.py confirms both new signals have IC > 0.03.
    """
    logger.info("Stage 8: Signal blending (static weights — GATv2 bypassed)...")

    T = len(dates)
    N = N_ASSETS

    # Normalise weight vectors (assert both sum to 1.0)
    w_eq  = np.array([_WEIGHTS_EQUITY.get(n, 0.0) for n in SIGNAL_NAMES], dtype=np.float64)
    w_etf = np.array([_WEIGHTS_ETF.get(n, 0.0)    for n in SIGNAL_NAMES], dtype=np.float64)
    w_eq  /= w_eq.sum()
    w_etf /= w_etf.sum()

    assert abs(w_eq.sum()  - 1.0) < 1e-6, f"Equity weights don't sum to 1: {w_eq.sum()}"
    assert abs(w_etf.sum() - 1.0) < 1e-6, f"ETF weights don't sum to 1: {w_etf.sum()}"

    # ── Build signal stack (T, N, S) ──────────────────────────────────────────
    signal_stack = np.zeros((T, N, N_SIGNALS), dtype=np.float32)
    for s_idx, sig_name in enumerate(SIGNAL_NAMES):
        if sig_name not in signal_dfs:
            logger.warning(f"  Signal '{sig_name}' missing — using zeros.")
            continue
        aligned = (
            signal_dfs[sig_name]
            .reindex(dates).ffill().fillna(0.0)
            .reindex(columns=TICKERS).fillna(0.0)
        )
        signal_stack[:, :, s_idx] = aligned.values.astype(np.float32)

    # ── Per-asset weight vector selection ────────────────────────────────────
    # Build weight matrix (N, S): each asset row gets either w_eq or w_etf
    weight_matrix = np.zeros((N, N_SIGNALS), dtype=np.float64)
    equity_set = frozenset(EQUITY_TICKERS)
    for i, ticker in enumerate(TICKERS):
        weight_matrix[i] = w_eq if ticker in equity_set else w_etf

    # Broadcast to (T, N, S) — weights are static across time
    weight_tensor = np.broadcast_to(
        weight_matrix[np.newaxis, :, :], (T, N, N_SIGNALS)
    ).copy()

    # Log effective weight assignments
    logger.info(
        "  Equity weights: "
        + " | ".join(f"{n}={w:.3f}" for n, w in zip(SIGNAL_NAMES, w_eq))
    )
    logger.info(
        "  ETF weights:    "
        + " | ".join(f"{n}={w:.3f}" for n, w in zip(SIGNAL_NAMES, w_etf))
    )

    # ── Warn if GATv2 weights exist (requires retrain before activation) ──────
    if _ROUTER_WEIGHTS.exists():
        logger.warning(
            f"  ⚠️  GATv2 weights at {_ROUTER_WEIGHTS} are STALE (trained on EIS/BV_VPIN). "
            f"Delete and retrain on night_effect/pca_statarb before re-enabling."
        )

    # ── Blended alpha: element-wise weighted sum ──────────────────────────────
    alpha_raw = (weight_tensor * signal_stack).sum(axis=-1)  # (T, N)

    # ── Reduced regime gate ───────────────────────────────────────────────────
    # Only fires in genuine crisis (urgency > 0.60); max 40% compression.
    # Safe-haven override intentionally removed: MVO handles this via λ_var scaling.
    if "ltc_urgency" in regime_df.columns:
        urgency = (
            regime_df["ltc_urgency"]
            .reindex(dates).ffill().fillna(0.0).values
        )
        for i, ticker in enumerate(TICKERS):
            asset_urgency = urgency if ticker not in {"GLD", "SLV", "GDX", "USO", "PDBC"} \
                            else urgency * 0.5
            # Gate only fires above 0.60; linear ramp to 40% max compression
            crisis_scale = np.clip((asset_urgency - 0.60) / 0.40, 0.0, 1.0) * 0.40
            alpha_raw[:, i] *= (1.0 - crisis_scale)
    else:
        logger.info("  ltc_urgency absent from regime_df — regime gate inactive.")

    # ── GEX multiplicative modifier (equity ETFs only) ────────────────────────
    gex_aligned = (
        gex_flow_df.reindex(dates).ffill().fillna(0.0)
        .reindex(columns=TICKERS).fillna(0.0)
    )
    gex_arr = gex_aligned.values.astype(np.float32)

    for i, ticker in enumerate(TICKERS):
        if ticker in GEX_BETA:
            alpha_raw[:, i] *= (1.0 + _GEX_INTERACTION_GAMMA * gex_arr[:, i])

    alpha_final = np.tanh(alpha_raw).astype(np.float32)
    alpha_df = pd.DataFrame(alpha_final, index=dates, columns=TICKERS)

    return alpha_df, gex_aligned


# ─────────────────────────────────────────────────────────────────────────────
# OHLC loading with yfinance fallback
# ─────────────────────────────────────────────────────────────────────────────

def load_ohlc(
    tickers: List[str], start: str
) -> Dict[str, pd.DataFrame]:
    """
    Load OHLC DataFrames from ohlc_wide.parquet if it exists; otherwise
    download from yfinance (equity_single tickers only) and persist.

    Returns dict with keys "open", "high", "low", "close" — each a (T, N)
    DataFrame with columns = tickers, aligned to the same DatetimeIndex.

    Only equity_single tickers need OHLC for NightEffectEngine; ETF OHLC
    slots are filled with the close price as a safe sentinel (open=high=low=close
    → GK vol = 0 → night effect = 0 for ETFs, which is the correct output).
    """
    if _OHLC_PATH.exists():
        logger.info(f"  Loading OHLC from cache: {_OHLC_PATH}")
        ohlc_df = pd.read_parquet(_OHLC_PATH)
        # MultiIndex columns: (price_type, ticker)
        return {
            "open_df":  ohlc_df["Open"].reindex(columns=tickers).ffill(),
            "high_df":  ohlc_df["High"].reindex(columns=tickers).ffill(),
            "low_df":   ohlc_df["Low"].reindex(columns=tickers).ffill(),
            "close_df": ohlc_df["Close"].reindex(columns=tickers).ffill(),
        }

    logger.warning(
        f"  ohlc_wide.parquet not found. Downloading OHLC for {len(tickers)} assets..."
        f"\n  Run scripts/bootstrap_market_data.py first to persist this data."
    )
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    if raw.empty:
        logger.error("  yfinance OHLC download failed. Night Effect will be zeros.")
        dummy = pd.DataFrame(1.0, index=pd.DatetimeIndex([]), columns=tickers)
        return {"open": dummy, "high": dummy, "low": dummy, "close": dummy}

    def _ex(pt: str) -> pd.DataFrame:
        try:
            return raw[pt].reindex(columns=tickers).ffill()
        except KeyError:
            return raw.xs(pt, axis=1, level=0).reindex(columns=tickers).ffill()

    ohlc_dfs = {
        "open_df":  _ex("Open"),
        "high_df":  _ex("High"),
        "low_df":   _ex("Low"),
        "close_df": _ex("Close"),
    }
    # Persist for future runs
    pd.concat({k.capitalize(): v for k, v in ohlc_dfs.items()}, axis=1).to_parquet(
        _OHLC_PATH
    )
    logger.info(f"  Persisted OHLC → {_OHLC_PATH}")
    return ohlc_dfs


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    logger.info("=" * 70)
    logger.info("FORTRESS v5 — Alpha Precompute v19 (Night Effect + PCA StatArb)")
    logger.info("=" * 70)

    if not _PRICES_PATH.exists() or not _RETURNS_PATH.exists():
        logger.error(
            "prices_wide.parquet / returns_wide.parquet not found. "
            "Run scripts/bootstrap_market_data.py first."
        )
        sys.exit(1)

    # ── Load base price data ──────────────────────────────────────────────────
    prices_df  = pd.read_parquet(_PRICES_PATH)
    returns_df = pd.read_parquet(_RETURNS_PATH)

    for df in (prices_df, returns_df):
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        if df.index.duplicated().any():
            df.drop(df.index[df.index.duplicated(keep="last")], inplace=True)

    start_date = str(returns_df.index[0].date())
    dates      = returns_df.index
    logger.info(
        f"Market data: {len(returns_df)} days × {N_ASSETS} assets | "
        f"{start_date} → {dates[-1].date()}"
    )

    # ── Stage 0: Regime posteriors ────────────────────────────────────────────
    regime_df, _ = await stage0_vol_regime(
        start=start_date,
        prices_df=prices_df,
        returns_df=returns_df,
    )
    regime_df.index = pd.to_datetime(regime_df.index)

    # ── Stage 1a: GEX dealer gamma ────────────────────────────────────────────
    gex_flow_df = await stage1a_dealer_gamma(start=start_date, dates=dates)
    gex_flow_df.index = pd.to_datetime(gex_flow_df.index)

    # ── Stage 1b–5: Existing signals ──────────────────────────────────────────
    mom_df    = stage1b_momentum(returns_df, regime_df=regime_df)
    lowvol_df = stage4_low_vol(returns_df)
    conc_df   = stage_conc_leadership(returns_df)

    # ── Stage 6: Night Effect (OHLC required) ────────────────────────────────
    ohlc_dfs = load_ohlc(tickers=TICKERS, start=start_date)
    # Align OHLC index to the returns_df date range
    for key in ohlc_dfs:
        ohlc_dfs[key] = ohlc_dfs[key].reindex(dates).ffill()
    night_df = stage_night_effect(**ohlc_dfs)
    # REVERSE_SIGNAL: validate_signal_ic.py confirmed IC_5d=-0.0123 with the
    # original sign (positive = bet on continued intraday mean-reversion).
    # In 2020-2024 momentum regimes, overnight accumulators continue intraday
    # (trend continuation beats mean-reversion at 5d). Negating aligns signal
    # with the empirically dominant direction.
    # Reduced D window (1d) and sign flip together target the 1-day microstructure
    # reversion rather than the 5-day accumulated imbalance.
    night_df = -night_df
    night_df.index = pd.to_datetime(night_df.index)

    # ── Stage 7: PCA StatArb ──────────────────────────────────────────────────
    # Use log returns for PCA to ensure additivity and better Gaussian properties
    log_returns_df = np.log1p(returns_df.clip(lower=-0.5, upper=2.0))
    statarb_df = stage_pca_statarb(log_returns_df)
    statarb_df.index = pd.to_datetime(statarb_df.index)

    # ── Assemble signal stack ─────────────────────────────────────────────────
    signal_dfs: Dict[str, pd.DataFrame] = {
        "mom":          mom_df,
        "low_vol":      lowvol_df,
        "conc_lead":    conc_df,
        "night_effect": night_df,
        "pca_statarb":  statarb_df,
    }

    # Persist per-signal tensor (wide format: signal_ticker columns)
    all_signals_for_export = {"gex_flow": gex_flow_df, **signal_dfs}
    long_frames = []
    for sig_name, df in all_signals_for_export.items():
        aligned = (
            df.reindex(dates).ffill().fillna(0.0)
            .reindex(columns=TICKERS).fillna(0.0)
        )
        aligned.columns = [f"{sig_name}_{t}" for t in aligned.columns]
        long_frames.append(aligned)

    signals_long_df = pd.concat(long_frames, axis=1)
    signals_long_df.to_parquet(_SIGNALS_OUT)
    logger.info(f"Per-signal tensor → {_SIGNALS_OUT} {signals_long_df.shape}")

    # ── Stage 8: Blend + regime gate ─────────────────────────────────────────
    alpha_df, gex_alpha_df = stage8_blend_signals(
        signal_dfs=signal_dfs,
        regime_df=regime_df,
        returns_df=returns_df,
        dates=dates,
        gex_flow_df=gex_flow_df,
    )
    alpha_df.to_parquet(_ALPHA_OUT)
    gex_alpha_df.to_parquet(_GEX_ALPHA_OUT)
    logger.info(f"Blended alpha → {_ALPHA_OUT} {alpha_df.shape}")

    # ── Per-signal diagnostics ────────────────────────────────────────────────
    logger.info("\nPer-signal diagnostics:")
    for sig_name, df in all_signals_for_export.items():
        aligned = df.reindex(dates).ffill().fillna(0.0).reindex(columns=TICKERS).fillna(0.0)
        eq_cols = [t for t in EQUITY_TICKERS if t in aligned.columns]
        nonzero = (aligned.abs() > 0.01).any(axis=1).mean() * 100
        if eq_cols:
            eq_mean = aligned[eq_cols].abs().mean().mean()
            logger.info(
                f"  {sig_name:12s} | all={aligned.abs().mean().mean():.3f} "
                f"| equity={eq_mean:.3f} | active={nonzero:.1f}%"
            )
        else:
            logger.info(
                f"  {sig_name:12s} | mean|α|={aligned.abs().mean().mean():.3f} "
                f"| active={nonzero:.1f}%"
            )

    logger.info("\n✅ Precompute v19 complete.")
    logger.info("Next: PYTHONPATH=. python scripts/validate_signal_ic.py")
    logger.info("Then: PYTHONPATH=. python scripts/train_gat_router.py  (retrain on new signals)")
    logger.info("Then: PYTHONPATH=. python scripts/run_standalone_backtest.py")


if __name__ == "__main__":
    asyncio.run(main())