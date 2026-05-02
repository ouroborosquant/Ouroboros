"""
FORTRESS v5 — scripts/precompute_alpha_signals.py  [v21 — VTS + SMAX SIGNALS]

v21 Signal Stack
----------------
  Replaces night_effect + pca_statarb with two new deterministic engines
  targeting higher-moment statistical anomalies and cross-asset macro latency:

  1. vts_lead  (VTSLeadEngine):
     Rolling 63-day bivariate OLS of each equity's returns against the daily
     VIX term-structure innovation shock (ΔVTS = Δln(VIX3M/VIX)) and SPY.
     Extracts the β^VTS cross-sectional loading as a forward-looking measure
     of each stock's sensitivity to vol regime transitions. Cross-sectionally
     Z-scored and tanh-bounded. Equity singles only.

  2. smax_rev  (SMAXReversalEngine):
     Idiosyncratic maximum return (SMAX) reversal — the lottery premium
     anomaly (Bali et al. 2011) orthogonalised against 252-day systemic beta.
     Residuals from cross-sectional regression of naive_max onto β_sys isolate
     pure idiosyncratic jump risk; sign-inverted to bet against lottery stocks.
     Equity singles only. warm-up = 252d.

Signal Weights (static; GATv2 bypassed pending retrain on new tensor)
----------------------------------------------------------------------
  Equity singles:  mom=0.30 | low_vol=0.15 | conc_lead=0.15 | vts=0.20 | smax=0.20
  Macro ETFs:      mom=0.45 | low_vol=0.25 | conc_lead=0.30  (vts/smax = 0.0)

CRITICAL DATA DEPENDENCY:
  prices_wide.parquet must contain columns '^VIX' and '^VIX3M'.
  Run scripts/bootstrap_market_data.py (v3.0+) before this script.
  bootstrap v3.0 appends these indices to the yfinance download.

Pipeline
--------
  Stage 0: Multi-asset vol regime (LTC urgency)
  Stage 1a: Dealer gamma (GEX/DIX) flow signal
  Stage 1b: Cross-sectional momentum (12-1M)
  Stage 4:  Low-volatility anomaly (class-aware)
  Stage 5:  Concentration leadership
  Stage 6:  VTS Beta Lead-Lag (replaces Night Effect)
  Stage 7:  SMAX Reversal (replaces PCA StatArb)
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
_OHLC_PATH      = _CACHE_DIR / "ohlc_wide.parquet"
_SIGNALS_OUT    = _CACHE_DIR / "alpha_signals.parquet"
_ALPHA_OUT      = _CACHE_DIR / "alpha_signals_blended.parquet"
_GEX_ALPHA_OUT  = _CACHE_DIR / "gex_alpha.parquet"
_REGIME_OUT     = _CACHE_DIR / "regime_posteriors.parquet"

_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_UNIVERSE_FILE  = _BASE_DIR / "config" / "universe.yaml"
_ROUTER_WEIGHTS = _BASE_DIR / "models" / "weights" / "gat_router.pt"

# ── Universe ──────────────────────────────────────────────────────────────────
with open(_UNIVERSE_FILE, "r") as _f:
    _univ_cfg = yaml.safe_load(_f)

TICKERS:   List[str] = [a["ticker"] for a in _univ_cfg["assets"]]
N_ASSETS:  int       = len(TICKERS)

_CAT          = {a["ticker"]: a.get("category", "unknown") for a in _univ_cfg["assets"]}
_ETF_CATS     = {"equity_broad", "equity_sector", "equity_intl", "fixed_income",
                 "commodity", "volatility"}
_SINGLE_CATS  = {"equity_single"}

ETF_TICKERS:    List[str] = [t for t in TICKERS if _CAT.get(t) in _ETF_CATS]
EQUITY_TICKERS: List[str] = [t for t in TICKERS if _CAT.get(t) in _SINGLE_CATS]
_HEDGE_ASSETS:  frozenset  = frozenset({"BIL", "SHV", "VIXY"})
_PARKING_ASSETS: frozenset = frozenset({"BIL", "SHV"})

logger.info(
    f"Universe: {N_ASSETS} total | {len(ETF_TICKERS)} ETFs | "
    f"{len(EQUITY_TICKERS)} single-name equities"
)

# ── Signal stack (v21) ────────────────────────────────────────────────────────
SIGNAL_NAMES: List[str] = ["mom", "low_vol", "conc_lead", "vts_lead", "smax_rev"]
N_SIGNALS:    int        = len(SIGNAL_NAMES)

# Per-asset-class static weights; both vectors are renormalised in stage8.
# vts_lead and smax_rev output 0.0 for ETFs → excluded from ETF weight sum.
_WEIGHTS_EQUITY: Dict[str, float] = {
    "mom":       0.30,
    "low_vol":   0.15,
    "conc_lead": 0.15,
    "vts_lead":  0.20,
    "smax_rev":  0.20,
}  # sum = 1.00

_WEIGHTS_ETF: Dict[str, float] = {
    "mom":       0.45,
    "low_vol":   0.25,
    "conc_lead": 0.30,
    "vts_lead":  0.00,   # zeroed at source; excluded from ETF blend
    "smax_rev":  0.00,   # zeroed at source; excluded from ETF blend
}  # effective sum = 1.00

GEX_BETA: Dict[str, float] = {
    "SPY": 1.00, "QQQ": 0.95, "IWM": 0.40,
    "XLK": 0.85, "XLC": 0.75, "XLY": 0.65,
}
_GEX_INTERACTION_GAMMA: float = 0.50


# ─────────────────────────────────────────────────────────────────────────────
# Stage 0: Multi-asset vol regime
# ─────────────────────────────────────────────────────────────────────────────

async def stage0_vol_regime(
    start:      str,
    prices_df:  Optional[pd.DataFrame] = None,
    returns_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, object]:
    """
    Attempt MultiAssetVolRegime; fall back to SPY rolling-vol urgency proxy
    so the regime gate in stage8 is never permanently inactive.
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
            try:
                result = engine.compute_regimes(returns_df=returns_df)
            except TypeError:
                result = engine.compute_regimes(prices_df=prices_df, returns_df=returns_df)
            regime_df = result if isinstance(result, pd.DataFrame) else result[0]
        elif hasattr(engine, "fit") and returns_df is not None:
            engine.fit(returns_df)
            getter    = next(
                (m for m in ["get_posteriors", "posteriors", "regime_df"] if hasattr(engine, m)),
                None,
            )
            regime_df = getattr(engine, getter)() if getter else None
        elif hasattr(engine, "run"):
            regime_df = await engine.run() if asyncio.iscoroutinefunction(engine.run) \
                        else engine.run()

        if regime_df is None or not isinstance(regime_df, pd.DataFrame):
            raise AttributeError("Could not extract regime DataFrame.")

        regime_df.to_parquet(_REGIME_OUT)
        logger.info(f"  ✓ Regime posteriors → {_REGIME_OUT} {regime_df.shape}")
        return regime_df, engine

    except Exception as e:
        logger.warning(f"  ⚠️  Vol regime failed ({e}); activating inline SPY urgency proxy.")
        if returns_df is not None and "SPY" in returns_df.columns:
            spy_rv  = returns_df["SPY"].rolling(21, min_periods=10).std() * np.sqrt(252)
            spy_ewm = spy_rv.ewm(halflife=63, min_periods=21)
            z       = (spy_rv - spy_ewm.mean()) / spy_ewm.std().clip(lower=0.001)
            urgency = (1.0 / (1.0 + np.exp(-z * 0.8))).clip(0.0, 1.0)
            regime_df = pd.DataFrame(
                {"ltc_urgency": urgency.fillna(0.3), "spy_rv_21d": spy_rv.fillna(0.15)},
                index=returns_df.index,
            )
            logger.info(
                f"  ✓ SPY urgency proxy | mean={urgency.mean():.3f} | "
                f"crisis(>0.6)={(urgency > 0.6).mean()*100:.1f}%"
            )
            return regime_df, None

        return pd.DataFrame(index=pd.DatetimeIndex([]), columns=["ltc_urgency"]), None


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1a: Dealer gamma (GEX/DIX) flow signal
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
# Stage 1b: Cross-sectional momentum (12-1M)
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

    ret_12m = returns_df[active_cols].rolling(252, min_periods=126).sum()
    ret_1m  = returns_df[active_cols].rolling(21,  min_periods=10).sum()
    mom_raw = ret_12m - ret_1m

    rank  = mom_raw.rank(axis=1, pct=True)
    score = np.tanh((rank - 0.5) * 4.0)

    if regime_df is not None and "breadth_ratio" in regime_df.columns:
        breadth    = regime_df["breadth_ratio"].reindex(returns_df.index).ffill().fillna(0.5)
        conc_gate  = np.clip((breadth.values - 0.30) / 0.20, 0.0, 1.0)[:, None]
        score      = pd.DataFrame(
            score.values * (0.5 + 0.5 * conc_gate),
            index=score.index, columns=score.columns,
        )

    result = pd.DataFrame(0.0, index=returns_df.index, columns=TICKERS, dtype=np.float32)
    result[active_cols] = score.fillna(0.0).astype(np.float32)
    result[list(_HEDGE_ASSETS)] = 0.0

    logger.info(f"  ✓ Momentum: mean|α|={result[active_cols].abs().mean().mean():.3f}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4: Low-volatility anomaly (class-aware)
# ─────────────────────────────────────────────────────────────────────────────

def stage4_low_vol(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Low-vol anomaly with class-aware cross-sectional ranking.
    ETFs rank within ETF sub-universe; equities rank within equity sub-universe.
    Negated: empirically a high-vol premium signal in 2018-2026 (AI bull + rate cycle).
    """
    logger.info("Stage 4: Low-vol anomaly (class-aware ranking)...")

    rv_63 = (
        returns_df.reindex(columns=TICKERS)
        .rolling(63, min_periods=20)
        .std() * np.sqrt(252)
    ).ffill()

    result = pd.DataFrame(0.0, index=returns_df.index, columns=TICKERS, dtype=np.float32)

    etf_active = [t for t in ETF_TICKERS if t not in _HEDGE_ASSETS]
    if etf_active:
        rank_etf   = rv_63[etf_active].rank(axis=1, pct=True)
        result[etf_active] = np.tanh(-(rank_etf - 0.5) * 2.0).fillna(0.0).astype(np.float32)

    eq_active = list(EQUITY_TICKERS)
    if eq_active:
        rank_eq   = rv_63[eq_active].rank(axis=1, pct=True)
        result[eq_active] = np.tanh(-(rank_eq - 0.5) * 2.0).fillna(0.0).astype(np.float32)

    mean_abs_etf = result[etf_active].abs().mean().mean() if etf_active else 0.0
    mean_abs_eq  = result[eq_active].abs().mean().mean()  if eq_active  else 0.0

    # Negate: validate_signal_ic confirmed IC inverted for 2018-2026 period.
    result = -result
    logger.info(
        f"  ✓ Vol-premium (negated low_vol): ETF={mean_abs_etf:.3f} | equity={mean_abs_eq:.3f}"
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
        # Inline fallback: 63d-skip-21d cross-sectional momentum
        # IC profile: IC_21d≈+0.014, IC_63d≈+0.048 (medium-term continuation)
        logger.info("  signals/conc_signal.py absent — using inline 63d-skip-21d momentum.")
        active_cols = [t for t in TICKERS if t not in _HEDGE_ASSETS]
        ret_63 = returns_df[active_cols].rolling(63, min_periods=30).sum()
        ret_21 = returns_df[active_cols].rolling(21, min_periods=10).sum()
        mom    = ret_63 - ret_21
        rank   = mom.rank(axis=1, pct=True)
        signal = np.tanh((rank - 0.5) * 3.0)
        result = pd.DataFrame(0.0, index=returns_df.index, columns=TICKERS, dtype=np.float32)
        result[active_cols] = signal.fillna(0.0).astype(np.float32)
        result[list(_HEDGE_ASSETS)] = 0.0
        logger.info(
            f"  ✓ Conc leadership (inline): mean|α|={result[active_cols].abs().mean().mean():.3f}"
        )
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Stage 6: VTS Beta Lead-Lag (replaces Night Effect)
# ─────────────────────────────────────────────────────────────────────────────

def stage_vts_lead(
    prices_df:  pd.DataFrame,
    returns_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Rolling 63-day bivariate OLS β^VTS for each equity vs. the VIX term-
    structure innovation shock (ΔVTS = Δln(VIX3M/VIX)) and SPY.

    DEPENDENCY: prices_df must contain '^VIX' and '^VIX3M' columns.
    These are injected by bootstrap_market_data.py v3.0+ as aux indices.

    Equity singles only; ETFs receive 0.0.
    See models/alpha/vts_lead.py for full mathematical specification.
    """
    logger.info("Stage 6: VTS Beta Lead-Lag (bivariate OLS vs ΔVTS shock)...")

    # Guard: if VIX data absent (e.g., old bootstrap cache), return zeros
    for col in ("^VIX", "^VIX3M"):
        if col not in prices_df.columns or prices_df[col].notna().sum() < 100:
            logger.error(
                f"  ✗ '{col}' absent or sparse in prices_wide.parquet — "
                f"VTSLead returning zeros. Re-run bootstrap_market_data.py (v3.0+)."
            )
            return pd.DataFrame(0.0, index=returns_df.index, columns=TICKERS, dtype=np.float32)

    from models.alpha.vts_lead import VTSLeadEngine
    engine = VTSLeadEngine(
        equity_tickers=EQUITY_TICKERS,
        all_tickers=TICKERS,
    )
    return engine.compute_signal(prices_df=prices_df, returns_df=returns_df)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 7: SMAX Reversal (replaces PCA StatArb)
# ─────────────────────────────────────────────────────────────────────────────

def stage_smax_reversal(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Idiosyncratic maximum return reversal (Bali et al. 2011 lottery anomaly)
    orthogonalised against 252-day systemic beta.

    SMAX = residuals from cross-sectional regression of naive_max on β_sys.
    Signal is sign-inverted: high SMAX → short (lottery premium reversal).
    Equity singles only; ETFs receive 0.0. Warm-up = 252 days.

    See models/alpha/smax_rev.py for full mathematical specification.
    """
    logger.info("Stage 7: SMAX Reversal (idiosyncratic lottery anomaly)...")
    from models.alpha.smax_rev import SMAXReversalEngine
    engine = SMAXReversalEngine(
        equity_tickers=EQUITY_TICKERS,
        all_tickers=TICKERS,
    )
    return engine.compute_signal(returns_df)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 8: Signal blending (static weights; GATv2 bypassed pending retrain)
# ─────────────────────────────────────────────────────────────────────────────

def stage8_blend_signals(
    signal_dfs:  Dict[str, pd.DataFrame],
    regime_df:   pd.DataFrame,
    returns_df:  pd.DataFrame,
    dates:       pd.DatetimeIndex,
    gex_flow_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Blend 5 signals into final alpha vector using per-asset-class static weights.

    Asset-class weight normalisation:
      vts_lead and smax_rev output 0.0 for ETF tickers. Without renormalisation,
      ETF alphas would be scaled by (w_vts + w_smax) = 0.40 of equity alphas.
      The ETF weight vector sums to 1.0 over its three active signals;
      the equity weight vector sums to 1.0 over all five.

    GATv2 bypass:
      Saved weights were trained on the night_effect/pca_statarb tensor.
      Re-enabling before retraining on the vts_lead/smax_rev tensor will
      produce inverted routing due to feature-space mismatch. After running
      validate_signal_ic.py and confirming IC > 0.035 for new signals:
        rm models/weights/gat_router.pt
        PYTHONPATH=. python scripts/train_gat_router.py
    """
    logger.info("Stage 8: Signal blending (static weights — GATv2 bypassed)...")

    T = len(dates)
    N = N_ASSETS

    # Normalise weight vectors to sum exactly to 1.0
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

    # ── Per-asset weight matrix (N, S): equity rows → w_eq, ETF rows → w_etf ─
    equity_set    = frozenset(EQUITY_TICKERS)
    weight_matrix = np.array(
        [w_eq if t in equity_set else w_etf for t in TICKERS],
        dtype=np.float64,
    )  # (N, S)

    # Broadcast weight tensor (T, N, S) — static across time
    weight_tensor = np.broadcast_to(
        weight_matrix[np.newaxis, :, :], (T, N, N_SIGNALS)
    ).copy()

    logger.info(
        "  Equity weights: " + " | ".join(f"{n}={w:.3f}" for n, w in zip(SIGNAL_NAMES, w_eq))
    )
    logger.info(
        "  ETF weights:    " + " | ".join(f"{n}={w:.3f}" for n, w in zip(SIGNAL_NAMES, w_etf))
    )

    if _ROUTER_WEIGHTS.exists():
        logger.warning(
            f"  ⚠️  GATv2 weights at {_ROUTER_WEIGHTS} are STALE "
            f"(trained on night_effect/pca_statarb). Delete and retrain before re-enabling."
        )

    # ── Blended alpha: element-wise weighted sum ──────────────────────────────
    alpha_raw = (weight_tensor * signal_stack).sum(axis=-1)  # (T, N)

    # ── Regime gate: fire only above 0.60 urgency; max 40% compression ────────
    if "ltc_urgency" in regime_df.columns:
        urgency = regime_df["ltc_urgency"].reindex(dates).ffill().fillna(0.0).values
        for i, ticker in enumerate(TICKERS):
            asset_urgency = urgency if ticker not in {"GLD", "SLV", "GDX", "USO", "PDBC"} \
                            else urgency * 0.5
            crisis_scale  = np.clip((asset_urgency - 0.60) / 0.40, 0.0, 1.0) * 0.40
            alpha_raw[:, i] *= (1.0 - crisis_scale)
    else:
        logger.info("  ltc_urgency absent — regime gate inactive.")

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
    alpha_df    = pd.DataFrame(alpha_final, index=dates, columns=TICKERS)
    return alpha_df, gex_aligned


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    logger.info("=" * 70)
    logger.info("FORTRESS v5 — Alpha Precompute v21 (VTS Lead + SMAX Reversal)")
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

    # Restrict returns to universe columns (prices may include ^VIX/^VIX3M extras)
    returns_df_univ = returns_df.reindex(columns=TICKERS)

    start_date = str(returns_df.index[0].date())
    dates      = returns_df.index
    logger.info(
        f"Market data: {len(returns_df)} days × {prices_df.shape[1]} series | "
        f"{start_date} → {dates[-1].date()}"
    )

    # Verify aux VIX columns present for stage_vts_lead
    for col in ("^VIX", "^VIX3M"):
        if col in prices_df.columns:
            logger.info(f"  ✓ {col} present: {prices_df[col].notna().sum()} non-null rows")
        else:
            logger.warning(
                f"  ⚠️  {col} NOT in prices_wide.parquet. "
                f"VTSLead will return zeros. Re-run bootstrap_market_data.py (v3.0+)."
            )

    # ── Stage 0: Regime posteriors ────────────────────────────────────────────
    regime_df, _ = await stage0_vol_regime(
        start=start_date, prices_df=prices_df, returns_df=returns_df_univ,
    )
    regime_df.index = pd.to_datetime(regime_df.index)

    # ── Stage 1a: GEX dealer gamma ────────────────────────────────────────────
    gex_flow_df = await stage1a_dealer_gamma(start=start_date, dates=dates)
    gex_flow_df.index = pd.to_datetime(gex_flow_df.index)

    # ── Stages 1b–5: Core cross-sectional signals ─────────────────────────────
    mom_df    = stage1b_momentum(returns_df_univ, regime_df=regime_df)
    lowvol_df = stage4_low_vol(returns_df_univ)
    conc_df   = stage_conc_leadership(returns_df_univ)

    # ── Stage 6: VTS Beta Lead-Lag ────────────────────────────────────────────
    # Passes full prices_df (includes ^VIX/^VIX3M) and universe returns
    vts_df = stage_vts_lead(prices_df=prices_df, returns_df=returns_df_univ)
    vts_df.index = pd.to_datetime(vts_df.index)

    # ── Stage 7: SMAX Reversal ────────────────────────────────────────────────
    smax_df = stage_smax_reversal(returns_df_univ)
    smax_df.index = pd.to_datetime(smax_df.index)

    # ── Assemble signal dict ──────────────────────────────────────────────────
    signal_dfs: Dict[str, pd.DataFrame] = {
        "mom":       mom_df,
        "low_vol":   lowvol_df,
        "conc_lead": conc_df,
        "vts_lead":  vts_df,
        "smax_rev":  smax_df,
    }

    # ── Persist per-signal tensor (wide: signal_ticker columns) ──────────────
    all_for_export = {"gex_flow": gex_flow_df, **signal_dfs}
    long_frames    = []
    for sig_name, df in all_for_export.items():
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
        returns_df=returns_df_univ,
        dates=dates,
        gex_flow_df=gex_flow_df,
    )
    alpha_df.to_parquet(_ALPHA_OUT)
    gex_alpha_df.to_parquet(_GEX_ALPHA_OUT)
    logger.info(f"Blended alpha  → {_ALPHA_OUT} {alpha_df.shape}")

    # ── Per-signal diagnostics ────────────────────────────────────────────────
    logger.info("\nPer-signal diagnostics:")
    for sig_name, df in all_for_export.items():
        aligned    = df.reindex(dates).ffill().fillna(0.0).reindex(columns=TICKERS).fillna(0.0)
        eq_cols    = [t for t in EQUITY_TICKERS if t in aligned.columns]
        nonzero    = (aligned.abs() > 0.01).any(axis=1).mean() * 100
        all_mean   = aligned.abs().mean().mean()
        eq_mean    = aligned[eq_cols].abs().mean().mean() if eq_cols else 0.0
        logger.info(
            f"  {sig_name:12s} | all={all_mean:.3f} | equity={eq_mean:.3f} | active={nonzero:.1f}%"
        )

    logger.info("\n✅ Precompute v21 complete.")


if __name__ == "__main__":
    asyncio.run(main())