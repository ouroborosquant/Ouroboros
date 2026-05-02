"""
FORTRESS v5 — scripts/precompute_alpha_signals.py  [v22.0 — Blueprint Suite]

5-Signal Stack: low_vol | ramom_ts | odpv_vwap | clv_flow | dtfe_trend

Pipeline
--------
Stage 0  : Multi-asset vol regime (LTC urgency proxy fallback)
Stage 1  : Raw OHLCV download via yfinance (Open, High, Low, Close, Volume)
Stage 2  : Low-volatility anomaly (class-aware cross-sectional ranking)
Stage 3  : RAMOM-TS  — Yang-Zhang-normalised EMA velocity
Stage 4  : ODPV-VWAP — rolling VWAP spread oscillator
Stage 5  : CLV-Flow  — close-location-value × volume accumulation
Stage 6  : DTFE-Trend — Kaufman fractal efficiency × direction
Stage 7  : Static-weight blend (GATv2 bypassed until retrained on new tensor)
Stage 8  : Persist signal tensor, alpha parquet, GEX-alpha parquet

GATv2 status: BYPASSED. Saved weights at models/weights/gat_router.pt were
trained on the legacy [mom, low_vol, conc_lead, night_effect, pca_statarb]
feature tensor. Loading them on the new 5-signal tensor will invert routing.
After running validate_signal_ic.py and confirming IC > 0.035 per signal:
    rm models/weights/gat_router.pt
    PYTHONPATH=. python scripts/train_gat_router.py

Causality guarantee
-------------------
All rolling operations use only past data: pandas rolling/ewm with
adjust=False (causal EMA), and all Z-score denominators are computed on
windows that are strictly left-closed (no centre=True).
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Ouroboros.Precompute")

# ─────────────────────────────────────────────────────────────────────────────
# Universe constants
# ─────────────────────────────────────────────────────────────────────────────

def _load_universe() -> List[str]:
    for candidate in ["config/universe.yaml", "universe.yaml", "../config/universe.yaml"]:
        p = Path(candidate)
        if p.exists():
            with open(p) as fh:
                data = yaml.safe_load(fh)
            return [a["ticker"] for a in data["assets"]]
    # Hard-coded fallback: canonical 25-ticker ETF universe
    return [
        "SPY","QQQ","IWM","TLT","HYG","LQD","GLD","SLV","GDX",
        "XLE","XLF","XLK","XLV","XLU","XLI","XLP","XLY","XLB","XLC",
        "VIXY","BIL","SHV","USO","PDBC","COWZ",
    ]


TICKERS: List[str] = _load_universe()
N_ASSETS:  int     = len(TICKERS)

# ── yfinance ticker normalisation ─────────────────────────────────────────────
# BRK.B is listed as BRK-B on yfinance; dots in ticker names cause silent 404s.
# Map canonical universe names → yfinance query strings; reverse-map on return.
_YF_TICKER_MAP: Dict[str, str] = {"BRK.B": "BRK-B"}
_YF_REVERSE_MAP: Dict[str, str] = {v: k for k, v in _YF_TICKER_MAP.items()}

def _yf_tickers(tickers: List[str]) -> List[str]:
    """Convert universe ticker list to yfinance-safe query strings."""
    return [_YF_TICKER_MAP.get(t, t) for t in tickers]

def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Reverse-map yfinance column names back to canonical universe names."""
    return df.rename(columns=_YF_REVERSE_MAP)

# ── Asset-class routing ───────────────────────────────────────────────────────
# Populated from universe.yaml categories; handles mixed ETF+equity universe.
def _categorise_universe() -> Tuple[List[str], List[str]]:
    """Return (etf_tickers, equity_single_tickers) from yaml if available."""
    for candidate in ["config/universe.yaml", "universe.yaml", "../config/universe.yaml"]:
        p = Path(candidate)
        if p.exists():
            with open(p) as fh:
                data = yaml.safe_load(fh)
            _cat = {a["ticker"]: a.get("category", "") for a in data["assets"]}
            _ETF_CATS = {
                "equity_broad", "equity_sector", "equity_intl",
                "fixed_income", "commodity", "volatility",
            }
            etfs    = [t for t in TICKERS if _cat.get(t, "") in _ETF_CATS]
            equities = [t for t in TICKERS if _cat.get(t, "") == "equity_single"]
            return etfs, equities
    # Fallback: everything is ETF
    return TICKERS, []

ETF_TICKERS, EQUITY_TICKERS = _categorise_universe()
_HEDGE_ASSETS: List[str] = ["BIL", "SHV", "TLT", "LQD"]

SIGNAL_NAMES: List[str] = ["low_vol", "ramom_ts", "odpv_vwap", "clv_flow", "dtfe_trend"]
N_SIGNALS:    int        = len(SIGNAL_NAMES)

# ── Per-asset-class static blend weights ─────────────────────────────────────
# ETF sub-universe: low_vol has the strongest IC (IC_21d≈+0.048 in validation).
#   Blueprint structural-flow signals have moderate IC on ETFs — volume patterns
#   are noisier at the ETF level due to creation/redemption flow mixing with
#   secondary market volume.
# Equity sub-universe: structural-flow signals (RAMOM, ODPV, CLV, DTFE) were
#   designed for single-name mega-caps where TWAP/VWAP execution is dominant.
#   low_vol retains meaningful weight as a risk-adjusted baseline.
_WEIGHTS_ETF: Dict[str, float] = {
    "low_vol":    0.35,
    "ramom_ts":   0.25,
    "odpv_vwap":  0.20,
    "clv_flow":   0.10,
    "dtfe_trend": 0.10,
}
_WEIGHTS_EQUITY: Dict[str, float] = {
    "low_vol":    0.15,
    "ramom_ts":   0.30,
    "odpv_vwap":  0.25,
    "clv_flow":   0.20,
    "dtfe_trend": 0.10,
}
assert abs(sum(_WEIGHTS_ETF.values())    - 1.0) < 1e-9
assert abs(sum(_WEIGHTS_EQUITY.values()) - 1.0) < 1e-9

# ── Output paths ──────────────────────────────────────────────────────────────
_OUTDIR      = Path("data/processed")
_ALPHA_OUT   = _OUTDIR / "alpha_signals.parquet"
_SIGNALS_OUT = _OUTDIR / "signal_tensor.parquet"
_REGIME_OUT  = _OUTDIR / "regime_posteriors.parquet"
_GEX_ALPHA_OUT = _OUTDIR / "gex_alpha_signals.parquet"

_START_DATE  = "2018-01-01"
_WARMUP_BARS = 273   # max(252 Z-score + 21 KER/VWAP) to ensure clean signals


# ─────────────────────────────────────────────────────────────────────────────
# Stage 0: Vol regime
# ─────────────────────────────────────────────────────────────────────────────

async def stage0_vol_regime(
    start:      str,
    prices_df:  Optional[pd.DataFrame] = None,
    returns_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, object]:
    try:
        from signals.vol_regime import MultiAssetVolRegime
        engine = MultiAssetVolRegime()
        
        # FIXED: Use the correct API methods from the VIX Patch
        await engine.load_history(start=start)
        regime_df, _ = engine.get_tensor_series(TICKERS)

        _OUTDIR.mkdir(parents=True, exist_ok=True)
        regime_df.to_parquet(_REGIME_OUT)
        logger.info(f"  ✓ Regime posteriors → {_REGIME_OUT} {regime_df.shape}")
        return regime_df, engine
    except Exception as exc:
        logger.warning(f"  ⚠️  Vol regime failed ({exc}); falling back to SPY urgency proxy.")
        if returns_df is not None and "SPY" in returns_df.columns:
            spy_rv  = returns_df["SPY"].rolling(21, min_periods=10).std() * np.sqrt(252)
            spy_ewm = spy_rv.ewm(halflife=63, min_periods=21)
            z       = (spy_rv - spy_ewm.mean()) / spy_ewm.std().clip(lower=0.001)
            urgency = (1.0 / (1.0 + np.exp(-z * 0.8))).clip(0.0, 1.0)
            regime_df = pd.DataFrame(
                {"ltc_urgency": urgency.fillna(0.3), "spy_rv_21d": spy_rv.fillna(0.15)},
                index=returns_df.index,
            )
            return regime_df, None
        return pd.DataFrame(index=pd.DatetimeIndex([]), columns=["ltc_urgency"]), None


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: OHLCV data download
# ─────────────────────────────────────────────────────────────────────────────

def stage1_download_ohlcv(start: str) -> Dict[str, pd.DataFrame]:
    """
    Download full OHLCV for all tickers via yfinance.

    Returns dict with keys: "open", "high", "low", "close", "volume"
    Each value is (T, N) DataFrame indexed by trading date.

    Adjusted prices (auto_adjust=True) are used for open/high/low/close to
    ensure split-continuity. Volume is kept as raw consolidated shares — the
    VWAP and CLV-Flow calculations require volume in consistent units.
    """
    import yfinance as yf

    logger.info(f"Stage 1: Downloading OHLCV for {N_ASSETS} tickers from {start}...")

    # BRK.B dot-notation silently 404s on yfinance; use BRK-B for the query
    yf_tickers = _yf_tickers(TICKERS)

    raw = yf.download(
        tickers    = yf_tickers,
        start      = start,
        auto_adjust= True,
        progress   = False,
        threads    = True,
    )

    def _extract(key: str) -> pd.DataFrame:
        """Pull a price field, flatten MultiIndex, reverse yfinance names, reindex."""
        if isinstance(raw.columns, pd.MultiIndex):
            df = raw[key]
        else:
            df = raw[[key]]
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        df = _normalise_columns(df)   # BRK-B -> BRK.B
        return df.reindex(columns=TICKERS)

    ohlcv = {
        "open":   _extract("Open"),
        "high":   _extract("High"),
        "low":    _extract("Low"),
        "close":  _extract("Close"),
        "volume": _extract("Volume"),
    }

    # ── Integrity check: guard against yfinance adjusted-close H < C ─────────
    ohlcv["high"] = np.maximum(ohlcv["high"], ohlcv["close"])
    ohlcv["low"]  = np.minimum(ohlcv["low"],  ohlcv["close"])

    close_valid = ohlcv["close"].notna().any(axis=1)
    dates       = ohlcv["close"].loc[close_valid].index
    logger.info(f"  ✓ OHLCV downloaded: {len(TICKERS)} tickers × {len(dates)} trading days")

    return ohlcv


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Low-volatility anomaly
# ─────────────────────────────────────────────────────────────────────────────

def stage2_low_vol(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Class-aware cross-sectional low-volatility anomaly.

    Ranks assets by 63-day realised volatility within their asset class,
    maps rank to [−1, +1] via 2·(rank_pct − 0.5), then negates (empirically
    IC is inverted for 2018-2026: high-vol premium dominates in momentum-bull
    environments). Returns tanh-bounded signal.
    """
    logger.info("Stage 2: Low-vol anomaly (class-aware ranking)...")

    rv_63 = (
        returns_df.reindex(columns=TICKERS)
        .rolling(63, min_periods=20)
        .std() * np.sqrt(252)
    ).ffill()

    result = pd.DataFrame(0.0, index=returns_df.index, columns=TICKERS, dtype=np.float32)

    etf_active = [t for t in ETF_TICKERS if t not in _HEDGE_ASSETS]
    if etf_active:
        rank_etf           = rv_63[etf_active].rank(axis=1, pct=True)
        result[etf_active] = np.tanh(-(rank_etf - 0.5) * 2.0).fillna(0.0).astype(np.float32)

    # Equity singles ranked within equity sub-universe only.
    # Cross-ranking with ETFs collapses equity IC: single-name vols are
    # structurally higher, so all equities rank high and get a uniform signal.
    eq_active = [t for t in EQUITY_TICKERS if t in rv_63.columns]
    if eq_active:
        rank_eq           = rv_63[eq_active].rank(axis=1, pct=True)
        result[eq_active] = np.tanh(-(rank_eq - 0.5) * 2.0).fillna(0.0).astype(np.float32)

    result = -result
    logger.info(
        f"  ✓ low_vol (negated): mean|α|={result.abs().mean().mean():.4f}"
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: RAMOM-TS
# ─────────────────────────────────────────────────────────────────────────────

def stage3_ramom_ts(ohlcv: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Risk-Adjusted Time-Series Momentum via Yang-Zhang volatility normalisation.
    See models/alpha/ramom_ts.py for full mathematical specification.
    """
    logger.info("Stage 3: RAMOM-TS (Yang-Zhang normalised momentum)...")
    from models.alpha.ramom_ts import RAMOMTSEngine

    engine = RAMOMTSEngine(tickers=TICKERS)
    result = engine.compute_signal(
        prices_df = ohlcv["close"],
        open_df   = ohlcv["open"],
        high_df   = ohlcv["high"],
        low_df    = ohlcv["low"],
    )
    logger.info(f"  ✓ ramom_ts: mean|α|={result.abs().mean().mean():.4f}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4: ODPV-VWAP
# ─────────────────────────────────────────────────────────────────────────────

def stage4_odpv_vwap(ohlcv: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Dynamic VWAP Oscillator — rolling 21-day VWAP spread Z-score.
    See models/alpha/odpv_vwap.py for full mathematical specification.
    """
    logger.info("Stage 4: ODPV-VWAP (institutional VWAP footprint)...")
    from models.alpha.odpv_vwap import ODPVEngine

    engine = ODPVEngine(tickers=TICKERS)
    result = engine.compute_signal(
        close_df  = ohlcv["close"],
        volume_df = ohlcv["volume"],
    )
    logger.info(f"  ✓ odpv_vwap: mean|α|={result.abs().mean().mean():.4f}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5: CLV-Flow
# ─────────────────────────────────────────────────────────────────────────────

def stage5_clv_flow(ohlcv: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Close Location Value × volume accumulation flow.
    See models/alpha/clv_flow.py for full mathematical specification.
    """
    logger.info("Stage 5: CLV-Flow (intraday close-location accumulation)...")
    from models.alpha.clv_flow import CLVFlowEngine

    engine = CLVFlowEngine(tickers=TICKERS)
    result = engine.compute_signal(
        high_df   = ohlcv["high"],
        low_df    = ohlcv["low"],
        close_df  = ohlcv["close"],
        volume_df = ohlcv["volume"],
    )
    logger.info(f"  ✓ clv_flow: mean|α|={result.abs().mean().mean():.4f}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Stage 6: DTFE-Trend
# ─────────────────────────────────────────────────────────────────────────────

def stage6_dtfe_trend(ohlcv: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Directional Trend Fractal Efficiency — Kaufman Efficiency Ratio ×
    sign(21-day return).
    See models/alpha/dtfe_trend.py for full mathematical specification.
    """
    logger.info("Stage 6: DTFE-Trend (fractal efficiency noise filter)...")
    from models.alpha.dtfe_trend import DTFETrendEngine

    engine = DTFETrendEngine(tickers=TICKERS)
    result = engine.compute_signal(close_df=ohlcv["close"])
    logger.info(f"  ✓ dtfe_trend: mean|α|={result.abs().mean().mean():.4f}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
    logger.info("Stage 7: Static-weight blend (GATv2 bypassed — retrain required)...")
# ─────────────────────────────────────────────────────────────────────────────

def stage7_blend(
    signal_dfs:  Dict[str, pd.DataFrame],
    regime_df:   pd.DataFrame,
    dates:       pd.DatetimeIndex,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Blend 5 signals into the final alpha vector.

    GATv2 bypass rationale: saved weights at models/weights/gat_router.pt
    were trained on legacy [mom, low_vol, conc_lead, night_effect, pca_statarb].
    Feature-space mismatch will produce inverted routing; bypass is safer.
    After retraining: remove the bypass block and call SignalRouterGAT.infer_alpha.

    Regime gate: ltc_urgency scalar compresses alpha toward zero during high-
    volatility regimes where signal IC degrades. Gate = 0.3 + 0.7 × (1 − urgency)
    preserves a 30% baseline alpha floor to avoid full position liquidation.
    """
    logger.info("Stage 7: Static-weight blend (GATv2 bypassed — retrain required)...")

    T = len(dates)
    N = N_ASSETS
    # Per-asset weight matrix (N, S): equity singles use equity-tuned weights
    equity_set = frozenset(EQUITY_TICKERS)
    w_etf_arr  = np.array([_WEIGHTS_ETF.get(n, 0.0)    for n in SIGNAL_NAMES], dtype=np.float32)
    w_eq_arr   = np.array([_WEIGHTS_EQUITY.get(n, 0.0) for n in SIGNAL_NAMES], dtype=np.float32)
    weight_mat = np.stack(
        [w_eq_arr if t in equity_set else w_etf_arr for t in TICKERS]
    )  # (N, S)

    # Build signal stack (T, N, S)
    stack = np.zeros((T, N, N_SIGNALS), dtype=np.float32)
    for s_idx, sig_name in enumerate(SIGNAL_NAMES):
        if sig_name not in signal_dfs:
            logger.warning(f"  Signal '{sig_name}' absent — using zeros for blend.")
            continue
        aligned = (
            signal_dfs[sig_name]
            .reindex(dates)
            .ffill()
            .fillna(0.0)
            .reindex(columns=TICKERS)
            .fillna(0.0)
            .values.astype(np.float32)
        )
        stack[:, :, s_idx] = aligned

    # Element-wise blend: (T,N,S) * (N,S)[broadcast] -> (T,N)
    alpha_raw = (stack * weight_mat[None, :, :]).sum(axis=2)  # (T, N)

    # ── Regime gate: scale alpha by vol-regime urgency proxy ─────────────────
    if "ltc_urgency" in regime_df.columns:
        urgency = (
            regime_df["ltc_urgency"]
            .reindex(dates)
            .ffill()
            .fillna(0.3)
            .clip(0.0, 1.0)
            .values[:, None]     # (T, 1) for broadcast over N
        )
        # Gate ∈ [0.30, 1.00]: preserves 30% floor when urgency = 1.0 (crisis)
        gate = 0.30 + 0.70 * (1.0 - urgency)
        alpha_gated = alpha_raw * gate
    else:
        logger.warning("  Regime gate unavailable; bypassing.")
        alpha_gated = alpha_raw

    # ── Turnover Smoothing & tanh re-normalisation ───────────────────────────
    # Apply a 3-day exponential smoothing to the gated alpha to kill high-frequency
    # noise and drop daily turnover. Then apply tanh to bound the optimizer.
    raw_alpha_df = pd.DataFrame(alpha_gated * 2.0, index=dates, columns=TICKERS)
    smoothed_alpha = raw_alpha_df.ewm(span=3, min_periods=1).mean()

    alpha_final = np.tanh(smoothed_alpha.values).astype(np.float32)
    alpha_df = pd.DataFrame(alpha_final, index=dates, columns=TICKERS)

    # ── GEX alpha: equity-only subset, currently returns alpha_df copy ────────
    # Full GEX/DIX integration deferred; placeholder preserves downstream API.
    gex_alpha_df = alpha_df.copy()

    logger.info(
        f"  ✓ Blend complete | mean|α|={alpha_df.abs().mean().mean():.4f} | "
        f"gate_mean={gate.mean():.3f}" if "ltc_urgency" in regime_df.columns
        else f"  ✓ Blend complete | mean|α|={alpha_df.abs().mean().mean():.4f}"
    )
    return alpha_df, gex_alpha_df


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    logger.info("=" * 70)
    logger.info("FORTRESS v5 — precompute_alpha_signals.py  [v22.0 — Blueprint Suite]")
    logger.info(f"Universe     : {TICKERS}")
    logger.info(f"Signal stack : {SIGNAL_NAMES}")
    logger.info(f"ETF blend: {_WEIGHTS_ETF} | Equity blend: {_WEIGHTS_EQUITY}")
    logger.info("=" * 70)

    _OUTDIR.mkdir(parents=True, exist_ok=True)

    # ── Stage 1: Download OHLCV ───────────────────────────────────────────────
    ohlcv = stage1_download_ohlcv(start=_START_DATE)
    close   = ohlcv["close"]
    returns = np.log(close / close.shift(1)).astype(np.float32)

    # Full date index (close-aligned, non-NaN majority rows)
    dates: pd.DatetimeIndex = close.dropna(how="all").index

    # ── Stage 0: Vol regime ───────────────────────────────────────────────────
    regime_df, _ = await stage0_vol_regime(
        start      = _START_DATE,
        prices_df  = close,
        returns_df = returns,
    )

    # ── Stages 2-6: Individual signal engines ────────────────────────────────
    lowvol_df   = stage2_low_vol(returns_df=returns)
    ramom_df    = stage3_ramom_ts(ohlcv=ohlcv)
    odpv_df     = stage4_odpv_vwap(ohlcv=ohlcv)
    clv_df      = stage5_clv_flow(ohlcv=ohlcv)
    dtfe_df     = stage6_dtfe_trend(ohlcv=ohlcv)

    signal_dfs: Dict[str, pd.DataFrame] = {
        "low_vol":    lowvol_df,
        "ramom_ts":   ramom_df,
        "odpv_vwap":  odpv_df,
        "clv_flow":   clv_df,
        "dtfe_trend": dtfe_df,
    }

    # ── Stage 7: Blend ────────────────────────────────────────────────────────
    alpha_df, gex_alpha_df = stage7_blend(
        signal_dfs = signal_dfs,
        regime_df  = regime_df,
        dates      = dates,
    )

    # ── Persist per-signal tensor (wide: {signal}_{ticker} columns) ──────────
    long_frames: list[pd.DataFrame] = []
    for sig_name, df in signal_dfs.items():
        aligned = (
            df.reindex(dates).ffill().fillna(0.0)
            .reindex(columns=TICKERS).fillna(0.0)
        )
        
        # --- TURNOVER SMOOTHING INJECTED HERE ---
        # Smooth the raw signals before the GATv2 sees them
        aligned = aligned.ewm(span=3, min_periods=1).mean()
        # ----------------------------------------
        
        aligned.columns = [f"{sig_name}_{t}" for t in aligned.columns]
        long_frames.append(aligned)

    signal_tensor = pd.concat(long_frames, axis=1)
    signal_tensor.to_parquet(_SIGNALS_OUT)
    alpha_df.to_parquet(_ALPHA_OUT)
    gex_alpha_df.to_parquet(_GEX_ALPHA_OUT)

    logger.info(f"Signal tensor → {_SIGNALS_OUT}  {signal_tensor.shape}")
    logger.info(f"Alpha parquet → {_ALPHA_OUT}     {alpha_df.shape}")

    # ── Per-signal diagnostics ────────────────────────────────────────────────
    logger.info("\n── Per-signal diagnostics ──")
    for sig_name, df in signal_dfs.items():
        aligned  = df.reindex(dates).ffill().fillna(0.0).reindex(columns=TICKERS).fillna(0.0)
        nonzero  = (aligned.abs() > 0.01).any(axis=1).mean() * 100
        mean_abs = aligned.abs().mean().mean()
        max_abs  = aligned.abs().max().max()
        logger.info(
            f"  {sig_name:12s} | mean|α|={mean_abs:.4f} | "
            f"max|α|={max_abs:.4f} | active={nonzero:.1f}%"
        )

    logger.info("\n✅ precompute v22.0 complete.")
    logger.info(
        "Next: validate_signal_ic.py → confirm IC > 0.035 per signal → "
        "rm models/weights/gat_router.pt → train_gat_router.py"
    )


if __name__ == "__main__":
    asyncio.run(main())