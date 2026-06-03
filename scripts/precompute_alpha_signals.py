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

SIGNAL_NAMES: List[str] = [
    "low_vol", "ramom_ts", "odpv_vwap", "clv_flow", "dtfe_trend",
    "res_mom", "cw_spread", "ami_impact", "real_skew", "vol_decouple"
]
N_SIGNALS: int = len(SIGNAL_NAMES)

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

# INSERT THIS CODE BLOCk IMMEDIATELY BEFORE stage7_blend:

def stage6b_res_mom(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Isolates factor-adjusted idiosyncratic momentum via rolling cross-sectional beta removal."""
    logger.info("Stage 6b: Residual Momentum (factor-insulated alpha)...")
    r_idx = returns_df.reindex(columns=TICKERS).fillna(0.0)
    mkt = r_idx.mean(axis=1) 
    
    mkt_var = mkt.rolling(63, min_periods=20).var().clip(lower=1e-6)
    res_df = pd.DataFrame(0.0, index=returns_df.index, columns=TICKERS, dtype=np.float32)
    
    for col in TICKERS:
        cov = r_idx[col].rolling(63, min_periods=20).cov(mkt)
        beta = cov / mkt_var
        res_df[col] = r_idx[col] - beta * mkt
        
    # Apply a causal rolling Z-score to bring numbers into standard normal scale before tanh
    res_smooth = res_df.ewm(span=21, adjust=False).mean()
    res_z = (res_smooth - res_smooth.rolling(63, min_periods=20).mean()) / res_smooth.rolling(63, min_periods=20).std().clip(lower=1e-6)
    
    return np.tanh(res_z.fillna(0.0)).astype(np.float32)

def stage6c_cw_spread(ohlcv: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Corwin-Schultz High-Low structural bid-ask spread estimator with cross-sectional ranking."""
    logger.info("Stage 6c: Corwin-Schultz implicit spread estimator...")
    h, l = np.log(ohlcv["high"]), np.log(ohlcv["low"])
    
    kh1 = (h / l) ** 2
    kh1_2d = kh1 + kh1.shift(1)
    
    h_2d = np.maximum(h, h.shift(1))
    l_2d = np.minimum(l, l.shift(1))
    k_2d = (h_2d / l_2d) ** 2
    
    const_inv = 3.0 - 2.0 * np.sqrt(2.0)
    alpha = (np.sqrt(2.0 * k_2d) - np.sqrt(k_2d)) / const_inv - np.sqrt(kh1_2d / const_inv)
    
    spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    spread_smoothed = spread.rolling(21, min_periods=5).mean().fillna(0.0)
    
    # Map to relative cross-sectional percentiles to completely destroy the saturation problem
    spread_rank = spread_smoothed.rank(axis=1, pct=True)
    return np.tanh(-(spread_rank - 0.5) * 2.0).astype(np.float32)

def stage6d_ami_impact(ohlcv: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Asymmetric daily price impact engine measuring continuous volume-liquidity tension."""
    logger.info("Stage 6d: Asymmetric Price Impact vector...")
    c, v = ohlcv["close"], ohlcv["volume"]
    ret = np.log(c / c.shift(1)).fillna(0.0)
    
    # Compute price impact relative to asset's own historical dollar volume profile
    dollar_vol = (v * c).clip(lower=1.0)
    raw_impact = ret.abs() / dollar_vol
    signed_impact = np.sign(ret) * raw_impact
    
    # Scale via a long-window rolling Z-score to normalize single-name volume variance
    roll_mean = signed_impact.rolling(63, min_periods=20).mean()
    roll_std = signed_impact.rolling(63, min_periods=20).std().clip(lower=1e-12)
    impact_z = (signed_impact - roll_mean) / roll_std
    
    smooth_impact = impact_z.rolling(21, min_periods=5).mean().fillna(0.0)
    return np.tanh(smooth_impact).astype(np.float32)

def stage6e_real_skew(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Exploits downstream alpha lottery decay via rolling realized third standardized moments."""
    logger.info("Stage 6e: Realized Skewness asymmetry mapping...")
    r_idx = returns_df.reindex(columns=TICKERS)
    skew = r_idx.rolling(63, min_periods=20).skew().fillna(0.0)
    return -np.tanh(skew).astype(np.float32)

def stage6f_vol_decouple(ohlcv: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Measures variance range to institutional volume decoupling (divergence signals)."""
    logger.info("Stage 6f: Volume-Range Decoupling profile...")
    h, l, c, v = ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
    
    parkinson = ((np.log(h / l)) ** 2) / (4.0 * np.log(2.0))
    parkinson_smooth = parkinson.rolling(63, min_periods=20).mean()
    p_mean = parkinson_smooth.rolling(63, min_periods=20).mean()
    p_std = parkinson_smooth.rolling(63, min_periods=20).std().clip(lower=1e-6)
    z_p = (parkinson_smooth - p_mean) / p_std
    
    v_smooth = v.rolling(63, min_periods=20).mean()
    v_mean = v_smooth.rolling(63, min_periods=20).mean()
    v_std = v_smooth.rolling(63, min_periods=20).std().clip(lower=1e-6)
    z_v = (v_smooth - v_mean) / v_std
    
    decouple = (z_p - z_v).fillna(0.0)
    return -np.tanh(decouple).astype(np.float32)

# ─────────────────────────────────────────────────────────────────────────────
    logger.info("Stage 7: Static-weight blend (GATv2 bypassed — retrain required)...")
# ─────────────────────────────────────────────────────────────────────────────



def stage7_blend(
    signal_dfs:  Dict[str, pd.DataFrame],
    regime_df:   pd.DataFrame,
    dates:       pd.DatetimeIndex,
    returns:     pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    use_router = os.getenv("USE_GAT_ROUTER", "0") == "1"
    
    T = len(dates)
    N = N_ASSETS
    
    # This will now safely inherit N_SIGNALS = 10 from the global definition block
    stack = np.zeros((T, N, N_SIGNALS), dtype=np.float32)
    # ... resto de la función ...
    for s_idx, sig_name in enumerate(SIGNAL_NAMES):
        if sig_name in signal_dfs:
            stack[:, :, s_idx] = signal_dfs[sig_name].reindex(dates).ffill().fillna(0.0).reindex(columns=TICKERS).fillna(0.0).values

    if use_router:
        logger.info("Stage 7: Dynamic-weight blend via active GATv2 Router...")
        try:
            # Drop N_SIGNALS from the import list completely
            from models.alpha.gat_signal_router import SignalRouterGAT
            import torch

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            router = SignalRouterGAT(n_signals=N_SIGNALS).to(device)

            raw_ckpt   = torch.load(
                "models/weights/gat_router.pt",
                map_location=device,
                weights_only=False,
            )
            # Tolerate both bare state_dict and {"model_state_dict": sd} checkpoint formats
            state_dict = (
                raw_ckpt.get("model_state_dict", raw_ckpt)
                if isinstance(raw_ckpt, dict)
                else raw_ckpt
            )
            router.load_state_dict(state_dict)
            router.eval()

            regime_t = torch.tensor(
                regime_df.reindex(dates).ffill().fillna(0.0).values,
                dtype=torch.float32,
            )  # (T, D) — stays on CPU; temporal_infer pulls to device internally
            rets_t  = torch.tensor(
                returns.reindex(dates).ffill().fillna(0.0).values,
                dtype=torch.float32,
            )  # (T, N)
            stack_t = torch.tensor(stack, dtype=torch.float32)  # (T, N, S)

            # FIXED: temporal_infer accepts (T,N,S)/(T,N)/(T,D)/List[str]
            # Previous call: router.infer_alpha(regime_t, rets_t, dates_t, TICKERS, stack_t)
            #   → TICKERS (list) landed on edge_attr positional slot → .to(device) crash
            alpha_raw = router.temporal_infer(
                signal_stack = stack_t,
                returns      = rets_t,
                regime_arr   = regime_t,
                tickers      = TICKERS[:N_ASSETS],
                device       = str(device),
            )  # (T, N) np.ndarray — no .cpu().numpy() required

        except Exception as exc:
            logger.warning(f"  ⚠️ GATv2 Router failed ({exc}). Falling back to robust ensemble.")
            use_router = False

    if not use_router:
        logger.info("Stage 7: Robust Cross-Sectional Ensemble Blend (Regime-Aware)...")
        
        # Normalización Cross-Sectional de las alphas individuales para eliminar sesgos de escala
        signal_z = {}
        for name, df in signal_dfs.items():
            x = df.reindex(dates).ffill().fillna(0.0).reindex(columns=TICKERS).fillna(0.0)
            x_mean = x.mean(axis=1)
            x_std  = x.std(axis=1).replace(0.0, np.nan)
            # Z-score cross-sectional + truncamiento rígido a 3 sigmas
            signal_z[name] = x.sub(x_mean, axis=0).div(x_std, axis=0).fillna(0.0).clip(-3.0, 3.0)

        # Matriz de pesos adaptativa según el nivel de urgencia macro del régimen
        alpha_raw = np.zeros((T, N), dtype=np.float32)
        
        # Petición de urgencia para modular la exposición defensiva
        urgency_arr = regime_df["ltc_urgency"].reindex(dates).ffill().fillna(0.3).values
        
        for t_idx in range(T):
            u = urgency_arr[t_idx]
            if u > 0.5:  # High Market Stress Regime (Defensive Realignment)
                w = {
                    "low_vol": 0.25, "ramom_ts": 0.05, "odpv_vwap": 0.10, "clv_flow": 0.05, "dtfe_trend": 0.05,
                    "res_mom": 0.10, "cw_spread": 0.15, "ami_impact": 0.10, "real_skew": 0.10, "vol_decouple": 0.05
                }
            else:        # Structural Expansion Regime (Alpha Capture)
                w = {
                    "low_vol": 0.10, "ramom_ts": 0.15, "odpv_vwap": 0.10, "clv_flow": 0.10, "dtfe_trend": 0.05,
                    "res_mom": 0.15, "cw_spread": 0.05, "ami_impact": 0.10, "real_skew": 0.05, "vol_decouple": 0.15
                }
            
            for name in SIGNAL_NAMES:
                alpha_raw[t_idx] += signal_z[name].iloc[t_idx].values * w[name]

    # ── Regime gate: scale alpha by vol-regime urgency proxy ─────────────────
    if "ltc_urgency" in regime_df.columns:
        urgency = regime_df["ltc_urgency"].reindex(dates).ffill().fillna(0.3).clip(0.0, 1.0).values[:, None]
        gate = 0.30 + 0.70 * (1.0 - urgency)
        alpha_gated = alpha_raw * gate
    else:
        alpha_gated = alpha_raw

    # ── Turnover Smoothing & tanh re-normalisation ───────────────────────────
    raw_alpha_df = pd.DataFrame(alpha_gated * 2.0, index=dates, columns=TICKERS)
    # Subir el span de 3 a 8 barras calma drásticamente las oscilaciones diarias de asignación
    smoothed_alpha = raw_alpha_df.ewm(span=8, min_periods=1).mean()
    alpha_final = np.tanh(smoothed_alpha.values).astype(np.float32)
    alpha_df = pd.DataFrame(alpha_final, index=dates, columns=TICKERS)

    return alpha_df, alpha_df.copy()

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
    # Legacy Core Blocks
    lowvol_df   = stage2_low_vol(returns_df=returns)
    ramom_df    = stage3_ramom_ts(ohlcv=ohlcv)
    odpv_df     = stage4_odpv_vwap(ohlcv=ohlcv)
    clv_df      = stage5_clv_flow(ohlcv=ohlcv)
    dtfe_df     = stage6_dtfe_trend(ohlcv=ohlcv)
    
    # Advanced Blocks
    resmom_df   = stage6b_res_mom(returns_df=returns)
    cwspread_df = stage6c_cw_spread(ohlcv=ohlcv)
    ami_df      = stage6d_ami_impact(ohlcv=ohlcv)
    skew_df     = stage6e_real_skew(returns_df=returns)
    decouple_df = stage6f_vol_decouple(ohlcv=ohlcv)

    # FIXED CONIC ALIGNMENT: Flipping signs based on empirical IC horizons
    # Conic Alignment Matched Tensors Mapping
    signal_dfs: Dict[str, pd.DataFrame] = {
        "low_vol":      -lowvol_df,   # Keep legacy mapping: Massive 63d macro alpha
        "ramom_ts":     -ramom_df,    # Elite short-term horizon momentum factor
        "clv_flow":     -clv_df,      # Capitalizes on intraday accumulation
        "dtfe_trend":   dtfe_df,      # Preserves fractal efficiency metrics
        
        # FIXED: Restored variable pointer to odpv_df to fix the duplicate tracking bug
        "odpv_vwap":    odpv_df,      
        
        "real_skew":    skew_df,      # Elite 63d tail risk factor
        "vol_decouple": decouple_df,  # Long-term variance-volume expansion anchor
        "res_mom":      -resmom_df,   # Flipped sign successfully captures idiosyncratic mean-reversion
        "ami_impact":   -ami_df,      # Flipped sign successfully tracks structural liquidity provision
        
        # FIXED: Removed negation to align the estimator with a positive alpha capture
        "cw_spread":    cwspread_df   
    }

    # ── Stage 7: Blend ────────────────────────────────────────────────────────
    alpha_df, gex_alpha_df = stage7_blend(
        signal_dfs = signal_dfs,
        regime_df  = regime_df,
        dates      = dates,
        returns    = returns,   # <── ENLACE DE MATRIZ CAUSAL
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