"""
FORTRESS v5 — scripts/precompute_alpha_signals.py  [v18 — ORTHOGONAL EXPANSION]

v18 INJECTS DEEP RESEARCH SIGNALS:
  1. Expected Idiosyncratic Skewness (EIS): Regresses equities against SPY/QQQ/IWM 
     to isolate idiosyncratic residuals. Calculates 63d trailing skewness.
  2. Bulk Volume VPIN (BV-VPIN): Uses the Normal CDF to probabilistically partition 
     daily volume into institutional accumulation/distribution order imbalances.
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
from scipy.stats import spearmanr, norm, skew
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger("Ouroboros.AlphaPrecompute")

_BASE_DIR    = Path(".")
_CACHE_DIR   = _BASE_DIR / "research" / "outputs" / "cache"
_UNIVERSE_FILE = _BASE_DIR / "config" / "universe.yaml"

_PRICES_PATH    = _CACHE_DIR / "prices_wide.parquet"
_RETURNS_PATH   = _CACHE_DIR / "returns_wide.parquet"
_VOLUMES_PATH   = _CACHE_DIR / "volumes_wide.parquet"
_REGIME_OUT     = _CACHE_DIR / "regime_posteriors.parquet"
_SIGNALS_OUT    = _CACHE_DIR / "alpha_signals.parquet"
_ALPHA_OUT      = _CACHE_DIR / "alpha_signals_blended.parquet"
_GEX_ALPHA_OUT  = _CACHE_DIR / "gex_alpha.parquet"
_PC_FLOW_CACHE  = _CACHE_DIR / "options_flow_pc.parquet"

_CACHE_DIR.mkdir(parents=True, exist_ok=True)

with open(_UNIVERSE_FILE, "r") as f:
    _univ_config = yaml.safe_load(f)

TICKERS: List[str] = [asset["ticker"] for asset in _univ_config["assets"]]
N_ASSETS = len(TICKERS)

_EQUITY_CATEGORIES = {"equity_broad", "equity_sector", "equity_single", "equity_intl"}
_EQUITY_SET: frozenset[str] = frozenset([a["ticker"] for a in _univ_config["assets"] if a.get("category") in _EQUITY_CATEGORIES])

_HEDGE_ASSETS:   frozenset[str] = frozenset({"BIL", "SHV", "VIXY"})
_EQUITY_TICKERS_FOR_MOM_GATE = _EQUITY_SET 

# v18: 5 signals in the blend stack
SIGNAL_NAMES: List[str] = ["mom", "low_vol", "conc_lead", "eis", "bv_vpin"]
N_SIGNALS = len(SIGNAL_NAMES)

GEX_BETA: Dict[str, float] = {"SPY": 1.0, "QQQ": 0.95, "IWM": 0.40, "XLK": 0.85, "XLC": 0.75, "XLY": 0.65}
_GEX_INTERACTION_GAMMA: float = 0.50

async def stage0_vol_regime(start: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    from signals.vol_regime import MultiAssetVolRegime
    logger.info("Stage 0: Multi-asset vol regime construction...")
    engine = MultiAssetVolRegime()
    await engine.load_history(start=start)
    regime_df, meta_df = engine.get_tensor_series(tickers=TICKERS)
    regime_df.to_parquet(_REGIME_OUT)
    return regime_df, meta_df

async def stage1a_dealer_gamma(start: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    logger.info("Stage 1a: Institutional GEX/DIX flow...")
    if _PC_FLOW_CACHE.exists():
        try:
            df = pd.read_parquet(_PC_FLOW_CACHE).reindex(dates).ffill().fillna(0.0).reindex(columns=TICKERS).fillna(0.0)
            for ticker in df.columns: df[ticker] *= GEX_BETA.get(ticker, 1.0)
            for ticker in TICKERS:
                if ticker not in _EQUITY_SET and ticker in df.columns: df[ticker] = 0.0
            return df
        except Exception as e:
            pass
    return pd.DataFrame(0.0, index=dates, columns=TICKERS)

def stage1b_momentum(returns_df: pd.DataFrame, regime_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    logger.info("Stage 1b: Cross-sectional momentum...")
    active_cols = [t for t in TICKERS if t not in _HEDGE_ASSETS]
    r = returns_df.reindex(columns=TICKERS).fillna(0.0)
    raw_mom = r[active_cols].rolling(252, min_periods=126).sum() - r[active_cols].rolling(21, min_periods=10).sum()
    result = pd.DataFrame(0.0, index=returns_df.index, columns=TICKERS)
    result[active_cols] = np.tanh((raw_mom.rank(axis=1, pct=True) - 0.5) * 2.0)
    breadth_score = (r[active_cols].std(axis=1).clip(lower=1e-6) / r[active_cols].std(axis=1).clip(lower=1e-6).rolling(252, min_periods=63).mean().clip(lower=1e-6)).clip(0.2, 2.0)
    result[active_cols] = result[active_cols].multiply(breadth_score, axis=0)
    return result

def stage4_low_vol(returns_df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Stage 4: Vol-rank proxy...")
    rv_63 = (returns_df.reindex(columns=TICKERS).rolling(63, min_periods=20).std() * np.sqrt(252)).ffill()
    active_cols = [t for t in TICKERS if t not in _HEDGE_ASSETS]
    result = pd.DataFrame(0.0, index=returns_df.index, columns=TICKERS)
    result[active_cols] = np.tanh((rv_63[active_cols].rank(axis=1, pct=True) - 0.5) * 1.0)
    return result.fillna(0.0)

def stage_conc_leadership(returns_df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Stage CONC: Concentration leadership signal...")
    equity_cols = sorted(_EQUITY_SET & set(TICKERS) & set(returns_df.columns))
    active_cols = [t for t in TICKERS if t not in _HEDGE_ASSETS]
    r = returns_df.reindex(columns=TICKERS).fillna(0.0)
    ew_ret_63 = r[equity_cols].mean(axis=1).rolling(63, min_periods=21).sum()
    conc_spread = r["QQQ"].rolling(63, min_periods=21).sum() - ew_ret_63
    conc_zscore = ((conc_spread - conc_spread.rolling(252, min_periods=63).mean()) / conc_spread.rolling(252, min_periods=63).std().clip(lower=1e-4)).clip(-3.0, 3.0).fillna(0.0)
    result = pd.DataFrame(0.0, index=returns_df.index, columns=TICKERS)
    result[active_cols] = np.tanh((r[active_cols].rolling(63, min_periods=21).sum().rank(axis=1, pct=True) - 0.5).multiply(conc_zscore, axis=0) * 1.5)
    for t in TICKERS:
        if t not in _EQUITY_SET: result[t] = 0.0
    return result.fillna(0.0)

def stage_eis(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Expected Idiosyncratic Skewness (EIS) Signal."""
    logger.info("Stage EIS: Calculating Idiosyncratic Skewness (Lottery Effect)...")
    eq_cols = list(_EQUITY_SET & set(TICKERS))
    
    # Simple market proxy using broad indices
    X = returns_df[['SPY', 'QQQ', 'IWM']].fillna(0.0).values
    eis_signal = pd.DataFrame(0.0, index=returns_df.index, columns=TICKERS)
    
    for i, ticker in enumerate(eq_cols):
        y = returns_df[ticker].fillna(0.0).values
        skew_arr = np.zeros(len(y))
        for t_idx in range(63, len(y)):
            X_w = X[t_idx-63:t_idx]
            y_w = y[t_idx-63:t_idx]
            try:
                beta = np.linalg.pinv(X_w.T @ X_w) @ X_w.T @ y_w
                eps = y_w - X_w @ beta
                if np.std(eps) > 1e-6:
                    skew_arr[t_idx] = skew(eps)
            except:
                pass
        
        skew_s = pd.Series(skew_arr, index=returns_df.index)
        med = skew_s.rolling(252, min_periods=63).median()
        mad = (skew_s - med).abs().rolling(252, min_periods=63).median() * 1.4826
        # Invert: High positive skew = overvalued lottery ticket = bearish signal
        z_skew = -(skew_s - med) / mad.clip(lower=1e-4)
        eis_signal[ticker] = np.tanh(z_skew.fillna(0.0) * 0.5)
        
    return eis_signal

def stage_bv_vpin(prices_df: pd.DataFrame, returns_df: pd.DataFrame) -> pd.DataFrame:
    """Bulk Volume VPIN Signal."""
    logger.info("Stage BV-VPIN: Calculating Volume Order Toxicity...")
    if not _VOLUMES_PATH.exists():
        logger.info("  Downloading volume data for BV-VPIN...")
        raw = yf.download(TICKERS, start=prices_df.index[0], auto_adjust=True, progress=False)
        volumes = raw["Volume"] if "Volume" in raw.columns else raw.xs("Volume", axis=1, level=0)
        volumes = volumes.reindex(columns=TICKERS).ffill().fillna(0.0)
        volumes.to_parquet(_VOLUMES_PATH)
    else:
        volumes = pd.read_parquet(_VOLUMES_PATH).reindex(columns=TICKERS, index=prices_df.index).fillna(0.0)
        
    vpin_signal = pd.DataFrame(0.0, index=returns_df.index, columns=TICKERS)
    eq_cols = list(_EQUITY_SET & set(TICKERS))
    
    for ticker in eq_cols:
        ret = returns_df[ticker].fillna(0.0)
        vol = volumes[ticker].fillna(0.0)
        
        # EMA bounds check to prevent data corruption from unadjusted splits
        vol_mean = vol.ewm(span=63).mean()
        vol_std = vol.ewm(span=63).std().clip(lower=1e-4)
        vol = vol.clip(upper=vol_mean + 5 * vol_std)
        
        sigma = ret.rolling(21, min_periods=5).std().clip(lower=1e-4)
        
        # Probabilistic Assignment via Normal CDF
        buy_pct = norm.cdf(ret / sigma)
        buy_vol = vol * buy_pct
        sell_vol = vol * (1 - buy_pct)
        
        oi = (buy_vol - sell_vol).abs()
        roll_oi = oi.rolling(21, min_periods=5).sum()
        roll_vol = vol.rolling(21, min_periods=5).sum().clip(lower=1e-4)
        vpin = roll_oi / roll_vol
        
        direction = np.sign((buy_vol - sell_vol).rolling(21).sum())
        raw_sig = direction * vpin
        
        sig_mean = raw_sig.rolling(252, min_periods=63).mean()
        sig_std = raw_sig.rolling(252, min_periods=63).std().clip(lower=1e-4)
        z_sig = (raw_sig - sig_mean) / sig_std
        
        vpin_signal[ticker] = np.tanh(z_sig.fillna(0.0) * 0.5)
        
    return vpin_signal

def stage5_blend_signals(
    signal_dfs:  Dict[str, pd.DataFrame],
    regime_df:   pd.DataFrame,
    returns_df:  pd.DataFrame,
    dates:       pd.DatetimeIndex,
    gex_flow_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Stage 5: IC-Weighted Signal Routing (GATv2 Neural Router)...")
    
    from models.alpha.gat_signal_router import SignalRouterGAT, build_economic_graph, build_node_features
    import torch
    
    S = N_SIGNALS
    N = N_ASSETS
    T = len(dates)

    signal_stack = np.zeros((T, N, S), dtype=np.float32)
    for s_idx, name in enumerate(SIGNAL_NAMES):
        aligned = signal_dfs[name].reindex(dates).ffill().fillna(0.0).reindex(columns=TICKERS).fillna(0.0)
        signal_stack[:, :, s_idx] = aligned.values

    logger.info("  Computing 63d rolling signal ICs...")
    ic_history = np.zeros((T, N, S), dtype=np.float32)
    ret_arr = returns_df.reindex(dates).reindex(columns=TICKERS).fillna(0.0).values
    
    for s_idx in range(S):
        sig_arr = signal_stack[:, :, s_idx]
        for t_idx in range(63, T):
            for n_idx in range(N):
                sig_slice = sig_arr[t_idx-63:t_idx, n_idx]
                ret_slice = ret_arr[t_idx-63:t_idx, n_idx]
                if np.std(sig_slice) > 1e-6 and np.std(ret_slice) > 1e-8:
                    ic, _ = spearmanr(sig_slice, ret_slice)
                    ic_history[t_idx, n_idx, s_idx] = ic if np.isfinite(ic) else 0.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    router = SignalRouterGAT(n_signals=S).to(device)
    
    weights_path = Path("models/weights/gat_router.pt")
    if weights_path.exists():
        router.load_state_dict(torch.load(weights_path, map_location=device))
        logger.info("  ✅ Loaded trained GATv2 weights.")
    else:
        logger.warning("  ⚠️ GATv2 weights not found. Falling back to non-neural router for precomputation.")
        from models.alpha.gat_signal_router import FallbackSignalRouter
        fallback = FallbackSignalRouter(temperature=0.1)
        alpha_raw = np.zeros((T, N), dtype=np.float32)
        for t_idx in range(T):
            alpha_raw[t_idx] = fallback.route_signals(ic_history[t_idx], signal_stack[t_idx])
            
        # TIER 1: GEX Multiplicative Modifier (Equity Only)
        gex_aligned = gex_flow_df.reindex(dates).ffill().fillna(0.0).reindex(columns=TICKERS).fillna(0.0)
        gex_arr = gex_aligned.values.astype(np.float32)

        for i, ticker in enumerate(TICKERS):
            if ticker in _EQUITY_SET:
                alpha_raw[:, i] *= (1.0 + _GEX_INTERACTION_GAMMA * gex_arr[:, i])

        return pd.DataFrame(np.tanh(alpha_raw).astype(np.float32), index=dates, columns=TICKERS), gex_aligned

    router.eval()
    alpha_raw = np.zeros((T, N), dtype=np.float32)
    edge_index, edge_attr = build_economic_graph(returns_df)
    edge_index, edge_attr = edge_index.to(device), edge_attr.to(device)

    vixy_col = returns_df["VIXY"] if "VIXY" in TICKERS else returns_df.mean(axis=1)
    tlt_col = returns_df["TLT"] if "TLT" in TICKERS else returns_df.mean(axis=1)
    vol_betas = returns_df.rolling(63).corr(vixy_col).fillna(0.0).values
    rate_betas = returns_df.rolling(63).corr(tlt_col).fillna(0.0).values
    liquidity_z = np.zeros((T, N), dtype=np.float32)

    for t_idx in range(T):
        z_mu_str = regime_df.iloc[t_idx]["z_mu"]
        if isinstance(z_mu_str, str):
            import ast; z_mu = np.array(ast.literal_eval(z_mu_str), dtype=np.float32)
        else:
            z_mu = np.zeros(16, dtype=np.float32)

        x_np = build_node_features(
            signal_ic_history=ic_history[t_idx],
            vol_betas=vol_betas[t_idx],
            rate_betas=rate_betas[t_idx],
            liquidity_z=liquidity_z[t_idx]
        )
        
        g_tensor = torch.from_numpy(z_mu[:16]).to(device)
        alpha_raw[t_idx] = router.infer_alpha(
            x=x_np, g=g_tensor, edge_index=edge_index, edge_attr=edge_attr, 
            signal_matrix=torch.from_numpy(signal_stack[t_idx]), device=str(device)
        )

    gex_aligned = gex_flow_df.reindex(dates).ffill().fillna(0.0).reindex(columns=TICKERS).fillna(0.0)
    gex_arr = gex_aligned.values.astype(np.float32)

    for i, ticker in enumerate(TICKERS):
        if ticker in _EQUITY_SET:
            alpha_raw[:, i] *= (1.0 + _GEX_INTERACTION_GAMMA * gex_arr[:, i])

    alpha_final = np.tanh(alpha_raw).astype(np.float32)
    result_df = pd.DataFrame(alpha_final, index=dates, columns=TICKERS)
    
    eq_cols_set = list(_EQUITY_SET & set(TICKERS))
    neq_cols_set = [t for t in TICKERS if t not in _EQUITY_SET]
    eq_mean  = result_df[eq_cols_set].abs().mean().mean() if eq_cols_set else 0.0
    neq_mean = result_df[neq_cols_set].abs().mean().mean() if neq_cols_set else 0.0
    
    logger.info(
        f"  Blended alpha (GATv2 Router + GEX modifier): {len(result_df)}d x {N} assets | "
        f"|alpha| equity={eq_mean:.3f} | non-equity={neq_mean:.3f}"
    )

    return result_df, gex_aligned

async def main() -> None:
    logger.info("====== Ouroboros Alpha Precompute v18 (ORTHOGONAL EXPANSION) ======")

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
    
    eis_df = stage_eis(returns_df)
    eis_df.index = pd.to_datetime(eis_df.index)
    
    vpin_df = stage_bv_vpin(prices_df, returns_df)
    vpin_df.index = pd.to_datetime(vpin_df.index)

    signal_dfs: Dict[str, pd.DataFrame] = {
        "mom":       mom_df,
        "low_vol":   lowvol_df,
        "conc_lead": conc_df,
        "eis":       eis_df,
        "bv_vpin":   vpin_df
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

    alpha_df = alpha_df.fillna(0.0)
    alpha_df.to_parquet(_ALPHA_OUT)
    gex_alpha_df.to_parquet(_GEX_ALPHA_OUT)

    logger.info("Per-signal statistics:")
    for sig_name, df in all_signals.items():
        df_a = df.reindex(dates).ffill().fillna(0.0).reindex(columns=TICKERS).fillna(0.0)
        nonzero = (df_a.abs() > 0.01).any(axis=1).mean() * 100
        logger.info(f"  {sig_name:10s} | mean|α|={df_a.abs().mean().mean():.3f} | active={nonzero:.1f}%")

if __name__ == "__main__":
    asyncio.run(main())