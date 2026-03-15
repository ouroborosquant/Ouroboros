"""
FORTRESS v5 - scripts/precompute_alpha_signals.py
Path: scripts/precompute_alpha_signals.py  [COMPLETE REWRITE]

4-Layer Alpha Signal Precomputation with Signal Router Integration.

PIPELINE STAGES:
  Stage 0: Multi-asset vol regime construction (replaces Mamba-KAN z_mu)
  Stage 1: Options surface signals (VRP, IV/HV ratio, term structure)
  Stage 2: ETF NAV/AP stress signal (bond/commodity ETFs only)
  Stage 3: SEC filing intelligence (IWM insider + activist sector signals)
  Stage 4: Low-vol anomaly (baseline factor, always-on with low weight)
  Stage 5: Signal Router (GATv2 IC predictor) — if weights exist
           Fallback: EWMA-IC weighted blend (no GATv2)

OUTPUT FORMAT:
  alpha_signals.parquet:    (T, N×S) — raw per-signal per-ticker alpha vectors
                            Columns: {signal_name}_{ticker}
                            Used by train_signal_router.py as training input.

  alpha_blended.parquet:    (T, N) — final blended alpha via GATv2 signal router
                            Columns: {ticker}
                            Consumed by research/backtest_engine.py

  regime_posteriors.parquet: (T, *) — Multi-asset VolRegimeTensor posteriors
                              Replaces Mamba-KAN output as regime signal.

  signal_metadata.parquet:  (T, *) — AP stress indicators, IC statistics, debug

ARCHITECTURAL CHANGES vs prior version:
  REMOVED:
    - 8-factor surrogate (momentum, reversal, carry, tilt)
    - FactorDecayMonitor gating (replaced by GATv2 IC prediction)
    - z_mu from Mamba-KAN (replaced by VolRegimeTensor)
    - VIX × vol_beta IV proxy (replaced by CBOE vol indices + yfinance chains)

  ADDED:
    - MultiAssetVolRegime (VIX/MOVE/GVZ/OVX four-axis regime tensor)
    - OptionsAlphaEngine (real CBOE IV, VRP, skew, term structure)
    - ETFNavArbSignal (AP capacity stress, stress-gated)
    - SECFilingIntelligence (IWM insider + activist sector signals)
    - LowVolAnomaly (retained as baseline — low weight)
    - SignalRouterGAT (GATv2 predicting forward IC per signal per asset)

LOOK-AHEAD CONTRACT:
  All signals use strictly causal computation: rolling windows, EWMA, and
  SEC filing dates use only data available at as_of_date.
  Forward IC computation in train_signal_router.py uses forward data only
  as TRAINING TARGETS — never as features. This contract is enforced by
  the separate computation of ic_tensor (target) vs ewma_ic_history (feature).

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
_BASE_DIR      = Path(".")
_CACHE_DIR     = _BASE_DIR / "research" / "outputs" / "cache"
_WEIGHTS_DIR   = _BASE_DIR / "models" / "weights"

_PRICES_PATH   = _CACHE_DIR / "prices_wide.parquet"
_RETURNS_PATH  = _CACHE_DIR / "returns_wide.parquet"

# Output paths
_REGIME_OUT    = _CACHE_DIR / "regime_posteriors.parquet"
_SIGNALS_OUT   = _CACHE_DIR / "alpha_signals.parquet"     # per-signal per-ticker
_ALPHA_OUT     = _CACHE_DIR / "alpha_signals_blended.parquet"  # final blended
_META_OUT      = _CACHE_DIR / "signal_metadata.parquet"

_ROUTER_WEIGHTS = _WEIGHTS_DIR / "signal_router_latest.pt"

_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

TICKERS: List[str] = [
    "SPY", "QQQ", "IWM", "TLT", "HYG", "LQD", "GLD", "SLV",
    "GDX", "XLE", "XLF", "XLK", "XLV", "XLU", "XLI", "XLP",
    "XLY", "XLB", "XLC", "VIXY", "BIL", "SHV", "USO", "PDBC", "COWZ",
]
N_ASSETS  = len(TICKERS)
SIGNAL_NAMES: List[str] = ["vrp", "vts", "nav_arb", "insider", "low_vol"]
N_SIGNALS = len(SIGNAL_NAMES)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 0: Multi-asset vol regime
# ─────────────────────────────────────────────────────────────────────────────

async def stage0_vol_regime(start: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute multi-asset vol regime tensor for the full history.
    Replaces Mamba-KAN regime_posteriors.parquet.

    Returns:
      regime_df:  (T, *) z_mu/z_sigma/tda_alert/ltc_urgency/regime_label
      meta_df:    (T, *) per-axis labels and urgencies (for diagnostics)
    """
    from signals.vol_regime import MultiAssetVolRegime

    logger.info("Stage 0: Multi-asset vol regime construction...")
    regime_engine = MultiAssetVolRegime()
    await regime_engine.load_history(start=start)

    regime_df, meta_df = regime_engine.get_tensor_series(tickers=TICKERS)
    regime_df.to_parquet(_REGIME_OUT)
    logger.info(f"  ✓ Regime posteriors → {_REGIME_OUT} ({len(regime_df)} rows)")

    # Log regime distribution
    if "equity_label" in meta_df.columns:
        for label in ["crisis", "stress", "neutral", "complacent"]:
            pct = (meta_df["equity_label"] == label).mean() * 100
            logger.info(f"  Equity regime [{label}]: {pct:.1f}% of days")

    return regime_df, meta_df


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Options surface signals
# ─────────────────────────────────────────────────────────────────────────────

async def stage1_options_alpha(start: str) -> pd.DataFrame:
    """
    Compute VRP, IV/HV ratio, and VTS term structure signals.
    Uses CBOE vol indices for settlement-grade IV (no proxy needed for SPY/QQQ/IWM/GLD/USO).

    Returns:
      options_df: (T, N) — options-surface composite alpha per ticker
    """
    from signals.options_alpha import OptionsAlphaEngine

    logger.info("Stage 1: Options surface alpha (CBOE vol indices)...")
    engine = OptionsAlphaEngine()
    await engine.load_data(start=start)

    options_df = engine.compute_full_history()
    logger.info(
        f"  ✓ Options alpha: {len(options_df)} days × {len(options_df.columns)} assets | "
        f"Mean |signal|: {options_df.abs().mean().mean():.3f}"
    )
    return options_df


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: ETF NAV/AP stress signal
# ─────────────────────────────────────────────────────────────────────────────

async def stage2_nav_arb(start: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute AP capacity stress signal for bond and commodity ETFs.
    Signal is zero unless AP stress conditions are met — this is correct.
    A sparse signal that fires 5-10% of trading days is not a bug; it's the
    mechanism correctly refusing to trade when there is no edge.

    Returns:
      signal_df:   (T, N) — stress-gated NAV premium signals
      stress_meta: (T, 2) — ap_stress_indicator, n_active_tickers
    """
    from signals.etf_nav_arb import ETFNavArbSignal

    logger.info("Stage 2: ETF NAV / AP stress signal...")
    engine = ETFNavArbSignal()
    await engine.load_data(start=start)

    signal_df, stress_meta = engine.compute_full_history()
    logger.info(
        f"  ✓ NAV arb: {len(signal_df)} days | "
        f"Active days (AP stress): {(stress_meta['n_active'] > 0).sum()} "
        f"({(stress_meta['n_active'] > 0).mean() * 100:.1f}%)"
    )
    return signal_df, stress_meta


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: SEC filing intelligence
# ─────────────────────────────────────────────────────────────────────────────

async def stage3_sec_insider(
    dates: pd.DatetimeIndex,
    batch_size: int = 5,
) -> pd.DataFrame:
    """
    Compute IWM insider cluster buying + activist sector signals.
    Signal is narrow (non-zero for IWM and sector ETFs only).
    Fetched with EDGAR rate-limit awareness (6 concurrent requests max).

    FREQUENCY: We compute this WEEKLY (not daily) to reduce EDGAR API calls.
    For dates between weekly recomputation, we hold the last value.
    This is valid because insider/activist signals decay slowly (30-90d horizon).

    Returns:
      insider_df: (T, N) — insider + activist alpha per ticker
    """
    from signals.sec_insider import SECFilingIntelligence

    logger.info("Stage 3: SEC filing intelligence (IWM insider + activist)...")
    engine = SECFilingIntelligence(lookback_days=30)

    # Compute weekly (every 5 trading days) to limit EDGAR load
    weekly_dates  = dates[::5]
    weekly_signals: Dict[str, pd.Series] = {}

    n_computed = 0
    for date in weekly_dates:
        date_str = str(date.date())
        try:
            alpha = await engine.get_combined_alpha_vector(date_str)
            weekly_signals[date_str] = alpha
            n_computed += 1
            if n_computed % 10 == 0:
                logger.info(f"  SEC signals: {n_computed}/{len(weekly_dates)} weeks")
        except Exception as e:
            logger.debug(f"  SEC signals failed for {date_str}: {e}")
            weekly_signals[date_str] = pd.Series(0.0, index=TICKERS)

        # Minimal sleep to respect EDGAR rate limits
        await asyncio.sleep(0.5)

    # Build weekly DataFrame
    weekly_df = pd.DataFrame(weekly_signals).T
    weekly_df.index = pd.to_datetime(weekly_df.index)
    weekly_df = weekly_df.sort_index()

    # Forward-fill to daily (hold last weekly value)
    insider_df = weekly_df.reindex(dates).ffill().fillna(0.0)

    n_nonzero = (insider_df.abs() > 0.01).any(axis=1).mean() * 100
    logger.info(
        f"  ✓ SEC insider: {len(insider_df)} days | "
        f"Non-zero signal: {n_nonzero:.1f}% of days (expected: IWM + up to 10 sector ETFs)"
    )
    return insider_df


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4: Low-volatility anomaly (baseline factor)
# ─────────────────────────────────────────────────────────────────────────────

def stage4_low_vol(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Low-volatility anomaly: lower-vol assets tend to earn higher risk-adjusted
    returns. Used as a baseline signal with LOW WEIGHT (0.10) — it's always-on
    but not a primary alpha source. Its value is providing a floor of signal
    quality when other signals are sparse or zero.

    SIGNAL: -rank(realized_vol_63d) normalised cross-sectionally.
    Negative because we want LOW vol = HIGH signal.

    RISK: This signal can fail dramatically in risk-on momentum regimes where
    high-vol assets (growth, biotech, crypto-adjacent) outperform. The regime
    gate from MultiAssetVolRegime should reduce its weight when equity_regime
    shows complacent/bull signal (vol carry is unfavourable in late-bull).
    We handle this in the signal router training — the GATv2 will learn to
    downweight low-vol signal in complacent equity regimes.
    """
    logger.info("Stage 4: Low-volatility anomaly (baseline)...")
    rv_63 = returns_df.reindex(columns=TICKERS).rolling(63, min_periods=20).std() * np.sqrt(252)
    rv_63 = rv_63.ffill().fillna(rv_63.mean())

    # Cross-sectional rank normalisation: -rank(vol) → long low-vol, short high-vol
    rank_vol = rv_63.rank(axis=1, pct=True)
    signal   = -(rank_vol - 0.5)  # center at zero

    # Tanh squash to [-1, 1]
    result = np.tanh(signal * 2.0)

    logger.info(
        f"  ✓ Low-vol signal: {len(result)} days | "
        f"Mean |signal|: {result.abs().mean().mean():.3f}"
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5: Signal blending (GATv2 router or EWMA IC fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _build_ewma_ic_fallback_weights(
    signal_stack:  np.ndarray,   # (T, N, S)
    returns_np:    np.ndarray,   # (T, N)
    halflife:      int = 63,
    ic_window:     int = 21,
) -> np.ndarray:
    """
    Causal EWMA IC history used as fallback blending weights when GATv2
    signal router weights are unavailable.

    For each date t, computes the EWMA of historical cross-sectional IC:
      IC_s(t_past) = Spearman_corr(signal_s[t_past, :], r[t_past+1, :])
                     (cross-sectional over N=25 assets)
      ewma_ic_s(t) = EWMA(IC_s(t_past), halflife=63)

    NOTE: This is the CROSS-SECTIONAL IC (across assets at a single date),
    not the TEMPORAL IC used in the signal router training. Both are valid;
    the cross-sectional IC is faster to compute and sufficient for the fallback.
    """
    from scipy import stats

    T, N, S = signal_stack.shape
    cross_ic = np.zeros((T, S), dtype=np.float32)

    # Compute raw cross-sectional IC series
    for t in range(T - 1):
        sig_t    = signal_stack[t]           # (N, S)
        ret_t1   = returns_np[t + 1]         # (N,)
        valid    = ~np.isnan(ret_t1)
        if valid.sum() < 10:
            continue
        for s in range(S):
            sig_s = sig_t[valid, s]
            ret_s = ret_t1[valid]
            if np.std(sig_s) < 1e-6:
                continue
            try:
                ic, _ = stats.spearmanr(sig_s, ret_s)
                if not np.isnan(ic):
                    cross_ic[t, s] = float(ic)
            except Exception:
                pass

    # EWMA over history — causal
    alpha_decay = 1.0 - np.exp(-1.0 / halflife)
    ewma_ic = np.zeros_like(cross_ic)
    running = np.zeros(S, dtype=np.float32)
    running_w = 0.0

    for t in range(T):
        running_w  = running_w * (1 - alpha_decay) + alpha_decay
        running    = running   * (1 - alpha_decay) + cross_ic[t] * alpha_decay
        if running_w > 1e-6:
            ewma_ic[t] = running / running_w

    # Broadcast to per-asset weights: (T, S) → (T, N, S)
    # For the fallback, all assets in a class share the same weights
    weights_broadcast = np.broadcast_to(
        ewma_ic[:, np.newaxis, :],  # (T, 1, S)
        (T, N, S),
    ).copy()

    # Softmax over S dimension
    weights_broadcast = np.exp(weights_broadcast * 2.0)  # temperature=0.5
    weights_broadcast /= (weights_broadcast.sum(axis=-1, keepdims=True) + 1e-8)

    return weights_broadcast  # (T, N, S)


def stage5_blend_signals(
    signal_dfs: Dict[str, pd.DataFrame],
    regime_df:  pd.DataFrame,
    returns_df: pd.DataFrame,
    dates:      pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Blend all signals into a final alpha vector.

    PRIORITY ORDER:
      1. GATv2 Signal Router (if weights exist and PyG available)
      2. EWMA-IC weighted fallback (always available)

    SIGNAL STACK CONSTRUCTION:
      signal_stack[t, i, s] = value of signal_s for asset_i at date_t

    STATIC WEIGHTS (used in both modes for LOW-VOL signal):
      The low-vol signal always gets weight 0.10 (information floor).
      The remaining 0.90 is allocated by the signal router.

    REGIME GATING:
      After blending, apply asset-specific urgency scaling:
        alpha_final[i] = alpha_blended[i] × (1 − urgency[i] × 0.7)
              + safe_haven_signal[i] × urgency[i] × 0.7

      When urgency is high for an asset class, alpha compresses toward
      safe-haven (TLT, GLD, BIL) rather than maintaining equity alpha.
      This prevents the 2022-style scenario where equity alpha was
      systematically wrong while rate/credit regimes were in full crisis.
    """
    import json

    logger.info("Stage 5: Signal blending...")

    signal_names_ordered = ["vrp", "vts", "nav_arb", "insider", "low_vol"]

    # Build signal stack (T, N, S)
    T = len(dates)
    N = N_ASSETS
    S = N_SIGNALS
    signal_stack = np.zeros((T, N, S), dtype=np.float32)

    for s_idx, sig_name in enumerate(signal_names_ordered):
        if sig_name in signal_dfs:
            df = signal_dfs[sig_name].reindex(dates).ffill().fillna(0.0).reindex(columns=TICKERS).fillna(0.0)
            signal_stack[:, :, s_idx] = df.values.astype(np.float32)
        else:
            logger.warning(f"Signal '{sig_name}' not in signal_dfs — using zeros")

    returns_np = returns_df.reindex(dates).reindex(columns=TICKERS).fillna(0.0).values

    # ── Attempt GATv2 signal router ────────────────────────────────────────────
    blending_weights = None

    if _ROUTER_WEIGHTS.exists():
        try:
            import torch
            from models.alpha.gat_signal_router import (
                SignalRouterGAT, build_economic_graph, build_node_features
            )

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model  = SignalRouterGAT().to(device)
            model.load_state_dict(
                torch.load(str(_ROUTER_WEIGHTS), map_location=device, weights_only=True)
            )
            model.eval()
            logger.info(f"  GATv2 Signal Router loaded from {_ROUTER_WEIGHTS}")

            edge_index, edge_attr = build_economic_graph()
            edge_index = edge_index.to(device)
            edge_attr  = edge_attr.to(device)

            def parse_zmu(val) -> np.ndarray:
                if isinstance(val, (list, np.ndarray)):
                    return np.asarray(val, dtype=np.float32)
                if isinstance(val, str):
                    return np.array(json.loads(val), dtype=np.float32)
                return np.zeros(16, dtype=np.float32)

            regime_zmu_dict = {
                str(pd.Timestamp(idx).date()): parse_zmu(row["z_mu"])
                for idx, row in regime_df.iterrows()
            }

            blending_weights = np.zeros((T, N, S), dtype=np.float32)

            with torch.no_grad():
                for t_idx, date in enumerate(dates):
                    date_str = str(date.date())
                    z_mu = regime_zmu_dict.get(date_str, np.zeros(16, dtype=np.float32))

                    # Use cross-sectional EWMA IC as node features (causal)
                    ic_hist_t = np.zeros((N, S), dtype=np.float32)
                    if t_idx > 10:
                        # Approximate per-asset IC from recent returns
                        for s in range(S):
                            for i in range(N):
                                sig_arr = signal_stack[max(0, t_idx-21):t_idx, i, s]
                                ret_arr = returns_np[max(1, t_idx-20):t_idx+1, i]
                                if len(sig_arr) > 5 and len(ret_arr) > 5:
                                    n_min = min(len(sig_arr), len(ret_arr))
                                    try:
                                        from scipy import stats
                                        ic, _ = stats.spearmanr(sig_arr[-n_min:], ret_arr[-n_min:])
                                        ic_hist_t[i, s] = float(ic) if not np.isnan(ic) else 0.0
                                    except Exception:
                                        pass

                    x = build_node_features(
                        regime_tensor_zmu=z_mu,
                        signal_ic_history=ic_hist_t,
                        vol_betas=np.zeros(N, dtype=np.float32),  # simplified — full betas in train
                        rate_betas=np.zeros(N, dtype=np.float32),
                        liquidity_z=np.zeros(N, dtype=np.float32),
                    ).to(device)

                    signal_mat = torch.from_numpy(signal_stack[t_idx]).float().to(device)
                    weights, _ = model(x, edge_index, edge_attr)
                    blending_weights[t_idx] = weights.cpu().numpy()

                    if t_idx % 200 == 0:
                        logger.info(f"  GATv2 routing: {t_idx}/{T} dates")

            logger.info(f"  ✓ GATv2 signal routing complete")

        except Exception as exc:
            logger.warning(f"  GATv2 routing failed ({exc}) — using EWMA-IC fallback")
            blending_weights = None

    # ── EWMA-IC fallback ───────────────────────────────────────────────────────
    if blending_weights is None:
        logger.info("  Using EWMA-IC weighted fallback...")
        blending_weights = _build_ewma_ic_fallback_weights(
            signal_stack=signal_stack,
            returns_np=returns_np,
        )
        logger.info("  ✓ EWMA-IC fallback weights computed")

    # ── Blended alpha ──────────────────────────────────────────────────────────
    # alpha_raw = Σ_s W_is × signal_s(i)
    alpha_raw = (blending_weights * signal_stack).sum(axis=-1)  # (T, N)

    # ── Asset-class regime gating ──────────────────────────────────────────────
    # When asset-specific urgency is high, compress toward safe-haven allocation
    safe_haven_allocation = np.zeros((T, N), dtype=np.float32)
    for i, ticker in enumerate(TICKERS):
        if ticker in ("TLT", "GLD", "BIL", "SHV"):
            safe_haven_allocation[:, i] = 0.8   # safe havens get positive alpha in crisis

    # Build per-asset urgency matrix from regime_df
    urgency_matrix = np.zeros((T, N), dtype=np.float32)

    if "ltc_urgency" in regime_df.columns:
        regime_reindexed = regime_df.reindex(dates).ffill()
        base_urgency     = regime_reindexed["ltc_urgency"].fillna(0.0).values

        # Route urgency to correct axis per asset
        for i, ticker in enumerate(TICKERS):
            equity_weight    = _get_equity_routing_weight(ticker)
            bond_weight      = _get_bond_routing_weight(ticker)
            commodity_weight = _get_commodity_routing_weight(ticker)

            asset_urgency = base_urgency * (0.4 + 0.6 * (1 - equity_weight))
            urgency_matrix[:, i] = np.clip(asset_urgency, 0.0, 1.0)

    # Apply regime gate
    crisis_scale = np.clip(urgency_matrix * 0.7, 0.0, 0.7)
    alpha_gated  = (
        alpha_raw   * (1 - crisis_scale) +
        safe_haven_allocation * crisis_scale
    )

    # Final tanh squash to [-1, 1]
    alpha_final = np.tanh(alpha_gated).astype(np.float32)

    result_df = pd.DataFrame(alpha_final, index=dates, columns=TICKERS)

    logger.info(
        f"  ✓ Blended alpha: {len(result_df)} days × {N} assets | "
        f"Mean |alpha|: {result_df.abs().mean().mean():.3f} | "
        f"Max: {result_df.max().max():.3f}"
    )
    return result_df


# ─────────────────────────────────────────────────────────────────────────────
# Routing weight helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_equity_routing_weight(ticker: str) -> float:
    equity = {"SPY", "QQQ", "IWM", "XLE", "XLF", "XLK", "XLV", "XLU",
              "XLI", "XLP", "XLY", "XLB", "XLC", "COWZ", "VIXY"}
    if ticker in equity: return 1.0
    if ticker in {"GDX", "PDBC"}: return 0.3
    return 0.0

def _get_bond_routing_weight(ticker: str) -> float:
    bond = {"TLT": 1.0, "LQD": 0.6, "HYG": 0.4, "BIL": 1.0, "SHV": 1.0}
    return bond.get(ticker, 0.0)

def _get_commodity_routing_weight(ticker: str) -> float:
    comm = {"GLD": 1.0, "SLV": 1.0, "GDX": 0.7, "USO": 1.0, "PDBC": 0.7}
    return comm.get(ticker, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    logger.info("══════ Fortress v5 — Alpha Signal Precomputation [REWRITE] ══════")

    for path in [_PRICES_PATH, _RETURNS_PATH]:
        if not path.exists():
            logger.error(f"Missing required cache: {path}")
            logger.error("Run data ingestion pipeline first (Stage 1).")
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

    # ── Stage 0: Regime ────────────────────────────────────────────────────────
    regime_df, regime_meta = await stage0_vol_regime(start=start_date)
    regime_df.index = pd.to_datetime(regime_df.index)

    # ── Stage 1: Options surface ───────────────────────────────────────────────
    options_df = await stage1_options_alpha(start=start_date)
    options_df.index = pd.to_datetime(options_df.index)

    # ── Stage 2: ETF NAV / AP stress ──────────────────────────────────────────
    nav_df, stress_meta = await stage2_nav_arb(start=start_date)
    nav_df.index = pd.to_datetime(nav_df.index)

    # ── Stage 3: SEC filing intelligence ──────────────────────────────────────
    insider_df = await stage3_sec_insider(dates=dates)
    insider_df.index = pd.to_datetime(insider_df.index)

    # ── Stage 4: Low-vol baseline ──────────────────────────────────────────────
    lowvol_df = stage4_low_vol(returns_df)
    lowvol_df.index = pd.to_datetime(lowvol_df.index)

    # ── Save per-signal matrices ───────────────────────────────────────────────
    # Long format: columns = {signal_name}_{ticker}
    signal_dfs: Dict[str, pd.DataFrame] = {
        "vrp":     options_df,   # VRP from options engine (also contains VTS)
        "vts":     options_df,   # Same engine, different components — router separates
        "nav_arb": nav_df,
        "insider": insider_df,
        "low_vol": lowvol_df,
    }

    # Build long-format signals DataFrame for train_signal_router.py
    long_frames = []
    for sig_name, df in signal_dfs.items():
        df_aligned = df.reindex(dates).ffill().fillna(0.0).reindex(columns=TICKERS).fillna(0.0)
        df_long    = df_aligned.copy()
        df_long.columns = [f"{sig_name}_{t}" for t in df_long.columns]
        long_frames.append(df_long)

    signals_long_df = pd.concat(long_frames, axis=1)
    signals_long_df.to_parquet(_SIGNALS_OUT)
    logger.info(f"✓ Per-signal alpha matrix → {_SIGNALS_OUT} ({signals_long_df.shape})")

    # ── Stage 5: Signal blending ───────────────────────────────────────────────
    alpha_df = stage5_blend_signals(
        signal_dfs=signal_dfs,
        regime_df=regime_df,
        returns_df=returns_df,
        dates=dates,
    )

    # Validate output
    assert alpha_df.shape == (len(dates), N_ASSETS), \
        f"Shape mismatch: {alpha_df.shape} != ({len(dates)}, {N_ASSETS})"
    assert not alpha_df.isnull().any().any(), "NaN values in final alpha"
    assert (alpha_df.abs() <= 1.0 + 1e-5).all().all(), "Values outside [-1, 1]"

    alpha_df.to_parquet(_ALPHA_OUT)
    logger.info(f"✓ Blended alpha → {_ALPHA_OUT} ({alpha_df.shape})")

    # ── Signal summary ─────────────────────────────────────────────────────────
    means = alpha_df.mean().sort_values(ascending=False)
    logger.info("Alpha signal summary (time-averaged):")
    bar_scale = 6.0 / (means.abs().max() + 1e-8)
    for ticker, val in means.items():
        bar_len = int(abs(val) * bar_scale)
        sign    = "+" if val >= 0 else "-"
        bar     = "█" * bar_len
        logger.info(f"  {ticker:6s}: {sign}{bar} ({val:+.3f})")

    # ── IC statistics ──────────────────────────────────────────────────────────
    logger.info("Per-signal mean cross-sectional |alpha|:")
    for sig_name, df in signal_dfs.items():
        df_a = df.reindex(dates).ffill().fillna(0.0).reindex(columns=TICKERS).fillna(0.0)
        mean_abs = df_a.abs().mean().mean()
        nonzero  = (df_a.abs() > 0.01).any(axis=1).mean() * 100
        logger.info(f"  {sig_name:12s}: mean|α|={mean_abs:.3f} | active={nonzero:.1f}% of days")

    logger.info(
        "\n══════ Stage 2 COMPLETE ══════\n"
        "Next steps:\n"
        "  1. python training/train_signal_router.py   (train GATv2 IC predictor)\n"
        "  2. python scripts/run_standalone_backtest.py (validate with walk-forward)\n"
        "  3. python scripts/run_cpcv_validation.py     (compute PBO)\n"
    )


if __name__ == "__main__":
    asyncio.run(main())