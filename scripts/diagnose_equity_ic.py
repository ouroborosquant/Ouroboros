"""
FORTRESS v5 — scripts/diagnose_equity_ic.py

Equity-only IC validation for signals that output 0.0 for macro ETF tickers.

Problem
-------
validate_signal_ic.py computes Spearman IC across all 100 assets. For signals
like night_effect and pca_statarb that are 0.0 for the 25 macro ETFs, including
those 25 zero-signal assets in the cross-sectional rank correlation systematically
dilutes (and can invert) the measured IC.

Illustration: if equity-only IC = +0.025 but 25 zero-signal ETFs are added,
the mixed-universe IC can drop to +0.008 or flip to -0.012 depending on how
ETF forward returns happen to rank relative to the zero-signal midpoint.

This script computes IC separately for:
  1. Equity-only subset (75 single-name equities) — for night_effect, pca_statarb
  2. ETF-only subset (25 macro ETFs) — for mom, low_vol signals
  3. Full universe (100 assets) — baseline comparison

Run
---
  PYTHONPATH=. python scripts/diagnose_equity_ic.py
"""
from __future__ import annotations

import logging
import yaml
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("EquityIC")

_CACHE_DIR = Path("research/outputs/cache")
_UNIVERSE_FILE = Path("config/universe.yaml")

HORIZONS = [1, 5, 21, 63]
IC_VIABLE = 0.03

# ── Load universe ─────────────────────────────────────────────────────────────
with open(_UNIVERSE_FILE) as f:
    _cfg = yaml.safe_load(f)

ALL_TICKERS = [a["ticker"] for a in _cfg["assets"]]
_CAT = {a["ticker"]: a.get("category", "") for a in _cfg["assets"]}
_SINGLE = {"equity_single"}
_ETF_CATS = {"equity_broad", "equity_sector", "equity_intl", "fixed_income",
             "commodity", "volatility"}

ETF_TICKERS    = [t for t in ALL_TICKERS if _CAT.get(t) in _ETF_CATS]
EQUITY_TICKERS = [t for t in ALL_TICKERS if _CAT.get(t) in _SINGLE]

# Signals that should be evaluated on the equity-only subset
EQUITY_SCOPED_SIGNALS = {"night_effect", "pca_statarb"}
# Signals that should be evaluated on ETF-only subset
ETF_SCOPED_SIGNALS    = {"low_vol", "mom", "conc_lead", "gex_flow"}


def compute_ic_subset(
    sig_arr: np.ndarray,   # (T, N_sub)
    ret_arr: np.ndarray,   # (T, N_sub)
    horizon: int,
    min_assets: int = 3,
) -> float:
    """
    Per-date cross-sectional Spearman IC, averaged over active dates.
    Active date = at least `min_assets` non-zero signal values.
    """
    T = sig_arr.shape[0]
    ic_series = []

    for t in range(T - horizon):
        s = sig_arr[t]
        r = ret_arr[t + horizon]
        active = np.isfinite(s) & np.isfinite(r) & (np.abs(s) > 1e-6)
        if active.sum() < min_assets:
            continue
        ic_val, _ = spearmanr(s[active], r[active])
        if np.isfinite(ic_val):
            ic_series.append(ic_val)

    return float(np.mean(ic_series)) if ic_series else np.nan


def main() -> None:
    signals_df = pd.read_parquet(_CACHE_DIR / "alpha_signals.parquet")
    returns_df = pd.read_parquet(_CACHE_DIR / "returns_wide.parquet")
    returns_df.index = pd.to_datetime(returns_df.index)
    signals_df.index = pd.to_datetime(signals_df.index)

    common_dates = signals_df.index.intersection(returns_df.index)
    returns_df = returns_df.reindex(common_dates)
    signals_df = signals_df.reindex(common_dates)

    # Identify signal names from column prefixes
    all_cols = signals_df.columns.tolist()
    signal_names = sorted({c.rsplit("_", 1)[0] if "_" in c else c
                            for c in all_cols
                            if not c.startswith("gex_flow_")})
    # Handle gex_flow separately
    if any(c.startswith("gex_flow_") for c in all_cols):
        signal_names.append("gex_flow")
    signal_names = sorted(set(signal_names))

    logger.info(f"Signals found: {signal_names}")
    logger.info(f"Full universe: {len(ALL_TICKERS)} | ETFs: {len(ETF_TICKERS)} | Equities: {len(EQUITY_TICKERS)}")
    logger.info("=" * 72)

    for sig_name in signal_names:
        # Extract signal columns for this signal
        sig_cols_all    = [f"{sig_name}_{t}" for t in ALL_TICKERS    if f"{sig_name}_{t}" in signals_df.columns]
        sig_cols_etf    = [f"{sig_name}_{t}" for t in ETF_TICKERS    if f"{sig_name}_{t}" in signals_df.columns]
        sig_cols_equity = [f"{sig_name}_{t}" for t in EQUITY_TICKERS if f"{sig_name}_{t}" in signals_df.columns]

        ret_all    = returns_df[ALL_TICKERS].values
        ret_etf    = returns_df[ETF_TICKERS].values
        ret_equity = returns_df[EQUITY_TICKERS].values

        sig_all    = signals_df[sig_cols_all].values    if sig_cols_all    else None
        sig_etf    = signals_df[sig_cols_etf].values    if sig_cols_etf    else None
        sig_equity = signals_df[sig_cols_equity].values if sig_cols_equity else None

        logger.info(f"\n  Signal: {sig_name}")
        logger.info(f"  {'Horizon':>8}  {'Full-100':>10}  {'ETF-only':>10}  {'Equity-only':>12}")
        logger.info(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*12}")

        for h in HORIZONS:
            ic_full   = compute_ic_subset(sig_all,    ret_all,    h) if sig_all    is not None else np.nan
            ic_etf    = compute_ic_subset(sig_etf,    ret_etf,    h) if sig_etf    is not None else np.nan
            ic_equity = compute_ic_subset(sig_equity, ret_equity, h) if sig_equity is not None else np.nan

            def _fmt(v: float) -> str:
                if not np.isfinite(v):
                    return "   nan"
                star = " ✓" if abs(v) >= IC_VIABLE else "  "
                return f"{v:+.4f}{star}"

            logger.info(f"  {h:>7}d  {_fmt(ic_full):>12}  {_fmt(ic_etf):>12}  {_fmt(ic_equity):>14}")

        # Scope verdict
        scope_tag = "(equity-scoped)" if sig_name in EQUITY_SCOPED_SIGNALS else \
                    "(etf-scoped)"    if sig_name in ETF_SCOPED_SIGNALS    else "(full-universe)"
        logger.info(f"  → Primary scope: {scope_tag}")

    logger.info("\n" + "=" * 72)
    logger.info("INTERPRETATION:")
    logger.info("  night_effect / pca_statarb: use 'Equity-only' column as ground truth.")
    logger.info("  low_vol / mom / conc_lead:  ETF-only and equity-only are now separate signals")
    logger.info("                              (class-aware ranking). Full-100 is still diluted.")
    logger.info("  IC threshold for viability: ±0.03")


if __name__ == "__main__":
    main()