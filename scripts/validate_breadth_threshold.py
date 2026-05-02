#!/usr/bin/env python3
"""
scripts/validate_breadth_threshold.py
======================================
Pre-flight diagnostic: validates that the breadth_ratio threshold of 0.35
correctly identifies F8's concentrated regime window before you commit to
a full 20-minute walk-forward run.

Outputs:
  1. Per-bar breadth_ratio time series
  2. Regime classification (CONC vs DISPERSED)
  3. IC-halt suppression rate by regime
  4. Alpha concentration (HHI) aligned with regime classification

Run first. If breadth_ratio in F8 OOS window (2023-07-01 to 2023-12-31)
is NOT consistently below 0.35, adjust CONC_BREADTH_THRESHOLD before patching
the backtest.

Usage:
    PYTHONPATH=. python scripts/validate_breadth_threshold.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Project imports
sys.path.insert(0, str(Path(__file__).parents[1]))
from regime_allocator import (
    ALPHA_POSITIVE_FLOOR,
    CONC_BREADTH_THRESHOLD,
    compute_breadth_ratio,
    compute_hhi,
    is_conc_regime,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

ALPHA_PATH   = Path("research/outputs/alpha_signals.parquet")
RETURNS_PATH = Path("research/outputs/returns.parquet")
F8_OOS_START = pd.Timestamp("2023-07-01")
F8_OOS_END   = pd.Timestamp("2023-12-31")
IC_WINDOW    = 21   # Spearman rolling window (reverted from v11.0's 42)
FWD_HORIZON  = 5    # 5-day forward return for IC


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    alpha   = pd.read_parquet(ALPHA_PATH)
    returns = pd.read_parquet(RETURNS_PATH)
    # Align on common dates
    common  = alpha.index.intersection(returns.index)
    return alpha.loc[common], returns.loc[common]


def compute_rolling_spearman_ic(
    alpha_df:   pd.DataFrame,
    returns_df: pd.DataFrame,
    horizon:    int = FWD_HORIZON,
    window:     int = IC_WINDOW,
) -> pd.Series:
    """
    Cross-sectional Spearman IC: rank correlation between composite alpha
    and horizon-day forward returns, rolled over time.

    NOTE: forward returns use shift(-horizon) — causality-clean because this
    is a validation script only; the backtest loop uses realized returns at t+h.
    """
    fwd_returns = returns_df.shift(-horizon)
    composite   = alpha_df.mean(axis=1) if alpha_df.ndim > 1 else alpha_df

    # Compute composite alpha column-sum if multi-signal df
    if isinstance(alpha_df, pd.DataFrame) and alpha_df.shape[1] > 1:
        # Assume alpha_df has one row per date, N_TICKERS columns
        ic_series = pd.Series(index=alpha_df.index, dtype=float)
        for i in range(window, len(alpha_df)):
            window_alpha = alpha_df.iloc[i - window:i]
            window_fwd   = fwd_returns.iloc[i - window:i]
            # Cross-sectional IC at each step: mean over window
            ics = []
            for j in range(len(window_alpha)):
                a_row = window_alpha.iloc[j].dropna()
                f_row = window_fwd.iloc[j].reindex(a_row.index).dropna()
                a_row = a_row.reindex(f_row.index)
                if len(f_row) < 5:
                    continue
                rho, _ = spearmanr(a_row.values, f_row.values)
                if not np.isnan(rho):
                    ics.append(rho)
            ic_series.iloc[i] = np.nanmean(ics) if ics else np.nan
        return ic_series
    else:
        # Single composite series path
        flat_alpha = alpha_df.iloc[:, 0] if isinstance(alpha_df, pd.DataFrame) else alpha_df
        return flat_alpha.rolling(window).corr(fwd_returns.mean(axis=1))


def main() -> None:
    logger.info("Loading alpha and returns...")
    alpha_df, returns_df = load_data()

    tickers   = list(alpha_df.columns)
    n_tickers = len(tickers)
    logger.info(f"Universe: {n_tickers} tickers | Dates: {alpha_df.index[0]} → {alpha_df.index[-1]}")

    # ── Breadth ratio time series ─────────────────────────────────────────────
    breadth_series = alpha_df.apply(
        lambda row: (row > ALPHA_POSITIVE_FLOOR).sum() / n_tickers,
        axis=1,
    )
    hhi_series = alpha_df.apply(compute_hhi, axis=1)

    regime_series = breadth_series.apply(
        lambda b: "CONC" if is_conc_regime(b) else "DISP"
    )

    # ── F8 window analysis ────────────────────────────────────────────────────
    f8_mask  = (breadth_series.index >= F8_OOS_START) & (breadth_series.index <= F8_OOS_END)
    f8_bread = breadth_series[f8_mask]
    f8_hhi   = hhi_series[f8_mask]
    f8_reg   = regime_series[f8_mask]

    conc_pct  = (f8_reg == "CONC").mean() * 100
    disp_pct  = (f8_reg == "DISP").mean() * 100

    logger.info("=" * 60)
    logger.info(f"F8 OOS window ({F8_OOS_START.date()} → {F8_OOS_END.date()}):")
    logger.info(f"  CONC days:   {(f8_reg=='CONC').sum()} ({conc_pct:.1f}%)")
    logger.info(f"  DISP days:   {(f8_reg=='DISP').sum()} ({disp_pct:.1f}%)")
    logger.info(f"  Breadth mean:  {f8_bread.mean():.3f}  (threshold={CONC_BREADTH_THRESHOLD})")
    logger.info(f"  Breadth min:   {f8_bread.min():.3f}")
    logger.info(f"  HHI mean:      {f8_hhi.mean():.3f}  (1.0=monopolar)")
    logger.info("=" * 60)

    # ── IC-halt suppression audit ─────────────────────────────────────────────
    # Compute rolling IC with Spearman (v8.2 method, NOT v11.0 Pearson)
    logger.info("Computing rolling Spearman IC (this may take ~30s)...")

    # Flatten alpha to composite score per date for IC computation
    alpha_composite = alpha_df.stack().groupby(level=0).mean()  # mean across tickers per date
    fwd_ret_composite = returns_df.shift(-FWD_HORIZON).stack().groupby(level=0).mean()

    common_idx = alpha_composite.index.intersection(fwd_ret_composite.index)
    alpha_composite    = alpha_composite.loc[common_idx]
    fwd_ret_composite  = fwd_ret_composite.loc[common_idx]

    rolling_ic = alpha_composite.rolling(IC_WINDOW).corr(fwd_ret_composite)

    # Simulate IC halt decisions with and without breadth gate
    IC_HALT_THRESHOLD = -0.02
    IC_HALT_WINDOW    = 15

    halt_days_counter   = 0
    halted_old          = []  # v8.2 logic
    halted_new          = []  # v8.3 gated logic

    for date in common_idx:
        ic_val    = rolling_ic.loc[date]
        breadth   = breadth_series.loc[date] if date in breadth_series.index else 0.5
        in_conc   = is_conc_regime(breadth)

        if pd.isna(ic_val):
            halted_old.append(0)
            halted_new.append(0)
            halt_days_counter = 0
            continue

        if ic_val < IC_HALT_THRESHOLD:
            halt_days_counter += 1
        else:
            halt_days_counter = 0

        # Old logic (v8.2): halt if IC bad for N days
        old_halt = int(ic_val < IC_HALT_THRESHOLD and halt_days_counter >= IC_HALT_WINDOW)

        # New logic (v8.3): same but suppressed in CONC regime
        new_halt = 0 if in_conc else old_halt

        halted_old.append(old_halt)
        halted_new.append(new_halt)

    halt_df = pd.DataFrame({
        "rolling_ic":   rolling_ic.loc[common_idx].values,
        "breadth":      breadth_series.reindex(common_idx).values,
        "regime":       regime_series.reindex(common_idx).values,
        "halt_v82":     halted_old,
        "halt_v83":     halted_new,
    }, index=common_idx)

    f8_halt = halt_df[f8_mask.reindex(common_idx, fill_value=False)]
    if not f8_halt.empty:
        logger.info(f"F8 IC-halt comparison:")
        logger.info(f"  v8.2 halt rate: {f8_halt['halt_v82'].mean()*100:.1f}%")
        logger.info(f"  v8.3 halt rate: {f8_halt['halt_v83'].mean()*100:.1f}%  (gated)")
        logger.info(f"  IC halts suppressed by breadth gate: "
                    f"{(f8_halt['halt_v82'] - f8_halt['halt_v83']).clip(lower=0).sum()} days")

    # ── Threshold sensitivity ─────────────────────────────────────────────────
    logger.info("\nThreshold sensitivity (F8 CONC day coverage):")
    for thresh in [0.25, 0.30, 0.35, 0.40, 0.45]:
        pct = (f8_bread < thresh).mean() * 100
        logger.info(f"  threshold={thresh:.2f} → {pct:.1f}% of F8 classified CONC")

    logger.info("\nRecommendation:")
    if conc_pct > 50:
        logger.info(f"  ✅ threshold=0.35 covers {conc_pct:.1f}% of F8 — proceed with patch")
    elif conc_pct > 30:
        logger.info(f"  ⚠️  threshold=0.35 only covers {conc_pct:.1f}% of F8 — consider 0.40")
    else:
        logger.info(f"  ❌ threshold=0.35 covers only {conc_pct:.1f}% of F8 — alpha signal "
                    f"may not be sparse in this window. Check precompute output for F8 dates.")

    # ── Save diagnostic parquet ───────────────────────────────────────────────
    out_path = Path("research/outputs/breadth_diagnostic.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    halt_df.to_parquet(out_path)
    logger.info(f"\n✅ Diagnostic → {out_path}")
    logger.info("Next: if F8 CONC coverage >50%, run v8.3 backtest.")
    logger.info("      If coverage <30%, debug alpha signal for F8 window via:")
    logger.info("      PYTHONPATH=. python scripts/validate_signal_ic.py "
                "--start 2023-07-01 --end 2023-12-31 --per-ticker")


if __name__ == "__main__":
    main()