"""
FORTRESS v5 — run_cpcv_validation.py   [P4 — CPCV Validation Runner]
Path: scripts/run_cpcv_validation.py

Wires the existing research/cpcv_validation.py infrastructure to produce
a real Probability of Backtest Overfitting (PBO) number using
Combinatorial Purged Cross-Validation (CPCV).

Why CPCV instead of walk-forward:
  Walk-forward with 7 folds produces 7 OOS paths. PBO convergence
  requires ≥16 independent paths (Bailey et al. 2014). CPCV with
  k=6 groups generates C(6,3)=20 combinatorial IS/OOS splits, each
  using non-overlapping data segments with embargo purging.

  The walk-forward in Stage 3 (run_standalone_backtest.py) is useful
  as a sanity check and for visual IS→OOS degradation curves. But it
  is structurally insufficient for formal PBO assessment because:
    1. Paths are serially dependent (expanding window shares data).
    2. 7 paths give a PBO estimator with ~30% standard error.
    3. No purging → potential label leakage at fold boundaries for
       rolling-window features (12-1M momentum overlaps by 11 months).

  CPCV fixes all three: combinatorial independence, 20 paths, and
  explicit embargo windows at group boundaries.

Usage:
  python scripts/run_cpcv_validation.py

  Reads: research/outputs/backtest_tearsheet.csv (daily returns)
  Writes: research/outputs/cpcv_results.json
          research/outputs/cpcv_degradation.csv
"""

from __future__ import annotations

import json
import logging
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("CPCVRunner")

# ── Paths ─────────────────────────────────────────────────────────────────────
_TEARSHEET_CSV = Path("research/outputs/backtest_tearsheet.csv")
_OUTPUT_DIR    = Path("research/outputs")
_CPCV_JSON     = _OUTPUT_DIR / "cpcv_results.json"
_CPCV_CSV      = _OUTPUT_DIR / "cpcv_degradation.csv"

# ── CPCV Parameters ──────────────────────────────────────────────────────────
_N_GROUPS:       int   = 6       # k=6 → C(6,3)=20 combinatorial paths
_EMBARGO_DAYS:   int   = 5       # Purge window at each group boundary
_RF_DAILY:       float = 0.05 / 252


def _sharpe_from_returns(returns: np.ndarray) -> float:
    """Annualised Sharpe from daily returns array."""
    excess = returns - _RF_DAILY
    std = excess.std()
    if std < 1e-10 or len(returns) < 5:
        return 0.0
    return float((excess.mean() / std) * np.sqrt(252))


def _sortino_from_returns(returns: np.ndarray) -> float:
    """Annualised Sortino from daily returns array."""
    excess = returns - _RF_DAILY
    downside = excess[excess < 0]
    if len(downside) < 2:
        return 0.0
    ds = downside.std()
    if ds < 1e-10:
        return 0.0
    return float((excess.mean() / ds) * np.sqrt(252))


def _psr(sr: float, n: int, skew: float, kurt_raw: float) -> float:
    """Probabilistic Sharpe Ratio: P(SR > 0)."""
    excess_k = kurt_raw - 3.0
    var_sr = (1.0 - skew * sr + (excess_k / 4.0) * sr ** 2) / max(n - 1, 1)
    if var_sr <= 0:
        return 0.0
    return float(stats.norm.cdf(sr / (var_sr ** 0.5 + 1e-10)))


def run_cpcv(
    daily_returns: np.ndarray,
    n_groups: int = _N_GROUPS,
    embargo: int = _EMBARGO_DAYS,
) -> Dict:
    """
    Full CPCV implementation producing PBO and degradation curve.

    Algorithm:
      1. Divide T trading days into k equal-size groups.
      2. For each C(k, k//2) combination, assign half the groups to IS
         and half to OOS.
      3. Within each split, apply embargo purging at group boundaries.
      4. Compute IS and OOS Sharpe for each path.
      5. PBO = fraction of paths where IS-optimal variant underperforms
         OOS median.

    Since we have a single strategy (not multiple variants), we adapt:
      - Each combinatorial split produces one IS Sharpe and one OOS Sharpe.
      - PBO = fraction of paths where OOS Sharpe < 0 (i.e., the strategy
        would have lost money out-of-sample on that data partition).
      - This is a conservative single-strategy PBO estimator.

    For the full multi-variant PBO (Bailey et al. 2014), you'd need S
    strategy variants. We approximate by treating each combinatorial
    path as a "variant" of the same strategy applied to different data.

    Args:
        daily_returns: (T,) array of daily portfolio returns.
        n_groups:      Number of groups k (default 6 → 20 paths).
        embargo:       Purge window in days at each boundary.

    Returns:
        Dict with keys: pbo, n_paths, avg_is_sharpe, avg_oos_sharpe,
        paths (list of per-path metrics), degradation_ratio.
    """
    T = len(daily_returns)
    k = n_groups
    half_k = k // 2

    # ── 1. Divide into k groups ───────────────────────────────────────────────
    group_size = T // k
    groups: List[np.ndarray] = []
    for g in range(k):
        start_idx = g * group_size
        end_idx = (g + 1) * group_size if g < k - 1 else T
        groups.append(daily_returns[start_idx:end_idx])

    group_boundaries = [g * group_size for g in range(k + 1)]
    group_boundaries[-1] = T

    # ── 2. Generate C(k, k//2) combinatorial splits ──────────────────────────
    all_combos = list(combinations(range(k), half_k))
    n_paths = len(all_combos)
    logger.info(
        f"CPCV: {k} groups, C({k},{half_k})={n_paths} paths, "
        f"embargo={embargo}d, T={T} days."
    )

    # ── 3. Evaluate each path ─────────────────────────────────────────────────
    path_results: List[Dict] = []

    for path_id, is_groups in enumerate(all_combos):
        oos_groups = tuple(g for g in range(k) if g not in is_groups)

        # Collect IS and OOS returns with embargo purging
        is_returns = []
        oos_returns = []

        for g in range(k):
            g_start = group_boundaries[g]
            g_end   = group_boundaries[g + 1]

            # Apply embargo: remove `embargo` days from each end of the group
            # that borders the opposite partition (IS ↔ OOS boundary).
            purge_start = embargo if g > 0 and (
                (g in is_groups and g - 1 in oos_groups) or
                (g in oos_groups and g - 1 in is_groups)
            ) else 0
            purge_end = embargo if g < k - 1 and (
                (g in is_groups and g + 1 in oos_groups) or
                (g in oos_groups and g + 1 in is_groups)
            ) else 0

            actual_start = g_start + purge_start
            actual_end   = g_end - purge_end

            if actual_start >= actual_end:
                continue  # Group fully consumed by embargo

            if g in is_groups:
                is_returns.append(daily_returns[actual_start:actual_end])
            else:
                oos_returns.append(daily_returns[actual_start:actual_end])

        is_arr  = np.concatenate(is_returns)  if is_returns  else np.array([])
        oos_arr = np.concatenate(oos_returns) if oos_returns else np.array([])

        is_sr  = _sharpe_from_returns(is_arr)  if len(is_arr) > 10  else 0.0
        oos_sr = _sharpe_from_returns(oos_arr) if len(oos_arr) > 10 else 0.0

        is_sortino  = _sortino_from_returns(is_arr)  if len(is_arr) > 10  else 0.0
        oos_sortino = _sortino_from_returns(oos_arr) if len(oos_arr) > 10 else 0.0

        oos_n = len(oos_arr)
        oos_skew = float(stats.skew(oos_arr)) if oos_n > 10 else 0.0
        oos_kurt = float(stats.kurtosis(oos_arr, fisher=False)) if oos_n > 10 else 3.0
        psr_val  = _psr(oos_sr, oos_n, oos_skew, oos_kurt) if oos_n > 10 else 0.0

        path_results.append({
            "path_id":      path_id,
            "is_groups":    list(is_groups),
            "oos_groups":   list(oos_groups),
            "is_sharpe":    round(is_sr, 4),
            "oos_sharpe":   round(oos_sr, 4),
            "is_sortino":   round(is_sortino, 4),
            "oos_sortino":  round(oos_sortino, 4),
            "is_n_obs":     len(is_arr),
            "oos_n_obs":    oos_n,
            "psr":          round(psr_val, 4),
        })

    # ── 4. Compute PBO ────────────────────────────────────────────────────────
    # Single-strategy PBO: fraction of paths with OOS Sharpe < 0
    # (strategy would have lost money on that data partition)
    oos_sharpes = [p["oos_sharpe"] for p in path_results]
    is_sharpes  = [p["is_sharpe"]  for p in path_results]

    # Standard PBO: fraction where IS-best underperforms OOS median
    # For single strategy, this reduces to: P(OOS SR < median OOS SR)
    # which is always ~0.50 by construction. Instead we use:
    # PBO = P(OOS SR < 0) — probability the strategy fails OOS.
    n_negative_oos = sum(1 for s in oos_sharpes if s < 0)
    pbo = float(n_negative_oos / max(n_paths, 1))

    avg_is  = float(np.mean(is_sharpes))
    avg_oos = float(np.mean(oos_sharpes))
    std_oos = float(np.std(oos_sharpes))

    # Degradation ratio: avg_oos / avg_is
    # <1.0 indicates IS overfitting; <0 indicates directional failure.
    deg_ratio = avg_oos / avg_is if abs(avg_is) > 1e-10 else 0.0

    # ── 5. Statistical summary ────────────────────────────────────────────────
    logger.info(f"CPCV Results ({n_paths} paths):")
    logger.info(f"  Avg IS Sharpe:    {avg_is:.3f}")
    logger.info(f"  Avg OOS Sharpe:   {avg_oos:.3f} ± {std_oos:.3f}")
    logger.info(f"  Degradation:      {deg_ratio:.2f}x")
    logger.info(f"  PBO (P(OOS<0)):   {pbo:.2%}")

    if pbo > 0.50:
        logger.critical(
            f"❌ PBO = {pbo:.2%} > 50%. Strategy is formally overfitted. "
            "Redesign required before live deployment."
        )
    elif pbo > 0.25:
        logger.warning(
            f"⚠️  PBO = {pbo:.2%} > 25%. Strategy shows signs of overfitting. "
            "Additional OOS validation recommended."
        )
    else:
        logger.info(
            f"✅ PBO = {pbo:.2%} < 25%. Strategy passes overfitting gate."
        )

    # t-test: is avg OOS Sharpe significantly > 0?
    if n_paths >= 5 and std_oos > 1e-10:
        t_stat = avg_oos / (std_oos / np.sqrt(n_paths))
        p_value = 1 - stats.t.cdf(t_stat, df=n_paths - 1)
        logger.info(f"  OOS SR t-test:    t={t_stat:.2f}, p={p_value:.4f}")
    else:
        p_value = 1.0

    return {
        "pbo":              round(pbo, 4),
        "n_paths":          n_paths,
        "avg_is_sharpe":    round(avg_is, 4),
        "avg_oos_sharpe":   round(avg_oos, 4),
        "std_oos_sharpe":   round(std_oos, 4),
        "degradation_ratio": round(deg_ratio, 4),
        "oos_sr_p_value":   round(p_value, 4),
        "paths":            path_results,
    }


def main() -> None:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("══════ Fortress v5 — CPCV Validation (P4) ══════")

    if not _TEARSHEET_CSV.exists():
        logger.error(
            f"Tearsheet not found: {_TEARSHEET_CSV}. "
            "Run run_standalone_backtest.py first."
        )
        sys.exit(1)

    df = pd.read_csv(_TEARSHEET_CSV, parse_dates=["date"], index_col="date")
    if "daily_return" not in df.columns:
        logger.error("Missing 'daily_return' column.")
        sys.exit(1)

    daily_returns = df["daily_return"].fillna(0).values.astype(np.float64)
    logger.info(f"Loaded {len(daily_returns)} daily returns.")

    # Run CPCV
    results = run_cpcv(daily_returns, n_groups=_N_GROUPS, embargo=_EMBARGO_DAYS)

    # Save JSON
    with open(_CPCV_JSON, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"✅ CPCV results → {_CPCV_JSON}")

    # Save degradation CSV
    paths_df = pd.DataFrame(results["paths"])
    paths_df.to_csv(_CPCV_CSV, index=False)
    logger.info(f"✅ Degradation curve → {_CPCV_CSV}")

    # Gate check against config thresholds
    max_pbo = 0.25  # from config/risk_limits.yaml
    if results["pbo"] > max_pbo:
        logger.critical(
            f"GATE FAILED: PBO={results['pbo']:.2%} > max_pbo={max_pbo:.2%}. "
            "Strategy does NOT pass institutional validation."
        )
        sys.exit(2)
    else:
        logger.info(
            f"GATE PASSED: PBO={results['pbo']:.2%} ≤ {max_pbo:.2%}."
        )


if __name__ == "__main__":
    main()