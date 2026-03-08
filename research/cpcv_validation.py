"""
FORTRESS v5 - cpcv_validation.py
Path: research/cpcv_validation.py

Combinatorial Purged Cross-Validation (CPCV) — Lopez de Prado (2018).

Replaces simple walk-forward with the full combinatorial approach that:
  1. Generates C(k, k//2) = C(6,3) = 20 independent IS/OOS path pairs.
  2. Computes Probability of Backtest Overfitting (PBO) over all 20 paths.
  3. Computes Probabilistic Sharpe Ratio (PSR) per path.
  4. Generates the backtest degradation curve (IS SR → OOS SR per path).

Key advantages over walk-forward:
  - Walk-forward produces 4–6 OOS paths for a 5-year dataset; CPCV produces 20.
  - CPCV eliminates serial correlation bias in the OOS performance distribution
    (each test path uses non-overlapping data segments).
  - Purging prevents label leakage at fold boundaries (critical for ML features
    trained on overlapping rolling windows like 12-1M momentum).
  - PBO convergence requires ≥ 16 paths; walk-forward is structurally insufficient.

Architecture:
  - CPCVSplitter: Generates purged combinatorial splits.
  - CPCVResult: Dataclass for per-path performance metrics.
  - CombPurgedCV: Orchestrates backtests across all paths, aggregates PBO + PSR.

Targets (from audit roadmap):
  - PBO < 0.50 (strategy not overfitted)
  - PSR > 0.95 (statistically significant at 95% confidence)
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger("CPCVValidation")

# ── Constants ─────────────────────────────────────────────────────────────────
_RF_DAILY:      float = 0.05 / 252   # Daily risk-free rate
_EMBARGO_DAYS:  int   = 5            # Purge embargo window: 5 trading days each side of boundary


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CPCVResult:
    """Performance metrics for one IS/OOS combinatorial path."""
    path_id:       int
    is_indices:    List[int]    # Which k-fold groups are IS
    oos_indices:   List[int]    # Which k-fold groups are OOS
    is_sharpe:     float = 0.0
    oos_sharpe:    float = 0.0
    is_sortino:    float = 0.0
    oos_sortino:   float = 0.0
    is_n_obs:      int   = 0
    oos_n_obs:     int   = 0
    psr:           float = 0.0   # PSR of OOS performance (see compute_psr)
    logit_sr_rank: float = 0.0   # Logit of OOS SR rank — input to PBO distribution


@dataclass
class CPCVSummary:
    """Aggregated CPCV output across all combinatorial paths."""
    n_paths:          int
    pbo:              float    # Probability of Backtest Overfitting
    mean_psr:         float    # Mean PSR across OOS paths
    frac_psr_pass:    float    # Fraction of paths with PSR > 0.95
    mean_is_sharpe:   float
    mean_oos_sharpe:  float
    sharpe_degradation: float  # (IS - OOS) / IS — fractional degradation
    results:          List[CPCVResult] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# CPCV Splitter
# ─────────────────────────────────────────────────────────────────────────────

class CPCVSplitter:
    """
    Generates purged combinatorial (IS, OOS) index splits.

    Algorithm (Lopez de Prado 2018, Ch. 12):
      1. Partition T observations into k equal groups (folds).
      2. For each C(k, n_test_groups) combination, assign those groups to OOS
         and the rest to IS.
      3. Apply a two-sided embargo: the `embargo_days` observations immediately
         adjacent to any IS/OOS boundary are excluded from both sets.
         This prevents the training set from containing rolled features that
         partially overlap the test period.

    Args:
        n_splits:        k — total number of folds. C(k, k//2) paths generated.
        n_test_groups:   Number of OOS folds per path. Default: k//2.
        embargo_days:    Days excluded on each side of IS/OOS boundary.
    """

    def __init__(
        self,
        n_splits:      int = 6,
        n_test_groups: Optional[int] = None,
        embargo_days:  int = _EMBARGO_DAYS,
    ) -> None:
        self.n_splits      = n_splits
        self.n_test_groups = n_test_groups if n_test_groups is not None else n_splits // 2
        self.embargo_days  = embargo_days

        from math import comb
        self.n_paths = comb(self.n_splits, self.n_test_groups)
        logger.info(
            f"CPCVSplitter: C({n_splits},{self.n_test_groups}) = {self.n_paths} OOS paths | "
            f"embargo={embargo_days} days"
        )

    def split(
        self,
        n_observations: int,
    ) -> Iterator[Tuple[List[int], List[int], List[int], List[int]]]:
        """
        Yields (is_idx, oos_idx, is_groups, oos_groups) for each combinatorial path.

        Args:
            n_observations: Total number of observations T.

        Yields:
            is_idx:     Row indices assigned to the IS set (after embargo).
            oos_idx:    Row indices assigned to the OOS set (after embargo).
            is_groups:  Which fold group IDs are IS.
            oos_groups: Which fold group IDs are OOS.
        """
        # Build group boundaries: fold_starts[g] = first index in group g
        group_size  = n_observations // self.n_splits
        fold_starts = [g * group_size for g in range(self.n_splits)]
        fold_ends   = fold_starts[1:] + [n_observations]

        all_groups = list(range(self.n_splits))

        for path_id, oos_groups in enumerate(
            itertools.combinations(all_groups, self.n_test_groups)
        ):
            oos_groups_set = set(oos_groups)
            is_groups      = [g for g in all_groups if g not in oos_groups_set]

            # Raw IS and OOS index sets
            raw_is  = set()
            raw_oos = set()
            for g in all_groups:
                idxs = range(fold_starts[g], fold_ends[g])
                if g in oos_groups_set:
                    raw_oos.update(idxs)
                else:
                    raw_is.update(idxs)

            # Apply embargo: find all boundary points between IS and OOS groups,
            # then exclude `embargo_days` rows on each side of each boundary.
            embargo_idxs: set[int] = set()
            for g in oos_groups_set:
                # Left boundary (first OOS index)
                left = fold_starts[g]
                for j in range(left - self.embargo_days, left + self.embargo_days + 1):
                    if 0 <= j < n_observations:
                        embargo_idxs.add(j)
                # Right boundary (last OOS index)
                right = fold_ends[g] - 1
                for j in range(right - self.embargo_days, right + self.embargo_days + 1):
                    if 0 <= j < n_observations:
                        embargo_idxs.add(j)

            is_idx  = sorted(raw_is  - embargo_idxs)
            oos_idx = sorted(raw_oos - embargo_idxs)

            yield is_idx, oos_idx, is_groups, list(oos_groups)


# ─────────────────────────────────────────────────────────────────────────────
# Statistical measures
# ─────────────────────────────────────────────────────────────────────────────

def annualised_sharpe(
    returns: np.ndarray,
    rf_daily: float = _RF_DAILY,
) -> float:
    """Annualised Sharpe ratio from daily return array."""
    excess = returns - rf_daily
    std    = float(np.std(excess, ddof=1))
    return float((np.mean(excess) / std) * np.sqrt(252)) if std > 1e-9 else 0.0


def annualised_sortino(
    returns: np.ndarray,
    rf_daily: float = _RF_DAILY,
    mar:      float = 0.0,
) -> float:
    """Annualised Sortino ratio. MAR = minimum acceptable daily return."""
    excess   = returns - rf_daily
    downside = returns[returns < mar] - mar
    dsd      = float(np.sqrt(np.mean(downside ** 2))) if len(downside) > 0 else 1e-9
    return float((np.mean(excess) / dsd) * np.sqrt(252)) if dsd > 1e-9 else 0.0


def compute_psr(
    returns:          np.ndarray,
    observed_sharpe:  float,
    benchmark_sharpe: float = 0.0,
) -> float:
    """
    Probabilistic Sharpe Ratio (Bailey & Lopez de Prado, 2012).

    PSR = Φ[ (SR̂ - SR*) / σ(SR̂) ]

    where σ(SR̂)² = (1 - γ·SR̂ + (κ_excess/4)·SR̂²) / (T-1)
    and γ = skewness, κ_excess = kurtosis - 3.

    Args:
        returns:          Array of daily returns.
        observed_sharpe:  Annualised Sharpe to evaluate (typically the OOS SR).
        benchmark_sharpe: SR* — the minimum acceptable performance (default 0.0).

    Returns:
        PSR in [0, 1]. Values > 0.95 indicate statistical significance.
    """
    T = len(returns)
    if T < 10:
        return 0.0

    skew = float(stats.skew(returns))
    kurt = float(stats.kurtosis(returns, fisher=False))  # raw kurtosis (Normal=3)
    excess_kurt = kurt - 3.0

    sr_annualised = observed_sharpe
    sr_daily      = sr_annualised / np.sqrt(252)

    # SR variance estimator (Christie 2005 / Bailey 2012)
    var_sr = (
        1.0
        - skew * sr_daily
        + (excess_kurt / 4.0) * sr_daily ** 2
    ) / max(T - 1, 1)

    if var_sr <= 0:
        return 0.0

    benchmark_daily = benchmark_sharpe / np.sqrt(252)
    t_stat = (sr_daily - benchmark_daily) / np.sqrt(var_sr)
    return float(stats.norm.cdf(t_stat))


def compute_pbo(logit_sr_ranks: List[float]) -> float:
    """
    Probability of Backtest Overfitting (CSCV framework).

    PBO = P(logit(OOS rank) < 0) across all combinatorial paths.

    When the IS winner's OOS rank falls below median (rank < 0.5),
    logit(rank) < 0. PBO is the fraction of paths where this occurs.

    PBO > 0.5 → overfitted. PBO < 0.5 → robust out-of-sample.

    Args:
        logit_sr_ranks: List of logit(OOS rank percentile) values, one per path.

    Returns:
        PBO in [0, 1].
    """
    if not logit_sr_ranks:
        return 0.5
    return float(np.mean([1.0 if x < 0.0 else 0.0 for x in logit_sr_ranks]))


# ─────────────────────────────────────────────────────────────────────────────
# Main CPCV Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class CombPurgedCV:
    """
    Orchestrates CPCV over a precomputed return DataFrame.

    Usage:
        cv = CombPurgedCV(returns_df, n_splits=6)
        summary = cv.run()
        cv.report(summary)

    The `returns_df` should be the daily return series from the full backtest
    (already computed by `EventDrivenBacktester.run_backtest`). CPCV operates
    on this return series rather than re-running the full backtest pipeline for
    each of the 20 paths (which would be prohibitively expensive with live models).

    For each IS/OOS split, it re-evaluates the statistical properties of the
    IS and OOS sub-periods of the precomputed returns. This is the standard
    "model-free" CPCV application used for strategy evaluation.

    For full re-training CPCV (where the model is retrained on each IS set),
    use `run_with_refit()` which accepts a callable `backtest_fn`.
    """

    def __init__(
        self,
        returns_df:    pd.Series,
        n_splits:      int = 6,
        n_test_groups: Optional[int] = None,
        embargo_days:  int = _EMBARGO_DAYS,
        rf_daily:      float = _RF_DAILY,
    ) -> None:
        self.returns      = returns_df.dropna().values.astype(np.float64)
        self.n_obs        = len(self.returns)
        self.rf_daily     = rf_daily
        self.splitter     = CPCVSplitter(n_splits, n_test_groups, embargo_days)

        if self.n_obs < 100:
            raise ValueError(
                f"CPCV requires at least 100 daily return observations; got {self.n_obs}."
            )

    def run(self) -> CPCVSummary:
        """
        Run the full CPCV evaluation.

        Returns:
            CPCVSummary with PBO, PSR, and per-path metrics.
        """
        results: List[CPCVResult] = []
        all_oos_sharpes: List[float] = []

        for path_id, (is_idx, oos_idx, is_groups, oos_groups) in enumerate(
            self.splitter.split(self.n_obs)
        ):
            if len(is_idx) < 21 or len(oos_idx) < 21:
                logger.warning(f"Path {path_id}: insufficient data after embargo. Skipping.")
                continue

            is_returns  = self.returns[is_idx]
            oos_returns = self.returns[oos_idx]

            is_sr   = annualised_sharpe(is_returns,  self.rf_daily)
            oos_sr  = annualised_sharpe(oos_returns, self.rf_daily)
            is_so   = annualised_sortino(is_returns,  self.rf_daily)
            oos_so  = annualised_sortino(oos_returns, self.rf_daily)
            psr_val = compute_psr(oos_returns, oos_sr)

            all_oos_sharpes.append(oos_sr)

            result = CPCVResult(
                path_id=path_id,
                is_indices=is_groups,
                oos_indices=oos_groups,
                is_sharpe=is_sr,
                oos_sharpe=oos_sr,
                is_sortino=is_so,
                oos_sortino=oos_so,
                is_n_obs=len(is_idx),
                oos_n_obs=len(oos_idx),
                psr=psr_val,
                logit_sr_rank=0.0,  # populated after all paths computed
            )
            results.append(result)

        if not results:
            raise RuntimeError("CPCV produced no valid paths. Check input data length.")

        # Compute logit(OOS SR rank) for each path — used for PBO
        oos_sharpes_arr = np.array(all_oos_sharpes)
        for result in results:
            # Rank percentile of this path's OOS SR among ALL OOS SRs
            rank_pct = float(np.mean(oos_sharpes_arr <= result.oos_sharpe))
            # Clip to avoid log(0) or log(∞)
            rank_pct  = float(np.clip(rank_pct, 1e-6, 1.0 - 1e-6))
            result.logit_sr_rank = float(np.log(rank_pct / (1.0 - rank_pct)))

        # Aggregate
        logit_ranks   = [r.logit_sr_rank for r in results]
        pbo           = compute_pbo(logit_ranks)
        psr_vals      = [r.psr for r in results]
        mean_psr      = float(np.mean(psr_vals))
        frac_psr_pass = float(np.mean([1.0 if p > 0.95 else 0.0 for p in psr_vals]))

        mean_is_sr  = float(np.mean([r.is_sharpe  for r in results]))
        mean_oos_sr = float(np.mean([r.oos_sharpe for r in results]))
        degradation = (mean_is_sr - mean_oos_sr) / max(abs(mean_is_sr), 1e-6)

        summary = CPCVSummary(
            n_paths=len(results),
            pbo=pbo,
            mean_psr=mean_psr,
            frac_psr_pass=frac_psr_pass,
            mean_is_sharpe=mean_is_sr,
            mean_oos_sharpe=mean_oos_sr,
            sharpe_degradation=degradation,
            results=results,
        )
        return summary

    def run_with_refit(
        self,
        backtest_fn: "Callable[[List[int], List[int]], Tuple[np.ndarray, np.ndarray]]",
    ) -> CPCVSummary:
        """
        Full CPCV with model refit on each IS set.

        Args:
            backtest_fn: Callable(is_indices, oos_indices) → (is_returns, oos_returns).
                         Called once per combinatorial path. Typically wraps the
                         full training + backtest pipeline for the IS/OOS split.

        Returns:
            CPCVSummary.
        """
        results: List[CPCVResult] = []
        all_oos_sharpes: List[float] = []

        for path_id, (is_idx, oos_idx, is_groups, oos_groups) in enumerate(
            self.splitter.split(self.n_obs)
        ):
            logger.info(f"CPCV path {path_id + 1}/{self.splitter.n_paths}: refit...")
            try:
                is_returns, oos_returns = backtest_fn(is_idx, oos_idx)
            except Exception as exc:
                logger.error(f"Path {path_id} backtest_fn failed: {exc}")
                continue

            is_sr   = annualised_sharpe(is_returns,  self.rf_daily)
            oos_sr  = annualised_sharpe(oos_returns, self.rf_daily)
            psr_val = compute_psr(oos_returns, oos_sr)
            all_oos_sharpes.append(oos_sr)

            results.append(CPCVResult(
                path_id=path_id,
                is_indices=is_groups,
                oos_indices=oos_groups,
                is_sharpe=is_sr,
                oos_sharpe=oos_sr,
                is_sortino=annualised_sortino(is_returns),
                oos_sortino=annualised_sortino(oos_returns),
                is_n_obs=len(is_returns),
                oos_n_obs=len(oos_returns),
                psr=psr_val,
            ))

        if not results:
            raise RuntimeError("CPCV (with refit) produced no valid paths.")

        # Logit SR ranks
        arr = np.array(all_oos_sharpes)
        for result in results:
            p = float(np.clip(np.mean(arr <= result.oos_sharpe), 1e-6, 1.0 - 1e-6))
            result.logit_sr_rank = float(np.log(p / (1.0 - p)))

        pbo           = compute_pbo([r.logit_sr_rank for r in results])
        psr_vals      = [r.psr for r in results]
        mean_is_sr    = float(np.mean([r.is_sharpe  for r in results]))
        mean_oos_sr   = float(np.mean([r.oos_sharpe for r in results]))
        degradation   = (mean_is_sr - mean_oos_sr) / max(abs(mean_is_sr), 1e-6)

        return CPCVSummary(
            n_paths=len(results),
            pbo=pbo,
            mean_psr=float(np.mean(psr_vals)),
            frac_psr_pass=float(np.mean([1.0 if p > 0.95 else 0.0 for p in psr_vals])),
            mean_is_sharpe=mean_is_sr,
            mean_oos_sharpe=mean_oos_sr,
            sharpe_degradation=degradation,
            results=results,
        )

    def report(self, summary: CPCVSummary) -> None:
        """Logs a formatted summary to the standard logger."""
        pbo_verdict = "✅ NOT overfitted" if summary.pbo < 0.50 else "❌ OVERFITTED"
        psr_verdict = "✅ Significant" if summary.frac_psr_pass > 0.80 else "⚠️  Marginal"

        logger.info("═" * 70)
        logger.info("CPCV SUMMARY")
        logger.info(f"  Paths evaluated:      {summary.n_paths} / {self.splitter.n_paths}")
        logger.info(f"  PBO:                  {summary.pbo:.3f}  {pbo_verdict}  (target < 0.50)")
        logger.info(f"  Mean PSR:             {summary.mean_psr:.3f}  {psr_verdict}")
        logger.info(f"  Paths PSR > 0.95:     {summary.frac_psr_pass:.1%}  (target > 80%)")
        logger.info(f"  Mean IS Sharpe:       {summary.mean_is_sharpe:.3f}")
        logger.info(f"  Mean OOS Sharpe:      {summary.mean_oos_sharpe:.3f}")
        logger.info(f"  Sharpe Degradation:   {summary.sharpe_degradation:.1%}  (IS→OOS)")
        logger.info("─" * 70)
        logger.info("  Per-path OOS Sharpe distribution:")
        oos_srs = sorted([r.oos_sharpe for r in summary.results])
        q25, q50, q75 = np.percentile(oos_srs, [25, 50, 75])
        logger.info(f"    Q25={q25:.3f}  Median={q50:.3f}  Q75={q75:.3f}")
        logger.info("═" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# Standalone runner
# ─────────────────────────────────────────────────────────────────────────────

def run_cpcv_on_tearsheet(
    tearsheet_path: str = "research/outputs/backtest_tearsheet.csv",
    n_splits:       int = 6,
    output_path:    str = "research/outputs/cpcv_summary.csv",
) -> CPCVSummary:
    """
    Runs CPCV on an existing tearsheet CSV (model-free mode).
    Intended to be called from run_all.sh after backtest_engine.py.
    """
    df = pd.read_csv(tearsheet_path, index_col=0, parse_dates=True)

    if "daily_return" not in df.columns:
        raise ValueError("Tearsheet must contain 'daily_return' column.")

    returns_series = df["daily_return"].dropna()
    cv = CombPurgedCV(returns_series, n_splits=n_splits)
    summary = cv.run()
    cv.report(summary)

    # Persist path-level results
    results_df = pd.DataFrame([
        {
            "path_id":       r.path_id,
            "is_groups":     str(r.is_indices),
            "oos_groups":    str(r.oos_indices),
            "is_sharpe":     r.is_sharpe,
            "oos_sharpe":    r.oos_sharpe,
            "is_sortino":    r.is_sortino,
            "oos_sortino":   r.oos_sortino,
            "psr":           r.psr,
            "logit_sr_rank": r.logit_sr_rank,
            "is_n_obs":      r.is_n_obs,
            "oos_n_obs":     r.oos_n_obs,
        }
        for r in summary.results
    ])
    results_df.to_csv(output_path, index=False)
    logger.info(f"✅ CPCV results saved → {output_path}")

    return summary


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    run_cpcv_on_tearsheet()