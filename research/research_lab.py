"""
FORTRESS v5 - research_lab.py  [PRODUCTION REWRITE]
Path: research/research_lab.py

Institutional Statistical Rigor.

AUDIT FIXES:
  BUG #9 (DSR):  Used raw kurtosis κ in the SR variance formula.
                 Bailey & Lopez de Prado (2012, p.7) require EXCESS kurtosis (κ - 3)
                 because they assume a Normal baseline (κ_Normal = 3).
                 Using κ instead of κ-3 systematically understates SR variance,
                 producing an overconfident (inflated) DSR.
                 Fix: formula now uses (kurtosis - 3) as the excess kurtosis term.

  BUG #10 (PBO): calculate_pbo() always returned a hardcoded 0.15 constant.
                 Replaced with full CSCV (Combinatorially Symmetric Cross-Validation)
                 as described in Bailey et al. (2014), "The Probability of Backtest
                 Overfitting". Requires a (T × S) return matrix where T = trading days,
                 S = number of strategy variants or walk-forward folds.
"""

from __future__ import annotations

import logging
from itertools import combinations
from typing import List, Optional

import numpy as np
import scipy.stats as stats

logger = logging.getLogger("ResearchLab")

# ── Bailey & Lopez de Prado (2012): Deflated Sharpe Ratio ─────────────────────


def calculate_dsr(
    trials_history: List[float],
    current_sharpe: float,
    benchmark_sharpe: float = 0.0,
    skewness: float = 0.0,
    kurtosis: float = 3.0,      # CONVENTION: raw (not excess) kurtosis. Normal = 3.
    num_observations: int = 252,
) -> float:
    """
    Deflated Sharpe Ratio (DSR) — Bailey & Lopez de Prado (2012).

    Adjusts the Sharpe Ratio for:
      1. Non-Normality of returns (via skewness & excess kurtosis).
      2. Multiple testing inflation (via the expected maximum Sharpe across all trials).

    The SR variance estimator (Equation 7 in Bailey 2012) is:
        Var(SR) = [1 - γ · SR + ((κ_excess)/4) · SR²] / (T - 1)
    where:
        γ = skewness of returns
        κ_excess = kurtosis - 3  (excess kurtosis; Normal baseline = 0)
        T = number of observations

    AUDIT FIX #9: The original implementation used raw κ instead of (κ - 3).
    For a Normal distribution, raw κ = 3, so the original formula added
    (3/4) * SR² rather than 0 — systematically underestimating SR variance.

    Args:
        trials_history:     Sharpe ratios from all PREVIOUS strategy trials / folds.
                            Pass [] for the very first trial (no multi-testing correction).
        current_sharpe:     Sharpe of the strategy being evaluated.
        benchmark_sharpe:   Minimum acceptable Sharpe (typically 0.0 or risk-free proxy).
        skewness:           Third standardised moment of the strategy's returns.
        kurtosis:           RAW fourth standardised moment (Normal = 3.0).
                            Excess kurtosis = kurtosis - 3 is computed internally.
        num_observations:   Number of return observations (trading days).

    Returns:
        dsr: P-value in [0, 1]. Values > 0.95 indicate the strategy's Sharpe
             is statistically significant even after multi-testing deflation.
    """
    # ── 1. Expected maximum Sharpe across all prior trials ────────────────────
    if not trials_history:
        # No prior trials: benchmark is the user-defined minimum acceptable Sharpe
        expected_max_sr = benchmark_sharpe
    else:
        n_trials = len(trials_history)
        mu_sr    = np.mean(trials_history)
        std_sr   = np.std(trials_history)

        # Euler–Mascheroni constant (γ ≈ 0.5772)
        em_const = 0.5772156649

        # Expected maximum of N i.i.d. Normal RVs — asymptotic approximation
        # (Equation 10 in Bailey & Lopez de Prado 2014)
        expected_max_sr = mu_sr + std_sr * (
            (1 - em_const) * stats.norm.ppf(1 - 1.0 / n_trials)
            + em_const      * stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
        )

    # ── 2. SR variance: AUDIT FIX #9 — use excess kurtosis ──────────────────
    excess_kurtosis = kurtosis - 3.0   # Normal excess κ = 0; fat-tails > 0
    sr_variance = (
        1.0
        - skewness * current_sharpe
        + (excess_kurtosis / 4.0) * (current_sharpe ** 2)
    ) / max(num_observations - 1, 1)

    if sr_variance <= 0:
        return 0.0

    # ── 3. Standardise and compute P-value ───────────────────────────────────
    t_stat = (current_sharpe - expected_max_sr) / np.sqrt(sr_variance)
    return float(stats.norm.cdf(t_stat))


# ── Bailey et al. (2014): Probability of Backtest Overfitting (CSCV) ──────────


def calculate_pbo(
    matrix_of_returns: Optional[np.ndarray] = None,
    n_combinations: int = 16,
    risk_free_daily: float = 0.05 / 252,
) -> float:
    """
    Probability of Backtest Overfitting (PBO) via Combinatorially Symmetric
    Cross-Validation (CSCV) — Bailey, Borwein, Lopez de Prado & Zhu (2014).

    Algorithm:
      1. Split the (T × S) return matrix into S equal sub-periods along axis 0.
      2. For each C(S, S/2) IS/OOS partition:
           a. IS Sharpe of each strategy variant (column) → pick the IS winner.
           b. Map the IS winner's index → OOS Sharpe of the same variant.
           c. Compute logit of the IS winner's OOS rank percentile among all OOS Sharpes.
      3. PBO = P(logit < 0) = fraction of partitions where the IS winner
         underperforms the median OOS strategy.

    A PBO > 0.5 signals the backtest is more likely overfitted than not.

    Args:
        matrix_of_returns: Shape (T, S) where T = trading days, S = number of
                           strategy variants or walk-forward fold OOS return arrays.
                           If None or too small, returns a 0.15 conservative prior.
        n_combinations:    Number of C(S, S/2) partitions to evaluate.
                           Capped for computational tractability.
        risk_free_daily:   Daily risk-free rate for Sharpe computation.

    Returns:
        pbo: Probability of backtest overfitting in [0, 1].
    """
    # ── Input validation ──────────────────────────────────────────────────────
    if matrix_of_returns is None or matrix_of_returns.ndim != 2:
        logger.warning("PBO: invalid input. Returning conservative prior 0.15.")
        return 0.15

    T, S = matrix_of_returns.shape
    if T < 50 or S < 2:
        logger.warning(
            f"PBO: insufficient data ({T} days, {S} variants). "
            "Returning conservative prior 0.15."
        )
        return 0.15

    # ── 1. Split T observations into S equal sub-periods ─────────────────────
    # Truncate T to be divisible by S
    T_trunc = (T // S) * S
    R = matrix_of_returns[:T_trunc, :]          # (T_trunc, S)
    sub_period_len = T_trunc // S

    sub_periods: list = []
    for i in range(S):
        sub_periods.append(R[i * sub_period_len : (i + 1) * sub_period_len, :])

    # ── 2. Enumerate C(S, S/2) IS/OOS splits ─────────────────────────────────
    half = S // 2
    all_combos = list(combinations(range(S), half))

    # Cap for speed — randomly sample if too many
    rng = np.random.default_rng(seed=42)
    if len(all_combos) > n_combinations:
        idx = rng.choice(len(all_combos), size=n_combinations, replace=False)
        all_combos = [all_combos[i] for i in idx]

    logits: list[float] = []

    for is_indices in all_combos:
        oos_indices = tuple(i for i in range(S) if i not in is_indices)

        # IS returns: concatenate the chosen sub-periods
        is_returns  = np.vstack([sub_periods[i] for i in is_indices])   # (IS_T, S)
        oos_returns = np.vstack([sub_periods[i] for i in oos_indices])  # (OOS_T, S)

        # Sharpe per strategy variant on IS and OOS
        is_sharpes  = _compute_sharpe_per_column(is_returns,  risk_free_daily)
        oos_sharpes = _compute_sharpe_per_column(oos_returns, risk_free_daily)

        # IS winner: strategy variant with the highest IS Sharpe
        is_winner_idx = int(np.argmax(is_sharpes))

        # OOS rank of the IS winner (percentile among all OOS Sharpes)
        is_winner_oos_sharpe = oos_sharpes[is_winner_idx]
        oos_rank_pct = float(np.mean(oos_sharpes <= is_winner_oos_sharpe))

        # Logit transform: negative when IS winner is below OOS median
        # Clip to avoid log(0) / log(1)
        oos_rank_clipped = float(np.clip(oos_rank_pct, 1e-6, 1.0 - 1e-6))
        logit = float(np.log(oos_rank_clipped / (1.0 - oos_rank_clipped)))
        logits.append(logit)

    if not logits:
        return 0.15

    # PBO = fraction of partitions where IS winner underperforms OOS median
    pbo = float(np.mean(np.array(logits) < 0.0))
    logger.info(f"PBO (CSCV, {len(logits)} partitions): {pbo:.4f}")
    return pbo


def _compute_sharpe_per_column(
    return_matrix: np.ndarray,
    risk_free_daily: float,
) -> np.ndarray:
    """
    Computes annualised Sharpe ratio for each column (strategy variant).

    Args:
        return_matrix: Shape (T, S).
        risk_free_daily: Daily risk-free rate.

    Returns:
        sharpes: Shape (S,).
    """
    excess = return_matrix - risk_free_daily       # (T, S)
    mean   = excess.mean(axis=0)                  # (S,)
    std    = excess.std(axis=0)                   # (S,)
    # Avoid division by zero for flat-return strategies
    std    = np.where(std == 0, 1e-8, std)
    return (mean / std) * np.sqrt(252)