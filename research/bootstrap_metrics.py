"""
FORTRESS v5 - bootstrap_metrics.py
Path: research/bootstrap_metrics.py

Stationary Block Bootstrap Confidence Intervals — Politis & Romano (1994).

Replaces naive i.i.d. bootstrap (which destroys the autocorrelation structure
of daily return series) with the stationary variant, which:
  1. Samples blocks of random geometrically-distributed length L ~ Geom(1/block_size).
  2. Wraps the series circularly so block draws at the tail can continue from
     the head — this ensures the bootstrap distribution is stationary.
  3. Tiles sampled blocks until the resampled series matches the original length T.

Why stationary block bootstrap over circular block bootstrap (Politis & Romano 1992)?
  - Circular block: fixed block length b → bootstrap distribution depends on b,
    which is not truly stationary because blocks at different offsets have
    different transition probabilities.
  - Stationary block: random L ~ Geom(1/b) → each bootstrap replicate is
    drawn from a stationary distribution, making inference asymptotically
    consistent even for strongly autocorrelated series.

Optimal block length selection: Hall, Horowitz & Jing (1995) / Politis & White (2004):
  b* = (4/3)^(1/3) * c^(1/3) * T^(1/3)
where c is an estimate of the long-run variance. We provide `optimal_block_size()`
which implements the Politis–White (2004) automatic plug-in estimator.

Metrics computed with 95% CI:
  - Sharpe Ratio (annualised)
  - Sortino Ratio (annualised)
  - Calmar Ratio
  - CAGR
  - Maximum Drawdown
  - CVaR-95
  - Hit Rate
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("BootstrapMetrics")

# ── Constants ─────────────────────────────────────────────────────────────────
_RF_DAILY:          float = 0.05 / 252
_N_BOOTSTRAP:       int   = 10_000
_DEFAULT_BLOCK:     int   = 21          # ~1 trading month default
_CI_ALPHA:          float = 0.05        # 95% CI (two-sided)
_MIN_BLOCK:         int   = 5
_MAX_BLOCK:         int   = 63


@dataclass
class BootstrapCI:
    """Confidence interval for a single metric."""
    metric:  str
    point:   float          # Point estimate from the original sample
    lower:   float          # CI lower bound (α/2 percentile)
    upper:   float          # CI upper bound (1 - α/2 percentile)
    std_err: float          # Bootstrap standard error
    n_boot:  int


@dataclass
class BootstrapReport:
    """Full bootstrap report across all metrics."""
    n_observations: int
    block_size:     int
    n_bootstrap:    int
    alpha:          float
    metrics:        Dict[str, BootstrapCI] = field(default_factory=dict)

    def summary_df(self) -> pd.DataFrame:
        rows = []
        for name, ci in self.metrics.items():
            rows.append({
                "Metric":    name,
                "Point Est": ci.point,
                "CI Lower":  ci.lower,
                "CI Upper":  ci.upper,
                "Std Error": ci.std_err,
            })
        return pd.DataFrame(rows).set_index("Metric")


# ─────────────────────────────────────────────────────────────────────────────
# Optimal block size estimator
# ─────────────────────────────────────────────────────────────────────────────

def optimal_block_size(
    returns: np.ndarray,
    max_lag: Optional[int] = None,
) -> int:
    """
    Politis & White (2004) automatic plug-in block size estimator.

    Estimates the optimal stationary block bootstrap block size b* via:
        b* = (4/3)^(1/3) * (2 * Ĝ²/D̂)^(1/3) * T^(1/3)

    where Ĝ = Σ_{k=-∞}^{∞} |k| γ(k),  D̂ = Σ_{k=-∞}^{∞} γ(k)
    and γ(k) is the sample autocovariance at lag k.

    The autocovariance sum is truncated at lag M using Bartlett kernel tapering
    to avoid noise amplification at large lags:
        M = max(1, floor(2 * T^(1/3)))

    Args:
        returns: Array of daily returns, shape (T,).
        max_lag: Manual lag truncation override. If None, uses 2*T^(1/3).

    Returns:
        b*: Integer block size, clipped to [_MIN_BLOCK, _MAX_BLOCK].
    """
    T = len(returns)
    if T < 10:
        return _DEFAULT_BLOCK

    M = max_lag if max_lag is not None else max(1, int(2 * T ** (1 / 3)))
    r = returns - returns.mean()

    # Bartlett-tapered autocovariance sum
    G_hat = 0.0   # Σ |k| * γ(k)
    D_hat = float(np.var(r))  # γ(0)

    for k in range(1, M + 1):
        gamma_k = float(np.mean(r[:T - k] * r[k:]))
        weight  = 1.0 - k / M           # Bartlett taper
        G_hat  += weight * abs(k) * gamma_k
        D_hat  += weight * 2.0 * gamma_k

    D_hat = max(D_hat, 1e-12)
    G_hat = max(abs(G_hat), 1e-12)

    # Plug-in formula
    b_star = (4 / 3) ** (1 / 3) * (2.0 * G_hat ** 2 / D_hat) ** (1 / 3) * T ** (1 / 3)
    b_star = int(np.round(b_star))
    return int(np.clip(b_star, _MIN_BLOCK, _MAX_BLOCK))


# ─────────────────────────────────────────────────────────────────────────────
# Stationary Block Bootstrap resampler
# ─────────────────────────────────────────────────────────────────────────────

class StationaryBlockBootstrap:
    """
    Politis & Romano (1994) stationary block bootstrap.

    Block lengths are drawn i.i.d. from Geom(1/block_size) to preserve
    the stationarity of the resampled series. Circular wrap-around ensures
    blocks starting near the end of the series can wrap to the beginning.

    Args:
        returns:    Daily return array, shape (T,).
        block_size: Mean block length b. Defaults to optimal_block_size().
        seed:       Optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        returns:    np.ndarray,
        block_size: Optional[int] = None,
        seed:       Optional[int] = 42,
    ) -> None:
        self.returns    = returns.astype(np.float64)
        self.T          = len(returns)
        self.block_size = block_size if block_size is not None else optimal_block_size(returns)
        self.rng        = np.random.default_rng(seed)
        logger.info(
            f"StationaryBlockBootstrap: T={self.T}, block_size={self.block_size}"
        )

    def resample(self) -> np.ndarray:
        """
        Draw one bootstrap replicate of length T.

        Algorithm:
          1. Draw starting index s ~ Uniform{0, ..., T-1}.
          2. Draw block length L ~ Geom(1/b).  E[L] = b.
          3. Collect returns[s], returns[s+1 mod T], ..., returns[s+L-1 mod T].
          4. Repeat from step 1 until accumulated length ≥ T.
          5. Truncate to exactly T observations.

        Returns:
            Resampled return array, shape (T,).
        """
        p          = 1.0 / self.block_size
        resampled  = np.empty(self.T, dtype=np.float64)
        filled     = 0

        while filled < self.T:
            start   = self.rng.integers(0, self.T)
            # Geometric(p) draw: L = 1 + number of failures before first success
            L       = int(self.rng.geometric(p))
            for j in range(L):
                if filled >= self.T:
                    break
                resampled[filled] = self.returns[(start + j) % self.T]
                filled += 1

        return resampled

    def bootstrap_distribution(
        self,
        stat_fn:  Callable[[np.ndarray], float],
        n_boot:   int = _N_BOOTSTRAP,
    ) -> np.ndarray:
        """
        Compute the bootstrap distribution of `stat_fn` over n_boot replicates.

        Args:
            stat_fn: Function mapping return array → scalar statistic.
            n_boot:  Number of bootstrap replicates.

        Returns:
            Array of shape (n_boot,) with bootstrapped statistic values.
        """
        dist = np.empty(n_boot, dtype=np.float64)
        for i in range(n_boot):
            dist[i] = stat_fn(self.resample())
        return dist

    def confidence_interval(
        self,
        stat_fn:  Callable[[np.ndarray], float],
        name:     str,
        n_boot:   int  = _N_BOOTSTRAP,
        alpha:    float = _CI_ALPHA,
    ) -> BootstrapCI:
        """
        Compute a percentile bootstrap CI for a scalar statistic.

        Uses the basic/percentile bootstrap (Hall 1992): the CI is the
        [α/2, 1-α/2] quantile of the bootstrap distribution of stat_fn.

        Note: We use the percentile method (not the studentized pivot) because
        financial return distributions are heavy-tailed. The studentized pivot
        requires variance estimation of each bootstrap replicate and suffers
        from instability when the denominator (e.g. vol) approaches zero.

        Args:
            stat_fn: Callable(returns: np.ndarray) → float.
            name:    Human-readable metric name.
            n_boot:  Bootstrap replicates.
            alpha:   Significance level (default 0.05 → 95% CI).

        Returns:
            BootstrapCI dataclass.
        """
        point = stat_fn(self.returns)
        dist  = self.bootstrap_distribution(stat_fn, n_boot)

        lower   = float(np.percentile(dist, 100 * alpha / 2))
        upper   = float(np.percentile(dist, 100 * (1 - alpha / 2)))
        std_err = float(np.std(dist, ddof=1))

        logger.debug(f"{name}: point={point:.4f}, 95% CI=[{lower:.4f}, {upper:.4f}]")

        return BootstrapCI(
            metric=name, point=point, lower=lower, upper=upper,
            std_err=std_err, n_boot=n_boot,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Metric functions
# ─────────────────────────────────────────────────────────────────────────────

def _annualised_sharpe(returns: np.ndarray, rf: float = _RF_DAILY) -> float:
    excess = returns - rf
    std    = float(np.std(excess, ddof=1))
    return float((np.mean(excess) / std) * np.sqrt(252)) if std > 1e-9 else 0.0


def _annualised_sortino(returns: np.ndarray, rf: float = _RF_DAILY) -> float:
    excess   = returns - rf
    downside = returns[returns < 0.0]
    dsd      = float(np.sqrt(np.mean(downside ** 2))) * np.sqrt(252) if len(downside) > 0 else 1e-9
    return float(np.mean(excess) * 252 / dsd) if dsd > 1e-9 else 0.0


def _max_drawdown(returns: np.ndarray) -> float:
    """Max drawdown from daily return array (negative value)."""
    cum = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(cum)
    dd   = (cum - peak) / peak
    return float(np.min(dd))


def _calmar(returns: np.ndarray) -> float:
    T    = len(returns)
    cagr = float(np.prod(1.0 + returns) ** (252.0 / max(T, 1)) - 1.0)
    mdd  = abs(_max_drawdown(returns))
    return cagr / mdd if mdd > 1e-9 else 0.0


def _cagr(returns: np.ndarray) -> float:
    T = len(returns)
    return float(np.prod(1.0 + returns) ** (252.0 / max(T, 1)) - 1.0)


def _cvar_95(returns: np.ndarray) -> float:
    var = float(np.percentile(returns, 5))
    tail = returns[returns <= var]
    return float(np.mean(tail)) if len(tail) > 0 else var


def _hit_rate(returns: np.ndarray) -> float:
    return float(np.mean(returns > 0.0))


# ─────────────────────────────────────────────────────────────────────────────
# Full report builder
# ─────────────────────────────────────────────────────────────────────────────

# Registry: metric name → callable(np.ndarray) → float
_METRIC_REGISTRY: Dict[str, Callable[[np.ndarray], float]] = {
    "Sharpe Ratio":     _annualised_sharpe,
    "Sortino Ratio":    _annualised_sortino,
    "Calmar Ratio":     _calmar,
    "CAGR":             _cagr,
    "Max Drawdown":     _max_drawdown,
    "CVaR-95":          _cvar_95,
    "Hit Rate":         _hit_rate,
}


def compute_bootstrap_report(
    returns:    np.ndarray,
    block_size: Optional[int] = None,
    n_boot:     int   = _N_BOOTSTRAP,
    alpha:      float = _CI_ALPHA,
    seed:       int   = 42,
    metrics:    Optional[List[str]] = None,
) -> BootstrapReport:
    """
    Run the full stationary block bootstrap for all registered metrics.

    Args:
        returns:    Daily return array.
        block_size: Mean block length. Auto-selected via Politis-White if None.
        n_boot:     Bootstrap replicates (10,000 gives <0.5% CI Monte Carlo error).
        alpha:      CI significance level.
        seed:       RNG seed.
        metrics:    Subset of metric names to compute. None → all registered.

    Returns:
        BootstrapReport with point estimates and 95% CIs for each metric.
    """
    b = block_size if block_size is not None else optimal_block_size(returns)
    bbs = StationaryBlockBootstrap(returns, block_size=b, seed=seed)

    selected = metrics if metrics is not None else list(_METRIC_REGISTRY.keys())
    report   = BootstrapReport(
        n_observations=len(returns),
        block_size=b,
        n_bootstrap=n_boot,
        alpha=alpha,
    )

    for name in selected:
        if name not in _METRIC_REGISTRY:
            logger.warning(f"Unknown metric '{name}'. Skipping.")
            continue
        fn = _METRIC_REGISTRY[name]
        ci = bbs.confidence_interval(fn, name, n_boot=n_boot, alpha=alpha)
        report.metrics[name] = ci

    return report


def log_report(report: BootstrapReport) -> None:
    """Pretty-print a BootstrapReport to the standard logger."""
    ci_pct  = int(round((1.0 - report.alpha) * 100))
    logger.info("═" * 72)
    logger.info(
        f"STATIONARY BLOCK BOOTSTRAP  —  "
        f"n={report.n_observations:,}  |  b={report.block_size}  |  "
        f"B={report.n_bootstrap:,}  |  {ci_pct}% CI"
    )
    logger.info(f"  {'Metric':<22} {'Point':>10} {'Lower':>10} {'Upper':>10} {'SE':>10}")
    logger.info("─" * 72)
    for name, ci in report.metrics.items():
        fmt = ".2%" if name in ("CAGR", "Max Drawdown", "CVaR-95", "Hit Rate") else ".4f"
        logger.info(
            f"  {name:<22} {ci.point:{fmt}!s:>10} "
            f"{ci.lower:{fmt}!s:>10} {ci.upper:{fmt}!s:>10} "
            f"{ci.std_err:{fmt}!s:>10}"
        )
    logger.info("═" * 72)


# ─────────────────────────────────────────────────────────────────────────────
# Standalone runner
# ─────────────────────────────────────────────────────────────────────────────

def run_bootstrap_on_tearsheet(
    tearsheet_path: str = "research/outputs/backtest_tearsheet.csv",
    output_path:    str = "research/outputs/bootstrap_ci.csv",
    n_boot:         int = _N_BOOTSTRAP,
) -> BootstrapReport:
    """
    Compute stationary block bootstrap CIs from an existing tearsheet CSV.
    Intended to be called from run_all.sh after backtest_engine.py.
    """
    df = pd.read_csv(tearsheet_path, index_col=0, parse_dates=True)

    if "daily_return" not in df.columns:
        raise ValueError("Tearsheet must contain 'daily_return' column.")

    returns = df["daily_return"].dropna().values.astype(np.float64)

    b = optimal_block_size(returns)
    logger.info(f"Politis-White optimal block size: {b} days")

    report = compute_bootstrap_report(returns, block_size=b, n_boot=n_boot)
    log_report(report)

    report.summary_df().to_csv(output_path)
    logger.info(f"✅ Bootstrap CI report saved → {output_path}")

    return report


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    run_bootstrap_on_tearsheet()