"""
FORTRESS v5 - strategy_health.py  [PRODUCTION REWRITE]
Path: monitoring/strategy_health.py

Bayesian Strategy Health Monitor.

AUDIT FIX (Round 2):
  BUG #HEALTH-1 (STATELESS POSTERIOR):
    The `update_posterior()` method computed a posterior correctly on a single
    call, but the updated hyper-parameters (kappa_n, alpha_n, beta_n, mu_n)
    were assigned to local variables and DISCARDED. On every subsequent call
    to `evaluate_health()`, the computation restarted from the original prior
    (mu_0, kappa_0, alpha_0, beta_0). The posterior never accumulated evidence.

    In practice: 200 live trading days of negative returns had zero effect on
    the decay probability reported, because the prior was fully reinitialised
    on each call. The health monitor was a dressed-up z-score.

    Fix: The conjugate prior hyper-parameters are now INSTANCE ATTRIBUTES that
    are mutated in-place by every call to `update_and_persist_posterior()`.
    The sequential Bayesian update now correctly accumulates all observations.

  BUG #HEALTH-2 (NO META-AGENT TRIGGER):
    `evaluate_health()` returned a string status ("RED", "YELLOW", "GREEN")
    but had NO mechanism to actually trigger the ConstitutionalMetaAgent.
    The organism had alpha decay detection but no response wiring.

    Fix: `evaluate_health()` now accepts an optional `meta_agent` parameter.
    On RED status, it calls `await meta_agent.analyze_and_evolve(...)` in an
    asyncio-safe way.

  BUG #HEALTH-3 (WRONG POSTERIOR MARGINAL):
    The decay probability calculation used the Normal CDF directly on the
    posterior mean, treating the posterior as if the variance were known.
    Under a Normal-Inverse-Gamma conjugate prior, the marginal distribution
    of μ is a Student's t-distribution with 2α_n degrees of freedom.

    Fix: `calculate_decay_probability()` now uses the Student-t CDF with
    the correct degrees of freedom: t ~ t(2α_n) centered at mu_n.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any, Optional, Tuple

import numpy as np
import scipy.stats as stats
import yaml

logger = logging.getLogger("BayesianHealthMonitor")


class BayesianHealthMonitor:
    """
    Sequential Bayesian health monitor using a Normal-Inverse-Gamma conjugate prior.

    Prior specification:
        μ | σ² ~ N(μ_0, σ²/κ_0)
        σ²      ~ InvGamma(α_0, β_0)

    After observing returns x_1, ..., x_n the conjugate posterior is:
        μ | σ², data ~ N(μ_n, σ²/κ_n)
        σ² | data    ~ InvGamma(α_n, β_n)
    where the update equations are given in `update_and_persist_posterior()`.

    The marginal posterior of μ (integrating out σ²) is a Student-t:
        μ | data ~ t(2α_n, μ_n, β_n / (α_n * κ_n))
    """

    def __init__(
        self,
        config_path: str = "config/risk_limits.yaml",
        backtest_tearsheet_path: Optional[str] = "research/outputs/backtest_tearsheet.csv",
    ):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        limits = config.get("health_monitor", {}) if config else {}
        self.yellow_alert_prob: float = limits.get("bayesian_yellow_alert_prob", 0.20)
        self.red_alert_prob:    float = limits.get("bayesian_red_alert_prob",    0.50)

        # ── Initialise prior from backtest tearsheet if available ─────────────
        prior_mu, prior_sigma = self._load_backtest_priors(backtest_tearsheet_path)
        logger.info(
            f"Prior initialised: μ₀={prior_mu:.6f} ({prior_mu*252:.2%} ann.), "
            f"σ₀={prior_sigma:.6f} ({prior_sigma*np.sqrt(252):.2%} ann.)"
        )

        # ── AUDIT FIX #HEALTH-1: Hyper-parameters as mutable instance state ──
        # These are updated in-place on every call to update_and_persist_posterior().
        # The prior is never reset between calls — this is intentional and correct.
        self.mu_0:    float = prior_mu
        self.kappa_0: float = 10.0      # Low confidence in prior (10 "virtual observations")
        self.alpha_0: float = 2.0       # Shape: > 1 ensures finite mean for InvGamma
        self.beta_0:  float = prior_sigma ** 2 * (self.alpha_0 - 1)  # Mode at prior_sigma²

        # Current posterior state (updated sequentially)
        self.mu_n:    float = self.mu_0
        self.kappa_n: float = self.kappa_0
        self.alpha_n: float = self.alpha_0
        self.beta_n:  float = self.beta_0
        self.n_total: int   = 0   # Total observations accumulated

    @staticmethod
    def _load_backtest_priors(
        tearsheet_path: Optional[str],
    ) -> Tuple[float, float]:
        """
        Loads the expected daily return and volatility from the backtest tearsheet.
        Falls back to conservative defaults if the tearsheet is unavailable.
        """
        defaults = (0.0004, 0.010)   # ~10% ann. return, ~15% ann. vol

        if tearsheet_path is None or not os.path.exists(tearsheet_path):
            logger.warning(
                "Backtest tearsheet not found. Using default priors "
                "(mu=0.0004/day, sigma=0.010/day)."
            )
            return defaults

        try:
            import pandas as pd
            df = pd.read_csv(tearsheet_path)
            if "daily_return" not in df.columns:
                return defaults
            daily_returns = df["daily_return"].dropna().values
            if len(daily_returns) < 50:
                return defaults
            return float(daily_returns.mean()), float(daily_returns.std())
        except Exception as exc:
            logger.warning(f"Could not parse backtest tearsheet: {exc}. Using defaults.")
            return defaults

    # ── AUDIT FIX #HEALTH-1: Stateful posterior update ────────────────────────

    def update_and_persist_posterior(self, new_returns: np.ndarray) -> None:
        """
        Performs the Normal-Inverse-Gamma conjugate posterior update and
        PERSISTS the updated hyper-parameters as instance state.

        This is the correct sequential Bayesian update. Calling this method
        multiple times with successive batches of returns is equivalent to
        calling it once with all returns concatenated.

        Args:
            new_returns: Array of daily returns since the last update call.
        """
        n = len(new_returns)
        if n == 0:
            return

        x_bar = float(np.mean(new_returns))

        # Standard NIG conjugate update equations
        kappa_n_new = self.kappa_n + n
        mu_n_new    = (self.kappa_n * self.mu_n + n * x_bar) / kappa_n_new
        alpha_n_new = self.alpha_n + n / 2.0
        ss          = float(np.sum((new_returns - x_bar) ** 2))
        beta_n_new  = (
            self.beta_n
            + 0.5 * ss
            + (n * self.kappa_n) / (2.0 * kappa_n_new) * (x_bar - self.mu_n) ** 2
        )

        # ── Persist — this is the fix ─────────────────────────────────────────
        self.kappa_n = kappa_n_new
        self.mu_n    = mu_n_new
        self.alpha_n = alpha_n_new
        self.beta_n  = beta_n_new
        self.n_total += n

        logger.debug(
            f"Posterior updated (+{n} obs, total={self.n_total}): "
            f"μ_n={self.mu_n:.6f}, α_n={self.alpha_n:.2f}, β_n={self.beta_n:.6f}"
        )

    def get_posterior_moments(self) -> Tuple[float, float]:
        """
        Returns the posterior expected mean and posterior predictive std dev.

        Returns:
            (post_mean, post_std): Both in daily return units.
        """
        post_mean = self.mu_n
        # Posterior variance of μ: β_n / (α_n * κ_n)
        post_var_mu = self.beta_n / max(self.alpha_n * self.kappa_n, 1e-12)
        return post_mean, float(np.sqrt(post_var_mu))

    # ── AUDIT FIX #HEALTH-3: Student-t marginal ────────────────────────────────

    def calculate_decay_probability(self, threshold: float = 0.0) -> float:
        """
        Computes P(μ < threshold | data) using the exact Student-t marginal
        posterior of μ under the Normal-Inverse-Gamma conjugate.

        The marginal of μ after integrating out σ² is:
            (μ - μ_n) / √(β_n / (α_n * κ_n))  ~  t(2α_n)

        For large n, this converges to the Normal CDF.
        For small n, the heavy tails of the t-distribution correctly reflect
        uncertainty about the variance.

        Args:
            threshold: Daily return threshold below which we call it "decay".
                       Default 0.0 (positive edge).

        Returns:
            decay_prob: P(μ < threshold | data) in [0, 1].
        """
        post_var_mu = self.beta_n / max(self.alpha_n * self.kappa_n, 1e-12)
        post_std_mu = np.sqrt(post_var_mu)

        if post_std_mu < 1e-12:
            return 1.0 if self.mu_n < threshold else 0.0

        # Standardised deviation
        t_stat = (threshold - self.mu_n) / post_std_mu
        # Degrees of freedom for the Student-t marginal
        df = 2.0 * self.alpha_n

        return float(stats.t.cdf(t_stat, df=df))

    # ── PUBLIC INTERFACE ───────────────────────────────────────────────────────

    async def evaluate_health(
        self,
        new_returns: np.ndarray,
        meta_agent: Optional[Any] = None,
        performance_report: str = "",
        recent_logs: str = "",
    ) -> str:
        """
        LIVE INFERENCE METHOD.
        Incorporates new live returns into the posterior and evaluates system health.

        AUDIT FIX #HEALTH-2: On RED status, triggers meta_agent.analyze_and_evolve()
        if a meta_agent instance is provided.

        Args:
            new_returns:       Array of daily live returns since last call.
            meta_agent:        Optional ConstitutionalMetaAgent instance.
                               If provided, RED status triggers `analyze_and_evolve`.
            performance_report: String summary of recent live metrics (for LLM prompt).
            recent_logs:        String of recent service logs (for LLM prompt).

        Returns:
            status: "RED", "YELLOW", or "GREEN".
        """
        self.update_and_persist_posterior(new_returns)
        decay_prob = self.calculate_decay_probability(threshold=0.0)

        post_mean, post_std = self.get_posterior_moments()
        ann_edge = post_mean * 252

        logger.info(
            f"Health Check | n_obs={self.n_total} | "
            f"E[μ|data]={ann_edge:+.2%} ann. | "
            f"P(decay)={decay_prob:.2%} | DoF={2*self.alpha_n:.1f}"
        )

        if decay_prob >= self.red_alert_prob:
            logger.critical(
                f"🔴 RED ALERT: P(alpha decay)={decay_prob:.2%} ≥ {self.red_alert_prob:.2%}. "
                f"Posterior mean={ann_edge:+.2%} ann. Triggering Meta-Agent evolution."
            )
            if meta_agent is not None:
                # Fire-and-forget — do not block the health monitor on LLM latency
                asyncio.create_task(
                    meta_agent.analyze_and_evolve(
                        performance_report=performance_report
                        or f"Bayesian alpha decay: P={decay_prob:.2%}, ann_edge={ann_edge:+.2%}",
                        recent_logs=recent_logs,
                    )
                )
            return "RED"

        elif decay_prob >= self.yellow_alert_prob:
            logger.warning(
                f"🟡 YELLOW ALERT: P(alpha decay)={decay_prob:.2%} ≥ {self.yellow_alert_prob:.2%}. "
                "Reduce trade sizing."
            )
            return "YELLOW"

        return "GREEN"

    def state_dict(self) -> dict:
        """Serialises the current posterior state for Redis persistence."""
        return {
            "mu_n":    self.mu_n,
            "kappa_n": self.kappa_n,
            "alpha_n": self.alpha_n,
            "beta_n":  self.beta_n,
            "n_total": self.n_total,
            "updated_at": datetime.utcnow().isoformat(),
        }

    def load_state_dict(self, d: dict) -> None:
        """Restores posterior state from Redis on container restart."""
        self.mu_n    = float(d.get("mu_n",    self.mu_0))
        self.kappa_n = float(d.get("kappa_n", self.kappa_0))
        self.alpha_n = float(d.get("alpha_n", self.alpha_0))
        self.beta_n  = float(d.get("beta_n",  self.beta_0))
        self.n_total = int(d.get("n_total",   0))
        logger.info(
            f"Posterior state restored: n_total={self.n_total}, "
            f"μ_n={self.mu_n:.6f}"
        )