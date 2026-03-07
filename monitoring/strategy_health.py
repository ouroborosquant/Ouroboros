"""
FORTRESS v5 - strategy_health.py
Path: monitoring/strategy_health.py

Bayesian Strategy Health Monitor.
Continuously evaluates live trading returns against backtest expectations.
If the probability of alpha decay crosses the critical threshold, it triggers the Meta-Learning Agent.
"""

import os
import yaml
import logging
import numpy as np
import scipy.stats as stats
from typing import Tuple

logger = logging.getLogger("BayesianHealthMonitor")

class BayesianHealthMonitor:
    def __init__(self, config_path: str = 'config/risk_limits.yaml'):
        # Safely load the Bayesian thresholds we defined earlier
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            limits = config.get('health_monitor', {}) if config else {}
            
        self.yellow_alert_prob = limits.get('bayesian_yellow_alert_prob', 0.20)
        self.red_alert_prob = limits.get('bayesian_red_alert_prob', 0.50)
        
        # Prior assumptions based on backtest (e.g., Expected Daily Return and Volatility)
        # In a fully deployed system, these priors are injected dynamically from the backtest tearsheet.
        self.prior_mu = 0.0004  # ~10% annualized target return
        self.prior_sigma = 0.01 # ~15% annualized target volatility
        
        # Conjugate prior parameters for a Normal distribution with unknown mean and variance
        self.mu_0 = self.prior_mu
        self.kappa_0 = 1.0  # Weight of the prior (confidence in the backtest)
        self.alpha_0 = 1.0  # Shape parameter for variance
        self.beta_0 = self.prior_sigma ** 2  # Scale parameter for variance

    def update_posterior(self, live_returns: np.ndarray) -> Tuple[float, float]:
        """
        Performs a rigorous Bayesian update on the expected strategy return 
        using a Normal-Inverse-Gamma conjugate prior.
        """
        n = len(live_returns)
        if n == 0:
            return self.mu_0, np.sqrt(self.beta_0 / self.alpha_0)
            
        sample_mean = np.mean(live_returns)
        
        # Update hyper-parameters
        kappa_n = self.kappa_0 + n
        mu_n = (self.kappa_0 * self.mu_0 + n * sample_mean) / kappa_n
        
        alpha_n = self.alpha_0 + n / 2.0
        
        # Update sum of squares
        ss_update = 0.5 * sum((x - sample_mean) ** 2 for x in live_returns)
        mean_diff = (n * self.kappa_0) / (self.kappa_0 + n) * (sample_mean - self.mu_0) ** 2
        beta_n = self.beta_0 + ss_update + mean_diff
        
        # Expected posterior standard deviation
        posterior_sigma = np.sqrt(beta_n / max(alpha_n - 1.0, 1e-6))
        
        return mu_n, posterior_sigma

    def calculate_decay_probability(self, posterior_mu: float, posterior_sigma: float, threshold: float = 0.0) -> float:
        """
        Calculates the cumulative probability that the true strategy edge has fallen 
        below the failure threshold (0.0).
        """
        # Approximating the marginal posterior of mu using the Normal CDF
        z_score = (threshold - posterior_mu) / (posterior_sigma + 1e-8)
        decay_prob = stats.norm.cdf(z_score)
        return float(decay_prob)

    def evaluate_health(self, live_returns: np.ndarray) -> str:
        """
        LIVE INFERENCE METHOD.
        Evaluates the rolling window of live returns and returns the system health status.
        Called daily by the central supervisor.
        """
        logger.debug(f"Evaluating health against {len(live_returns)} recent live trading days...")
        
        post_mu, post_sigma = self.update_posterior(live_returns)
        decay_prob = self.calculate_decay_probability(post_mu, post_sigma, threshold=0.0)
        
        logger.info(f"Bayesian Alpha Decay Probability: {decay_prob:.2%}")
        
        if decay_prob >= self.red_alert_prob:
            logger.critical("🔴 RED ALERT: Strategy Alpha Decay verified by Bayesian bounds. Halting and requesting Meta-Agent intervention.")
            return "RED"
        elif decay_prob >= self.yellow_alert_prob:
            logger.warning("🟡 YELLOW ALERT: Strategy performance deteriorating beyond expected variance. Reducing trade sizing.")
            return "YELLOW"
        else:
            return "GREEN"