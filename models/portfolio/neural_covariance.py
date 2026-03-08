"""
FORTRESS v5 - neural_covariance.py
Path: models/portfolio/neural_covariance.py

Deep Factor Covariance Estimator — Regime-Conditional Neural Factor Model.

Architecture:

  DeepFactorCovariance learns a latent factor decomposition of the covariance matrix:

      Σ = B·F·Bᵀ + D

  where:
    B ∈ R^{N×K}  — factor loading matrix (N assets, K latent factors)
    F ∈ R^{K×K}  — factor covariance (symmetric PSD, Cholesky-parameterised)
    D ∈ R^{N}    — idiosyncratic variance (diagonal, positive)

  Both B and F are conditioned on the regime latent z_t from Mamba-KAN VAE.
  This makes Σ a smooth, differentiable function of the market regime:
    - Bull (low-vol): factors are market + sector + style; loadings spread
    - Crisis (high-vol): loadings collapse toward a single market factor (correlation → 1)
    - Recovery: gradual factor diversification resumes

  PSD Enforcement:
    - Factor covariance F is parameterised via its lower Cholesky factor L_F:
        F = L_F @ L_F.T   (guaranteed PSD by construction)
    - Idiosyncratic variance D uses softplus to enforce D > 0 element-wise.
    - The full Σ = B @ (L_F @ L_F.T) @ B.T + diag(D) is PSD by the outer product
      structure of B @ B.T plus the positive diagonal D.

  Why this over sample covariance?
    - Sample Σ̂ from T=21 days (monthly) has severe estimation error for N=25 assets:
      condition number can exceed 1000, making the minimum-variance portfolio
      extremely sensitive to noise (Ledoit & Wolf 2004).
    - The factor structure imposes a K-dimensional subspace for common risk,
      reducing the effective parameter count from N(N+1)/2 to K(K+1)/2 + N.
    - Regime-conditioning allows the factor structure to shift with z_t —
      capturing the empirically-observed compression of correlation during crises.

  Integration with portfolio_agent_svc:
    - `get_covariance(returns_history, z_t)` → (N,N) tensor for use in:
        1. Volatility targeting overlay (σ_realized from Σ diagonal)
        2. Risk parity rebalancing (equal risk contribution weights)
        3. DMRG/scipy portfolio optimiser as the covariance input

Parameters:
  N:          Number of assets (25 for Fortress ETF universe).
  K:          Number of latent factors (default 8: market + 3 sector + style×4).
  regime_dim: Dimension of z_t from Mamba-KAN (16).
  hidden_dim: MLP hidden width for factor network.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("NeuralCovariance")

# ── Architecture constants ─────────────────────────────────────────────────────
_DEFAULT_N:          int   = 25    # Assets
_DEFAULT_K:          int   = 8     # Latent factors
_DEFAULT_REGIME_DIM: int   = 16
_DEFAULT_HIDDEN:     int   = 128
_SOFTPLUS_BETA:      float = 5.0   # Steeper softplus → D closer to ReLU (less shrinkage)
_MIN_IDIO_VAR:       float = 1e-6  # Floor for idiosyncratic variance


# ─────────────────────────────────────────────────────────────────────────────
# Factor loading network
# ─────────────────────────────────────────────────────────────────────────────

class FactorLoadingNetwork(nn.Module):
    """
    Maps (returns_summary, z_t) → factor loadings B ∈ R^{N×K}.

    Returns summary: cross-sectional statistics of the lookback window
    that capture the current factor structure without requiring the full
    raw T×N return matrix as input.

    Summary features (dim = 3*N):
      - EWMA return (half-life 21d):         μ_i
      - EWMA volatility (half-life 21d):     σ_i
      - Cross-sectional z-score of μ/σ:      SR_i_normalised
    """

    def __init__(
        self,
        n_assets:    int = _DEFAULT_N,
        n_factors:   int = _DEFAULT_K,
        regime_dim:  int = _DEFAULT_REGIME_DIM,
        hidden_dim:  int = _DEFAULT_HIDDEN,
    ) -> None:
        super().__init__()
        self.n_assets  = n_assets
        self.n_factors = n_factors

        input_dim = 3 * n_assets + regime_dim   # summary stats + regime

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_assets * n_factors),
        )

    def forward(
        self,
        return_summary: torch.Tensor,   # (Batch, 3*N)
        z_t:            torch.Tensor,   # (Batch, regime_dim)
    ) -> torch.Tensor:
        """
        Returns:
            B: Shape (Batch, N, K) — factor loading matrix per sample.
        """
        x = torch.cat([return_summary, z_t], dim=-1)
        B_flat = self.net(x)                           # (Batch, N*K)
        return B_flat.view(-1, self.n_assets, self.n_factors)


class FactorCovarianceNetwork(nn.Module):
    """
    Maps z_t → Cholesky factor L_F ∈ R^{K×K} (lower triangular).

    F = L_F @ L_F.T is the factor covariance matrix (K×K, PSD by construction).

    Cholesky parameterisation:
      - Off-diagonal elements: unconstrained (can be positive or negative).
      - Diagonal elements: softplus-activated to ensure L_F[i,i] > 0,
        which is required for L_F to be the unique lower Cholesky factor of F.
    """

    def __init__(
        self,
        n_factors:  int = _DEFAULT_K,
        regime_dim: int = _DEFAULT_REGIME_DIM,
        hidden_dim: int = _DEFAULT_HIDDEN,
    ) -> None:
        super().__init__()
        self.n_factors = n_factors
        # Number of elements in lower triangular K×K matrix
        self.n_chol = n_factors * (n_factors + 1) // 2

        self.net = nn.Sequential(
            nn.Linear(regime_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.n_chol),
        )

        # Indices to reconstruct lower triangular matrix
        self._tril_idx = torch.tril_indices(n_factors, n_factors)

    def forward(self, z_t: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            F: Shape (Batch, K, K) — factor covariance matrix.
        """
        B     = z_t.shape[0]
        chol  = self.net(z_t)    # (Batch, n_chol)

        # Reconstruct lower triangular Cholesky factor
        L = torch.zeros(B, self.n_factors, self.n_factors, device=z_t.device)
        L[:, self._tril_idx[0].to(z_t.device), self._tril_idx[1].to(z_t.device)] = chol

        # Enforce positive diagonal via softplus
        diag_idx = torch.arange(self.n_factors, device=z_t.device)
        L[:, diag_idx, diag_idx] = F.softplus(
            L[:, diag_idx, diag_idx], beta=_SOFTPLUS_BETA
        ) + 1e-4   # absolute floor to prevent degenerate Cholesky

        # F = L @ Lᵀ : guaranteed symmetric PSD
        return torch.bmm(L, L.transpose(1, 2))


class IdiosyncraticVarianceNetwork(nn.Module):
    """
    Maps (return_summary, z_t) → idiosyncratic variance D ∈ R^{N} (diagonal).

    Softplus activation ensures D > 0. The idiosyncratic component captures
    variance orthogonal to the K latent factors — e.g. earnings surprises,
    index-specific flows, ETF creation/redemption pressure.
    """

    def __init__(
        self,
        n_assets:   int = _DEFAULT_N,
        regime_dim: int = _DEFAULT_REGIME_DIM,
        hidden_dim: int = _DEFAULT_HIDDEN,
    ) -> None:
        super().__init__()
        input_dim = 3 * n_assets + regime_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, n_assets),
        )

    def forward(
        self,
        return_summary: torch.Tensor,
        z_t:            torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns:
            D: Shape (Batch, N) — per-asset idiosyncratic variance (strictly positive).
        """
        x = torch.cat([return_summary, z_t], dim=-1)
        return F.softplus(self.net(x), beta=_SOFTPLUS_BETA) + _MIN_IDIO_VAR


# ─────────────────────────────────────────────────────────────────────────────
# Return summary extractor
# ─────────────────────────────────────────────────────────────────────────────

def _compute_return_summary(
    returns: torch.Tensor,
    halflife: int = 21,
) -> torch.Tensor:
    """
    Compute EWMA cross-sectional summary statistics from a return window.

    Args:
        returns: Shape (Batch, T, N) — T days of N-asset daily returns.
        halflife: EWMA half-life in days.

    Returns:
        summary: Shape (Batch, 3*N) — [μ_ewma | σ_ewma | sr_zscore].
    """
    T    = returns.shape[1]
    alpha = 1.0 - math.exp(-math.log(2.0) / halflife)   # type: ignore[attr-defined]

    import math as _math
    alpha = 1.0 - _math.exp(-_math.log(2.0) / halflife)

    # EWMA weights: w_t ∝ (1-α)^(T-1-t) for t = 0, ..., T-1
    t_idx   = torch.arange(T, device=returns.device, dtype=returns.dtype)
    weights = (1.0 - alpha) ** (T - 1 - t_idx)          # (T,)
    weights = weights / weights.sum()
    weights = weights.unsqueeze(0).unsqueeze(-1)         # (1, T, 1)

    # EWMA mean: (Batch, N)
    mu = (returns * weights).sum(dim=1)

    # EWMA variance: E[(r - μ)²] with EWMA weights
    dev      = returns - mu.unsqueeze(1)                 # (Batch, T, N)
    ewma_var = (dev ** 2 * weights).sum(dim=1)           # (Batch, N)
    sigma    = torch.sqrt(ewma_var.clamp(min=1e-8))     # (Batch, N)

    # Cross-sectional z-score of Sharpe ratios
    sr   = mu / sigma
    sr_mean = sr.mean(dim=-1, keepdim=True)
    sr_std  = sr.std(dim=-1, keepdim=True) + 1e-6
    sr_z    = (sr - sr_mean) / sr_std                   # (Batch, N)

    return torch.cat([mu, sigma, sr_z], dim=-1)         # (Batch, 3*N)


# ─────────────────────────────────────────────────────────────────────────────
# Full covariance model
# ─────────────────────────────────────────────────────────────────────────────

class DeepFactorCovariance(nn.Module):
    """
    Full regime-conditional deep factor covariance estimator.

    Σ(z_t, R_{t-T:t}) = B(z_t, R) · F(z_t) · B(z_t, R)ᵀ + diag(D(z_t, R))

    PSD by construction via Cholesky parameterisation of F and softplus D.

    Training objective (used in train_covariance.py — separate from this file):
        L = NLL(R_t | Σ_{t-1}) + λ * ||Σ - Σ_sample||²_F
    where NLL is the multivariate Gaussian negative log-likelihood on next-day returns.

    Standalone inference via `get_covariance()`:
        - Accepts raw numpy return arrays for compatibility with portfolio_agent_svc
        - Returns both the (N,N) covariance and the decomposed (B, F, D) factors
    """

    def __init__(self, config: Dict) -> None:
        super().__init__()

        self.n_assets   = config.get("n_assets",   _DEFAULT_N)
        self.n_factors  = config.get("n_factors",  _DEFAULT_K)
        self.regime_dim = config.get("latent_dim", _DEFAULT_REGIME_DIM)
        self.hidden_dim = config.get("hidden_dim", _DEFAULT_HIDDEN)

        self.loading_net = FactorLoadingNetwork(
            self.n_assets, self.n_factors, self.regime_dim, self.hidden_dim
        )
        self.factor_cov_net = FactorCovarianceNetwork(
            self.n_factors, self.regime_dim, self.hidden_dim
        )
        self.idio_net = IdiosyncraticVarianceNetwork(
            self.n_assets, self.regime_dim, self.hidden_dim
        )

    def forward(
        self,
        returns: torch.Tensor,    # (Batch, T, N)
        z_t:    torch.Tensor,    # (Batch, regime_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the full covariance matrix and its factor decomposition.

        Args:
            returns: Shape (Batch, T, N) — rolling return window.
            z_t:    Shape (Batch, regime_dim) — current regime posterior mean.

        Returns:
            Sigma: (Batch, N, N) — full covariance matrix.
            B:     (Batch, N, K) — factor loading matrix.
            F:     (Batch, K, K) — factor covariance matrix.
            D:     (Batch, N)    — idiosyncratic variance.
        """
        summary = _compute_return_summary(returns)     # (Batch, 3*N)

        B = self.loading_net(summary, z_t)             # (Batch, N, K)
        F_mat = self.factor_cov_net(z_t)               # (Batch, K, K)
        D = self.idio_net(summary, z_t)                # (Batch, N)

        # Σ = B @ F @ Bᵀ + diag(D)
        BF    = torch.bmm(B, F_mat)                    # (Batch, N, K)
        BFBt  = torch.bmm(BF, B.transpose(1, 2))      # (Batch, N, N)
        Sigma = BFBt + torch.diag_embed(D)             # (Batch, N, N)

        return Sigma, B, F_mat, D

    @torch.no_grad()
    def get_covariance(
        self,
        returns_np: np.ndarray,              # (T, N) daily returns
        z_t_np:     np.ndarray,              # (regime_dim,) regime posterior mean
        device:     str = "cpu",
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Inference interface for `services/portfolio_agent_svc.py`.

        Args:
            returns_np: Shape (T, N) — lookback window of daily returns.
            z_t_np:     Shape (regime_dim,) — z_mu from Mamba-KAN VAE.
            device:     Torch device string.

        Returns:
            sigma_np:   Shape (N, N) — numpy covariance matrix.
            factors:    Dict with keys 'B', 'F', 'D' (numpy arrays).
        """
        self.eval()
        dev = torch.device(device)
        self.to(dev)

        returns_t = torch.FloatTensor(returns_np).unsqueeze(0).to(dev)   # (1, T, N)
        z_t_t     = torch.FloatTensor(z_t_np).unsqueeze(0).to(dev)      # (1, regime_dim)

        Sigma, B, F_mat, D = self.forward(returns_t, z_t_t)

        sigma_np = Sigma.squeeze(0).cpu().numpy()
        factors  = {
            "B": B.squeeze(0).cpu().numpy(),
            "F": F_mat.squeeze(0).cpu().numpy(),
            "D": D.squeeze(0).cpu().numpy(),
        }
        return sigma_np, factors

    def nll_loss(
        self,
        next_returns: torch.Tensor,    # (Batch, N) — next-day returns
        Sigma:        torch.Tensor,    # (Batch, N, N)
    ) -> torch.Tensor:
        """
        Multivariate Gaussian NLL for training:
            L = 0.5 * [log|Σ| + r.T @ Σ^{-1} @ r + N*log(2π)]

        Uses Cholesky solve for numerical stability:
            Σ^{-1} @ r = chol_solve(r, cholesky(Σ))

        The N*log(2π) constant is omitted (does not affect gradients).

        Args:
            next_returns: Shape (Batch, N) — realised next-day returns.
            Sigma:        Shape (Batch, N, N) — predicted covariance.

        Returns:
            Scalar NLL loss.
        """
        # Add ridge regularisation to ensure positive-definiteness numerically
        N     = Sigma.shape[-1]
        ridge = 1e-5 * torch.eye(N, device=Sigma.device).unsqueeze(0)
        Sigma_reg = Sigma + ridge

        try:
            L_chol = torch.linalg.cholesky(Sigma_reg)
        except RuntimeError:
            # Fallback: add stronger ridge if Cholesky fails (e.g. in early training)
            Sigma_reg = Sigma + 1e-3 * torch.eye(N, device=Sigma.device).unsqueeze(0)
            L_chol = torch.linalg.cholesky(Sigma_reg)

        # Log determinant: log|Σ| = 2 * Σ_i log(L_ii)
        log_det = 2.0 * L_chol.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)   # (Batch,)

        # Mahalanobis: rᵀ Σ⁻¹ r via triangular solve
        r = next_returns.unsqueeze(-1)                    # (Batch, N, 1)
        v = torch.cholesky_solve(r, L_chol)               # (Batch, N, 1)
        mahal = (r * v).sum(dim=(-2, -1))                 # (Batch,)

        nll = 0.5 * (log_det + mahal)
        return nll.mean()

    def risk_parity_weights(
        self,
        Sigma: torch.Tensor,
        n_iter: int = 50,
    ) -> torch.Tensor:
        """
        Compute equal risk contribution (ERC) portfolio weights.

        ERC condition: w_i * (Σw)_i = constant for all i.
        Solved via Newton-Raphson on the risk contribution equations.
        Starting point: inverse volatility weights.

        Args:
            Sigma: Shape (Batch, N, N).
            n_iter: Newton-Raphson iterations.

        Returns:
            w: Shape (Batch, N) — ERC weights (sum to 1, all > 0).
        """
        B, N = Sigma.shape[0], Sigma.shape[1]
        # Start from inverse-vol weights
        vol = torch.sqrt(Sigma.diagonal(dim1=-2, dim2=-1).clamp(min=1e-8))  # (Batch, N)
        w   = (1.0 / vol) / (1.0 / vol).sum(dim=-1, keepdim=True)           # (Batch, N)

        target_rc = 1.0 / N   # Equal risk contribution fraction

        for _ in range(n_iter):
            # Portfolio variance: σ²_p = wᵀ Σ w
            Sw        = torch.bmm(Sigma, w.unsqueeze(-1)).squeeze(-1)   # (Batch, N)
            port_var  = (w * Sw).sum(dim=-1, keepdim=True)              # (Batch, 1)

            # Risk contribution: RC_i = w_i * (Σw)_i / σ²_p
            rc = w * Sw / (port_var + 1e-8)                             # (Batch, N)

            # Gradient of (RC_i - target)² w.r.t. w_i — simplified Newton step
            grad  = rc - target_rc
            # Approximate Hessian diagonal: Σ_ii / σ²_p (ignoring off-diagonal)
            h_diag = Sigma.diagonal(dim1=-2, dim2=-1) / (port_var + 1e-8)
            step  = grad / (h_diag.clamp(min=1e-8))

            w = w - 0.5 * step
            w = w.clamp(min=1e-6)                                        # long-only constraint
            w = w / w.sum(dim=-1, keepdim=True)

        return w