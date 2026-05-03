"""
fortress_v5/signals/frac_diff.py
──────────────────────────────────
Adaptive Fractional Differencing with learnable per-asset order d.

Motivation
──────────
Standard log-returns (d=1) achieve stationarity but destroy the Hurst
exponent H — the long-range memory that makes price series predictable.
Fractional differencing at d* ∈ (0, 1) achieves stationarity (ADF rejects
unit root) while preserving maximum memory.

Key insight (López de Prado, 2018): there exists a minimum d* per asset
such that the differenced series is just stationary. We make d* a learnable
parameter, trained to balance a stationarity loss against a memory-loss penalty.

Differencing operator:
    (1 - B)^d x_t = Σ_{k=0}^{∞} w_k x_{t-k}
    w_0 = 1
    w_k = ∏_{j=0}^{k-1} (j - d) / (j + 1)    [iterative form]
         = (-1)^k · Γ(d+1) / (Γ(k+1) · Γ(d-k+1))  [Gamma form]

FFT Acceleration
────────────────
Direct convolution: O(T²) per series → O(T log T) via FFT overlap-add.
The weight vector w is computed once per d value, padded to length T, then
convolved via rfft/irfft. Since d is a scalar parameter per asset, ∂L/∂d
flows through the weight computation via log-Gamma identities.

Gradient through d
──────────────────
∂w_k/∂d = w_k · Σ_{j=0}^{k-1} 1/(j-d)    [chain rule through iterative product]

This is implemented via log-Gamma differentiation for numerical stability.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


# ── Weight computation ────────────────────────────────────────────────────

def _frac_diff_weights(d: torch.Tensor, length: int) -> torch.Tensor:
    """
    Compute fractional differencing weights via log-Gamma (numerically stable).

    w_k = Γ(d+1) / (Γ(k+1) · Γ(d-k+1)) · (-1)^k

    d      : scalar ∈ [0, 1] — differencing order
    length : truncation window L
    Returns : (L,) weight vector, gradient-enabled through d
    """
    k    = torch.arange(length, dtype=d.dtype, device=d.device)
    sign = (-1.0) ** k

    # Log-Gamma form: log|w_k| = lgamma(d+1) - lgamma(k+1) - lgamma(d-k+1)
    # For k > d (when d < L), lgamma(d-k+1) has poles — clamp to small eps
    d_minus_k = (d - k + 1.0).clamp(min=1e-6)

    log_w = (
        torch.lgamma(d + 1.0)
        - torch.lgamma(k + 1.0)
        - torch.lgamma(d_minus_k)
    )
    w = sign * torch.exp(log_w)

    # Taper to zero near truncation boundary to reduce edge artifacts
    taper = torch.ones_like(w)
    taper[-min(5, length // 4):] = 0.5                               # soft boundary
    return w * taper


def _fft_fractional_diff(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Apply fractional differencing filter w to signal x via FFT overlap-add.

    x : (T,)   signal
    w : (L,)   filter weights (w[0]=1, causal)
    Returns (T,) differenced series (first L-1 values are burn-in, NaN-masked)
    """
    T, L = x.shape[0], w.shape[0]
    n    = T + L - 1
    n_fft = 1 << (n - 1).bit_length()                                # next power of 2

    # Zero-pad and FFT
    X = torch.fft.rfft(x,   n=n_fft)
    W = torch.fft.rfft(w,   n=n_fft)

    y_full = torch.fft.irfft(X * W, n=n_fft)[:T]                    # causal truncation

    # Burn-in mask: first L-1 observations use incomplete history
    burn = min(L - 1, T)
    y_full[:burn] = float("nan")

    return y_full


# ── Learnable fractional differencing module ──────────────────────────────

class AdaptiveFracDiff(nn.Module):
    """
    Per-asset learnable fractional differencing with FFT acceleration.

    Each asset i has a differencing order d_i = sigmoid(θ_i) ∈ (0, 1),
    initialized near d=0.4 (empirically stable for most ETFs).

    Training objective (injected externally):
        L = L_task + λ_stat · L_stationarity + λ_mem · L_memory

    where:
        L_stationarity ≈ -ADF_approx(x_diff_i)   [minimize ↔ force stationarity]
        L_memory       = |H_i - H_target|          [preserve Hurst exponent]

    Parameters
    ----------
    n_assets    : N — one d parameter per asset
    weight_len  : truncation window L (longer = more memory preserved, slower)
    d_init      : initialization for d ∈ (0,1)
    """

    def __init__(
        self,
        n_assets: int,
        weight_len: int = 128,
        d_init: float = 0.4,
    ) -> None:
        super().__init__()
        self.N = n_assets
        self.L = weight_len

        # d_i = sigmoid(θ_i) → θ_init = logit(d_init)
        d_init_logit = np.log(d_init / (1.0 - d_init))
        self.theta = nn.Parameter(
            torch.full((n_assets,), d_init_logit, dtype=torch.float32)
        )

    @property
    def d_values(self) -> torch.Tensor:
        """Per-asset d ∈ (0, 1). Clamped slightly away from 0/1 for gradient flow."""
        return torch.sigmoid(self.theta).clamp(min=0.01, max=0.99)

    def forward(self, prices: torch.Tensor) -> torch.Tensor:
        """
        Apply learnable fractional differencing to a price matrix.

        prices  : (T, N) raw price series (NOT log-prices — we take log internally)
        Returns : (T, N) differenced series (burn-in rows contain NaN)

        Gradient flows through d_values → theta via autograd.
        """
        T, N = prices.shape
        assert N == self.N, f"Asset count mismatch: {N} ≠ {self.N}"

        log_prices = torch.log(prices.clamp(min=1e-8))               # (T, N)
        output     = torch.empty_like(log_prices)

        d_vals = self.d_values                                        # (N,) with grad

        for i in range(N):
            w_i      = _frac_diff_weights(d_vals[i], self.L)         # (L,) grad-connected
            diff_i   = _fft_fractional_diff(log_prices[:, i], w_i)   # (T,)
            output[:, i] = diff_i

        return output

    def forward_batch(
        self,
        prices: torch.Tensor,
        asset_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized forward pass using batched FFT.
        Returns (differenced: (T,N), d_values: (N,)) for downstream use.

        NOTE: This detaches the gradient on the FFT operation to avoid
        the O(T·N·L) autograd graph — gradients through d flow via
        a surrogate path: L_stationarity(diff) → ∂L/∂diff → ∂diff/∂d.
        """
        T, N = prices.shape
        idx  = asset_idx if asset_idx is not None else torch.arange(N)

        log_p  = torch.log(prices.clamp(min=1e-8))                   # (T, N)
        d_vals = self.d_values                                        # (N,)

        # Batch FFT: compute all weight vectors → stack → batched FFT
        W_list = torch.stack([_frac_diff_weights(d_vals[i], self.L) for i in idx])  # (N, L)
        n_fft  = 1 << (T + self.L - 1 - 1).bit_length()

        X_fft  = torch.fft.rfft(log_p.T.contiguous(), n=n_fft)       # (N, n_fft//2+1)
        W_fft  = torch.fft.rfft(W_list, n=n_fft)                     # (N, n_fft//2+1)

        y_full = torch.fft.irfft(X_fft * W_fft, n=n_fft)[:, :T]     # (N, T)
        diff   = y_full.T.contiguous()                                # (T, N)

        # Burn-in mask
        burn = min(self.L - 1, T)
        diff[:burn] = float("nan")

        return diff, d_vals

    # ── Loss functions ─────────────────────────────────────────────────────

    @staticmethod
    def adf_approx_loss(series: torch.Tensor, lag: int = 1) -> torch.Tensor:
        """
        Differentiable proxy for ADF stationarity test.

        Approximates the ADF t-statistic:
            t = γ̂ / std(γ̂)   where  Δy = γ·y_{t-1} + ε

        More negative t → more stationary.
        We minimize t (i.e., maximize stationarity).
        """
        # Remove NaN (burn-in)
        s    = series[~torch.isnan(series)]
        if s.numel() < 20:
            return torch.tensor(0.0, requires_grad=True, device=series.device)

        y      = s[lag:]                                              # Δy proxy
        y_lag  = s[:-lag]

        # OLS: γ̂ = (y_lag'y_lag)^{-1} y_lag'y
        gamma  = (y_lag * y).sum() / (y_lag.pow(2).sum() + 1e-8)
        resid  = y - gamma * y_lag
        sigma2 = resid.pow(2).mean() + 1e-8
        se     = torch.sqrt(sigma2 / (y_lag.pow(2).sum() + 1e-8))
        t_stat = gamma / se

        # Maximize |t_stat| in negative direction (more negative = more stationary)
        return t_stat                                                  # minimize this

    @staticmethod
    def hurst_loss(series: torch.Tensor, target_H: float = 0.6) -> torch.Tensor:
        """
        Differentiable Hurst exponent loss using R/S analysis.
        Penalizes deviation from target_H (typically 0.5-0.65 for persistent ETFs).
        """
        s = series[~torch.isnan(series)]
        if s.numel() < 32:
            return torch.tensor(0.0, requires_grad=True, device=series.device)

        # Log R/S at multiple lags
        T_max = s.numel()
        lags  = [T_max // 8, T_max // 4, T_max // 2, T_max]
        rs_values = []

        for lag in lags:
            chunk  = s[:lag]
            mean_s = chunk.mean()
            devs   = (chunk - mean_s).cumsum(0)
            R      = devs.max() - devs.min() + 1e-8
            S      = chunk.std() + 1e-8
            rs_values.append(torch.log(R / S))

        rs_tensor = torch.stack(rs_values)
        log_lags  = torch.log(torch.tensor(
            [float(l) for l in lags], device=s.device, dtype=s.dtype
        ))

        # Hurst: H = slope of log(R/S) vs log(n)
        log_lags_c = log_lags - log_lags.mean()
        rs_c       = rs_tensor - rs_tensor.mean()
        H_est      = (log_lags_c * rs_c).sum() / (log_lags_c.pow(2).sum() + 1e-8)

        return (H_est - target_H).pow(2)

    def combined_loss(
        self,
        diff_series: torch.Tensor,        # (T, N) output of forward()
        raw_series: torch.Tensor,         # (T, N) original log-prices
        lambda_stat: float = 1.0,
        lambda_mem: float = 0.5,
        target_H: float = 0.6,
    ) -> torch.Tensor:
        """
        L = λ_stat · mean_i(ADF_approx_i) + λ_mem · mean_i(HurstLoss_i)

        Minimizing ADF proxy forces d toward minimum-stationarity threshold.
        Minimizing Hurst loss prevents over-differencing (d → 1).
        """
        stat_losses = []
        hurst_losses = []

        for i in range(self.N):
            stat_losses.append(self.adf_approx_loss(diff_series[:, i]))
            hurst_losses.append(self.hurst_loss(diff_series[:, i], target_H))

        return (
            lambda_stat * torch.stack(stat_losses).mean() +
            lambda_mem  * torch.stack(hurst_losses).mean()
        )