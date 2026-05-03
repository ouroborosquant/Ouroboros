"""
fortress_v5/risk/wasserstein_hmm.py
────────────────────────────────────
Wasserstein-anchored Gaussian HMM for regime detection.

Label-switching problem: standard EM/Viterbi optimizes a permutation-invariant
likelihood — components swap identities across windows, making persistence
signals useless. Solution: after every parameter update, solve a min-cost
assignment (Hungarian) on the K×K W2-distance matrix between current and
reference components, then permute parameters to restore canonical ordering.

W2²(N(μ₁,Σ₁), N(μ₂,Σ₂)) = ‖μ₁-μ₂‖² + B²(Σ₁,Σ₂)
For diagonal Σ: B²(diag(σ₁²), diag(σ₂²)) = ‖σ₁-σ₂‖²   (exact, O(D))

Sinkhorn (GPU, ε-regularized OT) is used for the soft assignment transport
plan — the W2 cost matrix itself uses the closed-form diagonal formula.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegimeState:
    regime_probs: torch.Tensor       # (K,) posterior at final timestep
    exposure_multiplier: float        # ∈ [0.0, 1.0]  — fed directly to position sizer
    dominant_regime: int
    w2_transport_plan: torch.Tensor  # (K,K) soft OT plan for diagnostics


class WassersteinHMM(nn.Module):
    """
    Gaussian HMM whose component identity is anchored by W2 optimal transport.

    Regime ordering convention (enforced by exposure_logits prior):
        0 → trending / low-vol  (exposure → 1.0)
        1 → transitional        (exposure → 0.5)
        2 → crisis / high-vol   (exposure → 0.0)

    Parameters
    ----------
    n_regimes      : K Gaussian components
    n_features     : D-dimensional observation space
    sinkhorn_reg   : ε entropy regularization for Sinkhorn OT
    sinkhorn_iters : number of Sinkhorn log-domain iterations
    ema_alpha      : EMA decay for reference component update
    """

    def __init__(
        self,
        n_regimes: int = 3,
        n_features: int = 5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        sinkhorn_reg: float = 0.05,
        sinkhorn_iters: int = 100,
        ema_alpha: float = 0.95,
    ) -> None:
        super().__init__()
        self.K = n_regimes
        self.D = n_features
        self.device = torch.device(device)
        self.eps = sinkhorn_reg
        self.sinkhorn_iters = sinkhorn_iters
        self.ema_alpha = ema_alpha

        # ── Gaussian component parameters ──────────────────────────────────
        self.means    = nn.Parameter(torch.randn(n_regimes, n_features) * 0.1)
        self.log_stds = nn.Parameter(torch.zeros(n_regimes, n_features))

        # ── Transition matrix in unconstrained log-space ────────────────────
        # Initialized near identity (high self-transition) for regime persistence
        log_trans_init = torch.eye(n_regimes) * 3.0 + torch.ones(n_regimes, n_regimes) * 0.5
        self.log_trans = nn.Parameter(log_trans_init)

        # ── Initial state distribution ─────────────────────────────────────
        self.log_pi0 = nn.Parameter(torch.zeros(n_regimes))

        # ── Per-regime exposure multipliers (sigmoid-parameterized) ────────
        # Prior: regime 0 bullish, 1 neutral, 2 bearish
        self.exposure_logits = nn.Parameter(
            torch.tensor([2.5, 0.0, -2.5], dtype=torch.float32)
        )

        # W2-reference anchors (set on first forward pass, never optimized)
        self.register_buffer("_ref_means", torch.zeros(n_regimes, n_features))
        self.register_buffer("_ref_stds",  torch.ones(n_regimes, n_features))
        self._anchored: bool = False

    # ── Derived properties ─────────────────────────────────────────────────

    @property
    def stds(self) -> torch.Tensor:
        return self.log_stds.exp().clamp(min=1e-4)

    @property
    def trans(self) -> torch.Tensor:
        # Row-stochastic: each row sums to 1
        return torch.softmax(self.log_trans, dim=-1)

    @property
    def pi0(self) -> torch.Tensor:
        return torch.softmax(self.log_pi0, dim=-1)

    @property
    def exposure_multipliers(self) -> torch.Tensor:
        return torch.sigmoid(self.exposure_logits)

    # ── Core algorithms ────────────────────────────────────────────────────

    def _log_emission(self, x: torch.Tensor) -> torch.Tensor:
        """
        Diagonal-Gaussian log p(x_t | regime k) for all t,k.
        x : (T, D) → returns (T, K)
        """
        # (T,1,D) - (1,K,D) → (T,K,D)
        diff    = x.unsqueeze(1) - self.means.unsqueeze(0)
        log_std = self.log_stds.unsqueeze(0)                        # (1,K,D)
        log_p   = -0.5 * (
            (diff / self.stds.unsqueeze(0)).pow(2)
            + 2.0 * log_std
            + np.log(2.0 * np.pi)
        ).sum(-1)                                                     # (T,K)
        return log_p

    def _forward_filter(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Log-space forward algorithm — numerically stable for T > 500.
        Returns (log_alpha: (T,K), log_likelihood: scalar).
        """
        T            = x.shape[0]
        log_emit     = self._log_emission(x)                         # (T,K)
        log_A        = (self.trans + 1e-12).log()                    # (K,K)
        log_alpha    = torch.empty(T, self.K, device=self.device)
        log_alpha[0] = (self.pi0 + 1e-12).log() + log_emit[0]

        for t in range(1, T):
            # logsumexp over previous state: log Σ_i α_{t-1,i} A_{i,j}
            log_pred     = torch.logsumexp(
                log_alpha[t - 1].unsqueeze(1) + log_A, dim=0
            )                                                         # (K,)
            log_alpha[t] = log_pred + log_emit[t]

        return log_alpha, torch.logsumexp(log_alpha[-1], dim=0)

    @torch.no_grad()
    def _sinkhorn_w2_cost(
        self,
        mu_a: torch.Tensor,  sigma_a: torch.Tensor,   # (K,D)
        mu_b: torch.Tensor,  sigma_b: torch.Tensor,   # (K,D)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cost matrix C[i,j] = W2²(component_i, component_j) [exact, diagonal Σ]
        Soft transport plan via Sinkhorn-Knopp in log-domain.

        Returns (C: (K,K), P: (K,K)) — C is the squared W2, P is the OT plan.
        """
        C = (
            (mu_a.unsqueeze(1) - mu_b.unsqueeze(0)).pow(2) +
            (sigma_a.unsqueeze(1) - sigma_b.unsqueeze(0)).pow(2)
        ).sum(-1)                                                     # (K,K)

        # Log-domain Sinkhorn for numerical stability
        log_K_mat = -C / self.eps                                     # (K,K)
        log_a     = torch.full((self.K,), -np.log(self.K), device=self.device)
        log_b     = log_a.clone()
        log_u     = torch.zeros(self.K, device=self.device)
        log_v     = torch.zeros(self.K, device=self.device)

        for _ in range(self.sinkhorn_iters):
            log_u = log_a - torch.logsumexp(log_K_mat + log_v.unsqueeze(0), dim=1)
            log_v = log_b - torch.logsumexp(log_K_mat + log_u.unsqueeze(1), dim=0)

        log_P = log_K_mat + log_u.unsqueeze(1) + log_v.unsqueeze(0)
        return C, log_P.exp()

    def resolve_label_switching(self) -> None:
        """
        Hungarian assignment on W2 cost matrix → permute parameters to restore
        canonical ordering. Called after every gradient step.
        Must not be inside autograd graph (operates on .data).
        """
        if not self._anchored:
            self._ref_means.copy_(self.means.data)
            self._ref_stds.copy_(self.stds.data)
            self._anchored = True
            return

        C, _ = self._sinkhorn_w2_cost(
            self.means.data,   self.stds.data,
            self._ref_means,   self._ref_stds,
        )

        # min-cost bijective assignment: new[i] ↔ ref[col_ind[i]]
        _, col_ind = linear_sum_assignment(C.cpu().numpy())
        perm = torch.from_numpy(col_ind).long().to(self.device)

        with torch.no_grad():
            self.means.data          = self.means.data[perm]
            self.log_stds.data       = self.log_stds.data[perm]
            self.exposure_logits.data = self.exposure_logits.data[perm]
            # Row + column permutation on transition matrix
            self.log_trans.data      = self.log_trans.data[perm][:, perm]

        # EMA reference update — allows slow structural drift
        self._ref_means.mul_(self.ema_alpha).add_(
            (1.0 - self.ema_alpha) * self.means.data
        )
        self._ref_stds.mul_(self.ema_alpha).add_(
            (1.0 - self.ema_alpha) * self.stds.data
        )

    # ── Public API ─────────────────────────────────────────────────────────

    def training_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood — minimize with Adam/LBFGS."""
        x = x.to(self.device)
        _, log_ll = self._forward_filter(x)
        return -log_ll

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> RegimeState:
        """
        Run forward filter on the feature window x and return the current
        RegimeState. Caller is responsible for feature construction
        (e.g., [realized_vol, skew, put_call_ratio, breadth, gex_delta]).

        x : (T, D) — typically the last 60 trading days
        """
        x = x.to(self.device)
        log_alpha, _ = self._forward_filter(x)

        posterior   = torch.softmax(log_alpha[-1], dim=0)            # (K,)
        dominant    = int(posterior.argmax())
        exposure    = float((posterior * self.exposure_multipliers).sum().clamp(0.0, 1.0))

        C, plan = self._sinkhorn_w2_cost(
            self.means, self.stds,
            self._ref_means, self._ref_stds,
        )

        log.debug(
            "Regime posterior=%s dominant=%d exposure=%.3f",
            posterior.tolist(), dominant, exposure
        )

        return RegimeState(
            regime_probs      = posterior.cpu(),
            exposure_multiplier = exposure,
            dominant_regime   = dominant,
            w2_transport_plan = plan.cpu(),
        )