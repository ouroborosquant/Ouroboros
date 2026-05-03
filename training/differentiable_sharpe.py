"""
fortress_v5/training/differentiable_sharpe.py
───────────────────────────────────────────────
End-to-end Differentiable Sharpe Ratio Loss.

Motivation
──────────
MSE loss trains a signal to minimize forecast error, not trading profit.
A 1% prediction error on a low-vol asset is strategically irrelevant; a 0.5%
error on a high-vol asset at a position inflection may be catastrophic.

Training on the net Sharpe ratio directly aligns the loss landscape with the
realized objective function under realistic frictions.

Net Sharpe (discrete-time, annualized):
    r_net,t = rᵀw_t - c · ‖w_t - w_{t-1}‖₁_smooth
    SR_net  = √252 · E[r_net] / std(r_net)
    L       = -SR_net  (minimize ↔ maximize SR)

Smooth absolute value (Huber-like):
    smooth_abs(x; ε) = √(x² + ε)   ← differentiable everywhere, ε > 0
    lim_{x→0} smooth_abs = √ε  (smooth near zero unlike |x|)
    ∂/∂x smooth_abs = x / √(x² + ε)   ← well-defined at x=0

This is strictly better than ‖·‖₁ for training because it provides
non-zero gradient even when turnover is small — the optimizer can
"lean away" from marginal trades before they occur.

Differential Sharpe (Moody & Saffell, 2001):
    For online gradient computation, the differential Sharpe decomposes
    the gradient through time-series statistics without full unrolling:
    ∂SR/∂w_t = (∂r_net,t/∂w_t) · [Ā - r_net,t · B̄] / (σ · T)
    where Ā, B̄ are running EMA estimates of mean and second moment.
    Used for memory-efficient backprop on very long sequences (T > 1000).
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

ANNUAL_FACTOR: float = 252.0 ** 0.5


# ── Smooth absolute value ─────────────────────────────────────────────────

def smooth_abs(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Pseudo-Huber smooth approximation to |x|.
    Gradient: x / √(x² + ε), bounded ∈ (-1, 1).
    Chose ε small (1e-6) to minimize bias vs |x| for TC > ~0.1%.
    """
    return torch.sqrt(x.pow(2) + eps)


def smooth_abs_sum(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Σ smooth_abs(x_i) — differentiable proxy for L1 norm."""
    return smooth_abs(x, eps).sum(dim=-1)


# ── Transaction cost model ────────────────────────────────────────────────

class TransactionCostModel(nn.Module):
    """
    Differentiable transaction cost estimator.

    Models: TC_t = c_fixed · ‖Δw_t‖₁_smooth + c_impact · ‖Δw_t‖₂²

    where:
        c_fixed  : proportional cost (bid-ask + commission), per unit turnover
        c_impact : market impact coefficient (quadratic, Kyle 1985)
        ‖Δw‖₁   : one-way turnover
        ‖Δw‖₂²  : squared turnover (impact grows with size²)

    The smooth L1 ensures gradient flow even at Δw ≈ 0.
    """

    def __init__(
        self,
        c_fixed: float  = 5e-4,      # 5bps one-way (ETF typical)
        c_impact: float = 1e-5,      # minimal for liquid ETFs
        eps: float      = 1e-6,
    ) -> None:
        super().__init__()
        # Make costs learnable if calibration data is available
        self.log_c_fixed  = nn.Parameter(torch.tensor(c_fixed).log())
        self.log_c_impact = nn.Parameter(torch.tensor(c_impact).log())
        self.eps = eps

    @property
    def c_fixed(self) -> torch.Tensor:
        return self.log_c_fixed.exp()

    @property
    def c_impact(self) -> torch.Tensor:
        return self.log_c_impact.exp()

    def forward(self, w_t: torch.Tensor, w_prev: torch.Tensor) -> torch.Tensor:
        """
        TC_t per timestep.
        w_t, w_prev : (..., N) weight tensors
        Returns     : (...,) scalar cost per timestep
        """
        delta = w_t - w_prev
        l1    = smooth_abs_sum(delta, self.eps)                      # (...,)
        l2sq  = delta.pow(2).sum(dim=-1)                             # (...,)
        return self.c_fixed * l1 + self.c_impact * l2sq


# ── Differentiable Sharpe Loss ────────────────────────────────────────────

class DifferentiableSharpe(nn.Module):
    """
    End-to-end differentiable Sharpe ratio loss with transaction costs.

    Usage pattern:
        1. Forward pass: model predicts weight sequence w (T, N)
        2. Compute r_net from realized returns R and TC model
        3. Compute SR = annualized Sharpe of r_net
        4. Backprop: -SR flows gradients through R → w → model params

    Parameters
    ----------
    tc_model        : TransactionCostModel instance
    min_periods     : minimum T before Sharpe is meaningful (guard divide-by-0)
    sharpe_eps      : std denominator floor (avoids blowup in trending regimes)
    annual_factor   : √252 for daily returns
    differential    : use online differential Sharpe estimator (memory-efficient)
    ema_decay       : EMA decay for differential Sharpe running stats
    """

    def __init__(
        self,
        tc_model: Optional[TransactionCostModel] = None,
        min_periods: int = 20,
        sharpe_eps: float = 1e-6,
        annual_factor: float = ANNUAL_FACTOR,
        differential: bool = False,
        ema_decay: float = 0.99,
    ) -> None:
        super().__init__()
        self.tc_model      = tc_model or TransactionCostModel()
        self.min_periods   = min_periods
        self.sharpe_eps    = sharpe_eps
        self.annual_factor = annual_factor
        self.differential  = differential

        # EMA state for online differential Sharpe
        self.register_buffer("_ema_mean",  torch.zeros(1))
        self.register_buffer("_ema_sq",    torch.zeros(1))
        self.register_buffer("_ema_count", torch.zeros(1))
        self.ema_decay = ema_decay

    def net_returns(
        self,
        weights: torch.Tensor,          # (T, N) predicted portfolio weights
        realized_ret: torch.Tensor,     # (T, N) realized asset returns
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-timestep net portfolio returns with transaction costs.

        r_net,t = Rₜᵀwₜ - TC(wₜ, w_{t-1})

        Handles w_{t-1} = 0 at t=0 (initial portfolio construction).
        Returns (r_net: (T,), r_gross: (T,)).
        """
        T, N = weights.shape

        # Prepend zero-weight initial state for TC at t=0
        w_prev = torch.cat([
            torch.zeros(1, N, device=weights.device, dtype=weights.dtype),
            weights[:-1]
        ], dim=0)                                                     # (T, N)

        r_gross = (weights * realized_ret).sum(dim=-1)               # (T,)
        tc      = self.tc_model(weights, w_prev)                     # (T,)
        r_net   = r_gross - tc

        return r_net, r_gross

    def sharpe(self, r_net: torch.Tensor) -> torch.Tensor:
        """
        Annualized Sharpe ratio — differentiable via standard autograd.
        Unbiased std (Bessel correction, ddof=1).
        """
        T = r_net.shape[0]
        if T < self.min_periods:
            # Return zero loss with gradient (not nan) to keep training stable
            return torch.zeros(1, device=r_net.device, requires_grad=True).squeeze()

        mean_r = r_net.mean()
        std_r  = r_net.std(unbiased=True).clamp(min=self.sharpe_eps)
        return self.annual_factor * mean_r / std_r

    def differential_sharpe(self, r_t: torch.Tensor) -> torch.Tensor:
        """
        Online differential Sharpe (Moody & Saffell, 2001).
        Avoids full-sequence unrolling — suitable for very long T or streaming.

        Updates running EMA of mean and E[r²] to approximate:
            SR ≈ Ā / √(B̄ - Ā²)    where Ā = EMA(r), B̄ = EMA(r²)
        """
        decay = self.ema_decay
        self._ema_mean = decay * self._ema_mean + (1 - decay) * r_t
        self._ema_sq   = decay * self._ema_sq   + (1 - decay) * r_t.pow(2)
        self._ema_count += 1

        # Bias correction
        bc   = 1.0 - decay ** self._ema_count
        A    = self._ema_mean / bc
        B    = self._ema_sq   / bc

        var  = (B - A.pow(2)).clamp(min=self.sharpe_eps)
        return self.annual_factor * A / var.sqrt()

    def forward(
        self,
        weights: torch.Tensor,          # (T, N)
        realized_ret: torch.Tensor,     # (T, N)
        regime_mask: Optional[torch.Tensor] = None,   # (T,) bool — exclude stress periods
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute -SR_net (loss to minimize).

        Returns (loss: scalar tensor, diagnostics: dict).
        """
        r_net, r_gross = self.net_returns(weights, realized_ret)

        if regime_mask is not None:
            # Mask certain timesteps (e.g., exclude fat-tail regime from signal training)
            r_net   = r_net[regime_mask]
            r_gross = r_gross[regime_mask]

        if self.differential:
            sr_val = self.differential_sharpe(r_net.mean())
        else:
            sr_val = self.sharpe(r_net)

        loss = -sr_val                                                 # maximize SR

        # ── Diagnostics (detached — no memory leak risk) ─────────────────
        with torch.no_grad():
            tc_total = (r_gross - r_net).mean()
            turnover = (weights - torch.cat([
                torch.zeros(1, weights.shape[-1], device=weights.device),
                weights[:-1]
            ], 0)).abs().sum(dim=-1).mean()

        diag = {
            "sharpe_net":   float(sr_val),
            "mean_r_net":   float(r_net.mean()),
            "std_r_net":    float(r_net.std()),
            "mean_tc":      float(tc_total),
            "mean_turnover": float(turnover),
        }

        return loss, diag


# ── Composite training loss ───────────────────────────────────────────────

class FortressLoss(nn.Module):
    """
    Full composite loss combining:
        1. -SR_net (primary objective)
        2. CVaR tail risk penalty
        3. Drawdown constraint (soft barrier)
        4. Concentration penalty (Herfindahl index)

    L_total = L_sharpe + λ_cvar · CVaR + λ_dd · DD_barrier + λ_hhi · HHI
    """

    def __init__(
        self,
        sharpe_loss: DifferentiableSharpe,
        lambda_cvar: float = 0.1,
        lambda_dd:   float = 5.0,     # high weight — drawdown is a kill switch
        lambda_hhi:  float = 0.05,
        cvar_alpha:  float = 0.95,
        max_dd:      float = 0.07,    # 7% soft barrier (firm limit is 8%)
    ) -> None:
        super().__init__()
        self.sharpe_loss = sharpe_loss
        self.lambda_cvar = lambda_cvar
        self.lambda_dd   = lambda_dd
        self.lambda_hhi  = lambda_hhi
        self.cvar_alpha  = cvar_alpha
        self.max_dd      = max_dd

    def _cvar_penalty(self, r_net: torch.Tensor) -> torch.Tensor:
        """ES at cvar_alpha — differentiable via sort."""
        T   = r_net.shape[0]
        k   = int((1 - self.cvar_alpha) * T)
        k   = max(k, 1)
        # Bottom-k returns (worst losses)
        tail = r_net.topk(k, largest=False).values
        return -tail.mean()                                           # ES = -mean(tail)

    def _drawdown_barrier(self, r_net: torch.Tensor) -> torch.Tensor:
        """
        Soft drawdown barrier: L_dd = ReLU(max_drawdown - max_dd)²
        Differentiable everywhere; zero when DD < limit.
        """
        cum_ret = (1.0 + r_net).log().cumsum(0).exp()                # cumulative return
        running_max = torch.cummax(cum_ret, dim=0).values
        dd = ((running_max - cum_ret) / (running_max + 1e-8)).max()
        return F.relu(dd - self.max_dd).pow(2)

    def _hhi_penalty(self, weights: torch.Tensor) -> torch.Tensor:
        """Herfindahl-Hirschman concentration index — penalizes large positions."""
        return weights.pow(2).sum(dim=-1).mean()

    def forward(
        self,
        weights: torch.Tensor,
        realized_ret: torch.Tensor,
        regime_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        sharpe_loss, diag = self.sharpe_loss(weights, realized_ret, regime_mask)
        r_net, _ = self.sharpe_loss.net_returns(weights, realized_ret)

        cvar_pen  = self._cvar_penalty(r_net)
        dd_pen    = self._drawdown_barrier(r_net)
        hhi_pen   = self._hhi_penalty(weights)

        total = (
            sharpe_loss
            + self.lambda_cvar * cvar_pen
            + self.lambda_dd   * dd_pen
            + self.lambda_hhi  * hhi_pen
        )

        diag.update({
            "cvar_penalty":  float(cvar_pen),
            "dd_penalty":    float(dd_pen),
            "hhi_penalty":   float(hhi_pen),
            "loss_total":    float(total),
        })

        return total, diag