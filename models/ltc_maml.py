"""
fortress_v5/models/ltc_maml.py
───────────────────────────────
Liquid Time-Constant (LTC/CfC) nodes wrapping GATv2 with MAML meta-learning.

Architecture
────────────
Each GATv2 node aggregation output is passed through a CfC (Closed-Form
Continuous-time) cell instead of a vanilla MLP/GRU. CfC cells solve:

    h(t) = [ σ(-f(x,h)) · A ] · exp(-Δt · exp(τ(x,h))) + σ(-f(x,h)) · B

in closed form (no ODE solver required at inference), where Δt is the elapsed
calendar time between observations — directly encoding market microstructure
gaps (weekends, holidays, earnings blackouts).

MAML outer loop optimizes meta-parameters θ such that 2-5 gradient steps
on a small task-specific support set S (a recent regime episode) yield a
well-adapted θ' that generalizes to the query set Q.

Loss hierarchy:
    Inner: task-specific MSE on support set (few-shot adapt)
    Outer: sum of query-set losses at θ' (meta-objective)

`higher.innerloop_ctx` maintains a differentiable copy of the optimizer state,
enabling second-order MAML gradients to flow through the inner loop without
manual Jacobian computation.

Catastrophic forgetting mitigation:
    - EWC (Elastic Weight Consolidation) penalty on meta-parameters
    - Fisher information estimated on the recent task distribution
    - Reptile fallback (first-order) if memory budget is exceeded

Dependencies: ncps, torchdiffeq (CfC fallback), higher
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ncps provides CfC: the practical closed-form LTC implementation
try:
    from ncps.torch import CfC
    from ncps.wirings import AutoNCP
    _NCPS_AVAILABLE = True
except ImportError:
    _NCPS_AVAILABLE = False
    import warnings
    warnings.warn("ncps not installed — CfC will fall back to GRU. pip install ncps")

# higher for differentiable inner-loop
try:
    import higher
    _HIGHER_AVAILABLE = True
except ImportError:
    _HIGHER_AVAILABLE = False
    import warnings
    warnings.warn("higher not installed — MAML will use first-order Reptile. pip install higher")

log = logging.getLogger(__name__)


# ── CfC node encoder ──────────────────────────────────────────────────────

class LTCNodeEncoder(nn.Module):
    """
    Replaces the GATv2 per-node MLP with a CfC temporal cell.

    CfC advantages over GRU/LSTM for financial time series:
    1. Δt-awareness: models irregular sampling (holidays, micro-structure gaps)
    2. Continuous-time dynamics: interpolates between observations on a manifold
    3. Faster wall-clock than neural ODE (no Dormand-Prince solver)

    Parameters
    ----------
    input_dim   : GATv2 aggregated node feature dimension
    hidden_dim  : CfC hidden state size
    output_dim  : signal embedding dimension
    n_neurons   : NCP motor neuron count (AutoNCP wiring)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 32,
        n_neurons: int = 16,
    ) -> None:
        super().__init__()

        if _NCPS_AVAILABLE:
            # AutoNCP wiring: sparse, biologically-inspired connectivity
            wiring = AutoNCP(hidden_dim, n_neurons)
            self.rnn = CfC(input_dim, wiring, batch_first=True, return_sequences=True)
        else:
            # Graceful GRU fallback
            self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
        )
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def forward(
        self,
        x: torch.Tensor,                          # (B, T, input_dim)
        time_delta: Optional[torch.Tensor] = None, # (B, T) elapsed time in days
        hx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (embeddings: (B, T, output_dim), h_T: (B, hidden_dim)).
        """
        if _NCPS_AVAILABLE and time_delta is not None:
            out, h_n = self.rnn(x, hx=hx, timespans=time_delta)
        else:
            out, h_n = self.rnn(x, hx)

        if isinstance(h_n, tuple):
            h_n = h_n[0]                                              # GRU compat

        # Take last hidden state for sequence output
        h_n = h_n.squeeze(0)                                          # (B, hidden_dim)
        emb = self.proj(out)                                          # (B, T, output_dim)
        return emb, h_n


# ── EWC Fisher penalty ────────────────────────────────────────────────────

class EWCRegularizer:
    """
    Elastic Weight Consolidation — penalizes deviation from prior task params
    weighted by Fisher information. Prevents catastrophic forgetting across
    regime transitions.

    F̂_θ ≈ (1/|T|) Σ_t (∂ log p(y_t|θ)/∂θ)²   [diagonal Fisher approximation]
    L_EWC = λ/2 · Σ_i F̂_i · (θ_i - θ*_i)²
    """

    def __init__(self, lambda_ewc: float = 1000.0) -> None:
        self.lambda_ewc  = lambda_ewc
        self._theta_star: Optional[dict] = None
        self._fisher:     Optional[dict] = None

    def consolidate(
        self,
        model: nn.Module,
        task_data: Tuple[torch.Tensor, ...],
        loss_fn: nn.Module,
    ) -> None:
        """
        Estimate Fisher information on current task and snapshot parameters.
        Call at end of each regime episode.
        """
        model.eval()
        model.zero_grad()

        x, *rest = task_data
        pred = model(x)
        loss = loss_fn(pred, *rest)
        loss.backward()

        # Diagonal Fisher: squared gradient per parameter
        self._fisher = {
            name: (p.grad.data.clone() ** 2)
            for name, p in model.named_parameters()
            if p.grad is not None
        }
        self._theta_star = {
            name: p.data.clone()
            for name, p in model.named_parameters()
        }
        model.zero_grad()

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """L_EWC to add to the outer-loop loss."""
        if self._fisher is None:
            return torch.tensor(0.0, requires_grad=False)

        penalty = torch.tensor(0.0, device=next(model.parameters()).device)
        for name, p in model.named_parameters():
            if name in self._fisher:
                penalty += (self._fisher[name] * (p - self._theta_star[name]) ** 2).sum()

        return 0.5 * self.lambda_ewc * penalty


# ── MAML wrapper ─────────────────────────────────────────────────────────

@dataclass
class MAMLTask:
    """A single regime episode: (support, query) splits."""
    support_x: torch.Tensor    # (K_s, T, D) few-shot support features
    support_y: torch.Tensor    # (K_s, N) support targets
    query_x:   torch.Tensor    # (K_q, T, D) query features
    query_y:   torch.Tensor    # (K_q, N) query targets
    time_delta: Optional[torch.Tensor] = None


class MAMLSignalAdapter(nn.Module):
    """
    MAML meta-learner wrapping LTCNodeEncoder (or any base model).

    Meta-objective:
        θ* = argmin_θ Σ_τ L_τ(U^k(θ, S_τ), Q_τ)
    where U^k is k inner gradient steps, S_τ support set, Q_τ query set.

    At inference, the model adapts to the current regime episode in 2-5 steps
    without forgetting the meta-prior — enabling rapid structural break response.

    Parameters
    ----------
    base_model      : LTCNodeEncoder or any nn.Module with compatible I/O
    inner_lr        : task-specific adaptation learning rate
    inner_steps     : number of adaptation gradient steps
    outer_lr        : meta-optimizer learning rate
    use_higher      : second-order MAML (True) or first-order Reptile (False)
    ewc_lambda      : EWC regularization strength (0 = disabled)
    """

    def __init__(
        self,
        base_model: nn.Module,
        inner_lr: float = 0.01,
        inner_steps: int = 3,
        outer_lr: float = 1e-4,
        use_higher: bool = True,
        ewc_lambda: float = 500.0,
    ) -> None:
        super().__init__()
        self.model       = base_model
        self.inner_lr    = inner_lr
        self.inner_steps = inner_steps
        self.use_higher  = use_higher and _HIGHER_AVAILABLE

        # Meta-optimizer (outer loop)
        self.meta_optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=outer_lr, weight_decay=1e-4
        )

        # EWC consolidation for catastrophic forgetting
        self.ewc = EWCRegularizer(lambda_ewc=ewc_lambda)

    def _inner_loss(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        time_delta: Optional[torch.Tensor],
    ) -> torch.Tensor:
        emb, _ = model(x, time_delta)
        pred   = emb[:, -1, :]                                        # use final timestep embedding
        # MSE on signal embedding — substitute task-specific head loss here
        return F.mse_loss(pred, y)

    def meta_train_step(self, tasks: List[MAMLTask]) -> float:
        """
        One outer-loop meta-gradient step over a batch of tasks.
        Returns: scalar outer loss (float, for logging).
        """
        self.meta_optimizer.zero_grad()
        outer_loss = torch.tensor(0.0, device=next(self.model.parameters()).device)

        if self.use_higher:
            outer_loss = self._maml_second_order(tasks)
        else:
            outer_loss = self._reptile_step(tasks)

        # EWC penalty on meta-parameters
        if tasks:
            outer_loss = outer_loss + self.ewc.penalty(self.model)

        outer_loss.backward()

        # Gradient clipping — second-order MAML gradients can be large
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.meta_optimizer.step()

        return float(outer_loss.detach())

    def _maml_second_order(self, tasks: List[MAMLTask]) -> torch.Tensor:
        """
        Full MAML (Finn et al. 2017) with `higher` differentiable inner loop.
        Second-order gradients flow through the adaptation steps back to θ.
        """
        device       = next(self.model.parameters()).device
        outer_loss   = torch.zeros(1, device=device)
        inner_opt    = torch.optim.SGD(self.model.parameters(), lr=self.inner_lr)

        for task in tasks:
            sx = task.support_x.to(device)
            sy = task.support_y.to(device)
            qx = task.query_x.to(device)
            qy = task.query_y.to(device)
            td = task.time_delta.to(device) if task.time_delta is not None else None

            # `higher` tracks gradient history through inner SGD steps
            with higher.innerloop_ctx(
                self.model, inner_opt,
                copy_initial_weights=False,
                track_higher_grads=True,
            ) as (f_model, diff_inner_opt):

                for _ in range(self.inner_steps):
                    s_loss = self._inner_loss(f_model, sx, sy, td)
                    diff_inner_opt.step(s_loss)

                # Query loss at θ' (adapted params) — gradients flow to θ
                q_loss = self._inner_loss(f_model, qx, qy, td)
                outer_loss = outer_loss + q_loss

        return outer_loss / max(len(tasks), 1)

    def _reptile_step(self, tasks: List[MAMLTask]) -> torch.Tensor:
        """
        First-order Reptile (Nichol et al. 2018) — O(k) memory vs O(k²) MAML.
        θ ← θ + ε · (θ' - θ) aggregated over tasks.
        """
        device    = next(self.model.parameters()).device
        meta_loss = torch.zeros(1, device=device)

        for task in tasks:
            # Clone model for task-specific adaptation
            task_model = copy.deepcopy(self.model)
            task_opt   = torch.optim.SGD(task_model.parameters(), lr=self.inner_lr)

            sx = task.support_x.to(device)
            sy = task.support_y.to(device)
            td = task.time_delta.to(device) if task.time_delta is not None else None

            for _ in range(self.inner_steps):
                task_opt.zero_grad()
                loss = self._inner_loss(task_model, sx, sy, td)
                loss.backward()
                task_opt.step()

            # Query evaluation for logging
            with torch.no_grad():
                q_loss = self._inner_loss(
                    task_model,
                    task.query_x.to(device),
                    task.query_y.to(device),
                    td
                )
            meta_loss = meta_loss + q_loss

            # Reptile: move meta-params toward task-adapted params
            with torch.no_grad():
                for p_meta, p_task in zip(self.model.parameters(), task_model.parameters()):
                    p_meta.grad = p_meta.data - p_task.data           # direction toward θ'

        # Outer step applies these pseudo-gradients
        # (meta_optimizer.step() called by caller)
        return meta_loss / max(len(tasks), 1)

    @torch.no_grad()
    def adapt_and_predict(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        time_delta: Optional[torch.Tensor] = None,
        n_steps: int = 3,
    ) -> torch.Tensor:
        """
        At-inference few-shot adaptation: run n_steps on the current regime
        support set, then predict on query_x.

        CAUTION: operates on a deep copy — meta-parameters are never mutated
        at inference time. The adapted θ' is ephemeral.
        """
        device      = next(self.model.parameters()).device
        adapt_model = copy.deepcopy(self.model).to(device)
        adapt_opt   = torch.optim.SGD(adapt_model.parameters(), lr=self.inner_lr)

        adapt_model.train()
        for _ in range(n_steps):
            adapt_opt.zero_grad()
            loss = self._inner_loss(
                adapt_model,
                support_x.to(device),
                support_y.to(device),
                time_delta.to(device) if time_delta is not None else None,
            )
            loss.backward()
            adapt_opt.step()

        adapt_model.eval()
        emb, h_n = adapt_model(
            query_x.to(device),
            time_delta.to(device) if time_delta is not None else None,
        )
        return emb[:, -1, :]                                          # (B, output_dim)