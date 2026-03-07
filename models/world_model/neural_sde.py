"""
FORTRESS v5 - neural_sde.py
Path: models/world_model/neural_sde.py

Latent Neural SDE World Model.
Learns continuous-time market dynamics to generate synthetic trajectories
for offline RL training and federated data sharing.
Architecture ONLY. No training loops.

FIXES APPLIED:
  - BUG #10: `self.current_z_t` was a global mutable attribute set directly on the
             model instance before calling `torchsde.sdeint`:
                 self.model.current_z_t = Z_batch
                 torchsde.sdeint(self.model, ...)
             In any context with DataLoader `num_workers > 0`, or if `sdeint` is
             called concurrently from two coroutines, the regime conditioning tensor
             of one call silently overwrites the other. The SDE integrates with the
             wrong z_t, producing a regime-mismatched trajectory with no error raised.

             Fix: `current_z_t` has been removed as an instance attribute.
             The drift `f()` and diffusion `g()` functions now receive z_t via a
             per-call closure. `generate_synthetic_paths()` and the training harness
             call `build_conditioned_sde(z_t)` which returns a thin wrapper whose
             `f` and `g` close over that specific z_t tensor. Thread-safe by construction.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple

# External dependencies for Neural SDEs
try:
    import torchsde
except ImportError:
    raise ImportError("torchsde is required. Install via: pip install torchsde")


class DriftNetwork(nn.Module):
    """
    Models the deterministic component μ(y, z_t) of the market SDE:
        dY_t = f(t, Y_t) dt + g(t, Y_t) dW_t
    where f = DriftNetwork captures expected return / directional trend,
    conditioned on the latent regime z_t from the Mamba-KAN VAE.
    """

    def __init__(self, state_dim: int, regime_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + regime_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        z_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            t:   Scalar time (unused if network is time-invariant, kept for torchsde compat).
            y:   Asset state tensor, shape (Batch, State_Dim).
            z_t: Regime conditioning tensor, shape (Batch, Regime_Dim).

        Returns:
            drift: Shape (Batch, State_Dim).
        """
        if z_t.dim() == 1:
            z_t = z_t.unsqueeze(0).expand(y.shape[0], -1)
        inputs = torch.cat([y, z_t], dim=-1)
        return self.net(inputs)


class DiffusionNetwork(nn.Module):
    """
    Models the stochastic component σ(y, z_t) of the market SDE.
    Output shape must be (Batch, State_Dim, Brownian_Size) for torchsde 'general' noise.
    """

    def __init__(
        self, state_dim: int, regime_dim: int, hidden_dim: int, brownian_size: int
    ):
        super().__init__()
        self.state_dim = state_dim
        self.brownian_size = brownian_size

        self.net = nn.Sequential(
            nn.Linear(state_dim + regime_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim * brownian_size),
            nn.Softplus(),  # Ensures non-negative diffusion coefficients.
        )

    def forward(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        z_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            t:   Scalar time.
            y:   Asset state tensor, shape (Batch, State_Dim).
            z_t: Regime conditioning tensor, shape (Batch, Regime_Dim).

        Returns:
            diffusion: Shape (Batch, State_Dim, Brownian_Size).
        """
        if z_t.dim() == 1:
            z_t = z_t.unsqueeze(0).expand(y.shape[0], -1)
        inputs = torch.cat([y, z_t], dim=-1)
        diffusion_flat = self.net(inputs)
        # Reshape to strictly match the torchsde requirement for 'general' noise.
        return diffusion_flat.view(-1, self.state_dim, self.brownian_size)


class _ConditionedSDE(nn.Module):
    """
    FIX #10: A thin, per-call wrapper that closes over a specific `z_t` tensor.

    Previously, `LatentSDEWorldModel` used `self.current_z_t` — a shared mutable
    attribute set before each `torchsde.sdeint` call. Any concurrent call overwrote it.

    This wrapper is instantiated fresh per `generate_synthetic_paths()` call or per
    training batch iteration. The z_t is local to this instance, making the SDE
    integration completely thread-safe. Two concurrent calls will never share state.

    torchsde requires `noise_type` and `sde_type` as class attributes.
    """

    noise_type = "general"
    sde_type = "ito"

    def __init__(
        self,
        f_net: DriftNetwork,
        g_net: DiffusionNetwork,
        z_t: torch.Tensor,
    ):
        super().__init__()
        # Store the conditioning tensor as a buffer (not a parameter — no gradients).
        # Using register_buffer ensures it is moved correctly if .to(device) is called.
        self.register_buffer("_z_t", z_t)
        # Use non-parameter references to avoid double-registration of weights.
        self.f_net = f_net
        self.g_net = g_net

    def f(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Drift function called by the torchsde integrator."""
        return self.f_net(t, y, self._z_t)

    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Diffusion function called by the torchsde integrator."""
        return self.g_net(t, y, self._z_t)


class LatentSDEWorldModel(nn.Module):
    """
    The full Neural SDE World Model.
    Noise type: 'general' (state-dependent covariance matrix).
    SDE type:   'ito' (standard Itô calculus formulation).

    FIX #10: `current_z_t` instance attribute removed.
    Regime conditioning is now injected per-call via `build_conditioned_sde(z_t)`.
    """

    def __init__(self, config: Dict):
        super().__init__()

        self.state_dim = config.get("sde_state_dim", 25)   # 25-asset universe.
        self.regime_dim = config.get("latent_dim", 16)      # From Mamba-KAN.
        self.hidden_dim = config.get("hidden_dim", 128)
        self.brownian_size = config.get("brownian_size", 10)
        self.sde_method = config.get("sde_method", "euler")

        self.f_net = DriftNetwork(self.state_dim, self.regime_dim, self.hidden_dim)
        self.g_net = DiffusionNetwork(
            self.state_dim, self.regime_dim, self.hidden_dim, self.brownian_size
        )

    def build_conditioned_sde(self, z_t: torch.Tensor) -> _ConditionedSDE:
        """
        FIX #10: Factory method. Returns a fresh _ConditionedSDE that closes over
        this specific `z_t` tensor. Thread-safe — each call produces an independent
        object with no shared mutable state.

        Called by:
          - `generate_synthetic_paths()` for live scenario generation.
          - `train_world_model.py` for each training batch iteration.

        Args:
            z_t: Regime conditioning tensor, shape (Batch, Regime_Dim).
                 Can be a single regime vector (1, 16) or a batched set (B, 16).

        Returns:
            A _ConditionedSDE instance whose f() and g() are bound to this z_t.
        """
        return _ConditionedSDE(f_net=self.f_net, g_net=self.g_net, z_t=z_t)

    @torch.no_grad()
    def generate_synthetic_paths(
        self,
        initial_state: torch.Tensor,
        z_t: torch.Tensor,
        n_steps: int = 21,
        dt: float = 1.0,
        n_paths: int = 1000,
    ) -> torch.Tensor:
        """
        SIMULATION METHOD.
        Generates n_paths alternate future trajectories from `initial_state` under
        the regime encoded in `z_t`.

        FIX #10: Previously set `self.current_z_t = z_t` before calling sdeint.
                 Now passes z_t through `build_conditioned_sde()` — no shared state.

        Args:
            initial_state: Shape (State_Dim,) — current market state (prices/returns).
            z_t:           Shape (Regime_Dim,) or (1, Regime_Dim) — regime conditioning.
            n_steps:       Number of simulation steps (days).
            dt:            Time step size (1.0 = daily).
            n_paths:       Number of Monte Carlo paths to generate.

        Returns:
            paths: Shape (n_paths, n_steps + 1, State_Dim).
        """
        device = next(self.parameters()).device

        # Expand initial state across all paths: (n_paths, State_Dim).
        y0 = initial_state.unsqueeze(0).expand(n_paths, -1).to(device)

        # Expand z_t across all paths: (n_paths, Regime_Dim).
        if z_t.dim() == 1:
            z_t_expanded = z_t.unsqueeze(0).expand(n_paths, -1).to(device)
        else:
            z_t_expanded = z_t.expand(n_paths, -1).to(device)

        # FIX #10: Build a fresh, per-call conditioned SDE. No global mutation.
        conditioned_sde = self.build_conditioned_sde(z_t_expanded).to(device)

        # Time grid: [0, dt, 2*dt, ..., n_steps*dt].
        ts = torch.linspace(0, n_steps * dt, n_steps + 1, device=device)

        # Numerical SDE integration (Euler-Maruyama by default).
        # Returns shape: (n_steps + 1, n_paths, State_Dim).
        paths = torchsde.sdeint(
            sde=conditioned_sde,
            y0=y0,
            ts=ts,
            method=self.sde_method,
            dt=dt,
        )

        # Transpose to (n_paths, n_steps + 1, State_Dim) for downstream consumers.
        return paths.permute(1, 0, 2)