"""
FORTRESS v5 - neural_sde.py
Path: models/world_model/neural_sde.py

Latent Neural SDE World Model.
Learns continuous-time market dynamics to generate synthetic trajectories 
for offline RL training and federated data sharing.
Architecture ONLY. No training loops.
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
    Models the deterministic component of the market dynamics (f_theta).
    This captures the expected return / directional trend.
    """
    def __init__(self, state_dim: int, regime_dim: int, hidden_dim: int):
        super().__init__()
        # The network takes the current market state and the latent regime context
        self.net = nn.Sequential(
            nn.Linear(state_dim + regime_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, t: torch.Tensor, y: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        """
        t: Current timestep
        y: Current market state (Batch, State_Dim)
        z_t: Current Mamba-KAN regime latent (Batch, Regime_Dim)
        """
        # Concatenate state and regime context
        # Expand z_t to match the batch dimension if generating multiple paths
        if z_t.dim() == 1:
            z_t = z_t.unsqueeze(0).expand(y.shape[0], -1)
            
        inputs = torch.cat([y, z_t], dim=-1)
        return self.net(inputs)


class DiffusionNetwork(nn.Module):
    """
    Models the stochastic component of the market dynamics (g_phi).
    This captures volatility, correlations, and Brownian shocks (dW_t).
    """
    def __init__(self, state_dim: int, regime_dim: int, hidden_dim: int, brownian_size: int):
        super().__init__()
        self.state_dim = state_dim
        self.brownian_size = brownian_size
        
        self.net = nn.Sequential(
            nn.Linear(state_dim + regime_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            # Output maps to a matrix of shape (state_dim, brownian_size)
            nn.Linear(hidden_dim, state_dim * brownian_size)
        )

    def forward(self, t: torch.Tensor, y: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        """
        Returns the diffusion matrix to be multiplied by the Brownian motion.
        Output shape must be (Batch, State_Dim, Brownian_Size).
        """
        if z_t.dim() == 1:
            z_t = z_t.unsqueeze(0).expand(y.shape[0], -1)
            
        inputs = torch.cat([y, z_t], dim=-1)
        diffusion_flat = self.net(inputs)
        
        # Reshape to strictly match the torchsde requirement for matrix outputs
        return diffusion_flat.view(-1, self.state_dim, self.brownian_size)


class LatentSDEWorldModel(nn.Module):
    """
    The full Neural SDE wrapper utilizing the torchsde numerical solvers.
    Noise type: 'general' (allows complex state-dependent correlations).
    SDE type: 'ito' (standard Itô calculus formulation).
    """
    noise_type = 'general'
    sde_type = 'ito'

    def __init__(self, config: Dict):
        super().__init__()
        
        self.state_dim = config.get('sde_state_dim', 25) # e.g., simulating 25 asset returns
        self.regime_dim = config.get('latent_dim', 16)   # From Mamba-KAN
        self.hidden_dim = config.get('hidden_dim', 128)
        self.brownian_size = config.get('brownian_size', 10) # Number of independent shock factors
        self.sde_method = config.get('sde_method', 'euler')  # 'euler', 'milstein', 'srk'
        
        # We must explicitly define f and g for the torchsde.sdeint solver
        self.f_net = DriftNetwork(self.state_dim, self.regime_dim, self.hidden_dim)
        self.g_net = DiffusionNetwork(self.state_dim, self.regime_dim, self.hidden_dim, self.brownian_size)
        
        # State required by torchsde to pass external context (z_t) cleanly
        self.current_z_t = None

    def f(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Drift function called internally by the solver."""
        return self.f_net(t, y, self.current_z_t)

    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Diffusion function called internally by the solver."""
        return self.g_net(t, y, self.current_z_t)

    @torch.no_grad()
    def generate_synthetic_paths(self, initial_state: torch.Tensor, z_t: torch.Tensor, 
                                 n_steps: int = 21, dt: float = 1.0, 
                                 n_paths: int = 1000) -> torch.Tensor:
        """
        SIMULATION METHOD.
        Generates thousands of alternate future trajectories based on the current regime.
        
        Args:
            initial_state: Starting market prices/features (State_Dim,)
            z_t: The current Mamba-KAN regime vector (Regime_Dim,)
            n_steps: Number of forward timesteps to simulate (e.g., 21 trading days)
            dt: Step size
            n_paths: Number of Monte Carlo paths to generate
            
        Returns:
            Tensor of shape (n_steps, n_paths, State_Dim)
        """
        self.eval()
        
        # Set the latent context for the solver
        self.current_z_t = z_t.to(initial_state.device)
        
        # Expand initial state for batch processing
        y0 = initial_state.unsqueeze(0).expand(n_paths, -1)
        
        # Define the time grid
        t_grid = torch.linspace(0, n_steps * dt, n_steps + 1, device=initial_state.device)
        
        # Solve the SDE forward in time
        # The solver handles the complex stochastic calculus automatically
        with torch.no_grad():
            synthetic_trajectories = torchsde.sdeint(
                self, 
                y0, 
                t_grid, 
                method=self.sde_method, 
                dt=dt
            )
            
        return synthetic_trajectories