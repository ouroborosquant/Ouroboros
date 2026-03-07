"""
FORTRESS v5 - opportunistic_sac.py
Path: models/execution/opportunistic_sac.py

Opportunistic Soft Actor-Critic (SAC) Execution Agent.
Objective: Capture the spread and provide liquidity when the book is calm, 
but rapidly cross the spread if adverse momentum builds.
Architecture ONLY. No training loops.
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict

class GaussianActor(nn.Module):
    """
    Outputs a stochastic policy (mean and log_std) for continuous actions.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish()
        )
        self.mu_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

        # Action bounds for numerical stability
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.net(state)
        mu = self.mu_layer(features)
        
        log_std = self.log_std_layer(features)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)
        
        return mu, std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples an action using the reparameterization trick."""
        mu, std = self.forward(state)
        
        dist = torch.distributions.Normal(mu, std)
        x_t = dist.rsample() # Reparameterization trick
        
        # Enforce action bounds [-1, 1] using Tanh
        y_t = torch.tanh(x_t)
        action = y_t
        
        # Calculate log probability (with correction for the Tanh squashing)
        log_prob = dist.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob


class TwinCritic(nn.Module):
    """
    SAC uses two separate Q-networks to mitigate the overestimation bias 
    inherent in standard Q-learning.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        # Q-Network 1
        self.q1_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Q-Network 2
        self.q2_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], 1)
        q1 = self.q1_net(sa)
        q2 = self.q2_net(sa)
        return q1, q2


class OpportunisticSAC(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.state_dim = config.get('opport_state_dim', 12)
        self.action_dim = config.get('opport_action_dim', 2)
        self.hidden_dim = config.get('opport_hidden_dim', 128)
        
        # Core Networks
        self.actor = GaussianActor(self.state_dim, self.action_dim, self.hidden_dim)
        self.critic = TwinCritic(self.state_dim, self.action_dim, self.hidden_dim)
        self.critic_target = TwinCritic(self.state_dim, self.action_dim, self.hidden_dim)
        
        # Initialize target networks to match local networks
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Freeze target networks
        for p in self.critic_target.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        LIVE INFERENCE METHOD.
        Called by execution_svc.py.
        """
        self.eval()
        
        if not isinstance(state, torch.Tensor):
            import numpy as np
            state = torch.FloatTensor(state).unsqueeze(0)
            
        if deterministic:
            # During live trading, we bypass the sampling and just use the mean
            mu, _ = self.actor(state)
            return torch.tanh(mu).squeeze(0).numpy()
        else:
            # During self-play exploration, we sample the distribution
            action, _ = self.actor.sample(state)
            return action.squeeze(0).numpy()