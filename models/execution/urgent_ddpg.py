"""
FORTRESS v5 - urgent_ddpg.py
Path: models/execution/urgent_ddpg.py

Urgent Deep Deterministic Policy Gradient (DDPG) Execution Agent.
Objective: Minimize time-to-completion during risk-off events, guaranteeing fills.
Architecture ONLY. No training loops.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict

class DeterministicActor(nn.Module):
    """
    Unlike PPO which outputs a probability distribution, DDPG outputs a 
    single deterministic action. This is ideal for emergency execution where 
    we don't want stochastic exploration during a live market crash.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh() # Bounds the output between [-1, 1]
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Outputs a continuous aggressiveness vector.
        e.g., +1.0 = Max market sweep (take all available liquidity at any price)
              -1.0 = Passive limit (fallback if the crash suddenly reverses)
        """
        return self.net(state)


class QValueCritic(nn.Module):
    """
    Evaluates how good a specific deterministic action is, given the current state.
    Q(s, a) -> Expected future reward (which in this case is negative time-to-fill).
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        # The Critic takes BOTH the state and the action as input
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1) # Single Q-value
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa_cat = torch.cat([state, action], dim=-1)
        return self.net(sa_cat)


class UrgentDDPG(nn.Module):
    """
    The complete Urgent Execution Agent.
    
    Inputs (State Dimension = 12):
        - Current Bid-Ask Spread
        - Order Book Imbalance (Bids vs Asks)
        - Micro-price momentum (1-second, 5-second)
        - Remaining Inventory (%)
        - Urgency Multiplier (from the Meta-Controller / LTC network)
        
    Outputs:
        - Aggressiveness score [-1.0 to 1.0]
    """
    def __init__(self, config: Dict):
        super().__init__()
        
        self.state_dim = config.get('urgent_state_dim', 12)
        # Action 1: Aggressiveness (Price marketable limit offset)
        # Action 2: Sweep Size (How much to dump in this specific micro-second)
        self.action_dim = config.get('urgent_action_dim', 2)
        self.hidden_dim = config.get('urgent_hidden_dim', 128)
        
        # The DDPG architecture requires a Target Network for stable training
        # These are used strictly during train_execution.py to compute the Bellman error
        self.actor = DeterministicActor(self.state_dim, self.action_dim, self.hidden_dim)
        self.actor_target = DeterministicActor(self.state_dim, self.action_dim, self.hidden_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = QValueCritic(self.state_dim, self.action_dim, self.hidden_dim)
        self.critic_target = QValueCritic(self.state_dim, self.action_dim, self.hidden_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Freeze target networks with respect to optimizers
        for p in self.actor_target.parameters():
            p.requires_grad = False
        for p in self.critic_target.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def get_action(self, state: np.ndarray, noise_scale: float = 0.0) -> np.ndarray:
        """
        LIVE INFERENCE METHOD.
        Called by services/execution_svc.py when the Meta-Controller screams "SELL".
        """
        self.actor.eval()
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        action = self.actor(state_tensor)
        
        # Add exploration noise ONLY during training, never during live execution
        if noise_scale > 0.0:
            noise = torch.randn_like(action) * noise_scale
            action = torch.clamp(action + noise, -1.0, 1.0)
            
        return action.squeeze(0).numpy()

    def update_targets(self, tau: float = 0.005):
        """
        Soft update of the target networks.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Called during the training loop.
        """
        for target_param, local_param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
            
        for target_param, local_param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)