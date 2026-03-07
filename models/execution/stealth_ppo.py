"""
FORTRESS v5 - stealth_ppo.py
Path: models/execution/stealth_ppo.py

Stealth Proximal Policy Optimization (PPO) Execution Agent.
Objective: Execute parent orders with zero market impact and minimal detectability.
Architecture ONLY. No training loops.
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict

class ActorNetwork(nn.Module):
    """
    The Policy network. 
    Observes the Level-2 order book and remaining inventory, and outputs actions.
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
        
        # Action 1: Limit Price Offset (from the mid-price, in basis points)
        # Action 2: Order Size % (fraction of remaining inventory to execute now)
        self.action_mean = nn.Linear(hidden_dim, action_dim)
        
        # Trainable standard deviation for continuous action exploration
        self.action_log_std = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the mean and standard deviation of the action distribution.
        """
        features = self.net(state)
        action_mean = self.action_mean(features)
        
        # Tanh bounds the actions [-1, 1], which are then scaled by the environment
        action_mean = torch.tanh(action_mean) 
        
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        
        return action_mean, action_std


class CriticNetwork(nn.Module):
    """
    The Value network.
    Estimates the Expected Implementation Shortfall from the current state.
    Used to compute Advantages during PPO training.
    """
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1) # Outputs a single scalar value V(s)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class StealthPPO(nn.Module):
    """
    The complete Stealth Execution Agent.
    
    Inputs (State Dimension = 12):
        - Current Bid-Ask Spread
        - Order Book Imbalance (Bids vs Asks)
        - Micro-price momentum (1-second, 5-second)
        - Remaining Inventory (%)
        - Remaining Time in Execution Window (%)
        - Detectability Penalty (Rolling uniformity of recent slice sizes)
        
    Outputs:
        - Limit price placement
        - Execution size
    """
    def __init__(self, config: Dict):
        super().__init__()
        
        # Standard Level-2 Microstructure state space
        self.state_dim = config.get('stealth_state_dim', 12)
        # 2 continuous actions: [Price Offset, Size Fraction]
        self.action_dim = config.get('stealth_action_dim', 2)
        self.hidden_dim = config.get('stealth_hidden_dim', 128)
        
        # The penalty applied to the reward function if the agent's trades become predictable
        self.detectability_lambda = config.get('detectability_lambda', 0.15)
        
        self.actor = ActorNetwork(self.state_dim, self.action_dim, self.hidden_dim)
        self.critic = CriticNetwork(self.state_dim, self.hidden_dim)

    @torch.no_grad()
    def get_action(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        LIVE INFERENCE METHOD.
        Called by services/execution_svc.py when the Meta-Controller delegates to Stealth.
        """
        self.eval()
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        action_mean, action_std = self.actor(state_tensor)
        
        if deterministic:
            # Live trading: exploit the learned policy directly
            action = action_mean
        else:
            # Self-play training: explore using the standard deviation
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()
            
        return action.squeeze(0).numpy()

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Used during the PPO update phase (train_execution.py) to compute 
        the Clipped Surrogate Objective and Value Loss.
        """
        action_mean, action_std = self.actor(states)
        dist = torch.distributions.Normal(action_mean, action_std)
        
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        state_values = self.critic(states).squeeze(-1)
        
        return log_probs, state_values, entropy