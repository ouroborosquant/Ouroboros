"""
FORTRESS v5 - meta_controller.py
Path: models/execution/meta_controller.py

MARL Execution Meta-Controller.
Dynamically routes sub-orders to the optimal execution agent (Stealth, Urgent, or Opportunistic)
based on current market microstructure and macro regime alerts.
Architecture ONLY. No training loops.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple

class ExecutionMetaController(nn.Module):
    """
    The Meta-Controller is a lightweight policy network.
    It must be extremely fast (<1ms latency) to evaluate tick-by-tick order book changes.
    
    Inputs:
        - z_t: 16-dim latent regime posterior
        - tda_alert: Binary flag from FPGA (0 or 1)
        - ltc_urgency: Continuous float [0, 1] from the intraday Liquid Neural Net
        - spread_z: Current bid-ask spread z-score (liquidity proxy)
        - vol_pace: 5-minute volume momentum
        
    Outputs:
        - Softmax probabilities across the 3 execution agents:
          [P(Stealth), P(Urgent), P(Opportunistic)]
    """
    
    AGENT_NAMES = ['stealth_ppo', 'urgent_ddpg', 'opportunistic_sac']
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Input dimension: 16 (z_t) + 1 (tda) + 1 (ltc) + 1 (spread) + 1 (vol) = 20
        self.input_dim = config.get('meta_input_dim', 20)
        self.hidden_dim = config.get('meta_hidden_dim', 64)
        
        # Lightweight Feed-Forward Network for sub-millisecond inference
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.Mish(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.Mish(),
            nn.Linear(self.hidden_dim // 2, 3) # 3 specialized agents
        )
        
    def forward(self, state_vector: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning log probabilities for the agents.
        state_vector shape: (Batch, 20)
        """
        logits = self.net(state_vector)
        return F.softmax(logits, dim=-1)

    @torch.no_grad()
    def select_agent(self, z_t: np.ndarray, tda_alert: bool, ltc_urgency: float, 
                     spread_z: float, vol_pace: float, 
                     deterministic: bool = True) -> str:
        """
        LIVE INFERENCE METHOD.
        Called by services/execution_svc.py for every child order slice.
        
        Args:
            deterministic: If True, picks the argmax. If False, samples from the distribution
                           (useful for exploration during self-play training).
        """
        self.eval()
        
        # Construct the state vector
        # Note: tda_alert is cast to float (1.0 for True, 0.0 for False)
        scalar_features = np.array([float(tda_alert), ltc_urgency, spread_z, vol_pace], dtype=np.float32)
        
        # Combine the 16-dim regime vector with the 4 scalar features
        state = np.concatenate([z_t, scalar_features])
        state_tensor = torch.FloatTensor(state).unsqueeze(0) # (1, 20)
        
        # Get routing probabilities
        probs = self.forward(state_tensor).squeeze(0).numpy()
        
        # Hard Safety Override:
        # If the FPGA detects a structural topological breakdown OR the LTC detects severe 
        # intraday drift, we mathematically force the Urgent agent, bypassing the neural network.
        if tda_alert or ltc_urgency > 0.85:
            return 'urgent_ddpg'
            
        if deterministic:
            agent_idx = np.argmax(probs)
        else:
            agent_idx = np.random.choice(3, p=probs)
            
        return self.AGENT_NAMES[agent_idx]

    def compute_loss(self, state: torch.Tensor, chosen_agent: torch.Tensor, 
                     implementation_shortfall: torch.Tensor) -> torch.Tensor:
        """
        Loss function for REINFORCE (Policy Gradient) training.
        The goal is to minimize the implementation shortfall (execution cost).
        
        Args:
            state: (Batch, 20)
            chosen_agent: (Batch,) - Integer index of the agent that was used
            implementation_shortfall: (Batch,) - The realized cost in basis points
        """
        probs = self.forward(state)
        
        # Get the probability of the agent that was actually selected
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(chosen_agent)
        
        # Reward is negative shortfall (we want to minimize costs)
        # We use standard REINFORCE objective: -E[log_prob * reward]
        reward = -implementation_shortfall
        
        loss = -(log_probs * reward).mean()
        return loss