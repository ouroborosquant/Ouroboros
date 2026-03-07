"""
FORTRESS v5 - liquid_neural_net.py
Path: models/regime/liquid_neural_net.py

Continuous-time intraday regime drift detector.
Uses Liquid Time-Constant (LTC) networks to process irregular market ticks.
Architecture ONLY. No training loops.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Optional

# External dependencies for Liquid Neural Networks
try:
    from ncps.torch import LTC
    from ncps.wirings import AutoNCP
except ImportError:
    raise ImportError("ncps is required. Install via: pip install ncps")


class IntraRegimeMonitor(nn.Module):
    """
    Intraday regime monitor utilizing Neural Circuit Policies (NCPs) and LTCs.
    
    This network does not output a regime class; instead, it outputs an 
    `urgency_score` [0, 1] representing the probability that the current intraday
    microstructure has dangerously decoupled from the morning's expected regime.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.input_size = config.get('input_size', 15)
        self.n_units = config.get('n_units', 64)
        self.sparsity_level = config.get('sparsity_level', 0.50)
        
        # Thresholds loaded from hyperparams.yaml
        self.urgency_threshold = config.get('urgency_threshold', 0.70)
        
        # 1. Wiring: AutoNCP creates a brain-inspired sparse neural circuit
        # We need 1 output: the drift/urgency score
        self.wiring = AutoNCP(
            units=self.n_units, 
            output_size=1, 
            sparsity_level=self.sparsity_level
        )
        
        # 2. Liquid Time-Constant (LTC) core
        # batch_first=True aligns with standard PyTorch time-series shape (B, T, F)
        self.ltc = LTC(
            input_size=self.input_size, 
            wiring=self.wiring, 
            batch_first=True
        )
        
        # The hidden state persists across intraday ticks, but must be reset daily
        self.hidden_state: Optional[torch.Tensor] = None

    def reset_hidden_state(self):
        """
        MUST be called exactly once per day at market open (09:30 AM EST).
        Clears the persistent intraday memory.
        """
        self.hidden_state = None

    def forward(self, x: torch.Tensor, timespans: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Standard forward pass used primarily during training (train_ltc.py).
        
        Args:
            x: Shape (Batch, Seq_Len, Input_Size)
            timespans: Shape (Batch, Seq_Len) containing continuous time steps (e.g., seconds since open)
            
        Returns:
            out: Shape (Batch, Seq_Len, 1) - The urgency score trajectory
            hidden: The final hidden state
        """
        out, hidden = self.ltc(x, timespans=timespans)
        
        # Sigmoid to bound the urgency score strictly between [0.0, 1.0]
        urgency_score = torch.sigmoid(out)
        return urgency_score, hidden

    @torch.no_grad()
    def step(self, obs: np.ndarray, elapsed_seconds: float, device: str = 'cuda') -> Tuple[float, bool]:
        """
        LIVE INFERENCE METHOD. 
        Called by services/regime_encoder_svc.py upon receiving a new market tick.
        
        Args:
            obs: 15-dim numpy array of current intraday features (e.g., volume pace, spread z-score)
            elapsed_seconds: Exact time elapsed since the previous observation
            
        Returns:
            drift_score: Float between [0, 1]
            urgency_flag: Boolean, True if score exceeds urgency_threshold
        """
        self.eval()
        
        # Reshape to (Batch=1, Seq_Len=1, Features=15)
        x = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0).to(device)
        
        # Reshape to (Batch=1, Seq_Len=1)
        ts = torch.FloatTensor([[elapsed_seconds]]).to(device)
        
        # Advance the continuous-time network by exactly `elapsed_seconds`
        out, self.hidden_state = self.ltc(x, timespans=ts, hx=self.hidden_state)
        
        drift_score = torch.sigmoid(out[0, 0, 0]).item()
        urgency_flag = bool(drift_score > self.urgency_threshold)
        
        return drift_score, urgency_flag