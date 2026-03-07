"""
FORTRESS v5 - train_hedging.py
Path: training/train_hedging.py

Deep Hedging Training Loop.
Trains the derivative overlay network to minimize CVaR across SDE-simulated crashes.
"""

import os
import yaml
import torch
import logging
import numpy as np
from torch.optim import AdamW

from models.hedging.deep_hedging import DeepHedgingNetwork
from models.world_model.neural_sde import LatentSDEWorldModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("HedgingTrainer")

class HedgingTrainer:
    def __init__(self, config_path: str = 'config/hyperparams.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Initializing CVaR Minimization Optimizer on {self.device}...")
        
        # Load the SDE Simulator (Weights frozen during hedging training)
        self.world_model = LatentSDEWorldModel(self.config.get('world_model', {})).to(self.device)
        self.world_model.eval()
        
        # Load the untrained Hedging Network
        self.hedger = DeepHedgingNetwork(self.config.get('hedging', {})).to(self.device)
        self.optimizer = AdamW(self.hedger.parameters(), lr=1e-3, weight_decay=1e-4)
        
        self.epochs = 50
        self.cvar_alpha = self.config.get('hedging', {}).get('cvar_alpha', 0.05)

    def train(self):
        logger.info("Initiating Deep Hedging optimization via World Model simulation...")
        batch_size = 128
        
        for epoch in range(1, self.epochs + 1):
            self.hedger.train()
            
            # 1. Scaffold random market states (Regime z_t, Portfolio state, Alerts)
            z_t = torch.randn(batch_size, 16, device=self.device)
            port_state = torch.randn(batch_size, 3, device=self.device)
            alerts = torch.rand(batch_size, 2, device=self.device)
            
            state_vector = torch.cat([z_t, port_state, alerts], dim=-1)
            
            self.optimizer.zero_grad()
            
            # 2. Predict the hedge weights (e.g., how much VIXY or GLD to hold)
            hedge_weights = self.hedger(state_vector)
            
            # 3. Simulate future market trajectories using the SDE
            # Simulate 10 days forward, generating 100 paths per batch item
            sde_initial_state = torch.randn(batch_size, 25, device=self.device)
            
            with torch.no_grad():
                # Shape: (Steps, Batch, State_Dim)
                trajectories = self.world_model.generate_synthetic_paths(
                    initial_state=sde_initial_state, 
                    z_t=z_t, 
                    n_steps=10, 
                    n_paths=batch_size
                )
                
            # 4. Calculate final simulated returns (End vs Start)
            final_returns = (trajectories[-1] - trajectories[0]) / trajectories[0].abs()
            
            # Scaffold logic: Split the 25 assets into 'base portfolio' vs 'hedge proxies'
            # In reality, this requires strict index mapping
            simulated_base_returns = final_returns[:, :20].mean(dim=-1) # Mean return of equities
            simulated_proxy_returns = final_returns[:, 20:25] # Returns of VIXY, GLD, TLT, UUP, SH
            
            # 5. Compute CVaR Loss and Backpropagate
            loss = self.hedger.compute_cvar_loss(
                predicted_hedges=hedge_weights,
                simulated_portfolio_returns=simulated_base_returns,
                simulated_proxy_returns=simulated_proxy_returns,
                alpha=self.cvar_alpha
            )
            
            loss.backward()
            self.optimizer.step()
            
            if epoch % 5 == 0 or epoch == 1:
                logger.info(f"Epoch [{epoch:02d}/{self.epochs}] | Synthetic CVaR Loss: {loss.item():.5f}")

        self._save_weights()

    def _save_weights(self):
        os.makedirs('models/weights', exist_ok=True)
        save_path = 'models/weights/deep_hedging_latest.pt'
        torch.save(self.hedger.state_dict(), save_path)
        logger.info(f"Hedge Overlay weights successfully saved to {save_path}")

if __name__ == "__main__":
    trainer = HedgingTrainer()
    trainer.train()