"""
FORTRESS v5 - train_world_model.py
Path: training/train_world_model.py

Neural SDE Training Loop.
Teaches the model the physics of market crashes by optimizing the 
drift and diffusion networks against historical volatility regimes.
"""

import os
import yaml
import torch
import logging
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
import torchsde

from models.world_model.neural_sde import LatentSDEWorldModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("WorldModelTrainer")

class WorldModelTrainer:
    def __init__(self, config_path: str = 'config/hyperparams.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f).get('world_model', {})
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Initializing Neural SDE Trainer on {self.device}...")
        
        self.model = LatentSDEWorldModel(self.config).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        self.scaler = GradScaler()
        
        self.epochs = self.config.get('epochs', 150)
        self.batch_size = self.config.get('batch_size', 64)
        
        # Dimensions
        self.state_dim = self.config.get('sde_state_dim', 25)
        self.regime_dim = self.config.get('latent_dim', 16)

    def _load_historical_paths(self) -> DataLoader:
        """
        Loads actual historical market trajectories. 
        Input Shape: (Num_Paths, Seq_Len, State_Dim)
        """
        logger.info("Loading continuous-time historical trajectories...")
        
        # Scaffold: 5,000 overlapping 21-day historical sequences
        num_paths = 5000
        seq_len = 21 
        
        # Historical prices/returns
        Y = torch.randn(num_paths, seq_len, self.state_dim, dtype=torch.float32)
        # Corresponding Mamba-KAN regime states
        Z = torch.randn(num_paths, self.regime_dim, dtype=torch.float32)
        
        dataset = TensorDataset(Y, Z)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

    def train(self):
        dataloader = self._load_historical_paths()
        logger.info("Initiating Itô Calculus Optimization...")
        
        # We simulate dt = 1.0 (daily steps) for the training grid
        t_grid = torch.linspace(0, 20, 21, device=self.device)
        dt = 1.0
        
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss = 0.0
            
            for Y_batch, Z_batch in dataloader:
                Y_batch, Z_batch = Y_batch.to(self.device), Z_batch.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                
                # SDEs are memory intensive; use mixed precision
                with autocast():
                    # Set the latent regime context for the SDE networks
                    self.model.current_z_t = Z_batch
                    
                    # 1. Calculate the Drift (f) and Diffusion (g) for the true historical states
                    # Flatten the sequence for batch processing through the networks
                    Y_flat = Y_batch[:, :-1, :].reshape(-1, self.state_dim)
                    Z_expanded = Z_batch.unsqueeze(1).expand(-1, 20, -1).reshape(-1, self.regime_dim)
                    
                    # Dummy time tensor (not strictly used if networks are time-invariant)
                    t_dummy = torch.zeros(Y_flat.shape[0], device=self.device)
                    
                    drift = self.model.f_net(t_dummy, Y_flat, Z_expanded)
                    diffusion = self.model.g_net(t_dummy, Y_flat, Z_expanded)
                    
                    # 2. Maximum Likelihood / Euler-Maruyama Loss
                    # How well does the drift predict the actual next step?
                    actual_delta = Y_batch[:, 1:, :].reshape(-1, self.state_dim) - Y_flat
                    
                    # Approximate the negative log-likelihood of the transition
                    # Loss = (Delta - Drift * dt)^2 / (2 * Diffusion^2 * dt) + log(Diffusion)
                    # We use a simplified pseudo-Huber loss for numerical stability
                    
                    diffusion_variance = torch.sum(diffusion ** 2, dim=-1) + 1e-4
                    expected_delta = drift * dt
                    
                    residual = actual_delta - expected_delta
                    
                    loss = torch.mean((residual ** 2) / (2 * diffusion_variance * dt) + torch.log(diffusion_variance))
                
                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0) # SDEs need looser clipping
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                total_loss += loss.item()
                
            avg_loss = total_loss / len(dataloader)
            
            if epoch % 10 == 0 or epoch == 1:
                logger.info(f"Epoch [{epoch:03d}/{self.epochs}] | SDE Negative Log-Likelihood: {avg_loss:.4f}")

        self._save_weights()

    def _save_weights(self):
        os.makedirs('models/weights', exist_ok=True)
        save_path = 'models/weights/neural_sde_latest.pt'
        torch.save(self.model.state_dict(), save_path)
        logger.info(f"World Model physics saved to {save_path}")

if __name__ == "__main__":
    trainer = WorldModelTrainer()
    trainer.train()