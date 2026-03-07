"""
FORTRESS v5 - train_regime.py
Path: training/train_regime.py

Mamba-KAN VAE Training Loop.
Learns to encode the 52-dimensional historical market state into a 
16-dimensional interpretable latent regime vector (z_t).
"""

import os
import yaml
import torch
import logging
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler

# Internal Model Import
from models.regime.mamba_kan_vae import MambaKANVAE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RegimeTrainer")

class RegimeTrainer:
    def __init__(self, config_path: str = 'config/hyperparams.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f).get('mamba_kan', {})
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Initializing Mamba-KAN Trainer on device: {self.device}")
        
        self.model = MambaKANVAE(self.config).to(self.device)
        
        # Institutional-grade optimization setup
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config.get('learning_rate', 1e-4),
            weight_decay=1e-5
        )
        
        # Cosine Annealing learning rate schedule for stable convergence
        self.epochs = self.config.get('epochs', 100)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        
        # Mixed Precision scaler for VRAM efficiency and speed
        self.scaler = GradScaler()
        
        self.batch_size = self.config.get('batch_size', 64)
        self.seq_len = self.config.get('seq_len', 252) # 1 trading year of context

    def _load_training_data(self) -> DataLoader:
        """
        In production, this queries TimescaleDB to build rolling sequence windows.
        For architectural demonstration, we generate synthetic tensors matching the schema.
        Input Shape: (Num_Samples, Seq_Len, Obs_Dim)
        """
        logger.info("Loading sequential market observations from TimescaleDB...")
        obs_dim = self.config.get('obs_dim', 52)
        
        # Scaffold: 10,000 synthetic trading days of historical data
        num_samples = 10000 
        X = torch.randn(num_samples, self.seq_len, obs_dim, dtype=torch.float32)
        
        dataset = TensorDataset(X)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

    def train(self):
        dataloader = self._load_training_data()
        
        logger.info("Initiating Mamba-KAN Optimization...")
        
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss, recon_loss, kl_loss = 0.0, 0.0, 0.0
            
            # KL Annealing (Beta Warm-up) to prevent posterior collapse early in training
            beta = min(4.0, 0.01 + (epoch / (self.epochs * 0.5)) * 4.0)
            
            for batch_idx, (X_batch,) in enumerate(dataloader):
                X_batch = X_batch.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                
                # Forward pass with mixed precision
                with autocast():
                    loss, recon, kl = self.model.compute_loss(X_batch, beta=beta)
                
                # Backward pass and gradient clipping
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                total_loss += loss.item()
                recon_loss += recon.item()
                kl_loss += kl.item()
                
            self.scheduler.step()
            
            # Logging metrics
            avg_loss = total_loss / len(dataloader)
            avg_recon = recon_loss / len(dataloader)
            avg_kl = kl_loss / len(dataloader)
            
            if epoch % 5 == 0 or epoch == 1:
                logger.info(f"Epoch [{epoch:03d}/{self.epochs}] | Beta: {beta:.2f} | "
                            f"Total Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | KL: {avg_kl:.4f}")

        self._save_weights()

    def _save_weights(self):
        """Saves the optimized tensors and extracts the KAN symbolic rules."""
        os.makedirs('models/weights', exist_ok=True)
        save_path = 'models/weights/mamba_kan_latest.pt'
        torch.save(self.model.state_dict(), save_path)
        logger.info(f"Model weights successfully saved to {save_path}")
        
        # Extract the interpretable mathematical formulas from the Kolmogorov-Arnold Network
        logger.info("Extracting symbolic regime formulas from KAN...")
        try:
            rules = self.model.extract_symbolic_rules()
            for latent_dim, formula in rules.items():
                logger.debug(f"Latent Dimension z_{latent_dim}: {formula}")
        except Exception as e:
            logger.warning(f"Could not extract symbolic rules (requires pykan fix_symbolic): {e}")

if __name__ == "__main__":
    trainer = RegimeTrainer()
    trainer.train()