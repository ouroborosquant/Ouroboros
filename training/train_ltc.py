"""
FORTRESS v5 - train_ltc.py
Path: training/train_ltc.py

Liquid Neural Network (LTC) Training Loop.
Teaches the continuous-time network to detect intraday flash crashes 
from irregularly spaced limit order book ticks.
"""

import os
import yaml
import torch
import logging
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler

from models.regime.liquid_neural_net import IntraRegimeMonitor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LTC_Trainer")

class LTCTrainer:
    def __init__(self, config_path: str = 'config/hyperparams.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f).get('ltc', {})
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Initializing Liquid Time-Constant Optimizer on {self.device}...")
        
        self.model = IntraRegimeMonitor(self.config).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.get('learning_rate', 1e-3))
        self.scaler = GradScaler()
        
        self.epochs = 50
        self.batch_size = 64

    def _load_irregular_time_series(self) -> DataLoader:
        """
        Simulates high-frequency tick data with irregular timestamps.
        Inputs: (Features, Timespans), Targets: Binary Flash Crash Event (0 or 1)
        """
        num_samples = 2000
        seq_len = 100 # 100 ticks per training window
        input_size = self.config.get('input_size', 15)
        
        # Random market features (OBI, spread, etc.)
        X = torch.randn(num_samples, seq_len, input_size, dtype=torch.float32)
        
        # Irregular timespans (e.g., 0.1s, 2.5s, 0.05s between trades)
        timespans = torch.abs(torch.randn(num_samples, seq_len, dtype=torch.float32)) * 2.0
        
        # Target: Did this sequence result in a micro-crash? (1 = Yes, 0 = No)
        y = torch.randint(0, 2, (num_samples, 1), dtype=torch.float32)
        
        return DataLoader(TensorDataset(X, timespans, y), batch_size=self.batch_size, shuffle=True)

    def train(self):
        dataloader = self._load_irregular_time_series()
        logger.info("Initiating Continuous-Time ODE Optimization...")
        
        # Binary Cross Entropy for the probability of a crash
        criterion = torch.nn.BCELoss()
        
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss = 0.0
            
            for X_batch, ts_batch, y_batch in dataloader:
                X_batch, ts_batch, y_batch = X_batch.to(self.device), ts_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                
                with autocast():
                    # The LTC network processes the irregular timespans directly
                    urgency_scores, _ = self.model(X_batch, timespans=ts_batch)
                    
                    # We evaluate the network's prediction at the final timestep
                    final_score = urgency_scores[:, -1, :] 
                    loss = criterion(final_score, y_batch)
                
                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                total_loss += loss.item()
                
            avg_loss = total_loss / len(dataloader)
            if epoch % 5 == 0 or epoch == 1:
                logger.info(f"Epoch [{epoch:02d}/{self.epochs}] | LTC Binary Cross-Entropy Loss: {avg_loss:.5f}")

        self._save_weights()

    def _save_weights(self):
        os.makedirs('models/weights', exist_ok=True)
        save_path = 'models/weights/ltc_latest.pt'
        torch.save(self.model.state_dict(), save_path)
        logger.info(f"LTC weights successfully saved to {save_path}")

if __name__ == "__main__":
    trainer = LTCTrainer()
    trainer.train()