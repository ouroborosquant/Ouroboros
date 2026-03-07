"""
FORTRESS v5 - train_edt.py
Path: training/train_edt.py

Elastic Decision Transformer Training Loop.
Trains the portfolio allocator using sequence modeling and diffusion denoising.
"""

import os
import yaml
import torch
import logging
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler

from models.portfolio.edt_agent import ElasticDecisionTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("EDTTrainer")

class EDTTrainer:
    def __init__(self, config_path: str = 'config/hyperparams.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f).get('edt', {})
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Initializing EDT Trainer on {self.device}...")
        
        # Instantiate the architecture
        self.model = ElasticDecisionTransformer(self.config).to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config.get('learning_rate', 3e-4),
            weight_decay=1e-4
        )
        self.scaler = GradScaler()
        
        self.epochs = self.config.get('epochs', 200)
        self.batch_size = self.config.get('batch_size', 128)
        self.action_dim = self.config.get('action_dim', 25) # 25-asset universe

    def _load_offline_trajectories(self) -> DataLoader:
        """
        Loads historical sequences of (Market_State, Target_Return, Optimal_Weights).
        In reality, these target weights are calculated via retrospective dynamic programming
        (hindsight optimization) to show the EDT what the "perfect" portfolio was.
        """
        logger.info("Loading offline hindsight trajectories...")
        state_dim = self.config.get('state_dim', 192)
        
        # Scaffold: Synthetic dataset representing 50,000 historical sequences
        num_samples = 50000
        states = torch.randn(num_samples, state_dim, dtype=torch.float32)
        returns = torch.randn(num_samples, 1, dtype=torch.float32) # Target Returns
        
        # Target weights (Softmaxed so they sum to 1.0)
        raw_weights = torch.randn(num_samples, self.action_dim, dtype=torch.float32)
        target_weights = torch.softmax(raw_weights, dim=-1)
        
        dataset = TensorDataset(states, returns, target_weights)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

    def train(self):
        dataloader = self._load_offline_trajectories()
        logger.info("Initiating EDT Diffusion Optimization...")
        
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss = 0.0
            
            for batch_idx, (states, target_returns, optimal_weights) in enumerate(dataloader):
                states, target_returns, optimal_weights = states.to(self.device), target_returns.to(self.device), optimal_weights.to(self.device)
                
                self.optimizer.zero_grad(set_to_none=True)
                
                with autocast():
                    # 1. Embed inputs (Return and State)
                    ret_emb = self.model.embed_return(target_returns.unsqueeze(1))
                    state_emb = self.model.embed_state(states.unsqueeze(1))
                    seq_input = torch.cat([ret_emb, state_emb], dim=1)
                    
                    # 2. Forward through Transformer to get context
                    transformer_out = self.model.transformer(seq_input)
                    context = transformer_out[:, -1, :] # Final hidden state
                    
                    # 3. Diffusion Denoising Objective
                    # Sample random timesteps for the diffusion process
                    t = torch.randint(0, self.model.action_head.n_steps, (states.size(0), 1), device=self.device).float() / self.model.action_head.n_steps
                    
                    # Add Gaussian noise to the optimal historical weights
                    noise = torch.randn_like(optimal_weights)
                    noisy_weights = optimal_weights + noise * t
                    
                    # Predict the noise using the Action Head
                    nn_input = torch.cat([context, noisy_weights, t], dim=-1)
                    predicted_noise = self.model.action_head.noise_predictor(nn_input)
                    
                    # Loss is simply the Mean Squared Error between actual noise and predicted noise
                    loss = torch.nn.functional.mse_loss(predicted_noise, noise)
                
                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                total_loss += loss.item()
                
            avg_loss = total_loss / len(dataloader)
            
            if epoch % 10 == 0 or epoch == 1:
                logger.info(f"Epoch [{epoch:03d}/{self.epochs}] | Diffusion Denoising Loss: {avg_loss:.6f}")

        self._save_weights()

    def _save_weights(self):
        os.makedirs('models/weights', exist_ok=True)
        save_path = 'models/weights/edt_latest.pt'
        torch.save(self.model.state_dict(), save_path)
        logger.info(f"EDT weights saved to {save_path}")

if __name__ == "__main__":
    trainer = EDTTrainer()
    trainer.train()