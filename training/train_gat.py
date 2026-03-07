"""
FORTRESS v5 - train_gat.py
Path: training/train_gat.py

Multi-Relational GATv2 Training Loop.
Optimizes the Alpha Engine to predict cross-asset shock propagation.
"""

import os
import yaml
import torch
import logging
import numpy as np
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch.cuda.amp import autocast, GradScaler

from models.alpha.gat_alpha import MultiRelationalGAT, AssetGraph

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GAT_Trainer")

class GATTrainer:
    def __init__(self, config_path: str = 'config/hyperparams.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f).get('gat_alpha', {})
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Initializing GATv2 Optimizer on {self.device}...")
        
        self.model = MultiRelationalGAT(
            node_feat_dim=self.config.get('node_feat_dim', 78),
            edge_feat_dim=self.config.get('edge_feat_dim', 5),
            hidden_dim=self.config.get('hidden_dim', 128),
            n_heads=self.config.get('n_heads', 8),
            n_layers=self.config.get('n_layers', 3)
        ).to(self.device)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-4, weight_decay=1e-3)
        self.scaler = GradScaler()
        self.epochs = self.config.get('epochs', 100)

    def _build_synthetic_graph_dataset(self) -> DataLoader:
        """
        Scaffolds historical multi-relational graphs.
        In production, this pulls dynamically from Neo4j/DYNOTEARS outputs.
        """
        dataset = []
        num_graphs = 2000 # Trading days
        num_nodes = 25    # ETF Universe
        
        for _ in range(num_graphs):
            # 78-dim node features
            x = torch.randn(num_nodes, self.config.get('node_feat_dim', 78), dtype=torch.float32)
            edge_index, edge_attr = AssetGraph.build_dummy_edge_index(num_nodes)
            
            # Target: 5-day forward residual alpha
            y = torch.randn(num_nodes, dtype=torch.float32)
            dataset.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
            
        return DataLoader(dataset, batch_size=32, shuffle=True)

    def train(self):
        dataloader = self._build_synthetic_graph_dataset()
        logger.info("Initiating Message Passing Optimization...")
        
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss = 0.0
            
            for batch in dataloader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                
                with autocast():
                    predicted_alphas = self.model(batch.x, batch.edge_index, batch.edge_attr)
                    
                    # Mean Squared Error against true forward alpha
                    mse_loss = F.mse_loss(predicted_alphas, batch.y)
                    # L1 penalty enforces sparsity (only strong signals survive)
                    l1_penalty = 0.01 * torch.norm(predicted_alphas, p=1)
                    loss = mse_loss + l1_penalty
                
                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                total_loss += loss.item() * batch.num_graphs
                
            avg_loss = total_loss / len(dataloader.dataset)
            if epoch % 10 == 0 or epoch == 1:
                logger.info(f"Epoch [{epoch:03d}/{self.epochs}] | GAT Alpha Loss: {avg_loss:.5f}")

        self._save_weights()

    def _save_weights(self):
        os.makedirs('models/weights', exist_ok=True)
        save_path = 'models/weights/gat_alpha_latest.pt'
        torch.save(self.model.state_dict(), save_path)
        logger.info(f"Graph weights successfully saved to {save_path}")

if __name__ == "__main__":
    trainer = GATTrainer()
    trainer.train()