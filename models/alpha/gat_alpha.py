"""
FORTRESS v5 - gat_alpha.py
Path: models/alpha/gat_alpha.py

Multi-Relational Graph Attention Network (GATv2).
Processes the causal asset graph to produce per-asset expected alpha scores.
Architecture ONLY. No training loops.
"""
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# External dependencies (PyTorch Geometric)
try:
    import torch_geometric.nn as pyg_nn
    from torch_geometric.data import Data
except ImportError:
    raise ImportError("PyTorch Geometric is required. Install via: pip install torch-geometric")


class AssetGraph:
    """
    Utility class to manage the dynamic multi-relational asset graph structure.
    In the live system, these edges are continuously updated by the Knowledge Graph Manager
    and the TDA Topology microservice.
    """
    EDGE_TYPES = [
        'granger_causal',       # 0: Lead-lag relationships discovered by DYNOTEARS
        'dcc_correlation',      # 1: Dynamic Conditional Correlation (financial coupling)
        'macro_sensitivity',    # 2: Shared sensitivity to FRED macro shocks
        'institutional_flow',   # 3: Overlapping ETF ownership / block trade flow
        'supply_chain'          # 4: Physical dependencies (BEA Input-Output / Satellite)
    ]

    @staticmethod
    def build_dummy_edge_index(num_nodes: int = 25) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates a placeholder graph for scaffolding/testing.
        Returns:
            edge_index: Shape (2, Num_Edges)
            edge_attr: Shape (Num_Edges, 5) - The 5-dim edge features
        """
        # Fully connected graph for dummy initialization (excluding self-loops)
        sources, targets = [], []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    sources.append(i)
                    targets.append(j)
                    
        edge_index = torch.tensor([sources, targets], dtype=torch.long)
        
        # 5-dimensional edge features [edge_type_one_hot (3-dim proxy), correlation, causal_strength]
        num_edges = edge_index.shape[1]
        edge_attr = torch.rand((num_edges, 5), dtype=torch.float32)
        
        return edge_index, edge_attr


class MultiRelationalGAT(nn.Module):
    """
    Produces a 25-dimensional alpha vector from the dynamic asset graph.
    
    Node Features (78-dim total):
      - 47 obs: Raw TimescaleDB price/macro features
      - 16 z_t: The latent regime posterior from Mamba-KAN
      - 15 LLM: Vectorized alpha signals from the Agentic LLM / Satellite pipelines
      
    GATv2Conv is used over GATConv because it applies the non-linearity BEFORE 
    the attention scoring: e_ij = a^T * LeakyReLU(W * [h_i || h_j]). This allows 
    the network to learn attention weights that depend on the specific interaction 
    between nodes, rather than just their absolute values.
    """
    
    def __init__(self, node_feat_dim: int = 78, edge_feat_dim: int = 5, 
                 hidden_dim: int = 128, n_heads: int = 8, n_layers: int = 3, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.n_layers = n_layers
        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        # Build GATv2 layers
        for i in range(n_layers):
            in_channels = node_feat_dim if i == 0 else hidden_dim * n_heads
            
            # The final layer concatenates heads=1 to output a stable dimension
            is_final_layer = (i == n_layers - 1)
            out_channels = hidden_dim
            heads = 1 if is_final_layer else n_heads
            concat = not is_final_layer
            
            self.convs.append(pyg_nn.GATv2Conv(
                in_channels=in_channels, 
                out_channels=out_channels,
                heads=heads, 
                edge_dim=edge_feat_dim, 
                concat=concat, 
                dropout=dropout
            ))
            
            # Layer normalization for training stability across graph sizes
            norm_dim = out_channels if is_final_layer else out_channels * heads
            self.layer_norms.append(nn.LayerNorm(norm_dim))
            
        # Final projection to a single scalar 'expected alpha' per asset
        self.alpha_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features, shape (Num_Nodes, 78) -> usually Num_Nodes=25
            edge_index: Graph connectivity, shape (2, Num_Edges)
            edge_attr: Edge features, shape (Num_Edges, 5)
            
        Returns:
            alpha_scores: Shape (Num_Nodes,) -> The predicted alpha for each asset
        """
        h = x
        
        for i in range(self.n_layers):
            # Message passing and attention
            h = self.convs[i](h, edge_index, edge_attr)
            
            # Non-linearity and normalization (skip activation on final layer output)
            if i < self.n_layers - 1:
                h = F.elu(h)
                
            h = self.layer_norms[i](h)
            
        # Project hidden state to a single alpha score per node
        alpha_scores = self.alpha_head(h).squeeze(-1)  # Shape: (25,)
        
        # Apply Tanh to bound expected alphas between [-1, 1] for optimization stability
        return torch.tanh(alpha_scores)

    @torch.no_grad()
    def infer_live_alpha(self, graph_data: Data, device: str = 'cuda') -> np.ndarray:
        """
        LIVE INFERENCE METHOD.
        Called by services/alpha_engine_svc.py.
        """
        self.eval()
        x = graph_data.x.to(device)
        edge_index = graph_data.edge_index.to(device)
        edge_attr = graph_data.edge_attr.to(device)
        
        alphas = self.forward(x, edge_index, edge_attr)
        return alphas.cpu().numpy()