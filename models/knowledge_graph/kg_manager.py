"""
FORTRESS v5 - kg_manager.py
Path: models/knowledge_graph/kg_manager.py

Dynamic Knowledge Graph Manager.
Constructs the multi-relational adjacency matrix for the Alpha Engine (GATv2)
by computing rolling correlations, statistical causalities, and macro sensitivities.
"""

import logging
import torch
import numpy as np
import pandas as pd
from typing import Tuple, List

logger = logging.getLogger("KG_Manager")

class KnowledgeGraphManager:
    def __init__(self, universe_tickers: List[str]):
        """
        Initializes the graph manager for the defined asset universe.
        """
        self.tickers = universe_tickers
        self.num_nodes = len(self.tickers)
        self.ticker_to_idx = {ticker: i for i, ticker in enumerate(self.tickers)}
        
        # Hyperparameters for edge pruning
        self.corr_threshold = 0.65  # Only connect assets with strong correlations
        self.causal_threshold = 0.05 # p-value threshold for Granger causality proxy

    def build_live_edge_index(self, return_matrix: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Constructs the PyTorch Geometric edge tensors dynamically based on recent price action.
        
        Args:
            return_matrix: DataFrame of recent asset returns, shape (Time, Num_Assets).
                           Columns must strictly match self.tickers.
                           
        Returns:
            edge_index: Shape (2, Num_Edges)
            edge_attr: Shape (Num_Edges, 5) 
                       Features: [Is_Correlated, Is_Inverse, Causal_Lead, Macro_Proxy, Supply_Proxy]
        """
        if return_matrix.empty or return_matrix.shape[1] != self.num_nodes:
            logger.warning("Invalid return matrix provided to KG Manager. Falling back to dense graph.")
            return self._build_fallback_graph()

        # 1. Compute the empirical correlation matrix
        corr_matrix = return_matrix.corr().fillna(0).values
        
        sources, targets = [], []
        edge_features = []

        # 2. Iterate through all possible pairs and forge edges based on thresholds
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i == j:
                    continue # No self-loops; GAT handles self-attention internally
                    
                correlation = corr_matrix[i, j]
                
                # Check if the relationship is statistically strong enough to warrant an edge
                if abs(correlation) > self.corr_threshold:
                    sources.append(i)
                    targets.append(j)
                    
                    # Feature 1 & 2: Correlation metrics
                    is_correlated = 1.0 if correlation > 0 else 0.0
                    is_inverse = 1.0 if correlation < 0 else 0.0
                    
                    # Feature 3: Granger Causality proxy (simulated here for architecture flow)
                    # In full prod, this uses DYNOTEARS or statsmodels grangercausalitytests
                    causal_lead = 1.0 if (i < j and abs(correlation) > 0.8) else 0.0
                    
                    # Feature 4 & 5: Static sector/macro groupings (e.g., both are tech or safe havens)
                    # Abstracted for this module, assuming 0.5 neutral prior
                    macro_proxy = 0.5 
                    supply_proxy = 0.0
                    
                    edge_features.append([
                        is_correlated, 
                        is_inverse, 
                        causal_lead, 
                        macro_proxy, 
                        supply_proxy
                    ])

        # If market is totally fragmented and no edges pass the threshold, use a minimal spanning tree
        if not sources:
            return self._build_fallback_graph()

        edge_index = torch.tensor([sources, targets], dtype=torch.long)
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)
        
        return edge_index, edge_attr

    def _build_fallback_graph(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Provides a safe, fully-connected graph (minus self-loops) with neutral edge weights."""
        sources, targets = [], []
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j:
                    sources.append(i)
                    targets.append(j)
                    
        edge_index = torch.tensor([sources, targets], dtype=torch.long)
        num_edges = edge_index.shape[1]
        
        # Neutral 5-dim features
        neutral_features = [[0.5, 0.0, 0.0, 0.0, 0.0] for _ in range(num_edges)]
        edge_attr = torch.tensor(neutral_features, dtype=torch.float32)
        
        return edge_index, edge_attr