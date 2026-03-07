"""
FORTRESS v5 - edt_agent.py
Path: models/portfolio/edt_agent.py

Elastic Decision Transformer (EDT) with Diffusion Action Head.
Reframes portfolio optimization as conditional sequence modeling.
Architecture ONLY. No training loops.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict

class MultiScaleContextAttention(nn.Module):
    """
    Evaluates three different temporal context windows (21d, 63d, 252d) 
    and learns to dynamically weight which timeframe is most relevant 
    for the current market regime.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.scale = d_model ** -0.5

    def forward(self, short_ctx: torch.Tensor, med_ctx: torch.Tensor, long_ctx: torch.Tensor) -> torch.Tensor:
        # Stack contexts: (Batch, 3, d_model)
        contexts = torch.stack([short_ctx, med_ctx, long_ctx], dim=1)
        
        Q = self.query_proj(contexts)
        K = self.key_proj(contexts)
        V = self.value_proj(contexts)
        
        # Self-attention across the 3 time horizons
        attention_scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale
        attention_weights = torch.softmax(attention_scores, dim=-1)
        
        # Weighted fusion of the multi-scale contexts
        fused_context = torch.bmm(attention_weights, V)
        return fused_context.mean(dim=1)  # (Batch, d_model)


class DiffusionActionHead(nn.Module):
    """
    Instead of outputting a single deterministic weight vector, this head uses 
    denoising diffusion to generate a full probability distribution over possible 
    portfolios. This quantifies epistemic uncertainty.
    """
    def __init__(self, d_model: int, action_dim: int = 25, n_steps: int = 20):
        super().__init__()
        self.action_dim = action_dim
        self.n_steps = n_steps
        
        # Denoising network: predicts the noise added to the action
        self.noise_predictor = nn.Sequential(
            nn.Linear(d_model + action_dim + 1, 256),  # +1 for timestep embedding
            nn.Mish(),
            nn.Linear(256, 256),
            nn.Mish(),
            nn.Linear(256, action_dim)
        )

    def sample(self, context_embedding: torch.Tensor, n_samples: int = 20) -> torch.Tensor:
        """
        Runs the reverse diffusion process to sample multiple valid weight vectors.
        """
        device = context_embedding.device
        batch_size = context_embedding.shape[0]
        
        # Start with pure Gaussian noise (Batch, n_samples, Action_Dim)
        x_t = torch.randn((batch_size, n_samples, self.action_dim), device=device)
        
        # Iteratively denoise
        for step in reversed(range(self.n_steps)):
            t_tensor = torch.full((batch_size, n_samples, 1), step / self.n_steps, device=device)
            
            # Expand context to match n_samples
            ctx_expanded = context_embedding.unsqueeze(1).expand(-1, n_samples, -1)
            
            # Predict noise
            nn_input = torch.cat([ctx_expanded, x_t, t_tensor], dim=-1)
            predicted_noise = self.noise_predictor(nn_input)
            
            # Remove a fraction of the noise (simplified DDIM step)
            alpha = step / self.n_steps
            x_t = x_t - (1 - alpha) * predicted_noise
            
        # Apply Softmax to ensure weights sum to 1.0 (long-only constraint)
        return torch.softmax(x_t, dim=-1)


class ElasticDecisionTransformer(nn.Module):
    """
    Main EDT architecture.
    Conditions on: (Target_Return, Market_State, GAT_Alpha)
    Outputs: Distribution of Target Portfolio Weights
    """
    def __init__(self, config: Dict):
        super().__init__()
        self.state_dim = config.get('state_dim', 192)  # Raw obs + latent z_t + GAT Alpha
        self.action_dim = config.get('action_dim', 25) # 25 ETFs in universe
        self.d_model = config.get('d_model', 512)
        
        # Modality Embeddings
        self.embed_return = nn.Linear(1, self.d_model)
        self.embed_state = nn.Linear(self.state_dim, self.d_model)
        self.embed_action = nn.Linear(self.action_dim, self.d_model)
        
        self.embed_timestep = nn.Embedding(1000, self.d_model)
        
        # Multi-Scale Attention Fusion
        self.multi_scale_fusion = MultiScaleContextAttention(self.d_model)
        
        # Core Transformer (GPT-2 style standard block)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=config.get('n_heads', 8), 
            dim_feedforward=self.d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.get('n_layers', 6))
        
        # Diffusion Action Head
        self.action_head = DiffusionActionHead(
            d_model=self.d_model, 
            action_dim=self.action_dim, 
            n_steps=config.get('diffusion_action_steps', 20)
        )

    def get_regime_return_target(self, z_t: np.ndarray, volatility_targets: Dict[str, float]) -> float:
        """
        Dynamically calculates the Return-To-Go (RTG) target.
        Instead of a fixed scalar, the EDT is prompted to achieve a return
        commensurate with the current regime's volatility constraint.
        """
        # Placeholder mapping: in production, this interprets the Mamba-KAN latent z_t 
        # to determine the current regime (e.g., 'high_vol_bull' -> 0.08 vol target)
        # return target = Risk_Free_Rate + (Sharpe_Target * Vol_Target)
        vol_target = volatility_targets.get('low_vol_bull', 0.10)
        expected_annual_return = 0.04 + (1.5 * vol_target) # Assumes 4% RFR, 1.5 Sharpe
        return expected_annual_return

    def select_optimal_context_length(self, returns_history: np.ndarray, target_return: float) -> int:
        """
        History-Length Elasticity.
        If recent performance severely lags the target return, the model
        truncates its context window to "forget" the bad trajectory.
        """
        windows = [21, 63, 252]
        regrets = []
        
        for w in windows:
            if len(returns_history) < w:
                regrets.append(float('inf'))
                continue
            
            recent_ret = np.mean(returns_history[-w:]) * 252 # Annualized
            regret = abs(target_return - recent_ret)
            regrets.append(regret)
            
        # Select the context window with the lowest regret relative to the target
        best_idx = np.argmin(regrets)
        return windows[best_idx]

    @torch.no_grad()
    def get_weights(self, state: np.ndarray, target_return: float, device: str = 'cuda') -> Tuple[np.ndarray, np.ndarray]:
        """
        LIVE INFERENCE METHOD.
        Called by services/portfolio_agent_svc.py.
        
        Returns:
            mean_weights: The central allocation vector (25,)
            std_weights: Epistemic uncertainty of the allocation (25,)
        """
        self.eval()
        
        # Prepare sequence: [Target_Return, State]
        ret_tensor = torch.FloatTensor([[target_return]]).unsqueeze(0).to(device)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device)
        
        ret_emb = self.embed_return(ret_tensor)
        state_emb = self.embed_state(state_tensor)
        
        # Sequence input for causal transformer
        seq_input = torch.cat([ret_emb, state_emb], dim=1)
        
        # Forward pass
        transformer_out = self.transformer(seq_input)
        
        # Use the final hidden state to condition the diffusion head
        context_embedding = transformer_out[:, -1, :]
        
        # Sample multiple valid portfolios from the diffusion head
        sampled_portfolios = self.action_head.sample(context_embedding, n_samples=20)
        
        # Calculate mean and uncertainty
        mean_weights = sampled_portfolios.mean(dim=1).squeeze(0).cpu().numpy()
        std_weights = sampled_portfolios.std(dim=1).squeeze(0).cpu().numpy()
        
        return mean_weights, std_weights