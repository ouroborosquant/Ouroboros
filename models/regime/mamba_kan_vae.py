"""
FORTRESS v5 - mamba_kan_vae.py
Path: models/regime/mamba_kan_vae.py

Hybrid Mamba-KAN Variational Autoencoder.
Architecture ONLY. No training loops.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict

# External dependencies (Require CUDA for Mamba)
try:
    from mamba_ssm import Mamba
except ImportError:
    raise ImportError("mamba_ssm is required. Install via: pip install mamba-ssm")

try:
    from kan import KAN
except ImportError:
    raise ImportError("pykan is required. Install via: pip install pykan")


class MambaKANEncoder(nn.Module):
    """
    Temporal processing (Mamba) -> interpretable latent transformation (KAN).
    """
    def __init__(self, obs_dim: int = 52, latent_dim: int = 16, d_model: int = 256, 
                 n_layers: int = 4, kan_width: list = [256, 128, 64, 16], 
                 kan_grid: int = 5, kan_k: int = 3):
        super().__init__()
        
        # 1. Input Projection
        self.input_proj = nn.Linear(obs_dim, d_model)
        
        # 2. Temporal Sequence Processing (Mamba SSM)
        self.mamba_blocks = nn.ModuleList([
            Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(n_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])
        
        # 3. Symbolic Transformation (Kolmogorov-Arnold Network)
        # Note: KAN takes the final temporal state and maps it to the latent space
        self.kan = KAN(width=kan_width, grid=kan_grid, k=kan_k, seed=42)
        
        # 4. VAE Reparameterization Heads
        # Ensure kan_width[-1] is treated as a plain integer
        # Flatten the list if it's nested (e.g., [[128]] -> [128] -> 128)
        last_val = kan_width[-1]
        while isinstance(last_val, list):
            last_val = last_val[0]

        in_features = int(last_val)
        
        self.mu_head = nn.Linear(in_features, latent_dim)
        self.log_sigma_head = nn.Linear(in_features, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input x shape: (Batch, Seq_Len, Obs_Dim)
        Returns: mu, log_sigma each shape (Batch, Latent_Dim)
        """
        # Temporal processing over the sequence
        h = self.input_proj(x)
        for block, norm in zip(self.mamba_blocks, self.layer_norms):
            h = h + block(norm(h))  # Residual connection with pre-norm
            
        # Extract the final timestep's hidden state
        h_last = h[:, -1, :]  # Shape: (Batch, d_model)
        
        # Pass through KAN for interpretable non-linear transformation
        z_pre = self.kan(h_last)
        
        return self.mu_head(z_pre), self.log_sigma_head(z_pre)


class StudentTMixtureDecoder(nn.Module):
    """
    Fat-tailed decoder designed for financial returns.
    Standard Gaussian decoders fail during 5-sigma market crashes.
    """
    def __init__(self, latent_dim: int = 16, obs_dim: int = 52, n_components: int = 4):
        super().__init__()
        self.n_components = n_components
        self.obs_dim = obs_dim
        
        self.pi_net = nn.Linear(latent_dim, n_components)
        self.mu_net = nn.Linear(latent_dim, n_components * obs_dim)
        self.sigma_net = nn.Linear(latent_dim, n_components * obs_dim)
        
        # Learnable Degrees of Freedom (nu) to dynamically adapt to kurtosis
        self.log_nu = nn.Parameter(torch.ones(n_components) * 2.0)

    def log_likelihood(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Computes the log-likelihood of the observation x given latent state z
        under a Student-t mixture model.
        """
        pi_logits = self.pi_net(z)
        pi = F.softmax(pi_logits, dim=-1)  # (Batch, n_components)
        
        mu = self.mu_net(z).view(-1, self.n_components, self.obs_dim)
        sigma = F.softplus(self.sigma_net(z)).view(-1, self.n_components, self.obs_dim) + 1e-4
        
        # Shape nu to (1, 4, 1) so it broadcasts safely across (Batch, 4, 52)
        nu = (F.softplus(self.log_nu) + 2.0).view(1, self.n_components, 1)
        
        # Expand x to match mixture components
        x_expanded = x.unsqueeze(1).expand(-1, self.n_components, -1)
        
        # Student-t log probability calculation
        term1 = torch.lgamma((nu + 1) / 2) - torch.lgamma(nu / 2)
        term2 = -0.5 * torch.log(nu * np.pi) - torch.log(sigma)
        term3 = -((nu + 1) / 2) * torch.log1p(((x_expanded - mu) / sigma) ** 2 / nu)
        
        log_prob_components = term1 + term2 + term3
        log_prob_sum = log_prob_components.sum(dim=-1)  # Sum log probs across obs_dim
        
        # Log-Sum-Exp for numerical stability of mixture weights
        weighted_log_prob = torch.log(pi + 1e-8) + log_prob_sum
        return torch.logsumexp(weighted_log_prob, dim=-1)


class MambaKANVAE(nn.Module):
    """
    Main model wrapper. Orchestrates the Encoder and Decoder.
    """
    def __init__(self, config: Dict):
        super().__init__()
        self.encoder = MambaKANEncoder(
            obs_dim=config.get('obs_dim', 52),
            latent_dim=config.get('latent_dim', 16),
            d_model=config.get('d_model', 256),
            n_layers=config.get('n_mamba_layers', 4),
            kan_width=config.get('kan_width', [256, 128, 64, 16])
        )
        self.decoder = StudentTMixtureDecoder(
            latent_dim=config.get('latent_dim', 16),
            obs_dim=config.get('obs_dim', 52),
            n_components=config.get('n_mixture_components', 4)
        )

    def reparameterize(self, mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * log_sigma)

    def compute_loss(self, x: torch.Tensor, beta: float = 4.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the beta-VAE ELBO loss.
        x shape: (Batch, Seq_Len, Obs_Dim)
        """
        mu, log_sigma = self.encoder(x)
        z = self.reparameterize(mu, log_sigma)
        
        # Reconstruction loss against the final timestep
        recon_loss = -self.decoder.log_likelihood(x[:, -1, :], z).mean()
        
        # KL Divergence vs Standard Normal N(0,I)
        kl_loss = -0.5 * torch.sum(1 + log_sigma - mu**2 - log_sigma.exp(), dim=-1).mean()
        
        total_loss = recon_loss + beta * kl_loss
        return total_loss, recon_loss, kl_loss

    @torch.no_grad()
    def get_posterior(self, x_history: np.ndarray, device: str = 'cuda') -> Tuple[np.ndarray, np.ndarray]:
        """
        LIVE INFERENCE METHOD.
        Called by services/regime_encoder_svc.py.
        Thread-safe and low-latency.
        """
        self.eval()
        x = torch.FloatTensor(x_history).unsqueeze(0).to(device)
        mu, log_sigma = self.encoder(x)
        return mu[0].cpu().numpy(), torch.exp(0.5 * log_sigma[0]).cpu().numpy()

    def extract_symbolic_rules(self) -> Dict[int, str]:
        """
        Extracts human-readable mathematical formulas defining the regime space.
        Called post-training by scripts/export_fpga_rules.py.
        """
        # Utilizes pykan's symbolic functionality
        self.encoder.kan.fix_symbolic()
        formulas = self.encoder.kan.symbolic_formulas()
        return {i: str(f) for i, f in enumerate(formulas[0])}