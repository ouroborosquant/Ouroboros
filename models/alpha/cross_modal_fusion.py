"""
FORTRESS v5 - cross_modal_fusion.py
Path: models/alpha/cross_modal_fusion.py

Cross-Modal Alpha Fusion Network.
Combines three heterogeneous signal modalities into a single 78-dim node
feature tensor consumed by the GATv2 MultiRelationalGAT.

Previously this file contained only a docstring (empty stub). The GATv2
was receiving its node features from an unspecified source — in practice,
random noise — because no code constructed the 78-dim feature vector.

ARCHITECTURE:
  The 78-dimensional node feature vector for each asset is composed of:
    [0:47]  obs_features   — 47 raw market features (price/vol/options/macro)
                             Source: TimescaleDB / Redis 'obs:current'
    [47:63] regime_z_t     — 16-dim Mamba-KAN latent regime posterior
                             Source: Redis 'regime:z_mu'
    [63:78] llm_alpha      — 15-dim LLM/satellite alpha signal per asset
                             Source: LLM agent debate output + satellite pipeline

FUSION DESIGN:
  Rather than simple concatenation (which treats all modalities as equally
  reliable), this module uses a Gated Cross-Attention mechanism:

    1. Each modality is projected to a shared d_fusion dimensional space.
    2. A learned gating vector determines the contribution weight of each
       modality, conditioned on the regime z_t.
       Gate logic: α_obs, α_regime, α_llm = Softmax(W_gate * [z_t])
       This allows the network to downweight noisy LLM signals during
       macro-dominated regimes and upweight them during earnings seasons.
    3. The fused representation is projected back to 78 dims per node.

The final output matches the node_feat_dim=78 expected by gat_alpha.py.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("CrossModalFusion")

# ── Dimensionality contract ───────────────────────────────────────────────────
# These constants must match hyperparams.yaml and gat_alpha.py::node_feat_dim.
_OBS_DIM:     int = 47    # Raw market observation features per asset
_REGIME_DIM:  int = 16    # Mamba-KAN latent posterior dim (z_t)
_LLM_DIM:     int = 15    # LLM agent + satellite alpha signal dim per asset
_NODE_FEAT_DIM: int = _OBS_DIM + _REGIME_DIM + _LLM_DIM  # = 78, must match gat_alpha.py
_N_ASSETS:    int = 25    # Size of the ETF universe


class ModalityProjector(nn.Module):
    """
    Projects a single modality from its raw dimension into the shared fusion space.
    Includes LayerNorm for training stability (handles different scales across modalities).
    """

    def __init__(self, in_dim: int, d_fusion: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, d_fusion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_fusion, d_fusion),
            nn.LayerNorm(d_fusion),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class RegimeConditionedGate(nn.Module):
    """
    Learns the contribution weights for each modality, conditioned on the
    current market regime z_t.

    Intuition:
      - During high-vol / crash regimes (z_t encodes crisis), the LLM signals
        are noisy (news is lagging) → gate should downweight LLM, upweight obs.
      - During earnings season (z_t encodes idiosyncratic vol), LLM signals
        from the 3-agent debate carry more alpha → gate should upweight LLM.

    The gate learns this regime-conditional weighting end-to-end from labels.
    """

    def __init__(self, regime_dim: int, n_modalities: int = 3) -> None:
        super().__init__()
        self.n_modalities = n_modalities
        # Maps the regime posterior to a n_modalities-dim logit vector
        self.gate_net = nn.Sequential(
            nn.Linear(regime_dim, 64),
            nn.SiLU(),
            nn.Linear(64, n_modalities),
        )

    def forward(self, z_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_t: Regime posterior, shape (Batch, regime_dim) or (regime_dim,).

        Returns:
            gate_weights: Softmax-normalised weights, shape (Batch, n_modalities).
                          Sums to 1.0 across the modality dimension.
        """
        if z_t.dim() == 1:
            z_t = z_t.unsqueeze(0)
        return F.softmax(self.gate_net(z_t), dim=-1)


class CrossModalFusionNetwork(nn.Module):
    """
    Full fusion model that combines the three alpha modalities.

    Inputs (per asset, per timestep):
        obs_features:  (N_Assets, _OBS_DIM)   — 47 raw market features
        regime_z_t:    (_REGIME_DIM,)          — shared regime for ALL assets
        llm_alphas:    (N_Assets, _LLM_DIM)    — 15-dim LLM signal per asset

    Output:
        node_features: (N_Assets, _NODE_FEAT_DIM=78) — GATv2-ready tensor

    Design note:
        The output is NOT the gated-fusion embedding alone. We retain the
        raw concatenation [obs | z_t | llm] structure expected by the GATv2,
        but apply the gated fusion as a residual correction:

            output = concat(obs, broadcast(z_t), llm)
                     + gamma * gated_fusion_residual

        where gamma is a learnable scalar initialised near 0 (identity bypass
        at the start of training). This ensures the GATv2 can train stably
        from random initialisation before the fusion gate has converged.
    """

    def __init__(self, config: Dict) -> None:
        super().__init__()
        d_fusion      = config.get("d_fusion",  128)
        dropout       = config.get("dropout",   0.1)

        # ── Modality projectors ──────────────────────────────────────────────
        self.proj_obs    = ModalityProjector(_OBS_DIM,    d_fusion, dropout)
        self.proj_regime = ModalityProjector(_REGIME_DIM, d_fusion, dropout)
        self.proj_llm    = ModalityProjector(_LLM_DIM,    d_fusion, dropout)

        # ── Regime-conditioned gate ──────────────────────────────────────────
        self.gate = RegimeConditionedGate(regime_dim=_REGIME_DIM, n_modalities=3)

        # ── Cross-attention (obs queries regime+llm keys/values) ─────────────
        # Allows obs features to attend to regime and LLM context
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_fusion,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )

        # ── Output projection: fused → _NODE_FEAT_DIM residual ─────────────
        self.out_proj = nn.Sequential(
            nn.Linear(d_fusion, _NODE_FEAT_DIM),
            nn.LayerNorm(_NODE_FEAT_DIM),
        )

        # ── Learnable residual scale (initialised near 0 for stable training) ─
        self.gamma = nn.Parameter(torch.zeros(1))

        # ── Observation normalisation ─────────────────────────────────────────
        # Applied to raw obs features before projection to handle scale differences.
        self.obs_norm    = nn.LayerNorm(_OBS_DIM)
        self.llm_norm    = nn.LayerNorm(_LLM_DIM)
        self.regime_norm = nn.LayerNorm(_REGIME_DIM)

    def forward(
        self,
        obs_features: torch.Tensor,
        regime_z_t:  torch.Tensor,
        llm_alphas:  torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuses all three modalities into the 78-dim GATv2 node feature tensor.

        Args:
            obs_features: (N, _OBS_DIM=47)   — raw per-asset market features
            regime_z_t:   (_REGIME_DIM=16,)  — global regime posterior (broadcast)
            llm_alphas:   (N, _LLM_DIM=15)   — per-asset LLM alpha signals

        Returns:
            node_features: (N, _NODE_FEAT_DIM=78)
        """
        N = obs_features.shape[0]

        # ── Normalise inputs ─────────────────────────────────────────────────
        obs    = self.obs_norm(obs_features)          # (N, 47)
        z_t    = self.regime_norm(regime_z_t)          # (16,)
        llm    = self.llm_norm(llm_alphas)             # (N, 15)

        # ── Project each modality to d_fusion space ──────────────────────────
        h_obs    = self.proj_obs(obs)                  # (N, d_fusion)
        h_regime = self.proj_regime(z_t)               # (d_fusion,)
        h_llm    = self.proj_llm(llm)                  # (N, d_fusion)

        # Broadcast regime across all N assets
        h_regime_broadcast = h_regime.unsqueeze(0).expand(N, -1)  # (N, d_fusion)

        # ── Regime-conditioned gating ─────────────────────────────────────────
        # z_t determines the relative weight of each modality for this regime.
        gate_weights = self.gate(z_t.unsqueeze(0))     # (1, 3)
        w_obs, w_reg, w_llm = gate_weights.unbind(dim=-1)  # Each: scalar tensor

        # Weighted modality mixture
        h_mixed = (
            w_obs * h_obs
            + w_reg * h_regime_broadcast
            + w_llm * h_llm
        )  # (N, d_fusion)

        # ── Cross-attention: obs features attend to regime + LLM context ─────
        # Query: obs projection
        # Key/Value: stacked [regime, llm] context
        #   This allows each asset's observation to be refined by the global
        #   regime and its own LLM thesis simultaneously.
        kv_context = torch.stack([h_regime_broadcast, h_llm], dim=1)  # (N, 2, d_fusion)
        q          = h_mixed.unsqueeze(1)                               # (N, 1, d_fusion)

        attn_out, _ = self.cross_attn(query=q, key=kv_context, value=kv_context)
        # attn_out: (N, 1, d_fusion) → (N, d_fusion)
        h_attn = attn_out.squeeze(1)

        # ── Residual correction on the base concatenation ────────────────────
        # Base: raw concatenation [obs | broadcast(z_t) | llm]
        z_t_broadcast = z_t.unsqueeze(0).expand(N, -1)   # (N, 16)
        base_concat   = torch.cat([obs, z_t_broadcast, llm], dim=-1)  # (N, 78)

        # Residual: gated fusion projected back to 78-dim
        fusion_residual = self.out_proj(h_attn)           # (N, 78)

        # gamma starts near 0 → identity bypass at init, learned over training
        node_features = base_concat + torch.tanh(self.gamma) * fusion_residual  # (N, 78)

        return node_features

    @torch.no_grad()
    def build_node_features(
        self,
        obs_np:    np.ndarray,
        z_t_np:    np.ndarray,
        llm_np:    np.ndarray,
        device:    str = "cuda",
    ) -> torch.Tensor:
        """
        LIVE INFERENCE METHOD.
        Called by services/alpha_engine_svc.py to build GATv2 node features
        from Redis-fetched raw arrays.

        Args:
            obs_np:  (N, 47) numpy array of market observations
            z_t_np:  (16,)   numpy array of regime posterior
            llm_np:  (N, 15) numpy array of LLM alpha signals

        Returns:
            node_features: (N, 78) torch.Tensor on `device`
        """
        self.eval()
        obs   = torch.FloatTensor(obs_np).to(device)
        z_t   = torch.FloatTensor(z_t_np).to(device)
        llm   = torch.FloatTensor(llm_np).to(device)
        return self.forward(obs, z_t, llm)


# ── Standalone feature assembler (no learnable parameters) ───────────────────

class RawFeatureAssembler:
    """
    Deterministic baseline assembler.
    Used during pre-training or when the CrossModalFusionNetwork has not
    yet been trained. Produces the same 78-dim vector by simple concatenation
    without any learned gating or attention.

    This ensures the GATv2 can operate even before fusion network weights exist.
    """

    @staticmethod
    def assemble(
        obs_np: np.ndarray,
        z_t_np: np.ndarray,
        llm_np: np.ndarray,
    ) -> np.ndarray:
        """
        Args:
            obs_np: (N, 47) float32
            z_t_np: (16,)   float32 — broadcast across all assets
            llm_np: (N, 15) float32

        Returns:
            node_features: (N, 78) float32
        """
        N = obs_np.shape[0]
        z_broadcast = np.tile(z_t_np[np.newaxis, :], (N, 1))  # (N, 16)
        result = np.concatenate([obs_np, z_broadcast, llm_np], axis=-1)  # (N, 78)

        assert result.shape == (N, _NODE_FEAT_DIM), (
            f"Feature assembly shape mismatch: got {result.shape}, "
            f"expected ({N}, {_NODE_FEAT_DIM}). "
            f"Check _OBS_DIM={_OBS_DIM}, _REGIME_DIM={_REGIME_DIM}, _LLM_DIM={_LLM_DIM}."
        )
        return result.astype(np.float32)

    @staticmethod
    def get_zero_llm_features(n_assets: int = _N_ASSETS) -> np.ndarray:
        """
        Returns a zero-filled LLM feature matrix when the LLM agent pipeline
        is not yet producing output (e.g., at system startup or API downtime).
        The GATv2 will treat this as neutral / no opinion.
        """
        return np.zeros((n_assets, _LLM_DIM), dtype=np.float32)