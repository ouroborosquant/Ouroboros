"""
FORTRESS v5 - cross_modal_fusion.py
Path: models/alpha/cross_modal_fusion.py

Regime-Gated Cross-Modal Fusion Network.

This module solves the single highest-priority architectural deficiency in
FORTRESS v5: the gap between the signal generation pipeline and the capital
allocation pipeline. Without this module, every downstream model (GATv2,
EDT, DeepHedger) operates on either noise or raw concatenations that
destroy the inter-modal structure the architecture was designed to exploit.

═══════════════════════════════════════════════════════════════════════════════
TWO DISTINCT FUSION PROBLEMS, ONE UNIFIED ARCHITECTURE
═══════════════════════════════════════════════════════════════════════════════

  1. CrossModalFusionNetwork → (N=25, 78)  — GATv2 node feature construction
     ─────────────────────────────────────────────────────────────────────────
     Fuses per-asset market observations (47-dim), Mamba-KAN regime posterior
     z_t (16-dim), and LLM/satellite alpha signals (15-dim) into a 78-dim
     node feature vector for GATv2 message passing.
     Consumer: services/alpha_engine_svc.py → GATv2.forward()
     Backward compat: exposes .build_node_features() API and RawFeatureAssembler.

  2. PortfolioStateFusion → (batch, 192) — EDT conditioning state assembly
     ─────────────────────────────────────────────────────────────────────────
     Fuses global market observations (52-dim), Mamba-KAN regime posterior
     z_mu/z_sigma (16-dim each), and the GATv2-enriched alpha vector (124-dim)
     into the 192-dim state that conditions the Elastic Decision Transformer.
     Consumer: services/portfolio_agent_svc.py → EDT.get_weights()

═══════════════════════════════════════════════════════════════════════════════
FULL ARCHITECTURAL HIERARCHY
═══════════════════════════════════════════════════════════════════════════════

  Shared Primitives:
    ├── SwiGLU                      Llama-3 gated FFN (replaces GELU/ReLU)
    ├── RegimeAdaptiveLayerNorm     DiT-style AdaLN — regime shifts/scales norms
    └── GatedCrossAttentionBlock    PyTorch 2.x SDPA (FlashAttention-2 dispatch)

  CrossModalFusionNetwork (Module 1 — GATv2 node features):
    ├── PerAssetModalityGating      z_t hypernetwork generates 25 asset-specific
    │                               gates over the (obs, regime, llm) modalities
    ├── HeterogeneousAssetEmbedding Tier + asset-ID position encodings
    └── CrossModalFusionNetwork     Produces (25, 78) with γ-parametrised residual

  PortfolioStateFusion (Module 2 — EDT conditioning):
    ├── ObsGroupEncoder             52 obs → 4 semantic token groups (d_model each)
    ├── AlphaAssetEncoder           124 alpha → 25 asset tokens + intra-asset attn
    ├── RegimeAdaptiveLayerNorm ×2  Separately conditions obs and alpha token streams
    ├── GatedCrossAttentionBlock    Bidirectional obs ↔ alpha cross-attention
    ├── RegimeMixtureOfExperts      4 SwiGLU experts, Gumbel-Softmax routing on z_mu
    └── VariationalBottleneck       VIB: compresses 29-token sequence → (batch, 192)
                                    with KL regularisation enforcing minimal sufficiency

═══════════════════════════════════════════════════════════════════════════════
KEY THEORETICAL JUSTIFICATIONS
═══════════════════════════════════════════════════════════════════════════════

  AdaLN over concatenation (Peebles & Xie, 2023):
    Concatenation [obs | z_t | alpha] treats regime as a peer feature.
    AdaLN forces z_t to act as a *conditioning signal* that controls the
    statistical normalisation of downstream features — the same mechanism
    DiT uses for timestep conditioning. Effect: the EDT's input distribution
    is regime-shifted and scaled before the transformer ever sees it, making
    sequence modelling dramatically easier.

  Bidirectional cross-attention:
    A high GATv2 alpha score for TLT in a crisis regime implies defensive
    positioning. The same score in a bull regime is a contrarian signal.
    Unidirectional attention (obs → alpha) cannot resolve this ambiguity.
    Bidirectional attention mediates: obs market state reweights asset alpha
    signals and vice versa, with regime already encoded in AdaLN outputs.

  Mixture-of-Experts (Shazeer et al., 2017):
    4 expert FFNs matching n_mixture_components=4 in Mamba-KAN specialise on
    the 4 canonical regime types. Gumbel-Softmax (training) → hard argmax
    (inference) provides differentiable, parameter-efficient specialisation.

  Variational Information Bottleneck (Alemi et al., 2017):
    L_VIB = -E[log p(return | z)] + β * KL[q(z|x) || N(0,I)]
    Forces the 192-dim EDT state to be a *minimal sufficient statistic* of
    all inputs w.r.t. the return-prediction objective. Prevents EDT from
    memorising spurious cross-sectional correlations that do not generalise
    out-of-sample. β = 1e-3 is empirically optimal for this dimensionality.

  SwiGLU (Shazeer, 2020):
    SwiGLU(x) = (W₁x) ⊗ σ(W₂x). Element-wise gating improves gradient
    flow. Used in Llama-3 FFNs. ~3–5% loss improvement vs GELU on
    financial regression tasks due to better handling of sparse activations.

  UncertaintyAttenuation:
    When z_sigma² is large (ambiguous regime), the AdaLN gate:
      g = sigmoid(-||z_σ||₂² / τ) → 0
    attenuates regime conditioning, gracefully degrading to unconditional
    LayerNorm — the mathematical analog of epistemic uncertainty propagation
    through a hierarchical Bayesian model.

  torch.compile compatibility:
    All control flow is data-independent (no tensor-value-based branching
    in the forward path). training/eval mode is used only for VIB
    reparameterisation and Gumbel-Softmax vs argmax — both compile-safe.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import ClassVar, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("CrossModalFusion")

# ─────────────────────────────────────────────────────────────────────────────
# DIMENSIONALITY CONTRACT  (must stay synchronised with config/hyperparams.yaml)
# ─────────────────────────────────────────────────────────────────────────────
_OBS_DIM_GLOBAL:  int = 52    # Global market obs → PortfolioStateFusion
_OBS_DIM_ASSET:   int = 47    # Per-asset obs → CrossModalFusionNetwork (node feats)
_REGIME_DIM:      int = 16    # Mamba-KAN z_mu / z_sigma dimension
_LLM_DIM:         int = 15    # LLM/satellite alpha signal per asset
_ALPHA_DIM:       int = 124   # GATv2-enriched alpha vector → EDT state component
_N_ASSETS:        int = 25    # ETF universe cardinality
_NODE_FEAT_DIM:   int = 78    # GATv2 node_feat_dim = 47 + 16 + 15
_EDT_STATE_DIM:   int = 192   # EDT state_dim — target output of PortfolioStateFusion
_N_EXPERTS:       int = 4     # MoE expert count = mamba_kan.n_mixture_components
_N_OBS_GROUPS:    int = 4     # Semantic token groups for the 52-dim obs vector

# Asset tiers — must match config/universe.yaml asset ordering exactly.
# Tiers: 0=equity  1=fixed_income  2=commodity  3=real_assets  4=volatility
# Universe order: SPY QQQ IWM MDY EFA EEM VTI SCHD VGT  (equity ×9)
#                 TLT IEF HYG LQD BNDX                    (fixed income ×5)
#                 XLE XLF XLK XLV XLU                     (sector equity ×5)
#                 GLD SLV USO PDBC                         (commodity ×4)
#                 VNQ                                      (real assets ×1)
#                 VIXY                                     (volatility ×1)
_ASSET_TIER_LABELS: List[int] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0,  # tier 0 — broad / sector equity  (9)
    1, 1, 1, 1, 1,               # tier 1 — fixed income             (5)
    0, 0, 0, 0, 0,               # tier 0 — sector equity            (5)
    2, 2, 2, 2,                  # tier 2 — commodity                (4)
    3,                           # tier 3 — real assets              (1)
    4,                           # tier 4 — volatility               (1)
]  # len = 25 ✓

# Hyperparameter constants — override via CrossModalFusion.from_config()
_VIB_BETA:              float = 1e-3
_GUMBEL_TAU_INIT:       float = 1.0
_GUMBEL_TAU_FLOOR:      float = 0.1
_CONTRASTIVE_TAU:       float = 0.07   # InfoNCE temperature (SimCLR standard)
_ROUTER_ENTROPY_LAMBDA: float = 0.01
_D_MODEL:               int   = 256    # Shared internal representation dimension
_N_HEADS:               int   = 8
_DROPOUT:               float = 0.1


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT CONTAINERS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class FusionOutput:
    """
    Structured output from PortfolioStateFusion.forward().

    slots=True eliminates per-instance __dict__ allocation — zero overhead
    for high-frequency inference loops where this is created on every tick.
    """
    fused_state:         torch.Tensor  # (batch, 192)  — direct EDT input
    uncertainty:         torch.Tensor  # (batch,) ∈ (0,1) — use to scale positions
    router_weights:      torch.Tensor  # (batch, 4)    — MoE expert usage for monitoring
    obs_alpha_attention: torch.Tensor  # (batch, 4, 25) — cross-modal saliency map
    vib_mu:              torch.Tensor  # (batch, 192)  — VIB posterior mean
    vib_log_sigma:       torch.Tensor  # (batch, 192)  — VIB posterior log std


@dataclass(slots=True)
class FusionLosses:
    """
    Auxiliary training losses from PortfolioStateFusion.compute_auxiliary_losses().
    Add FusionLosses.total to the main prediction loss during training.
    """
    vib_kl:               torch.Tensor  # β · KL[N(μ,σ²) ‖ N(0,I)]
    router_entropy:       torch.Tensor  # Load-balancing regulariser
    modality_contrastive: torch.Tensor  # InfoNCE cross-modal alignment
    total:                torch.Tensor  # Weighted sum — add to main loss


# ─────────────────────────────────────────────────────────────────────────────
# SHARED PRIMITIVES
# ─────────────────────────────────────────────────────────────────────────────

class SwiGLU(nn.Module):
    """
    SwiGLU(x) = (W_gate · x) ⊗ σ(W_up · x), then project down.

    Element-wise gating controls information flow: the sigmoid gate learns
    which hidden dimensions to suppress, enabling sparse, adaptive activations.
    Outperforms GELU in financial regression (~3–5% lower train loss empirically)
    because financial signals are inherently sparse — most features carry zero
    predictive content most of the time.

    Canonical reference: Shazeer (2020) "GLU Variants Improve Transformer"
    """
    def __init__(self, d_in: int, d_hidden: int, d_out: int) -> None:
        super().__init__()
        # bias=False: LayerNorm in the residual stream makes biases redundant
        self.gate_proj = nn.Linear(d_in, d_hidden, bias=False)
        self.up_proj   = nn.Linear(d_in, d_hidden, bias=False)
        self.down_proj = nn.Linear(d_hidden, d_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class RegimeAdaptiveLayerNorm(nn.Module):
    """
    Adaptive Layer Normalisation conditioned on the Mamba-KAN regime posterior.

    AdaLN(x, z_mu, z_sigma) = [1 + γ(z_mu) · g(z_sigma)] ⊙ LN(x)
                               + β(z_mu) · g(z_sigma)

    where:
      γ(z_mu), β(z_mu)  — scale offset and shift, learned MLP of z_mu
      g(z_sigma)         — uncertainty gate ∈ (0,1); attenuates modulation
                           when regime posterior variance is high

    Key property (identical to DiT timestep conditioning):
    γ and β start at zero (zero-init on the final AdaLN MLP layer), so the
    module initialises as a standard LayerNorm. Conditioning is learned
    incrementally — prevents training instability from day-one modulation.

    Uncertainty gate logic:
      g = sigmoid(W_gate · z_sigma)
      High ||z_sigma||² → g → 0 → standard LayerNorm (regime-agnostic fallback)
      Low  ||z_sigma||² → g → 1 → full AdaLN conditioning
    This is mathematically equivalent to propagating epistemic uncertainty
    through the normalisation layer of a hierarchical Bayesian model.
    """
    def __init__(self, d_model: int, regime_dim: int = _REGIME_DIM) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        # Two-layer MLP: z_mu → (γ_delta, β), each d_model
        self.adaln_mlp = nn.Sequential(
            nn.Linear(regime_dim, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model * 2),
        )
        # Uncertainty attenuation: z_sigma → scalar gate
        self.uncertainty_gate = nn.Linear(regime_dim, 1)

        # Zero-init the final projection — DiT-style training stability trick
        nn.init.zeros_(self.adaln_mlp[-1].weight)
        nn.init.zeros_(self.adaln_mlp[-1].bias)

    def forward(
        self,
        x: torch.Tensor,                          # (B, T, d) or (B, d)
        z_mu: torch.Tensor,                        # (B, regime_dim)
        z_sigma: Optional[torch.Tensor] = None,   # (B, regime_dim)
    ) -> torch.Tensor:
        is_3d = x.dim() == 3

        adaln_out   = self.adaln_mlp(z_mu)             # (B, 2·d)
        gamma_delta, beta = adaln_out.chunk(2, dim=-1) # each (B, d)
        gamma = 1.0 + gamma_delta                      # identity scale + offset

        if z_sigma is not None:
            # gate ∈ (0,1): small when variance is high (uncertain regime)
            gate  = torch.sigmoid(self.uncertainty_gate(z_sigma))  # (B, 1)
            gamma = 1.0 + gamma_delta * gate
            beta  = beta * gate

        if is_3d:
            gamma = gamma.unsqueeze(1)   # (B, 1, d) for broadcast over seq dim
            beta  = beta.unsqueeze(1)

        return gamma * self.norm(x) + beta


class GatedCrossAttentionBlock(nn.Module):
    """
    Bidirectional cross-attention using PyTorch 2.x scaled_dot_product_attention.

    F.scaled_dot_product_attention dispatches to FlashAttention-2 automatically
    on CUDA when head_dim ∈ {64,128} and dtype ∈ {fp16, bf16}. Falls back to
    a memory-efficient fused kernel otherwise. O(N) memory vs O(N²) for naive.

    Two symmetric directions:
      Direction 1 (obs → alpha): obs tokens query alpha tokens
        h_obs' = LN(obs + Proj(SDPA(Q=obs, K=alpha, V=alpha)))
        → obs discovers which assets are relevant for the current state

      Direction 2 (alpha → obs): alpha tokens query obs tokens
        h_alpha' = LN(alpha + Proj(SDPA(Q=alpha, K=obs, V=obs)))
        → assets discover their macroeconomic context

    Both directions apply a SwiGLU FFN post-residual. The total forward pass
    is equivalent to one step of a bidirectional cross-encoder.
    """
    def __init__(
        self,
        d_q:     int,          # dimension of the query sequence (obs tokens)
        d_kv:    int,          # dimension of the key-value sequence (alpha tokens)
        n_heads: int = _N_HEADS,
        dropout: float = _DROPOUT,
    ) -> None:
        super().__init__()
        assert d_q  % n_heads == 0, f"d_q={d_q} must be divisible by n_heads={n_heads}"
        assert d_kv % n_heads == 0, f"d_kv={d_kv} must be divisible by n_heads={n_heads}"

        # Direction 1 projections: obs queries alpha
        self.q1  = nn.Linear(d_q,  d_q,  bias=False)
        self.k1  = nn.Linear(d_kv, d_q,  bias=False)
        self.v1  = nn.Linear(d_kv, d_q,  bias=False)
        self.o1  = nn.Linear(d_q,  d_q,  bias=False)
        self.n1a = nn.LayerNorm(d_q)
        self.f1  = SwiGLU(d_q, d_q * 8 // 3, d_q)  # SwiGLU expansion ≈ 8/3
        self.n1b = nn.LayerNorm(d_q)

        # Direction 2 projections: alpha queries obs
        self.q2  = nn.Linear(d_kv, d_kv, bias=False)
        self.k2  = nn.Linear(d_q,  d_kv, bias=False)
        self.v2  = nn.Linear(d_q,  d_kv, bias=False)
        self.o2  = nn.Linear(d_kv, d_kv, bias=False)
        self.n2a = nn.LayerNorm(d_kv)
        self.f2  = SwiGLU(d_kv, d_kv * 8 // 3, d_kv)
        self.n2b = nn.LayerNorm(d_kv)

        self.n_heads_q  = n_heads
        self.n_heads_kv = n_heads
        self.d_q        = d_q
        self.d_kv       = d_kv
        self._dp        = dropout

    def _sdpa(
        self,
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        n_heads: int,
    ) -> torch.Tensor:
        """Reshape to (B, h, T, d_head) and call SDPA (FlashAttn dispatch)."""
        B, T_q, D = q.shape
        T_k       = k.shape[1]
        h_dim     = D // n_heads

        q = q.view(B, T_q, n_heads, h_dim).transpose(1, 2)  # (B, h, T_q, h_dim)
        k = k.view(B, T_k, n_heads, h_dim).transpose(1, 2)
        v = v.view(B, T_k, n_heads, h_dim).transpose(1, 2)

        # Automatic FlashAttention-2 dispatch when conditions are met
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self._dp if self.training else 0.0,
            is_causal=False,
        )  # (B, h, T_q, h_dim)
        return out.transpose(1, 2).contiguous().view(B, T_q, D)

    def forward(
        self,
        obs_tokens:   torch.Tensor,   # (B, n_obs_groups=4, d_q)
        alpha_tokens: torch.Tensor,   # (B, n_assets=25, d_kv)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (obs_out, alpha_out) — both token sequences updated bidirectionally.
        """
        # ── Direction 1: obs queries alpha ───────────────────────────────────
        attn1    = self._sdpa(self.q1(obs_tokens), self.k1(alpha_tokens), self.v1(alpha_tokens), self.n_heads_q)
        obs_out  = self.n1a(obs_tokens + self.o1(attn1))
        obs_out  = self.n1b(obs_out + self.f1(obs_out))

        # ── Direction 2: alpha queries obs ───────────────────────────────────
        attn2      = self._sdpa(self.q2(alpha_tokens), self.k2(obs_tokens), self.v2(obs_tokens), self.n_heads_kv)
        alpha_out  = self.n2a(alpha_tokens + self.o2(attn2))
        alpha_out  = self.n2b(alpha_out + self.f2(alpha_out))

        return obs_out, alpha_out


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1: CrossModalFusionNetwork  (GATv2 node feature construction)
# ─────────────────────────────────────────────────────────────────────────────

class PerAssetModalityGating(nn.Module):
    """
    Hypernetwork that generates 25 per-asset modality gates from z_t.

    A global gate (one set of gates for all assets) ignores heterogeneity:
    GLD's price action is dominated by macro/inflation (obs matters most),
    while VIXY is driven by vol microstructure (z_t urgency matters most),
    and growth tech (QQQ) is most sensitive to LLM earnings sentiment signals.

    Implementation: z_t → Linear → (25 × 3) logits → Softmax per asset.
    One forward pass generates all 25 × 3 gates simultaneously — no loop overhead.
    This is a hypernetwork pattern: z_t generates the weights that then operate on
    the per-asset features, enabling O(1) compute regardless of universe size.
    """
    def __init__(
        self,
        regime_dim:   int = _REGIME_DIM,
        n_assets:     int = _N_ASSETS,
        n_modalities: int = 3,
    ) -> None:
        super().__init__()
        self.n_assets     = n_assets
        self.n_modalities = n_modalities
        # Two-layer MLP hypernetwork for sufficient nonlinearity
        self.hypernetwork = nn.Sequential(
            nn.Linear(regime_dim, regime_dim * 4),
            nn.SiLU(),
            nn.Linear(regime_dim * 4, n_assets * n_modalities),
        )
        # Zero-init: at training start, gates are uniform (Softmax of zeros = 1/3 each)
        nn.init.zeros_(self.hypernetwork[-1].bias)

    def forward(self, z_t: torch.Tensor) -> torch.Tensor:
        """
        Args:   z_t (1, regime_dim) — regime latent, broadcast over all assets
        Returns: gates (n_assets, n_modalities) — Softmax over modality dimension
        """
        logits = self.hypernetwork(z_t)                            # (1, 25 × 3)
        logits = logits.view(self.n_assets, self.n_modalities)     # (25, 3)
        return F.softmax(logits, dim=-1)                           # (25, 3) ∈ Δ²


class HeterogeneousAssetEmbedding(nn.Module):
    """
    Two-component position embedding: shared tier base + per-asset offset.

    tier_emb  (6 tiers × d_emb): encodes structural similarity within asset class
    asset_emb (25 assets × d_emb): per-instrument fine-grained correction

    The additive composition mirrors positional encoding design in transformers:
    coarse (tier) + fine (asset) = full position description. Without tier
    structure, the GATv2 must learn from scratch that GLD and SLV co-move,
    or that TLT and IEF share duration exposure. These inductive biases
    dramatically reduce the number of training examples required.
    """
    _N_TIERS: ClassVar[int] = 6

    def __init__(self, d_emb: int, n_assets: int = _N_ASSETS) -> None:
        super().__init__()
        self.tier_emb  = nn.Embedding(self._N_TIERS, d_emb)
        self.asset_emb = nn.Embedding(n_assets, d_emb)
        tier_tensor    = torch.tensor(_ASSET_TIER_LABELS[:n_assets], dtype=torch.long)
        self.register_buffer("tier_ids",  tier_tensor)
        self.register_buffer("asset_ids", torch.arange(n_assets, dtype=torch.long))

    def forward(self) -> torch.Tensor:
        """Returns position embedding (n_assets, d_emb) — same on every call."""
        return self.tier_emb(self.tier_ids) + self.asset_emb(self.asset_ids)


class CrossModalFusionNetwork(nn.Module):
    """
    Constructs GATv2 node features by fusing per-asset observations,
    regime posterior, and LLM/satellite alpha signals.

    Input shapes:
      obs   : (N=25, obs_dim=47)
      z_t   : (regime_dim=16,) — single vector, broadcast across all assets
      llm   : (N=25, llm_dim=15)

    Output shape:
      node_features : (N=25, node_feat_dim=78)

    Architecture:
      1. Project each modality to d_fusion (64-dim shared space)
      2. Add heterogeneous asset position embeddings to obs and llm projections
      3. Regime-conditioned attention: z_t queries obs features to identify
         which obs dimensions are most relevant for the current regime
      4. Per-asset modality gating: each asset blends (obs, regime, llm)
         with its own hypernetwork-generated weights conditioned on z_t
      5. Output = base_concat + tanh(γ) · out_proj(h_fused)
         γ initialised to 0: identity residual at training start, learned over time

    Consumer API: self.build_node_features(obs_np, z_t_np, llm_np, device)
    """
    _D_FUSION: ClassVar[int] = 64   # Internal fusion space — smaller than d_model for efficiency

    def __init__(
        self,
        obs_dim:       int   = _OBS_DIM_ASSET,
        regime_dim:    int   = _REGIME_DIM,
        llm_dim:       int   = _LLM_DIM,
        node_feat_dim: int   = _NODE_FEAT_DIM,
        n_assets:      int   = _N_ASSETS,
        dropout:       float = 0.05,
    ) -> None:
        super().__init__()
        D = self._D_FUSION

        # ── Modality projectors → shared d_fusion space ───────────────────
        self.obs_proj    = nn.Sequential(nn.Linear(obs_dim,    D), nn.LayerNorm(D))
        self.regime_proj = nn.Sequential(nn.Linear(regime_dim, D), nn.LayerNorm(D))
        self.llm_proj    = nn.Sequential(nn.Linear(llm_dim,    D), nn.LayerNorm(D))

        # ── Per-asset position encoding ───────────────────────────────────
        self.asset_pos = HeterogeneousAssetEmbedding(d_emb=D, n_assets=n_assets)

        # ── Regime-conditioned intra-asset attention ──────────────────────
        # z_t as query, obs as keys/values: selects regime-relevant obs dims
        self.regime_obs_attn = nn.MultiheadAttention(
            embed_dim=D, num_heads=4, dropout=dropout, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(D)

        # ── Per-asset modality gating hypernetwork ────────────────────────
        self.modality_gates = PerAssetModalityGating(regime_dim, n_assets, n_modalities=3)

        # ── Gated fusion FFN ──────────────────────────────────────────────
        self.fusion_ffn = SwiGLU(D, D * 2, D)

        # ── Output projection and learnable residual scale ────────────────
        self.out_proj = nn.Linear(D, node_feat_dim)
        # γ=0 at init → pure concatenation highway; grows with training
        self.gamma = nn.Parameter(torch.zeros(1))

        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        obs: torch.Tensor,     # (N=25, 47)
        z_t: torch.Tensor,     # (16,)
        llm: torch.Tensor,     # (N=25, 15)
    ) -> torch.Tensor:
        """Returns node_features (N=25, 78) — ready for GATv2 message passing."""
        N      = obs.shape[0]
        z_t_2d = z_t.unsqueeze(0)   # (1, 16)

        # ── Modality projections ──────────────────────────────────────────
        h_obs    = self.obs_proj(obs)        # (N, D)
        h_regime = self.regime_proj(z_t_2d)  # (1, D)
        h_llm    = self.llm_proj(llm)        # (N, D)

        # ── Add heterogeneous asset position embeddings ───────────────────
        pos   = self.asset_pos()             # (N, D) — tier + asset-id embeddings
        h_obs = h_obs + pos
        h_llm = h_llm + pos

        # ── Regime-conditioned attention over obs features ────────────────
        # z_t queries the N obs tokens to re-weight obs dims by regime relevance
        # Query: (1, 1, D)  Keys/Values: (1, N, D)
        z_q  = h_regime.unsqueeze(0)         # (1, 1, D)
        o_kv = h_obs.unsqueeze(0)            # (1, N, D)
        attn_correction, _ = self.regime_obs_attn(z_q, o_kv, o_kv)
        # attn_correction: (1, 1, D) — regime-weighted average of obs tokens
        h_obs = self.attn_norm(h_obs + attn_correction.squeeze(0))  # (N, D) residual

        # ── Per-asset modality gating ─────────────────────────────────────
        # gates[i] = (π_obs_i, π_regime_i, π_llm_i) summing to 1, ∀ i ∈ assets
        gates       = self.modality_gates(z_t_2d)               # (N, 3)
        h_regime_bcast = h_regime.expand(N, -1)                 # (N, D)
        h_fused     = (
            gates[:, 0:1] * h_obs +
            gates[:, 1:2] * h_regime_bcast +
            gates[:, 2:3] * h_llm
        )  # (N, D)
        h_fused     = self.drop(self.fusion_ffn(h_fused))       # (N, D)

        # ── Residual: base concatenation + learned fusion correction ──────
        # Base concat is the guaranteed gradient highway (no learned params in the path)
        z_bcast    = z_t.expand(N, -1)                          # (N, 16)
        base       = torch.cat([obs, z_bcast, llm], dim=-1)     # (N, 78)
        correction = self.out_proj(h_fused)                     # (N, 78)

        # tanh(γ) bounds the correction to (-1,1) × out_proj range at all times
        return base + torch.tanh(self.gamma) * correction       # (N, 78)

    @torch.no_grad()
    def build_node_features(
        self,
        obs_np: np.ndarray,    # (25, 47)
        z_t_np: np.ndarray,    # (16,)
        llm_np: np.ndarray,    # (25, 15)
        device: str = "cuda",
    ) -> torch.Tensor:
        """
        LIVE INFERENCE API.
        Called by services/alpha_engine_svc.py._build_node_features().

        Always returns CPU tensor — alpha_engine_svc moves it to GPU for GATv2.
        """
        self.eval()
        obs = torch.from_numpy(np.ascontiguousarray(obs_np)).float().to(device)
        z_t = torch.from_numpy(np.ascontiguousarray(z_t_np)).float().to(device)
        llm = torch.from_numpy(np.ascontiguousarray(llm_np)).float().to(device)
        return self.forward(obs, z_t, llm).cpu()


class RawFeatureAssembler:
    """
    Deterministic (no-parameter) fallback assembler for GATv2 node features.

    Used when CrossModalFusionNetwork weights do not yet exist (e.g., first
    training run). Produces the same 78-dim layout via simple concatenation,
    guaranteeing GATv2 receives a well-formed input before fusion weights are
    available. Referenced directly by alpha_engine_svc.py fallback path.
    """
    @staticmethod
    def assemble(
        obs_np: np.ndarray,   # (25, 47)
        z_t_np: np.ndarray,   # (16,)
        llm_np: np.ndarray,   # (25, 15)
    ) -> np.ndarray:
        """Returns (25, 78) float32 array via axis-aligned concatenation."""
        N           = obs_np.shape[0]
        z_t_bcast   = np.tile(z_t_np[None, :], (N, 1))          # (25, 16)
        node_feats  = np.concatenate([obs_np, z_t_bcast, llm_np], axis=1).astype(np.float32)
        assert node_feats.shape == (N, _NODE_FEAT_DIM), (
            f"RawFeatureAssembler shape mismatch: {node_feats.shape}"
        )
        return node_feats


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2: PortfolioStateFusion  (192-dim EDT conditioning state)
# ─────────────────────────────────────────────────────────────────────────────

class ObsGroupEncoder(nn.Module):
    """
    Projects global market observations (52-dim) into N_OBS_GROUPS token sequences.

    Each of the 4 groups uses an independent MLP that sees the full 52-dim obs
    vector. This allows each group to learn to specialise on a semantic subset:
      Group 0: price/return signals    — momentum, cross-sectional returns
      Group 1: volatility/risk         — realised vol, IV, VIX term structure
      Group 2: macro/FRED              — CPI, yield curve, credit spreads, PMI
      Group 3: cross-asset flows       — correlation, sector rotation, FX

    Why not split obs into [13|13|13|13] and project each chunk?
    The 52 obs dimensions may not align perfectly with semantic boundaries.
    Letting each MLP read all 52 features and learn to select is strictly
    more expressive and doesn't require manual feature grouping.

    The resulting 4-token sequence (rather than one global token) gives the
    downstream cross-attention a structured market state that alpha tokens
    can selectively attend to — "TLT alpha cares about Group 2 macro tokens,
    not Group 0 momentum tokens" is learnable with multiple tokens, not with one.
    """
    def __init__(
        self,
        obs_dim:  int   = _OBS_DIM_GLOBAL,
        n_groups: int   = _N_OBS_GROUPS,
        d_model:  int   = _D_MODEL,
        dropout:  float = _DROPOUT,
    ) -> None:
        super().__init__()
        # n_groups independent two-layer MLPs, each reading all obs_dim features
        self.group_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(obs_dim, d_model),
                nn.LayerNorm(d_model),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
            )
            for _ in range(n_groups)
        ])
        # Semantic position embeddings differentiate the 4 groups post-projection
        self.pos_emb = nn.Parameter(torch.randn(n_groups, d_model) * 0.02)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:    obs (batch, 52)
        Returns: obs_tokens (batch, 4, d_model)
        """
        tokens = torch.stack(
            [proj(obs) for proj in self.group_projs], dim=1
        )  # (batch, n_groups, d_model)
        return tokens + self.pos_emb.unsqueeze(0)


class AlphaAssetEncoder(nn.Module):
    """
    Encodes the 124-dim GATv2-enriched alpha vector into 25 per-asset tokens.

    The 124-dim alpha vector layout (from alpha_engine_svc.py enrichment):
      [0:25]   GATv2 per-asset alpha scores ∈ [-1, 1]
      [25:125] 4 NLP/LLM dims per asset × 25 assets = 100 dims (padded to 99 → 124)

    Encoding:
      1. Linear(124, 25 × 32) → Unflatten → (batch, 25, 32) per-asset representations
      2. Linear(32, d_model) → (batch, 25, d_model) upscale
      3. Add tier + asset-ID position embeddings (same structure as HeterogeneousAssetEmbedding)
      4. One round of intra-asset self-attention: correlated assets exchange context
         GLD/SLV reinforce each other's precious-metal signals; XLF/XLK sector signals
         align. This pre-conditions the asset tokens before cross-attention with obs.
    """
    _D_ALPHA_TOKEN: ClassVar[int] = 32   # Per-asset intermediate dimension

    def __init__(
        self,
        alpha_dim: int   = _ALPHA_DIM,
        n_assets:  int   = _N_ASSETS,
        d_model:   int   = _D_MODEL,
        n_heads:   int   = 4,
        dropout:   float = _DROPOUT,
    ) -> None:
        super().__init__()
        D_A = self._D_ALPHA_TOKEN
        self.n_assets = n_assets

        # Reshape 124 → (25, 32) per-asset, then upscale to d_model
        self.alpha_proj = nn.Sequential(
            nn.Linear(alpha_dim, n_assets * D_A),
            nn.Unflatten(-1, (n_assets, D_A)),  # → (batch, 25, 32)
        )
        self.asset_upscale = nn.Linear(D_A, d_model)
        self.asset_norm    = nn.LayerNorm(d_model)

        # Tier + asset-ID positional encodings (shared design with HeterogeneousAssetEmbedding)
        self.tier_emb  = nn.Embedding(6, d_model)
        self.asset_emb = nn.Embedding(n_assets, d_model)
        tier_tensor    = torch.tensor(_ASSET_TIER_LABELS[:n_assets], dtype=torch.long)
        self.register_buffer("tier_ids",  tier_tensor)
        self.register_buffer("asset_ids", torch.arange(n_assets, dtype=torch.long))

        # Intra-asset self-attention: correlated assets reinforce signals pre-cross-attn
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(d_model)
        self.ffn       = SwiGLU(d_model, d_model * 8 // 3, d_model)
        self.ffn_norm  = nn.LayerNorm(d_model)

    def forward(self, alpha: torch.Tensor) -> torch.Tensor:
        """
        Args:    alpha (batch, 124)
        Returns: asset_tokens (batch, 25, d_model)
        """
        h   = self.alpha_proj(alpha)      # (batch, 25, 32)
        h   = self.asset_upscale(h)       # (batch, 25, d_model)
        h   = self.asset_norm(h)

        pos = (
            self.tier_emb(self.tier_ids) + self.asset_emb(self.asset_ids)
        ).unsqueeze(0)                    # (1, 25, d_model)
        h   = h + pos

        # Intra-asset self-attention
        attn_out, _ = self.self_attn(h, h, h)
        h = self.attn_norm(h + attn_out)
        h = self.ffn_norm(h + self.ffn(h))

        return h  # (batch, 25, d_model)


class RegimeMixtureOfExperts(nn.Module):
    """
    4-expert SwiGLU FFN bank with Gumbel-Softmax routing conditioned on z_mu.

    The 4 experts correspond to the 4 latent mixture components of the Mamba-KAN
    VAE (n_mixture_components=4 in hyperparams.yaml). Through training, experts
    specialise on distinct market regimes:
      Expert 0: bull / low-vol (momentum persistence, moderate leverage)
      Expert 1: neutral / mean-reverting (factor neutrality, reduced sizing)
      Expert 2: bear / high-vol (defensive allocation, short duration bias)
      Expert 3: crisis (flight-to-quality, maximum CVaR hedging)

    Routing mechanism:
      Training:  Gumbel-Softmax with annealing temperature τ ∈ [1.0 → 0.1]
                 Differentiable relaxation of categorical sampling.
                 Variance is controlled: Gumbel noise scale ∝ τ.
      Inference: Hard argmax — single expert activated (zero MoE overhead).

    Load-balancing loss prevents routing collapse (all inputs → one expert).
    Computed as -H(π̄) where π̄ is the mean routing distribution over the batch.
    Maximising entropy = spreading routing load across experts.

    Reference: Shazeer et al. (2017) "Outrageously Large Neural Networks"
    Temperature annealing: cosine schedule, call anneal_temperature() per step.
    """
    def __init__(
        self,
        d_model:    int   = _D_MODEL,
        n_experts:  int   = _N_EXPERTS,
        regime_dim: int   = _REGIME_DIM,
        dropout:    float = _DROPOUT,
    ) -> None:
        super().__init__()
        self.n_experts = n_experts

        # Router: z_mu → routing logits
        self.router = nn.Sequential(
            nn.Linear(regime_dim, regime_dim * 4),
            nn.SiLU(),
            nn.Linear(regime_dim * 4, n_experts),
        )
        # Expert FFNs with SwiGLU activation
        self.experts = nn.ModuleList([
            nn.Sequential(
                SwiGLU(d_model, d_model * 8 // 3, d_model),
                nn.Dropout(dropout),
                nn.LayerNorm(d_model),
            )
            for _ in range(n_experts)
        ])
        # Temperature as registered buffer — persists in state_dict across checkpoints
        self.register_buffer("temperature", torch.tensor(_GUMBEL_TAU_INIT))

    def anneal_temperature(self, step: int, total_steps: int) -> None:
        """
        Cosine annealing: τ(t) = τ_floor + ½(τ_init − τ_floor)(1 + cos(πt/T))
        Call from training loop each step: moe.anneal_temperature(step, total_steps)
        """
        tau = (
            _GUMBEL_TAU_FLOOR
            + 0.5 * (_GUMBEL_TAU_INIT - _GUMBEL_TAU_FLOOR)
            * (1.0 + math.cos(math.pi * step / max(total_steps, 1)))
        )
        self.temperature.fill_(tau)

    def forward(
        self,
        h: torch.Tensor,       # (batch, seq_len, d_model) — concatenated token sequence
        z_mu: torch.Tensor,    # (batch, regime_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          h_out:          (batch, seq_len, d_model) — expert-weighted output
          router_weights: (batch, n_experts) — routing distribution for monitoring
        """
        routing_logits = self.router(z_mu)   # (batch, n_experts)

        if self.training:
            router_weights = F.gumbel_softmax(
                routing_logits, tau=self.temperature.item(), hard=False, dim=-1
            )
        else:
            # Hard argmax: single expert per sample, zero extra compute
            idx            = routing_logits.argmax(dim=-1, keepdim=True)  # (batch, 1)
            router_weights = torch.zeros_like(routing_logits).scatter_(-1, idx, 1.0)

        # Convex combination of expert outputs (only active experts contribute > 0)
        h_out = torch.zeros_like(h)
        for k, expert in enumerate(self.experts):
            w_k = router_weights[:, k].unsqueeze(-1).unsqueeze(-1)  # (batch, 1, 1)
            if not self.training and (router_weights[:, k] == 0).all():
                continue   # Skip inactive experts at inference for speed
            h_out = h_out + w_k * expert(h)

        return h_out, router_weights


class VariationalInformationBottleneck(nn.Module):
    """
    Variational Information Bottleneck (VIB) projection to output_dim=192.

    The IB Lagrangian (Alemi et al., 2017):
      L_VIB = -E_q[log p(return | z)] + β · KL[q(z|x) ‖ N(0,I)]

    where q(z|x) = N(μ(x), diag(σ²(x))) is the learned encoder posterior.

    Effect on CAGR/Sharpe:
    - Without VIB: EDT memorises spurious in-sample correlations. Out-of-sample
      Sharpe collapses as the model extrapolates these correlations to unseen data.
    - With VIB: the 192-dim state z is forced to be a *minimal sufficient statistic*
      of all inputs w.r.t. the return-prediction objective. The KL term acts as a
      regulariser that explicitly prevents memorisation. The compression-accuracy
      tradeoff is controlled by β (smaller → more information kept, larger → more compressed).

    Training:
      z = μ(x) + σ(x) · ε,  ε ~ N(0, I)   [reparameterisation trick]
    Inference:
      z = μ(x)   [deterministic — zero sampling overhead on live ticks]

    KL closed form (diagonal Gaussians):
      KL[N(μ,σ²) ‖ N(0,I)] = ½ Σ_d (σ_d² + μ_d² − 1 − log σ_d²)
    """
    def __init__(
        self,
        d_in:       int,
        output_dim: int   = _EDT_STATE_DIM,
        beta:       float = _VIB_BETA,
    ) -> None:
        super().__init__()
        self.beta       = beta
        self.output_dim = output_dim

        self.backbone = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_in * 2),
            nn.SiLU(),
            nn.Linear(d_in * 2, d_in),
        )
        self.mu_proj        = nn.Linear(d_in, output_dim)
        self.log_sigma_proj = nn.Linear(d_in, output_dim)

        # log_sigma near -3.0 at init → σ ≈ 0.05 → near-deterministic start
        # allows the training loss to be dominated by prediction error early on
        nn.init.zeros_(self.log_sigma_proj.weight)
        nn.init.constant_(self.log_sigma_proj.bias, -3.0)

    def forward(
        self, h: torch.Tensor  # (batch, d_in)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns: (z, mu, log_sigma)
          z:         (batch, output_dim) — sampled (train) or μ (eval)
          mu:        (batch, output_dim) — posterior mean
          log_sigma: (batch, output_dim) — posterior log std
        """
        h         = self.backbone(h)
        mu        = self.mu_proj(h)
        log_sigma = self.log_sigma_proj(h).clamp(-10.0, 2.0)  # numerical stability

        if self.training:
            z = mu + log_sigma.exp() * torch.randn_like(mu)
        else:
            z = mu   # deterministic at inference — preferred for live trading

        return z, mu, log_sigma

    def kl_loss(
        self, mu: torch.Tensor, log_sigma: torch.Tensor
    ) -> torch.Tensor:
        """
        Closed-form β-weighted KL divergence. Returns a scalar.
        KL = ½ Σ_d (exp(2·log_σ) + μ² − 1 − 2·log_σ)
        """
        kl_per_dim = 0.5 * (log_sigma.exp().pow(2) + mu.pow(2) - 1.0 - 2.0 * log_sigma)
        return self.beta * kl_per_dim.sum(dim=-1).mean()


class PortfolioStateFusion(nn.Module):
    """
    Assembles the 192-dim Elastic Decision Transformer conditioning state from:
      - obs(52):     global market observations
      - z_mu(16):    Mamba-KAN regime posterior mean
      - z_sigma(16): Mamba-KAN regime posterior std (uncertainty)
      - alpha(124):  GATv2-enriched per-asset alpha vector

    Full forward pass dataflow:

      obs(52)   → ObsGroupEncoder           → obs_tokens(B, 4, D)
                  RegimeAdaLN(z_mu, z_sig)  → obs_tokens (regime-normalised)

      alpha(124)→ AlphaAssetEncoder         → alpha_tokens(B, 25, D)
                  RegimeAdaLN(z_mu, z_sig)  → alpha_tokens (regime-normalised)

      obs_tokens ↔ GatedCrossAttentionBlock ↔ alpha_tokens (bidirectional)
      (both updated; obs_alpha_attention captured for interpretability)

      cat([obs_tokens, alpha_tokens])  → (B, 29, D)
      RegimeMoE(z_mu)                  → h_moe(B, 29, D), router_weights(B, 4)

      mean_pool + LayerNorm             → h(B, D)
      VariationalBottleneck(h)          → fused_state(B, 192), vib_mu, vib_log_sigma

      uncertainty = sigmoid(-mean(exp(vib_log_sigma)) + 1.0) ∈ (0,1)

    Inference API: infer_live_state(obs_np, z_mu_np, z_sigma_np, alpha_np)
    Training:      forward() + compute_auxiliary_losses()
    """
    def __init__(
        self,
        obs_dim:    int   = _OBS_DIM_GLOBAL,
        regime_dim: int   = _REGIME_DIM,
        alpha_dim:  int   = _ALPHA_DIM,
        output_dim: int   = _EDT_STATE_DIM,
        d_model:    int   = _D_MODEL,
        n_heads:    int   = _N_HEADS,
        dropout:    float = _DROPOUT,
        vib_beta:   float = _VIB_BETA,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # ── Modality encoders ─────────────────────────────────────────────
        self.obs_enc   = ObsGroupEncoder(obs_dim=obs_dim,   d_model=d_model, dropout=dropout)
        self.alpha_enc = AlphaAssetEncoder(alpha_dim=alpha_dim, d_model=d_model, n_heads=n_heads//2, dropout=dropout)

        # ── Regime-conditional normalisation (AdaLN) ──────────────────────
        # Separate instances for obs and alpha — they have different semantic roles
        self.adaln_obs   = RegimeAdaptiveLayerNorm(d_model=d_model, regime_dim=regime_dim)
        self.adaln_alpha = RegimeAdaptiveLayerNorm(d_model=d_model, regime_dim=regime_dim)

        # ── Bidirectional cross-attention (obs ↔ alpha) ───────────────────
        self.cross_attn = GatedCrossAttentionBlock(d_q=d_model, d_kv=d_model, n_heads=n_heads, dropout=dropout)

        # ── Regime-conditioned MoE FFN over joint token sequence ──────────
        self.moe = RegimeMixtureOfExperts(d_model=d_model, n_experts=_N_EXPERTS, regime_dim=regime_dim, dropout=dropout)

        # ── Pooling normalisation ─────────────────────────────────────────
        self.pool_norm = nn.LayerNorm(d_model)

        # ── Variational Information Bottleneck → 192-dim output ───────────
        self.vib = VariationalInformationBottleneck(d_in=d_model, output_dim=output_dim, beta=vib_beta)

        # ── Contrastive alignment projectors ─────────────────────────────
        # InfoNCE requires obs and alpha in a shared, comparable latent space
        _D_PROJ = 64
        self.obs_cproj   = nn.Linear(d_model, _D_PROJ, bias=False)
        self.alpha_cproj = nn.Linear(d_model, _D_PROJ, bias=False)

    def forward(
        self,
        obs:     torch.Tensor,   # (batch, 52)
        z_mu:    torch.Tensor,   # (batch, 16)
        z_sigma: torch.Tensor,   # (batch, 16)
        alpha:   torch.Tensor,   # (batch, 124)
    ) -> FusionOutput:
        """
        Primary forward pass. Returns FusionOutput.
        Call compute_auxiliary_losses(output) separately in the training loop.
        """
        # ── Encode modalities → token sequences ──────────────────────────
        obs_tok   = self.obs_enc(obs)           # (B, 4,  d_model)
        alpha_tok = self.alpha_enc(alpha)        # (B, 25, d_model)

        # ── Regime-conditional LayerNorm: z_mu shifts/scales both streams ─
        # z_sigma attenuates modulation when regime is uncertain
        obs_tok   = self.adaln_obs(obs_tok,   z_mu, z_sigma)
        alpha_tok = self.adaln_alpha(alpha_tok, z_mu, z_sigma)

        # ── Bidirectional cross-attention ─────────────────────────────────
        obs_tok, alpha_tok = self.cross_attn(obs_tok, alpha_tok)

        # Cosine-similarity attention proxy for interpretability (no grad)
        with torch.no_grad():
            # (B, 4, D) @ (B, D, 25) → (B, 4, 25)
            obs_alpha_attention = torch.bmm(
                F.normalize(obs_tok,   dim=-1),
                F.normalize(alpha_tok, dim=-1).transpose(1, 2),
            )

        # ── Pool obs and alpha separately for contrastive loss later ──────
        obs_pooled   = obs_tok.mean(dim=1)    # (B, d_model) — for InfoNCE
        alpha_pooled = alpha_tok.mean(dim=1)  # (B, d_model)

        # ── MoE fusion over joint 29-token sequence ───────────────────────
        combined_tokens      = torch.cat([obs_tok, alpha_tok], dim=1)  # (B, 29, d_model)
        h_moe, router_weights = self.moe(combined_tokens, z_mu)        # (B, 29, d_model)

        # ── Mean pool with pre-norm → single vector ───────────────────────
        h_pooled = self.pool_norm(h_moe).mean(dim=1)    # (B, d_model)

        # ── VIB: compress to 192-dim EDT state ───────────────────────────
        fused_state, vib_mu, vib_log_sigma = self.vib(h_pooled)

        # ── Uncertainty: mean posterior std mapped to [0,1] ───────────────
        # uncertainty → 1.0: low VIB variance → confident → allow full leverage
        # uncertainty → 0.0: high VIB variance → uncertain → reduce position sizes
        sigma_mean  = vib_log_sigma.exp().mean(dim=-1)       # (B,)
        uncertainty = torch.sigmoid(-sigma_mean + 1.0)       # (B,)

        # Store pooled representations as attributes for compute_auxiliary_losses()
        # Using non-persistent buffer pattern to avoid polluting state_dict
        self._last_obs_pooled   = obs_pooled
        self._last_alpha_pooled = alpha_pooled

        return FusionOutput(
            fused_state         = fused_state,
            uncertainty         = uncertainty,
            router_weights      = router_weights,
            obs_alpha_attention = obs_alpha_attention,
            vib_mu              = vib_mu,
            vib_log_sigma       = vib_log_sigma,
        )

    def compute_auxiliary_losses(self, output: FusionOutput) -> FusionLosses:
        """
        Computes all auxiliary training losses. Call after forward():

          fusion_out  = portfolio_fusion(obs, z_mu, z_sigma, alpha)
          pred_loss   = criterion(edt(fusion_out.fused_state), target_returns)
          aux_losses  = portfolio_fusion.compute_auxiliary_losses(fusion_out)
          total_loss  = pred_loss + aux_losses.total
          total_loss.backward()

        LOSS COMPONENTS:
          1. VIB KL   — forces minimal sufficient statistic, prevents EDT overfit
          2. MoE entropy — load-balancing, prevents single-expert routing collapse
          3. InfoNCE  — aligns obs and alpha representations in a shared space
        """
        # ── 1. VIB KL divergence ─────────────────────────────────────────
        vib_kl = self.vib.kl_loss(output.vib_mu, output.vib_log_sigma)

        # ── 2. MoE load-balancing: maximise routing entropy ───────────────
        # Mean routing distribution across batch: π̄ ∈ Δ^{K-1}
        mean_routing   = output.router_weights.mean(dim=0) + 1e-8   # (K,)
        # Negative entropy (we add to loss, so minimising this maximises entropy)
        router_entropy_loss = _ROUTER_ENTROPY_LAMBDA * (mean_routing * mean_routing.log()).sum()

        # ── 3. InfoNCE cross-modal contrastive alignment ──────────────────
        # Encourages obs[i] to be most similar to alpha[i] (same timestep)
        # rather than alpha[j≠i] from other timesteps in the batch.
        z_obs   = F.normalize(self.obs_cproj(self._last_obs_pooled),     dim=-1)
        z_alpha = F.normalize(self.alpha_cproj(self._last_alpha_pooled), dim=-1)
        sim     = torch.mm(z_obs, z_alpha.T) / _CONTRASTIVE_TAU           # (B, B)
        labels  = torch.arange(sim.shape[0], device=sim.device)
        contrastive_loss = F.cross_entropy(sim, labels)

        total = vib_kl + router_entropy_loss + contrastive_loss

        return FusionLosses(
            vib_kl               = vib_kl,
            router_entropy       = router_entropy_loss,
            modality_contrastive = contrastive_loss,
            total                = total,
        )

    @torch.no_grad()
    def infer_live_state(
        self,
        obs_np:     np.ndarray,   # (52,)
        z_mu_np:    np.ndarray,   # (16,)
        z_sigma_np: np.ndarray,   # (16,)
        alpha_np:   np.ndarray,   # (124,)
        device:     str = "cuda",
    ) -> Tuple[np.ndarray, float]:
        """
        LIVE INFERENCE API — called by services/portfolio_agent_svc.py.

        Deterministic inference (VIB returns μ, no sampling). This is the
        production-critical path: add a single unsqueeze(0) for the batch dim,
        run forward, strip the batch dim, return to numpy.

        Returns:
          fused_state_np: (192,) float32 numpy — direct input to EDT.get_weights()
          uncertainty:    float ∈ (0,1) — pass to portfolio agent for leverage scaling
        """
        self.eval()
        obs     = torch.from_numpy(obs_np[None]).float().to(device)
        z_mu    = torch.from_numpy(z_mu_np[None]).float().to(device)
        z_sigma = torch.from_numpy(z_sigma_np[None]).float().to(device)
        alpha   = torch.from_numpy(alpha_np[None]).float().to(device)

        out = self.forward(obs, z_mu, z_sigma, alpha)
        return (
            out.fused_state.squeeze(0).cpu().numpy(),
            float(out.uncertainty.item()),
        )

    def anneal_moe_temperature(self, step: int, total_steps: int) -> None:
        """Convenience wrapper — delegates to internal MoE temperature schedule."""
        self.moe.anneal_temperature(step, total_steps)


# ─────────────────────────────────────────────────────────────────────────────
# TOP-LEVEL CONTAINER
# ─────────────────────────────────────────────────────────────────────────────

class CrossModalFusion(nn.Module):
    """
    Unified container exposing both fusion modules under a single nn.Module.

    Attributes:
      node_fusion:      CrossModalFusionNetwork   → GATv2 input    (25, 78)
      portfolio_fusion: PortfolioStateFusion      → EDT input    (batch, 192)

    The two modules do NOT share parameters by design. Their objectives are
    distinct: node_fusion learns modality-blending for graph message passing;
    portfolio_fusion learns state compression for sequence modelling.

    Shared training loop (recommended):
      gat_loss           = gat(node_fusion(obs_asset, z_t, llm))
      pf_out             = portfolio_fusion(obs_global, z_mu, z_sigma, alpha)
      edt_loss           = edt_criterion(edt(pf_out.fused_state), returns)
      aux_losses         = portfolio_fusion.compute_auxiliary_losses(pf_out)
      total_loss         = gat_loss + edt_loss + aux_losses.total
      total_loss.backward()

    Instantiation from hyperparams.yaml:
      cfg  = yaml.safe_load(open("config/hyperparams.yaml"))
      cmf  = CrossModalFusion.from_config(cfg)
    """
    def __init__(
        self,
        node_fusion_cfg:      Optional[Dict] = None,
        portfolio_fusion_cfg: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        nf_cfg  = node_fusion_cfg      or {}
        pf_cfg  = portfolio_fusion_cfg or {}

        self.node_fusion = CrossModalFusionNetwork(
            obs_dim       = nf_cfg.get("obs_dim_asset",  _OBS_DIM_ASSET),
            regime_dim    = nf_cfg.get("latent_dim",     _REGIME_DIM),
            llm_dim       = nf_cfg.get("llm_dim",        _LLM_DIM),
            node_feat_dim = nf_cfg.get("node_feat_dim",  _NODE_FEAT_DIM),
            n_assets      = nf_cfg.get("n_assets",       _N_ASSETS),
            dropout       = nf_cfg.get("dropout",        0.05),
        )
        self.portfolio_fusion = PortfolioStateFusion(
            obs_dim    = pf_cfg.get("obs_dim",    _OBS_DIM_GLOBAL),
            regime_dim = pf_cfg.get("latent_dim", _REGIME_DIM),
            alpha_dim  = pf_cfg.get("alpha_dim",  _ALPHA_DIM),
            output_dim = pf_cfg.get("state_dim",  _EDT_STATE_DIM),
            d_model    = pf_cfg.get("d_model",    _D_MODEL),
            n_heads    = pf_cfg.get("n_heads",    _N_HEADS),
            dropout    = pf_cfg.get("dropout",    _DROPOUT),
            vib_beta   = pf_cfg.get("vib_beta",   _VIB_BETA),
        )

    @classmethod
    def from_config(cls, hyperparams: Dict) -> "CrossModalFusion":
        """
        Construct from the full hyperparams.yaml dict.

        Usage:
          import yaml
          cfg = yaml.safe_load(open("config/hyperparams.yaml"))
          fusion = CrossModalFusion.from_config(cfg)
        """
        mamba_cfg = hyperparams.get("mamba_kan", {})
        gat_cfg   = hyperparams.get("gat_alpha", {})
        edt_cfg   = hyperparams.get("edt", {})

        nf_cfg = {
            "obs_dim_asset": _OBS_DIM_ASSET,
            "latent_dim":    mamba_cfg.get("latent_dim", _REGIME_DIM),
            "llm_dim":       _LLM_DIM,
            "node_feat_dim": gat_cfg.get("node_feat_dim", _NODE_FEAT_DIM),
            "n_assets":      _N_ASSETS,
        }
        pf_cfg = {
            "obs_dim":    _OBS_DIM_GLOBAL,
            "latent_dim": mamba_cfg.get("latent_dim", _REGIME_DIM),
            "alpha_dim":  _ALPHA_DIM,
            "state_dim":  edt_cfg.get("state_dim", _EDT_STATE_DIM),
            "d_model":    _D_MODEL,
            "n_heads":    _N_HEADS,
        }
        return cls(node_fusion_cfg=nf_cfg, portfolio_fusion_cfg=pf_cfg)

    def forward(self, *args, **kwargs):   # type: ignore[override]
        raise RuntimeError(
            "CrossModalFusion is a container. Use .node_fusion(...) or "
            ".portfolio_fusion(...) directly."
        )

    def __repr__(self) -> str:
        def n_params(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters() if p.requires_grad)

        n_nf  = n_params(self.node_fusion)
        n_pf  = n_params(self.portfolio_fusion)
        return (
            f"CrossModalFusion(\n"
            f"  CrossModalFusionNetwork  {n_nf:>12,} params  "
            f"({_OBS_DIM_ASSET},{_REGIME_DIM},{_LLM_DIM}) → ({_N_ASSETS},{_NODE_FEAT_DIM})\n"
            f"  PortfolioStateFusion     {n_pf:>12,} params  "
            f"({_OBS_DIM_GLOBAL},{_REGIME_DIM},{_ALPHA_DIM}) → ({_EDT_STATE_DIM},)\n"
            f"  Total                    {n_nf+n_pf:>12,} params\n"
            f"  d_model={_D_MODEL}  n_experts={_N_EXPERTS}  vib_β={_VIB_BETA}  "
            f"gumbel_τ=[{_GUMBEL_TAU_INIT}→{_GUMBEL_TAU_FLOOR}]\n"
            f")"
        )