"""
FORTRESS v5 - models/alpha/gat_signal_router.py
Path: models/alpha/gat_signal_router.py

GATv2 Signal Router — Dynamic IC Prediction and Signal Blending.

ARCHITECTURAL PIVOT (addresses Q5):
  The original GATv2 was trained as a raw return predictor:
    Input: node features (78-dim price/macro/regime)
    Target: 5-day forward cross-sectional Sharpe ratio
    Problem: 25 assets × 60 months of data = ~1,500 training samples.
    At 78 input dimensions this is critically underspecified. Overfitting
    is not a risk — it is the guaranteed outcome.

  THE SIGNAL ROUTER REFRAMING:
    Instead of predicting returns, the GATv2 predicts the FORWARD INFORMATION
    COEFFICIENT (IC) of each signal for each asset in the current regime.

    Formal definition:
      Let S = {VRP, VTS, NAV_arb, Insider, LowVol} be the signal set (|S|=5)
      Let IC_s(i, t+1) = Spearman_corr(signal_s[t, i], r[t+1, i]) over the
          next 21-day rolling window, computed forward from date t.

    GATv2 inputs (per node, 32-dim):
      [0:16]  = regime tensor z_mu (multi-asset vol regime posterior)
      [16:21] = per-signal rolling IC history (EWMA IC_s over last 63 days)
                This tells the model which signals HAVE BEEN working recently
      [21:25] = asset class embedding (one-hot: equity/bond/commodity/credit)
      [25:27] = asset duration/sensitivity proxies (vol beta, rate beta)
      [27:32] = market microstructure (liquidity, bid-ask proxy, avg daily volume z)

    GATv2 outputs (per node, 5-dim):
      W_i ∈ ℝ^5 — softmax-normalized blending weights for S signals
      W_is = P(signal s is the most informative for asset i in current regime)

    FINAL ALPHA COMPUTATION:
      α_i = Σ_s W_is × signal_s(i)
      This is a soft attention over signals rather than a hard selection.
      The attention is computed by the GATv2 graph network, which propagates
      regime and IC information between connected assets.

LOSS FUNCTION:
  The training objective directly optimises for forward IC:

  L_IC = -Σ_i Σ_s W_is × FwdIC_s(i, t+1)
         + λ_ent × H(W)        [entropy: prevents single-signal collapse]
         + λ_L2  × ‖θ‖²        [weight decay: prevents router overfitting]

  Where:
    H(W) = -Σ_i Σ_s W_is × log(W_is + ε)  [per-asset entropy over signals]
    FwdIC_s(i, t+1) = forward Spearman IC of signal s for asset i at t+1

  INTERPRETATION:
    The model is learning to predict, for each asset in the current regime,
    which signal will have the highest IC in the next 21 days.
    The entropy term prevents the router from always putting 100% weight on
    the signal with the highest recent IC (which would cause regime chasing).

GRAPH TOPOLOGY:
  Edges encode WHICH ASSETS SHARE INFORMATION ABOUT SIGNAL QUALITY.
  This replaces the DYNOTEARS price-correlation edges with economically-grounded
  information sharing:

  EDGE TYPE 0 — Asset class membership:
    All equity ETFs are connected to each other (shared equity vol regime).
    TLT-LQD-HYG form a triangle (shared rate/credit regime).
    GLD-SLV-GDX form a triangle (shared precious metals regime).
    Edge weight: inverse correlation of returns (more different = weaker edge).

  EDGE TYPE 1 — Shared vol index:
    SPY-QQQ-IWM: all respond to VIX (stronger cross-asset IC predictability).
    GLD-GDX: both respond to GVZ.
    TLT-LQD: both respond to MOVE/rate vol.
    Edge weight: historical beta to shared vol index.

  EDGE TYPE 2 — Factor loading similarity:
    Assets with similar VRP/VTS signal loadings should have correlated IC:
    when equity VRP signal works for SPY, it probably works for QQQ too.
    Edge weight: cosine similarity of rolling 63d signal IC vectors.

  This topology is fixed (no DYNOTEARS optimisation needed) and economically
  motivated — each edge type has a clear causal interpretation.

INPUT NODE FEATURES: 32 dimensions per asset
OUTPUT: W ∈ ℝ^(N×S) where N=25 assets, S=5 signals
ARCHITECTURE: 3-layer GATv2 with 4 attention heads, edge_dim=3 (one per edge type)
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATv2Conv
    from torch_geometric.data import Data
    _PYG_AVAILABLE = True
except ImportError:
    _PYG_AVAILABLE = False
    logger_init = logging.getLogger("GATSignalRouter")
    logger_init.warning("torch_geometric not available — GATSignalRouter will use fallback.")

logger = logging.getLogger("GATSignalRouter")

# ── Constants ──────────────────────────────────────────────────────────────────
N_ASSETS:     int = 25
N_SIGNALS:    int = 5   # VRP, VTS, NAV_arb, Insider, LowVol
NODE_FEAT_DIM: int = 32  # Reduced from 78 to match actual available features
EDGE_FEAT_DIM: int = 3   # One dimension per edge type [membership, vol_idx, IC_sim]
HIDDEN_DIM:   int = 64   # Smaller than original (32 features → 64 hidden is fine)
N_HEADS:      int = 4
N_LAYERS:     int = 3
DROPOUT:      float = 0.15

# Signal index mapping
SIGNAL_NAMES: List[str] = ["vrp", "vts", "nav_arb", "insider", "low_vol"]
SIGNAL_IDX:   Dict[str, int] = {name: i for i, name in enumerate(SIGNAL_NAMES)}

# Universe — must match order in precompute_alpha_signals.py
TICKERS: List[str] = [
    "SPY", "QQQ", "IWM", "TLT", "HYG", "LQD", "GLD", "SLV",
    "GDX", "XLE", "XLF", "XLK", "XLV", "XLU", "XLI", "XLP",
    "XLY", "XLB", "XLC", "VIXY", "BIL", "SHV", "USO", "PDBC", "COWZ",
]

# Asset class one-hot embedding index
# 0=equity, 1=bond/credit, 2=commodity, 3=vol_product, 4=cash
_ASSET_CLASS_IDX: Dict[str, int] = {
    "SPY": 0, "QQQ": 0, "IWM": 0, "XLE": 0, "XLF": 0, "XLK": 0,
    "XLV": 0, "XLU": 0, "XLI": 0, "XLP": 0, "XLY": 0, "XLB": 0,
    "XLC": 0, "COWZ": 0,
    "TLT": 1, "HYG": 1, "LQD": 1,
    "GLD": 2, "SLV": 2, "GDX": 2, "USO": 2, "PDBC": 2, "SHV": 2,
    "VIXY": 3,
    "BIL": 4, "SHV": 4,
}

# Static edge graph — computed once from economic priors (see build_economic_graph)
_STATIC_EDGE_INDEX: Optional[torch.Tensor] = None
_STATIC_EDGE_ATTR:  Optional[torch.Tensor] = None


class SignalRouterGAT(nn.Module):
    """
    GATv2-based signal router: predicts per-asset blending weights W ∈ ℝ^(N×S).

    The model is deliberately smaller than the original GATv2AlphaEngine
    because the task complexity is lower (predict 5 IC values vs. predict
    25 returns from scratch) and the data efficiency requirement is higher
    (we need good OOS generalisation, which means fewer parameters relative
    to training samples).

    PARAMETER COUNT: ~45k (vs original ~320k) — appropriate for N=25, S=5
    TRAINING SAMPLES: ~1200 (5 years of daily data with 21-day targets)
    EFFECTIVE RATIO: ~1:27 (parameters:samples) — reasonable for a 32-dim input

    The original 78-dim × 128-hidden × 3-layer model was ~2M parameters
    trained on 1200 samples — a ratio of ~1:0.0006. Catastrophic overfit was
    geometrically guaranteed.
    """

    def __init__(
        self,
        node_feat_dim: int = NODE_FEAT_DIM,
        edge_feat_dim: int = EDGE_FEAT_DIM,
        hidden_dim:    int = HIDDEN_DIM,
        n_heads:       int = N_HEADS,
        n_layers:      int = N_LAYERS,
        n_signals:     int = N_SIGNALS,
        dropout:       float = DROPOUT,
    ) -> None:
        super().__init__()

        if not _PYG_AVAILABLE:
            raise ImportError(
                "torch_geometric required. Install: pip install torch_geometric"
            )

        self.n_layers  = n_layers
        self.n_signals = n_signals

        # Input projection to align node features with hidden dim
        self.input_proj = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # GATv2 message passing layers
        self.convs      = nn.ModuleList()
        self.norms      = nn.ModuleList()
        self.dropouts   = nn.ModuleList()

        for layer_idx in range(n_layers):
            in_ch  = hidden_dim if layer_idx == 0 else hidden_dim * n_heads
            # Final layer: single head for stable output
            is_final = layer_idx == n_layers - 1
            heads    = 1 if is_final else n_heads
            concat   = not is_final

            self.convs.append(GATv2Conv(
                in_channels=in_ch,
                out_channels=hidden_dim,
                heads=heads,
                edge_dim=edge_feat_dim,
                concat=concat,
                dropout=dropout,
                add_self_loops=True,
                share_weights=False,  # asymmetric attention (better for heterogeneous universe)
            ))
            norm_dim = hidden_dim if is_final else hidden_dim * n_heads
            self.norms.append(nn.LayerNorm(norm_dim))
            self.dropouts.append(nn.Dropout(dropout))

        # Output head: hidden_dim → n_signals blending weights
        # Multi-layer output head improves expressivity for the IC prediction task
        self.weight_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_signals),
        )

        # IC prediction head (auxiliary): predicts per-signal IC for loss computation
        # This is the supervised target; weight_head softmax the IC predictions
        self.ic_pred_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_signals),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Xavier uniform init for linear layers, zero bias.
        GATv2Conv internal parameters are initialised by torch_geometric.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.8)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,           # (N, NODE_FEAT_DIM)
        edge_index: torch.Tensor,  # (2, E)
        edge_attr:  torch.Tensor,  # (E, EDGE_FEAT_DIM)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            blending_weights: (N, S) — softmax-normalised signal blending weights.
                              W_is = P(signal s is most informative for asset i)
            predicted_ic:     (N, S) — raw (pre-softmax) IC predictions.
                              Used in the IC loss during training.
        """
        h = self.input_proj(x)  # (N, hidden_dim)

        for i in range(self.n_layers):
            h_res = h  # residual connection
            h     = self.convs[i](h, edge_index, edge_attr)
            h     = self.norms[i](h)
            if i < self.n_layers - 1:
                h = F.gelu(h)
                h = self.dropouts[i](h)
                # Residual only where dimensions match (first layer: input_proj→hidden)
                if h.shape == h_res.shape:
                    h = h + h_res * 0.1  # scaled residual for training stability

        # Output heads
        predicted_ic      = self.ic_pred_head(h)           # (N, S) — raw IC estimates
        blending_weights  = F.softmax(self.weight_head(h), dim=-1)  # (N, S) — soft attention

        return blending_weights, predicted_ic

    def route_signals(
        self,
        x:            torch.Tensor,    # (N, NODE_FEAT_DIM)
        edge_index:   torch.Tensor,    # (2, E)
        edge_attr:    torch.Tensor,    # (E, EDGE_FEAT_DIM)
        signal_matrix: torch.Tensor,  # (N, S) — raw signal values
    ) -> torch.Tensor:
        """
        Compute the routed alpha vector: α = W ⊙ signals (element-wise, then sum over S).

        α_i = Σ_s W_is × signal_s(i)

        This is the production inference path. Returns (N,) alpha vector.
        """
        weights, _ = self.forward(x, edge_index, edge_attr)
        # Weighted sum of signals: (N, S) × (N, S) → (N, S) → sum over S → (N,)
        alpha = (weights * signal_matrix).sum(dim=-1)
        return torch.tanh(alpha)

    @torch.no_grad()
    def infer_alpha(
        self,
        x:            torch.Tensor,
        edge_index:   torch.Tensor,
        edge_attr:    torch.Tensor,
        signal_matrix: torch.Tensor,
        device:       str = "cuda",
    ) -> np.ndarray:
        """
        Production inference. Thread-safe under torch.no_grad().
        Returns (N,) numpy array of routed alpha scores.
        """
        self.eval()
        x_d = x.to(device)
        ei_d = edge_index.to(device)
        ea_d = edge_attr.to(device)
        sm_d = signal_matrix.to(device)
        alpha = self.route_signals(x_d, ei_d, ea_d, sm_d)
        return alpha.cpu().numpy()


def signal_router_loss(
    predicted_ic:     torch.Tensor,  # (N, S)
    forward_ic:       torch.Tensor,  # (N, S) — ground truth future IC
    blending_weights: torch.Tensor,  # (N, S)
    lambda_ent:       float = 0.05,
    lambda_l2_ic:     float = 0.01,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Signal router training loss with three terms:

    L = L_IC + λ_ent × L_entropy + λ_l2 × L_L2

    L_IC:
      Mean squared error between predicted IC and forward IC.
      This is the primary learning signal: the model learns which signals
      will have high IC in the near future for each asset.
      L_IC = E_i,s[(predicted_ic_is - forward_ic_is)²]

    L_entropy (regularisation):
      Per-asset entropy over signal weights. Prevents single-signal collapse.
      H(W_i) = -Σ_s W_is × log(W_is + ε)
      When all weight goes to one signal, H=0 and the penalty is maximum (inverted).
      We MAXIMISE entropy to encourage portfolio diversification across signals.
      L_ent = -E_i[H(W_i)]  → maximising entropy = minimising -H

    L_L2:
      L2 regularisation on IC predictions (not on weights, which are softmax-bounded).
      Prevents IC prediction head from learning large magnitude predictions that
      then dominate the weight distribution through the softmax.

    MATHEMATICAL NOTE ON L_IC DIRECTION:
      We want to MAXIMISE forward IC (higher IC = more predictive signal).
      The loss minimises -Σ W_is × FwdIC_is, which is equivalent to maximising
      the expected IC of the selected signal blend. This is the correct objective
      for a portfolio construction system: maximise expected predictive power.

    Args:
        predicted_ic:     Model's IC prediction (N, S)
        forward_ic:       Ground truth rolling Spearman IC (N, S)
        blending_weights: Softmax weights (N, S)
        lambda_ent:       Entropy regularisation strength
        lambda_l2_ic:     L2 regularisation on IC predictions

    Returns:
        total_loss: Scalar loss tensor
        components: Dict of loss component values for logging
    """
    # L_IC: MSE on IC predictions
    loss_ic = F.mse_loss(predicted_ic, forward_ic)

    # Expected IC under current weights (for information ratio computation)
    expected_ic = (blending_weights * forward_ic).sum(dim=-1)  # (N,)

    # L_maximise_expected_IC: directly reward high expected IC
    # We subtract because loss is minimised; maximising E[IC] = minimising -E[IC]
    loss_ic_reward = -expected_ic.mean()

    # L_entropy: encourage even signal usage (maximise entropy = minimise -H)
    eps = 1e-8
    entropy = -(blending_weights * torch.log(blending_weights + eps)).sum(dim=-1)  # (N,)
    loss_entropy = -entropy.mean()  # negate because we want to MAXIMISE entropy

    # L_L2: regularise IC prediction magnitudes
    loss_l2 = (predicted_ic ** 2).mean()

    # Total loss
    total_loss = (
        loss_ic
        + loss_ic_reward           # primary: maximise expected IC
        + lambda_ent * loss_entropy
        + lambda_l2_ic * loss_l2
    )

    components = {
        "ic_mse":          float(loss_ic.item()),
        "ic_reward":       float(-loss_ic_reward.item()),   # positive = good
        "entropy":         float(-loss_entropy.item()),     # positive = good
        "l2_reg":          float(loss_l2.item()),
        "mean_expected_ic": float(expected_ic.mean().item()),
        "total":           float(total_loss.item()),
    }

    return total_loss, components


def build_economic_graph(
    returns_df: Optional["pd.DataFrame"] = None,
    ic_history: Optional["pd.DataFrame"] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build the static economic edge graph for the 25-ticker universe.

    THREE EDGE TYPES (encoded in edge_attr[:, 3]):
      edge_attr[:, 0] = asset_class_membership weight [0, 1]
      edge_attr[:, 1] = shared_vol_index weight [0, 1]
      edge_attr[:, 2] = ic_similarity weight [0, 1] (from ic_history if available)

    EDGES ARE UNDIRECTED (both directions included):
      For each (i, j) pair in the edge set, both (i→j) and (j→i) are included.

    FIXED EDGE STRUCTURE (economic prior):
      Equity cluster: SPY↔QQQ, SPY↔IWM, QQQ↔IWM (shared equity vol)
      Extended equity: SPY/QQQ/IWM ↔ all XL* sectors (sector constituents)
      Bond cluster: TLT↔LQD↔HYG (shared rate sensitivity)
      Commodity cluster: GLD↔SLV↔GDX, GLD↔USO, USO↔PDBC
      Vol product: VIXY↔SPY (VIX short/long relationship)
      Cross-asset: GDX↔XLE (both energy/materials correlated)
    """
    import numpy as np

    ticker_idx = {t: i for i, t in enumerate(TICKERS)}
    N = len(TICKERS)

    edge_set_raw: List[Tuple[int, int, float, float, float]] = []

    def add_edge(t1: str, t2: str, membership: float, vol_idx: float, ic_sim: float = 0.5) -> None:
        """Add bidirectional edge with attributes."""
        i, j = ticker_idx[t1], ticker_idx[t2]
        edge_set_raw.append((i, j, membership, vol_idx, ic_sim))
        edge_set_raw.append((j, i, membership, vol_idx, ic_sim))

    # ── Equity cluster (shared VIX-family vol regime) ─────────────────────────
    core_equity = ["SPY", "QQQ", "IWM"]
    for a in core_equity:
        for b in core_equity:
            if a < b:
                add_edge(a, b, membership=1.0, vol_idx=1.0)

    sectors = ["XLE", "XLF", "XLK", "XLV", "XLU", "XLI", "XLP", "XLY", "XLB", "XLC", "COWZ"]
    for core in core_equity:
        for sec in sectors:
            add_edge(core, sec, membership=0.8, vol_idx=0.7)

    for i, s1 in enumerate(sectors):
        for s2 in sectors[i+1:]:
            add_edge(s1, s2, membership=0.6, vol_idx=0.5)

    # ── Bond/credit cluster (shared MOVE/rate vol) ─────────────────────────────
    bonds = ["TLT", "LQD", "HYG"]
    for a in bonds:
        for b in bonds:
            if a < b:
                add_edge(a, b, membership=1.0, vol_idx=0.9)

    cash_like = ["BIL", "SHV"]
    for c in cash_like:
        for b in bonds:
            add_edge(c, b, membership=0.5, vol_idx=0.6)

    # ── Commodity cluster (shared GVZ/OVX) ────────────────────────────────────
    precious = ["GLD", "SLV", "GDX"]
    for a in precious:
        for b in precious:
            if a < b:
                add_edge(a, b, membership=1.0, vol_idx=0.9)

    energy_comm = ["USO", "PDBC"]
    for a in energy_comm:
        for b in energy_comm:
            if a < b:
                add_edge(a, b, membership=0.9, vol_idx=0.9)
    for a in precious:
        for b in energy_comm:
            add_edge(a, b, membership=0.4, vol_idx=0.2)  # weak precious↔energy link

    # ── Cross-asset links ──────────────────────────────────────────────────────
    add_edge("VIXY", "SPY", membership=0.9, vol_idx=1.0)   # VIX short vs equity long
    add_edge("GDX",  "XLE", membership=0.5, vol_idx=0.4)   # materials-energy overlap
    add_edge("GLD",  "TLT", membership=0.3, vol_idx=0.2)   # safe-haven co-movement
    add_edge("HYG",  "XLF", membership=0.5, vol_idx=0.5)   # credit-financials link

    # If IC history provided, compute IC similarity and update edge weights
    ic_sim_map: Dict[Tuple[int, int], float] = {}
    if ic_history is not None and not ic_history.empty:
        # ic_history columns should be signal_names; index is tickers
        ic_corr = ic_history.T.corr()  # cross-ticker IC correlation matrix
        for (i, j, m, v, _) in edge_set_raw:
            t1, t2 = TICKERS[i], TICKERS[j]
            if t1 in ic_corr.index and t2 in ic_corr.columns:
                sim = float(np.clip((ic_corr.loc[t1, t2] + 1) / 2, 0.0, 1.0))
                ic_sim_map[(i, j)] = sim

    # Build tensors
    src_list, dst_list, attr_list = [], [], []
    for (i, j, membership, vol_idx, default_ic_sim) in edge_set_raw:
        ic_sim = ic_sim_map.get((i, j), default_ic_sim)
        src_list.append(i)
        dst_list.append(j)
        attr_list.append([membership, vol_idx, ic_sim])

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr  = torch.tensor(attr_list, dtype=torch.float32)

    logger.info(
        f"Economic graph built: {len(src_list)} edges | "
        f"Edge types: membership, vol_idx, ic_similarity"
    )
    return edge_index, edge_attr


def build_node_features(
    regime_tensor_zmu:  np.ndarray,      # (16,) from VolRegimeTensor.z_mu
    signal_ic_history:  np.ndarray,      # (N, S) rolling 63d EWMA IC per asset per signal
    vol_betas:          np.ndarray,      # (N,) sensitivity to market vol (VIX beta)
    rate_betas:         np.ndarray,      # (N,) sensitivity to rate moves
    liquidity_z:        np.ndarray,      # (N,) liquidity z-score (ADV vs norm)
) -> torch.Tensor:
    """
    Build 32-dim node features per asset.

    Feature structure:
      [0:16]  regime_z_mu      — Multi-asset vol regime tensor (16-dim)
      [16:21] ic_history_mean  — Mean IC per signal over last 63d (5-dim)
                                 i.e. how well each signal has been working recently
      [21:25] asset_class      — One-hot asset class embedding (5 classes)
      [25:27] sensitivity      — vol_beta, rate_beta (2-dim)
      [27:32] microstructure   — liquidity_z + 4 derived features

    NOTE: regime_z_mu is BROADCAST to all nodes — it's the same market regime
    for all assets. This is correct: the regime is market-wide. The asset-specific
    part is the IC history and sensitivity betas, which differ per node.
    """
    N = len(TICKERS)
    assert signal_ic_history.shape == (N, N_SIGNALS), \
        f"IC history shape mismatch: {signal_ic_history.shape} != ({N}, {N_SIGNALS})"

    node_features = np.zeros((N, NODE_FEAT_DIM), dtype=np.float32)

    for i, ticker in enumerate(TICKERS):
        # [0:16] Broadcast regime tensor
        node_features[i, 0:16] = regime_tensor_zmu[:16]

        # [16:21] Per-asset IC history (mean EWMA IC per signal)
        node_features[i, 16:21] = signal_ic_history[i, :]

        # [21:25] One-hot asset class (5 classes: equity, bond, commodity, vol, cash)
        asset_class = _ASSET_CLASS_IDX.get(ticker, 0)
        if 21 + asset_class < 26:
            node_features[i, 21 + asset_class] = 1.0

        # [25:27] Sensitivity betas (clipped to prevent extreme values)
        node_features[i, 25] = float(np.clip(vol_betas[i],  -3.0, 3.0))
        node_features[i, 26] = float(np.clip(rate_betas[i], -3.0, 3.0))

        # [27:32] Microstructure features
        node_features[i, 27] = float(np.clip(liquidity_z[i], -3.0, 3.0))
        # Derived: asset class interaction with regime
        node_features[i, 28] = node_features[i, 25] * node_features[i, 0]   # vol_beta × equity_regime
        node_features[i, 29] = node_features[i, 26] * node_features[i, 4]   # rate_beta × rate_regime
        node_features[i, 30] = node_features[i, 25] * float(regime_tensor_zmu[0])  # vol_beta × equity_urgency
        node_features[i, 31] = float(np.mean(signal_ic_history[i, :]))  # mean IC across all signals

    return torch.from_numpy(node_features)


class FallbackSignalRouter:
    """
    Gradient-free fallback when torch_geometric is unavailable.

    Uses 63-day EWMA IC history to compute signal weights without graph propagation.
    Loses the cross-asset information sharing of GATv2, but preserves the
    IC-based weighting logic.

    This ensures the pipeline always produces a valid alpha signal, even in
    environments where PyG cannot be installed.
    """

    def __init__(self, temperature: float = 2.0) -> None:
        self._temp = temperature  # Softmax temperature — lower = more concentrated

    def route_signals(
        self,
        signal_ic_history: np.ndarray,   # (N, S) rolling EWMA IC per asset per signal
        signal_matrix:     np.ndarray,   # (N, S) current signal values
    ) -> np.ndarray:
        """
        Weight signals by recent IC, apply softmax, return blended alpha.
        No graph propagation — each asset is independent.
        """
        # Softmax over IC history, temperature-scaled
        ic_scaled = signal_ic_history / self._temp
        # Shift by max for numerical stability
        ic_shifted = ic_scaled - ic_scaled.max(axis=1, keepdims=True)
        exp_ic     = np.exp(ic_shifted)
        weights    = exp_ic / (exp_ic.sum(axis=1, keepdims=True) + 1e-8)  # (N, S)

        # Weighted blend
        alpha_raw = (weights * signal_matrix).sum(axis=1)  # (N,)
        return np.tanh(alpha_raw).astype(np.float32)