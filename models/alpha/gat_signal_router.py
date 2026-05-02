"""
FORTRESS v5 - models/alpha/gat_signal_router.py
Path: models/alpha/gat_signal_router.py

GATv2 Signal Router — 5-Signal Orthogonal Stack (MOM, LOW_VOL, CONC_LEAD, EIS, BV_VPIN).
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
    _PYG_AVAILABLE = True
except ImportError:
    _PYG_AVAILABLE = False

logger = logging.getLogger("GATSignalRouter")

# ── Constants ──────────────────────────────────────────────────────────────────
N_SIGNALS:    int = 5    # mom, low_vol, conc_lead, eis, bv_vpin
NODE_FEAT_DIM: int = 16  # 5(IC) + 5(Class) + 2(Beta) + 4(Stats)
GLOBAL_FEAT_DIM: int = 16 
EDGE_FEAT_DIM: int = 3   
HIDDEN_DIM:   int = 64
N_HEADS:      int = 4
N_LAYERS:     int = 3
DROPOUT:      float = 0.15
K_NEIGHBORS:  int = 5    

SIGNAL_NAMES: List[str] = ["mom", "low_vol", "conc_lead", "night_effect", "pca_statarb"]

import yaml
with open("config/universe.yaml", "r") as f:
    _univ = yaml.safe_load(f)
TICKERS = [a["ticker"] for a in _univ["assets"]]
N_ASSETS = len(TICKERS)

_ASSET_CLASS_IDX: Dict[str, int] = {
    "TLT": 1, "HYG": 1, "LQD": 1,
    "GLD": 2, "SLV": 2, "GDX": 2, "USO": 2, "PDBC": 2, "SHV": 2,
    "VIXY": 3,
    "BIL": 4, "SHV": 4,
}
for t in TICKERS:
    if t not in _ASSET_CLASS_IDX:
        _ASSET_CLASS_IDX[t] = 0

class SignalRouterGAT(nn.Module):
    def __init__(
        self,
        node_feat_dim: int = NODE_FEAT_DIM,
        global_feat_dim: int = GLOBAL_FEAT_DIM,
        edge_feat_dim: int = EDGE_FEAT_DIM,
        hidden_dim:    int = HIDDEN_DIM,
        n_heads:       int = N_HEADS,
        n_layers:      int = N_LAYERS,
        n_signals:     int = N_SIGNALS,
        dropout:       float = DROPOUT,
    ) -> None:
        super().__init__()

        self.n_layers  = n_layers
        self.n_signals = n_signals

        self.node_proj = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
        )
        
        self.global_proj = nn.Sequential(
            nn.Linear(global_feat_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
        )

        self.convs      = nn.ModuleList()
        self.norms      = nn.ModuleList()
        self.dropouts   = nn.ModuleList()

        for layer_idx in range(n_layers):
            in_ch  = hidden_dim if layer_idx == 0 else hidden_dim * n_heads
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
                share_weights=False, 
            ))
            norm_dim = hidden_dim if is_final else hidden_dim * n_heads
            self.norms.append(nn.LayerNorm(norm_dim))
            self.dropouts.append(nn.Dropout(dropout))

        self.weight_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_signals),
        )

        self.ic_pred_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_signals),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.8)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,           
        g: torch.Tensor,           
        edge_index: torch.Tensor,  
        edge_attr:  torch.Tensor,  
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        h_node = self.node_proj(x)       
        h_glob = self.global_proj(g)     
        
        h_glob_expanded = h_glob.unsqueeze(0).expand(x.size(0), -1) 
        h = torch.cat([h_node, h_glob_expanded], dim=1) 

        for i in range(self.n_layers):
            h_res = h
            h     = self.convs[i](h, edge_index, edge_attr)
            h     = self.norms[i](h)
            if i < self.n_layers - 1:
                h = self.dropouts[i](F.gelu(h))
                if h.shape == h_res.shape:
                    h = h + h_res * 0.1 

        predicted_ic      = self.ic_pred_head(h)           
        blending_weights  = F.softmax(self.weight_head(h), dim=-1)  

        return blending_weights, predicted_ic

    def route_signals(
        self,
        x:            torch.Tensor,
        g:            torch.Tensor,
        edge_index:   torch.Tensor,    
        edge_attr:    torch.Tensor,    
        signal_matrix: torch.Tensor,  
    ) -> torch.Tensor:
        weights, _ = self.forward(x, g, edge_index, edge_attr)
        alpha = (weights * signal_matrix).sum(dim=-1)
        return torch.tanh(alpha)

    @torch.no_grad()
    def infer_alpha(
        self,
        x:            torch.Tensor,
        g:            torch.Tensor,
        edge_index:   torch.Tensor,
        edge_attr:    torch.Tensor,
        signal_matrix: torch.Tensor,
        device:       str = "cuda",
    ) -> np.ndarray:
        self.eval()
        x_d = x.to(device)
        g_d = g.to(device)
        ei_d = edge_index.to(device)
        ea_d = edge_attr.to(device)
        sm_d = signal_matrix.to(device)
        alpha = self.route_signals(x_d, g_d, ei_d, ea_d, sm_d)
        return alpha.cpu().numpy()


def signal_router_loss(
    predicted_ic:     torch.Tensor,  
    forward_ic:       torch.Tensor,  
    blending_weights: torch.Tensor,  
    lambda_ent:       float = 0.05,
    lambda_l2_ic:     float = 0.01,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    
    loss_ic = F.mse_loss(predicted_ic, forward_ic)
    expected_ic = (blending_weights * forward_ic).sum(dim=-1)  
    loss_ic_reward = -expected_ic.mean()

    eps = 1e-8
    entropy = -(blending_weights * torch.log(blending_weights + eps)).sum(dim=-1)  
    loss_entropy = -entropy.mean()  

    loss_l2 = (predicted_ic ** 2).mean()

    total_loss = loss_ic + loss_ic_reward + lambda_ent * loss_entropy + lambda_l2_ic * loss_l2

    components = {
        "ic_mse":          float(loss_ic.item()),
        "ic_reward":       float(-loss_ic_reward.item()),   
        "entropy":         float(-loss_entropy.item()),     
        "l2_reg":          float(loss_l2.item()),
        "mean_expected_ic": float(expected_ic.mean().item()),
        "total":           float(total_loss.item()),
    }
    return total_loss, components


def build_economic_graph(
    returns_df: Optional["pd.DataFrame"] = None,
    ic_history: Optional["pd.DataFrame"] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    import numpy as np

    N = len(TICKERS)
    edge_set_raw: List[Tuple[int, int, float, float, float]] = []
    
    if ic_history is not None and not ic_history.empty:
        aligned_df = ic_history.reindex(TICKERS).fillna(0.0)
        sim_matrix = aligned_df.T.corr().fillna(0.0).values
    elif returns_df is not None and not returns_df.empty:
        aligned_df = returns_df.reindex(columns=TICKERS).fillna(0.0)
        sim_matrix = aligned_df.corr().fillna(0.0).values
    else:
        sim_matrix = np.eye(N)

    structural_edges = [
        ("TLT", "LQD"), ("LQD", "HYG"), ("TLT", "HYG"),
        ("GLD", "SLV"), ("GLD", "GDX"), ("USO", "PDBC"),
        ("VIXY", "SPY")
    ]
    for t1, t2 in structural_edges:
        if t1 in TICKERS and t2 in TICKERS:
            i, j = TICKERS.index(t1), TICKERS.index(t2)
            edge_set_raw.extend([
                (i, j, 1.0, 0.9, float(sim_matrix[i, j])), 
                (j, i, 1.0, 0.9, float(sim_matrix[i, j]))
            ])

    for i in range(N):
        top_k_idx = np.argsort(sim_matrix[i])[-K_NEIGHBORS-1:-1]
        
        for j in top_k_idx:
            ic_sim = float(sim_matrix[i, j])
            membership = 1.0 if _ASSET_CLASS_IDX[TICKERS[i]] == _ASSET_CLASS_IDX[TICKERS[j]] else 0.0
            vol_idx = float(np.clip((ic_sim + 1.0) / 2.0, 0.0, 1.0))
            edge_set_raw.append((i, j, membership, vol_idx, ic_sim))

    unique_edges = {}
    for (i, j, m, v, s) in edge_set_raw:
        if (i, j) not in unique_edges:
            unique_edges[(i, j)] = (m, v, s)

    src_list, dst_list, attr_list = [], [], []
    for (i, j), (m, v, s) in unique_edges.items():
        src_list.append(i)
        dst_list.append(j)
        attr_list.append([m, v, s])

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr  = torch.tensor(attr_list, dtype=torch.float32)
    return edge_index, edge_attr


def build_node_features(
    signal_ic_history:  np.ndarray,      
    vol_betas:          np.ndarray,      
    rate_betas:         np.ndarray,      
    liquidity_z:        np.ndarray,     
) -> torch.Tensor:
    
    N = len(TICKERS)
    assert signal_ic_history.shape == (N, N_SIGNALS), \
        f"IC history shape mismatch: {signal_ic_history.shape} != ({N}, {N_SIGNALS})"

    node_features = np.zeros((N, NODE_FEAT_DIM), dtype=np.float32)

    for i, ticker in enumerate(TICKERS):
        # [0:5] Per-asset IC history (5 signals)
        node_features[i, 0:5] = signal_ic_history[i, :]

        # [5:10] One-hot asset class 
        asset_class = _ASSET_CLASS_IDX.get(ticker, 0)
        node_features[i, 5 + asset_class] = 1.0

        # [10:12] Sensitivity betas
        node_features[i, 10] = float(np.clip(vol_betas[i],  -3.0, 3.0))
        node_features[i, 11] = float(np.clip(rate_betas[i], -3.0, 3.0))

        # [12:16] Microstructure and derived stats
        node_features[i, 12] = float(np.clip(liquidity_z[i], -3.0, 3.0))
        node_features[i, 13] = float(np.mean(signal_ic_history[i, :])) 
        node_features[i, 14] = float(np.std(signal_ic_history[i, :]))
        node_features[i, 15] = float(np.max(signal_ic_history[i, :]))

    return torch.from_numpy(node_features)


class FallbackSignalRouter:
    def __init__(self, temperature: float = 2.0) -> None:
        self._temp = temperature 

    def route_signals(self, signal_ic_history: np.ndarray, signal_matrix: np.ndarray) -> np.ndarray:
        ic_scaled = signal_ic_history / self._temp
        ic_shifted = ic_scaled - ic_scaled.max(axis=1, keepdims=True)
        exp_ic     = np.exp(ic_shifted)
        weights    = exp_ic / (exp_ic.sum(axis=1, keepdims=True) + 1e-8)  
        alpha_raw = (weights * signal_matrix).sum(axis=1)  
        return np.tanh(alpha_raw).astype(np.float32)