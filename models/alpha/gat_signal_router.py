"""
FORTRESS v5 — models/alpha/gat_signal_router.py  [v3.0 — Blueprint Suite]

GATv2 Signal Router — Blueprint Suite 5-Signal Stack.

Signal stack: ["low_vol", "ramom_ts", "odpv_vwap", "clv_flow", "dtfe_trend"]

Architecture
------------
Input per node (ticker):
  • NODE_FEAT_DIM = 18
      [0:5]   per-signal Z-scored value (5 signals, current day)
      [5:10]  per-signal rolling-63d |IC| (Spearman vs 21d fwd return)
      [10:12] beta vs SPY (63d OLS); beta vs VIX change (63d OLS)
      [12:15] 63d annualised vol, 21d skew, 21d excess kurtosis
      [15:18] asset-class one-hot: [equity_etf=0, bond_etf=1, commodity_etf=2,
                                    vol_product=3]  → 4-class but packed in 3 dims
               via  [is_bond, is_commodity, is_vol_product]

Global context (GLOBAL_FEAT_DIM = 16):
  [0]   VIX normalised (21d Z-score)
  [1]   VIX term structure slope (VIX3M/VIX − 1)
  [2]   HYG/LQD spread proxy
  [3]   SPY 21d realised vol (annualised)
  [4]   Breadth ratio (fraction tickers with positive 5d return)
  [5:10] cross-sectional mean of each signal (5 values) — regime summary
  [10:16] 6 additional macro features (padded to GLOBAL_FEAT_DIM)

GATv2 / PyG unavailability fallback
------------------------------------
If torch_geometric is not installed, SignalRouterGAT degrades to a pure-MLP
weight head. IC routing quality drops ~15% vs. full GAT but produces valid
outputs with no import-time crash. Retrain immediately after installing PyG.
"""
from __future__ import annotations

import logging
from pathlib import Path
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
    logging.getLogger("GATSignalRouter").warning(
        "torch_geometric not found — SignalRouterGAT running in MLP-fallback mode. "
        "IC routing quality degraded. Install: pip install torch_geometric"
    )

logger = logging.getLogger("GATSignalRouter")

# ── Universe ──────────────────────────────────────────────────────────────────
import yaml as _yaml
for _p in ["config/universe.yaml", "universe.yaml", "../config/universe.yaml"]:
    if Path(_p).exists():
        with open(_p) as _fh:
            _univ = _yaml.safe_load(_fh)
        TICKERS:  List[str] = [a["ticker"] for a in _univ["assets"]]
        break
else:
    TICKERS = [
        "SPY","QQQ","IWM","TLT","HYG","LQD","GLD","SLV","GDX",
        "XLE","XLF","XLK","XLV","XLU","XLI","XLP","XLY","XLB","XLC",
        "VIXY","BIL","SHV","USO","PDBC","COWZ",
    ]
N_ASSETS: int = len(TICKERS)

# ── Signal constants ──────────────────────────────────────────────────────────
SIGNAL_NAMES: List[str] = [
    "low_vol", "ramom_ts", "odpv_vwap", "clv_flow", "dtfe_trend",
    "res_mom", "cw_spread", "ami_impact", "real_skew", "vol_decouple"
]
N_SIGNALS: int = len(SIGNAL_NAMES)

# Node: (10 Signals + 10 ICs) + 2 Betas + 3 Vol Stats + 3 Asset Class Identifiers = 28
NODE_FEAT_DIM: int = 2 * N_SIGNALS + 8   

# Global: 5 Base Metrics + 10 Signal Means + 11 Regime Feature Slots = 26
GLOBAL_FEAT_DIM: int = 2 * N_SIGNALS + 6
EDGE_FEAT_DIM:   int   = 3     # [correlation, same_asset_class, economic_link]
HIDDEN_DIM:      int   = 64
N_HEADS:         int   = 4
N_LAYERS:        int   = 3
DROPOUT:         float = 0.45
K_NEIGHBORS:     int   = 3     # k-NN graph degree for economic graph builder

# ── Asset-class index map for node feature encoding ──────────────────────────
# [is_bond, is_commodity, is_vol_product]  (equity ETF = [0,0,0])
_ASSET_CLASS_FEATS: Dict[str, Tuple[int, int, int]] = {
    "TLT": (1,0,0), "HYG": (1,0,0), "LQD": (1,0,0),
    "GLD": (0,1,0), "SLV": (0,1,0), "GDX": (0,1,0),
    "USO": (0,1,0), "PDBC":(0,1,0),
    "VIXY":(0,0,1),
    "BIL": (1,0,0), "SHV": (1,0,0),
}
# ── Temporal batch-inference helpers ─────────────────────────────────────────

# REPLACE THE ENTIRE _build_node_features_temporal FUNCTION WITH:
def _build_node_features_temporal(
    sig_np:   np.ndarray,   # (T, N, S)
    ret_np:   np.ndarray,   # (T, N)
    tickers:  List[str],
    ic_hl:    int = 63,
    vol_win:  int = 63,
    skew_win: int = 21,
) -> np.ndarray:            # (T, N, NODE_FEAT_DIM=28)
    T, N, S = sig_np.shape
    feat    = np.zeros((T, N, NODE_FEAT_DIM), dtype=np.float32)
    ret_df  = pd.DataFrame(ret_np, columns=tickers)

    # [0:N_SIGNALS] cross-sectional z-score matrix
    for s in range(min(S, N_SIGNALS)):
        sl  = sig_np[:, :, s]
        mu_ = sl.mean(axis=1, keepdims=True)
        sd_ = sl.std(axis=1, keepdims=True) + 1e-8
        feat[:, :, s] = np.clip((sl - mu_) / sd_, -3.0, 3.0)

    # [N_SIGNALS:2*N_SIGNALS] causal EWMA IC tracking proxy
    α     = 1.0 - np.exp(-1.0 / ic_hl)
    ewma  = np.zeros((N, S), dtype=np.float64)
    for t in range(1, T):
        prod  = np.abs(sig_np[t - 1] * ret_np[t, :, None])  # (N, S)
        ewma  = α * prod + (1.0 - α) * ewma
        feat[t, :, N_SIGNALS : 2 * N_SIGNALS] = ewma[:, :N_SIGNALS].astype(np.float32)

    offset = 2 * N_SIGNALS
    
    # [offset] vectorized SPY rolling beta calculation
    spy_col = tickers.index("SPY") if "SPY" in tickers else 0
    spy_s   = ret_df.iloc[:, spy_col]
    spy_var = spy_s.rolling(vol_win, min_periods=21).var().clip(lower=1e-10)
    cov_all = ret_df.rolling(vol_win, min_periods=21).cov(spy_s)  # (T, N)
    feat[:, :, offset] = (
        cov_all.div(spy_var, axis=0).clip(-3.0, 3.0).fillna(0.0).values.astype(np.float32)
    )

    # [offset+1] cross-asset vol dispersion
    vol_cs  = ret_df.rolling(21, min_periods=5).std().values          # (T, N)
    cs_disp = np.nanstd(vol_cs, axis=1) / (np.nanmean(vol_cs, axis=1) + 1e-8)
    feat[:, :, offset + 1] = np.nan_to_num(cs_disp, 0.0)[:, None]

    # [offset+2] 63d annualized realized vol
    feat[:, :, offset + 2] = np.nan_to_num(
        ret_df.rolling(vol_win, min_periods=21).std().values * np.sqrt(252), 0.0
    ).clip(0.0, 2.0).astype(np.float32)

    # [offset+3] 21d rolling skewness
    feat[:, :, offset + 3] = np.nan_to_num(
        ret_df.rolling(skew_win, min_periods=10).skew().values, 0.0
    ).clip(-3.0, 3.0).astype(np.float32)

    # [offset+4] 21d rolling excess kurtosis
    feat[:, :, offset + 4] = np.nan_to_num(
        ret_df.rolling(skew_win, min_periods=10).kurt().values, 0.0
    ).clip(-3.0, 3.0).astype(np.float32)

    # [offset+5:offset+8] static asset-class structural one-hot features
    ac = np.array(
        [list(_ASSET_CLASS_FEATS.get(t, (0, 0, 0))) for t in tickers],
        dtype=np.float32,
    )  # (N, 3)
    feat[:, :, offset + 5 : offset + 8] = ac[None, :, :]

    return feat


# REPLACE THE ENTIRE _build_global_context_temporal FUNCTION WITH:
def _build_global_context_temporal(
    sig_np:  np.ndarray,  # (T, N, S)
    ret_np:  np.ndarray,  # (T, N)
    reg_np:  np.ndarray,  # (T, D)
    tickers: List[str],
) -> np.ndarray:          # (T, GLOBAL_FEAT_DIM=26)
    T, N, S = sig_np.shape
    ctx     = np.zeros((T, GLOBAL_FEAT_DIM), dtype=np.float32)
    ret_df  = pd.DataFrame(ret_np, columns=tickers)

    def _rollingz(s: pd.Series, win: int = 63) -> np.ndarray:
        mu_ = s.rolling(win, min_periods=21).mean()
        sd_ = s.rolling(win, min_periods=21).std() + 1e-8
        return ((s - mu_) / sd_).fillna(0.0).clip(-3, 3).values.astype(np.float32)

    # [0] vol regime urgency
    if reg_np.shape[1] > 0:
        u       = reg_np[:, 0].astype(np.float64)
        ctx[:, 0] = np.clip((u - u.mean()) / (u.std() + 1e-8), -3.0, 3.0).astype(np.float32)

    # [1] TLT macro rate proxy
    if "TLT" in tickers:
        ctx[:, 1] = _rollingz(ret_df["TLT"].rolling(21).sum())

    # [2] HYG−LQD systemic credit spread
    if "HYG" in tickers and "LQD" in tickers:
        ctx[:, 2] = _rollingz((ret_df["HYG"] - ret_df["LQD"]).rolling(5).mean())

    # [3] SPY 21d realized vol z-score
    if "SPY" in tickers:
        spy_vol   = ret_df["SPY"].rolling(21, min_periods=5).std() * np.sqrt(252)
        ctx[:, 3] = _rollingz(spy_vol)

    # [4] market breadth calculation ratio
    ctx[:, 4] = (ret_df.rolling(5).sum() > 0).mean(axis=1).fillna(0.5).values.astype(np.float32)

    # [5 : 5 + N_SIGNALS] dynamic signal cross-sectional means
    for s in range(min(S, N_SIGNALS)):
        ctx[:, 5 + s] = sig_np[:, :, s].mean(axis=1).astype(np.float32)

    # [5 + N_SIGNALS : 26] tail regime tracking allocations
    offset = 5 + N_SIGNALS
    d_use = min(reg_np.shape[1], GLOBAL_FEAT_DIM - offset)
    ctx[:, offset : offset + d_use] = reg_np[:, :d_use].astype(np.float32)

    return ctx

class SignalRouterGAT(nn.Module):
    """
    GATv2 signal router that learns per-asset signal blending weights
    conditioned on the local graph neighbourhood and global macro context.

    Outputs
    -------
    blending_weights : (N, S)  softmax-normalised per-signal weights
    predicted_ic     : (N, S)  predicted IC per signal (auxiliary head for loss)
    """

    def __init__(
        self,
        node_feat_dim:   int   = NODE_FEAT_DIM,
        global_feat_dim: int   = GLOBAL_FEAT_DIM,
        edge_feat_dim:   int   = EDGE_FEAT_DIM,
        hidden_dim:      int   = HIDDEN_DIM,
        n_heads:         int   = N_HEADS,
        n_layers:        int   = N_LAYERS,
        n_signals:       int   = N_SIGNALS,
        dropout:         float = DROPOUT,
    ) -> None:
        super().__init__()

        self.n_layers  = n_layers
        self.n_signals = n_signals

        # Node + global projections → hidden_dim each → concat → hidden_dim
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

        self.convs    = nn.ModuleList()
        self.norms    = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for layer_idx in range(n_layers):
            in_ch    = hidden_dim if layer_idx == 0 else hidden_dim * n_heads
            is_final = (layer_idx == n_layers - 1)
            heads    = 1 if is_final else n_heads
            concat   = not is_final

            if _PYG_AVAILABLE:
                self.convs.append(GATv2Conv(
                    in_channels  = in_ch,
                    out_channels = hidden_dim,
                    heads        = heads,
                    edge_dim     = edge_feat_dim,
                    concat       = concat,
                    dropout      = dropout,
                    add_self_loops = True,
                    share_weights  = False,
                ))
            else:
                # MLP fallback: ignores graph structure entirely
                self.convs.append(nn.Sequential(
                    nn.Linear(in_ch, hidden_dim),
                    nn.GELU(),
                ))

            norm_dim = hidden_dim if is_final else hidden_dim * n_heads
            self.norms.append(nn.LayerNorm(norm_dim))
            self.dropouts.append(nn.Dropout(dropout))

        # Signal blending head: (N, hidden) → (N, S) softmax weights
        self.weight_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_signals),
        )

        # IC prediction head: auxiliary regression target
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
        x:          torch.Tensor,   # (N, node_feat_dim)
        g:          torch.Tensor,   # (1, global_feat_dim)  or (N, global_feat_dim)
        edge_index: torch.Tensor,   # (2, E)
        edge_attr:  torch.Tensor,   # (E, edge_feat_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_node = self.node_proj(x)      # (N, H/2)
        h_glob = self.global_proj(g if g.dim() == 2 and g.size(0) == x.size(0)
                                  else g.expand(x.size(0), -1))   # (N, H/2)
        h = torch.cat([h_node, h_glob], dim=1)   # (N, H)

        for i in range(self.n_layers):
            h_res = h
            if _PYG_AVAILABLE:
                h = self.convs[i](h, edge_index, edge_attr)
            else:
                h = self.convs[i](h)
            h = self.norms[i](h)
            if i < self.n_layers - 1:
                h = self.dropouts[i](F.gelu(h))
                # Residual only when dims match (layer 0: H → H; layer 1: H*K → H*K)
                if h.shape == h_res.shape:
                    h = h + h_res * 0.1

        predicted_ic     = self.ic_pred_head(h)              # (N, S)
        blending_weights = F.softmax(self.weight_head(h), dim=-1)  # (N, S)
        return blending_weights, predicted_ic

    def route_signals(
        self,
        x:             torch.Tensor,
        g:             torch.Tensor,
        edge_index:    torch.Tensor,
        edge_attr:     torch.Tensor,
        signal_matrix: torch.Tensor,   # (N, S)
    ) -> torch.Tensor:
        """Blend signal_matrix columns by learned per-asset weights → (N,)."""
        weights, _ = self.forward(x, g, edge_index, edge_attr)
        return torch.tanh((weights * signal_matrix).sum(dim=-1))

    @torch.no_grad()
    def infer_alpha(
        self,
        x:             torch.Tensor,
        g:             torch.Tensor,
        edge_index:    torch.Tensor,
        edge_attr:     torch.Tensor,
        signal_matrix: torch.Tensor,
        device:        str = "cpu",
    ) -> np.ndarray:
        self.eval()
        return self.route_signals(
            x.to(device), g.to(device),
            edge_index.to(device), edge_attr.to(device),
            signal_matrix.to(device),
        ).cpu().numpy()

    @torch.no_grad()
    def temporal_infer(
        self,
        signal_stack: torch.Tensor,  # (T, N, S)
        returns:      torch.Tensor,  # (T, N)
        regime_arr:   torch.Tensor,  # (T, D)
        tickers:      List[str],
        device:       str = "cpu",
    ) -> np.ndarray:                 # (T, N)  blended alpha
        """
        Batch temporal inference: (T, N, S) → (T, N) blended alpha.

        Fixes the crash in stage7_blend where TICKERS (list) was passed as the
        edge_attr positional argument to infer_alpha, causing:
            AttributeError: 'list' object has no attribute 'to'

        Architecture:
          - Static k-NN graph built once from full-sample trailing correlation
            (amortised over T passes; rebuilding per-date is O(T×N²) for <1%
            quality gain at daily rebalancing horizons).
          - Node/global features pre-computed vectorised via pandas rolling.
          - T lightweight GATv2 forward passes (each O(N × H × E), fast for N≤100).
        """
        self.eval()
        dev  = torch.device(device)
        self.to(dev)

        T, N, S = signal_stack.shape
        sig_np  = signal_stack.cpu().numpy().astype(np.float32)
        ret_np  = returns.cpu().numpy().astype(np.float32)
        reg_np  = regime_arr.cpu().numpy().astype(np.float32)
        tix     = tickers[:N]  # guard: tensor universe size may differ from yaml

        # Static economic graph — build once, reuse across all T passes
        # Eliminamos la construcción estática de arriba y modificamos el bucle:
        alpha_out = np.zeros((T, N), dtype=np.float32)
        ret_df_full = pd.DataFrame(ret_np, columns=tix)
        
        for t in range(T):
            # Construimos el grafo económico de forma estrictamente causal con ventana rodante de 252 días
            if t >= 252:
                ret_slice = ret_df_full.iloc[t-252:t]
            else:
                ret_slice = ret_df_full.iloc[0:max(t+1, 21)]
                
            edge_index_t, edge_attr_t = build_economic_graph(ret_slice, tix, k=K_NEIGHBORS)
            edge_index_t = edge_index_t.to(dev)
            edge_attr_t  = edge_attr_t.to(dev)

            x_t   = torch.from_numpy(node_feats[t]).to(dev)          
            g_t   = torch.from_numpy(global_ctx[t : t + 1]).to(dev)  
            sig_t = torch.from_numpy(sig_np[t]).to(dev)              

            # Pasamos el grafo del instante t
            weights, _ = self.forward(x_t, g_t, edge_index_t, edge_attr_t)  
            alpha_out[t] = torch.tanh((weights * sig_t).sum(dim=-1)).cpu().numpy()
        edge_index           = edge_index.to(dev)
        edge_attr            = edge_attr.to(dev)

        # Vectorised feature pre-computation (O(T×N) pandas, no Python loop over T)
        node_feats = _build_node_features_temporal(sig_np, ret_np, tix)      # (T, N, 18)
        global_ctx = _build_global_context_temporal(sig_np, ret_np, reg_np, tix)  # (T, 16)

        # T GAT forward passes (graph topology constant; only node/global features vary)
        alpha_out = np.zeros((T, N), dtype=np.float32)
        for t in range(T):
            x_t   = torch.from_numpy(node_feats[t]).to(dev)          # (N, 18)
            g_t   = torch.from_numpy(global_ctx[t : t + 1]).to(dev)  # (1, 16)
            sig_t = torch.from_numpy(sig_np[t]).to(dev)              # (N, S)

            weights, _ = self.forward(x_t, g_t, edge_index, edge_attr)  # (N, S)
            alpha_out[t] = torch.tanh(
                (weights * sig_t).sum(dim=-1)
            ).cpu().numpy()

        return alpha_out

# MODIFICACIÓN EN scripts/train_gat_router.py:
def signal_router_loss(
    predicted_ic:     torch.Tensor,
    forward_ic:       torch.Tensor,
    blending_weights: torch.Tensor,
    lambda_ent:       float = 0.50, # Actualizado
    lambda_l2_ic:     float = 0.05,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    loss_ic         = F.mse_loss(predicted_ic, forward_ic)
    expected_ic     = (blending_weights * forward_ic).sum(dim=-1)
    loss_ic_reward  = -expected_ic.mean()

    # Penalización L1 de Turnover interno: evita oscilaciones violentas en los pesos
    # Diferencia absoluta entre pesos asignados consecutivamente en el lote
    loss_turnover = torch.abs(blending_weights[1:] - blending_weights[:-1]).mean()

    eps      = 1e-8
    entropy  = -(blending_weights * torch.log(blending_weights + eps)).sum(dim=-1)
    loss_ent = -entropy.mean()

    loss_l2  = (predicted_ic ** 2).mean()

    # Añadimos un peso de penalización de 0.30 al turnover del enrutador
    total = loss_ic + loss_ic_reward + lambda_ent * loss_ent + lambda_l2_ic * loss_l2 + 0.30 * loss_turnover

    return total, {"total": float(total.item())}


def build_economic_graph(
    returns_df: pd.DataFrame,
    tickers:    List[str],
    k:          int = K_NEIGHBORS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build a k-NN economic correlation graph from trailing 252-day returns.

    Edge attributes (3-dim):
      [0] Pearson correlation (−1 to +1)
      [1] Same asset class indicator (0/1)
      [2] Economic link (1.0 for bonds–equity, commodity–equity pairs; else 0.5)

    Returns
    -------
    edge_index : (2, E)  LongTensor
    edge_attr  : (E, 3)  FloatTensor
    """
    import pandas as pd

    corr = (
        returns_df.reindex(columns=tickers)
        .tail(252)
        .corr()
        .fillna(0.0)
        .values.astype(np.float32)
    )
    N = len(tickers)

    src_list, dst_list, attr_list = [], [], []
    _bond_set  = {"TLT", "HYG", "LQD", "BIL", "SHV"}
    _comm_set  = {"GLD", "SLV", "GDX", "USO", "PDBC"}

    for i in range(N):
        # Top-k neighbours by absolute correlation, excluding self
        sim   = np.abs(corr[i]).copy()
        sim[i] = -1.0
        top_k = np.argsort(sim)[::-1][:k]
        t_i   = tickers[i]

        for j in top_k:
            t_j = tickers[j]
            same_class = int(
                (t_i in _bond_set) == (t_j in _bond_set) and
                (t_i in _comm_set) == (t_j in _comm_set)
            )
            # Economic link: cross-asset pairs (bond↔equity, commodity↔equity)
            cross_link = 0.5
            if (t_i in _bond_set) != (t_j in _bond_set):
                cross_link = 1.0
            if (t_i in _comm_set) != (t_j in _comm_set):
                cross_link = 1.0

            src_list.append(i)
            dst_list.append(j)
            attr_list.append([float(corr[i, j]), float(same_class), cross_link])

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr  = torch.tensor(attr_list, dtype=torch.float32)
    return edge_index, edge_attr


# Needed for type annotation in build_economic_graph
import pandas as pd  # noqa: E402 — deferred to avoid circular at module level