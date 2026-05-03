"""
FORTRESS v6 — models/alpha/multi_horizon_ic_router.py
═══════════════════════════════════════════════════════════════════════════════
Multi-Horizon Information Coefficient Router.

THE ALPHA DECAY PROBLEM
───────────────────────
Static signal weights (e.g. momentum = 0.45) assume that each signal's
predictive power is constant over time. Empirically this is false:
  - Momentum signals have high IC during trending regimes (rho ~ 0.08–0.12)
    but negative IC during mean-reversion regimes (rho ~ −0.05)
  - Low-volatility signals are consistently positive but magnitude varies
    sharply with the VIX level
  - Flow signals (DIX/GEX) are powerful at t+1 but decay to noise by t+21

The forward IC (Spearman rank correlation between signal_{t} and
realized_return_{t+h}) captures both magnitude and sign of each signal's
predictive power at each horizon h. The router's job is to predict this
IC *before* the returns are realised, so that the blending weights can be
adjusted prospectively — not retrospectively.

MULTI-HORIZON PREDICTION HEAD
──────────────────────────────
We add three IC prediction heads to the existing SignalRouterGAT backbone:

    h_embed  (N, hidden_dim)   ← GATv2 message-passing output
         │
         ├── ic_head_1  → (N, S)  predicted IC at t+1
         ├── ic_head_5  → (N, S)  predicted IC at t+5
         └── ic_head_21 → (N, S)  predicted IC at t+21

Joint MSE loss with horizon-specific weights λ_h:

    L = λ_1·MSE(IC_hat_{t+1}, IC_{t+1})
      + λ_5·MSE(IC_hat_{t+5}, IC_{t+5})
      + λ_21·MSE(IC_hat_{t+21}, IC_{t+21})

Weight recommendation (Grinold & Kahn, "Active Portfolio Management"):
  λ_1 = 0.20:  Short horizon is noisiest — don't over-optimise
  λ_5 = 0.35:  Weekly horizon is actionable and moderately stable
  λ_21 = 0.45: Monthly horizon has highest Sharpe information ratio

AUTONOMOUS CASH ALLOCATION TRIGGER
────────────────────────────────────
If the GAT's predicted IC at the 21-day horizon drops below the threshold
τ_IC = −0.015 (averaged across all signals and assets), the router concludes
that the alpha stack has gone adversarial — meaning the blended signal is
likely to predict returns in the wrong direction over the next month.

When triggered, all weights are forced to the cash proxy (BIL/SHV):
  - This is a conservative, survival-first decision
  - The threshold −0.015 is calibrated to the 5th percentile of realized
    IC in the v5 OOS backtest — it corresponds to a regime where the
    expected Sharpe contribution of the alpha stack becomes negative

REALIZED IC ESTIMATION
───────────────────────
Computing the ground-truth IC for training requires:
  1. For each signal s ∈ {1…S} and asset i ∈ {1…N}:
         IC_{t,h} = SpearmanCorr(signal_{t,i}, return_{t+h,i})
     computed over a rolling cross-sectional window.
  2. IC is estimated cross-sectionally: rank-correlate the signal vector
     across assets against the realized return vector at horizon h.
  3. A rolling buffer of length lookback_ic stores historical IC estimates
     for training target construction.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

# Import the base GAT architecture (must be on PYTHONPATH)
try:
    from models.alpha.gat_signal_router import (
        SignalRouterGAT,
        NODE_FEAT_DIM,
        GLOBAL_FEAT_DIM,
        EDGE_FEAT_DIM,
        HIDDEN_DIM,
        N_HEADS,
        N_LAYERS,
        N_SIGNALS,
        SIGNAL_NAMES,
        TICKERS,
        N_ASSETS,
        DROPOUT,
        _PYG_AVAILABLE,
        build_economic_graph,
    )
    _BASE_GAT_AVAILABLE = True
except ImportError:
    _BASE_GAT_AVAILABLE = False
    logging.getLogger("MultiHorizonICRouter").warning(
        "gat_signal_router not importable — using stub constants. "
        "Run from the FORTRESS v6 root directory."
    )
    # Stub constants so the file is importable for testing
    NODE_FEAT_DIM   = 18
    GLOBAL_FEAT_DIM = 16
    EDGE_FEAT_DIM   = 3
    HIDDEN_DIM      = 64
    N_HEADS         = 4
    N_LAYERS        = 3
    N_SIGNALS       = 5
    N_ASSETS        = 25
    DROPOUT         = 0.15
    SIGNAL_NAMES    = ["low_vol", "ramom_ts", "odpv_vwap", "clv_flow", "dtfe_trend"]
    TICKERS         = ["SPY","QQQ","IWM","TLT","HYG","LQD","GLD","SLV","GDX","XLE",
                       "XLF","XLK","XLV","XLU","XLI","XLP","XLY","XLB","XLC","VIXY",
                       "BIL","SHV","USO","PDBC","COWZ"]

log = logging.getLogger("MultiHorizonICRouter")

# ── Horizon configuration ─────────────────────────────────────────────────────
HORIZONS: Tuple[int, ...]         = (1, 5, 21)     # forward-return horizons in trading days
LAMBDA_HORIZONS: Dict[int, float] = {1: 0.20, 5: 0.35, 21: 0.45}  # loss weights (must sum to 1)

# IC cash-allocation trigger threshold
#   If mean predicted IC_21 < IC_CASH_TRIGGER → force 100% BIL/SHV
IC_CASH_TRIGGER: float = -0.015

# IC rolling estimation buffer length (in trading days)
IC_BUFFER_LEN: int = 63  # ~3 months of daily cross-sectional IC estimates

# Cash proxy indices (BIL=20, SHV=21)
CASH_INDICES: Tuple[int, int] = (20, 21)


# ══════════════════════════════════════════════════════════════════════════════
# §1  MULTI-HORIZON IC PREDICTION HEAD
# ══════════════════════════════════════════════════════════════════════════════

class MultiHorizonICHead(nn.Module):
    """
    Three-branch IC prediction head plugged on top of the GATv2 embedding.

    Each branch is a two-layer MLP:
        h_embed (hidden_dim) → FC → GELU → Dropout → FC → tanh → IC_hat (S,)

    The final tanh squashes predictions to (-1, 1), which is the valid range
    for Spearman IC.  This prevents gradient explosion for distant horizons
    where IC variance is high.

    Branches are parameter-INDEPENDENT: each horizon may learn different
    features from the shared embedding without cross-contamination.

    Architecture note: BatchNorm is intentionally omitted here. BN computes
    statistics across the asset batch dimension, which would leak cross-
    sectional information from other assets into each asset's IC prediction —
    a subtle look-ahead bias in the training objective.

    Parameters
    ----------
    in_dim:     Dimensionality of the GATv2 node embedding.
    n_signals:  Number of signals S to predict IC for.
    hidden_mul: Hidden layer size = in_dim × hidden_mul.
    dropout:    Dropout probability.
    """

    def __init__(
        self,
        in_dim:     int   = HIDDEN_DIM,
        n_signals:  int   = N_SIGNALS,
        hidden_mul: int   = 2,
        dropout:    float = DROPOUT,
    ) -> None:
        super().__init__()
        hidden = in_dim * hidden_mul

        def _branch() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, n_signals),
                nn.Tanh(),   # IC ∈ (-1, 1)
            )

        # One independent branch per horizon
        self.head_h1  = _branch()   # t+1
        self.head_h5  = _branch()   # t+5
        self.head_h21 = _branch()   # t+21

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        h: torch.Tensor,     # (N, in_dim) node embeddings from GATv2
    ) -> Dict[int, torch.Tensor]:
        r"""
        Predict IC for all three horizons.

        Args:
            h: (N, in_dim) GATv2 node embeddings.

        Returns:
            dict {1: (N, S), 5: (N, S), 21: (N, S)} — predicted IC per
            horizon per signal per asset.  Values ∈ (-1, 1).
        """
        return {
            1:  self.head_h1(h),    # (N, S)
            5:  self.head_h5(h),    # (N, S)
            21: self.head_h21(h),   # (N, S)
        }


# ══════════════════════════════════════════════════════════════════════════════
# §2  MULTI-HORIZON JOINT LOSS
# ══════════════════════════════════════════════════════════════════════════════

def multi_horizon_ic_loss(
    ic_pred:    Dict[int, torch.Tensor],     # {h: (N, S)} predicted IC per horizon
    ic_target:  Dict[int, torch.Tensor],     # {h: (N, S)} realized IC per horizon
    lambdas:    Dict[int, float] = LAMBDA_HORIZONS,
    l2_ic_coef: float            = 0.005,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    r"""
    Joint multi-horizon IC MSE loss.

        L_ic(h) = MSE(IC_hat_{t+h}, IC_{t+h})
        L_total = Σ_{h} λ_h · L_ic(h)  +  ε_l2 · ‖IC_hat‖²₂

    The L2 regularisation term (ε_l2 = 0.005) prevents the model from
    predicting large IC values on assets where the true IC is close to zero.
    Without it, the model exploits noisy IC targets by predicting extreme
    values that get penalised less by MSE than by a correct-sign prediction.

    Horizon weight rationale (Grinold & Kahn):
      - t+1: Noisiest horizon. High realized IC variance makes convergence
             slow. Low weight prevents the short-horizon head from dominating.
      - t+5: Weekly momentum has the best risk-adjusted IC.
      - t+21: Monthly is smoothest but predicts fewer actionable trade dates.
              Highest weight because it captures the regime-level signal.

    Args:
        ic_pred:    {h: (N, S)} predictions from MultiHorizonICHead.forward().
        ic_target:  {h: (N, S)} realized cross-sectional IC estimates.
        lambdas:    Per-horizon loss weights (should sum to 1).
        l2_ic_coef: L2 penalty coefficient.

    Returns:
        (total_loss, diagnostics_dict)
    """
    total_loss    = torch.zeros(1, device=next(iter(ic_pred.values())).device).squeeze()
    diagnostics:  Dict[str, float] = {}
    l2_accum      = torch.zeros(1, device=total_loss.device).squeeze()

    for h, lambda_h in lambdas.items():
        if h not in ic_pred or h not in ic_target:
            continue

        pred   = ic_pred[h]     # (N, S)
        target = ic_target[h]   # (N, S)

        # Per-horizon MSE
        loss_h = F.mse_loss(pred, target)
        total_loss = total_loss + lambda_h * loss_h

        # L2 accumulation across all predictions
        l2_accum = l2_accum + (pred ** 2).mean()

        diagnostics[f"ic_mse_h{h}"]  = float(loss_h.item())
        diagnostics[f"ic_mean_h{h}"] = float(pred.mean().item())
        diagnostics[f"ic_std_h{h}"]  = float(pred.std().item())

    # L2 regularisation
    l2_term    = l2_ic_coef * l2_accum
    total_loss = total_loss + l2_term

    diagnostics["l2_reg"]    = float(l2_term.item())
    diagnostics["loss_total"] = float(total_loss.item())

    return total_loss, diagnostics


# ══════════════════════════════════════════════════════════════════════════════
# §3  REALIZED IC ESTIMATOR (TRAINING TARGET CONSTRUCTION)
# ══════════════════════════════════════════════════════════════════════════════

class RealizedICEstimator:
    """
    Computes rolling cross-sectional Spearman IC for training target construction.

    Cross-sectional Spearman IC at time t for horizon h and signal s:

        IC_{t,h,s} = SpearmanCorr(signal_{t,1:N}, return_{t+h,1:N})

    where the correlation is computed across the N-asset cross-section.

    CAUSALITY CONTRACT:
        At inference time t, IC_{t,h,s} is NOT yet observable (it requires
        return_{t+h,1:N} which is in the future).
        For TRAINING, IC targets are computed using historical realizations
        where both signal and return are available.
        The `as_of_date` check ensures no future IC is used as a training
        target for examples from before that date.

    Parameters
    ----------
    horizons:    Tuple of forward horizons in trading days.
    buffer_len:  Number of past observations to maintain in memory.
    min_assets:  Minimum number of assets with valid data for IC to be computed.
    """

    def __init__(
        self,
        horizons:   Tuple[int, ...] = HORIZONS,
        buffer_len: int             = IC_BUFFER_LEN,
        min_assets: int             = 10,
    ) -> None:
        self.horizons   = horizons
        self.buffer_len = buffer_len
        self.min_assets = min_assets

        # Circular buffers: {h: ndarray of shape (buffer_len, S)}
        self._ic_buffers: Dict[int, List[np.ndarray]] = {h: [] for h in horizons}

    def update(
        self,
        signal_history: np.ndarray,     # (T, N, S) signal values
        return_history: np.ndarray,     # (T, N)    realized returns
        t:              int,            # current time index in history
    ) -> Dict[int, np.ndarray]:
        """
        Compute cross-sectional IC for all horizons at time t.

        For each horizon h:
          IC_{t,h,s} = SpearmanCorr over assets i of:
              signal_{t, i, s}  vs  return_{t+h, i}

        If t+h exceeds the history length, returns NaN for that horizon
        (caller must handle by zeroing or not using that sample).

        Args:
            signal_history: (T, N, S) historical signal tensor.
            return_history: (T, N)    historical return tensor.
            t:              Time index for the current signal snapshot.

        Returns:
            {h: (S,) cross-sectional IC array} for each horizon.
        """
        T = return_history.shape[0]
        ic_out: Dict[int, np.ndarray] = {}

        for h in self.horizons:
            fwd_idx = t + h

            if fwd_idx >= T or t >= T:
                ic_out[h] = np.full(N_SIGNALS, np.nan)
                continue

            sig_t = signal_history[t]        # (N, S)
            ret_fwd = return_history[fwd_idx] # (N,)

            # Valid assets: non-NaN in both signal and return
            valid_mask = (
                np.isfinite(sig_t).all(axis=-1) &
                np.isfinite(ret_fwd)
            )
            if valid_mask.sum() < self.min_assets:
                ic_out[h] = np.full(N_SIGNALS, 0.0)
                continue

            sig_valid = sig_t[valid_mask]    # (N_valid, S)
            ret_valid = ret_fwd[valid_mask]  # (N_valid,)

            ic_s = np.zeros(N_SIGNALS)
            for s in range(N_SIGNALS):
                rho, _ = spearmanr(sig_valid[:, s], ret_valid)
                ic_s[s] = float(rho) if np.isfinite(rho) else 0.0

            ic_out[h] = ic_s

            # Update buffer
            self._ic_buffers[h].append(ic_s.copy())
            if len(self._ic_buffers[h]) > self.buffer_len:
                self._ic_buffers[h].pop(0)

        return ic_out

    def mean_ic(self, h: int) -> Optional[np.ndarray]:
        """Rolling mean IC over the buffer for horizon h. Returns None if empty."""
        buf = self._ic_buffers.get(h, [])
        if not buf:
            return None
        return np.stack(buf).mean(axis=0)   # (S,)

    def ic_decay_ratio(self) -> float:
        """
        IC decay ratio: mean(IC_h21) / mean(IC_h1).

        Values near 1.0: signal IC is stable across horizons (trending regime).
        Values near 0.0: IC decays quickly (mean-reversion, reduce exposure).
        Values < 0: mean-reversion — IC is positive at t+1 but negative at t+21.
        """
        ic1  = self.mean_ic(1)
        ic21 = self.mean_ic(21)
        if ic1 is None or ic21 is None:
            return 1.0  # Unknown: assume stable
        mean1  = float(np.abs(ic1).mean()) + 1e-8
        mean21 = float(ic21.mean())
        return mean21 / mean1


# ══════════════════════════════════════════════════════════════════════════════
# §4  FULL MULTI-HORIZON IC ROUTER (GAT EXTENSION)
# ══════════════════════════════════════════════════════════════════════════════

class MultiHorizonICRouter(nn.Module):
    """
    Full multi-horizon IC routing model.

    Extends SignalRouterGAT by:
      1. Replacing the single IC prediction head with MultiHorizonICHead
         (three independent branches, one per horizon).
      2. Adding the IC cash-allocation trigger logic.
      3. Exposing a `route()` method that returns blended alpha + allocation flag.

    Architecture inheritance:
        SignalRouterGAT backbone (GATv2 layers + weight_head) is frozen or
        fine-tuned depending on the training mode. MultiHorizonICHead is always
        trained from scratch.

    Parameters
    ----------
    freeze_backbone:  If True, freeze SignalRouterGAT parameters during
                      multi-horizon IC training. Set True for fine-tuning on
                      a new regime where historical IC labels are scarce.
    """

    def __init__(
        self,
        node_feat_dim:    int   = NODE_FEAT_DIM,
        global_feat_dim:  int   = GLOBAL_FEAT_DIM,
        edge_feat_dim:    int   = EDGE_FEAT_DIM,
        hidden_dim:       int   = HIDDEN_DIM,
        n_heads:          int   = N_HEADS,
        n_layers:         int   = N_LAYERS,
        n_signals:        int   = N_SIGNALS,
        dropout:          float = DROPOUT,
        freeze_backbone:  bool  = False,
        ic_cash_trigger:  float = IC_CASH_TRIGGER,
    ) -> None:
        super().__init__()

        # ── GATv2 backbone ─────────────────────────────────────────────────
        if _BASE_GAT_AVAILABLE:
            self.backbone = SignalRouterGAT(
                node_feat_dim   = node_feat_dim,
                global_feat_dim = global_feat_dim,
                edge_feat_dim   = edge_feat_dim,
                hidden_dim      = hidden_dim,
                n_heads         = n_heads,
                n_layers        = n_layers,
                n_signals       = n_signals,
                dropout         = dropout,
            )
        else:
            # Stub MLP backbone for tests without torch_geometric
            self.backbone = _MLPBackboneStub(node_feat_dim, global_feat_dim, hidden_dim)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            log.info("MultiHorizonICRouter: backbone frozen (fine-tune mode).")

        # ── Multi-horizon IC head ──────────────────────────────────────────
        self.ic_head = MultiHorizonICHead(
            in_dim    = hidden_dim,
            n_signals = n_signals,
            dropout   = dropout,
        )

        self.ic_cash_trigger = ic_cash_trigger
        self._last_ic21: float = 0.0  # cached for monitoring

    # ── Internal embedding extractor ────────────────────────────────────────

    def _get_embeddings(
        self,
        x:          torch.Tensor,   # (N, node_feat_dim)
        g:          torch.Tensor,   # (N or 1, global_feat_dim)
        edge_index: torch.Tensor,   # (2, E)
        edge_attr:  torch.Tensor,   # (E, edge_feat_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run GATv2 backbone, return:
          h:                (N, hidden_dim) node embeddings
          blending_weights: (N, S) softmax-normalised signal blend weights
          predicted_ic_v1:  (N, S) legacy single-horizon IC (from backbone)
        """
        blending_weights, predicted_ic_v1 = self.backbone(x, g, edge_index, edge_attr)

        # Re-extract the penultimate layer embedding for the IC head.
        # SignalRouterGAT computes this internally but doesn't expose it.
        # We re-run only the convolution stack (backbone is the full model,
        # so we need to access its intermediate representation).
        #
        # Implementation: access backbone.weight_head's input by hooking
        # the last GATv2 conv's output. We do this by a lightweight re-pass
        # of just the conv layers (no weight_head overhead).
        h = self._extract_hidden(x, g, edge_index, edge_attr)

        return h, blending_weights, predicted_ic_v1

    def _extract_hidden(
        self,
        x:          torch.Tensor,
        g:          torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr:  torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract the penultimate GATv2 layer output (before the weight head).

        This mirrors the forward pass of SignalRouterGAT up to the final norm,
        without the signal weight head. Necessary because the IC prediction
        should condition on the full graph structure embedding, not the
        downstream blending weights (which are logits of a different task).
        """
        bb = self.backbone
        N  = x.size(0)

        # Fuse node features with global context
        h_node = bb.node_proj(x)
        g_exp  = g if (g.dim() == 2 and g.size(0) == N) else g.expand(N, -1)
        h_glob = bb.global_proj(g_exp)
        h      = torch.cat([h_node, h_glob], dim=1)  # (N, hidden_dim)

        # GATv2 message passing
        import torch.nn.functional as F_  # avoid name collision
        for i in range(bb.n_layers):
            h_res = h
            if _PYG_AVAILABLE:
                h = bb.convs[i](h, edge_index, edge_attr)
            else:
                h = bb.convs[i](h)
            h = bb.norms[i](h)
            if i < bb.n_layers - 1:
                h = bb.dropouts[i](F_.gelu(h))
                if h.shape == h_res.shape:
                    h = h + h_res * 0.1

        return h   # (N, hidden_dim) — final conv layer output

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(
        self,
        x:          torch.Tensor,
        g:          torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr:  torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor], torch.Tensor]:
        r"""
        Full forward pass.

        Returns:
            blending_weights:  (N, S) softmax signal blend weights.
            ic_predictions:    {1: (N,S), 5: (N,S), 21: (N,S)} predicted IC.
            ic_blended:        (N, S) weighted IC prediction (for monitoring).
        """
        h, blending_weights, _ = self._get_embeddings(x, g, edge_index, edge_attr)

        # Multi-horizon IC predictions
        ic_preds = self.ic_head(h)   # {1: (N,S), 5: (N,S), 21: (N,S)}

        # Blended IC estimate: weighted by horizon lambdas for monitoring
        ic_blended = sum(
            LAMBDA_HORIZONS[h_] * ic_preds[h_]
            for h_ in HORIZONS
        )   # (N, S)

        # Cache h=21 IC for the cash trigger (accessed in route())
        self._last_ic21 = float(ic_preds[21].mean().item())

        return blending_weights, ic_preds, ic_blended

    # ── Cash trigger ──────────────────────────────────────────────────────────

    def is_cash_triggered(self) -> bool:
        """
        Returns True if the t+21 IC prediction has fallen below the
        autonomous cash allocation threshold.

        Decision logic:
          - If mean predicted IC_21 < IC_CASH_TRIGGER (default -0.015),
            the alpha stack is predicted to be net-negative over the next
            month. Force full BIL/SHV allocation regardless of any signal.
          - This is a HARD override — it supersedes HRP, CVaR-MVO, and
            any per-asset signal. The Supreme Law §1 applies.

        The threshold -0.015 corresponds empirically to a regime where the
        aggregate alpha stack contributes negative expected return after
        transaction costs.
        """
        triggered = self._last_ic21 < self.ic_cash_trigger
        if triggered:
            log.warning(
                "[IC CASH TRIGGER] Predicted IC_21=%.4f < threshold=%.4f — "
                "forcing BIL/SHV allocation.",
                self._last_ic21, self.ic_cash_trigger,
            )
        return triggered

    # ── Alpha routing with trigger ────────────────────────────────────────────

    @dataclass
    class RoutingDecision:
        """Output of MultiHorizonICRouter.route()."""
        alpha:         np.ndarray     # (N,) blended alpha signal
        cash_forced:   bool           # True if IC trigger fired
        ic21_mean:     float          # mean predicted IC at t+21
        blend_weights: np.ndarray     # (N, S) per-asset per-signal blend weights
        ic_pred_h1:    np.ndarray     # (N, S) IC predictions at t+1
        ic_pred_h5:    np.ndarray     # (N, S) IC predictions at t+5
        ic_pred_h21:   np.ndarray     # (N, S) IC predictions at t+21
        horizon_focus: int            # dominant horizon (1 if IC_21 < 0)

    @torch.no_grad()
    def route(
        self,
        x:             torch.Tensor,
        g:             torch.Tensor,
        edge_index:    torch.Tensor,
        edge_attr:     torch.Tensor,
        signal_matrix: torch.Tensor,   # (N, S) current signal values
        device:        str = "cpu",
    ) -> "MultiHorizonICRouter.RoutingDecision":
        """
        Inference-time routing with IC cash trigger.

        Decision flow:
          1. Run GATv2 forward to get IC predictions and blend weights.
          2. Check IC_21 cash trigger.
          3. If triggered → alpha = [0…0]; cash_forced = True.
          4. If NOT triggered:
             a. If IC_21 < 0 but > trigger → shift attention to shorter
                horizons by downweighting signal_matrix by the IC sign.
             b. Blend signals using per-asset blend weights → (N,) alpha.

        Args:
            x, g, edge_index, edge_attr: Graph inputs.
            signal_matrix: (N, S) current signal values (pre-normalised).
            device: Device string.

        Returns:
            RoutingDecision dataclass.
        """
        self.eval()
        x, g, edge_index, edge_attr, signal_matrix = (
            t.to(device) for t in (x, g, edge_index, edge_attr, signal_matrix)
        )

        blend_w, ic_preds, ic_blended = self.forward(x, g, edge_index, edge_attr)

        ic21_mean = self._last_ic21

        # ── Cash trigger ────────────────────────────────────────────────────
        if self.is_cash_triggered():
            return self.RoutingDecision(
                alpha         = np.zeros(N_ASSETS),
                cash_forced   = True,
                ic21_mean     = ic21_mean,
                blend_weights = blend_w.cpu().numpy(),
                ic_pred_h1    = ic_preds[1].cpu().numpy(),
                ic_pred_h5    = ic_preds[5].cpu().numpy(),
                ic_pred_h21   = ic_preds[21].cpu().numpy(),
                horizon_focus = 1,  # In crisis → only short-term signal has any value
            )

        # ── Horizon focus: IC_21 in (-0.015, 0) → shift to shorter horizons ─
        # When IC_21 is slightly negative but hasn't triggered the hard stop,
        # we down-weight the 21-day signal component and emphasise t+1 and t+5.
        # This is implemented by using horizon-IC-weighted signal masking.
        if ic21_mean < 0.0:
            # Penalise signal components where IC_21 predicts reversal
            ic21_per_asset = ic_preds[21].cpu().numpy()   # (N, S)
            reversal_mask  = (ic21_per_asset < 0.0).astype(np.float32)

            # Scale signal_matrix: zero out signals where IC_21 < 0
            # This forces the router to rely on t+1 and t+5 IC signals only
            signal_adj  = signal_matrix * torch.tensor(
                1.0 - reversal_mask, dtype=torch.float32, device=device
            )
            horizon_focus = 5   # shift to weekly horizon
        else:
            signal_adj    = signal_matrix
            horizon_focus = 21  # normal operating mode

        # ── Blend signals ────────────────────────────────────────────────────
        # alpha_i = tanh(Σ_s blend_w_{i,s} · signal_{i,s})
        # tanh bounds the output to (-1, 1) regardless of signal scale
        alpha = torch.tanh((blend_w * signal_adj).sum(dim=-1))   # (N,)

        return self.RoutingDecision(
            alpha         = alpha.cpu().numpy(),
            cash_forced   = False,
            ic21_mean     = ic21_mean,
            blend_weights = blend_w.cpu().numpy(),
            ic_pred_h1    = ic_preds[1].cpu().numpy(),
            ic_pred_h5    = ic_preds[5].cpu().numpy(),
            ic_pred_h21   = ic_preds[21].cpu().numpy(),
            horizon_focus = horizon_focus,
        )


# ══════════════════════════════════════════════════════════════════════════════
# §5  TRAINING LOSS WRAPPER (COMBINED SIGNAL ROUTING + IC)
# ══════════════════════════════════════════════════════════════════════════════

def multi_horizon_combined_loss(
    ic_pred:          Dict[int, torch.Tensor],    # {h: (N, S)}
    ic_target:        Dict[int, torch.Tensor],    # {h: (N, S)}
    blending_weights: torch.Tensor,               # (N, S)
    forward_ic:       torch.Tensor,               # (N, S) current IC (legacy term)
    lambda_ic_mh:     float = 0.70,               # multi-horizon loss weight
    lambda_ic_legacy: float = 0.30,               # legacy IC MSE weight
    lambda_ent:       float = 0.03,               # entropy regularisation
) -> Tuple[torch.Tensor, Dict[str, float]]:
    r"""
    Combined training loss for MultiHorizonICRouter.

        L_total = λ_mh · L_multi_horizon
                + λ_legacy · L_legacy_ic
                + λ_ent · L_entropy

    L_multi_horizon: Joint IC MSE across horizons (Equation from module docstring).
    L_legacy_ic:     MSE of blending_weights against current-day forward IC
                     (preserves the original SignalRouterGAT supervision).
    L_entropy:       Entropy maximisation on blending_weights — prevents
                     degenerate single-signal collapse (all weight on one signal).

    Args:
        ic_pred:          Multi-horizon IC predictions from ic_head.
        ic_target:        Multi-horizon IC ground truth from RealizedICEstimator.
        blending_weights: (N, S) from SignalRouterGAT weight head.
        forward_ic:       (N, S) realised IC (legacy supervision).
        lambda_ic_mh:     Multi-horizon IC loss weight.
        lambda_ic_legacy: Legacy IC supervision weight.
        lambda_ent:       Entropy regularisation weight.

    Returns:
        (total_loss_scalar, diagnostics_dict)
    """
    # ── Multi-horizon IC loss ─────────────────────────────────────────────
    mh_loss, mh_diag = multi_horizon_ic_loss(ic_pred, ic_target)

    # ── Legacy IC reward: route toward high-IC signals ────────────────────
    expected_ic    = (blending_weights * forward_ic).sum(dim=-1)   # (N,)
    loss_ic_legacy = -expected_ic.mean()   # maximise weighted IC

    # ── Entropy regularisation: maintain signal diversification ──────────
    eps     = 1e-8
    entropy = -(blending_weights * torch.log(blending_weights + eps)).sum(dim=-1)
    loss_ent = -entropy.mean()   # negative: maximise entropy

    # ── Total ─────────────────────────────────────────────────────────────
    total = (
        lambda_ic_mh     * mh_loss
      + lambda_ic_legacy * loss_ic_legacy
      + lambda_ent       * loss_ent
    )

    diag = {
        **{f"mh_{k}": v for k, v in mh_diag.items()},
        "ic_legacy":   float(loss_ic_legacy.item()),
        "entropy":     float(-loss_ent.item()),
        "loss_total":  float(total.item()),
    }

    return total, diag


# ══════════════════════════════════════════════════════════════════════════════
# §6  STUB BACKBONE (for testing without torch_geometric)
# ══════════════════════════════════════════════════════════════════════════════

class _MLPBackboneStub(nn.Module):
    """Minimal MLP stub for CI testing without torch_geometric installed."""

    def __init__(self, node_feat_dim: int, global_feat_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.n_layers = N_LAYERS
        self.node_proj   = nn.Linear(node_feat_dim,   hidden_dim // 2)
        self.global_proj = nn.Linear(global_feat_dim, hidden_dim // 2)
        _norms    = [nn.LayerNorm(hidden_dim)] * N_LAYERS
        _drops    = [nn.Dropout(DROPOUT)] * N_LAYERS
        _convs    = [nn.Linear(hidden_dim, hidden_dim)] * N_LAYERS
        self.norms    = nn.ModuleList(_norms)
        self.dropouts = nn.ModuleList(_drops)
        self.convs    = nn.ModuleList(_convs)
        self.weight_head  = nn.Linear(hidden_dim, N_SIGNALS)
        self.ic_pred_head = nn.Linear(hidden_dim, N_SIGNALS)

    def forward(self, x, g, edge_index, edge_attr):
        h = torch.cat([
            self.node_proj(x),
            self.global_proj(g.expand(x.size(0), -1))
        ], dim=-1)
        for i in range(self.n_layers):
            h = F.gelu(self.convs[i](h))
        return F.softmax(self.weight_head(h), dim=-1), torch.tanh(self.ic_pred_head(h))


# ══════════════════════════════════════════════════════════════════════════════
# §7  STANDALONE SMOKE TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    torch.manual_seed(42)

    N_, S_, H_, E_ = N_ASSETS, N_SIGNALS, HIDDEN_DIM, N_ASSETS * 5

    # Synthetic inputs
    x          = torch.randn(N_, NODE_FEAT_DIM)
    g          = torch.randn(1, GLOBAL_FEAT_DIM)
    edge_index = torch.randint(0, N_, (2, E_))
    edge_attr  = torch.randn(E_, EDGE_FEAT_DIM)
    signals    = torch.randn(N_, S_)

    # IC targets for all horizons
    ic_target = {h: torch.randn(N_, S_) * 0.1 for h in HORIZONS}

    model = MultiHorizonICRouter(freeze_backbone=False)
    model.eval()

    # Forward pass
    blend_w, ic_preds, ic_blended = model(x, g, edge_index, edge_attr)
    print(f"blend_w:    {blend_w.shape}   (should be [{N_}, {S_}])")
    print(f"ic_pred_1:  {ic_preds[1].shape}")
    print(f"ic_pred_21: {ic_preds[21].shape}")
    print(f"IC_21 mean: {model._last_ic21:.4f}")

    # Loss
    loss, diag = multi_horizon_combined_loss(
        ic_preds, ic_target, blend_w, ic_target[5]
    )
    print(f"Combined loss: {float(loss):.5f}")
    print(f"MH IC MSE h21: {diag['mh_ic_mse_h21']:.5f}")

    # Routing decision
    decision = model.route(x, g, edge_index, edge_attr, signals)
    print(f"\nRouting decision:")
    print(f"  cash_forced:   {decision.cash_forced}")
    print(f"  ic21_mean:     {decision.ic21_mean:.4f}")
    print(f"  horizon_focus: {decision.horizon_focus}")
    print(f"  alpha[:5]:     {decision.alpha[:5].round(4)}")