"""
FORTRESS v5 — training/train_gat.py   [ARCHITECTURE v2]
═══════════════════════════════════════════════════════════════════════════════
End-to-End Differentiable Portfolio Optimization Training Loop.

UPGRADES OVER v1
────────────────
  LOSS        MSE on alpha labels  →  Differentiable Sharpe Ratio on *net*
              portfolio returns.  The model now trains on what it is actually
              paid to maximise.

  TX COSTS    Standard |Δw| (non-differentiable kink at 0)  →  smooth abs:
                  smooth_abs(x) = √(x² + ε),   ε = 1 × 10⁻⁶
              Gradient flows cleanly through near-zero weight changes that
              would otherwise produce autograd subgradient = 0.

  TEMPERATURE Constant τ  →  cosine annealing  τ: 1.0 → 0.01 over epochs.
              Eliminates the two failure modes of fixed temperature:
                (a) τ large  → softmax is uniform → ∂L/∂α ≈ 0 (vanishing)
                (b) τ small  → softmax is one-hot → noisy, high-variance update

  EWC         Elastic Weight Consolidation consolidation triggered on every
              detected regime transition.  Fisher diagonal is computed via
              the empirical Fisher approximation on the departing regime's
              data.  Prevents catastrophic forgetting of prior-regime
              representations during fine-tuning on the arriving regime.

GRADIENT FLOW CONTRACT
──────────────────────
    GAT(x, edge) → α  ∈  ℝᴺ
         ↓  softmax(α/τ)
         w  ∈  Δᴺ                          (differentiable proxy for CVaR-MVO)
         ↓  _compute_net_returns(W, R, γ)
         r_net  ∈  ℝᵀ
         ↓  differentiable_sharpe(r_net)
         L_sharpe  ∈  ℝ                    (scalar, minimise)
         +
         L_ewc     ∈  ℝ                    (scalar, zero until first consolidation)
         ─────────────────────────────────
         L_total                           ← backward()

    The cvxpy CVaR-MVO optimizer is used ONLY at inference time (non-differentiable).
    During training the softmax proxy provides the differentiable bridge.
"""
from __future__ import annotations

import copy
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

from models.alpha.gat_alpha import AssetGraph, MultiRelationalGAT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("GAT_Trainer_v2")


# ══════════════════════════════════════════════════════════════════════════════
# §1  DIFFERENTIABLE PRIMITIVES
# ══════════════════════════════════════════════════════════════════════════════

#: Smoothing term ε — keeps smooth_abs C∞ at the origin.
#: At ε = 1e-6, approximation error is < 0.001 bps per weight unit.
_EPS_SMOOTH_ABS: float = 1e-6

#: Denominator guard for Sharpe ratio (prevents σ → 0 blow-up).
_EPS_SHARPE: float = 1e-8

#: Annualisation factor: √252.
_SQRT_252: float = math.sqrt(252.0)


def smooth_abs(x: torch.Tensor, eps: float = _EPS_SMOOTH_ABS) -> torch.Tensor:
    r"""
    Smooth, everywhere-differentiable approximation of |x|.

        smooth_abs(x) = √(x² + ε)

    Motivation
    ──────────
    The L1 turnover penalty  γ · Σᵢ |w_{t,i} − w_{t-1,i}|  has a
    non-differentiable kink at zero.  When a weight barely moves,
    torch.autograd falls back to the Clarke subgradient (= 0), silently
    zeroing the turnover gradient signal for near-stationary weights.

    √(x² + ε)  is C∞ everywhere:
        d/dx √(x² + ε)  =  x / √(x² + ε)   →   sign(x)  as  ε → 0

    At ε = 1e-6 the approximation error |√(x² + ε) − |x|| < 1e-3 for any
    |x| > 1e-3, which covers all realistic weight changes.

    Args:
        x:   Any tensor (shape, dtype, device are preserved).
        eps: Smoothing constant.  Default: 1e-6.

    Returns:
        Tensor with same shape / dtype / device as x.
    """
    return torch.sqrt(x.pow(2) + eps)


def differentiable_sharpe(
    net_returns: torch.Tensor,
    annualize:   bool  = True,
    eps:         float = _EPS_SHARPE,
) -> torch.Tensor:
    r"""
    Differentiable (negative) annualised Sharpe Ratio.

        SR = E[r_net] / σ[r_net] × √252

    Loss convention: returns −SR  so that minimisation = Sharpe maximisation.

    Gradient hazard: σ → 0 when all returns in the window are identical
    (degenerate weight sequence or near-zero regime).  Guarded by `eps`.

    Args:
        net_returns: (T,) daily net portfolio return series.
                     Must have T ≥ 2 (std needs at least two observations).
        annualize:   Multiply by √252.  Default True.
        eps:         Denominator guard.

    Returns:
        Scalar tensor: −SR.  Differentiable w.r.t. net_returns.
    """
    if net_returns.numel() < 2:
        # Degenerate window — return zero loss, no gradient signal.
        return torch.zeros(1, device=net_returns.device, requires_grad=True).squeeze()

    mu    = net_returns.mean()
    sigma = net_returns.std(unbiased=True).clamp(min=eps)
    sr    = mu / sigma
    if annualize:
        sr = sr * _SQRT_252
    return -sr  # negative → minimise = maximise SR


def cosine_temperature(
    epoch:        int,
    total_epochs: int,
    tau_start:    float = 1.0,
    tau_end:      float = 0.01,
) -> float:
    r"""
    Cosine annealing schedule for the softmax temperature τ.

        τ(e) = τ_end + ½·(τ_start − τ_end)·(1 + cos(π · e / E))

    Boundary values:
        e = 0  →  τ = τ_start  (near-uniform weights, smooth gradients)
        e = E  →  τ = τ_end    (near-argmax weights, sharp but noisy gradients)

    The schedule keeps τ in the gradient-productive band throughout training,
    rather than committing to a fixed value that may be wrong for all epochs.

    Args:
        epoch:        Current epoch index (0-based).
        total_epochs: Total number of training epochs E.
        tau_start:    Initial temperature (default 1.0).
        tau_end:      Final temperature (default 0.01).

    Returns:
        Float scalar τ(e).
    """
    # Clamp progress to [0, 1] to be safe against epoch > total_epochs calls.
    progress = float(epoch) / float(max(total_epochs - 1, 1))
    progress = min(max(progress, 0.0), 1.0)
    return float(tau_end + 0.5 * (tau_start - tau_end) * (1.0 + math.cos(math.pi * progress)))


# ══════════════════════════════════════════════════════════════════════════════
# §2  ALLOCATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def alpha_to_weights(alpha: torch.Tensor, tau: float) -> torch.Tensor:
    r"""
    Temperature-scaled softmax allocation:  w = softmax(α / τ).

    This is the differentiable proxy for the CVaR-MVO optimizer used during
    training.  It satisfies the simplex constraint  Σᵢ wᵢ = 1,  wᵢ ≥ 0  by
    construction, and is fully autograd-compatible.

    Jacobian:
        ∂wᵢ/∂αⱼ  =  (1/τ)·[wᵢ·δᵢⱼ − wᵢ·wⱼ]
                  →  (1/τ)·diag(w)  −  (1/τ)·wwᵀ

    The (1/τ) prefactor means:
        τ large  →  Jacobian ≈ 0                    (gradient vanishes)
        τ small  →  Jacobian spikes near argmax      (gradient is noisy)
        τ ∈ [0.05, 0.3] is empirically well-behaved for 25-asset universes.

    Args:
        alpha: (N,) raw GAT alpha predictions.
        tau:   Temperature scalar τ > 0.

    Returns:
        (N,) portfolio weight vector  w ∈ Δᴺ.
    """
    return F.softmax(alpha / tau, dim=-1)


def compute_net_returns(
    weights:     torch.Tensor,  # (T, N)
    fwd_returns: torch.Tensor,  # (T, N)
    gamma_tc:    float,
) -> torch.Tensor:
    r"""
    Compute a (T,) sequence of net portfolio returns.

        r_p,t  =  wₜᵀ · rₜ  −  γ · Σᵢ smooth_abs(wₜ,ᵢ − wₜ₋₁,ᵢ)

    The smooth_abs approximation ensures the turnover gradient flows back
    through the full weight sequence.  Standard |·| would produce zero
    subgradient for near-stationary weights.

    Boundary convention:
        w_{-1} := w_0  (no turnover cost charged on the very first step;
                        the prior portfolio is unknown and assumed equal to w_0).

    Args:
        weights:     (T, N) time-indexed weight tensor — output of alpha_to_weights.
        fwd_returns: (T, N) realized asset returns aligned with weights.
        gamma_tc:    Transaction cost rate γ (same unit as returns, e.g. 0.003 = 30bps).

    Returns:
        (T,) tensor of net daily portfolio returns.  Differentiable w.r.t. weights.
    """
    # Gross portfolio return per day: rᵢ,t = wₜ · rₜ
    gross_ret = (weights * fwd_returns).sum(dim=-1)                  # (T,)

    # Lagged weights — boundary: w_{-1} = w_0
    w_prev    = torch.cat([weights[:1].detach(), weights[:-1]], dim=0)  # (T, N)
    # Note: w_0 is detached for the boundary term only — the rest of the chain
    # remains fully differentiable.

    delta_w   = weights - w_prev                                     # (T, N)
    tc        = gamma_tc * smooth_abs(delta_w).sum(dim=-1)           # (T,) per-step cost

    return gross_ret - tc                                             # (T,) net returns


# ══════════════════════════════════════════════════════════════════════════════
# §3  ELASTIC WEIGHT CONSOLIDATION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EWCCheckpoint:
    r"""
    Stores the Fisher diagonal and MAP parameters for one consolidated regime.

    EWC Penalty (Kirkpatrick et al. 2017):

        L_ewc  =  λ/2 · Σᵢ  Fᵢ · (θᵢ − θ*ᵢ)²

    where:
        Fᵢ   = i-th diagonal of the empirical Fisher information matrix.
        θ*ᵢ  = parameter value at the consolidation checkpoint.
        λ    = EWC strength — controls the plasticity/stability trade-off.

    Larger Fᵢ means parameter i was highly influential on the loss of the
    departing regime, so it should be penalised more heavily for drifting.
    """
    fisher:             Dict[str, torch.Tensor]  # {param_name: F_diag}
    theta_star:         Dict[str, torch.Tensor]  # {param_name: θ*}
    regime_id:          int
    consolidated_epoch: int
    n_samples:          int                       # batches used to estimate Fisher


class EWCConsolidator:
    r"""
    Computes and accumulates Fisher diagonals for Elastic Weight Consolidation.

    Empirical Fisher (diagonal approximation):

        F̂ᵢ  ≈  (1/B) · Σ_b  (∂L_b/∂θᵢ)²

    where L_b is the per-batch Sharpe loss on the regime being consolidated.

    This avoids the O(P²) full Fisher matrix while retaining parameter
    importance information.  The approximation quality degrades when L_b
    gradients are correlated across batches (typical in trending regimes),
    but is sufficient for the EWC penalty to prevent catastrophic forgetting.

    Args:
        ewc_lambda:      λ penalty weight.  500–1000 is typical; larger values
                         increase stability at the cost of plasticity.
        max_batches:     Cap on the number of batches used per consolidation.
                         Limits wall-clock cost of the Fisher pass.
    """

    def __init__(self, ewc_lambda: float = 500.0, max_batches: int = 32) -> None:
        self.ewc_lambda  = ewc_lambda
        self.max_batches = max_batches
        self.history:    List[EWCCheckpoint] = []

    def consolidate(
        self,
        model:      nn.Module,
        graphs:     List[Data],
        device:     torch.device,
        regime_id:  int,
        epoch:      int,
        tau:        float,
        gamma_tc:   float,
        num_assets: int,
    ) -> None:
        """
        Compute the empirical Fisher diagonal over `graphs` and snapshot θ*.

        Timing:  called at the START of a regime transition (before any gradient
                 steps on the new regime), so θ* captures the model's knowledge
                 of the departing regime at peak fidelity.

        The consolidation forward pass uses single-step returns (one graph per
        step) and a mean-return loss instead of Sharpe, because Sharpe requires
        T ≥ 2 and the consolidation is deliberately capped at `max_batches`
        for speed.  The gradient signal for Fisher estimation does not require
        Sharpe's normalisation — it only needs to identify which parameters
        were most active on the regime's data.

        Args:
            model:      The GATv2 model being trained.
            graphs:     List of PyG Data objects from the departing regime.
            device:     Compute device.
            regime_id:  Integer regime label being consolidated.
            epoch:      Current training epoch (for bookkeeping).
            tau:        Current softmax temperature.
            gamma_tc:   Transaction cost penalty rate.
            num_assets: Asset universe size N.
        """
        logger.info(
            "[EWC] Consolidating regime %d at epoch %d | λ=%.1f | %d graphs available",
            regime_id, epoch, self.ewc_lambda, len(graphs),
        )
        model.eval()

        # Accumulator for Σ_b (∂L/∂θ)²
        fisher_acc: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(p.data)
            for name, p in model.named_parameters()
            if p.requires_grad
        }

        n_processed = 0
        loader      = DataLoader(graphs, batch_size=1, shuffle=True)

        for data in loader:
            if n_processed >= self.max_batches:
                break

            data = data.to(device)
            model.zero_grad()

            alpha  = model(data.x, data.edge_index, data.edge_attr)   # (N,)
            w      = alpha_to_weights(alpha[:num_assets], tau)         # (N,)
            r_fwd  = data.y[:num_assets].to(device)                   # (N,)

            # Single-step net return (mean-return proxy — suitable for Fisher)
            gross  = (w * r_fwd).sum()
            tc     = gamma_tc * smooth_abs(w - w.detach()).sum()
            loss   = -(gross - tc)                                     # minimise → maximise

            loss.backward()

            for name, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher_acc[name].add_(p.grad.data.pow(2))

            n_processed += 1

        # Normalise and save
        fisher_diag = {
            name: (acc / max(n_processed, 1)).detach().clone()
            for name, acc in fisher_acc.items()
        }
        theta_star = {
            name: p.data.detach().clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }

        self.history.append(EWCCheckpoint(
            fisher             = fisher_diag,
            theta_star         = theta_star,
            regime_id          = regime_id,
            consolidated_epoch = epoch,
            n_samples          = n_processed,
        ))

        model.train()
        logger.info(
            "[EWC] Consolidation complete: regime %d | %d batches used | "
            "%d total regime(s) in memory",
            regime_id, n_processed, len(self.history),
        )

    def penalty(self, model: nn.Module) -> torch.Tensor:
        r"""
        Compute the total EWC penalty across all consolidated regimes.

            L_ewc  =  (λ/2) · Σ_regimes  Σᵢ  Fᵢ · (θᵢ − θ*ᵢ)²

        Returns a zero scalar (no gradient) during the warm-up phase before
        any regime has been consolidated.

        Args:
            model: The model being trained (provides current θ).

        Returns:
            Scalar tensor — differentiable w.r.t. model parameters.
        """
        if not self.history:
            # No consolidation yet — return a detached zero to avoid
            # adding a spurious term to the computation graph.
            return torch.zeros(1, device=next(model.parameters()).device).squeeze()

        total = torch.zeros(1, device=next(model.parameters()).device).squeeze()

        for ckpt in self.history:
            for name, p in model.named_parameters():
                if name not in ckpt.fisher:
                    continue
                F_i    = ckpt.fisher[name].to(p.device)
                th_i   = ckpt.theta_star[name].to(p.device)
                total  = total + (F_i * (p - th_i).pow(2)).sum()

        return (self.ewc_lambda / 2.0) * total


# ══════════════════════════════════════════════════════════════════════════════
# §4  REGIME-AWARE DATASET
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RegimeSegment:
    """
    A time-contiguous block of graph snapshots belonging to a single regime.

    fields:
        graphs:      List[Data] — one PyG graph per trading day.
        fwd_returns: (T, N) realized forward returns aligned with graphs[t].
        regime_id:   Regime label from WassersteinHMM.predict().
        span:        (start_day, end_day) indices for logging/debugging.
    """
    graphs:      List[Data]
    fwd_returns: torch.Tensor              # (T, N)
    regime_id:   int
    span:        Tuple[int, int] = field(default=(0, 0))


def build_regime_segments(
    num_graphs:    int   = 2000,
    num_nodes:     int   = 25,
    node_feat_dim: int   = 78,
    num_regimes:   int   = 3,
    seed:          int   = 42,
) -> List[RegimeSegment]:
    """
    Scaffold a synthetic regime-segmented graph dataset for training.

    Regime structure mimics realistic macro cycles:
        Regime 0 (~40%): Low-vol bull trend     (+10% ann., σ = 8%)
        Regime 1 (~35%): High-vol bear trend    (−5% ann.,  σ = 18%)
        Regime 2 (~25%): Crisis / decorrelation (−20% ann., σ = 28%)

    In production, replace this with:
        DataPipeline.get_returns_dataframe()  →  fwd_returns
        WassersteinHMM.predict()              →  regime_id per day
        Neo4j / DYNOTEARS causal graphs       →  edge_index, edge_attr

    Args:
        num_graphs:    Total number of trading-day snapshots T.
        num_nodes:     Number of assets N (must match GAT config).
        node_feat_dim: Node feature dimensionality (must match GAT config).
        num_regimes:   Number of HMM regimes.
        seed:          RNG seed.

    Returns:
        List[RegimeSegment] in chronological order.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # ── Markov regime sequence ────────────────────────────────────────────
    # Persistent transition matrix: regimes last O(weeks/months)
    _P_full = np.array([
        [0.97, 0.02, 0.01],
        [0.03, 0.94, 0.03],
        [0.05, 0.05, 0.90],
    ])
    P = _P_full[:num_regimes, :num_regimes]
    P /= P.sum(axis=1, keepdims=True)

    regime_seq        = np.empty(num_graphs, dtype=int)
    regime_seq[0]     = 0
    for t in range(1, num_graphs):
        regime_seq[t] = rng.choice(num_regimes, p=P[regime_seq[t - 1]])

    # ── Return distribution per regime ───────────────────────────────────
    # (daily_mean, daily_vol) pairs — annualised: mean×252, vol×√252
    regime_dists = [
        ( 4e-4,  8e-3),    # Bull
        (-2e-4, 18e-3),    # Bear
        (-8e-4, 28e-3),    # Crisis
    ]

    # ── Generate graphs + forward returns ────────────────────────────────
    all_graphs:   List[Data]          = []
    all_fwd_rets: List[torch.Tensor]  = []

    for t in range(num_graphs):
        r_id          = int(regime_seq[t])
        mu_r, sig_r   = regime_dists[r_id % len(regime_dists)]
        x             = torch.randn(num_nodes, node_feat_dim)
        edge_index, edge_attr = AssetGraph.build_dummy_edge_index(num_nodes)
        y             = torch.tensor(
            rng.normal(mu_r, sig_r, num_nodes), dtype=torch.float32
        )
        all_graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
        all_fwd_rets.append(y)

    # ── Segment into contiguous regime runs ───────────────────────────────
    segments: List[RegimeSegment] = []
    run_start = 0

    for t in range(1, num_graphs + 1):
        is_end       = (t == num_graphs)
        is_break     = (not is_end) and (regime_seq[t] != regime_seq[run_start])

        if is_break or is_end:
            end = t
            fwd = torch.stack(all_fwd_rets[run_start:end], dim=0)   # (T_seg, N)
            segments.append(RegimeSegment(
                graphs      = all_graphs[run_start:end],
                fwd_returns = fwd,
                regime_id   = int(regime_seq[run_start]),
                span        = (run_start, end - 1),
            ))
            run_start = t

    logger.info(
        "Dataset scaffolded: %d graphs → %d regime segments (k=%d regimes)",
        num_graphs, len(segments), num_regimes,
    )
    return segments


# ══════════════════════════════════════════════════════════════════════════════
# §5  TRAINING DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EpochRecord:
    """One row of the training log."""
    epoch:          int
    loss_total:     float
    loss_sharpe:    float
    loss_ewc:       float
    sharpe_ratio:   float    # positive (−L_sharpe)
    mean_turnover:  float
    tau:            float
    lr:             float
    ewc_terms:      int
    regime_transitions: int


# ══════════════════════════════════════════════════════════════════════════════
# §6  MAIN TRAINER
# ══════════════════════════════════════════════════════════════════════════════

class GATTrainer:
    """
    End-to-end differentiable training loop for MultiRelationalGAT.

    Loss:
        L_total(θ)  =  L_sharpe(w(α(θ)), r)  +  L_ewc(θ)

        L_sharpe    =  −SR_net(τ)              [Sharpe of net-of-TC returns]
        L_ewc       =  (λ/2)·Σ_regimes Σᵢ Fᵢ·(θᵢ−θ*ᵢ)²

    Temperature:
        τ(e) = τ_end + ½·(τ_start−τ_end)·(1+cos(π·e/E))    [cosine decay]

    EWC:
        Consolidation fires at every detected regime transition, provided the
        departing regime ran for at least `min_ewc_run_len` consecutive steps
        (guards against short HMM noise blips triggering spurious Fisher passes).
    """

    def __init__(self, config_path: str = "config/hyperparams.yaml") -> None:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        gat_cfg   = cfg.get("gat_alpha", {})
        train_cfg = cfg.get("training",  {})

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("GATTrainer v2 initialising on %s", self.device)

        # ── Model ─────────────────────────────────────────────────────────
        self.model = MultiRelationalGAT(
            node_feat_dim = gat_cfg.get("node_feat_dim", 78),
            edge_feat_dim = gat_cfg.get("edge_feat_dim",  5),
            hidden_dim    = gat_cfg.get("hidden_dim",    128),
            n_heads       = gat_cfg.get("n_heads",         8),
            n_layers      = gat_cfg.get("n_layers",        3),
        ).to(self.device)

        # ── Hyperparameters ───────────────────────────────────────────────
        self.num_assets     = gat_cfg.get("num_nodes",         25)
        self.node_feat_dim  = gat_cfg.get("node_feat_dim",     78)
        self.epochs         = int(train_cfg.get("epochs",     100))
        self.gamma_tc       = float(train_cfg.get("gamma_tc", 3e-3))
        self.tau_start      = float(train_cfg.get("tau_start", 1.0))
        self.tau_end        = float(train_cfg.get("tau_end",  0.01))
        self.ewc_lambda     = float(train_cfg.get("ewc_lambda", 500.0))
        self.grad_clip      = float(train_cfg.get("grad_clip",   1.0))
        self.num_regimes    = int(train_cfg.get("num_regimes",     3))

        # Minimum consecutive steps in a regime before its departure triggers
        # EWC consolidation.  Below this threshold, the run is treated as noise.
        self.min_ewc_run_len: int = int(train_cfg.get("min_ewc_run_len", 10))

        # Maximum segment length to process in a single forward pass.
        # Long segments are chunked to keep the (T,N) weight tensor tractable.
        self.max_segment_chunk: int = int(train_cfg.get("max_segment_chunk", 63))

        # ── Optimiser + LR scheduler ──────────────────────────────────────
        self.optim = torch.optim.AdamW(
            self.model.parameters(),
            lr           = float(train_cfg.get("lr",         5e-4)),
            weight_decay = float(train_cfg.get("wd",         1e-3)),
        )
        # Cosine LR annealing is orthogonal to temperature annealing:
        # LR annealing slows the *step size*; temperature annealing sharpens
        # the *gradient signal*.  Both are needed.
        self.lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optim,
            T_max   = self.epochs,
            eta_min = 1e-6,
        )
        self.scaler = GradScaler()

        # ── EWC ───────────────────────────────────────────────────────────
        self.ewc = EWCConsolidator(
            ewc_lambda  = self.ewc_lambda,
            max_batches = int(train_cfg.get("ewc_max_batches", 32)),
        )

        # Dataset config (passed through to scaffold)
        self._dataset_cfg = {
            "num_graphs":    int(train_cfg.get("num_graphs",  2000)),
            "num_regimes":   self.num_regimes,
            "seed":          int(train_cfg.get("seed",          42)),
        }

        self.log: List[EpochRecord] = []

    # ── Forward: single regime segment ────────────────────────────────────

    def _forward_segment(
        self,
        seg: RegimeSegment,
        tau: float,
    ) -> Tuple[torch.Tensor, float]:
        r"""
        Differentiable forward pass over one time-contiguous regime segment.

        Algorithm
        ─────────
        For each time step t = 0 … T−1:
            αₜ  =  GAT(xₜ, edge_index, edge_attr)          (N,)
            wₜ  =  softmax(αₜ[:N] / τ)                      (N,)

        After the full sequence:
            W   =  stack(w₀, …, w_{T-1})                    (T, N)
            r_net = compute_net_returns(W, R, γ)             (T,)
            L   =  differentiable_sharpe(r_net)              scalar

        Long segments are chunked at `max_segment_chunk` steps to keep GPU
        memory tractable.  Chunk losses are averaged (not summed) to keep
        the Sharpe scale consistent regardless of chunk size.

        Args:
            seg: RegimeSegment with T graphs and (T, N) fwd_returns.
            tau: Current softmax temperature.

        Returns:
            (sharpe_loss, mean_turnover_diagnostic)
        """
        T           = len(seg.graphs)
        fwd_ret     = seg.fwd_returns.to(self.device)   # (T, N)
        chunk_size  = self.max_segment_chunk
        n_chunks    = max(1, math.ceil(T / chunk_size))

        chunk_losses:    List[torch.Tensor] = []
        chunk_turnovers: List[float]        = []

        for chunk_idx in range(n_chunks):
            t0 = chunk_idx * chunk_size
            t1 = min(t0 + chunk_size, T)
            chunk_graphs  = seg.graphs[t0:t1]
            chunk_fwd_ret = fwd_ret[t0:t1]               # (t1-t0, N)

            weights_list: List[torch.Tensor] = []

            for graph in chunk_graphs:
                g     = graph.to(self.device)
                alpha = self.model(g.x, g.edge_index, g.edge_attr)  # (N_nodes,)
                w     = alpha_to_weights(alpha[:self.num_assets], tau)
                weights_list.append(w)                               # (N,)

            W         = torch.stack(weights_list, dim=0)             # (chunk_T, N)
            net_rets  = compute_net_returns(W, chunk_fwd_ret, self.gamma_tc)
            loss_c    = differentiable_sharpe(net_rets)              # scalar

            chunk_losses.append(loss_c)

            # Diagnostic: mean L1 turnover within the chunk
            if W.shape[0] > 1:
                to = smooth_abs(W[1:] - W[:-1]).sum(dim=-1).mean().item()
            else:
                to = 0.0
            chunk_turnovers.append(to)

        sharpe_loss    = torch.stack(chunk_losses).mean()
        mean_turnover  = float(np.mean(chunk_turnovers))

        return sharpe_loss, mean_turnover

    # ── EWC: conditional consolidation ────────────────────────────────────

    def _maybe_consolidate(
        self,
        departing_regime_graphs: List[Data],
        regime_id:               int,
        run_length:              int,
        epoch:                   int,
        tau:                     float,
    ) -> bool:
        """
        Trigger EWC consolidation if the departing regime ran long enough.

        The run-length gate (min_ewc_run_len) prevents the Fisher pass from
        running on noise blips from the WassersteinHMM — short runs provide
        too few gradient samples for a reliable Fisher diagonal estimate.

        Args:
            departing_regime_graphs: All graphs from the departing regime run.
            regime_id:               Regime label being consolidated.
            run_length:              Number of steps in the departing run.
            epoch:                   Current training epoch.
            tau:                     Current softmax temperature.

        Returns:
            True if consolidation was performed, False if skipped.
        """
        if run_length < self.min_ewc_run_len:
            logger.debug(
                "[EWC] Regime %d run too short (%d < %d) — consolidation skipped.",
                regime_id, run_length, self.min_ewc_run_len,
            )
            return False

        self.ewc.consolidate(
            model       = self.model,
            graphs      = departing_regime_graphs,
            device      = self.device,
            regime_id   = regime_id,
            epoch       = epoch,
            tau         = tau,
            gamma_tc    = self.gamma_tc,
            num_assets  = self.num_assets,
        )
        return True

    # ── Main training loop ─────────────────────────────────────────────────

    def train(self) -> None:
        """
        Outer loop: epochs.  Inner loop: regime segments (chronological).

        Per-epoch:
            1.  Compute τ(e) via cosine schedule.
            2.  Iterate over regime segments in chronological order.
            3.  On each regime transition:
                    a.  Run EWC consolidation on the departing regime's graphs
                        (if the run was long enough).
                    b.  Reset the run-length counter for the arriving regime.
            4.  Forward → L_sharpe + L_ewc → backward → clip → step.
            5.  LR scheduler step (once per epoch).
            6.  Log epoch diagnostics.
        """
        segments = build_regime_segments(
            num_graphs    = self._dataset_cfg["num_graphs"],
            num_nodes     = self.num_assets,
            node_feat_dim = self.node_feat_dim,
            num_regimes   = self._dataset_cfg["num_regimes"],
            seed          = self._dataset_cfg["seed"],
        )

        self.model.train()
        logger.info(
            "═══ Training start ═══  epochs=%d | segments=%d | "
            "γ_tc=%.4f | τ: %.2f → %.2f | λ_EWC=%.1f | device=%s",
            self.epochs, len(segments), self.gamma_tc,
            self.tau_start, self.tau_end, self.ewc_lambda, self.device,
        )

        for epoch in range(1, self.epochs + 1):

            tau = cosine_temperature(epoch - 1, self.epochs, self.tau_start, self.tau_end)

            # Per-epoch accumulators
            seg_sharpe_loss:   List[float] = []
            seg_ewc_loss:      List[float] = []
            seg_total_loss:    List[float] = []
            seg_turnovers:     List[float] = []
            n_transitions:     int         = 0

            # Regime run tracker for EWC consolidation trigger
            cur_regime_id:    Optional[int]   = None
            cur_regime_graphs: List[Data]     = []
            cur_regime_run:    int            = 0

            for seg in segments:

                # ── Regime transition detection ───────────────────────────
                if cur_regime_id is not None and seg.regime_id != cur_regime_id:
                    n_transitions += 1
                    # EWC: consolidate the regime that just ended
                    self._maybe_consolidate(
                        departing_regime_graphs = cur_regime_graphs,
                        regime_id               = cur_regime_id,
                        run_length              = cur_regime_run,
                        epoch                   = epoch,
                        tau                     = tau,
                    )
                    # Reset accumulator for new regime
                    cur_regime_graphs = []
                    cur_regime_run    = 0

                cur_regime_id = seg.regime_id
                cur_regime_graphs.extend(seg.graphs)
                cur_regime_run += len(seg.graphs)

                # ── Forward pass ──────────────────────────────────────────
                self.optim.zero_grad(set_to_none=True)

                # AMP wraps the GAT message-passing and softmax allocation.
                # The Sharpe loss operates on float32 accumulations; autocast
                # keeps those in fp32 automatically.
                with autocast():
                    sharpe_loss, mean_to = self._forward_segment(seg, tau)
                    ewc_pen              = self.ewc.penalty(self.model)
                    total_loss           = sharpe_loss + ewc_pen

                # ── Backward pass ─────────────────────────────────────────
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optim)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optim)
                self.scaler.update()

                seg_sharpe_loss.append(float(sharpe_loss.item()))
                seg_ewc_loss.append(float(ewc_pen.item()))
                seg_total_loss.append(float(total_loss.item()))
                seg_turnovers.append(mean_to)

            # ── Consolidate the final regime in the dataset ───────────────
            # (it has no successor transition, so we consolidate at epoch end)
            if cur_regime_id is not None and cur_regime_run >= self.min_ewc_run_len:
                self._maybe_consolidate(
                    departing_regime_graphs = cur_regime_graphs,
                    regime_id               = cur_regime_id,
                    run_length              = cur_regime_run,
                    epoch                   = epoch,
                    tau                     = tau,
                )

            # ── LR step ───────────────────────────────────────────────────
            self.lr_sched.step()

            # ── Epoch summary ─────────────────────────────────────────────
            mean_sharpe_loss = float(np.mean(seg_sharpe_loss))
            mean_ewc_loss    = float(np.mean(seg_ewc_loss))
            mean_total_loss  = float(np.mean(seg_total_loss))
            mean_turnover    = float(np.mean(seg_turnovers))
            cur_lr           = float(self.lr_sched.get_last_lr()[0])

            record = EpochRecord(
                epoch            = epoch,
                loss_total       = mean_total_loss,
                loss_sharpe      = mean_sharpe_loss,
                loss_ewc         = mean_ewc_loss,
                sharpe_ratio     = -mean_sharpe_loss,   # positive SR
                mean_turnover    = mean_turnover,
                tau              = tau,
                lr               = cur_lr,
                ewc_terms        = len(self.ewc.history),
                regime_transitions = n_transitions,
            )
            self.log.append(record)

            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    "Epoch [%03d/%03d] | L_total=%+.5f  L_Sharpe=%+.5f  "
                    "L_EWC=%+.5f | SR=%.4f | TO=%.4f | τ=%.4f | "
                    "LR=%.2e | EWC_k=%d | transitions=%d",
                    epoch, self.epochs,
                    mean_total_loss, mean_sharpe_loss, mean_ewc_loss,
                    -mean_sharpe_loss, mean_turnover, tau,
                    cur_lr, len(self.ewc.history), n_transitions,
                )

        self._save_artifacts()

    # ── Persistence ───────────────────────────────────────────────────────

    def _save_artifacts(self) -> None:
        """
        Save model weights, EWC state, and training log.

        The EWC state is saved separately from model weights because it must
        be reloaded if training is resumed — without it, future regime
        transitions would compute the EWC penalty against stale θ* anchors.
        """
        os.makedirs("models/weights", exist_ok=True)

        # ── Model weights ─────────────────────────────────────────────────
        model_path = "models/weights/gat_alpha_latest.pt"
        torch.save(self.model.state_dict(), model_path)
        logger.info("Model weights saved → %s", model_path)

        # ── EWC state ─────────────────────────────────────────────────────
        if self.ewc.history:
            ewc_path = "models/weights/ewc_state_latest.pt"
            serialisable = [
                {
                    "fisher":             {k: v.cpu() for k, v in ckpt.fisher.items()},
                    "theta_star":         {k: v.cpu() for k, v in ckpt.theta_star.items()},
                    "regime_id":          ckpt.regime_id,
                    "consolidated_epoch": ckpt.consolidated_epoch,
                    "n_samples":          ckpt.n_samples,
                }
                for ckpt in self.ewc.history
            ]
            torch.save(serialisable, ewc_path)
            logger.info(
                "EWC state saved → %s  (%d terms)", ewc_path, len(self.ewc.history)
            )

        # ── Training log ──────────────────────────────────────────────────
        log_path = "models/weights/train_log_v2.pt"
        torch.save(self.log, log_path)

    # ── Resume / inference ─────────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        config_path:  str           = "config/hyperparams.yaml",
        weights_path: str           = "models/weights/gat_alpha_latest.pt",
        ewc_path:     Optional[str] = "models/weights/ewc_state_latest.pt",
    ) -> "GATTrainer":
        """
        Reinstantiate trainer from saved weights + EWC state.

        Used for:
            - Resuming training (EWC state must be loaded to preserve consolidations)
            - Live inference (model weights only; EWC state optional)
        """
        trainer = cls(config_path)
        trainer.model.load_state_dict(
            torch.load(weights_path, map_location=trainer.device)
        )
        logger.info("Model weights loaded ← %s", weights_path)

        if ewc_path and os.path.exists(ewc_path):
            saved = torch.load(ewc_path, map_location=trainer.device)
            for entry in saved:
                trainer.ewc.history.append(EWCCheckpoint(
                    fisher             = entry["fisher"],
                    theta_star         = entry["theta_star"],
                    regime_id          = entry["regime_id"],
                    consolidated_epoch = entry["consolidated_epoch"],
                    n_samples          = entry.get("n_samples", 0),
                ))
            logger.info(
                "EWC state loaded ← %s  (%d terms)", ewc_path, len(trainer.ewc.history)
            )
        return trainer


# ══════════════════════════════════════════════════════════════════════════════
# §7  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    trainer = GATTrainer()
    trainer.train()