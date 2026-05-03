"""
fortress_v5/pipeline.py
────────────────────────
Integration layer: wires all 6 architectural modules into a coherent
daily execution loop.

Signal flow:
    raw_prices
        → AdaptiveFracDiff                (stationary features, learnable d)
        → LTCNodeEncoder / MAMLSignalAdapter (temporal embeddings, fast adapt)
        → TemporalConformalPredictor      (conformal intervals, mask abstention)
        → WassersteinHMM.predict()        (regime state, exposure multiplier)
        → CVaRMVOOptimizer.solve()        (constrained allocation with γ-friction)
        → FortressLoss (training only)    (differentiable Sharpe objective)
        → positions

Causality contract:
    All inputs to the optimizer at time t use strictly t-1 or earlier data.
    as_of_date guards are enforced at data ingestion, not here — this module
    assumes clean causal features are passed in.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from risk.wasserstein_hmm      import WassersteinHMM, RegimeState
from models.portfolio.cvar_optimizer  import CVaRMVOOptimizer, OptimResult
from signals.conformal_predictor import TemporalConformalPredictor, ConformalResult
from models.ltc_maml           import LTCNodeEncoder, MAMLSignalAdapter
from signals.frac_diff         import AdaptiveFracDiff
from training.differentiable_sharpe import (
    DifferentiableSharpe, TransactionCostModel, FortressLoss
)

log = logging.getLogger(__name__)


@dataclass
class DailyDecision:
    weights:            np.ndarray          # (N,) final target weights
    regime_state:       RegimeState
    conformal_result:   ConformalResult
    opt_result:         OptimResult
    masked_count:       int
    exposure_scalar:    float


class FortressV5Pipeline(nn.Module):
    """
    Fully differentiable daily allocation pipeline.

    Differentiability: the entire forward pass from frac-diff features through
    LTC embeddings to the Sharpe loss is autograd-compatible. Portfolio weights
    from the optimizer are NOT differentiable (cvxpy is not). For end-to-end
    training, use `training_forward()` which bypasses the optimizer and uses
    softmax-normalized alpha-weighted allocation.

    Live allocation uses `inference_forward()` which calls the full cvxpy optimizer.
    """

    TICKERS = [
        "SPY","QQQ","IWM","TLT","HYG","LQD","GLD","SLV","GDX","XLE",
        "XLF","XLK","XLV","XLU","XLI","XLP","XLY","XLB","XLC","VIXY",
        "BIL","SHV","USO","PDBC","COWZ",
    ]
    CASH_IDX = 20  # BIL

    def __init__(
        self,
        n_features:    int   = 16,       # fracDiff output features per asset
        embed_dim:     int   = 64,       # LTC hidden size
        signal_dim:    int   = 32,       # conformal model output dim
        n_regimes:     int   = 3,
        cvar_alpha:    float = 0.95,
        gamma:         float = 0.003,
        inner_lr:      float = 0.01,
        inner_steps:   int   = 3,
        device:        str   = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        super().__init__()
        N           = len(self.TICKERS)
        self.N      = N
        self.device = torch.device(device)

        # ── Module instantiation ─────────────────────────────────────────
        self.frac_diff = AdaptiveFracDiff(n_assets=N, weight_len=128)

        base_encoder   = LTCNodeEncoder(n_features, embed_dim, signal_dim)
        self.ltc_maml  = MAMLSignalAdapter(
            base_encoder, inner_lr=inner_lr, inner_steps=inner_steps
        )

        # Quantile head: maps LTC embedding to (q_low, q_high) per asset
        # N assets × signal_dim → N quantile pairs
        self.quantile_head = nn.Sequential(
            nn.Linear(signal_dim, signal_dim * 2),
            nn.SiLU(),
            nn.Linear(signal_dim * 2, N * 2),   # N × [q_low, q_high]
        )

        self.regime_hmm   = WassersteinHMM(n_regimes=n_regimes, n_features=5).to(self.device)
        self.conformal    = TemporalConformalPredictor(
            base_model    = self._quantile_model,
            n_assets      = N,
            alpha         = 0.10,
            cash_asset_idx= self.CASH_IDX,
        )
        self.optimizer_   = CVaRMVOOptimizer(
            n_assets    = N,
            alpha       = cvar_alpha,
            gamma       = gamma,
        )

        tc_model        = TransactionCostModel()
        sharpe_loss_fn  = DifferentiableSharpe(tc_model=tc_model)
        self.loss_fn    = FortressLoss(sharpe_loss_fn)

        self.to(self.device)

    def _quantile_model(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Bridge between TemporalConformalPredictor and LTC + quantile head."""
        emb, _ = self.ltc_maml.model(x)
        last   = emb[:, -1, :]                                        # (B, signal_dim)
        out    = self.quantile_head(last)                             # (B, N*2)
        q_low  = out[:, :self.N]                                     # (B, N)
        q_high = out[:, self.N:]
        return q_low, q_high

    # ── Inference path ─────────────────────────────────────────────────────

    @torch.no_grad()
    def inference_forward(
        self,
        prices_window: torch.Tensor,            # (T, N) price history ending at t-1
        regime_features: torch.Tensor,          # (T, 5) HMM feature window
        scenario_returns: np.ndarray,           # (T_scen, N) return scenarios
        time_delta: Optional[torch.Tensor] = None,  # (T,) calendar-day deltas
        support_xy: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        w_prev: Optional[np.ndarray] = None,
    ) -> DailyDecision:
        """
        Production inference: return target weights for tomorrow's open.
        Strictly no lookahead — all inputs are as_of EOD(t-1).
        """
        prices_window    = prices_window.to(self.device)
        regime_features  = regime_features.to(self.device)

        # 1. Fractional differencing → stationary features
        diff_prices, d_vals = self.frac_diff.forward_batch(prices_window)

        # 2. Regime detection
        regime_state = self.regime_hmm.predict(regime_features)

        # 3. LTC context switching — adapt if support set provided
        if support_xy is not None:
            sx, sy = support_xy
            x_inp  = diff_prices.unsqueeze(0)                        # (1, T, N) dummy batch
            alpha_emb = self.ltc_maml.adapt_and_predict(
                sx.to(self.device), sy.to(self.device),
                x_inp, time_delta
            )
        else:
            x_inp = diff_prices.T.unsqueeze(0)                       # (1, N, T) — treat assets as features
            emb, _ = self.ltc_maml.model(x_inp, time_delta)
            alpha_emb = emb[:, -1, :]                                # (1, signal_dim)

        # 4. Quantile head → alpha signal + conformal intervals
        out       = self.quantile_head(alpha_emb)                    # (1, N*2)
        q_low_np  = out[0, :self.N].cpu().numpy()
        q_high_np = out[0, self.N:].cpu().numpy()
        mu        = (q_low_np + q_high_np) / 2.0                    # midpoint as expected return

        # 5. Conformal prediction intervals + masking
        conf_result = self.conformal.predict(x_inp)
        mu_masked   = self.conformal.apply_masking(mu, conf_result.masked_assets)

        # 6. CVaR-MVO allocation
        opt_result = self.optimizer_.solve(
            mu                  = mu_masked,
            scenario_returns    = scenario_returns,
            w_prev              = w_prev,
            exposure_multiplier = regime_state.exposure_multiplier,
        )

        log.info(
            "FortressV5 decision: SR_exposure=%.2f regime=%d masked=%d "
            "turnover=%.4f CVaR=%.4f",
            regime_state.exposure_multiplier,
            regime_state.dominant_regime,
            len(conf_result.masked_assets),
            opt_result.turnover,
            opt_result.cvar,
        )

        return DailyDecision(
            weights          = opt_result.weights,
            regime_state     = regime_state,
            conformal_result = conf_result,
            opt_result       = opt_result,
            masked_count     = len(conf_result.masked_assets),
            exposure_scalar  = regime_state.exposure_multiplier,
        )

    # ── Training path ──────────────────────────────────────────────────────

    def training_forward(
        self,
        prices_window:  torch.Tensor,    # (B, T, N)
        realized_ret:   torch.Tensor,    # (B, T, N) OOS realized returns
        time_delta:     Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Differentiable forward for gradient-based training.
        Bypasses cvxpy — uses softmax-weighted alpha allocation.
        All operations are autograd-compatible.
        """
        B, T, N = prices_window.shape

        diff_prices = torch.stack([
            self.frac_diff.forward(prices_window[b])[0]   # (T, N) — note: uses per-sample frac diff
            for b in range(B)
        ], dim=0)                                                     # (B, T, N)

        emb, _ = self.ltc_maml.model(diff_prices.view(B * T, 1, N), time_delta)
        emb    = emb.view(B, T, -1)

        out    = self.quantile_head(emb)                              # (B, T, N*2)
        mu     = out[..., :N]                                         # midpoint alpha

        # Differentiable weight allocation: softmax with temperature
        # Real optimizer would go here; softmax is a differentiable proxy
        temp    = 0.1
        weights = torch.softmax(mu / temp, dim=-1)                   # (B, T, N)

        total_loss = torch.tensor(0.0, device=self.device)
        all_diag: Dict[str, List[float]] = {}

        for b in range(B):
            loss_b, diag_b = self.loss_fn(weights[b], realized_ret[b])
            total_loss = total_loss + loss_b
            for k, v in diag_b.items():
                all_diag.setdefault(k, []).append(v)

        avg_loss = total_loss / B
        avg_diag = {k: float(np.mean(v)) for k, v in all_diag.items()}

        return avg_loss, avg_diag