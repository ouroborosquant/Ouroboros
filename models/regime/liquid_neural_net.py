"""
FORTRESS v5 - liquid_neural_net.py
Path: models/regime/liquid_neural_net.py

Continuous-time intraday regime drift detector.
Uses Liquid Time-Constant (LTC) networks to process irregular market ticks.
Architecture ONLY. No training loops.

FIXES APPLIED:
  - BUG #12: `self.hidden_state` had no device pinning. If `step()` was called after
             a container restart (CPU fallback) or with a different device argument,
             the old hidden state tensor remained on the previous device, causing a
             device mismatch RuntimeError inside `self.ltc(x, hx=self.hidden_state)`.

             Fix 1: `self.hidden_state` is now pinned to the device used during `step()`.
             Fix 2: `.detach()` is called on the returned hidden state to sever it
                    from the autograd graph — without this, every `step()` call
                    accumulates gradient history across the entire intraday session,
                    causing unbounded memory growth (~O(num_ticks) memory leak).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Optional

# External dependencies for Liquid Neural Networks
try:
    from ncps.torch import LTC
    from ncps.wirings import AutoNCP
except ImportError:
    raise ImportError("ncps is required. Install via: pip install ncps")


class IntraRegimeMonitor(nn.Module):
    """
    Intraday regime monitor utilizing Neural Circuit Policies (NCPs) and LTCs.

    This network does not output a regime class; instead, it outputs an
    `urgency_score` [0, 1] representing the probability that the current intraday
    microstructure has dangerously decoupled from the morning's expected regime.
    """

    def __init__(self, config: Dict):
        super().__init__()

        self.input_size = config.get("input_size", 15)
        self.n_units = config.get("n_units", 64)
        self.sparsity_level = config.get("sparsity_level", 0.50)

        # Thresholds loaded from hyperparams.yaml
        self.urgency_threshold = config.get("urgency_threshold", 0.70)

        # 1. Wiring: AutoNCP creates a brain-inspired sparse neural circuit.
        # We need 1 output: the drift/urgency score.
        self.wiring = AutoNCP(
            units=self.n_units,
            output_size=1,
            sparsity_level=self.sparsity_level,
        )

        # 2. Liquid Time-Constant (LTC) core.
        # batch_first=True aligns with standard PyTorch time-series shape (B, T, F).
        self.ltc = LTC(
            input_size=self.input_size,
            wiring=self.wiring,
            batch_first=True,
        )

        # FIX #12: Hidden state is tracked with device awareness.
        # Initialised to None; set/updated exclusively through `step()`.
        self.hidden_state: Optional[torch.Tensor] = None
        self._hidden_device: Optional[torch.device] = None

    def reset_hidden_state(self) -> None:
        """
        MUST be called exactly once per day at market open (09:30 AM EST).
        Clears the persistent intraday memory.
        Called by services/regime_encoder_svc.py at session start.
        """
        self.hidden_state = None
        self._hidden_device = None

    def forward(
        self, x: torch.Tensor, timespans: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Standard forward pass used primarily during training (train_ltc.py).

        Args:
            x:          Shape (Batch, Seq_Len, Input_Size)
            timespans:  Shape (Batch, Seq_Len) — continuous time deltas in seconds.

        Returns:
            urgency_score: Shape (Batch, Seq_Len, 1) — urgency trajectory in [0, 1].
            hidden:        Final hidden state for stateful inference.
        """
        out, hidden = self.ltc(x, timespans=timespans)
        # Sigmoid bounds the urgency score strictly to [0.0, 1.0].
        urgency_score = torch.sigmoid(out)
        return urgency_score, hidden

    @torch.no_grad()
    def step(
        self,
        obs: np.ndarray,
        elapsed_seconds: float,
        device: str = "cuda",
    ) -> Tuple[float, bool]:
        """
        LIVE INFERENCE METHOD.
        Called by services/regime_encoder_svc.py upon receiving each market tick.

        FIX #12 — Device Pinning:
            The hidden state is pinned to `device` on every call. If the container
            restarts and falls back from CUDA to CPU, the hidden state is moved
            automatically rather than causing a device mismatch RuntimeError.

        FIX #12 — Gradient Detach:
            `.detach()` is called on the returned hidden state before storing it.
            Without this, the autograd graph grows unboundedly across the entire
            intraday session (potentially thousands of ticks), causing a memory leak
            that will OOM the GPU well before market close.

        Args:
            obs:             15-dim numpy array of intraday features (OBI, spread, vol, etc.)
            elapsed_seconds: Time elapsed since the previous observation (irregular intervals).
            device:          PyTorch device string ('cuda' or 'cpu').

        Returns:
            drift_score:  Float in [0, 1]. Higher = more dangerous regime drift.
            urgency_flag: True if drift_score exceeds self.urgency_threshold.
        """
        self.eval()
        target_device = torch.device(device)

        # Reshape to (Batch=1, Seq_Len=1, Features=15) for the LTC.
        x = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0).to(target_device)

        # Reshape to (Batch=1, Seq_Len=1) — the irregular time step.
        ts = torch.FloatTensor([[elapsed_seconds]]).to(target_device)

        # FIX #12 (Device Pinning): Move the hidden state to the current device
        # if it exists and is on a different device (e.g., after a CUDA -> CPU fallback).
        hx: Optional[torch.Tensor] = None
        if self.hidden_state is not None:
            if self.hidden_state.device != target_device:
                logger.warning(
                    f"LTC hidden state device mismatch: "
                    f"was {self.hidden_state.device}, moving to {target_device}."
                )
                self.hidden_state = self.hidden_state.to(target_device)
            hx = self.hidden_state

        # Advance the continuous-time ODE by exactly `elapsed_seconds`.
        out, new_hidden = self.ltc(x, timespans=ts, hx=hx)

        # FIX #12 (Gradient Detach): Sever the stored hidden state from the
        # computation graph. Without `.detach()`, every tick adds a new node to
        # the autograd graph — O(ticks_per_day) memory accumulation = GPU OOM.
        self.hidden_state = new_hidden.detach()
        self._hidden_device = target_device

        drift_score = torch.sigmoid(out[0, 0, 0]).item()
        urgency_flag = bool(drift_score > self.urgency_threshold)

        return drift_score, urgency_flag


# Module-level logger (imported by step() for device mismatch warnings).
import logging
logger = logging.getLogger("IntraRegimeMonitor")