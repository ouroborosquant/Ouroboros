"""
fortress_v5/signals/conformal_predictor.py
───────────────────────────────────────────
Temporal Conformal Prediction (TCP) wrapper for GATv2 signal outputs.

Theory
------
Standard split conformal prediction gives marginal coverage ≥ 1-α:
    P(Y_{n+1} ∈ Ĉ(X_{n+1})) ≥ 1 - α

For temporal data, exchangeability is violated. We use a rolling calibration
window [t-L, t-1] and weight nonconformity scores by recency:

    ŝ_i = ρ^{n-i}   (geometric decay, ρ < 1)

This gives coverage conditional on recent distribution (SPCI-style):
    Coverage degrades gracefully as distribution shifts — a feature, not a bug.

Nonconformity score for interval prediction:
    R_i = max(q̂_low(x_i) - y_i,  y_i - q̂_high(x_i))

Calibrated correction:
    q̂_α = weighted (1-α)(1 + 1/n) quantile of {R_i}

Adjusted interval: [q̂_low - q̂_α, q̂_high + q̂_α]

Signal masking logic
--------------------
If the calibrated interval width > Δ_threshold (regime-dependent), the signal
is masked: the asset's target weight is forced to 0 and reallocated to BIL/SHV.
This is "I don't know" mode — the network explicitly abstains.
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger(__name__)


@dataclass
class ConformalResult:
    lower: np.ndarray                  # (N,) lower bound on predicted return
    upper: np.ndarray                  # (N,) upper bound
    interval_widths: np.ndarray        # (N,) = upper - lower
    masked_assets: List[int]           # indices of masked (abstained) assets
    calibration_coverage: float        # empirical coverage on calibration set
    q_correction: float                # calibration quantile correction q̂_α


class RollingCalibrationBuffer:
    """
    Fixed-size ring buffer for nonconformity score accumulation.
    Thread-unsafe by design — single-threaded pipeline assumed.
    """

    def __init__(self, max_size: int = 252, decay: float = 0.98) -> None:
        self.max_size = max_size
        self.decay    = decay
        self._scores: Deque[float] = deque(maxlen=max_size)

    def push(self, score: float) -> None:
        self._scores.append(score)

    def push_batch(self, scores: np.ndarray) -> None:
        for s in scores:
            self._scores.append(float(s))

    def quantile(self, alpha: float) -> float:
        """
        Weighted empirical quantile with geometric recency weighting.
        Recent scores (rightmost) get weight ρ^0 = 1; oldest get ρ^{n-1}.
        """
        n = len(self._scores)
        if n < 10:
            return float("inf")         # insufficient calibration data → mask all

        scores  = np.array(self._scores)                              # oldest → newest
        weights = self.decay ** np.arange(n - 1, -1, -1)             # recent = high weight
        weights /= weights.sum()

        # Weighted empirical CDF
        sort_idx     = np.argsort(scores)
        sorted_s     = scores[sort_idx]
        sorted_w     = weights[sort_idx]
        cum_w        = np.cumsum(sorted_w)

        # Bonferroni-style inflation for finite-sample coverage
        target = (1 - alpha) * (1 + 1.0 / n)
        target = min(target, 1.0)

        idx = np.searchsorted(cum_w, target)
        idx = min(idx, n - 1)
        return float(sorted_s[idx])

    @property
    def empirical_coverage(self) -> float:
        """Fraction of calibration scores ≤ last q̂_α (diagnostic)."""
        if len(self._scores) < 10:
            return float("nan")
        return float(np.mean(np.array(self._scores) <= self.quantile(0.1)))


class TemporalConformalPredictor:
    """
    Wraps any base quantile regression model (GATv2, LTC, etc.) and
    adds calibrated conformal prediction intervals per asset.

    The base_model must expose:
        model(x) → (q_low: Tensor[N], q_high: Tensor[N])
    where q_low, q_high are the α/2 and 1-α/2 quantile forecasts.

    Parameters
    ----------
    base_model          : quantile-regression callable
    n_assets            : N — universe size
    alpha               : miscoverage rate (0.10 → 90% intervals)
    width_threshold     : if interval width > this, mask the asset
                          Can be a float (uniform) or (N,) per-asset array.
    calibration_window  : max rolling calibration buffer size
    decay               : recency weight decay ρ ∈ (0,1)
    cash_asset_idx      : index of BIL/SHV in the universe (masked → reallocate here)
    """

    def __init__(
        self,
        base_model: Callable,
        n_assets: int,
        alpha: float = 0.10,
        width_threshold: float | np.ndarray = 0.08,   # 8% annualized return width
        calibration_window: int = 252,
        decay: float = 0.98,
        cash_asset_idx: int = -1,
    ) -> None:
        self.model             = base_model
        self.N                 = n_assets
        self.alpha             = alpha
        self.decay             = decay
        self.cash_idx          = cash_asset_idx % n_assets

        # Per-asset thresholds (uniform if scalar provided)
        if np.isscalar(width_threshold):
            self.width_threshold = np.full(n_assets, float(width_threshold))
        else:
            self.width_threshold = np.asarray(width_threshold, dtype=float)

        # One calibration buffer per asset
        self._buffers: List[RollingCalibrationBuffer] = [
            RollingCalibrationBuffer(calibration_window, decay)
            for _ in range(n_assets)
        ]

    # ── Calibration update ─────────────────────────────────────────────────

    def calibrate(
        self,
        x_cal: torch.Tensor,     # (T, ...) calibration features
        y_cal: np.ndarray,        # (T, N) realized returns
    ) -> None:
        """
        Compute nonconformity scores on calibration set and push to buffers.
        Called once per walk-forward fold before live prediction.
        CAUTION: x_cal must be strictly before any in-sample training period.
        """
        self.model.eval()
        with torch.no_grad():
            q_low, q_high = self.model(x_cal)                        # each (T, N)

        q_low  = q_low.cpu().numpy()
        q_high = q_high.cpu().numpy()
        y_cal  = np.asarray(y_cal, dtype=float)

        # R_i = max(q̂_low_i - y_i, y_i - q̂_high_i) per asset per timestep
        scores = np.maximum(q_low - y_cal, y_cal - q_high)           # (T, N)

        for j in range(self.N):
            self._buffers[j].push_batch(scores[:, j])

        log.info(
            "TCP calibrated: T=%d, mean_score=%.4f±%.4f",
            len(y_cal), scores.mean(), scores.std()
        )

    def update_online(
        self,
        q_low_t: np.ndarray,   # (N,) predicted quantiles at t
        q_high_t: np.ndarray,  # (N,)
        y_t: np.ndarray,       # (N,) realized return at t (for online update)
    ) -> None:
        """Online score update — call after each day's realized return is known."""
        scores = np.maximum(q_low_t - y_t, y_t - q_high_t)
        for j in range(self.N):
            self._buffers[j].push(float(scores[j]))

    # ── Prediction ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, x_t: torch.Tensor) -> ConformalResult:
        """
        Produce conformal-adjusted prediction intervals and masked asset set.

        x_t : (1, ...) or (...) feature tensor for current timestep
        """
        self.model.eval()
        q_low_t, q_high_t = self.model(x_t)                         # each (N,) or (1,N)
        q_low_t  = q_low_t.squeeze().cpu().numpy()
        q_high_t = q_high_t.squeeze().cpu().numpy()

        # Per-asset calibration corrections
        corrections = np.array([
            buf.quantile(self.alpha) for buf in self._buffers
        ])                                                            # (N,)

        # Calibrated intervals
        lower  = q_low_t  - corrections
        upper  = q_high_t + corrections
        widths = upper - lower                                        # (N,)

        # Mask assets where interval is too wide (epistemic uncertainty > threshold)
        masked = [
            j for j in range(self.N)
            if widths[j] > self.width_threshold[j]
        ]

        if masked:
            log.info("TCP masking %d assets: %s", len(masked), masked)

        avg_coverage = float(np.mean([
            buf.empirical_coverage for buf in self._buffers
        ]))

        return ConformalResult(
            lower               = lower,
            upper               = upper,
            interval_widths     = widths,
            masked_assets       = masked,
            calibration_coverage = avg_coverage,
            q_correction        = float(np.mean(corrections)),
        )

    def apply_masking(
        self,
        alpha_weights: np.ndarray,       # (N,) proposed alpha/weight vector
        masked_assets: List[int],
    ) -> np.ndarray:
        """
        Zero-out masked asset weights and reallocate mass to cash equivalent.
        Preserves total exposure.
        """
        w = alpha_weights.copy()
        if not masked_assets:
            return w

        freed_mass = w[masked_assets].sum()
        w[masked_assets] = 0.0
        w[self.cash_idx] += freed_mass                                # reallocate to BIL

        # Renormalize if cash would exceed 1.0 (shouldn't happen but defensive)
        total = w.sum()
        if total > 1e-8:
            w /= total

        return w

    def adjust_width_threshold(self, regime_vol: float, base_vol: float = 0.15) -> None:
        """
        Dynamically tighten/loosen mask threshold with realized vol regime.
        High-vol regime → tighter threshold (more masking).
        """
        scale = np.clip(regime_vol / base_vol, 0.5, 3.0)
        self.width_threshold = self.width_threshold / scale
        log.debug("TCP threshold rescaled by %.2f (regime_vol=%.2f)", scale, regime_vol)