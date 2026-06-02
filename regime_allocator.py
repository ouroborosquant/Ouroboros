"""
regime_allocator.py — Fortress v5 | Regime-Split Capital Allocator
==================================================================

Architectural contract:
  - DISPERSED regime  (breadth_ratio >= CONC_THRESHOLD): route to CVaR-MVO as before
  - CONCENTRATED regime (breadth_ratio <  CONC_THRESHOLD): bypass covariance matrix entirely,
    use risk-adjusted heuristic weighting on positive-alpha assets only

Breadth diagnostic:
  breadth_ratio_t = (alpha_t > ALPHA_POSITIVE_FLOOR).sum() / N_ASSETS

  In a 25-ticker ETF universe a Mag7-style regime collapses breadth to ~5–8 tickers
  (breadth_ratio ≈ 0.20–0.32). The covariance penalty in MVO is designed to diversify;
  feeding it a concentrated signal vector is a category error, not a parameter problem.

IC-Halt gate:
  Standard Spearman/Pearson IC computed against a sparse alpha vector
  (15+ tickers near zero) produces large negative IC by construction — not signal failure.
  The IC-halt must be suppressed in concentrated regimes or you park 40% of the best
  momentum windows in BIL.

Usage (in run_standalone_backtest.py main loop):
  from regime_allocator import (
      compute_breadth_ratio,
      is_conc_regime,
      ic_halt_gated,
      allocate_conc_heuristic,
  )
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Tunable thresholds ────────────────────────────────────────────────────────

# REPLACE WITH:
CONC_BREADTH_THRESHOLD:   float = 0.35
ALPHA_POSITIVE_FLOOR:     float = 0.02
CONC_TOP_ALPHA_FLOOR:     float = 0.06   # top-end alpha must be genuinely elevated
CONC_TURNOVER_BLEND:      float = 0.35   # max fraction of portfolio to rotate per bar in CONC
CONC_MAX_WEIGHT:       float = 0.40    # hard cap per asset in CONC heuristic
CONC_MIN_ASSETS:       int   = 2       # refuse heuristic if fewer than this many positive


# ── Breadth diagnostics ───────────────────────────────────────────────────────

def compute_breadth_ratio(alpha: pd.Series | np.ndarray) -> float:
    """
    Fraction of tickers exceeding ALPHA_POSITIVE_FLOOR.

    O(N), no copy. Returns [0, 1]; <0.35 == concentrated regime.
    """
    arr = alpha.values if isinstance(alpha, pd.Series) else np.asarray(alpha)
    return float((arr > ALPHA_POSITIVE_FLOOR).mean())


def compute_hhi(alpha: pd.Series | np.ndarray) -> float:
    """
    Herfindahl-Hirschman Index on positive-alpha weights.
    HHI → 1.0 == monopolar, HHI → 1/N == maximally dispersed.
    Used for logging/diagnostics; not in the hot path.
    """
    arr = alpha.values if isinstance(alpha, pd.Series) else np.asarray(alpha)
    pos = arr[arr > 0]
    if pos.sum() < 1e-9:
        return 1.0
    w = pos / pos.sum()
    return float((w ** 2).sum())


# REPLACE WITH — gate now requires BOTH sparse AND elevated top-end alpha:
def is_conc_regime(breadth_ratio: float, top3_alpha: float = 0.0) -> bool:
    """
    True CONC: few tickers with positive alpha AND the leaders have meaningfully
    high alpha scores. Rejects bear markets where all alpha is suppressed
    (breadth is low but top alpha is also near zero).
    """
    return breadth_ratio < CONC_BREADTH_THRESHOLD and top3_alpha > CONC_TOP_ALPHA_FLOOR

# ── IC-halt gate ──────────────────────────────────────────────────────────────

def ic_halt_gated(
    rolling_ic:    float,
    ic_halt_days:  int,
    breadth_ratio: float,
    top3_alpha:    float = 0.0,  # Added to accept top performance tracking
    *,
    ic_halt_threshold: float = -0.02,
    ic_halt_window:    int   = 15,
) -> tuple[bool, str]:
    """
    Returns (should_halt: bool, reason: str).

    Gate logic:
      - If CONC regime: IC metric is structurally invalid (sparse alpha → negative IC
        by construction). Suppress halt entirely. Log the suppression.
      - If DISPERSED regime: apply standard IC-halt rule.
    """
    # FIX: Pass top3_alpha to prevent bear-market flat alpha arrays from triggering bypass
    if is_conc_regime(breadth_ratio, top3_alpha):
        # Suppress halt: IC invalidity is structural, not informational
        return False, (
            f"CONC_REGIME_BYPASS breadth={breadth_ratio:.3f}<{CONC_BREADTH_THRESHOLD} "
            f"top3_alpha={top3_alpha:.4f} — IC halt suppressed"
        )

    if rolling_ic < ic_halt_threshold and ic_halt_days >= ic_halt_window:
        return True, (
            f"IC-DECAY HALT — rolling_IC={rolling_ic:+.4f} below {ic_halt_threshold} "
            f"for {ic_halt_days}d. Moving to BIL."
        )

    return False, ""


# ── Heuristic concentrated-regime allocator ───────────────────────────────────

def allocate_conc_heuristic(
    alpha:        pd.Series | np.ndarray,
    vols:         pd.Series | np.ndarray,
    tickers:      list[str],
    prev_weights: Optional[np.ndarray] = None,   # ADD
    *,
    group_constraints: Optional[dict[str, list[str]]] = None,
    max_weight: float = CONC_MAX_WEIGHT,
) -> pd.Series:
    """
    Risk-adjusted heuristic allocator for concentrated regimes.

    Bypasses the covariance matrix entirely. MVO covariance penalty is correct
    math for diversification; it is the *wrong tool* when the signal explicitly
    demands concentration.

    Weight construction:
      1. Restrict universe to positive-alpha tickers (alpha > ALPHA_POSITIVE_FLOOR)
      2. Compute risk-adjusted scores: s_i = alpha_i / sigma_i  (Sharpe contribution proxy)
      3. Softmax-normalize scores for numerical stability under extreme skew
      4. Clip to max_weight and renormalize (preserves relative ordering)
      5. Apply optional group concentration constraints (same logic as MVO path)

    Softmax vs linear normalization:
      Linear norm collapses to near-100% in winner when one asset dominates by 3×.
      Softmax with temperature T=1 provides natural entropy regularization, keeping
      secondary positions investable. T is implicitly 1 here; could be made regime-adaptive.

    Args:
        alpha:            Raw composite alpha scores (cross-sectionally z-scored recommended)
        vols:             Rolling realized volatility per ticker (same index as alpha)
        tickers:          Ordered list matching alpha/vols arrays
        group_constraints: Optional {group_name: [ticker_list]} for sector caps;
                           applies same 0.40 group max as MVO path
        max_weight:       Per-asset hard cap (default 0.40)

    Returns:
        pd.Series indexed by tickers, weights summing to 1.0 (or 0.0 if no valid assets)
    """
    alpha_arr = alpha.values if isinstance(alpha, pd.Series) else np.asarray(alpha, dtype=float)
    vols_arr  = vols.values  if isinstance(vols,  pd.Series) else np.asarray(vols,  dtype=float)
    n         = len(tickers)

    # Step 1: restrict to positive-alpha assets
    pos_mask = alpha_arr > ALPHA_POSITIVE_FLOOR
    if pos_mask.sum() < CONC_MIN_ASSETS:
        # Degenerate: fall back to equal-weight cash equivalents
        # Caller should detect this via weights.sum() ≈ 0
        logger.warning(
            "allocate_conc_heuristic: only %d positive-alpha assets (< %d minimum). "
            "Returning zero weights — caller should route to BIL.",
            int(pos_mask.sum()), CONC_MIN_ASSETS
        )
        return pd.Series(np.zeros(n), index=tickers)

    # Step 2: risk-adjusted scores — alpha per unit vol
    safe_vols = np.where(vols_arr > 1e-6, vols_arr, 1e-6)
    scores    = np.where(pos_mask, alpha_arr / safe_vols, -np.inf)

    # Step 3: softmax normalization (temperature=1, masked)
    valid_scores  = scores[pos_mask]
    shifted       = valid_scores - valid_scores.max()   # numerical stability
    softmax_vals  = np.exp(shifted)
    softmax_vals /= softmax_vals.sum()

    weights = np.zeros(n)
    weights[pos_mask] = softmax_vals

    # Step 4: per-asset cap + renormalize
    weights = np.minimum(weights, max_weight)
    total   = weights.sum()
    if total < 1e-9:
        return pd.Series(np.zeros(n), index=tickers)
    weights /= total

    # Step 5: optional group concentration constraints
    if group_constraints:
        GROUP_CAP = 0.40
        changed = True
        for _ in range(20):     # iterative capping converges in <5 rounds typically
            changed = False
            for _grp, members in group_constraints.items():
                idxs      = [i for i, t in enumerate(tickers) if t in members]
                grp_total = weights[idxs].sum()
                if grp_total > GROUP_CAP:
                    scale             = GROUP_CAP / grp_total
                    weights[idxs]    *= scale
                    total             = weights.sum()
                    weights          /= total
                    changed           = True
            if not changed:
                break
    if prev_weights is not None and len(prev_weights) == len(tickers):
        # Constrains daily portfolio rotation to CONC_TURNOVER_BLEND fraction.
        # Without this, daily alpha/vol changes produce ~30% turnover.
        weights = CONC_TURNOVER_BLEND * weights + (1.0 - CONC_TURNOVER_BLEND) * np.clip(prev_weights, 0, 1)
        s = weights.sum()
        if s > 1e-9:
            weights /= s

    return pd.Series(weights, index=tickers)


# ── Dispatch entrypoint ───────────────────────────────────────────────────────

# REPLACE WITH:
def route_allocator(
    alpha:         pd.Series,
    vols:          pd.Series,
    tickers:       list[str],
    breadth_ratio: float,
    mvo_fn,
    prev_weights:  Optional[np.ndarray] = None,   # ADD
    *,
    group_constraints: Optional[dict[str, list[str]]] = None,
) -> tuple[pd.Series, str]:
    # Top-3 alpha: distinguishes Mag7 concentration from depressed bear markets
    alpha_arr = alpha.values
    top3_alpha = float(np.sort(alpha_arr)[-3:].mean()) if len(alpha_arr) >= 3 else 0.0

    if is_conc_regime(breadth_ratio, top3_alpha):
        w   = allocate_conc_heuristic(
                  alpha, vols, tickers,
                  prev_weights=prev_weights,
                  group_constraints=group_constraints
              )
        tag = f"HEURISTIC(breadth={breadth_ratio:.3f},top3={top3_alpha:.3f})"
    else:
        w_arr = mvo_fn(alpha.values, vols.values)
        w     = pd.Series(w_arr, index=tickers)
        tag   = f"MVO(breadth={breadth_ratio:.3f},top3={top3_alpha:.3f})"

    return w, tag
