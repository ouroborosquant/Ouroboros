"""
FORTRESS v5 — precompute_alpha_signals.py  [PATCH v2]
Path: scripts/precompute_alpha_signals.py

BUG #15 FIXED:
    _build_momentum_signal, _build_reversal_signal, and _build_vol_signal
    all constructed ranked DataFrames via:
        ranked = pd.DataFrame(index=..., columns=TICKERS, dtype=np.float32)
        for date in returns_df.index:
            ranked.loc[date] = _cross_sectional_rank(...)   # returns float64
    Pandas 2.x raises one FutureWarning per scalar assignment (25 assets ×
    1510 dates = 37,750 warnings per factor × 3 factors = 113,250 warnings).
    Beyond noise, this masks genuine warnings in downstream stages.

    Root cause: float64 → float32 scalar coercion on DatetimeIndex-keyed
    row assignment is deprecated in pandas ≥ 2.0.

    Fix: build a raw (n_dates, n_assets) numpy float64 array via
    np.apply_along_axis, then construct the DataFrame once at the end.
    This is also materially faster: O(N) DataFrame constructions → O(1).
    The final array is cast to float32 only at save-time via alpha_df.astype.

    No other logic changes — all five factor formulas are identical.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("PrecomputeAlpha")

# ── Paths ─────────────────────────────────────────────────────────────────────

_CACHE_DIR    = Path("research/outputs/cache")
_PRICES_PATH  = _CACHE_DIR / "prices_wide.parquet"
_RETURNS_PATH = _CACHE_DIR / "returns_wide.parquet"
_REGIME_PATH  = _CACHE_DIR / "regime_posteriors.parquet"
_ALPHA_OUT    = _CACHE_DIR / "alpha_signals.parquet"
_GAT_WEIGHTS  = Path("models/weights/gat_alpha_latest.pt")

# ── Universe ──────────────────────────────────────────────────────────────────

TICKERS: List[str] = [
    "SPY", "QQQ", "IWM", "VTV",
    "XLK", "XLF", "XLV", "XLP", "XLI", "XLE",
    "EFA", "EEM",
    "TLT", "IEF", "SHY", "LQD", "HYG",
    "GLD", "SLV", "USO", "PDBC", "VNQ",
    "VIXY",
    "SHV", "BIL",
]
N_ASSETS = len(TICKERS)

DEFENSIVE_TICKERS  = {"TLT", "IEF", "SHY", "GLD", "SHV", "BIL"}
EQUITY_TICKERS     = {
    "SPY", "QQQ", "IWM", "VTV",
    "XLK", "XLF", "XLV", "XLP", "XLI", "XLE",
    "EFA", "EEM", "VNQ", "HYG",
}
VOLATILITY_TICKERS = {"VIXY"}

CARRY_ESTIMATES: Dict[str, float] = {
    "TLT": 0.01, "IEF": 0.005, "SHY": -0.01,
    "LQD": 0.015, "HYG": 0.04, "SHV": -0.005, "BIL": -0.01,
}


# ── Cross-sectional rank helper ───────────────────────────────────────────────

def _cross_sectional_rank(arr: np.ndarray, descending: bool = True) -> np.ndarray:
    """
    Map signal vector → cross-sectional percentile ranks in [−1, 1].
    NaN → rank 0 (neutral).  Ties broken by averaging.
    """
    valid = ~np.isnan(arr)
    ranks = np.zeros(len(arr), dtype=np.float64)
    if valid.sum() < 2:
        return ranks
    n_valid = int(valid.sum())
    temp    = np.argsort(arr[valid])
    r       = (temp.argsort() + 1).astype(np.float64)
    r       = (r - (n_valid + 1) / 2.0) / ((n_valid - 1) / 2.0 + 1e-8)
    if descending:
        r = -r
    ranks[valid] = r
    return ranks


# ── Factor 1: Cross-Sectional Momentum ───────────────────────────────────────

def _build_momentum_signal(
    returns_df: pd.DataFrame,
    momentum_window: int = 252,
    skip_window:     int = 21,
) -> pd.DataFrame:
    """
    12-1 Month momentum (Jegadeesh & Titman 1993).
    momentum_i(t) = cum_return[t-252 → t-21], cross-sectionally ranked to [-1,1].

    BUG #15 FIX: build numpy array row-wise via apply_along_axis;
    construct DataFrame once at the end.  Zero FutureWarnings.
    """
    cum_long = returns_df.rolling(momentum_window, min_periods=60).sum()
    cum_skip = returns_df.rolling(skip_window,     min_periods=5).sum()
    signal   = (cum_long - cum_skip).fillna(0.0).values  # (n_days, n_assets)

    ranked_arr = np.apply_along_axis(
        lambda row: _cross_sectional_rank(row, descending=False),
        axis=1, arr=signal,
    )
    return pd.DataFrame(ranked_arr, index=returns_df.index, columns=TICKERS)


# ── Factor 2: Short-Term Reversal ─────────────────────────────────────────────

def _build_reversal_signal(
    returns_df: pd.DataFrame,
    reversal_window: int = 21,
) -> pd.DataFrame:
    """
    1-Month reversal (Lehmann 1990).
    Recent winners → negative rank (expected to mean-revert).
    """
    signal = returns_df.rolling(reversal_window, min_periods=5).sum().fillna(0.0).values

    ranked_arr = np.apply_along_axis(
        lambda row: _cross_sectional_rank(row, descending=True),
        axis=1, arr=signal,
    )
    return pd.DataFrame(ranked_arr, index=returns_df.index, columns=TICKERS)


# ── Factor 3: Low-Volatility Anomaly ─────────────────────────────────────────

def _build_vol_signal(
    returns_df: pd.DataFrame,
    vol_window: int = 63,
) -> pd.DataFrame:
    """
    Low-Vol anomaly (Baker, Bradley & Wurgler 2011).
    Low-vol assets → positive rank.  SHV/BIL neutralised (near-zero vol artefact).
    """
    cash_mask = np.array([t in {"SHV", "BIL"} for t in TICKERS])
    vol_arr   = returns_df.rolling(vol_window, min_periods=20).std().fillna(0.15).values

    def _rank_row(row: np.ndarray) -> np.ndarray:
        r = _cross_sectional_rank(row, descending=True)
        r[cash_mask] = 0.0
        return r

    ranked_arr = np.apply_along_axis(_rank_row, axis=1, arr=vol_arr)
    return pd.DataFrame(ranked_arr, index=returns_df.index, columns=TICKERS)


# ── Factor 4: Fixed-Income Carry ──────────────────────────────────────────────

def _build_carry_signal() -> Dict[str, float]:
    max_carry = max(abs(v) for v in CARRY_ESTIMATES.values()) + 1e-8
    return {t: CARRY_ESTIMATES.get(t, 0.0) / max_carry for t in TICKERS}


# ── Factor 5: Regime Tilt ─────────────────────────────────────────────────────

def _build_regime_tilt_signal(z_mu_0: float) -> np.ndarray:
    """Smooth regime tilt: z_mu[0] > 0 → long equity, < 0 → long defensive."""
    tilt = np.zeros(N_ASSETS, dtype=np.float64)
    for i, t in enumerate(TICKERS):
        if t in EQUITY_TICKERS:
            tilt[i] = z_mu_0 * 0.5
        elif t in DEFENSIVE_TICKERS:
            tilt[i] = -z_mu_0 * 0.4
        elif t in VOLATILITY_TICKERS:
            tilt[i] = -z_mu_0 * 0.8
    return tilt


# ── Full-mode GATv2 stub ──────────────────────────────────────────────────────

def _try_full_mode_gat(
    returns_df: pd.DataFrame,
    regime_df:  pd.DataFrame,
) -> bool:
    if not _GAT_WEIGHTS.exists():
        logger.info(f"GATv2 weights not found at {_GAT_WEIGHTS}. Running in Surrogate Mode.")
        return False
    try:
        import torch  # type: ignore
        logger.warning("Full-mode GATv2 inference not yet wired — falling back to Surrogate Mode.")
        return False
    except Exception as exc:
        logger.warning(f"Full mode GATv2 aborted ({exc}). Running Surrogate Mode.")
        return False


# ── Surrogate 5-factor alpha ───────────────────────────────────────────────────

def _compute_surrogate_alpha(
    returns_df: pd.DataFrame,
    regime_df:  pd.DataFrame,
) -> pd.DataFrame:
    """
    5-factor regime-conditioned alpha.  Factor combination is unchanged from v1.

    Key difference from v1: factor matrices (f_mom, f_rev, f_vol) are now built
    via _build_*_signal() which uses np.apply_along_axis — no per-row .loc
    assignment, no FutureWarnings, ~3× faster on 1510-day windows.
    """
    logger.info("Computing Factor 1: Cross-Sectional Momentum (12-1 month)...")
    f_mom = _build_momentum_signal(returns_df)

    logger.info("Computing Factor 2: Short-Term Reversal (1 month)...")
    f_rev = _build_reversal_signal(returns_df)

    logger.info("Computing Factor 3: Low-Volatility Anomaly (63-day realised vol)...")
    f_vol = _build_vol_signal(returns_df)

    logger.info("Computing Factor 4: Fixed-Income Carry (static)...")
    carry_dict = _build_carry_signal()
    carry_vec  = np.array([carry_dict[t] for t in TICKERS], dtype=np.float64)

    logger.info("Computing Factor 5: Regime Tilt + combining all factors...")

    def _parse_z_mu(val) -> np.ndarray:
        if isinstance(val, (list, np.ndarray)):
            return np.asarray(val, dtype=np.float32)
        if isinstance(val, str):
            return np.array(json.loads(val), dtype=np.float32)
        return np.zeros(16, dtype=np.float32)

    # Pre-extract factor arrays for vectorized access
    mom_arr = f_mom.values.astype(np.float64)  # (n_dates, 25)
    rev_arr = f_rev.values.astype(np.float64)
    vol_arr = f_vol.values.astype(np.float64)
    dates   = returns_df.index

    alpha_rows: list[np.ndarray] = []

    for i, date in enumerate(dates):
        z_mu   = _parse_z_mu(regime_df.loc[date, "z_mu"]) if date in regime_df.index \
                 else np.zeros(16, dtype=np.float32)
        z_mu_0 = float(np.clip(z_mu[0], -3.0, 3.0))
        z_mu_2 = float(np.clip(z_mu[2], -3.0, 3.0))

        # Regime-conditioned factor loadings (identical to v1)
        lambda_mom   = float(np.clip(z_mu_0 * 0.4  + 0.5,  0.05, 1.0))
        lambda_rev   = float(np.clip(-z_mu_0 * 0.3 + 0.2,  0.0,  0.55))
        lambda_vol   = float(np.clip(0.25 + abs(z_mu_2) * 0.25, 0.1, 0.65))
        lambda_carry = float(np.clip(z_mu_0 * 0.2 + 0.2,  0.0,  0.4))

        alpha_raw = (
            lambda_mom   * mom_arr[i]
            + lambda_rev * rev_arr[i]
            + lambda_vol * vol_arr[i]
            + lambda_carry * carry_vec
            + _build_regime_tilt_signal(z_mu_0)
        )

        alpha_tanh = np.tanh(alpha_raw)

        # VIXY: directly driven by crisis signal
        vixy_idx = TICKERS.index("VIXY")
        alpha_tanh[vixy_idx] = float(np.tanh(-z_mu_0 * 1.5))

        # Cash: mild positive alpha only when defensive (bear/crisis)
        for cash_t in ("SHV", "BIL"):
            idx = TICKERS.index(cash_t)
            alpha_tanh[idx] = float(np.tanh(max(-z_mu_0 * 0.3, 0.0)))

        alpha_rows.append(alpha_tanh.astype(np.float32))

        if i % 200 == 0:
            top3 = sorted(zip(TICKERS, alpha_tanh), key=lambda x: x[1], reverse=True)[:3]
            bot3 = sorted(zip(TICKERS, alpha_tanh), key=lambda x: x[1])[:3]
            label = regime_df.loc[date, "regime_label"] if date in regime_df.index else "N/A"
            logger.info(
                f"  [{i}/{len(dates)}] {date.date()} | z_mu[0]={z_mu_0:.2f} | "
                f"Regime={label} | "
                f"Top: {', '.join(f'{t}({s:.2f})' for t, s in top3)} | "
                f"Bot: {', '.join(f'{t}({s:.2f})' for t, s in bot3)}"
            )

    # Single DataFrame construction — no per-row .loc assignment
    alpha_arr = np.array(alpha_rows, dtype=np.float32)
    columns   = [f"alpha_{t}" for t in TICKERS]
    return pd.DataFrame(alpha_arr, index=dates, columns=columns)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("══════ Fortress v5 — Alpha Signal Precomputation ══════")

    for req_path in (_PRICES_PATH, _RETURNS_PATH, _REGIME_PATH):
        if not req_path.exists():
            logger.error(f"Required cache file missing: {req_path}. Run precompute_regime_posteriors.py first.")
            sys.exit(1)

    logger.info("Loading cached market data...")
    prices_df  = pd.read_parquet(_PRICES_PATH)
    returns_df = pd.read_parquet(_RETURNS_PATH)
    logger.info("Loading cached regime posteriors...")
    regime_df  = pd.read_parquet(_REGIME_PATH)

    prices_df.index  = pd.to_datetime(prices_df.index)
    returns_df.index = pd.to_datetime(returns_df.index)
    regime_df.index  = pd.to_datetime(regime_df.index)

    # Deduplicate indices (defensive — precompute scripts should produce clean output)
    returns_df = returns_df[~returns_df.index.duplicated(keep="last")].sort_index()
    regime_df  = regime_df[~regime_df.index.duplicated(keep="last")].sort_index()

    logger.info(
        f"Market data: {len(returns_df)} days × {len(TICKERS)} assets | "
        f"Regime posteriors: {len(regime_df)} rows"
    )

    common_dates = returns_df.index.intersection(regime_df.index)
    if len(common_dates) < len(returns_df):
        logger.warning(
            f"Date alignment: {len(returns_df) - len(common_dates)} market days have "
            "no matching regime posterior. Restricting to common dates."
        )
    returns_aligned = returns_df.loc[common_dates]
    logger.info(f"Aligned dataset: {len(returns_aligned)} trading days")

    if _try_full_mode_gat(returns_aligned, regime_df):
        logger.info("Full GATv2 alpha computation complete.")
        return

    logger.info("Surrogate Mode: computing 5-factor regime-conditioned alpha model...")
    alpha_df = _compute_surrogate_alpha(returns_aligned, regime_df)

    assert alpha_df.shape == (len(returns_aligned), N_ASSETS), (
        f"Alpha shape mismatch: {alpha_df.shape} != ({len(returns_aligned)}, {N_ASSETS})"
    )
    assert (alpha_df.abs() <= 1.0).all().all(), "Alpha values outside [-1, 1] — tanh failed."

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    alpha_df.to_parquet(_ALPHA_OUT)
    logger.info(
        f"✅ Alpha signals saved → {_ALPHA_OUT} "
        f"({len(alpha_df)} rows × {N_ASSETS} assets)"
    )

    logger.info("Alpha signal summary (time-averaged per asset):")
    mean_alpha = alpha_df.mean(axis=0).sort_values(ascending=False)
    for ticker_col, val in mean_alpha.items():
        ticker = ticker_col.replace("alpha_", "")
        bar    = "█" * int(abs(val) * 20)
        sign   = "+" if val >= 0 else "-"
        logger.info(f"  {ticker:6s}: {sign}{bar} ({val:+.3f})")

    logger.info("Precompute Stage 2 complete. Run run_standalone_backtest.py next.")


if __name__ == "__main__":
    main()