"""
FORTRESS v5 — precompute_alpha_signals.py
Path: scripts/precompute_alpha_signals.py

Batch precomputation of per-asset expected alpha scores for the full backtest window.
Runs AFTER precompute_regime_posteriors.py. Reads cached market data and regime
posteriors, writes alpha_signals.parquet.

TWO EXECUTION MODES (auto-detected):

  A) Full Mode  — GATv2 weights at models/weights/gat_alpha_latest.pt:
       1. Loads GATv2 + CrossModalFusionNetwork from models/alpha/.
       2. Builds daily PyTorch Geometric graphs (25 nodes, 5-dim edge features).
       3. Runs batch GATv2.infer_live_alpha() per day.
       4. Writes alpha_signals.parquet.

  B) Surrogate Mode — weights unavailable:
       5-factor regime-conditioned alpha model:
         F1: 12-1 Month Cross-Sectional Momentum   (Jegadeesh & Titman 1993)
         F2: 1-Month Short-Term Reversal            (Lehmann 1990)
         F3: Low-Volatility Anomaly                 (Baker, Bradley & Wurgler 2011)
         F4: Fixed-Income Yield Carry               (applicable to TLT, IEF, SHY, LQD)
         F5: Regime Tilt                            (z_mu[0] controls equity/defensive blend)

       Regime modulation:
         λ_mom  = clip(z_mu[0] × 0.5 + 0.5, 0.1, 1.0)   # ↑ in bull → momentum
         λ_rev  = clip(-z_mu[0] × 0.3 + 0.3, 0.0, 0.6)  # ↑ in bear → mean-reversion
         λ_vol  = clip(0.3 + |z_mu[2]| × 0.3, 0.1, 0.7) # always active, scales with regime vol
         λ_tilt = z_mu[0]                                  # direct equity/defensive tilt

       Final alpha: tanh( F1·λ_mom + F2·λ_rev + F3·λ_vol + F5·λ_tilt )
       Output range matches GATv2's tanh output: [-1.0, 1.0].

Output:
  research/outputs/cache/alpha_signals.parquet
    index: date
    columns: alpha_{ticker} for each of the 25 assets
"""

from __future__ import annotations

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

_CACHE_DIR       = Path("research/outputs/cache")
_PRICES_PATH     = _CACHE_DIR / "prices_wide.parquet"
_RETURNS_PATH    = _CACHE_DIR / "returns_wide.parquet"
_REGIME_PATH     = _CACHE_DIR / "regime_posteriors.parquet"
_ALPHA_OUT       = _CACHE_DIR / "alpha_signals.parquet"
_GAT_WEIGHTS     = Path("models/weights/gat_alpha_latest.pt")

# ── Universe (25 assets, must match precompute_regime_posteriors.py) ──────────

TICKERS: List[str] = [
    "SPY", "QQQ", "IWM", "VTV",
    "XLK", "XLF", "XLV", "XLP", "XLI", "XLE",
    "EFA", "EEM",
    "TLT", "IEF", "SHY", "LQD", "HYG",
    "GLD", "SLV", "USO", "PDBC", "VNQ",
    "VIXY",
    "SHV", "BIL",
]
N_ASSETS = len(TICKERS)   # 25

# Fixed income tickers that have a "carry" signal (yield differential vs T-bills)
FIXED_INCOME_TICKERS = {"TLT", "IEF", "LQD", "HYG"}

# Defensive/safe-haven assets (positive alpha in bear/crisis regimes)
DEFENSIVE_TICKERS = {"TLT", "IEF", "SHY", "GLD", "SHV", "BIL"}

# Equity-like assets (positive alpha in bull regimes)
EQUITY_TICKERS = {
    "SPY", "QQQ", "IWM", "VTV",
    "XLK", "XLF", "XLV", "XLP", "XLI", "XLE",
    "EFA", "EEM", "VNQ", "HYG",
}

# VIXY is a special case: positive alpha in crisis, strongly negative in bull
VOLATILITY_TICKERS = {"VIXY"}

# Static carry estimates (rough annualised yield spread vs 3M T-bill at ~5%)
# Positive = yield premium, negative = yield discount
CARRY_ESTIMATES: Dict[str, float] = {
    "TLT": 0.01,   # ~1% above T-bill (duration premium - credit minimal)
    "IEF": 0.005,  # ~0.5%
    "SHY": -0.01,  # slightly below in inverted curve
    "LQD": 0.015,  # investment grade credit spread
    "HYG": 0.04,   # high yield spread
    "SHV": -0.005,
    "BIL": -0.01,
}


# ── Factor construction helpers ───────────────────────────────────────────────

def _cross_sectional_rank(arr: np.ndarray, descending: bool = True) -> np.ndarray:
    """
    Converts a vector of signals to cross-sectional percentile ranks in [-1, 1].
    NaN entries receive rank 0 (neutral). Handles ties by averaging.
    """
    valid = ~np.isnan(arr)
    ranks = np.zeros(len(arr))
    if valid.sum() < 2:
        return ranks

    n_valid = valid.sum()
    temp = np.argsort(arr[valid])
    r    = (temp.argsort() + 1).astype(float)
    # Normalise to [-1, 1]
    r    = (r - (n_valid + 1) / 2) / ((n_valid - 1) / 2 + 1e-8)
    if descending:
        r = -r
    ranks[valid] = r
    return ranks


def _build_momentum_signal(
    returns_df: pd.DataFrame,
    momentum_window: int = 252,
    skip_window:     int = 21,
) -> pd.DataFrame:
    """
    12-1 Month Cross-Sectional Momentum (Jegadeesh & Titman 1993).

    For each date t:
        momentum_i = Σ_{d=t-252}^{t-21} r_{i,d}   (log cumulative return)
    Skipping the most recent 21 days avoids the short-term reversal contamination.

    Cross-sectionally ranked to [-1, 1].
    """
    cum_long  = returns_df.rolling(momentum_window, min_periods=60).sum()
    cum_skip  = returns_df.rolling(skip_window,     min_periods=5).sum()
    momentum  = (cum_long - cum_skip).fillna(0.0)

    ranked = pd.DataFrame(index=returns_df.index, columns=TICKERS, dtype=np.float32)
    for date in returns_df.index:
        ranked.loc[date] = _cross_sectional_rank(momentum.loc[date].values, descending=False)

    return ranked


def _build_reversal_signal(
    returns_df: pd.DataFrame,
    reversal_window: int = 21,
) -> pd.DataFrame:
    """
    1-Month Short-Term Reversal (Lehmann 1990).
    Assets with high past-month returns are expected to mean-revert.
    Ranked negatively relative to momentum.
    """
    cum_ret_21 = returns_df.rolling(reversal_window, min_periods=5).sum().fillna(0.0)

    ranked = pd.DataFrame(index=returns_df.index, columns=TICKERS, dtype=np.float32)
    for date in returns_df.index:
        # Reversal = NEGATIVE rank (winners → negative alpha, losers → positive)
        ranked.loc[date] = _cross_sectional_rank(cum_ret_21.loc[date].values, descending=True)

    return ranked


def _build_vol_signal(
    returns_df: pd.DataFrame,
    vol_window: int = 63,
) -> pd.DataFrame:
    """
    Low-Volatility Anomaly (Baker, Bradley & Wurgler 2011).
    Assets with lower realised volatility tend to have higher risk-adjusted returns.
    Assets with extremely low vol (cash-like: SHV, BIL) receive neutral score
    to prevent spurious positioning in near-zero-return assets.

    Score: negatively ranked on realised vol → high vol = negative alpha.
    """
    vol_63 = returns_df.rolling(vol_window, min_periods=20).std().fillna(0.15)

    # Neutralise near-zero-vol cash instruments (avoid artefact)
    cash_mask = np.array([t in {"SHV", "BIL"} for t in TICKERS])

    ranked = pd.DataFrame(index=returns_df.index, columns=TICKERS, dtype=np.float32)
    for date in returns_df.index:
        v = vol_63.loc[date].values.copy()
        r = _cross_sectional_rank(v, descending=True)  # Low vol → high rank
        r[cash_mask] = 0.0  # Cash is neutral in vol signal
        ranked.loc[date] = r

    return ranked


def _build_carry_signal() -> Dict[str, float]:
    """
    Static carry signal for fixed income ETFs.
    Returns a dict mapping ticker → carry_score in [-1, 1].
    Non-fixed-income assets receive 0.
    """
    carry = {}
    max_carry = max(abs(v) for v in CARRY_ESTIMATES.values()) + 1e-8
    for ticker in TICKERS:
        carry[ticker] = CARRY_ESTIMATES.get(ticker, 0.0) / max_carry
    return carry


def _build_regime_tilt_signal(z_mu_0: float) -> np.ndarray:
    """
    Regime-conditioned tilt signal constructed from the first PCA component.

    z_mu[0] > 0 → bull regime → tilt toward EQUITY_TICKERS.
    z_mu[0] < 0 → bear/crisis → tilt toward DEFENSIVE_TICKERS.
    VIXY receives the inverse of the market factor (crisis hedge).

    The tilt is a smooth function of z_mu[0] to avoid cliff-edge regime switches.
    """
    tilt = np.zeros(N_ASSETS)
    for i, ticker in enumerate(TICKERS):
        if ticker in EQUITY_TICKERS:
            tilt[i] = z_mu_0 * 0.5          # Long equity in bull
        elif ticker in DEFENSIVE_TICKERS:
            tilt[i] = -z_mu_0 * 0.4         # Long defensive in bear
        elif ticker in VOLATILITY_TICKERS:
            tilt[i] = -z_mu_0 * 0.8         # VIXY is strongly inverse market
    return tilt


# ── Full-mode GATv2 inference ─────────────────────────────────────────────────

def _try_full_mode_gat(
    returns_df: pd.DataFrame,
    regime_df:  pd.DataFrame,
) -> bool:
    """
    Attempts full-mode GATv2 batch inference. Returns True on success.
    """
    if not _GAT_WEIGHTS.exists():
        logger.info(f"GATv2 weights not found at {_GAT_WEIGHTS}. Running in Surrogate Mode.")
        return False

    try:
        import torch                                          # type: ignore
        from torch_geometric.data import Data                # type: ignore
        from models.alpha.gat_alpha import GATv2AlphaEngine  # type: ignore
        from models.alpha.cross_modal_fusion import RawFeatureAssembler  # type: ignore
        import yaml                                           # type: ignore

        with open("config/hyperparams.yaml") as f:
            cfg = yaml.safe_load(f)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = GATv2AlphaEngine(cfg["gat_alpha"]).to(device)
        model.load_state_dict(torch.load(_GAT_WEIGHTS, map_location=device))
        model.eval()
        logger.info(f"✅ GATv2 weights loaded. Full Mode inference beginning...")

        # Full GATv2 inference would go here.
        # Requires building PyG graphs per date using RawFeatureAssembler.
        # Omitted — implement once model weights and DB are available.
        logger.warning("Full-mode GATv2 inference not yet wired — falling back to Surrogate Mode.")
        return False

    except Exception as exc:
        logger.warning(f"Full mode GATv2 aborted ({exc}). Running Surrogate Mode.")
        return False


# ── Surrogate alpha computation ───────────────────────────────────────────────

def _compute_surrogate_alpha(
    returns_df: pd.DataFrame,
    regime_df:  pd.DataFrame,
) -> pd.DataFrame:
    """
    Computes the 5-factor surrogate alpha matrix for all dates.

    Factor weights are regime-modulated using z_mu[0] from the PCA posteriors.
    The final signal is squashed through tanh to match GATv2's output range.

    Returns DataFrame of shape (n_dates, 25) with columns alpha_{ticker}.
    """
    logger.info("Computing Factor 1: Cross-Sectional Momentum (12-1 month)...")
    f_mom = _build_momentum_signal(returns_df)

    logger.info("Computing Factor 2: Short-Term Reversal (1 month)...")
    f_rev = _build_reversal_signal(returns_df)

    logger.info("Computing Factor 3: Low-Volatility Anomaly (63-day realised vol)...")
    f_vol = _build_vol_signal(returns_df)

    logger.info("Computing Factor 4: Fixed-Income Carry (static)...")
    carry_dict = _build_carry_signal()
    carry_vec  = np.array([carry_dict[t] for t in TICKERS], dtype=np.float32)

    logger.info("Computing Factor 5: Regime Tilt + combining all factors...")

    # Pre-parse z_mu arrays from regime_df (stored as list-of-float strings)
    def _parse_z_mu(row_val) -> np.ndarray:
        if isinstance(row_val, list):
            return np.array(row_val, dtype=np.float32)
        if isinstance(row_val, str):
            import json
            return np.array(json.loads(row_val), dtype=np.float32)
        return np.array(row_val, dtype=np.float32)

    alpha_rows: list[np.ndarray] = []
    dates = returns_df.index

    for i, date in enumerate(dates):
        # ── Retrieve regime posterior for this date ────────────────────────────
        if date in regime_df.index:
            z_mu = _parse_z_mu(regime_df.loc[date, "z_mu"])
        else:
            z_mu = np.zeros(16, dtype=np.float32)

        z_mu_0 = float(np.clip(z_mu[0], -3.0, 3.0))
        z_mu_2 = float(np.clip(z_mu[2], -3.0, 3.0))  # Vol/regime-spread component

        # ── Compute regime-conditioned factor loadings ─────────────────────────
        # λ_mom scales from 0.1 (deep crisis) to 1.0 (strong bull)
        lambda_mom  = float(np.clip(z_mu_0 * 0.4 + 0.5, 0.05, 1.0))
        # λ_rev scales inversely: high in bear/crisis
        lambda_rev  = float(np.clip(-z_mu_0 * 0.3 + 0.2, 0.0, 0.55))
        # λ_vol is always active but scales with regime vol (|z_mu[2]|)
        lambda_vol  = float(np.clip(0.25 + abs(z_mu_2) * 0.25, 0.1, 0.65))
        # λ_carry is regime-scaled: carry matters more in low-vol bull
        lambda_carry = float(np.clip(z_mu_0 * 0.2 + 0.2, 0.0, 0.4))

        # ── Fetch factor signals for this date ────────────────────────────────
        mom_t   = f_mom.loc[date].values.astype(np.float32)
        rev_t   = f_rev.loc[date].values.astype(np.float32)
        vol_t   = f_vol.loc[date].values.astype(np.float32)
        tilt_t  = _build_regime_tilt_signal(z_mu_0).astype(np.float32)

        # ── Combine factors (linear with regime-conditioned loadings) ──────────
        alpha_raw = (
            lambda_mom   * mom_t
            + lambda_rev * rev_t
            + lambda_vol * vol_t
            + lambda_carry * carry_vec
            + tilt_t
        )

        # Squash to [-1, 1] matching GATv2's tanh output head
        alpha_tanh = np.tanh(alpha_raw).astype(np.float32)

        # ── Force VIXY to negative alpha in bull, strong positive in crisis ────
        vixy_idx = TICKERS.index("VIXY")
        alpha_tanh[vixy_idx] = float(np.tanh(-z_mu_0 * 1.5))

        # ── Neutralise cash in all but crisis regime ───────────────────────────
        for cash_t in ["SHV", "BIL"]:
            idx = TICKERS.index(cash_t)
            # Mild positive alpha for cash in bear/crisis (capital preservation)
            alpha_tanh[idx] = float(np.tanh(max(-z_mu_0 * 0.3, 0.0)))

        alpha_rows.append(alpha_tanh)

        if i % 200 == 0:
            top3 = sorted(zip(TICKERS, alpha_tanh), key=lambda x: x[1], reverse=True)[:3]
            bot3 = sorted(zip(TICKERS, alpha_tanh), key=lambda x: x[1])[:3]
            logger.info(
                f"  [{i}/{len(dates)}] {date.date()} | "
                f"z_mu[0]={z_mu_0:.2f} | "
                f"Regime={regime_df.loc[date,'regime_label'] if date in regime_df.index else 'N/A'} | "
                f"Top: {', '.join(f'{t}({s:.2f})' for t, s in top3)} | "
                f"Bot: {', '.join(f'{t}({s:.2f})' for t, s in bot3)}"
            )

    alpha_arr = np.array(alpha_rows, dtype=np.float32)
    columns   = [f"alpha_{t}" for t in TICKERS]
    return pd.DataFrame(alpha_arr, index=dates, columns=columns)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("══════ Fortress v5 — Alpha Signal Precomputation ══════")

    # ── Validate prerequisites ────────────────────────────────────────────────
    for req_path in [_PRICES_PATH, _RETURNS_PATH, _REGIME_PATH]:
        if not req_path.exists():
            logger.error(
                f"Required cache file missing: {req_path}. "
                "Run precompute_regime_posteriors.py first."
            )
            sys.exit(1)

    # ── Load cached data ──────────────────────────────────────────────────────
    logger.info("Loading cached market data...")
    prices_df  = pd.read_parquet(_PRICES_PATH)
    returns_df = pd.read_parquet(_RETURNS_PATH)

    logger.info("Loading cached regime posteriors...")
    regime_df = pd.read_parquet(_REGIME_PATH)

    # Ensure DatetimeIndex alignment
    prices_df.index  = pd.to_datetime(prices_df.index)
    returns_df.index = pd.to_datetime(returns_df.index)
    regime_df.index  = pd.to_datetime(regime_df.index)

    logger.info(
        f"Market data: {len(returns_df)} days × {len(TICKERS)} assets | "
        f"Regime posteriors: {len(regime_df)} rows"
    )

    # ── Align date indices ────────────────────────────────────────────────────
    common_dates = returns_df.index.intersection(regime_df.index)
    if len(common_dates) < len(returns_df):
        logger.warning(
            f"Date alignment: {len(returns_df) - len(common_dates)} market days have "
            "no matching regime posterior. Restricting to common dates."
        )
    returns_aligned = returns_df.loc[common_dates]
    logger.info(f"Aligned dataset: {len(returns_aligned)} trading days")

    # ── Attempt full GATv2 mode, fall back to surrogate ───────────────────────
    if _try_full_mode_gat(returns_aligned, regime_df):
        logger.info("Full GATv2 alpha computation complete.")
        return

    # ── Surrogate mode ────────────────────────────────────────────────────────
    logger.info("Surrogate Mode: computing 5-factor regime-conditioned alpha model...")
    alpha_df = _compute_surrogate_alpha(returns_aligned, regime_df)

    # ── Validate output shape ─────────────────────────────────────────────────
    assert alpha_df.shape == (len(returns_aligned), N_ASSETS), (
        f"Alpha shape mismatch: {alpha_df.shape} != ({len(returns_aligned)}, {N_ASSETS})"
    )
    assert (alpha_df.abs() <= 1.0).all().all(), "Alpha values outside [-1, 1] — tanh failed."

    # ── Save ─────────────────────────────────────────────────────────────────
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    alpha_df.to_parquet(_ALPHA_OUT)
    logger.info(f"✅ Alpha signals saved → {_ALPHA_OUT} ({len(alpha_df)} rows × {N_ASSETS} assets)")

    # ── Summary statistics ─────────────────────────────────────────────────────
    logger.info("Alpha signal summary (time-averaged per asset):")
    mean_alpha = alpha_df.mean(axis=0).sort_values(ascending=False)
    for ticker_col, val in mean_alpha.items():
        ticker = ticker_col.replace("alpha_", "")
        bar = "█" * int(abs(val) * 20)
        sign = "+" if val >= 0 else "-"
        logger.info(f"  {ticker:6s}: {sign}{bar} ({val:+.3f})")

    logger.info("Precompute Stage 2 complete. Run run_standalone_backtest.py next.")


if __name__ == "__main__":
    main()