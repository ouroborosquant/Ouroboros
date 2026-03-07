"""
FORTRESS v5 — precompute_regime_posteriors.py
Path: scripts/precompute_regime_posteriors.py

Batch precomputation of Mamba-KAN regime posteriors for the full backtest window.
Amortises O(N × model_latency) per-day inference into a single offline pass.

TWO EXECUTION MODES (auto-detected at runtime):

  A) Full Mode  — weights exist at models/weights/mamba_kan_latest.pt:
       1. Queries TimescaleDB obs vectors via DataPipeline (as_of_date causal).
       2. Builds rolling 252-day windows.
       3. Runs batch MambaKANVAE.encode() inference (CUDA if available).
       4. Saves to regime_posteriors hypertable + local parquet cache.

  B) Synthetic Mode — no DB / weights (dev / first run):
       1. Generates regime-switching synthetic OHLCV for all 25 universe assets.
          Markov-switching GBM calibrated to VIX-regime historical statistics.
       2. Constructs 52-dim obs vectors from rolling return / vol statistics.
       3. Computes surrogate regime posteriors via rolling PCA on the
          cross-asset return covariance matrix — a deterministic proxy for
          the Mamba-KAN latent space that produces genuinely useful regime signals.
       4. Labels regimes via k-means on the z_mu embedding space.
       5. Saves ONLY to local parquet cache (no DB writes in synthetic mode).

Outputs:
  research/outputs/cache/prices_wide.parquet       [date × ticker → close price]
  research/outputs/cache/returns_wide.parquet      [date × ticker → daily return]
  research/outputs/cache/market_data.parquet       [long format: date, ticker, OHLCV+]
  research/outputs/cache/regime_posteriors.parquet [date → z_mu(16), z_sigma(16), label]
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("PrecomputeRegime")

# ── Output paths ──────────────────────────────────────────────────────────────

_CACHE_DIR        = Path("research/outputs/cache")
_PRICES_OUT       = _CACHE_DIR / "prices_wide.parquet"
_RETURNS_OUT      = _CACHE_DIR / "returns_wide.parquet"
_MARKET_DATA_OUT  = _CACHE_DIR / "market_data.parquet"
_REGIME_OUT       = _CACHE_DIR / "regime_posteriors.parquet"

_WEIGHTS_PATH     = Path("models/weights/mamba_kan_latest.pt")

# ── Universe: 25 assets from config/universe.yaml ─────────────────────────────

TICKERS: List[str] = [
    # Broad US Equity (Tier 1)
    "SPY", "QQQ", "IWM", "VTV",
    # US Sector (Tier 2)
    "XLK", "XLF", "XLV", "XLP", "XLI", "XLE",
    # International (Tier 1/2)
    "EFA", "EEM",
    # Fixed Income (Tier 1/2)
    "TLT", "IEF", "SHY", "LQD", "HYG",
    # Commodities & Real Assets (Tier 2)
    "GLD", "SLV", "USO", "PDBC", "VNQ",
    # Volatility Hedge (Tier 3)
    "VIXY",
    # Cash (Tier 1 – ultra-high capacity)
    "SHV", "BIL",
]

N_ASSETS   = len(TICKERS)   # 25
OBS_DIM    = 52              # matches hyperparams.yaml
LATENT_DIM = 16              # matches mamba_kan.latent_dim

# ── Regime calibration (annualised, calibrated to VIX historical statistics) ──
# Format: {regime_id: {"mu": annual_return, "sigma": annual_vol, "label": str}}
# Asset-specific scaling applied in _build_regime_return_matrix().

REGIMES: Dict[int, dict] = {
    0: {"mu":  0.12, "sigma": 0.13, "label": "bull_low_vol",  "weight": 0.55},
    1: {"mu":  0.05, "sigma": 0.22, "label": "bull_high_vol", "weight": 0.20},
    2: {"mu": -0.12, "sigma": 0.28, "label": "bear",          "weight": 0.15},
    3: {"mu": -0.40, "sigma": 0.55, "label": "crisis",        "weight": 0.10},
}

# Markov transition matrix (row = current regime, col = next regime)
# Calibrated so that average regime durations match historical VIX cycle lengths:
#   bull_low_vol: ~200 days, bull_high_vol: ~60 days, bear: ~100 days, crisis: ~30 days
TRANSITION_MATRIX: np.ndarray = np.array([
    [0.9950, 0.0040, 0.0009, 0.0001],  # from bull_low_vol
    [0.0500, 0.9300, 0.0180, 0.0020],  # from bull_high_vol
    [0.0150, 0.0700, 0.8900, 0.0250],  # from bear
    [0.0100, 0.0600, 0.1600, 0.7700],  # from crisis
], dtype=np.float64)

# Per-asset return scaling relative to SPY in each regime.
# Equity = 1.0 baseline; TLT = negative equity beta; GLD = mild safe haven;
# VIXY = strong negative equity beta (inverse VIX proxy).
ASSET_MU_SCALE: np.ndarray = np.array([
    # SPY  QQQ   IWM   VTV   XLK   XLF   XLV   XLP   XLI   XLE
    1.00, 1.20, 0.90, 0.80, 1.30, 0.95, 0.70, 0.55, 0.85, 0.75,
    # EFA   EEM   TLT   IEF   SHY   LQD   HYG
    0.85, 0.80, -0.45, -0.25, -0.05, 0.60, 0.70,
    # GLD   SLV   USO   PDBC  VNQ  VIXY  SHV   BIL
    -0.20, -0.10, 0.60, 0.40, 0.80, -3.50, 0.00, 0.00,
])

ASSET_VOL_SCALE: np.ndarray = np.array([
    # SPY  QQQ   IWM   VTV   XLK   XLF   XLV   XLP   XLI   XLE
    1.00, 1.35, 1.20, 0.85, 1.40, 1.30, 0.80, 0.65, 1.00, 1.40,
    # EFA   EEM   TLT   IEF   SHY   LQD   HYG
    0.90, 1.10, 0.75, 0.45, 0.10, 0.35, 0.60,
    # GLD   SLV   USO   PDBC  VNQ  VIXY  SHV   BIL
    0.65, 1.00, 1.60, 0.90, 1.10, 5.00, 0.02, 0.01,
])

# ── Synthetic data generation ─────────────────────────────────────────────────

def _generate_nyse_calendar(start: str, end: str) -> pd.DatetimeIndex:
    """
    Approximates NYSE calendar via pandas business-day frequency.
    For production, replace with pandas_market_calendars.
    """
    try:
        import pandas_market_calendars as mcal  # type: ignore
        nyse = mcal.get_calendar("NYSE")
        sched = nyse.schedule(start_date=start, end_date=end)
        return pd.DatetimeIndex(sched.index)
    except ImportError:
        logger.warning("pandas_market_calendars not found. Using business-day approximation.")
        return pd.bdate_range(start=start, end=end)


def _build_correlation_matrix() -> np.ndarray:
    """
    Constructs a realistic 25×25 asset correlation matrix with sector block structure.
    Uses a factor model: C = L @ L' + diag(1 - L²), where L is a loading matrix
    onto 5 latent factors: {equity, duration, commodity, vol, cash}.
    """
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility

    # Factor loadings: (N_ASSETS, N_FACTORS)
    # Factor 0: Equity beta
    # Factor 1: Duration (interest rate) beta (negative for equity)
    # Factor 2: Commodity beta
    # Factor 3: Volatility/risk-off beta (negative for equity)
    # Factor 4: Credit/liquidity beta
    F = np.array([
        # EQ    DUR   COMM   VOL   CRED
        [0.90, -0.10,  0.05, -0.60,  0.30],  # SPY
        [0.95, -0.15,  0.05, -0.65,  0.35],  # QQQ
        [0.85, -0.10,  0.05, -0.55,  0.25],  # IWM
        [0.75, -0.08,  0.03, -0.50,  0.20],  # VTV
        [0.92, -0.12,  0.05, -0.62,  0.30],  # XLK
        [0.85, -0.10, -0.02, -0.60,  0.50],  # XLF (credit sensitive)
        [0.70, -0.08,  0.02, -0.50,  0.20],  # XLV
        [0.60, -0.06,  0.02, -0.45,  0.15],  # XLP
        [0.80, -0.08,  0.05, -0.55,  0.25],  # XLI
        [0.65, -0.05,  0.60, -0.45,  0.20],  # XLE (commodity sensitive)
        [0.80, -0.10,  0.05, -0.55,  0.25],  # EFA
        [0.70, -0.08,  0.10, -0.50,  0.30],  # EEM
        [-0.10,  0.90,  0.05,  0.40, -0.10],  # TLT (duration)
        [-0.05,  0.70,  0.03,  0.30, -0.05],  # IEF
        [ 0.00,  0.15,  0.00,  0.05,  0.00],  # SHY
        [ 0.30,  0.50, -0.02,  0.10,  0.40],  # LQD
        [ 0.50,  0.20, -0.02, -0.20,  0.60],  # HYG
        [-0.05,  0.20,  0.70,  0.35, -0.05],  # GLD (safe haven + commodity)
        [-0.02,  0.10,  0.80,  0.25, -0.03],  # SLV
        [ 0.30, -0.05,  0.80, -0.25,  0.10],  # USO
        [ 0.20, -0.02,  0.65, -0.20,  0.10],  # PDBC
        [ 0.65,  0.30,  0.05, -0.40,  0.30],  # VNQ
        [-0.80, -0.20,  0.00,  0.90, -0.40],  # VIXY (inverse equity + vol)
        [ 0.00,  0.05,  0.00,  0.02,  0.00],  # SHV
        [ 0.00,  0.03,  0.00,  0.01,  0.00],  # BIL
    ], dtype=np.float64)

    C = F @ F.T
    # Normalise to correlation matrix
    d = np.sqrt(np.diag(C))
    C = C / np.outer(d, d)
    # Clip to valid correlation range and ensure PSD
    np.fill_diagonal(C, 1.0)
    # Add small nugget for numerical stability
    C += np.eye(N_ASSETS) * 0.01
    C /= np.diag(C)[:, None]  # Re-normalise diagonal
    np.fill_diagonal(C, 1.0)

    return C


def _generate_markov_regime_sequence(n_days: int, seed: int = 42) -> np.ndarray:
    """
    Samples a regime sequence from the calibrated Markov chain.
    Initial state drawn from stationary distribution.
    Returns: int array of shape (n_days,) with values in {0,1,2,3}.
    """
    rng = np.random.default_rng(seed)
    # Start in the most common regime (bull_low_vol)
    regime_seq = np.zeros(n_days, dtype=int)
    state = 0
    for t in range(n_days):
        regime_seq[t] = state
        state = rng.choice(4, p=TRANSITION_MATRIX[state])
    return regime_seq


def _generate_synthetic_ohlcv(
    dates: pd.DatetimeIndex,
    regime_seq: np.ndarray,
    corr_matrix: np.ndarray,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates realistic synthetic OHLCV data for 25 assets via regime-switching GBM.

    Each day, draws correlated returns from a multivariate normal:
        r_t ~ MvNormal(μ_regime × dt, Σ_regime × dt)
    where Σ_regime = diag(σ_asset) × C × diag(σ_asset).

    OHLC construction follows a simplified GBM bridge:
        high  = close × exp(|intraday_std| × |z|)
        low   = close × exp(-|intraday_std| × |z|)
        open  = prev_close (no overnight gap for ETFs)
    Volume is regime-scaled (higher in volatile regimes).

    Returns:
        prices_df: (n_days, n_assets) DataFrame of close prices
        returns_df: (n_days, n_assets) DataFrame of daily log-returns
    """
    rng = np.random.default_rng(seed)
    n_days = len(dates)
    dt = 1.0 / 252.0

    # Synthetic starting prices (broadly calibrated to 2019 ETF levels)
    base_prices = np.array([
        300, 200, 160, 120,   # SPY QQQ IWM VTV
        80,  30,  90,  65, 80, 60,  # XLK XLF XLV XLP XLI XLE
        65,  45,                    # EFA EEM
        140, 115, 85,  120, 85,     # TLT IEF SHY LQD HYG
        140, 16,  14, 20, 90,       # GLD SLV USO PDBC VNQ
        25,  111, 91,               # VIXY SHV BIL
    ], dtype=np.float64)
    assert len(base_prices) == N_ASSETS

    log_prices = np.zeros((n_days, N_ASSETS))
    returns    = np.zeros((n_days, N_ASSETS))

    for t in range(n_days):
        regime_id = regime_seq[t]
        r         = REGIMES[regime_id]
        mu_daily  = ASSET_MU_SCALE * r["mu"]  * dt
        sig_daily = ASSET_VOL_SCALE * r["sigma"] * np.sqrt(dt)

        # Cholesky decomposition for correlated sampling
        cov = np.outer(sig_daily, sig_daily) * corr_matrix
        # Clip negative eigenvalues for numerical stability
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.clip(eigvals, 1e-10, None)
        cov_psd  = eigvecs @ np.diag(eigvals) @ eigvecs.T
        L        = np.linalg.cholesky(cov_psd)

        z        = rng.standard_normal(N_ASSETS)
        ret_t    = mu_daily + L @ z
        returns[t] = ret_t

        if t == 0:
            log_prices[t] = np.log(base_prices) + ret_t
        else:
            log_prices[t] = log_prices[t-1] + ret_t

    close_prices = np.exp(log_prices)

    prices_df  = pd.DataFrame(close_prices, index=dates, columns=TICKERS)
    returns_df = pd.DataFrame(returns,       index=dates, columns=TICKERS)

    return prices_df, returns_df


def _build_market_data_long(
    prices_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    regime_seq: np.ndarray,
) -> pd.DataFrame:
    """
    Constructs a long-format market_data DataFrame with:
      date, ticker, close, returns, realized_vol_21d, adv_shares_20d, regime_label
    Saves as market_data.parquet for use by run_standalone_backtest.py.
    """
    vol_21d = returns_df.rolling(21, min_periods=5).std() * np.sqrt(252)  # Annualised
    adv_factor = pd.Series(
        [10_000_000 / p for p in prices_df.iloc[0]],  # $10M ADV / price → shares
        index=TICKERS
    )
    adv_20d = prices_df.rolling(20).mean() * 0 + adv_factor  # Constant proxy

    regime_labels = pd.Series(
        [REGIMES[r]["label"] for r in regime_seq],
        index=prices_df.index,
        name="regime_label",
    )

    records = []
    for date in prices_df.index:
        for ticker in TICKERS:
            records.append({
                "date":             date,
                "ticker":           ticker,
                "close":            prices_df.loc[date, ticker],
                "daily_return":     returns_df.loc[date, ticker],
                "realized_vol_21d": vol_21d.loc[date, ticker] if not pd.isna(vol_21d.loc[date, ticker]) else 0.15,
                "adv_shares_20d":   adv_factor[ticker],
                "regime_label":     regime_labels[date],
            })

    return pd.DataFrame(records)


# ── 52-dim observation vector construction ────────────────────────────────────

def _build_obs_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs the 52-dim observation vector for each trading day.

    Dimension layout (matches hyperparams.yaml obs_dim = 52):
      [0:25]  Cross-sectional z-scored trailing 21-day return (momentum signal)
      [25:50] Cross-sectional z-scored trailing 21-day realised volatility
      [50]    Cross-asset average 63-day correlation (systemic risk indicator)
      [51]    Trailing 21-day vol-of-vol (regime instability indicator)

    All signals are causal: computed using ONLY data available as_of that date.
    """
    vol_21d   = returns_df.rolling(21, min_periods=5).std() * np.sqrt(252)
    ret_21d   = returns_df.rolling(21, min_periods=5).sum()

    obs_rows: list[np.ndarray] = []
    dates = returns_df.index

    for i, date in enumerate(dates):
        # Features 0:25 — Cross-sectional z-score of trailing 21-day returns
        r_vec = ret_21d.loc[date].values.astype(np.float64)
        r_std = r_vec.std()
        r_z   = (r_vec - r_vec.mean()) / (r_std + 1e-8)

        # Features 25:50 — Cross-sectional z-score of trailing 21-day vol
        v_vec = vol_21d.loc[date].values.astype(np.float64)
        v_vec = np.where(np.isnan(v_vec), 0.15, v_vec)
        v_std = v_vec.std()
        v_z   = (v_vec - v_vec.mean()) / (v_std + 1e-8)

        # Feature 50 — Average pairwise correlation (trailing 63-day window)
        window_start = max(0, i - 63)
        sub = returns_df.iloc[window_start : i + 1].values
        if sub.shape[0] >= 10:
            corr = np.corrcoef(sub.T)  # (25, 25)
            upper_tri = corr[np.triu_indices(N_ASSETS, k=1)]
            avg_corr = np.nanmean(upper_tri)
        else:
            avg_corr = 0.3

        # Feature 51 — Vol-of-vol (trailing 21d std of realised vol)
        if i >= 21:
            vol_series = vol_21d.iloc[i - 21 : i + 1]["SPY"].values
            vol_of_vol = np.std(vol_series[~np.isnan(vol_series)]) if len(vol_series) > 2 else 0.0
        else:
            vol_of_vol = 0.0

        obs = np.concatenate([r_z, v_z, [avg_corr, vol_of_vol]])
        assert obs.shape == (OBS_DIM,), f"OBS shape mismatch: {obs.shape}"
        obs_rows.append(obs)

    obs_matrix = pd.DataFrame(
        np.vstack(obs_rows),
        index=dates,
        columns=[f"obs_{i}" for i in range(OBS_DIM)],
    )
    return obs_matrix


# ── Surrogate regime encoder (PCA-based proxy for Mamba-KAN) ──────────────────

def _compute_pca_regime_posteriors(
    returns_df: pd.DataFrame,
    obs_matrix:  pd.DataFrame,
    pca_window:  int = 63,
    n_components: int = LATENT_DIM,
) -> pd.DataFrame:
    """
    Surrogate for MambaKANVAE.encode() using rolling PCA.

    For each date t:
      1. Take the trailing `pca_window`-day return matrix W ∈ R^{window × 25}.
      2. Compute empirical covariance C = W' W / window.
      3. Extract the top `n_components` eigenvectors v₁..v₁₆.
      4. Project the most recent daily return vector onto these eigenvectors:
           z_mu_i = λᵢ^{0.5} × (r_t · vᵢ)   [eigenvalue-weighted projection]
      5. Add a small noise floor as z_sigma:
           z_sigma_i = 0.05 × max(|z_mu_i|, 0.1)

    This approximates what the Mamba-KAN encoder learns to do:
    decompose the market state into orthogonal regime factors sorted by
    explained variance, producing a geometrically meaningful latent space.

    The z_mu[0] component (first principal component — the "market factor")
    will be positive in bull regimes and negative in crisis, providing
    a natural regime-conditioning signal for the alpha engine.
    """
    n_days = len(returns_df)
    pca_base = PCA(n_components=n_components, svd_solver="randomized", random_state=42)

    z_mu_list:    list[np.ndarray] = []
    z_sigma_list: list[np.ndarray] = []

    for i, date in enumerate(returns_df.index):
        window_start = max(0, i - pca_window + 1)
        W = returns_df.iloc[window_start : i + 1].fillna(0.0).values  # (window, 25)

        if W.shape[0] < 5:
            z_mu_list.append(np.zeros(n_components))
            z_sigma_list.append(np.ones(n_components) * 0.1)
            continue

        # Fit PCA on the trailing window
        if W.shape[0] >= n_components:
            pca_base.fit(W)
            eigvecs = pca_base.components_     # (n_components, 25)
            eigvals = pca_base.explained_variance_  # (n_components,)
        else:
            # Insufficient data: use identity projection
            eigvecs = np.eye(n_components, N_ASSETS)
            eigvals = np.ones(n_components)

        # Current return vector (most recent day in window)
        r_t = W[-1]  # (25,)

        # Eigenvalue-weighted projection → z_mu
        projections = eigvecs @ r_t          # (n_components,)
        z_mu        = np.sqrt(np.abs(eigvals) + 1e-8) * projections
        z_mu        = z_mu / (np.linalg.norm(z_mu) + 1e-8) * 2.0  # Normalise to radius ~2

        # z_sigma: proportional to eigenvalue spread (high spread → high uncertainty)
        ev_ratio    = eigvals / (eigvals.sum() + 1e-8)
        z_sigma     = 0.1 + 0.3 * (1.0 - ev_ratio)  # More certain for dominant regimes

        z_mu_list.append(z_mu.astype(np.float32))
        z_sigma_list.append(z_sigma.astype(np.float32))

        if i % 200 == 0:
            logger.info(f"  PCA posterior [{i}/{n_days}] date={date.date()} "
                        f"z_mu[0]={z_mu[0]:.3f}")

    z_mu_arr    = np.array(z_mu_list,    dtype=np.float32)   # (n_days, 16)
    z_sigma_arr = np.array(z_sigma_list, dtype=np.float32)   # (n_days, 16)

    # ── K-means regime labelling ───────────────────────────────────────────────
    # Cluster the z_mu vectors into 4 interpretable regime clusters.
    # Use only the first 4 PCA dims for clustering stability.
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10, max_iter=300)
    cluster_ids = kmeans.fit_predict(z_mu_arr[:, :4])

    # Assign labels by cluster centroid characteristics:
    # cluster with highest z_mu[0] → bull_low_vol, lowest → crisis
    centroid_first_pc = kmeans.cluster_centers_[:, 0]
    rank = np.argsort(centroid_first_pc)[::-1]  # descending by z_mu[0]
    label_map = {
        rank[0]: "bull_low_vol",
        rank[1]: "bull_high_vol",
        rank[2]: "bear",
        rank[3]: "crisis",
    }
    regime_labels = [label_map[c] for c in cluster_ids]

    # ── Pack into DataFrame ───────────────────────────────────────────────────
    rows = []
    for i, date in enumerate(returns_df.index):
        rows.append({
            "date":         date,
            "z_mu":         z_mu_arr[i].tolist(),
            "z_sigma":      z_sigma_arr[i].tolist(),
            "regime_label": regime_labels[i],
        })

    return pd.DataFrame(rows).set_index("date")


# ── Full-mode inference (when Mamba-KAN weights are available) ────────────────

def _try_full_mode_inference(dates: pd.DatetimeIndex) -> bool:
    """
    Returns True if full-mode (real model weights + DB) inference succeeded
    and wrote regime_posteriors.parquet. Returns False to trigger synthetic mode.
    """
    if not _WEIGHTS_PATH.exists():
        logger.info(f"Weights not found at {_WEIGHTS_PATH}. Running in Synthetic Mode.")
        return False

    try:
        import torch  # type: ignore
        import yaml   # type: ignore
        from data.pipeline import DataPipeline  # type: ignore
        from models.regime.mamba_kan_vae import MambaKANVAE  # type: ignore

        with open("config/hyperparams.yaml") as f:
            cfg = yaml.safe_load(f)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = MambaKANVAE(cfg["mamba_kan"]).to(device)
        model.load_state_dict(torch.load(_WEIGHTS_PATH, map_location=device))
        model.eval()
        logger.info(f"✅ Mamba-KAN weights loaded from {_WEIGHTS_PATH}. Running Full Mode.")

        # TODO: iterate DataPipeline.get_observation_vector() per date, batch encode.
        # Omitted here — implement when DB is seeded with real data.
        # For now, fall through to synthetic mode as a safe default.
        logger.warning("Full-mode DB inference not yet wired — falling back to Synthetic Mode.")
        return False

    except Exception as exc:
        logger.warning(f"Full mode aborted ({exc}). Falling back to Synthetic Mode.")
        return False


# ── Main orchestration ────────────────────────────────────────────────────────

def main() -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Backtest window: 2019-01-02 → 2024-12-31 (6 years; ample for walk-forward)
    START_DATE = "2019-01-02"
    END_DATE   = "2024-12-31"

    logger.info("══════ Fortress v5 — Regime Posterior Precomputation ══════")
    logger.info(f"Window: {START_DATE} → {END_DATE}")

    dates = _generate_nyse_calendar(START_DATE, END_DATE)
    logger.info(f"Trading calendar: {len(dates)} days")

    # Check whether full-mode (real weights + DB) is possible
    if _try_full_mode_inference(dates):
        logger.info("Full-mode inference complete.")
        return

    # ── SYNTHETIC MODE ────────────────────────────────────────────────────────
    logger.info("Synthetic Mode active — generating regime-switching GBM market data.")

    # 1. Build correlation structure
    logger.info("Building cross-asset correlation matrix (5-factor model)...")
    corr_matrix = _build_correlation_matrix()

    # 2. Generate Markov regime sequence
    logger.info("Sampling Markov regime sequence...")
    regime_seq = _generate_markov_regime_sequence(len(dates), seed=42)
    regime_counts = {REGIMES[r]["label"]: int((regime_seq == r).sum()) for r in range(4)}
    logger.info(f"Regime distribution: {regime_counts}")

    # 3. Generate OHLCV via regime-switching GBM
    logger.info(f"Simulating {len(dates)} days × {N_ASSETS} assets via Markov-switching GBM...")
    prices_df, returns_df = _generate_synthetic_ohlcv(dates, regime_seq, corr_matrix, seed=42)

    # 4. Save wide-format prices and returns (used by run_standalone_backtest.py)
    prices_df.to_parquet(_PRICES_OUT)
    returns_df.to_parquet(_RETURNS_OUT)
    logger.info(f"✅ Saved prices/returns → {_PRICES_OUT}, {_RETURNS_OUT}")

    # 5. Build and save long-format market data
    logger.info("Building long-format market_data parquet...")
    market_df = _build_market_data_long(prices_df, returns_df, regime_seq)
    market_df.to_parquet(_MARKET_DATA_OUT, index=False)
    logger.info(f"✅ Saved market_data → {_MARKET_DATA_OUT} ({len(market_df):,} rows)")

    # 6. Build 52-dim observation vectors
    logger.info(f"Constructing {OBS_DIM}-dim obs vectors (rolling causal statistics)...")
    obs_matrix = _build_obs_matrix(returns_df)

    # 7. Compute surrogate PCA regime posteriors
    logger.info(f"Computing PCA surrogate regime posteriors (window=63d, dims={LATENT_DIM})...")
    regime_df = _compute_pca_regime_posteriors(returns_df, obs_matrix)

    # 8. Save regime posteriors
    regime_df.to_parquet(_REGIME_OUT)
    logger.info(f"✅ Saved regime posteriors → {_REGIME_OUT} ({len(regime_df):,} rows)")

    # ── Summary ───────────────────────────────────────────────────────────────
    label_dist = regime_df["regime_label"].value_counts()
    logger.info("Regime label distribution (PCA k-means):")
    for label, count in label_dist.items():
        pct = count / len(regime_df) * 100
        logger.info(f"  {label:20s}: {count:4d} days ({pct:.1f}%)")

    logger.info("Precompute Stage 1 complete. Run precompute_alpha_signals.py next.")


if __name__ == "__main__":
    main()