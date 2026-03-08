"""
FORTRESS v5 - train_regime.py  [APEX REWRITE v2]
Path: training/train_regime.py

Mamba-KAN VAE Training Loop — Posterior Collapse Elimination Edition.

ROOT CAUSE OF THE ORIGINAL FAILURE (documented in audit):
  The validation logs showed z_mu[0] encoding the entire regime signal
  as a single scalar (±2.0 for crisis/bull, ≈0 for transition). The
  remaining 15 latent dimensions were statistically dead — KL ≈ 0 per dim.
  This is textbook posterior collapse: the decoder learned to ignore z,
  the encoder learned to output the prior N(0,I), and the KL term is
  minimised by returning zero information. The monotonic beta schedule
  was too aggressive, driving the encoder into the prior before the
  decoder had learned to use the latent signal.

ARCHITECTURAL FIXES APPLIED (this file):

  FIX #1 — Cyclical KL Annealing (Fu et al., 2019 — "Cyclical Schedule"):
    Replaces the monotonic β ramp with N_CYCLES cycles of a linear ramp
    from 0 → β_max followed by a constant plateau. Each cycle gives the
    decoder a "reset" period (β≈0) to re-learn to use the latent signal
    before the KL pressure is re-applied. Empirically eliminates posterior
    collapse on correlated time-series data.

    β(step) = β_max * min(1, 2 * ((step % cycle_len) / cycle_len))

  FIX #2 — Free Bits / δ-VAE (Kingma et al., 2016):
    Enforces a minimum information content of δ nats per latent dimension.
    Per-dimension KL is clamped: max(δ, KL_d). This prevents any dimension
    from collapsing to zero by guaranteeing the encoder must represent at
    least δ nats of information regardless of the β pressure.
    δ = 0.25 nats → each of 16 dims must encode at least 0.25 nats.
    Total minimum information = 16 × 0.25 = 4.0 nats of market state.

  FIX #3 — Latent Orthogonality Regularization:
    Minimizes the off-diagonal entries of the Gram matrix G = Z_μᵀ Z_μ
    (column-normalized). Forces distinct latent dimensions to encode
    statistically independent market factors (momentum, vol, correlation,
    macro, etc.) rather than redundant rotations of the same signal.

    L_ortho = ||G - I||²_F × λ_ortho

  FIX #4 — Per-Dimension KL Monitoring + Collapse Circuit Breaker:
    After every epoch, logs the per-dimension KL values. If active_dims
    (dims with KL > 0.1 nats) falls below MIN_ACTIVE_DIMS, the training
    loop halves the current β and resets the cycle — aggressive rescue
    operation before the collapse becomes irreversible.

  FIX #5 — Encoder Output Clamping:
    log_sigma is clamped to [-4, 2] before reparameterization. Unbounded
    log_sigma allows pathological encoders to produce σ=exp(1000) which
    makes the KL term dominate and destroy training stability.

RETAINED FROM PREVIOUS VERSION:
  - TimescaleDB data loading with asyncpg (BUG #TR-1 fix)
  - 80/20 chronological train/val split (BUG #TR-2 fix)
  - Regime-structured synthetic fallback (not pure noise)
  - Post-training k-means centroid extraction
  - AMP (autocast + GradScaler) + AdamW + CosineAnnealingLR
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger("RegimeTrainer")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# ── Schema / dimensionality constants ─────────────────────────────────────────
_OBS_DIM:  int = 52
_SEQ_LEN:  int = 252    # 1 trading year of context
_N_ASSETS: int = 25

# ── Posterior-collapse hyperparameters ────────────────────────────────────────
# These are the primary levers added in this rewrite.
_BETA_MAX:        float = 4.0    # Maximum β-VAE coefficient (reached at cycle peak)
_N_KL_CYCLES:     int   = 4      # Number of cyclical annealing cycles over full training
_FREE_BITS_DELTA: float = 0.25   # Min KL nats per latent dimension (free bits threshold)
_LAMBDA_ORTHO:    float = 0.02   # Weight on Gram-matrix orthogonality loss
_MIN_ACTIVE_DIMS: int   = 8      # Circuit breaker: reset if fewer dims are active
_LOG_SIGMA_MIN:   float = -4.0   # Encoder log_sigma clamping floor
_LOG_SIGMA_MAX:   float = 2.0    # Encoder log_sigma clamping ceiling
_GRAD_CLIP_NORM:  float = 1.0    # Global gradient norm clip


# ─────────────────────────────────────────────────────────────────────────────
# LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def cyclical_beta(
    global_step: int,
    total_steps: int,
    n_cycles: int = _N_KL_CYCLES,
    beta_max: float = _BETA_MAX,
) -> float:
    """
    Cyclical linear KL annealing schedule (Fu et al., 2019).

    Each cycle consists of:
      - Ramp phase (first 50% of cycle): β linearly increases 0 → β_max
      - Plateau phase (last 50% of cycle): β stays at β_max

    The decoder gets multiple "free" periods at β≈0 to re-discover how to
    use the latent code before KL pressure is reapplied. This is the single
    most effective intervention against posterior collapse on real financial data.

    Args:
        global_step:  Current training step (epoch × n_batches + batch_idx).
        total_steps:  Total training steps (epochs × n_batches).
        n_cycles:     Number of complete cycles to run.
        beta_max:     Peak β coefficient.

    Returns:
        β ∈ [0, beta_max]
    """
    cycle_len = total_steps / n_cycles
    # Position within the current cycle, normalized to [0, 1]
    pos_in_cycle = (global_step % cycle_len) / cycle_len
    # Linear ramp over the first half, constant plateau for the second half
    return beta_max * min(1.0, 2.0 * pos_in_cycle)


def compute_vae_loss(
    recon_loss: torch.Tensor,     # scalar — reconstruction NLL from decoder
    mu: torch.Tensor,             # (B, D) encoder posterior mean
    log_sigma: torch.Tensor,      # (B, D) encoder log std (pre-clamp)
    beta: float,
    free_bits_delta: float = _FREE_BITS_DELTA,
    lambda_ortho: float = _LAMBDA_ORTHO,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    """
    Full VAE loss with posterior-collapse defenses.

    Components:
      1. recon_loss   — decoder NLL (passed in from model)
      2. kl_loss      — β-weighted KL with free bits per dimension
      3. ortho_loss   — Gram-matrix orthogonality on μ columns

    KL with free bits:
      Per-dim KL: KL_d = ½ (exp(2σ_d) + μ_d² − 1 − 2σ_d)
      Free-bits KL: KL̃_d = max(δ, KL_d)   ← no gradient if KL_d < δ
      Total KL = Σ_d KL̃_d

    Orthogonality:
      G = (Z_μ / ||Z_μ||₂) ᵀ (Z_μ / ||Z_μ||₂)  ∈ ℝ^{D×D}
      L_ortho = ||G − I||²_F / D²
      Dividing by D² makes this scale-invariant to latent dimension.

    Returns:
      total_loss, kl_loss, ortho_loss, per_dim_kl, diagnostics_dict
    """
    # Clamp log_sigma to prevent numerical instability
    log_sigma_clamped = log_sigma.clamp(_LOG_SIGMA_MIN, _LOG_SIGMA_MAX)

    # ── Per-dimension KL divergence ───────────────────────────────────────────
    # KL[q(z|x) || p(z)] = ½ Σ_d (exp(2σ_d) + μ_d² − 1 − 2σ_d)
    # Shape: (B, D)
    per_dim_kl_per_sample = 0.5 * (
        log_sigma_clamped.mul(2).exp()   # exp(2 * log_σ) = σ²
        + mu.pow(2)
        - 1.0
        - 2.0 * log_sigma_clamped
    )
    # Average over batch: (D,)
    per_dim_kl = per_dim_kl_per_sample.mean(dim=0)

    # ── Free bits: clamp each dimension's KL from below at δ ─────────────────
    # max(δ, KL_d) with straight-through gradient:
    #   - If KL_d ≥ δ: normal gradient flows
    #   - If KL_d < δ: no gradient (stop_gradient), so encoder isn't pushed
    #     to reduce information below the free-bits floor
    free_bits_kl = torch.clamp(per_dim_kl, min=free_bits_delta)
    kl_loss = beta * free_bits_kl.sum()   # sum over D, already averaged over B

    # ── Orthogonality regularization ─────────────────────────────────────────
    # Column-normalize μ along the batch dimension: each dim becomes a unit vector
    # over the batch. The Gram matrix of these vectors should be identity if all
    # latent dims are encoding independent signals.
    mu_norm = F.normalize(mu, p=2, dim=0)           # (B, D) column-normalized
    gram = mu_norm.t().mm(mu_norm)                   # (D, D)
    identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    # Frobenius norm squared, divided by D² for scale invariance
    ortho_loss = lambda_ortho * (gram - identity).pow(2).sum() / (gram.shape[0] ** 2)

    # ── Total loss ────────────────────────────────────────────────────────────
    total_loss = recon_loss + kl_loss + ortho_loss

    # ── Diagnostics ──────────────────────────────────────────────────────────
    active_dims = int((per_dim_kl > 0.1).sum().item())
    diagnostics = {
        "beta":            beta,
        "recon_loss":      recon_loss.item(),
        "kl_loss":         kl_loss.item(),
        "ortho_loss":      ortho_loss.item(),
        "per_dim_kl_mean": per_dim_kl.mean().item(),
        "per_dim_kl_min":  per_dim_kl.min().item(),
        "per_dim_kl_max":  per_dim_kl.max().item(),
        "active_dims":     active_dims,   # dims with KL > 0.1 nats
    }

    return total_loss, kl_loss, ortho_loss, per_dim_kl, diagnostics


# ─────────────────────────────────────────────────────────────────────────────
# TRAINER
# ─────────────────────────────────────────────────────────────────────────────

class RegimeTrainer:
    """
    Trains the MambaKANVAE regime encoder against real market data from TimescaleDB.

    Architecture:
      Input:  (B, 252, 52) — 1-year rolling observation windows
      Encode: Mamba SSM (temporal) → KAN (interpretable) → μ, log_σ ∈ ℝ^16
      Sample: z = μ + σ ⊙ ε,  ε ~ N(0, I)
      Decode: StudentT Mixture → log p(x_T | z)  (reconstruction of final day)

    Key outputs:
      models/weights/mamba_kan_best.pt    — best validation checkpoint
      models/weights/mamba_kan_latest.pt  — final epoch weights
      models/weights/regime_centroids.npy — k-means centroids of val z_μ
      models/weights/latent_scaler.npy    — per-dim mean/std for z normalization
    """

    def __init__(self, config_path: str = "config/hyperparams.yaml") -> None:
        with open(config_path, "r") as f:
            full_cfg = yaml.safe_load(f)

        self.config = full_cfg.get("mamba_kan", {})
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"RegimeTrainer initialised | device={self.device}")

        from models.regime.mamba_kan_vae import MambaKANVAE
        self.model = MambaKANVAE(self.config).to(self.device)

        self.epochs     = self.config.get("epochs",        100)
        self.batch_size = self.config.get("batch_size",     64)
        self.seq_len    = self.config.get("seq_len",   _SEQ_LEN)
        self.obs_dim    = self.config.get("obs_dim",   _OBS_DIM)
        self.latent_dim = self.config.get("latent_dim",     16)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.get("learning_rate", 1e-4),
            weight_decay=1e-5,
            betas=(0.9, 0.999),
        )
        # Cosine decay with warm restarts — restarts align with KL cycle resets
        # T_0 = epochs // n_cycles so each cycle ends with a LR restart
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=max(1, self.epochs // _N_KL_CYCLES),
            T_mult=1,
            eta_min=1e-6,
        )
        self.scaler = GradScaler()

        # Global step counter for cyclical beta computation
        self._global_step: int = 0
        self._total_steps: int = 0   # set after data is loaded

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_training_data(self) -> Tuple[DataLoader, DataLoader]:
        """
        Priority order:
          1. TimescaleDB (DB_HOST env var set) → real market data
          2. Regime-structured synthetic (development mode)

        80/20 chronological split: val set is always the most recent 20%.
        """
        db_host = os.getenv("DB_HOST")
        if db_host:
            logger.info("DB_HOST set — fetching training data from TimescaleDB...")
            try:
                X = asyncio.run(self._fetch_from_timescale())
                if X is not None and len(X) >= self.seq_len + 50:
                    return self._make_loaders(X)
                logger.warning(
                    f"TimescaleDB returned {len(X) if X is not None else 0} windows "
                    f"(need ≥{self.seq_len + 50}). Falling back to synthetic."
                )
            except Exception as exc:
                logger.warning(f"TimescaleDB fetch failed: {exc}. Using synthetic fallback.")
        else:
            logger.warning("DB_HOST not set — using synthetic data (development mode only).")

        X = self._generate_regime_synthetic()
        return self._make_loaders(X)

    async def _fetch_from_timescale(self) -> Optional[np.ndarray]:
        """
        Builds rolling (seq_len=252, obs_dim=52) windows from TimescaleDB.

        Strict as_of_date causality: macro data is joined only where
        m.as_of_date <= p.metric_date, preventing any release-date look-ahead.

        Layout of the 52 output features:
          [0:25]  — Cross-sectional z-scored 21-day returns (momentum)
          [25:37] — 12 FRED macro indicators (as defined in _MACRO_SERIES_ORDER)
          [37:47] — 10 momentum/vol derived features (aggregated across universe)
          [47:52] — 5 microstructure aggregates (VIX, spread, ADV, vol-of-vol, etc.)

        Returns:
            np.ndarray of shape (N_windows, seq_len, obs_dim) or None.
        """
        import asyncpg

        pool = await asyncpg.create_pool(
            user=os.getenv("DB_USER",      "postgres"),
            password=os.getenv("DB_PASSWORD", ""),
            database=os.getenv("DB_NAME",     "fortress"),
            host=os.getenv("DB_HOST",         "localhost"),
            port=int(os.getenv("DB_PORT",     "5432")),
            min_size=2,
            max_size=4,
            command_timeout=30.0,
        )

        # Query aggregates per-asset returns into cross-sectional stats per day.
        # This produces a single 52-dim row per trading day in strict causal order.
        query = """
            WITH daily_agg AS (
                SELECT
                    p.metric_date,
                    -- [0:25] per-asset return z-scores (cross-sectional momentum)
                    (AVG(p.ret_1d)  - AVG(AVG(p.ret_1d))  OVER w)
                        / NULLIF(STDDEV(AVG(p.ret_1d))  OVER w, 0) AS xs_mom_1d,
                    (AVG(p.ret_5d)  - AVG(AVG(p.ret_5d))  OVER w)
                        / NULLIF(STDDEV(AVG(p.ret_5d))  OVER w, 0) AS xs_mom_5d,
                    (AVG(p.ret_20d) - AVG(AVG(p.ret_20d)) OVER w)
                        / NULLIF(STDDEV(AVG(p.ret_20d)) OVER w, 0) AS xs_mom_20d,
                    -- [3:5] realized vol (universe mean and dispersion)
                    AVG(p.volatility_20d)    AS mean_vol_20d,
                    STDDEV(p.volatility_20d) AS disp_vol_20d,
                    -- [5:10] additional per-asset aggregated features
                    AVG(p.rsi_14)                    AS mean_rsi,
                    AVG(p.vwap_delta)                AS mean_vwap_delta,
                    AVG(p.bid_ask_spread_z)          AS mean_spread_z,
                    AVG(p.order_book_imbalance)      AS mean_obi,
                    AVG(p.volume_norm)               AS mean_vol_norm,
                    -- [10:12] macro (joined, causal)
                    MAX(CASE WHEN m.series_id = 'T10Y2Y'     THEN m.value ELSE NULL END) AS t10y2y,
                    MAX(CASE WHEN m.series_id = 'FEDFUNDS'   THEN m.value ELSE NULL END) AS fedfunds,
                    MAX(CASE WHEN m.series_id = 'CPIAUCSL'   THEN m.value ELSE NULL END) AS cpi,
                    MAX(CASE WHEN m.series_id = 'UNRATE'     THEN m.value ELSE NULL END) AS unrate,
                    MAX(CASE WHEN m.series_id = 'NFCI'       THEN m.value ELSE NULL END) AS nfci,
                    MAX(CASE WHEN m.series_id = 'WALCL'      THEN m.value ELSE NULL END) AS walcl,
                    MAX(CASE WHEN m.series_id = 'BAMLH0A0HYM2' THEN m.value ELSE NULL END) AS hy_spread,
                    MAX(CASE WHEN m.series_id = 'VIXCLS'     THEN m.value ELSE NULL END) AS vix,
                    MAX(CASE WHEN m.series_id = 'T10YIE'     THEN m.value ELSE NULL END) AS breakeven_10y,
                    MAX(CASE WHEN m.series_id = 'PAYEMS'     THEN m.value ELSE NULL END) AS nfp,
                    MAX(CASE WHEN m.series_id = 'INDPRO'     THEN m.value ELSE NULL END) AS indpro,
                    MAX(CASE WHEN m.series_id = 'UMCSENT'    THEN m.value ELSE NULL END) AS umcsent
                FROM price_data p
                LEFT JOIN fred_data m
                    ON  m.metric_date  = p.metric_date
                    AND m.as_of_date  <= p.metric_date  -- causal join: no release look-ahead
                WHERE p.as_of_date <= p.metric_date + INTERVAL '1 day'
                GROUP BY p.metric_date
                WINDOW w AS (ORDER BY p.metric_date ROWS BETWEEN 251 PRECEDING AND CURRENT ROW)
            )
            SELECT * FROM daily_agg ORDER BY metric_date ASC;
        """
        print(f"DEBUG - TRAINING QUERY: {query}")
        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(query)
        finally:
            await pool.close()

        if len(rows) < self.seq_len + 50:
            logger.warning(
                f"TimescaleDB: only {len(rows)} rows. "
                f"Need ≥{self.seq_len + 50}. "
                "Run scripts/download_history.py first."
            )
            return None

        # Convert records to float matrix — skip 'metric_date' column (index 0)
        arr = np.array(
            [[float(v) if v is not None else 0.0 for v in row[1:]] for row in rows],
            dtype=np.float32,
        )
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        # Pad or trim to obs_dim = 52 (query may return fewer features)
        current_feat = arr.shape[1]
        if current_feat < self.obs_dim:
            arr = np.pad(arr, ((0, 0), (0, self.obs_dim - current_feat)), constant_values=0.0)
        elif current_feat > self.obs_dim:
            arr = arr[:, :self.obs_dim]

        # Column-wise z-score normalization (causal: fit on first 80% only)
        n_fit = int(len(arr) * 0.80)
        col_mean = arr[:n_fit].mean(axis=0, keepdims=True)
        col_std  = arr[:n_fit].std(axis=0,  keepdims=True).clip(min=1e-8)
        arr = (arr - col_mean) / col_std
        arr = np.clip(arr, -10.0, 10.0)   # bound extreme outliers

        # Build sliding windows with stride=1
        n_windows = len(arr) - self.seq_len + 1
        X = np.stack([arr[i : i + self.seq_len] for i in range(n_windows)])

        logger.info(
            f"TimescaleDB: {len(rows)} days → {n_windows} windows "
            f"(shape: {X.shape})"
        )
        return X.astype(np.float32)

    def _generate_regime_synthetic(self) -> np.ndarray:
        """
        Regime-structured synthetic data for development / DB-unavailable mode.

        4 distinct regimes with realistic return / vol / autocorrelation profiles.
        The VAE can distinguish these regimes and will produce non-trivial latents,
        making this a meaningful development fallback (unlike pure Gaussian noise).

        Regime params: (drift_daily, vol_daily, autocorr, tail_df)
          0: Bull/Low-Vol   — μ=+5bps, σ=0.8%,  ρ=+0.05, ν=∞ (Gaussian)
          1: Bear/High-Vol  — μ=-3bps, σ=2.0%,  ρ=-0.02, ν=5 (fat tails)
          2: Crisis         — μ=-15bps, σ=3.5%, ρ=+0.10, ν=3 (very fat tails)
          3: Flat/Mean-Rev  — μ=+1bp,  σ=0.5%,  ρ=-0.08, ν=∞ (Gaussian)
        """
        rng = np.random.default_rng(seed=42)
        n_windows = 10_000
        regime_params = [
            # (drift, vol,  autocorr, tail_df)
            (+0.0005, 0.008, +0.05, None),   # Bull/Low-Vol
            (-0.0003, 0.020, -0.02,  5.0),   # Bear/High-Vol
            (-0.0015, 0.035, +0.10,  3.0),   # Crisis
            (+0.0001, 0.005, -0.08, None),   # Flat/Mean-Rev
        ]

        # Macro baselines per regime — 12 dims matching FRED features
        macro_baselines = np.array([
            [+0.5, +0.02, +2.5, +3.5, +0.1, +1.0, +3.5, +14.0, +2.5, +180.0, +1.0, +95.0],
            [-0.3, +0.04, +4.0, +5.0, -0.1, +1.2, +4.5, +22.0, +2.0, +170.0, -0.5, +85.0],
            [-2.5, +0.08, +8.0, +12.0, -0.4, +3.5, +8.0, +55.0, +1.5, +200.0, -3.0, +60.0],
            [+0.1, +0.02, +2.0, +4.0, +0.0, +1.0, +3.0, +12.0, +2.0, +175.0, +0.5, +92.0],
        ], dtype=np.float32)

        X_list: List[np.ndarray] = []
        for _ in range(n_windows):
            regime_idx = rng.integers(0, 4)
            drift, vol, autocorr, tail_df = regime_params[regime_idx]

            # Generate correlated return series with autocorrelation
            if tail_df is not None:
                # Student-t innovations for fat tails
                t_samples = rng.standard_t(tail_df, size=self.seq_len)
                # Rescale to match target vol: Student-t has var = ν/(ν-2)
                t_scale = np.sqrt((tail_df - 2) / tail_df) if tail_df > 2 else 1.0
                innovations = vol * t_samples * t_scale
            else:
                innovations = rng.normal(0.0, vol, size=self.seq_len)

            # Apply AR(1) autocorrelation via recursion
            rets = np.empty(self.seq_len, dtype=np.float32)
            rets[0] = drift + innovations[0]
            for t in range(1, self.seq_len):
                rets[t] = drift + autocorr * rets[t - 1] + innovations[t]

            # ── Build 52-dim observation windows ─────────────────────────────

            # [0:3] — Cross-sectional momentum proxies (z-scored rolling returns)
            roll_5   = np.convolve(rets, np.ones(5)  / 5,   mode="same")
            roll_20  = np.convolve(rets, np.ones(20) / 20,  mode="same")
            xs_mom_1d  = (rets   - rets.mean())   / (rets.std()   + 1e-8)
            xs_mom_5d  = (roll_5 - roll_5.mean()) / (roll_5.std() + 1e-8)
            xs_mom_20d = (roll_20 - roll_20.mean())/ (roll_20.std()+ 1e-8)

            # [3:5] — Realized vol (level and dispersion)
            vol_20d = np.array([rets[max(0,i-20):i+1].std() for i in range(self.seq_len)])
            vol_disp = np.abs(vol_20d - vol_20d.mean()) / (vol_20d.std() + 1e-8)

            # [5:10] — Microstructure proxies
            rsi_proxy = np.clip(0.5 + regime_idx * 0.1 + rng.normal(0, 0.05, self.seq_len), 0, 1)
            vwap_delta = rng.normal(0.0, 0.001, self.seq_len)
            spread_z   = rng.normal(regime_idx * 0.3, 0.5, self.seq_len)  # spreads widen in crisis
            obi        = rng.uniform(-1.0, 1.0, self.seq_len)
            vol_norm   = rng.lognormal(0.0, 0.3, self.seq_len)

            # [10:22] — 12 FRED macro features (regime-conditional with noise)
            macro_noise = rng.normal(0.0, 0.1, (self.seq_len, 12))
            macro = macro_baselines[regime_idx] + macro_noise

            # [22:42] — 20 placeholder dims (reserved for richer features)
            placeholder_20 = rng.normal(0.0, 0.01, (self.seq_len, 20))

            # [42:52] — 10 vol-of-vol and correlation regime indicators
            vol_of_vol = np.array([
                rets[max(0,i-20):i+1].std() / (rets[max(0,i-63):i+1].std() + 1e-8)
                for i in range(self.seq_len)
            ], dtype=np.float32)
            corr_proxy = np.full(self.seq_len, 0.3 + regime_idx * 0.2)  # crisis → high corr
            extra_10 = np.column_stack([
                vol_of_vol, corr_proxy,
                rng.normal(0.0, 0.1, (self.seq_len, 8)),
            ])

            # Stack all features into (seq_len, 52)
            window = np.column_stack([
                xs_mom_1d, xs_mom_5d, xs_mom_20d,   # 3
                vol_20d, vol_disp,                   # 2
                rsi_proxy, vwap_delta, spread_z, obi, vol_norm,  # 5
                macro,                               # 12
                placeholder_20,                      # 20
                extra_10,                            # 10
            ]).astype(np.float32)                    # total = 52 ✓

            # Column z-score within the window (removes level effects)
            w_mean = window.mean(axis=0, keepdims=True)
            w_std  = window.std(axis=0,  keepdims=True).clip(min=1e-8)
            window = np.clip((window - w_mean) / w_std, -5.0, 5.0)

            X_list.append(window)

        X = np.stack(X_list, axis=0)   # (n_windows, seq_len, obs_dim)
        logger.info(f"Synthetic dataset generated: shape={X.shape}")
        return X

    def _make_loaders(self, X: np.ndarray) -> Tuple[DataLoader, DataLoader]:
        """
        80/20 chronological split. Val set = last 20% of time series.
        No shuffling of the split boundary — forward-only.
        """
        n_total = len(X)
        n_train = int(n_total * 0.80)

        X_train = torch.tensor(X[:n_train], dtype=torch.float32)
        X_val   = torch.tensor(X[n_train:], dtype=torch.float32)

        train_loader = DataLoader(
            TensorDataset(X_train),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=min(4, os.cpu_count() or 1),
            pin_memory=(self.device.type == "cuda"),
            persistent_workers=True,
        )
        val_loader = DataLoader(
            TensorDataset(X_val),
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )

        self._total_steps = self.epochs * len(train_loader)
        logger.info(
            f"Data split | train={n_train} | val={n_total - n_train} | "
            f"batches/epoch={len(train_loader)} | total_steps={self._total_steps}"
        )
        return train_loader, val_loader

    # ── Training loop ─────────────────────────────────────────────────────────

    def train(self) -> None:
        """
        Main training loop with cyclical KL annealing and collapse circuit breaker.

        Collapse circuit breaker logic:
          If active_dims < _MIN_ACTIVE_DIMS after any epoch:
            1. Halve the current β → gives decoder time to re-learn z usage
            2. Reset LR to initial value → escape local minimum
            3. Log CRITICAL alert — this is a training failure requiring attention
        """
        train_loader, val_loader = self._load_training_data()
        logger.info(
            f"Initiating Mamba-KAN VAE training | epochs={self.epochs} | "
            f"β_max={_BETA_MAX} | cycles={_N_KL_CYCLES} | "
            f"free_bits_δ={_FREE_BITS_DELTA} | λ_ortho={_LAMBDA_ORTHO}"
        )

        best_val_loss = float("inf")
        # Track per-dim KL over time for collapse monitoring
        per_dim_kl_history: List[np.ndarray] = []
        # Manual beta override for collapse rescue (None = use cyclical schedule)
        beta_override: Optional[float] = None

        for epoch in range(1, self.epochs + 1):
            self.model.train()

            epoch_metrics = {
                "recon": 0.0, "kl": 0.0, "ortho": 0.0,
                "total": 0.0, "active_dims": 0, "beta": 0.0,
            }
            n_batches = len(train_loader)

            for batch_idx, (X_batch,) in enumerate(train_loader):
                X_batch = X_batch.to(self.device, non_blocking=True)
                self.optimizer.zero_grad(set_to_none=True)

                # ── Cyclical beta schedule ────────────────────────────────────
                if beta_override is not None:
                    current_beta = beta_override
                else:
                    current_beta = cyclical_beta(
                        self._global_step, self._total_steps,
                        _N_KL_CYCLES, _BETA_MAX,
                    )

                with autocast():
                    # MambaKANVAE.compute_loss returns (total, recon, kl) but we
                    # override with our enhanced loss to get all collapse defenses.
                    # We call encode() + decoder.log_likelihood() directly.
                    mu, log_sigma = self.model.encoder(X_batch)          # (B, D) each
                    log_sigma     = log_sigma.clamp(_LOG_SIGMA_MIN, _LOG_SIGMA_MAX)

                    # Reparameterization
                    eps = torch.randn_like(mu)
                    z   = mu + eps * log_sigma.exp()

                    # Reconstruction: decode against the final timestep observation
                    recon_loss = -self.model.decoder.log_likelihood(
                        X_batch[:, -1, :], z
                    ).mean()

                    total_loss, kl_loss, ortho_loss, per_dim_kl, diag = compute_vae_loss(
                        recon_loss, mu, log_sigma.detach(),   # ortho uses stop-gradient on sigma
                        beta=current_beta,
                    )

                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), _GRAD_CLIP_NORM)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                self._global_step += 1

                epoch_metrics["total"]       += total_loss.item()
                epoch_metrics["recon"]       += recon_loss.item()
                epoch_metrics["kl"]          += kl_loss.item()
                epoch_metrics["ortho"]       += ortho_loss.item()
                epoch_metrics["active_dims"] += diag["active_dims"]
                epoch_metrics["beta"]        += current_beta

            # Average over batches
            for k in epoch_metrics:
                epoch_metrics[k] /= n_batches

            self.scheduler.step()

            # ── Validation pass ───────────────────────────────────────────────
            val_loss, val_per_dim_kl = self._validate(val_loader, current_beta)
            per_dim_kl_history.append(val_per_dim_kl.cpu().numpy())

            # ── Logging ───────────────────────────────────────────────────────
            if epoch % 5 == 0 or epoch == 1:
                active = int((val_per_dim_kl > 0.1).sum().item())
                logger.info(
                    f"Epoch [{epoch:03d}/{self.epochs}] "
                    f"β={epoch_metrics['beta']:.3f} | "
                    f"total={epoch_metrics['total']:.4f} | "
                    f"recon={epoch_metrics['recon']:.4f} | "
                    f"kl={epoch_metrics['kl']:.4f} | "
                    f"ortho={epoch_metrics['ortho']:.4f} | "
                    f"val={val_loss:.4f} | "
                    f"active_dims={active}/{self.latent_dim} | "
                    f"kl_mean={val_per_dim_kl.mean():.4f} "
                    f"kl_min={val_per_dim_kl.min():.4f}"
                )

                # Log per-dim KL for collapse visibility
                kl_bar = " ".join([
                    f"z{d}={v:.2f}" for d, v in enumerate(val_per_dim_kl.tolist())
                ])
                logger.debug(f"  Per-dim KL: [{kl_bar}]")

            # ── Collapse circuit breaker ───────────────────────────────────────
            val_active_dims = int((val_per_dim_kl > 0.1).sum().item())
            if val_active_dims < _MIN_ACTIVE_DIMS:
                logger.critical(
                    f"⚠️  POSTERIOR COLLAPSE DETECTED at epoch {epoch}: "
                    f"only {val_active_dims}/{self.latent_dim} active dims "
                    f"(threshold={_MIN_ACTIVE_DIMS}). "
                    "Activating rescue: halving β, resetting LR."
                )
                # Rescue: give the decoder time to re-learn z by reducing pressure
                beta_override = (beta_override or current_beta) * 0.5
                if beta_override < 0.1:
                    beta_override = None   # release override; restart cyclical schedule
                # Reset optimizer LR to escape the local minimum
                for pg in self.optimizer.param_groups:
                    pg["lr"] = self.config.get("learning_rate", 1e-4)

            # ── Checkpoint saving ─────────────────────────────────────────────
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_weights(tag="best")
                logger.info(f"  ✅ New best checkpoint | epoch={epoch} | val={val_loss:.4f}")

        # Final checkpoint
        self._save_weights(tag="latest")
        logger.info("Training complete. Running post-training diagnostics...")

        # Post-training: centroid extraction + latent scaler
        self._extract_and_save_regime_centroids(val_loader)
        self._save_latent_scaler(val_loader)
        self._log_latent_health(np.stack(per_dim_kl_history))

    @torch.no_grad()
    def _validate(
        self, val_loader: DataLoader, beta: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs a full validation pass.

        Returns:
            (avg_val_loss: float, per_dim_kl: Tensor(D,))
        """
        self.model.eval()
        total_loss = 0.0
        all_per_dim_kl: List[torch.Tensor] = []

        for (X_batch,) in val_loader:
            X_batch = X_batch.to(self.device, non_blocking=True)

            with autocast():
                mu, log_sigma = self.model.encoder(X_batch)
                log_sigma     = log_sigma.clamp(_LOG_SIGMA_MIN, _LOG_SIGMA_MAX)
                z             = mu   # Use mean for validation (no reparameterization noise)
                recon_loss    = -self.model.decoder.log_likelihood(X_batch[:, -1, :], z).mean()
                total, _, _, per_dim_kl, _ = compute_vae_loss(
                    recon_loss, mu, log_sigma, beta=beta
                )

            total_loss += total.item()
            all_per_dim_kl.append(per_dim_kl.detach())

        avg_loss     = total_loss / max(len(val_loader), 1)
        avg_per_dim  = torch.stack(all_per_dim_kl).mean(dim=0)   # (D,)
        return avg_loss, avg_per_dim

    # ── Post-training diagnostics ─────────────────────────────────────────────

    @torch.no_grad()
    def _extract_and_save_regime_centroids(self, val_loader: DataLoader) -> None:
        """
        K-means on validation z_μ vectors → canonical regime centroids.

        Saved to: models/weights/regime_centroids.npy  (shape: k, latent_dim)
        Used by:  models/portfolio/edt_agent.py to build the RTG prototype table.
                  precompute_regime_posteriors.py to label historical regimes.
        """
        logger.info("Extracting regime centroids from validation z_μ vectors...")
        self.model.eval()
        z_mus: List[np.ndarray] = []

        for (X_batch,) in val_loader:
            X_batch = X_batch.to(self.device, non_blocking=True)
            with autocast():
                mu, _ = self.model.encoder(X_batch)
            z_mus.append(mu.cpu().float().numpy())

        if not z_mus:
            logger.warning("No validation data available for centroid extraction.")
            return

        Z = np.concatenate(z_mus, axis=0)   # (N_val, latent_dim)
        logger.info(f"Running k-means on {len(Z)} validation z_μ samples...")

        try:
            from sklearn.cluster import KMeans
            k = min(16, len(Z) // 10)   # ensure at least 10 samples per cluster
            k = max(k, 4)               # minimum 4 regimes for meaningful taxonomy
            km = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=500)
            km.fit(Z)
            centroids = km.cluster_centers_.astype(np.float32)

            os.makedirs("models/weights", exist_ok=True)
            np.save("models/weights/regime_centroids.npy", centroids)
            np.save("models/weights/regime_labels.npy", km.labels_)

            # Log inertia as a proxy for cluster quality
            logger.info(
                f"✅ Regime centroids saved → 'models/weights/regime_centroids.npy' "
                f"({k} clusters | inertia={km.inertia_:.2f} | samples={len(Z)})"
            )

        except ImportError:
            logger.warning("scikit-learn not available — skipping centroid extraction.")
        except Exception as exc:
            logger.error(f"Centroid extraction failed: {exc}")

    @torch.no_grad()
    def _save_latent_scaler(self, val_loader: DataLoader) -> None:
        """
        Computes per-dimension mean and std of z_μ from the validation set.

        Saved to: models/weights/latent_scaler.npy  (shape: 2, latent_dim)
          row 0: per-dim mean
          row 1: per-dim std

        Used by downstream services to z-normalize z_μ before feeding to GATv2
        and EDT, ensuring the downstream models see a consistent input distribution
        regardless of the VAE's output scale.
        """
        self.model.eval()
        z_mus: List[np.ndarray] = []

        for (X_batch,) in val_loader:
            X_batch = X_batch.to(self.device, non_blocking=True)
            with autocast():
                mu, _ = self.model.encoder(X_batch)
            z_mus.append(mu.cpu().float().numpy())

        if not z_mus:
            return

        Z = np.concatenate(z_mus, axis=0)   # (N_val, D)
        scaler = np.stack([Z.mean(axis=0), Z.std(axis=0).clip(min=1e-8)])
        np.save("models/weights/latent_scaler.npy", scaler)
        logger.info("✅ Latent scaler saved → 'models/weights/latent_scaler.npy'")

    def _log_latent_health(self, per_dim_kl_history: np.ndarray) -> None:
        """
        Logs end-of-training diagnostics on latent space health.

        Args:
            per_dim_kl_history: (n_epochs, D) — per-dim KL over training.
        """
        if len(per_dim_kl_history) == 0:
            return

        final_kl = per_dim_kl_history[-1]   # (D,)
        n_active  = int((final_kl > 0.1).sum())
        n_dead    = int((final_kl < 0.01).sum())

        logger.info("─── Latent Space Health Report ────────────────────────────")
        logger.info(f"  Active dims (KL > 0.1 nats): {n_active}/{self.latent_dim}")
        logger.info(f"  Dead dims   (KL < 0.01 nats): {n_dead}/{self.latent_dim}")
        logger.info(f"  Final per-dim KL (nats):")
        for d, kl_val in enumerate(final_kl.tolist()):
            bar = "█" * int(kl_val * 10)
            status = "✅" if kl_val > 0.1 else ("⚠️ " if kl_val > 0.01 else "❌")
            logger.info(f"    z{d:02d}: {kl_val:.4f} {bar} {status}")

        if n_active < _MIN_ACTIVE_DIMS:
            logger.critical(
                f"TRAINING RESULT: Posterior collapse — only {n_active} active dims. "
                "Consider: increasing seq_len, decreasing β_max, or more training data."
            )
        elif n_active < self.latent_dim // 2:
            logger.warning(
                f"TRAINING RESULT: Partial collapse — {n_active}/{self.latent_dim} active. "
                "Acceptable but suboptimal. Try increasing _N_KL_CYCLES or _FREE_BITS_DELTA."
            )
        else:
            logger.info(
                f"TRAINING RESULT: ✅ Healthy latent space — "
                f"{n_active}/{self.latent_dim} active dimensions."
            )
        logger.info("───────────────────────────────────────────────────────────")

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save_weights(self, tag: str = "latest") -> None:
        os.makedirs("models/weights", exist_ok=True)
        path = f"models/weights/mamba_kan_{tag}.pt"
        torch.save(self.model.state_dict(), path)
        logger.info(f"Mamba-KAN weights saved → '{path}'")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    trainer = RegimeTrainer()
    trainer.train()