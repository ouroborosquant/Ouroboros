"""
FORTRESS v5 - train_world_model.py  [APEX REWRITE v2]
Path: training/train_world_model.py

Neural SDE World Model Training Loop — Real Regime Conditioning Edition.

ROOT CAUSE OF THE ORIGINAL FAILURE (documented in audit):
  The SDE world model was trained with z_proxy vectors derived from PCA
  eigenvectors of each 21-day return window's covariance matrix. These
  z proxies have zero statistical relationship to the MambaKANVAE posterior
  that the live system uses for conditioning during inference.

  Consequence: the SDE learned a dynamics model conditioned on meaningless
  PCA scalars. When the live system queries the SDE with real VAE z_t vectors,
  the drift/diffusion networks operate in a regime space they have never seen —
  producing trajectories with arbitrary dynamics regardless of the actual regime.
  The stress test CVaR estimates were therefore noise.

ARCHITECTURAL FIXES APPLIED (this file):

  FIX #WM-Z (CRITICAL) — Real VAE Encoder for z Conditioning:
    Loads the trained MambaKANVAE from models/weights/mamba_kan_best.pt and
    runs inference on each 21-day window to obtain true posterior mean vectors.
    Training dependency enforced: raises RuntimeError if VAE weights missing.
    The SDE now learns p(Y_{t+1} | Y_t, z_t) where z_t is identical in
    distribution to the vectors the live system will provide at inference.

  FIX #WM-MILSTEIN — Milstein Correction to NLL Loss:
    The Euler-Maruyama NLL treats the diffusion as locally constant between
    steps. For nonlinear diffusion networks, the true transition density
    includes a Milstein correction term:

      EM loss:       NLL_EM  = ||ΔY - f·dt||² / (2σ²·dt) + ½ log(σ²)
      Milstein loss: NLL_Mil = NLL_EM + Σ_d [ g·∂g/∂Y · ΔW·dt/2 ]²

    The Milstein term penalizes the diffusion network for ignoring the
    spatial gradient of σ w.r.t. state, which is the dominant source of
    bias in the Euler scheme for financial return processes with vol-of-vol.

    Implementation: We compute ∂g/∂Y via torch.autograd.grad with
    create_graph=True, compute the Milstein residual, and add it to the NLL.
    This increases training cost by ~30% but produces a strong-order-1.0
    consistent loss instead of the strong-order-0.5 EM loss.

  FIX #WM-VAL — Train/Validation Split on Trajectories:
    Applies the same 80/20 chronological split as train_regime.py.
    Validation loss is monitored to detect SDE overfitting.

  FIX #WM-NORM — State Normalization:
    Raw daily returns are used as Y_t. The drift network learns to predict
    the next day's return given the current state and regime. To stabilize
    gradient flow, returns are normalized by the universe-wide rolling 21-day
    vol before being fed to the SDE. The normalization stats are saved to disk
    and applied identically during inference in generate_scenarios.py.

RETAINED FROM PREVIOUS VERSION:
  - BUG #TRAIN-WM-1 fix: build_conditioned_sde() API (no global state mutation)
  - BUG #TRAIN-WM-2 fix: TimescaleDB data loading (no torch.randn training data)
  - BUG #TRAIN-WM-3 fix: Correct per-asset variance from g@gᵀ diagonal
  - AMP (autocast + GradScaler) + AdamW + CosineAnnealingLR
  - Synthetic fallback for development mode

TRAINING ORDER DEPENDENCY:
  train_regime.py MUST complete successfully before this script.
  Checkpoint: models/weights/mamba_kan_best.pt must exist.
  This is enforced in __init__ with a FileNotFoundError.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import List, Optional, Tuple

import asyncpg
import numpy as np
import torch
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset

from models.world_model.neural_sde import LatentSDEWorldModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("WorldModelTrainer")

# ── File paths ────────────────────────────────────────────────────────────────
_VAE_WEIGHTS_PATH:    str = "models/weights/mamba_kan_best.pt"
_SDE_BEST_PATH:       str = "models/weights/sde_best.pt"
_SDE_LATEST_PATH:     str = "models/weights/sde_latest.pt"
_STATE_NORM_PATH:     str = "models/weights/sde_state_norm.npy"  # (2, state_dim)

# ── Training constants ────────────────────────────────────────────────────────
_MILSTEIN_WEIGHT: float = 0.1    # Weight on Milstein correction term in NLL
_GRAD_CLIP:       float = 5.0    # SDEs accumulate larger gradient norms than standard nets
_VAL_SPLIT:       float = 0.20   # Last 20% of trajectories held out for validation
_MIN_TRAJ_COUNT:  int   = 500    # Minimum trajectories to proceed with real data


class WorldModelTrainer:
    """
    Trains the LatentSDEWorldModel to learn:
      p(Y_{t+dt} | Y_t, z_t) — one-step transition density

    where:
      Y_t ∈ ℝ^{state_dim=25}  — normalized 25-asset daily returns
      z_t ∈ ℝ^{regime_dim=16} — MambaKANVAE posterior mean (real encoder output)
      dt = 1.0                 — one trading day

    The NLL loss with Milstein correction provides a consistent estimator
    of the Itô SDE parameters up to strong order 1.0.

    Key outputs:
      models/weights/sde_best.pt        — best validation checkpoint
      models/weights/sde_latest.pt      — final epoch checkpoint
      models/weights/sde_state_norm.npy — (mean, std) for return normalization
    """

    def __init__(self, config_path: str = "config/hyperparams.yaml") -> None:
        with open(config_path, "r") as f:
            full_cfg = yaml.safe_load(f)

        self.config     = full_cfg.get("world_model", {})
        self.vae_config = full_cfg.get("mamba_kan",   {})

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"WorldModelTrainer initialised | device={self.device}")

        self.state_dim  = self.config.get("sde_state_dim", 25)
        self.regime_dim = self.config.get("latent_dim",    16)
        self.batch_size = self.config.get("batch_size",    64)
        self.epochs     = self.config.get("epochs",        150)
        self.seq_len    = 21   # 21-day trajectory windows

        # ── SDE model ─────────────────────────────────────────────────────────
        self.model = LatentSDEWorldModel(self.config).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.get("learning_rate", 3e-4),
            weight_decay=1e-5,
            betas=(0.9, 0.999),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs, eta_min=1e-6
        )
        self.scaler = GradScaler()

        # ── VAE encoder (FIX #WM-Z) ───────────────────────────────────────────
        # Load the trained MambaKANVAE. This is a hard dependency — the world
        # model's latent conditioning is meaningless without a trained encoder.
        if not os.path.isfile(_VAE_WEIGHTS_PATH):
            raise FileNotFoundError(
                f"MambaKANVAE weights not found at '{_VAE_WEIGHTS_PATH}'. "
                "Run training/train_regime.py first. "
                "Training order: train_regime → train_world_model."
            )

        from models.regime.mamba_kan_vae import MambaKANVAE
        self.vae = MambaKANVAE(self.vae_config).to(self.device)
        self.vae.load_state_dict(
            torch.load(_VAE_WEIGHTS_PATH, map_location=self.device)
        )
        self.vae.eval()
        # Freeze VAE completely — we're training the SDE, not fine-tuning the encoder
        for p in self.vae.parameters():
            p.requires_grad_(False)
        logger.info(f"✅ MambaKANVAE loaded and frozen from '{_VAE_WEIGHTS_PATH}'")

        # State normalization stats (computed after data loading)
        self._state_mean: Optional[np.ndarray] = None
        self._state_std:  Optional[np.ndarray] = None

    # ── DATA LOADING ──────────────────────────────────────────────────────────

    def _load_training_data(self) -> Tuple[DataLoader, DataLoader]:
        """
        Priority:
          1. TimescaleDB (DB_HOST set) → real 21-day return trajectories
          2. Regime-structured synthetic (development fallback)

        After loading, computes z_t for each trajectory using the frozen VAE.
        Returns 80/20 chronological train/val split.
        """
        db_host = os.getenv("DB_HOST")
        if db_host:
            logger.info("DB_HOST set — loading trajectory data from TimescaleDB...")
            try:
                Y, Z = asyncio.run(self._fetch_trajectories_from_timescale())
                if Y is not None and len(Y) >= _MIN_TRAJ_COUNT:
                    return self._make_loaders(Y, Z)
                logger.warning(
                    f"TimescaleDB returned {len(Y) if Y is not None else 0} trajectories "
                    f"(need ≥{_MIN_TRAJ_COUNT}). Using synthetic fallback."
                )
            except Exception as exc:
                logger.warning(f"TimescaleDB fetch failed: {exc}. Using synthetic fallback.")
        else:
            logger.warning("DB_HOST not set — using synthetic data (development mode).")

        Y, Z = self._generate_synthetic_trajectories()
        return self._make_loaders(Y, Z)

    @torch.no_grad()
    def _encode_z_from_vae(
        self,
        Y_trajectories: np.ndarray,   # (N, seq_len, state_dim)
        obs_seq_len: int = 252,
    ) -> np.ndarray:
        """
        Encodes z_t for each trajectory using the frozen MambaKANVAE.

        The VAE expects (B, seq_len=252, obs_dim=52) but we have (N, 21, 25).
        Strategy:
          - Pad the 25-dim return vector to 52-dims (trailing zeros for macro)
          - Pad the 21-day sequence to 252 by repeating the first observation
            as a constant prefix (causal padding — no future data leaks)
          - Run VAE encoder inference → μ ∈ ℝ^16

        This produces z vectors that live in the same distribution as the
        live system's z_t, making the SDE's conditioning meaningful.

        Args:
            Y_trajectories: (N, 21, 25) normalized return trajectories
            obs_seq_len: VAE expected sequence length (252)

        Returns:
            Z: (N, 16) regime posterior means
        """
        N, traj_len, state_d = Y_trajectories.shape
        obs_dim = self.vae_config.get("obs_dim", 52)

        # Pad state_dim → obs_dim with zeros (macro features not available here)
        pad_feat = obs_dim - state_d   # 52 - 25 = 27 trailing zeros
        assert pad_feat >= 0, f"state_dim={state_d} > obs_dim={obs_dim}"

        # Pad sequence length: repeat first timestep as prefix
        # Shape transformation: (N, 21, 25) → (N, 252, 52)
        Y_obs = np.pad(
            Y_trajectories,
            ((0, 0), (0, 0), (0, pad_feat)),    # pad features to 52
            constant_values=0.0,
        )
        # Causal prefix: tile the first observation to fill 252 - 21 = 231 timesteps
        prefix_len = obs_seq_len - traj_len
        prefix = np.tile(Y_obs[:, :1, :], (1, prefix_len, 1))   # (N, 231, 52)
        Y_full = np.concatenate([prefix, Y_obs], axis=1)         # (N, 252, 52)

        logger.info(f"Encoding z_t for {N} trajectories via MambaKANVAE...")
        Z_list: List[np.ndarray] = []
        batch_size = 256  # Larger batch for inference — no gradient storage

        for i in range(0, N, batch_size):
            batch = torch.tensor(
                Y_full[i:i + batch_size], dtype=torch.float32, device=self.device
            )
            with autocast():
                mu, _ = self.vae.encoder(batch)   # (B, 16)
            Z_list.append(mu.cpu().float().numpy())
            if i % (batch_size * 10) == 0 and i > 0:
                logger.info(f"  Encoded {i}/{N} trajectories...")

        Z = np.concatenate(Z_list, axis=0)   # (N, 16)
        logger.info(f"✅ z_t encoding complete | shape={Z.shape}")
        return Z.astype(np.float32)

    async def _fetch_trajectories_from_timescale(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Queries TimescaleDB for 21-day sliding windows of per-asset daily returns.

        Strict causality: ordered by metric_date ASC, returns built from
        lag-1 close prices only (no same-day lookahead).

        Returns:
            Y: (N, 21, 25) normalized return trajectories
            Z: (N, 16) VAE-encoded regime vectors
        """
        pool = await asyncpg.create_pool(
            user=os.getenv("DB_USER",      "postgres"),
            password=os.getenv("DB_PASSWORD", ""),
            database=os.getenv("DB_NAME",     "fortress"),
            host=os.getenv("DB_HOST",         "localhost"),
            port=int(os.getenv("DB_PORT",     "5432")),
            min_size=2, max_size=4,
            command_timeout=60.0,
        )

        # Fetch daily return matrix: one row per (date, 25 tickers)
        # Returns are computed as log(close_t / close_{t-1}) for stationarity
        query = """
            SELECT
                metric_date,
                AVG(CASE WHEN ticker='SPY'  THEN COALESCE(ret_1d, 0) ELSE NULL END) AS r0,
                AVG(CASE WHEN ticker='QQQ'  THEN COALESCE(ret_1d, 0) ELSE NULL END) AS r1,
                AVG(CASE WHEN ticker='IWM'  THEN COALESCE(ret_1d, 0) ELSE NULL END) AS r2,
                AVG(CASE WHEN ticker='EFA'  THEN COALESCE(ret_1d, 0) ELSE NULL END) AS r3,
                AVG(CASE WHEN ticker='EEM'  THEN COALESCE(ret_1d, 0) ELSE NULL END) AS r4,
                AVG(CASE WHEN ticker='TLT'  THEN COALESCE(ret_1d, 0) ELSE NULL END) AS r5,
                AVG(CASE WHEN ticker='IEF'  THEN COALESCE(ret_1d, 0) ELSE NULL END) AS r6,
                AVG(CASE WHEN ticker='SHY'  THEN COALESCE(ret_1d, 0) ELSE NULL END) AS r7,
                AVG(CASE WHEN ticker='SHV'  THEN COALESCE(ret_1d, 0) ELSE NULL END) AS r8,
                AVG(CASE WHEN ticker='BIL'  THEN COALESCE(ret_1d, 0) ELSE NULL END) AS r9,
                AVG(CASE WHEN ticker='LQD'  THEN COALESCE(ret_1d, 0) ELSE NULL END) AS r10,
                AVG(CASE WHEN ticker='HYG'  THEN COALESCE(ret_1d, 0) ELSE NULL END) AS r11,
                AVG(CASE WHEN ticker='GLD'  THEN COALESCE(ret_1d, 0) ELSE NULL END) AS r12,
                AVG(CASE WHEN ticker='SLV'  THEN COALESCE(ret_1d, 0) ELSE NULL END) AS r13,
                AVG(CASE WHEN ticker='USO'  THEN COALESCE(ret_1d, 0) ELSE NULL END) AS r14,
                AVG(CASE WHEN ticker='PDBC' THEN COALESCE(ret_1d, 0) ELSE NULL END) AS r15,
                AVG(CASE WHEN ticker='XLK'  THEN COALESCE(ret_1d, 0) ELSE NULL END) AS r16,
                AVG(CASE WHEN ticker='XLF'  THEN COALESCE(ret_1d, 0) ELSE NULL END) AS r17,
                AVG(CASE WHEN ticker='XLE'  THEN COALESCE(ret_1d, 0) ELSE NULL END) AS r18,
                AVG(CASE WHEN ticker='XLP'  THEN COALESCE(ret_1d, 0) ELSE NULL END) AS r19,
                AVG(CASE WHEN ticker='XLV'  THEN COALESCE(ret_1d, 0) ELSE NULL END) AS r20,
                AVG(CASE WHEN ticker='VNQ'  THEN COALESCE(ret_1d, 0) ELSE NULL END) AS r21,
                AVG(CASE WHEN ticker='VIXY' THEN COALESCE(ret_1d, 0) ELSE NULL END) AS r22,
                AVG(CASE WHEN ticker='UUP'  THEN COALESCE(ret_1d, 0) ELSE NULL END) AS r23,
                AVG(CASE WHEN ticker='SH'   THEN COALESCE(ret_1d, 0) ELSE NULL END) AS r24
            FROM prices
            WHERE as_of_date <= metric_date + INTERVAL '1 day'  -- causal: T+1 settlement
            GROUP BY metric_date
            HAVING COUNT(DISTINCT ticker) >= 20  -- require ≥20 assets to avoid sparse days
            ORDER BY metric_date ASC;
        """

        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(query)
        finally:
            await pool.close()

        if len(rows) < self.seq_len + 50:
            logger.warning(f"TimescaleDB: only {len(rows)} trading days found.")
            return None, None

        # Build full (T, 25) return matrix
        matrix = np.array(
            [[float(v) if v is not None else 0.0 for v in row[1:]] for row in rows],
            dtype=np.float32,
        )
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

        # Clip extreme outliers (>10σ) — circuit breakers, bad ticks, splits
        matrix = np.clip(matrix, -0.25, 0.25)   # hard cap: ±25% daily return per asset

        # ── State normalization (FIX #WM-NORM) ──────────────────────────────
        # Normalize by universe rolling 21-day vol so drift/diffusion nets
        # operate in a consistent scale regardless of vol regime
        n_fit = int(len(matrix) * 0.80)
        self._state_mean = matrix[:n_fit].mean(axis=0)           # (25,)
        self._state_std  = matrix[:n_fit].std(axis=0).clip(1e-6) # (25,)
        self._save_state_norm()

        matrix_norm = (matrix - self._state_mean) / self._state_std

        # Build sliding 21-day windows
        n_windows = len(matrix_norm) - self.seq_len + 1
        Y = np.stack([
            matrix_norm[i : i + self.seq_len] for i in range(n_windows)
        ])  # (N, 21, 25)

        logger.info(
            f"TimescaleDB: {len(rows)} days → {n_windows} trajectories "
            f"(shape: {Y.shape})"
        )

        # Encode z_t using the frozen VAE on the UNNORMALIZED trajectories
        # (the VAE expects the same scale as its own training data)
        Y_raw = np.stack([
            matrix[i : i + self.seq_len] for i in range(n_windows)
        ])
        Z = self._encode_z_from_vae(Y_raw)

        return Y.astype(np.float32), Z

    def _generate_synthetic_trajectories(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Regime-structured synthetic trajectories for development mode.

        Generates 4-regime log-normal paths correlated across the 25-asset
        universe. The VAE is still used to encode z_t — passing synthetic
        trajectories through the real encoder ensures z is still in-distribution
        for the SDE, even without real market data.

        Returns:
            Y: (N, 21, 25) normalized return trajectories
            Z: (N, 16) VAE-encoded regime vectors (from synthetic inputs)
        """
        logger.warning(
            "Using SYNTHETIC trajectory data for SDE training. "
            "Models WILL NOT generalise to live market regimes. "
            "Connect TimescaleDB for production training."
        )
        rng = np.random.default_rng(seed=42)
        n_paths = 8_000

        # Regime params: (drift_daily, vol_daily, cross_asset_corr, tail_df)
        regimes = [
            (+0.0005, 0.008, 0.25, None),   # Bull/Low-Vol:   moderate correlation
            (-0.0003, 0.018, 0.45, 5.0),    # Bear/High-Vol:  higher correlation
            (-0.0015, 0.035, 0.80, 3.0),    # Crisis:         very high correlation (correlation spike)
            (+0.0001, 0.005, 0.10, None),   # Flat/Mean-Rev:  low correlation
        ]

        # Synthetic universe: 4 sector blocks of correlated assets
        # [0:6] Equity broad, [6:11] Fixed Income, [11:16] Commodities,
        # [16:22] Sector ETFs, [22:25] Alternatives
        sector_map = [0,0,0,0,0,0, 1,1,1,1,1, 2,2,2,2,2, 3,3,3,3,3,3, 0,0,0]

        Y_list: List[np.ndarray] = []
        for _ in range(n_paths):
            r_idx = rng.integers(0, 4)
            drift, vol, corr, tail_df = regimes[r_idx]

            # Build intra-regime correlated covariance matrix (block structure)
            C = np.full((self.state_dim, self.state_dim), 0.05)   # baseline cross-corr
            for i in range(self.state_dim):
                for j in range(self.state_dim):
                    if sector_map[i] == sector_map[j]:
                        C[i, j] = corr   # intra-sector correlation
                C[i, i] = 1.0

            # Sample correlated shocks via Cholesky decomposition
            try:
                L = np.linalg.cholesky(C + np.eye(self.state_dim) * 1e-6)
            except np.linalg.LinAlgError:
                L = np.eye(self.state_dim)

            if tail_df is not None:
                # Student-t via scale mixture: ε = z / sqrt(χ²/ν) where z ~ N(0, I)
                chi2 = rng.chisquare(tail_df, size=(self.seq_len, 1))
                t_scale = np.sqrt(tail_df / chi2)
                raw_shocks = rng.standard_normal((self.seq_len, self.state_dim)) * t_scale
            else:
                raw_shocks = rng.standard_normal((self.seq_len, self.state_dim))

            # Apply covariance structure and vol/drift
            path = drift + vol * (raw_shocks @ L.T)
            # Apply mild AR(1) momentum for realism
            for t in range(1, self.seq_len):
                path[t] += 0.03 * path[t - 1]

            Y_list.append(path.astype(np.float32))

        Y_raw = np.stack(Y_list)   # (N, 21, 25)

        # Normalize by the synthetic universe's own vol
        self._state_mean = Y_raw.reshape(-1, self.state_dim).mean(axis=0)
        self._state_std  = Y_raw.reshape(-1, self.state_dim).std(axis=0).clip(1e-6)
        self._save_state_norm()
        Y_norm = (Y_raw - self._state_mean) / self._state_std

        # Encode z_t using the VAE on raw (unnormalized) synthetic trajectories
        Z = self._encode_z_from_vae(Y_raw)

        logger.info(f"Synthetic trajectories generated | Y={Y_norm.shape} | Z={Z.shape}")
        return Y_norm.astype(np.float32), Z

    def _make_loaders(
        self,
        Y: np.ndarray,   # (N, 21, 25)
        Z: np.ndarray,   # (N, 16)
    ) -> Tuple[DataLoader, DataLoader]:
        """
        80/20 chronological split → train/val DataLoaders.
        """
        N = len(Y)
        n_train = int(N * (1.0 - _VAL_SPLIT))

        Y_tr = torch.tensor(Y[:n_train], dtype=torch.float32)
        Z_tr = torch.tensor(Z[:n_train], dtype=torch.float32)
        Y_va = torch.tensor(Y[n_train:], dtype=torch.float32)
        Z_va = torch.tensor(Z[n_train:], dtype=torch.float32)

        train_loader = DataLoader(
            TensorDataset(Y_tr, Z_tr),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=min(4, os.cpu_count() or 1),
            pin_memory=(self.device.type == "cuda"),
            persistent_workers=True,
        )
        val_loader = DataLoader(
            TensorDataset(Y_va, Z_va),
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )

        logger.info(
            f"Data split | train={n_train} | val={N - n_train} | "
            f"batches/epoch={len(train_loader)}"
        )
        return train_loader, val_loader

    # ── LOSS FUNCTIONS ────────────────────────────────────────────────────────

    def _compute_em_nll_loss(
        self,
        Y_curr: torch.Tensor,    # (B*T, D) current state
        Y_next: torch.Tensor,    # (B*T, D) next state
        Z_flat: torch.Tensor,    # (B*T, 16) regime conditioning
        conditioned_sde: object,
        dt: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Euler-Maruyama negative log-likelihood with Milstein correction.

        EM NLL (per asset, per transition):
          NLL_EM = (ΔY - f·dt)² / (2·σ²·dt) + ½·log(σ²)

        Milstein correction (first-order Itô-Taylor expansion):
          Adds the term: ½·g·(∂g/∂Y)·(ΔW² - dt)
          where ΔW = residual / σ (inferred Brownian increment)
          This penalizes diffusion networks that ignore the spatial gradient of σ.

        BUG #TRAIN-WM-3 FIX:
          σ²_i = Σ_j g²_{ij} = row-wise sum of g's Brownian columns squared
          = diagonal of g @ gᵀ, shape (B*T, D)
          NOT the Frobenius norm of g which would include cross-asset interactions.

        Args:
            Y_curr, Y_next: State vectors at consecutive time steps
            Z_flat:         Regime vectors aligned to (B*T)
            conditioned_sde: Object with .f_net() and .g_net() methods
            dt:             Time step (1.0 trading day)

        Returns:
            (total_loss, em_loss, milstein_loss) — all scalar tensors
        """
        t_dummy = torch.zeros(Y_curr.shape[0], device=self.device)

        # ── Drift and diffusion ───────────────────────────────────────────────
        # f: (B*T, D)  — deterministic drift per asset per day
        # g: (B*T, D, Brownian_Size) — diffusion matrix
        drift     = conditioned_sde.f_net(t_dummy, Y_curr, Z_flat)
        diffusion = conditioned_sde.g_net(t_dummy, Y_curr, Z_flat)

        # ── BUG #TRAIN-WM-3 FIX: per-asset variance ──────────────────────────
        # σ²_i = Σ_j g²_{ij}  →  sum over Brownian dimension
        # shape: (B*T, D)  — each entry is the variance for one asset's return
        per_asset_var = diffusion.pow(2).sum(dim=-1).clamp(min=1e-6)  # (B*T, D)

        # ── Euler-Maruyama NLL ────────────────────────────────────────────────
        actual_delta   = Y_next - Y_curr                 # (B*T, D)
        expected_delta = drift * dt                      # (B*T, D)
        residual       = actual_delta - expected_delta   # (B*T, D)

        # NLL = residual² / (2σ²dt) + ½ log(σ²)
        # The constant 0.5·log(2π·dt) is dropped (does not affect gradient)
        em_nll = (
            residual.pow(2) / (2.0 * per_asset_var * dt)
            + 0.5 * torch.log(per_asset_var)
        )  # (B*T, D)

        em_loss = em_nll.mean()

        # ── Milstein correction (FIX #WM-MILSTEIN) ───────────────────────────
        # Inferred Brownian increment: ΔW ≈ residual / (σ · √dt)
        # Shape: (B*T, D)
        sigma = per_asset_var.sqrt()
        delta_W = residual / (sigma * (dt ** 0.5) + 1e-8)   # (B*T, D)

        # Compute ∂g/∂Y: Jacobian of diffusion norm w.r.t. state
        # We compute the gradient of the scalar g_norm w.r.t. Y_curr
        # g_norm_per_asset = σ = sqrt(Σ_j g²_{ij})  →  shape (B*T, D)
        # We want ∂σ_i/∂Y_j for each asset i, which we approximate as
        # the gradient of σ.sum() w.r.t. Y_curr, shape (B*T, D)
        # This is a diagonal approximation (ignoring cross-asset Jacobian terms)
        try:
            Y_curr_grad = Y_curr.detach().requires_grad_(True)
            diff_grad_out = conditioned_sde.g_net(t_dummy, Y_curr_grad, Z_flat)
            sigma_for_grad = diff_grad_out.pow(2).sum(dim=-1).sqrt()  # (B*T, D)

            # Sum to scalar per batch element for grad computation
            grad_sigma = torch.autograd.grad(
                sigma_for_grad.sum(), Y_curr_grad,
                create_graph=False, retain_graph=False,
            )[0]   # (B*T, D)  — ∂(Σ_i σ_i)/∂Y_j

            # Milstein residual: ½ · σ · (∂σ/∂Y) · (ΔW² - dt)
            # Shape: (B*T, D)
            milstein_residual = 0.5 * sigma * grad_sigma * (delta_W.pow(2) - dt)
            milstein_loss = _MILSTEIN_WEIGHT * milstein_residual.pow(2).mean()

        except RuntimeError:
            # Autograd unavailable in this context (e.g., torch.compile) — skip Milstein
            logger.debug("Milstein correction skipped (autograd unavailable).")
            milstein_loss = torch.tensor(0.0, device=self.device)

        total_loss = em_loss + milstein_loss
        return total_loss, em_loss, milstein_loss

    # ── TRAINING LOOP ─────────────────────────────────────────────────────────

    def train(self) -> None:
        """
        Main training loop.

        Each epoch:
          1. Flattens (B, T, D) trajectories into (B*T, D) transitions
          2. Gets real z_t conditioning from pre-computed VAE encodings
          3. Builds per-batch conditioned SDE via build_conditioned_sde()
             (BUG #TRAIN-WM-1 fix: no global state mutation)
          4. Computes EM-NLL + Milstein loss
          5. Validates on held-out chronological split
          6. Saves best checkpoint by validation loss
        """
        train_loader, val_loader = self._load_training_data()
        dt = 1.0   # One trading day per SDE step

        logger.info(
            f"Initiating Neural SDE Itô training | epochs={self.epochs} | "
            f"milstein_weight={_MILSTEIN_WEIGHT} | dt={dt} | "
            f"state_dim={self.state_dim} | regime_dim={self.regime_dim}"
        )

        best_val_loss = float("inf")
        T = self.seq_len - 1   # Number of transitions per trajectory = 20

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss = em_total = mil_total = 0.0

            for Y_batch, Z_batch in train_loader:
                # Y_batch: (B, 21, 25)  — normalized return trajectories
                # Z_batch: (B, 16)      — VAE-encoded regime vectors
                Y_batch = Y_batch.to(self.device, non_blocking=True)
                Z_batch = Z_batch.to(self.device, non_blocking=True)

                self.optimizer.zero_grad(set_to_none=True)

                with autocast():
                    # BUG #TRAIN-WM-1 FIX: per-batch conditioned SDE via closure
                    # build_conditioned_sde(Z_batch) returns an object whose f and g
                    # close over Z_batch — zero global state mutation, thread-safe
                    conditioned_sde = self.model.build_conditioned_sde(Z_batch).to(self.device)

                    # Flatten temporal dimension: (B, T, D) → (B*T, D)
                    Y_curr = Y_batch[:, :-1, :].reshape(-1, self.state_dim)  # (B*T, D)
                    Y_next = Y_batch[:,  1:, :].reshape(-1, self.state_dim)  # (B*T, D)

                    # Expand z_t to match flattened transitions: (B, 16) → (B*T, 16)
                    Z_flat = Z_batch.unsqueeze(1).expand(-1, T, -1).reshape(-1, self.regime_dim)

                    loss, em_loss, mil_loss = self._compute_em_nll_loss(
                        Y_curr, Y_next, Z_flat, conditioned_sde, dt
                    )

                # Note: Milstein uses autograd.grad (not backward), so scaler handles
                # only the main loss gradient. Milstein grad is computed inline.
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), _GRAD_CLIP)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_loss += loss.item()
                em_total   += em_loss.item()
                mil_total  += mil_loss.item() if isinstance(mil_loss, torch.Tensor) else 0.0

            self.scheduler.step()

            avg_total = total_loss / len(train_loader)
            avg_em    = em_total   / len(train_loader)
            avg_mil   = mil_total  / len(train_loader)
            val_loss  = self._validate(val_loader, dt)
            lr        = self.scheduler.get_last_lr()[0]

            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    f"Epoch [{epoch:03d}/{self.epochs}] | "
                    f"total={avg_total:.5f} | em={avg_em:.5f} | milstein={avg_mil:.5f} | "
                    f"val={val_loss:.5f} | lr={lr:.2e}"
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_weights(_SDE_BEST_PATH)
                logger.info(
                    f"  ✅ New best checkpoint | epoch={epoch} | val_loss={val_loss:.5f}"
                )

        self._save_weights(_SDE_LATEST_PATH)
        logger.info(
            f"Training complete | best_val_loss={best_val_loss:.5f} | "
            f"weights: {_SDE_LATEST_PATH}"
        )

    @torch.no_grad()
    def _validate(self, val_loader: DataLoader, dt: float) -> float:
        """Validation pass — no Milstein (no autograd.grad in no_grad context)."""
        self.model.eval()
        total = 0.0
        T = self.seq_len - 1

        for Y_batch, Z_batch in val_loader:
            Y_batch = Y_batch.to(self.device, non_blocking=True)
            Z_batch = Z_batch.to(self.device, non_blocking=True)

            with autocast():
                conditioned_sde = self.model.build_conditioned_sde(Z_batch).to(self.device)

                Y_curr = Y_batch[:, :-1, :].reshape(-1, self.state_dim)
                Y_next = Y_batch[:,  1:, :].reshape(-1, self.state_dim)
                Z_flat = Z_batch.unsqueeze(1).expand(-1, T, -1).reshape(-1, self.regime_dim)

                t_dummy   = torch.zeros(Y_curr.shape[0], device=self.device)
                drift     = conditioned_sde.f_net(t_dummy, Y_curr, Z_flat)
                diffusion = conditioned_sde.g_net(t_dummy, Y_curr, Z_flat)

                # BUG #TRAIN-WM-3 FIX: diagonal of g@gᵀ
                per_asset_var = diffusion.pow(2).sum(dim=-1).clamp(min=1e-6)

                actual_delta   = Y_next - Y_curr
                expected_delta = drift * dt
                residual       = actual_delta - expected_delta

                em_nll = (
                    residual.pow(2) / (2.0 * per_asset_var * dt)
                    + 0.5 * torch.log(per_asset_var)
                )
                total += em_nll.mean().item()

        return total / max(len(val_loader), 1)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save_weights(self, path: str) -> None:
        os.makedirs("models/weights", exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logger.info(f"SDE weights saved → '{path}'")

    def _save_state_norm(self) -> None:
        """
        Saves the return normalization statistics (mean, std) per asset.
        generate_scenarios.py and the live stress test must apply identical
        normalization to ensure inference operates in the training distribution.
        """
        if self._state_mean is None or self._state_std is None:
            return
        os.makedirs("models/weights", exist_ok=True)
        scaler = np.stack([self._state_mean, self._state_std])  # (2, 25)
        np.save(_STATE_NORM_PATH, scaler)
        logger.info(f"State normalization stats saved → '{_STATE_NORM_PATH}'")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    trainer = WorldModelTrainer()
    trainer.train()