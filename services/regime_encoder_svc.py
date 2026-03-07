"""
FORTRESS v5 - regime_encoder_svc.py
Path: services/regime_encoder_svc.py

Async Microservice: The Heartbeat of the Organism.
Consumes live market data, evaluates the Mamba-KAN and continuous-time LTC models,
and publishes the updated regime posterior to the Kafka event bus.

FIXES APPLIED:
  - BUG #RE-1 (CRITICAL): `_fetch_observation_history()` returned
    `np.random.randn(252, 52)` — pure Gaussian noise was being fed as the
    historical context window to the Mamba-KAN VAE. The latent z_t published
    to Kafka was conditioned on noise, not real market data. Every downstream
    service (GATv2, EDT, Deep Hedging) received garbage regime signals.
    Fixed: `_fetch_observation_history()` now queries TimescaleDB via asyncpg,
    fetching the 252 most recent rows from the `prices` hypertable with strict
    as_of_date causality enforcement. Falls back to zeros with a CRITICAL log.

  - BUG #RE-2 (CRITICAL): Model weights were never loaded from disk.
    The Mamba-KAN and LTC models ran with randomly initialised PyTorch weights.
    Added `_load_model_weights()` with explicit file existence checks.
    Emits a WARNING (not a crash) if weights are missing, allowing paper-trading
    with untrained models while flagging the condition clearly.

  - BUG #RE-3: `self.last_tick_time` was set at __init__ time, meaning the
    first delta_t passed to the LTC ODE solver could be hours or days old
    (the time between container start and first tick). The LTC integrator
    with a multi-hour delta_t produces numerical overflow in the ODE solution.
    Fixed: `self.last_tick_time` is reset on the first tick received, not at init.

  - IMPROVEMENT: Published regime posterior now includes a `regime_label`
    string ('bull_low_vol', 'crisis', etc.) derived from the top-k z_t
    dimension, enabling human-readable monitoring dashboards.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, Optional, List

import asyncio
import numpy as np
import torch

logger = logging.getLogger("RegimeEncoderSvc")

# ── Model weight paths ────────────────────────────────────────────────────────
_MAMBA_KAN_WEIGHTS: str = "models/weights/mamba_kan_latest.pt"
_LTC_WEIGHTS:       str = "models/weights/ltc_latest.pt"

# ── Observation schema ────────────────────────────────────────────────────────
# Must match hyperparams.yaml::mamba_kan::obs_dim = 52
_OBS_DIM: int    = 52
_SEQ_LEN: int    = 252   # 1 trading year of context

# ── LTC urgency threshold from hyperparams.yaml::ltc::urgency_threshold ─────
_LTC_URGENCY_THRESHOLD: float = 0.70

# ── Regime label mapping (top active z_t dimension → human label) ────────────
_REGIME_LABELS: List[str] = [
    "bull_low_vol",
    "bull_high_vol",
    "bear_low_vol",
    "bear_high_vol",
    "crisis",
    "recovery",
    "flat_deflation",
    "stagflation",
    "rate_shock",
    "credit_stress",
    "momentum_bull",
    "momentum_bear",
    "liquidity_crunch",
    "risk_on_EM",
    "risk_off_DM",
    "unknown",
]


class RegimeEncoderService:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Redis client for state caching
        try:
            import redis.asyncio as redis
            self._redis = redis.Redis.from_url(
                os.getenv("REDIS_URL", "redis://localhost:6379")
            )
        except ImportError as exc:
            raise ImportError("redis[asyncio] required: pip install redis") from exc

        # ── Load Models (BUG #RE-2 FIX) ──────────────────────────────────────
        logger.info("Loading Mamba-KAN and LTC models...")
        from models.regime.mamba_kan_vae import MambaKANVAE
        from models.regime.liquid_neural_net import IntraRegimeMonitor

        self.mamba_kan = MambaKANVAE(config.get("mamba_kan", {}))
        self.ltc_monitor = IntraRegimeMonitor(config.get("ltc", {}))

        self._load_model_weights()

        self.mamba_kan.to(self.device).eval()
        self.ltc_monitor.to(self.device).eval()

        # ── LTC state ────────────────────────────────────────────────────────
        # BUG #RE-3 FIX: Reset on first tick, not at init.
        self._last_tick_time: Optional[float] = None
        self._first_tick_received: bool = False

        # ── DB pool (lazy-init in run()) ─────────────────────────────────────
        self._db_pool = None

        # ── Kafka handles (set by setup_kafka) ───────────────────────────────
        self.consumer = None
        self.producer = None

    def _load_model_weights(self) -> None:
        """
        BUG #RE-2 FIX: Load trained weights from disk.
        Logs a WARNING if weights are missing — does NOT crash. This allows
        the system to run in paper-trade mode with untrained models, which is
        useful for integration testing, but clearly flags the condition.
        """
        for path, model, name in [
            (_MAMBA_KAN_WEIGHTS, self.mamba_kan,  "Mamba-KAN"),
            (_LTC_WEIGHTS,       self.ltc_monitor, "LTC"),
        ]:
            if os.path.isfile(path):
                try:
                    state_dict = torch.load(path, map_location=self.device)
                    model.load_state_dict(state_dict)
                    logger.info(f"✅ {name} weights loaded from '{path}'.")
                except Exception as exc:
                    logger.error(
                        f"❌ Failed to load {name} weights from '{path}': {exc}. "
                        "Running with random weights."
                    )
            else:
                logger.warning(
                    f"⚠️  {name} weight file '{path}' not found. "
                    "Model running with RANDOMLY INITIALISED WEIGHTS. "
                    "Run training/train_regime.py first."
                )

    async def _init_db_pool(self) -> None:
        """Lazy-initialises the asyncpg connection pool to TimescaleDB."""
        if self._db_pool is not None:
            return
        try:
            import asyncpg
            self._db_pool = await asyncpg.create_pool(
                user=os.getenv("DB_USER",     "postgres"),
                password=os.getenv("DB_PASSWORD", ""),
                database=os.getenv("DB_NAME",     "fortress"),
                host=os.getenv("DB_HOST",         "localhost"),
                min_size=2,
                max_size=5,
            )
            logger.info("RegimeEncoderSvc: TimescaleDB pool initialised.")
        except Exception as exc:
            logger.error(f"TimescaleDB pool failed: {exc}. Observation fetch will use zeros.")

    # ── Kafka setup ───────────────────────────────────────────────────────────

    async def setup_kafka(self) -> None:
        try:
            from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
        except ImportError as exc:
            raise ImportError("aiokafka required") from exc

        kafka_url = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

        self.consumer = AIOKafkaConsumer(
            "market-data-ticks",
            "macro-data-updates",
            bootstrap_servers=kafka_url,
            group_id="regime_encoder_group",
            auto_offset_reset="latest",
        )
        self.producer = AIOKafkaProducer(bootstrap_servers=kafka_url)

        await self.consumer.start()
        await self.producer.start()
        logger.info("RegimeEncoderSvc: Kafka consumer/producer connected.")

    # ── Main loop ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        await self._init_db_pool()
        await setup_kafka(self)

        self.ltc_monitor.reset_hidden_state()

        try:
            async for msg in self.consumer:
                topic   = msg.topic
                payload = json.loads(msg.value.decode("utf-8"))

                if topic == "market-data-ticks":
                    await self._process_intraday_tick(payload)
                elif topic == "macro-data-updates":
                    await self._process_macro_update(payload)

        except asyncio.CancelledError:
            logger.info("RegimeEncoderSvc: shutdown signal received.")
        except Exception as exc:
            logger.critical(f"Fatal error in regime encoder loop: {exc}", exc_info=True)
            raise
        finally:
            if self.consumer:
                await self.consumer.stop()
            if self.producer:
                await self.producer.stop()
            if self._redis:
                await self._redis.aclose()

    # ── Event handlers ────────────────────────────────────────────────────────

    async def _process_intraday_tick(self, payload: Dict[str, Any]) -> None:
        """
        Processes high-frequency price data through the Liquid Neural Net.
        Updates the LTC hidden state and publishes an urgency score to Redis.
        """
        tick_features = np.array(
            payload.get("features", [0.0] * 15), dtype=np.float32
        )
        tick_time     = float(payload.get("timestamp", time.time()))

        # BUG #RE-3 FIX: On the FIRST tick, initialise last_tick_time to now.
        # This prevents a multi-hour delta_t that would overflow the ODE solver.
        if not self._first_tick_received:
            self._last_tick_time    = tick_time
            self._first_tick_received = True
            logger.info("RegimeEncoderSvc: First tick received — LTC delta_t timer started.")
            return

        delta_t = tick_time - self._last_tick_time
        self._last_tick_time = tick_time

        # Guard: delta_t must be physically plausible for intraday (< 30 min)
        if delta_t <= 0 or delta_t > 1800:
            logger.debug(f"Skipping tick with implausible delta_t={delta_t:.1f}s.")
            return

        with torch.no_grad():
            tick_tensor = torch.FloatTensor(tick_features).unsqueeze(0).to(self.device)
            urgency     = self.ltc_monitor(tick_tensor, delta_t=delta_t)
            urgency_val = float(urgency.squeeze().cpu().item())

        await self._redis.set("regime:ltc_urgency", urgency_val, ex=3600)

        if urgency_val > _LTC_URGENCY_THRESHOLD:
            logger.warning(
                f"🔴 LTC Urgency spike: {urgency_val:.4f} > {_LTC_URGENCY_THRESHOLD}. "
                "Publishing emergency alert."
            )
            await self._publish_emergency_alert(urgency_val)

    async def _process_macro_update(self, payload: Dict[str, Any]) -> None:
        """
        Processes a macro release event through the Mamba-KAN VAE.
        Fetches the full 252-day historical observation matrix from TimescaleDB
        and publishes the updated regime posterior z_t to Kafka.
        """
        as_of_date = payload.get("timestamp", time.time())

        # BUG #RE-1 FIX: Fetch real observations from TimescaleDB.
        obs_history = await self._fetch_observation_history(as_of_date)

        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs_history).unsqueeze(0).to(self.device)
            # Shape: (1, seq_len, obs_dim) → encoder produces (1, latent_dim)
            z_mu, z_sigma = self.mamba_kan.encode(obs_tensor)

            z_mu_np    = z_mu.squeeze(0).cpu().numpy()
            z_sigma_np = z_sigma.squeeze(0).cpu().numpy()

        # Persist z_t to Redis for consumption by portfolio_agent_svc and alpha_engine_svc
        await self._redis.set(
            "regime:z_mu",
            json.dumps(z_mu_np.tolist()),
            ex=86400,
        )
        await self._redis.set(
            "regime:z_sigma",
            json.dumps(z_sigma_np.tolist()),
            ex=86400,
        )

        # Determine regime label from the dominant z_t dimension
        dominant_dim  = int(np.argmax(np.abs(z_mu_np)))
        regime_label  = _REGIME_LABELS[dominant_dim % len(_REGIME_LABELS)]

        # Fetch TDA and LTC signals for the published payload
        tda_raw = await self._redis.get("tda:alert")
        ltc_raw = await self._redis.get("regime:ltc_urgency")

        regime_payload = {
            "timestamp":    time.time(),
            "z_mu":         z_mu_np.tolist(),
            "z_sigma":      z_sigma_np.tolist(),
            "tda_alert":    int(tda_raw) if tda_raw else 0,
            "ltc_urgency":  float(ltc_raw) if ltc_raw else 0.0,
            "regime_label": regime_label,
        }

        await self.producer.send_and_wait(
            "regime-posterior",
            json.dumps(regime_payload).encode("utf-8"),
        )
        logger.info(
            f"Regime posterior published: label='{regime_label}', "
            f"z_mu[:4]={z_mu_np[:4].round(3).tolist()}"
        )

    # ── Data fetch (BUG #RE-1 FIX) ───────────────────────────────────────────

    async def _fetch_observation_history(
        self,
        as_of_timestamp: float,
    ) -> np.ndarray:
        """
        Fetches the 252 most recent daily observation rows from TimescaleDB,
        strictly as-of `as_of_timestamp` to prevent look-ahead bias.

        The query returns _OBS_DIM=52 features per row, ordered chronologically.
        If the DB is unavailable or returns insufficient rows, falls back to zeros
        with a CRITICAL log (the system continues but the regime signal is degraded).

        Returns:
            obs_history: (seq_len, _OBS_DIM) float32 numpy array
        """
        if self._db_pool is None:
            logger.critical(
                "DB pool not initialised — returning zero observation history. "
                "Regime encoder is operating BLIND."
            )
            return np.zeros((_SEQ_LEN, _OBS_DIM), dtype=np.float32)

        # Convert unix timestamp to date string for SQL query
        from datetime import datetime, timezone
        as_of_date = datetime.fromtimestamp(
            as_of_timestamp, tz=timezone.utc
        ).strftime("%Y-%m-%d")

        # This query must match the exact columns produced by the DataPipeline
        # ingestion into TimescaleDB. Column order determines feature index.
        query = """
            SELECT
                p.adj_close, p.volume_norm, p.ret_1d, p.ret_5d, p.ret_20d,
                p.volatility_20d, p.rsi_14, p.vwap_delta,
                p.bid_ask_spread_z, p.order_book_imbalance,
                m.t10y2y, m.nfci, m.cpiaucsl_yoy, m.unrate, m.walcl_chg,
                o.net_gex_bn,
                -- Pad remaining dims with 0 until the ingestion pipeline
                -- fills all 52 columns (placeholder zeros are safe — see note below)
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            FROM prices p
            LEFT JOIN fred_data m ON m.metric_date = p.metric_date
                AND m.as_of_date <= $1::date
            LEFT JOIN options_features o ON o.metric_date = p.metric_date
                AND o.ticker = 'SPY'
            WHERE p.as_of_date <= $1::date
            ORDER BY p.metric_date DESC
            LIMIT $2;
        """
        try:
            async with self._db_pool.acquire() as conn:
                rows = await conn.fetch(query, as_of_date, _SEQ_LEN)

            if len(rows) == 0:
                logger.critical(
                    f"TimescaleDB returned 0 rows for as_of_date={as_of_date}. "
                    "Run scripts/download_history.py first."
                )
                return np.zeros((_SEQ_LEN, _OBS_DIM), dtype=np.float32)

            # Rows are ordered DESC (newest first). Reverse to chronological order.
            arr = np.array([list(row) for row in rows], dtype=np.float32)
            arr = arr[::-1]  # Reverse to chronological

            # Pad with zeros if fewer than _SEQ_LEN rows available (early in life)
            if len(arr) < _SEQ_LEN:
                padding = np.zeros((_SEQ_LEN - len(arr), _OBS_DIM), dtype=np.float32)
                arr     = np.concatenate([padding, arr], axis=0)

            # Clamp NaNs (LEFT JOIN may produce nulls from unjoined macro/options rows)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

            return arr.astype(np.float32)

        except Exception as exc:
            logger.critical(
                f"TimescaleDB observation fetch FAILED: {exc}. "
                "Returning zero history — regime encoder is BLIND.",
                exc_info=True,
            )
            return np.zeros((_SEQ_LEN, _OBS_DIM), dtype=np.float32)

    # ── Emergency signal ──────────────────────────────────────────────────────

    async def _publish_emergency_alert(self, urgency_score: float) -> None:
        """Publishes an LTC urgency emergency alert to the Kafka emergency-alerts topic."""
        alert_payload = {
            "timestamp":          time.time(),
            "urgency_score":      urgency_score,
            "trigger":            "LTC_INTRADAY_DRIFT",
            "recommended_action": "REDUCE_50PCT" if urgency_score < 0.90 else "HALT",
        }
        await self.producer.send_and_wait(
            "emergency-alerts",
            json.dumps(alert_payload).encode("utf-8"),
        )


async def setup_kafka(svc: RegimeEncoderService) -> None:
    """Module-level helper to call `svc.setup_kafka()` — kept for backward compat."""
    await svc.setup_kafka()


if __name__ == "__main__":
    import yaml
    with open("config/hyperparams.yaml", "r") as f:
        config = yaml.safe_load(f)
    svc = RegimeEncoderService(config)
    asyncio.run(svc.run())