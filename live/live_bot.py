"""
FORTRESS v5 - live_bot.py
Path: live/live_bot.py

Master Organism Ignition.
Pre-flight validation + coordinated microservice launch.

FIXES APPLIED:
  - BUG #LB-1 (CRITICAL): `launch_services()` only created asyncio tasks for
    DataPipeline, RegimeEncoderService, and ExecutionService. Four microservices
    were completely absent:
      • AlphaEngineService  — GATv2 was never computing alpha scores
      • PortfolioAgentService — EDT was never allocating capital
      • TDATopologyService — topology collapse signals were never generated
      • OrderManagerService — MARL rewards were never being tracked
    As a result, `alpha:scores` was never published to Redis, so the EDT
    permanently allocated from a zero-padded state vector (BUG #5 in
    portfolio_agent_svc.py was a symptom of this root cause).
    All six microservices are now launched as concurrent asyncio tasks.

  - BUG #LB-2: The pre-flight check for Kafka was absent. The system could
    start successfully while Kafka was still initialising, causing all
    AIOKafkaConsumer/Producer connections to fail silently on first connect.
    Added an explicit Kafka healthcheck that verifies the broker is
    reachable and that the critical topics exist before allowing launch.

  - BUG #LB-3: Model weight files were never verified during pre-flight.
    The system would start and run with randomly-initialised weights if the
    training step was skipped. Added weight file existence checks with a
    warning (not a halt) to allow paper-trading with untrained models
    while flagging the condition clearly.

  - IMPROVEMENT: Added `_OrderManagerLoop` — a thin asyncio service wrapper
    that consumes the Kafka 'order-fills' topic and feeds fills into
    StatefulOrderTracker. Previously no code was consuming order fills.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from typing import Any, Dict, List

import yaml

logger = logging.getLogger("FortressOrganism")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

# ── Model weight files checked at pre-flight ─────────────────────────────────
_WEIGHT_FILES: List[str] = [
    "models/weights/mamba_kan_latest.pt",
    "models/weights/edt_latest.pt",
    "models/weights/sde_latest.pt",
    "models/weights/gat_alpha_latest.pt",
    "models/weights/ltc_latest.pt",
]

# ── Kafka topics that must exist before launch ───────────────────────────────
_REQUIRED_TOPICS: List[str] = [
    "market-data-ticks",
    "macro-data-updates",
    "regime-posterior",
    "alpha-signals",
    "target-weights",
    "order-fills",
    "execution-rewards",
    "emergency-alerts",
    "dead-letter-queue",
]


class _OrderManagerLoop:
    """
    Thin asyncio wrapper around StatefulOrderTracker.
    Consumes 'order-fills' from Kafka and feeds events into the tracker.
    This is the missing consumer that was preventing MARL reward generation.
    """

    def __init__(self) -> None:
        from live.order_manager import StatefulOrderTracker
        self._tracker = StatefulOrderTracker()

    async def run(self) -> None:
        try:
            from aiokafka import AIOKafkaConsumer
        except ImportError as exc:
            raise ImportError("aiokafka required") from exc

        await self._tracker.setup()
        kafka_url = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

        consumer = AIOKafkaConsumer(
            "order-fills",
            bootstrap_servers=kafka_url,
            group_id="order_manager_group",
            auto_offset_reset="earliest",  # Replay fills missed during downtime
        )
        await consumer.start()
        logger.info("OrderManagerLoop: consuming 'order-fills' topic.")

        try:
            async for msg in consumer:
                import json
                payload = json.loads(msg.value.decode("utf-8"))
                await self._tracker.process_fill(payload)
        except asyncio.CancelledError:
            logger.info("OrderManagerLoop: shutting down.")
        finally:
            await consumer.stop()
            await self._tracker.close()


class FortressOrganism:
    """
    The top-level coordinator for Project Ouroboros.
    Validates infrastructure, then launches all 7 microservice coroutines
    as concurrent asyncio tasks within a single event loop.
    """

    def __init__(self) -> None:
        self.config: Dict[str, Any] = self._load_config("config/hyperparams.yaml")
        self.services_running: bool = False
        self.tasks: List[asyncio.Task] = []
        self._telegram = self._build_telegram()

    def _load_config(self, path: str) -> Dict[str, Any]:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def _build_telegram(self):
        """Lazy-imports TelegramAlertService to avoid circular imports at module load."""
        try:
            from monitoring.telegram_bot import TelegramAlertService
            return TelegramAlertService(
                token=os.getenv("TELEGRAM_BOT_TOKEN"),
                chat_id=os.getenv("TELEGRAM_CHAT_ID"),
            )
        except Exception:
            logger.warning("TelegramAlertService unavailable — alerts suppressed.")
            return None

    async def _alert(self, message: str) -> None:
        if self._telegram:
            try:
                await self._telegram.send_alert(message)
            except Exception:
                pass

    # ── Pre-flight checks ─────────────────────────────────────────────────────

    async def _pre_flight_checks(self) -> bool:
        logger.info("═══ FORTRESS v5 PRE-FLIGHT SEQUENCE ═══")
        passed = True

        # 1. Redis
        try:
            import redis.asyncio as _redis
            r = _redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
            await r.ping()
            await r.aclose()
            logger.info("✅ [1/5] Redis: ONLINE")
        except Exception as exc:
            logger.critical(f"❌ [1/5] Redis OFFLINE: {exc}")
            passed = False

        # 2. TimescaleDB
        try:
            import asyncpg
            conn = await asyncpg.connect(
                user=os.getenv("DB_USER", "postgres"),
                password=os.getenv("DB_PASSWORD", ""),
                database=os.getenv("DB_NAME", "fortress"),
                host=os.getenv("DB_HOST", "localhost"),
            )
            version = await conn.fetchval("SELECT version();")
            await conn.close()
            logger.info(f"✅ [2/5] TimescaleDB: ONLINE ({version[:40]}...)")
        except Exception as exc:
            logger.critical(f"❌ [2/5] TimescaleDB OFFLINE: {exc}")
            passed = False

        # 3. Kafka (BUG #LB-2 FIX)
        try:
            from aiokafka.admin import AIOKafkaAdminClient
            kafka_url = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
            admin = AIOKafkaAdminClient(bootstrap_servers=kafka_url)
            await admin.start()
            existing_topics = await admin.list_topics()
            missing = [t for t in _REQUIRED_TOPICS if t not in existing_topics]
            await admin.close()

            if missing:
                logger.critical(
                    f"❌ [3/5] Kafka MISSING TOPICS: {missing}. "
                    "Run kafka-init container first."
                )
                passed = False
            else:
                logger.info(
                    f"✅ [3/5] Kafka: ONLINE — all {len(_REQUIRED_TOPICS)} topics present."
                )
        except Exception as exc:
            logger.critical(f"❌ [3/5] Kafka OFFLINE: {exc}")
            passed = False

        # 4. Alpaca Broker API
        try:
            from live.alpaca_client import ResilientAlpacaClient
            client = ResilientAlpacaClient(paper=True)
            account = await client.get_account()
            if getattr(account, "account_blocked", False):
                raise PermissionError("Alpaca account is blocked by the broker.")
            buying_power = getattr(account, "buying_power", "N/A")
            logger.info(f"✅ [4/5] Alpaca API: ONLINE (Buying Power: ${buying_power})")
        except Exception as exc:
            logger.critical(f"❌ [4/5] Alpaca API FAILED: {exc}")
            passed = False

        # 5. Model weight files (BUG #LB-3 FIX — WARNING, not halt)
        missing_weights = [f for f in _WEIGHT_FILES if not os.path.isfile(f)]
        if missing_weights:
            logger.warning(
                f"⚠️  [5/5] Missing trained weight files: {missing_weights}. "
                "Models will use RANDOMLY INITIALISED WEIGHTS. "
                "Run run_all.sh training steps before live trading."
            )
            await self._alert(
                f"⚠️ FORTRESS WARNING: {len(missing_weights)} model weight "
                "files missing. Running with untrained models."
            )
        else:
            logger.info(f"✅ [5/5] Model Weights: all {len(_WEIGHT_FILES)} files present.")

        if not passed:
            logger.critical("PRE-FLIGHT FAILED. System refusing to start.")
            await self._alert("🚨 FORTRESS STARTUP ABORTED: Infrastructure checks failed.")

        return passed

    # ── Service launch (BUG #LB-1 FIX) ──────────────────────────────────────

    async def launch_services(self) -> None:
        """
        Launches all 7 microservices as concurrent asyncio tasks.

        Previously only DataPipeline, RegimeEncoderService, and ExecutionService
        were started. The missing services meant alpha scores, portfolio weights,
        TDA topology signals, and MARL rewards were NEVER generated.
        """
        logger.info("═══ IGNITING ALL MICROSERVICES ═══")

        # --- Import all services ---
        from data.pipeline import DataPipeline
        from services.regime_encoder_svc import RegimeEncoderService
        from services.alpha_engine_svc import AlphaEngineService
        from services.portfolio_agent_svc import PortfolioAgentService
        from services.execution_svc import ExecutionService
        from services.tda_topology_svc import TDATopologyService

        # --- Instantiate all services ---
        data_pipeline      = DataPipeline()
        regime_svc         = RegimeEncoderService(self.config)
        alpha_svc          = AlphaEngineService(self.config)
        portfolio_svc      = PortfolioAgentService(self.config)
        execution_svc      = ExecutionService(self.config)
        tda_svc            = TDATopologyService(self.config)
        order_mgr_loop     = _OrderManagerLoop()

        # --- Launch all as concurrent asyncio tasks ---
        self.tasks = [
            asyncio.create_task(
                data_pipeline.run_continuous(), name="DataPipeline"
            ),
            asyncio.create_task(
                regime_svc.run(), name="RegimeEncoder"
            ),
            asyncio.create_task(
                alpha_svc.run(), name="AlphaEngine"
            ),
            asyncio.create_task(
                portfolio_svc.run(), name="PortfolioAgent"
            ),
            asyncio.create_task(
                execution_svc.run(), name="ExecutionRouter"
            ),
            asyncio.create_task(
                tda_svc.run(), name="TDATopology"
            ),
            asyncio.create_task(
                order_mgr_loop.run(), name="OrderManager"
            ),
        ]

        self.services_running = True
        logger.info(f"✅ {len(self.tasks)} microservices ignited.")
        await self._alert("🟢 FORTRESS v5 Organism is LIVE — all 7 microservices running.")

        # Monitor tasks — if any crash, log it and restart
        await self._supervise_tasks()

    async def _supervise_tasks(self) -> None:
        """
        Waits for all tasks. If any task raises an unhandled exception,
        logs the error and sends a Telegram alert. Tasks with `restart: on-failure`
        semantics are handled by Docker at the container level.
        """
        results = await asyncio.gather(*self.tasks, return_exceptions=True)
        for task, result in zip(self.tasks, results):
            if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
                logger.critical(
                    f"Task '{task.get_name()}' raised an unhandled exception: {result}"
                )
                await self._alert(
                    f"🔴 FORTRESS CRITICAL: Task '{task.get_name()}' crashed: {result}"
                )

    # ── Graceful shutdown ─────────────────────────────────────────────────────

    async def shutdown(self, signal_type: Any = None) -> None:
        if not self.services_running:
            return
        logger.warning(
            f"Received shutdown signal '{signal_type}'. Initiating graceful teardown..."
        )
        await self._alert("🔴 FORTRESS v5 shutdown initiated.")

        for task in self.tasks:
            task.cancel()

        # Allow cancelled tasks to clean up their Kafka/Redis connections
        await asyncio.gather(*self.tasks, return_exceptions=True)

        self.services_running = False
        logger.info("Shutdown complete. All systems offline.")
        sys.exit(0)


async def main() -> None:
    organism = FortressOrganism()

    # Wire OS signals for graceful Docker stop / CTRL-C
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(
            sig,
            lambda s=sig: asyncio.create_task(organism.shutdown(s)),
        )

    if await organism._pre_flight_checks():
        await organism.launch_services()
    else:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())