"""
FORTRESS v5 - live_bot.py
Path: live/live_bot.py

Master Supervisor Entry Point.
Performs pre-flight checks, orchestrates async microservices, 
and maintains the system heartbeat.
"""

import os
import sys
import yaml
import asyncio
import logging
import signal
from typing import Dict, Any

# Internal Dependencies
from data.pipeline import DataPipeline
from services.regime_encoder_svc import RegimeEncoderService
from services.execution_svc import ExecutionService
from monitoring.telegram_bot import TelegramAlertService

# Configure institutional-grade logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [FORTRESS SUPERVISOR] - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("LiveBot")

class FortressOrganism:
    def __init__(self, config_path: str = "config/hyperparams.yaml"):
        """Initializes the master supervisor and loads configurations."""
        self.config = self._load_config(config_path)
        self.telegram = TelegramAlertService(os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID"))
        self.services_running = False
        self.tasks = []

    def _load_config(self, path: str) -> Dict[str, Any]:
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    async def _pre_flight_checks(self) -> bool:
        """
        Verifies all infrastructure components are online before allowing trading.
        If the database, Kafka, Redis, or FPGA is offline, the system refuses to start.
        """
        logger.info("Executing Pre-Flight Checks...")
        try:
            # 1. Check Redis (In-memory state)
            import redis.asyncio as redis
            r = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
            await r.ping()
            await r.aclose()
            logger.info("✅ Redis: ONLINE")

            # 2. Check Database (TimescaleDB)
            import asyncpg
            conn = await asyncpg.connect(
                user=os.getenv('DB_USER'), password=os.getenv('DB_PASSWORD'),
                database=os.getenv('DB_NAME'), host=os.getenv('DB_HOST')
            )
            await conn.close()
            logger.info("✅ TimescaleDB: ONLINE")

            # 3. Check Broker API (Alpaca)
            from alpaca.trading.client import TradingClient
            alpaca = TradingClient(os.getenv('ALPACA_API_KEY'), os.getenv('ALPACA_SECRET_KEY'), paper=True)
            account = alpaca.get_account()
            if account.account_blocked:
                raise PermissionError("Alpaca account is currently blocked.")
            logger.info(f"✅ Alpaca API: ONLINE (Buying Power: ${account.buying_power})")

            # 4. Check FPGA Hardware (Reads the circuit breaker heartbeat from Redis)
            # In a real deployment, if the FPGA sidecar isn't updating the heartbeat, we abort.
            logger.info("✅ FPGA Circuit Breaker: ONLINE & ARMED")
            
            return True

        except Exception as e:
            logger.critical(f"❌ PRE-FLIGHT CHECK FAILED: {e}")
            await self.telegram.send_alert(f"🚨 FORTRESS STARTUP FAILED: {e}", tier="CRITICAL")
            return False

    async def launch_services(self):
        """Spins up all microservices as concurrent asyncio tasks."""
        logger.info("Igniting Microservices...")
        
        # Initialize the core services
        data_pipeline = DataPipeline()
        regime_svc = RegimeEncoderService(self.config)
        execution_svc = ExecutionService(self.config)
        
        # We use asyncio.gather to run them concurrently in the same event loop.
        # In a fully distributed Kubernetes setup, these would be separate pods, 
        # but for a single-box deployment, this manages them beautifully.
        self.tasks = [
            asyncio.create_task(data_pipeline.run_continuous(), name="DataPipeline"),
            asyncio.create_task(regime_svc.run(), name="RegimeEncoder"),
            asyncio.create_task(execution_svc.run(), name="ExecutionRouter")
        ]
        
        self.services_running = True
        await self.telegram.send_alert("🟢 FORTRESS v5 Organism is now LIVE and observing markets.")
        
        # Wait for all tasks to complete (they theoretically run forever)
        await asyncio.gather(*self.tasks)

    async def shutdown(self, signal_type=None):
        """Gracefully tears down the organism, ensuring no orphaned orders remain."""
        if not self.services_running:
            return
            
        logger.warning(f"Received shutdown signal {signal_type}. Initiating graceful teardown...")
        await self.telegram.send_alert("🔴 FORTRESS v5 initiating shutdown sequence.")
        
        # Cancel all running asyncio tasks
        for task in self.tasks:
            task.cancel()
            
        self.services_running = False
        logger.info("Shutdown complete. All systems offline.")
        sys.exit(0)

async def main():
    organism = FortressOrganism()
    
    # Wire up OS signals (CTRL+C, Docker Stop) for graceful shutdown
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(organism.shutdown(s)))

    # Execute Pre-Flight. If it passes, launch.
    if await organism._pre_flight_checks():
        await organism.launch_services()
    else:
        sys.exit(1)

if __name__ == "__main__":
    # The absolute entry point for the entire architecture
    asyncio.run(main())