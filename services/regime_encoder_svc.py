"""
FORTRESS v5 - regime_encoder_svc.py
Path: services/regime_encoder_svc.py

Async Microservice: The Heartbeat of the Organism.
Consumes live market data, evaluates the Mamba-KAN and continuous-time LTC models, 
and publishes the updated regime posterior to the Kafka event bus.
"""

import os
import json
import time
import asyncio
import logging
import numpy as np
import torch
from typing import Dict, Any, Optional

# Async Kafka and Redis clients
try:
    from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
    import redis.asyncio as redis
except ImportError:
    raise ImportError("Requires aiokafka and redis packages.")

# Internal Model Imports
from models.regime.mamba_kan_vae import MambaKANVAE
from models.regime.liquid_neural_net import IntraRegimeMonitor

logger = logging.getLogger("RegimeEncoderSvc")

class RegimeEncoderService:
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the Kafka topics, Redis state store, and loads the PyTorch models 
        into GPU memory for ultra-low latency inference.
        """
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 1. Initialize State Storage
        self.redis_client = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
        
        # 2. Load Models into VRAM
        logger.info("Loading Mamba-KAN and LTC models into VRAM...")
        self.mamba_kan = MambaKANVAE(config.get('mamba_kan', {}))
        # In production: self.mamba_kan.load_state_dict(torch.load('models/weights/mamba_kan_latest.pt'))
        self.mamba_kan.to(self.device)
        self.mamba_kan.eval()
        
        self.ltc_monitor = IntraRegimeMonitor(config.get('ltc', {}))
        # In production: self.ltc_monitor.load_state_dict(torch.load('models/weights/ltc_latest.pt'))
        self.ltc_monitor.to(self.device)
        self.ltc_monitor.eval()

        # State tracking for the continuous-time LTC
        self.last_tick_time = time.time()

    async def setup_kafka(self):
        """Creates the async consumer and producer."""
        kafka_url = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        
        self.consumer = AIOKafkaConsumer(
            'market-data-ticks',
            'macro-data-updates',
            bootstrap_servers=kafka_url,
            group_id='regime_encoder_group',
            auto_offset_reset='latest'
        )
        
        self.producer = AIOKafkaProducer(bootstrap_servers=kafka_url)
        
        await self.consumer.start()
        await self.producer.start()
        logger.info("Kafka consumer and producer connected.")

    async def run(self):
        """
        The main event loop. 
        Listens for incoming data, triggers model inference, and broadcasts the results.
        Idempotent design: Safe to restart if the container crashes.
        """
        await self.setup_kafka()
        
        # Reset the intraday memory of the LTC at the start of the session
        self.ltc_monitor.reset_hidden_state()
        
        try:
            async for msg in self.consumer:
                topic = msg.topic
                payload = json.loads(msg.value.decode('utf-8'))
                
                if topic == 'market-data-ticks':
                    await self._process_intraday_tick(payload)
                elif topic == 'macro-data-updates':
                    await self._process_macro_update(payload)
                    
        except asyncio.CancelledError:
            logger.info("Service shutting down...")
        except Exception as e:
            logger.error(f"Critical error in Regime Encoder event loop: {e}")
            raise
        finally:
            await self.consumer.stop()
            await self.producer.stop()
            await self.redis_client.aclose()

    async def _process_intraday_tick(self, payload: Dict):
        """
        Processes high-frequency price data through the Liquid Neural Net.
        """
        current_time = time.time()
        elapsed_seconds = current_time - self.last_tick_time
        self.last_tick_time = current_time
        
        # Extract the 15-dim intraday features required by the LTC
        # (e.g., volume pace, spread z-scores)
        obs_array = np.array(payload['features'], dtype=np.float32)
        
        # 1. Run LTC Inference
        drift_score, urgency_flag = self.ltc_monitor.step(obs_array, elapsed_seconds, device=self.device)
        
        # 2. Update Redis Global State
        await self.redis_client.set("regime:ltc_urgency", float(drift_score))
        await self.redis_client.set("regime:urgency_flag", int(urgency_flag))
        
        # 3. If the LTC detects a flash crash, broadcast an emergency interrupt immediately
        if urgency_flag:
            alert_msg = json.dumps({
                "timestamp": current_time,
                "urgency_score": float(drift_score),
                "trigger": "LTC_INTRADAY_DRIFT"
            })
            await self.producer.send_and_wait('emergency-alerts', alert_msg.encode('utf-8'))
            logger.warning(f"🚨 LTC URGENCY FLAG TRIGGERED! Score: {drift_score:.3f}")

    async def _process_macro_update(self, payload: Dict):
        """
        Processes slower, daily-level macroeconomic or EOD price data through the Mamba-KAN.
        """
        # Fetch the complete 52-dim historical sequence from Redis or TimescaleDB
        # Shape: (Seq_Len, 52)
        historical_sequence = await self._fetch_observation_history()
        
        # 1. Run Mamba-KAN Inference to get the latent posterior distribution
        mu_z, sigma_z = self.mamba_kan.get_posterior(historical_sequence, device=self.device)
        
        # 2. Fetch the latest Topological Data Analysis (TDA) stats from the FPGA
        tda_alert = int(await self.redis_client.get("fpga:tda_alert") or 0)
        
        # 3. Construct the master regime message
        regime_msg = {
            "timestamp": time.time(),
            "z_mu": mu_z.tolist(),
            "z_sigma": sigma_z.tolist(),
            "tda_alert": tda_alert
        }
        
        # 4. Broadcast the new state of the world to the EDT and Execution agents
        await self.producer.send_and_wait('regime-posterior', json.dumps(regime_msg).encode('utf-8'))
        
        # Update cache for fast synchronous reads by other minor services
        await self.redis_client.set("regime:z_mu", json.dumps(mu_z.tolist()))
        
        logger.info(f"Updated Mamba-KAN Regime Posterior broadcasted. TDA Alert Status: {tda_alert}")

    async def _fetch_observation_history(self) -> np.ndarray:
        """
        Helper method to pull the historical state matrix. 
        In production, this queries the `data.pipeline.get_observation_vector()` interface.
        """
        # Scaffolding: Returns a dummy tensor of (Seq_Len=252, Obs_Dim=52)
        return np.random.randn(252, 52).astype(np.float32)

# Standard entry point for running the microservice
if __name__ == "__main__":
    import yaml
    with open("config/hyperparams.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    service = RegimeEncoderService(config)
    asyncio.run(service.run())