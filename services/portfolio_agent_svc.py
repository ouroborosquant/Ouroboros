"""
FORTRESS v5 - portfolio_agent_svc.py
Path: services/portfolio_agent_svc.py

Async Microservice: The Capital Allocator.
Listens to Kafka for regime updates, triggers the Elastic Decision Transformer (EDT) 
and Deep Hedging network, and broadcasts target portfolio weights to the execution router.
"""

import os
import json
import asyncio
import logging
import numpy as np
import torch
from typing import Dict, Any

# Async Kafka and Redis clients
try:
    from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
    import redis.asyncio as redis
except ImportError:
    raise ImportError("Requires aiokafka and redis packages.")

# Internal Model Imports
from models.portfolio.edt_agent import ElasticDecisionTransformer
from models.hedging.deep_hedging import DeepHedgingNetwork
import yaml

logger = logging.getLogger("PortfolioAgentSvc")

class PortfolioAgentService:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.redis_client = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
        
        logger.info("Loading EDT and Deep Hedging models into VRAM...")
        self.edt = ElasticDecisionTransformer(config.get('edt', {}))
        self.edt.to(self.device)
        self.edt.eval()
        
        self.hedger = DeepHedgingNetwork(config.get('hedging', {}))
        self.hedger.to(self.device)
        self.hedger.eval()
        
        # Load the ETF universe list to map the integer indices back to string tickers
        with open('config/universe.yaml', 'r') as f:
            univ = yaml.safe_load(f)
            self.universe_tickers = [asset['ticker'] for asset in univ.get('assets', [])]

    async def setup_kafka(self):
        kafka_url = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        
        self.consumer = AIOKafkaConsumer(
            'regime-posterior', # Listens to the output of regime_encoder_svc
            bootstrap_servers=kafka_url,
            group_id='portfolio_agent_group',
            auto_offset_reset='latest'
        )
        self.producer = AIOKafkaProducer(bootstrap_servers=kafka_url)
        
        await self.consumer.start()
        await self.producer.start()
        logger.info("Kafka Portfolio Consumer connected to event bus.")

    async def run(self):
        await self.setup_kafka()
        
        try:
            async for msg in self.consumer:
                payload = json.loads(msg.value.decode('utf-8'))
                await self._process_allocation(payload)
                    
        except asyncio.CancelledError:
            logger.info("Portfolio Service shutting down...")
        except Exception as e:
            logger.error(f"Critical error in Portfolio Agent loop: {e}")
            raise
        finally:
            await self.consumer.stop()
            await self.producer.stop()
            await self.redis_client.aclose()

    async def _process_allocation(self, regime_payload: Dict):
        """
        Calculates base weights via EDT, overlays Deep Hedging constraints, 
        and publishes the final target weights for execution.
        """
        z_mu = np.array(regime_payload.get('z_mu', []), dtype=np.float32)
        tda_alert = bool(regime_payload.get('tda_alert', 0))
        
        # In production, we build the full 192-dim state (obs + z_t + alphas) here
        # Scaffold: Dummy 192-dim state
        full_state = np.random.randn(self.config.get('edt', {}).get('state_dim', 192)).astype(np.float32)
        
        # 1. Base Portfolio Allocation (Elastic Decision Transformer)
        # Assuming a default 10% target return prompt for the Transformer
        target_return = 0.10 
        mean_weights, std_weights = self.edt.get_weights(full_state, target_return, device=self.device)
        
        # Map numpy array to dictionary
        base_allocation = {ticker: float(weight) for ticker, weight in zip(self.universe_tickers, mean_weights)}
        
        # 2. Risk Management & Hedging Overlay
        # Fetch current portfolio state [leverage, drawdown, unrealized_pnl]
        port_state = np.array([1.0, -0.02, 0.05], dtype=np.float32) 
        ltc_urgency = float(await self.redis_client.get("regime:ltc_urgency") or 0.0)
        
        # Simulated probability of crash from World Model / SDE
        crash_prob = 0.15 
        
        hedge_overlay = self.hedger.get_hedge_overlay(
            z_t=z_mu, 
            portfolio_state=port_state, 
            tda_alert=tda_alert, 
            ltc_urgency=ltc_urgency, 
            crash_probability=crash_prob
        )
        
        # 3. Combine Base and Hedge
        # Simple override strategy: If the hedger demands 10% VIXY, we subtract that 10% 
        # proportionately from the base equity allocations.
        final_allocation = self._merge_allocations(base_allocation, hedge_overlay)
        
        # 4. Broadcast the final targets to the Execution Router
        out_msg = {
            "timestamp": regime_payload['timestamp'],
            "weights": final_allocation
        }
        
        await self.producer.send_and_wait('target-weights', json.dumps(out_msg).encode('utf-8'))
        logger.info(f"Target weights calculated and broadcasted. Top holdings: {self._get_top_holdings(final_allocation)}")

    def _merge_allocations(self, base: Dict[str, float], hedge: Dict[str, float]) -> Dict[str, float]:
        """Blends the EDT output with the Deep Hedging output cleanly."""
        combined = base.copy()
        hedge_total = sum(hedge.values())
        
        if hedge_total > 0:
            # Scale down base positions to make room for hedges
            scale_factor = 1.0 - min(hedge_total, 1.0)
            for k in combined.keys():
                combined[k] *= scale_factor
                
            # Add hedge positions
            for k, v in hedge.items():
                combined[k] = combined.get(k, 0.0) + v
                
        # Normalize just to be safe
        total = sum(combined.values())
        if total > 0:
            combined = {k: v / total for k, v in combined.items()}
            
        return combined

    def _get_top_holdings(self, alloc: Dict[str, float], n: int = 3) -> str:
        sorted_items = sorted(alloc.items(), key=lambda item: item[1], reverse=True)
        return ", ".join([f"{k}: {v:.1%}" for k, v in sorted_items[:n]])

if __name__ == "__main__":
    with open("config/hyperparams.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    service = PortfolioAgentService(config)
    asyncio.run(service.run())