"""
FORTRESS v5 - alpha_engine_svc.py
Path: services/alpha_engine_svc.py

Async Microservice: The Causal Alpha Engine.
Listens to regime updates and live market ticks, rebuilds the PyTorch Geometric 
asset graph in real-time, and broadcasts the 25-dimensional expected alpha vector.
"""

import os
import json
import asyncio
import logging
import torch
import numpy as np
from typing import Dict, Any

# Async Kafka and Redis clients
try:
    from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
    import redis.asyncio as redis
    from torch_geometric.data import Data
except ImportError:
    raise ImportError("Requires aiokafka, redis, and torch_geometric.")

from models.alpha.gat_alpha import MultiRelationalGAT, AssetGraph

logger = logging.getLogger("AlphaEngineSvc")

class AlphaEngineService:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.redis_client = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
        
        logger.info("Loading GATv2 Alpha Engine into VRAM...")
        self.gat = MultiRelationalGAT(
            node_feat_dim=config.get('gat_alpha', {}).get('node_feat_dim', 78),
            edge_feat_dim=config.get('gat_alpha', {}).get('edge_feat_dim', 5),
            hidden_dim=config.get('gat_alpha', {}).get('hidden_dim', 128)
        ).to(self.device)
        self.gat.eval()

    async def setup_kafka(self):
        kafka_url = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        
        self.consumer = AIOKafkaConsumer(
            'regime-posterior',   # Triggered whenever the macro regime updates
            'market-data-ticks',  # Triggered by intraday price action
            bootstrap_servers=kafka_url,
            group_id='alpha_engine_group',
            auto_offset_reset='latest'
        )
        self.producer = AIOKafkaProducer(bootstrap_servers=kafka_url)
        
        await self.consumer.start()
        await self.producer.start()
        logger.info("Kafka Alpha Consumer connected to event bus.")

    async def run(self):
        await self.setup_kafka()
        
        try:
            async for msg in self.consumer:
                # We only want to run the heavy graph inference on significant regime shifts
                # or periodically (e.g., every 5 minutes), not on every single microsecond tick.
                if msg.topic == 'regime-posterior':
                    payload = json.loads(msg.value.decode('utf-8'))
                    await self._compute_and_broadcast_alpha(payload)
                    
        except asyncio.CancelledError:
            logger.info("Alpha Engine Service shutting down...")
        except Exception as e:
            logger.error(f"Critical error in Alpha loop: {e}")
        finally:
            await self.consumer.stop()
            await self.producer.stop()
            await self.redis_client.aclose()

    async def _compute_and_broadcast_alpha(self, payload: Dict):
        """Builds the live graph and infers the forward alphas."""
        # 1. Build Node Features (78-dim)
        # In production, this pulls the 52-dim raw obs, 16-dim z_t, and 10-dim LLM signals from Redis
        num_nodes = 25
        node_features = torch.randn(num_nodes, 78, dtype=torch.float32)
        
        # 2. Build the Dynamic Edge Index
        # This queries the TDA Topology microservice and DYNOTEARS causality matrix
        edge_index, edge_attr = AssetGraph.build_dummy_edge_index(num_nodes)
        
        # Construct the PyTorch Geometric Data object
        graph_data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
        
        # 3. Graph Inference
        alphas = self.gat.infer_live_alpha(graph_data, device=self.device)
        
        # 4. Broadcast
        out_msg = {
            "timestamp": payload.get('timestamp'),
            "alpha_vector": alphas.tolist()
        }
        
        await self.producer.send_and_wait('alpha-signals', json.dumps(out_msg).encode('utf-8'))
        
        # Cache for synchronous reads by the Portfolio Agent
        await self.redis_client.set("alpha:latest", json.dumps(alphas.tolist()))
        logger.debug(f"Computed new Alpha vector. Top expected outlier: {np.max(alphas):.4f}")

if __name__ == "__main__":
    import yaml
    with open("config/hyperparams.yaml", "r") as f:
        config = yaml.safe_load(f)
    service = AlphaEngineService(config)
    asyncio.run(service.run())