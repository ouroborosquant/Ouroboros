"""
FORTRESS v5 - order_manager.py
Path: live/order_manager.py

Stateful Order Manager & MARL Reward Calculator.
Tracks parent-to-child order fills, calculates Implementation Shortfall (IS) in basis points,
and publishes the exact execution cost back to Kafka so the RL agents can learn.
"""

import os
import json
import logging
import asyncio
from typing import Dict, Any

# External Async Dependencies
try:
    import redis.asyncio as redis
    from aiokafka import AIOKafkaProducer
except ImportError:
    raise ImportError("Requires redis and aiokafka packages.")

logger = logging.getLogger("OrderManager")

class OrderManager:
    def __init__(self):
        # We use Redis as an ultra-fast, in-memory state store to track partially filled orders
        self.redis = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
        self.kafka_producer = None

    async def setup(self):
        """Initializes the Kafka producer to broadcast RL rewards."""
        kafka_url = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        self.kafka_producer = AIOKafkaProducer(bootstrap_servers=kafka_url)
        await self.kafka_producer.start()
        logger.info("Order Manager connected to Kafka event bus.")

    async def register_parent_order(self, order_id: str, ticker: str, side: str, 
                                    target_qty: float, arrival_price: float, agent_name: str):
        """
        Locks in the exact state of the market when the trading decision was made.
        This prevents the algorithm from lying to itself about its slippage.
        """
        order_state = {
            "ticker": ticker,
            "side": side.upper(),
            "target_qty": target_qty,
            "filled_qty": 0.0,
            "arrival_price": arrival_price, 
            "vwap_execution": 0.0,
            "agent": agent_name,
            "status": "OPEN"
        }
        
        # Save the dict to Redis hash map
        await self.redis.hset(f"order:{order_id}", mapping=order_state)
        logger.debug(f"Registered Parent Order {order_id} for {ticker} routed to {agent_name}.")

    async def process_fill(self, order_id: str, fill_qty: float, fill_price: float):
        """
        Called by the Alpaca Trade Update WebSocket whenever a child slice actually executes.
        """
        # 1. Fetch current order state from Redis
        raw_state = await self.redis.hgetall(f"order:{order_id}")
        if not raw_state:
            logger.warning(f"Received fill for unknown or already closed order: {order_id}")
            return
            
        state = {k.decode('utf-8'): v.decode('utf-8') for k, v in raw_state.items()}
        
        # 2. Update math for iterative Volume Weighted Average Price (VWAP)
        prev_filled = float(state['filled_qty'])
        prev_vwap = float(state['vwap_execution'])
        
        new_filled = prev_filled + fill_qty
        new_vwap = ((prev_vwap * prev_filled) + (fill_price * fill_qty)) / new_filled
        
        target_qty = float(state['target_qty'])
        status = "CLOSED" if new_filled >= target_qty else "OPEN"
        
        # 3. Save updated state back to Redis
        await self.redis.hset(f"order:{order_id}", mapping={
            "filled_qty": new_filled,
            "vwap_execution": new_vwap,
            "status": status
        })
        
        logger.info(f"Order {order_id} [{state['ticker']}] filled {fill_qty} @ ${fill_price:.2f}. Progress: {new_filled}/{target_qty}")
        
        # 4. If fully executed, calculate the final grade and close the RL loop
        if status == "CLOSED":
            await self._close_feedback_loop(
                order_id=order_id, 
                ticker=state['ticker'], 
                side=state['side'], 
                arrival_price=float(state['arrival_price']), 
                execution_vwap=new_vwap, 
                agent_name=state['agent']
            )

    async def _close_feedback_loop(self, order_id: str, ticker: str, side: str, 
                                   arrival_price: float, execution_vwap: float, agent_name: str):
        """
        Calculates Implementation Shortfall (IS) and sends the penalty/reward to the MARL agents.
        """
        # Buy orders: (Execution - Arrival) / Arrival
        # Sell orders: (Arrival - Execution) / Arrival
        direction_mult = 1.0 if side == 'BUY' else -1.0
        
        shortfall_pct = ((execution_vwap - arrival_price) / arrival_price) * direction_mult
        shortfall_bps = shortfall_pct * 10000.0
        
        logger.info(f"[{ticker}] Execution Complete by {agent_name}. IS: {shortfall_bps:.2f} bps")
        
        # Construct the reward payload for the RL agents
        # The RL objective is to MAXIMIZE reward, so we NEGATE the shortfall cost.
        reward_payload = {
            "order_id": order_id,
            "agent": agent_name,
            "shortfall_bps": shortfall_bps,
            "reward": -shortfall_bps  
        }
        
        # Broadcast the reward to Kafka so train_execution.py can update the neural weights
        if self.kafka_producer:
            await self.kafka_producer.send_and_wait(
                'marl-rewards', 
                json.dumps(reward_payload).encode('utf-8')
            )
        
        # Clean up memory
        await self.redis.delete(f"order:{order_id}")

    async def close(self):
        """Graceful shutdown."""
        if self.kafka_producer:
            await self.kafka_producer.stop()
        await self.redis.aclose()