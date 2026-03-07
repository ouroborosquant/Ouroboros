"""
FORTRESS v5 - execution_svc.py
Path: services/execution_svc.py

Async Microservice: The Execution Router.
Listens to Kafka for target allocations and emergency interrupts.
Routes sub-orders through the MARL Meta-Controller to the Alpaca Trading API.
"""

import os
import json
import numpy as np
import asyncio
import logging
from typing import Dict, Any, List

# External Async Dependencies
try:
    from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
    import redis.asyncio as redis
    from live.alpaca_client import ResilientAlpacaClient
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
except ImportError:
    raise ImportError("Requires aiokafka, redis, and alpaca-py packages.")

# Internal Model Imports
from models.execution.meta_controller import ExecutionMetaController
from models.execution.stealth_ppo import StealthPPO
from models.execution.urgent_ddpg import UrgentDDPG
from models.execution.opportunistic_sac import OpportunisticSAC

logger = logging.getLogger("ExecutionSvc")

class ExecutionService:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 1. Initialize State Storage and Broker API
        self.redis_client = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
        self.alpaca = ResilientAlpacaClient(paper=True)
        
        # 2. Load the MARL Execution Ecosystem
        logger.info("Loading MARL Execution Agents...")
        self.meta_controller = ExecutionMetaController(config.get('meta_controller', {}))
        self.agents = {
            'stealth_ppo': StealthPPO(config.get('stealth_ppo', {})),
            'urgent_ddpg': UrgentDDPG(config.get('urgent_ddpg', {})),
            'opportunistic_sac': OpportunisticSAC(config.get('opport_sac', {}))
        }
        
        # In production, load the trained weights:
        # self.meta_controller.load_state_dict(torch.load('models/weights/meta_controller.pt'))

    async def setup_kafka(self):
        """Sets up async consumers for targets, regimes, and hardware interrupts."""
        kafka_url = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        
        self.consumer = AIOKafkaConsumer(
            'target-weights',
            'emergency-alerts',
            bootstrap_servers=kafka_url,
            group_id='execution_router_group',
            auto_offset_reset='latest'
        )
        
        self.producer = AIOKafkaProducer(bootstrap_servers=kafka_url)
        
        await self.consumer.start()
        await self.producer.start()
        logger.info("Kafka Execution Consumer connected to event bus.")

    async def run(self):
        """Main asynchronous event loop for the execution router."""
        await self.setup_kafka()
        
        try:
            async for msg in self.consumer:
                topic = msg.topic
                payload = json.loads(msg.value.decode('utf-8'))
                
                if topic == 'emergency-alerts':
                    # Highest Priority: Bypass all logic and liquidate
                    await self._handle_emergency_halt(payload)
                elif topic == 'target-weights':
                    # Standard portfolio rebalance
                    await self._process_rebalance(payload)
                    
        except asyncio.CancelledError:
            logger.info("Execution Service shutting down...")
        except Exception as e:
            logger.error(f"Critical execution error: {e}")
            raise
        finally:
            await self.consumer.stop()
            await self.producer.stop()
            await self.redis_client.aclose()

    async def _handle_emergency_halt(self, payload: Dict):
        """
        Triggered directly by the FPGA circuit breaker or the LTC Intraday monitor.
        Cancels all pending limit orders and submits market sell orders for all risk assets.
        """
        logger.critical(f"🛑 EMERGENCY HALT TRIGGERED by {payload.get('trigger')}. Liquidating risk...")
        
        # 1. Cancel all open working orders immediately
        try:
            cancel_responses = self.alpaca.cancel_orders()
            logger.info(f"Canceled {len(cancel_responses)} open orders.")
        except Exception as e:
            logger.error(f"Failed to cancel open orders during halt: {e}")

        # 2. Get current positions
        positions = self.alpaca.get_all_positions()
        
        # 3. Liquidate everything EXCEPT the safe havens (e.g., SHV, BIL)
        safe_havens = self.config.get('safe_havens', ['SHV', 'BIL', 'GLD'])
        
        for pos in positions:
            if pos.symbol not in safe_havens:
                try:
                    order_data = MarketOrderRequest(
                        symbol=pos.symbol,
                        qty=pos.qty,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.GTC
                    )
                    self.alpaca.submit_order(order_data=order_data)
                    logger.critical(f"Submitted emergency market SELL for {pos.qty} shares of {pos.symbol}.")
                except Exception as e:
                    logger.error(f"Failed to liquidate {pos.symbol}: {e}")
                    
        # Publish status to Telegram bot via Redis/Kafka
        await self.redis_client.set("system:halted", "1")

    async def _process_rebalance(self, payload: Dict):
        """
        Translates the EDT's target portfolio weights into DELTA market orders.
        """
        target_weights = payload.get('weights', {})
        
        # 1. Fetch current portfolio value and existing positions
        account = self.alpaca.get_account()
        portfolio_value = float(account.portfolio_value)
        
        raw_positions = self.alpaca.get_all_positions()
        current_positions = {pos.symbol: float(pos.market_value) for pos in raw_positions}
        
        # 2. Fetch the current market regime and microstructure state
        import numpy as np # Fixed import
        z_t = json.loads(await self.redis_client.get("regime:z_mu") or "[]")
        tda_alert = bool(int(await self.redis_client.get("fpga:tda_alert") or 0))
        ltc_urgency = float(await self.redis_client.get("regime:ltc_urgency") or 0.0)
        spread_z, vol_pace = 0.5, 1.2 # Abstracted microstructure
        
        # 3. Select Execution Agent
        selected_agent_name = self.meta_controller.select_agent(
            z_t=np.array(z_t, dtype=np.float32) if z_t else np.zeros(16, dtype=np.float32),
            tda_alert=tda_alert,
            ltc_urgency=ltc_urgency,
            spread_z=spread_z,
            vol_pace=vol_pace
        )
        agent = self.agents[selected_agent_name]
        
        # 4. Calculate Net Delta and Execute
        for ticker, target_weight in target_weights.items():
            target_dollar_amount = portfolio_value * target_weight
            current_dollar_amount = current_positions.get(ticker, 0.0)
            
            # The exact dollar amount we need to buy (+) or sell (-)
            delta_notional = target_dollar_amount - current_dollar_amount
            
            # Only trade if the change is greater than $100 to avoid micro-fees/spread burn
            if abs(delta_notional) > 100.0:
                await self._route_sub_order(ticker, delta_notional, agent, selected_agent_name)
                
    async def _route_sub_order(self, ticker: str, notional_value: float, agent: Any, agent_name: str):
        """
        Passes the order delta to the specific MARL agent (Stealth, Urgent, or Opport)
        to determine the exact price offset and size fraction to submit to the broker.
        """
        # Abstracted state construction for the specific agent
        micro_state = np.zeros(12, dtype=np.float32) 
        
        if agent_name == 'stealth_ppo':
            # Stealth agent decides limit price offset
            action = agent.get_action(micro_state)
            # Example action output: [-0.02, 0.10] -> Place limit 2 bps below mid, for 10% of total size
            
            # Since Alpaca Python SDK handles fractional notional value, we can submit directly
            slice_value = notional_value * float(action[1])
            
            # (In production: Calculate limit price based on current mid-price * (1 + action[0]))
            
            try:
                # Submitting as a basic market order here for scaffolding purposes.
                # A true Stealth implementation slices this into a LimitOrderRequest over hours.
                order_data = MarketOrderRequest(
                    symbol=ticker,
                    notional=slice_value,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY
                )
                self.alpaca.submit_order(order_data=order_data)
                logger.debug(f"Stealth Agent routed {ticker} slice for ${slice_value:.2f}")
            except Exception as e:
                logger.error(f"Failed to route {ticker}: {e}")
                
        elif agent_name == 'urgent_ddpg':
            # Urgent agent guarantees the fill by using market orders immediately
            try:
                order_data = MarketOrderRequest(
                    symbol=ticker,
                    notional=notional_value,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY
                )
                self.alpaca.submit_order(order_data=order_data)
                logger.info(f"Urgent Agent swept {ticker} for full amount ${notional_value:.2f}")
            except Exception as e:
                logger.error(f"Urgent routing failed for {ticker}: {e}")

# Standard entry point for running the microservice
if __name__ == "__main__":
    import yaml
    with open("config/hyperparams.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    service = ExecutionService(config)
    asyncio.run(service.run())