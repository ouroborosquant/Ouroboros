"""
FORTRESS v5 - execution_svc.py
Path: services/execution_svc.py

Async Microservice: The Execution Router.
Listens to Kafka for target allocations and emergency interrupts.
Routes sub-orders through the MARL Meta-Controller to the Alpaca Trading API.

FIXES APPLIED:
  - BUG #2: _handle_emergency_halt called `self.alpaca.cancel_orders()` (AttributeError).
            Corrected to `self.alpaca.cancel_all_orders()` to match the wrapper method name.
  - BUG #3: _route_sub_order hardcoded `OrderSide.BUY` for all agents. A negative
            delta_notional (sell rebalance) was structurally impossible. Side is now
            derived from the sign of `notional_value`.
  - BUG #4: The `opportunistic_sac` branch was entirely absent from _route_sub_order.
            When MetaController selected it, the order was silently dropped. Added.
  - BUG #7: All blocking Alpaca SDK calls (get_account, get_all_positions, submit_order)
            were called synchronously inside async methods, freezing the event loop.
            All sync calls are now offloaded via `asyncio.get_event_loop().run_in_executor`.
  - BUG #13: `import numpy as np` was inline inside _process_rebalance. Moved to module top.
"""

import os
import json
import asyncio
import logging
from typing import Dict, Any, List

import numpy as np

# External Async Dependencies
try:
    from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
    import redis.asyncio as redis
    from live.alpaca_client import ResilientAlpacaClient, RateLimitError, BrokerAPIError
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

# Minimum notional delta to trigger a trade — avoids micro-fee burn on noise.
_MIN_TRADE_NOTIONAL_USD: float = 100.0

# Maximum retry attempts when the broker returns a 429 before giving up on a slice.
_MAX_RATE_LIMIT_RETRIES: int = 3
_RATE_LIMIT_BACKOFF_SECONDS: float = 5.0


class ExecutionService:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # 1. Initialize State Storage and Broker API
        self.redis_client = redis.Redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379")
        )
        self.alpaca = ResilientAlpacaClient(paper=True)

        # Cache the event loop reference for run_in_executor calls
        self._loop: asyncio.AbstractEventLoop | None = None

        # 2. Load the MARL Execution Ecosystem
        logger.info("Loading MARL Execution Agents...")
        self.meta_controller = ExecutionMetaController(config.get("meta_controller", {}))
        self.agents = {
            "stealth_ppo": StealthPPO(config.get("stealth_ppo", {})),
            "urgent_ddpg": UrgentDDPG(config.get("urgent_ddpg", {})),
            # FIX #4: OpportunisticSAC was missing from the routing logic entirely.
            # Config key is 'opport_sac' to match hyperparams.yaml.
            "opportunistic_sac": OpportunisticSAC(config.get("opport_sac", {})),
        }
        # In production, load trained weights:
        # self.meta_controller.load_state_dict(torch.load('models/weights/meta_controller.pt'))

    async def _run_sync(self, fn, *args) -> Any:
        """
        FIX #7: Offloads blocking synchronous Alpaca SDK calls to a thread pool executor.
        This prevents any single broker call from freezing the event loop and blocking
        the emergency-alerts consumer.

        Usage: result = await self._run_sync(self.alpaca.get_account)
        """
        loop = self._loop or asyncio.get_event_loop()
        return await loop.run_in_executor(None, fn, *args)

    async def setup_kafka(self):
        """Sets up async consumers for allocation targets and emergency interrupts."""
        kafka_url = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

        self.consumer = AIOKafkaConsumer(
            "target-weights",
            "emergency-alerts",
            bootstrap_servers=kafka_url,
            group_id="execution_router_group",
            auto_offset_reset="latest",
        )

        self.producer = AIOKafkaProducer(bootstrap_servers=kafka_url)

        await self.consumer.start()
        await self.producer.start()
        logger.info("Kafka Execution Consumer connected to event bus.")

    async def run(self):
        """Main asynchronous event loop for the execution router."""
        self._loop = asyncio.get_event_loop()
        await self.setup_kafka()

        try:
            async for msg in self.consumer:
                topic = msg.topic
                payload = json.loads(msg.value.decode("utf-8"))

                if topic == "emergency-alerts":
                    # Highest Priority: bypass all allocation logic and liquidate immediately.
                    await self._handle_emergency_halt(payload)
                elif topic == "target-weights":
                    await self._process_rebalance(payload)

        except asyncio.CancelledError:
            logger.info("Execution Service shutting down gracefully...")
        except Exception as e:
            logger.error(f"Critical execution error: {e}")
            raise
        finally:
            await self.consumer.stop()
            await self.producer.stop()
            await self.redis_client.aclose()

    async def _handle_emergency_halt(self, payload: Dict):
        """
        Triggered by FPGA circuit breaker or LTC intraday monitor.
        Cancels all pending limit orders and submits market sell orders for all risk assets.

        FIX #2: Previously called `self.alpaca.cancel_orders()` which does not exist.
                Corrected to `self.alpaca.cancel_all_orders()`.
        FIX #7: Broker calls now run in executor threads to avoid blocking the event loop.
        """
        logger.critical(
            f"🛑 EMERGENCY HALT TRIGGERED by {payload.get('trigger')}. Liquidating risk..."
        )

        # 1. Cancel all open working orders immediately.
        # FIX #2: method name was cancel_orders(), corrected to cancel_all_orders().
        try:
            cancel_responses = await self._run_sync(self.alpaca.cancel_all_orders)
            logger.info(f"Canceled {len(cancel_responses)} open orders.")
        except Exception as e:
            logger.error(f"Failed to cancel open orders during halt: {e}")

        # 2. Get current positions (non-blocking).
        try:
            positions = await self._run_sync(self.alpaca.get_all_positions)
        except Exception as e:
            logger.error(f"Failed to fetch positions during halt: {e}")
            positions = []

        # 3. Liquidate all risk assets; preserve safe havens.
        safe_havens = self.config.get("safe_havens", ["SHV", "BIL", "GLD"])

        for pos in positions:
            if pos.symbol not in safe_havens:
                try:
                    order_data = MarketOrderRequest(
                        symbol=pos.symbol,
                        qty=pos.qty,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.GTC,
                    )
                    # FIX #7: submit_order offloaded to executor.
                    await self._run_sync(self.alpaca.submit_order, order_data)
                    logger.critical(
                        f"Emergency SELL submitted: {pos.qty} shares of {pos.symbol}."
                    )
                except BrokerAPIError as e:
                    logger.error(f"BrokerAPIError liquidating {pos.symbol}: {e}")
                except Exception as e:
                    logger.error(f"Failed to liquidate {pos.symbol}: {e}")

        # Mark system as halted in Redis so all other services can read the flag.
        await self.redis_client.set("system:halted", "1")

    async def _process_rebalance(self, payload: Dict):
        """
        Translates the EDT's target portfolio weights into net-delta market orders.

        FIX #7: get_account and get_all_positions offloaded to executor threads.
        FIX #13: `import numpy as np` moved to module top.
        """
        # Guard: refuse to trade if the system is in a halted state.
        halted = await self.redis_client.get("system:halted")
        if halted and int(halted):
            logger.warning("Rebalance received but system is HALTED. Discarding.")
            return

        target_weights = payload.get("weights", {})

        # 1. Fetch account value and existing positions — non-blocking.
        # FIX #7: Run synchronous Alpaca calls in thread executor.
        account = await self._run_sync(self.alpaca.get_account)
        portfolio_value = float(account.portfolio_value)

        raw_positions = await self._run_sync(self.alpaca.get_all_positions)
        current_positions = {
            pos.symbol: float(pos.market_value) for pos in raw_positions
        }

        # 2. Fetch regime and microstructure state from Redis.
        z_t_raw = await self.redis_client.get("regime:z_mu")
        tda_alert = bool(int(await self.redis_client.get("fpga:tda_alert") or 0))
        ltc_urgency = float(await self.redis_client.get("regime:ltc_urgency") or 0.0)

        z_t_list = json.loads(z_t_raw) if z_t_raw else []
        z_t = (
            np.array(z_t_list, dtype=np.float32)
            if z_t_list
            else np.zeros(16, dtype=np.float32)
        )

        # TODO: Pull live microstructure from Redis L2 state keys.
        spread_z: float = float(await self.redis_client.get("micro:spread_z") or 0.5)
        vol_pace: float = float(await self.redis_client.get("micro:vol_pace") or 1.0)

        # 3. Select the execution agent via the MetaController policy network.
        selected_agent_name = self.meta_controller.select_agent(
            z_t=z_t,
            tda_alert=tda_alert,
            ltc_urgency=ltc_urgency,
            spread_z=spread_z,
            vol_pace=vol_pace,
        )
        agent = self.agents[selected_agent_name]
        logger.info(f"MetaController selected agent: [{selected_agent_name}]")

        # 4. Calculate net delta for each asset and dispatch sub-orders.
        for ticker, target_weight in target_weights.items():
            target_dollar_amount = portfolio_value * target_weight
            current_dollar_amount = current_positions.get(ticker, 0.0)

            # Signed notional: positive = BUY, negative = SELL.
            delta_notional = target_dollar_amount - current_dollar_amount

            if abs(delta_notional) > _MIN_TRADE_NOTIONAL_USD:
                await self._route_sub_order(
                    ticker, delta_notional, agent, selected_agent_name
                )

    async def _route_sub_order(
        self,
        ticker: str,
        notional_value: float,
        agent: Any,
        agent_name: str,
    ) -> None:
        """
        Routes the order delta to the appropriate MARL execution agent.

        FIX #3: OrderSide is now derived from the sign of `notional_value`.
                Previously hardcoded to BUY, making sells structurally impossible.
        FIX #4: Added `opportunistic_sac` branch. Previously missing, causing
                silent order drops when MetaController selected this agent.
        FIX #7: submit_order offloaded to thread executor.

        Args:
            ticker:        Asset symbol.
            notional_value: Signed USD delta. Positive = BUY, negative = SELL.
            agent:         The MARL agent instance selected by MetaController.
            agent_name:    String key identifying the agent type.
        """
        # FIX #3: Derive trade direction from the sign of the notional delta.
        side = OrderSide.BUY if notional_value > 0 else OrderSide.SELL
        notional_abs = abs(notional_value)

        # Build the 12-dim L2 order book state for the agent.
        # TODO: Populate from Redis `l2_state:{ticker}` hash.
        micro_state = np.zeros(12, dtype=np.float32)

        try:
            if agent_name == "stealth_ppo":
                # Stealth agent: slice orders over time to minimize market impact.
                # action[0] = price offset fraction, action[1] = size fraction (clamped [0,1])
                action = agent.get_action(micro_state)
                size_fraction = float(np.clip(action[1], 0.0, 1.0))
                slice_notional = notional_abs * size_fraction

                if slice_notional < _MIN_TRADE_NOTIONAL_USD:
                    logger.debug(f"Stealth slice for {ticker} below min notional. Skipping.")
                    return

                # TODO: Replace MarketOrderRequest with a LimitOrderRequest using
                # mid_price * (1 + action[0]) for true stealth execution.
                order_data = MarketOrderRequest(
                    symbol=ticker,
                    notional=round(slice_notional, 2),
                    side=side,  # FIX #3: was hardcoded BUY
                    time_in_force=TimeInForce.DAY,
                )
                await self._submit_with_rate_limit_retry(order_data)
                logger.debug(
                    f"[Stealth] {side.value} {ticker} slice ${slice_notional:.2f}"
                )

            elif agent_name == "urgent_ddpg":
                # Urgent agent: sweep the full notional with a single market order.
                order_data = MarketOrderRequest(
                    symbol=ticker,
                    notional=round(notional_abs, 2),
                    side=side,  # FIX #3: was hardcoded BUY
                    time_in_force=TimeInForce.DAY,
                )
                await self._submit_with_rate_limit_retry(order_data)
                logger.info(
                    f"[Urgent] {side.value} {ticker} full notional ${notional_abs:.2f}"
                )

            # FIX #4: Added missing opportunistic_sac branch.
            # Previously, when the MetaController selected this agent, _route_sub_order
            # fell through without executing — the delta was silently dropped.
            elif agent_name == "opportunistic_sac":
                # Opportunistic agent: exploit liquidity windows for improved fill prices.
                # action[0] = price limit offset, action[1] = size fraction (clamped [0,1])
                action = agent.get_action(micro_state)
                size_fraction = float(np.clip(action[1], 0.0, 1.0))
                slice_notional = notional_abs * size_fraction

                if slice_notional < _MIN_TRADE_NOTIONAL_USD:
                    logger.debug(
                        f"Opportunistic slice for {ticker} below min notional. Skipping."
                    )
                    return

                # TODO: Replace with LimitOrderRequest once live mid-price is wired.
                order_data = MarketOrderRequest(
                    symbol=ticker,
                    notional=round(slice_notional, 2),
                    side=side,  # FIX #3: correct side from signed notional
                    time_in_force=TimeInForce.DAY,
                )
                await self._submit_with_rate_limit_retry(order_data)
                logger.debug(
                    f"[Opportunistic] {side.value} {ticker} slice ${slice_notional:.2f}"
                )

            else:
                # This should never occur unless MetaController returns an unknown agent name.
                logger.error(
                    f"Unknown agent name '{agent_name}' — order for {ticker} NOT submitted."
                )

        except BrokerAPIError as e:
            logger.error(f"Non-retryable broker error for {ticker}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error routing sub-order for {ticker}: {e}")

    async def _submit_with_rate_limit_retry(self, order_data: Any) -> Any:
        """
        FIX #6 / FIX #7: Wraps submit_order with async-safe rate limit handling.
        Uses `await asyncio.sleep()` instead of the blocking `time.sleep()` that
        previously existed inside alpaca_client.submit_order.

        On RateLimitError, backs off asynchronously and retries up to
        _MAX_RATE_LIMIT_RETRIES times before abandoning the slice.
        """
        for attempt in range(1, _MAX_RATE_LIMIT_RETRIES + 1):
            try:
                return await self._run_sync(self.alpaca.submit_order, order_data)
            except RateLimitError:
                if attempt < _MAX_RATE_LIMIT_RETRIES:
                    backoff = _RATE_LIMIT_BACKOFF_SECONDS * attempt
                    logger.warning(
                        f"Rate limit hit (attempt {attempt}/{_MAX_RATE_LIMIT_RETRIES}). "
                        f"Backing off {backoff}s asynchronously..."
                    )
                    await asyncio.sleep(backoff)
                else:
                    logger.error(
                        f"Rate limit persisted after {_MAX_RATE_LIMIT_RETRIES} retries. "
                        "Order slice abandoned."
                    )
                    raise


# Standard entry point for running the microservice
if __name__ == "__main__":
    import yaml

    with open("config/hyperparams.yaml", "r") as f:
        config = yaml.safe_load(f)

    service = ExecutionService(config)
    asyncio.run(service.run())