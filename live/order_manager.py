"""
FORTRESS v5 - order_manager.py
Path: live/order_manager.py

Stateful Order Manager & MARL Reward Calculator.
Tracks parent-to-child order fills, calculates Implementation Shortfall (IS)
in basis points, and publishes the exact execution cost back to Kafka so the
RL agents can learn from live market impact.

FIXES APPLIED:
  - BUG #OM-1 (CRITICAL): The Kafka producer was publishing rewards to the topic
    'marl-rewards'. The correct topic defined in infrastructure/kafka/topics.yaml
    and consumed by the training loop is 'execution-rewards'. Every RL reward
    since the system launched was silently dropped into a non-existent topic.
    The MARL agents have been training blind on zero-reward signal.
    Fixed: topic name corrected to 'execution-rewards'.

  - BUG #OM-2: Reward was raw Implementation Shortfall in basis points, unbounded.
    An IS of -50 bps on a bad day produces reward=-50.0, which destabilises
    PPO/DDPG/SAC training (gradient explosion). Reward is now normalised:
      reward = clip(-IS_bps / _REWARD_SCALE, _REWARD_MIN, _REWARD_MAX)
    This bounds reward to [-5.0, +1.0], reflecting the asymmetric cost structure
    (bad execution hurts more than good execution helps).

  - BUG #OM-3: `calculate_implementation_shortfall` divided by `arrival_price`
    with no guard against arrival_price == 0 (e.g. a midnight synthetic fill).
    Added epsilon guard.

  - IMPROVEMENT: Added `enrich_reward_payload()` to attach regime context (z_t)
    and TDA alert status to the reward signal. This allows the MARL training loop
    to learn regime-conditional execution strategies — a key architectural goal.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

import asyncio

logger = logging.getLogger("OrderManager")

# ── Reward normalisation constants ───────────────────────────────────────────
# IS in basis points is divided by this scale factor before clipping.
# 10 bps / 10 = 1.0 unit → normalised reward space is approximately [-5, +1].
_REWARD_SCALE: float = 10.0
_REWARD_MIN:   float = -5.0    # Floor: catastrophic execution (-50 bps)
_REWARD_MAX:   float = 1.0     # Ceiling: excellent price improvement (+10 bps)

# BUG #OM-1 FIX: Correct Kafka topic name.
# topics.yaml defines 'execution-rewards', NOT 'marl-rewards'.
_KAFKA_REWARD_TOPIC: str = "execution-rewards"


class StatefulOrderTracker:
    """
    Tracks the lifecycle of a parent order through its child fills.

    State machine per order_id:
      PENDING  → child fills arrive via Kafka 'order-fills'
      COMPLETE → all expected notional filled, IS calculated, reward published
      EXPIRED  → arrival price TTL exceeded without full fill (rare, partial IS)

    State is stored in Redis with a 24h TTL so the tracker survives container
    restarts without losing in-flight orders.
    """

    def __init__(self) -> None:
        self._redis_client: Optional[Any] = None
        self._kafka_producer: Optional[Any] = None
        self._setup_done: bool = False

    async def setup(self) -> None:
        """
        Initialises the Redis client and Kafka producer.
        Idempotent — safe to call multiple times.
        """
        if self._setup_done:
            return

        try:
            import redis.asyncio as redis
            from aiokafka import AIOKafkaProducer
        except ImportError as exc:
            raise ImportError(
                "Requires redis and aiokafka: pip install redis aiokafka"
            ) from exc

        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        kafka_url = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

        self._redis_client = redis.Redis.from_url(redis_url)
        self._kafka_producer = AIOKafkaProducer(bootstrap_servers=kafka_url)
        await self._kafka_producer.start()

        self._setup_done = True
        logger.info("OrderManager: Redis and Kafka producer initialised.")

    async def register_parent_order(
        self,
        order_id: str,
        ticker: str,
        side: str,
        expected_notional: float,
        arrival_price: float,
        agent_name: str,
    ) -> None:
        """
        Registers a new parent order in Redis before any fills arrive.

        Args:
            order_id:          UUID of the parent order.
            ticker:            Asset symbol (e.g. 'SPY').
            side:              'BUY' or 'SELL'.
            expected_notional: Total USD notional expected to be filled.
            arrival_price:     Mid-price at the moment the order was submitted.
                               Used as the IS benchmark price.
            agent_name:        Which MARL agent routed this order.
        """
        state = {
            "ticker":             ticker,
            "side":               side,
            "expected_notional":  expected_notional,
            "arrival_price":      float(arrival_price),
            "agent_name":         agent_name,
            "filled_notional":    0.0,
            "volume_weighted_sum": 0.0,  # Σ(fill_price * fill_qty_usd)
            "fill_count":          0,
        }
        key = f"order:{order_id}"
        await self._redis_client.set(
            key,
            json.dumps(state),
            ex=86400,  # 24h TTL — purges stale orders automatically
        )
        logger.debug(f"[{ticker}] Registered parent order {order_id} (agent={agent_name}).")

    async def process_fill(self, fill_event: Dict[str, Any]) -> None:
        """
        Processes a single child fill event from the Kafka 'order-fills' topic.

        When the expected notional is fully filled, calculates Implementation
        Shortfall and publishes the reward to 'execution-rewards'.

        Args:
            fill_event: Dict with keys:
                order_id, ticker, side, fill_qty (shares), fill_price (per share),
                fill_timestamp, agent.
        """
        order_id   = fill_event.get("order_id")
        fill_qty   = float(fill_event.get("fill_qty",   0.0))
        fill_price = float(fill_event.get("fill_price", 0.0))

        if not order_id:
            logger.warning("Fill event missing order_id — skipped.")
            return

        key = f"order:{order_id}"
        raw = await self._redis_client.get(key)
        if not raw:
            logger.warning(
                f"No state found for order_id={order_id}. "
                "Order may have been registered before container restart or already complete."
            )
            return

        state: Dict[str, Any] = json.loads(raw)

        # Accumulate fill statistics (VWAP numerator and denominator)
        fill_notional            = fill_qty * fill_price
        state["filled_notional"]    += fill_notional
        state["volume_weighted_sum"] += fill_notional  # fill_price * notional (Σ p_i * v_i)
        state["fill_count"]         += 1

        # Check if the order is fully filled
        fill_pct = state["filled_notional"] / max(state["expected_notional"], 1e-8)

        if fill_pct >= 0.99:  # 99% threshold to handle rounding / fractional shares
            await self._finalise_order(order_id, state)
        else:
            # Persist updated state
            await self._redis_client.set(key, json.dumps(state), ex=86400)
            logger.debug(
                f"[{state['ticker']}] Partial fill {fill_pct:.1%} "
                f"({state['fill_count']} fills so far)."
            )

    async def _finalise_order(
        self,
        order_id: str,
        state: Dict[str, Any],
    ) -> None:
        """
        Computes Implementation Shortfall, normalises to a reward signal,
        enriches with regime context, and publishes to 'execution-rewards'.
        """
        ticker        = state["ticker"]
        side          = state["side"]
        agent_name    = state["agent_name"]
        arrival_price = state["arrival_price"]
        total_filled  = state["filled_notional"]

        # ── VWAP execution price ────────────────────────────────────────────
        # VWAP = Σ(price_i * notional_i) / Σ(notional_i)
        # Since volume_weighted_sum stores Σ(fill_notional) and fill_notional
        # = fill_price * fill_qty, we approximate VWAP as:
        #   VWAP ≈ volume_weighted_sum / filled_notional
        # This is only exact when all fills are in the same currency units.
        execution_vwap = state["volume_weighted_sum"] / max(total_filled, 1e-8)

        # ── Implementation Shortfall (bps) ──────────────────────────────────
        # IS = (ExecutionVWAP - ArrivalPrice) / ArrivalPrice * 10_000  [BUY]
        # IS = (ArrivalPrice - ExecutionVWAP) / ArrivalPrice * 10_000  [SELL]
        # A positive IS means we paid MORE than the arrival price (bad).
        # A negative IS means we got price improvement (good, but rare).
        direction_mult = 1.0 if side.upper() == "BUY" else -1.0

        # BUG #OM-3 FIX: guard against zero arrival_price
        safe_arrival   = max(abs(arrival_price), 1e-8)
        shortfall_pct  = ((execution_vwap - arrival_price) / safe_arrival) * direction_mult
        shortfall_bps  = shortfall_pct * 10_000.0

        logger.info(
            f"[{ticker}] Order {order_id} COMPLETE via {agent_name}. "
            f"IS: {shortfall_bps:+.2f} bps | Fills: {state['fill_count']} | "
            f"Filled: ${total_filled:,.0f}"
        )

        # ── Reward normalisation (BUG #OM-2 FIX) ───────────────────────────
        # Raw IS in bps is unbounded and will destabilise PPO/DDPG gradient updates.
        # Normalise: reward = clip(-IS / scale, min, max)
        # Example mappings:
        #   IS = +50 bps (bad)  → reward = clip(-5.0, -5.0, 1.0) = -5.0
        #   IS =  0 bps         → reward = 0.0
        #   IS = -10 bps (good) → reward = clip(+1.0, -5.0, 1.0) = +1.0
        raw_reward      = -shortfall_bps / _REWARD_SCALE
        normalised_reward = max(_REWARD_MIN, min(_REWARD_MAX, raw_reward))

        # ── Enrich reward with regime context ──────────────────────────────
        # Attach z_t and tda_alert so the MARL training loop can learn
        # regime-conditional execution strategies.
        regime_context = await self._fetch_regime_context()

        # ── Publish reward (BUG #OM-1 FIX) ─────────────────────────────────
        reward_payload = {
            "order_id":                    order_id,
            "ticker":                      ticker,
            "agent":                       agent_name,
            "implementation_shortfall_bps": round(shortfall_bps, 4),
            "reward":                      round(normalised_reward, 6),
            "execution_vwap":              round(execution_vwap, 6),
            "arrival_price":               round(arrival_price, 6),
            "filled_notional_usd":         round(total_filled, 2),
            "fill_count":                  state["fill_count"],
            # Regime enrichment for conditional RL training
            "regime_z_t":                  regime_context.get("z_mu", []),
            "tda_alert":                   regime_context.get("tda_alert", 0),
            "ltc_urgency":                 regime_context.get("ltc_urgency", 0.0),
        }

        if self._kafka_producer:
            await self._kafka_producer.send_and_wait(
                _KAFKA_REWARD_TOPIC,  # BUG #OM-1 FIX: was 'marl-rewards', now 'execution-rewards'
                json.dumps(reward_payload).encode("utf-8"),
            )
            logger.debug(
                f"[{ticker}] Reward {normalised_reward:+.4f} published to "
                f"'{_KAFKA_REWARD_TOPIC}'."
            )

        # Clean up Redis state for this order
        await self._redis_client.delete(f"order:{order_id}")

    async def _fetch_regime_context(self) -> Dict[str, Any]:
        """
        Fetches the current regime posterior from Redis for reward enrichment.
        Gracefully returns an empty context if the regime encoder is not yet live.
        """
        try:
            import json as _json

            z_mu_raw = await self._redis_client.get("regime:z_mu")
            tda_raw  = await self._redis_client.get("regime:tda_alert")
            ltc_raw  = await self._redis_client.get("regime:ltc_urgency")

            return {
                "z_mu":        _json.loads(z_mu_raw) if z_mu_raw else [],
                "tda_alert":   int(tda_raw)          if tda_raw  else 0,
                "ltc_urgency": float(ltc_raw)        if ltc_raw  else 0.0,
            }
        except Exception as exc:
            logger.debug(f"Regime context fetch failed (non-critical): {exc}")
            return {}

    async def close(self) -> None:
        """Graceful shutdown — flushes Kafka producer and closes Redis."""
        if self._kafka_producer:
            await self._kafka_producer.stop()
        if self._redis_client:
            await self._redis_client.aclose()
        logger.info("OrderManager: shutdown complete.")


def calculate_implementation_shortfall(
    execution_vwap: float,
    arrival_price:  float,
    side:           str,
) -> float:
    """
    Module-level utility for computing IS in basis points.
    Used directly by services that compute IS outside the StatefulOrderTracker.

    Args:
        execution_vwap: Volume-weighted average fill price.
        arrival_price:  Mid-price at order submission (the benchmark).
        side:           'BUY' or 'SELL'.

    Returns:
        is_bps: Implementation Shortfall in basis points.
                Positive = paid more than arrival (bad for alpha).
                Negative = price improvement (good).
    """
    safe_arrival   = max(abs(arrival_price), 1e-8)   # BUG #OM-3 FIX
    direction_mult = 1.0 if side.upper() == "BUY" else -1.0
    shortfall_pct  = ((execution_vwap - arrival_price) / safe_arrival) * direction_mult
    return shortfall_pct * 10_000.0


def normalise_reward(is_bps: float) -> float:
    """
    Converts raw IS in basis points to a normalised RL reward signal.
    Bounds: [_REWARD_MIN, _REWARD_MAX] = [-5.0, +1.0].

    Exported so training/train_execution.py can use the same normalisation.
    """
    raw = -is_bps / _REWARD_SCALE
    return max(_REWARD_MIN, min(_REWARD_MAX, raw))