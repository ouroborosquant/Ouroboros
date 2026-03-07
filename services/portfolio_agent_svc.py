"""
FORTRESS v5 - portfolio_agent_svc.py
Path: services/portfolio_agent_svc.py

Async Microservice: The Capital Allocator.
Listens to Kafka for regime updates, triggers the Elastic Decision Transformer (EDT)
and Deep Hedging network, and broadcasts target portfolio weights to the execution router.

FIXES APPLIED:
  - BUG #5: The 192-dim EDT state vector was assembled as `np.random.randn(192)` — random
            noise was being submitted to the broker as a capital allocation signal.
            The state is now assembled from its three real components:
              [obs_dim=52]    — raw market observation fetched from Redis (key: obs:current)
              [latent_dim=16] — Mamba-KAN regime posterior (key: regime:z_mu)
              [alpha_dim=124] — GATv2 alpha scores per asset (key: alpha:scores)
            Each component falls back gracefully to zeros if the upstream service has not
            yet published its output (e.g., at startup). The EDT will NOT allocate from
            random noise under any condition.
  - BUG #14: Added explicit gross-leverage assertion before broadcasting weights.
             If sum(abs(weights)) > max_gross_leverage, allocation is rejected and
             the previous weights are held.
"""

import os
import json
import asyncio
import logging
import numpy as np
import torch
import yaml
from typing import Dict, Any, Optional, Tuple

# Async Kafka and Redis clients
try:
    from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
    import redis.asyncio as redis
except ImportError:
    raise ImportError("Requires aiokafka and redis packages.")

# Internal Model Imports
from models.portfolio.edt_agent import ElasticDecisionTransformer
from models.hedging.deep_hedging import DeepHedgingNetwork

logger = logging.getLogger("PortfolioAgentSvc")

# Dimensionality contract — must match hyperparams.yaml exactly.
_OBS_DIM: int = 52
_LATENT_DIM: int = 16
_ALPHA_DIM: int = 124   # 5 features * 25 assets (node_feat_dim reduced to per-asset summary)
_STATE_DIM: int = _OBS_DIM + _LATENT_DIM + _ALPHA_DIM  # = 192

# Risk guard: reject any weight vector whose gross leverage exceeds this threshold.
# Must match config/risk_limits.yaml::portfolio_limits::max_gross_leverage.
_MAX_GROSS_LEVERAGE: float = 1.50


class PortfolioAgentService:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.redis_client = redis.Redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379")
        )

        logger.info("Loading EDT and Deep Hedging models into VRAM...")
        self.edt = ElasticDecisionTransformer(config.get("edt", {}))
        self.edt.to(self.device)
        self.edt.eval()

        self.hedger = DeepHedgingNetwork(config.get("hedging", {}))
        self.hedger.to(self.device)
        self.hedger.eval()

        # Load the ETF universe list to map integer weight indices -> ticker strings.
        with open("config/universe.yaml", "r") as f:
            univ = yaml.safe_load(f)
            self.universe_tickers = [asset["ticker"] for asset in univ.get("assets", [])]

        # Cache the last valid allocation so we can hold it if the new one fails validation.
        self._last_valid_allocation: Optional[Dict[str, float]] = None

    async def setup_kafka(self):
        kafka_url = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

        self.consumer = AIOKafkaConsumer(
            "regime-posterior",  # Published by regime_encoder_svc
            bootstrap_servers=kafka_url,
            group_id="portfolio_agent_group",
            auto_offset_reset="latest",
        )
        self.producer = AIOKafkaProducer(bootstrap_servers=kafka_url)

        await self.consumer.start()
        await self.producer.start()
        logger.info("Kafka Portfolio Consumer connected to event bus.")

    async def run(self):
        await self.setup_kafka()

        try:
            async for msg in self.consumer:
                payload = json.loads(msg.value.decode("utf-8"))
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

    async def _assemble_state_vector(
        self, z_mu: np.ndarray
    ) -> Tuple[np.ndarray, bool]:
        """
        FIX #5: Assembles the real 192-dim EDT state vector from live Redis keys.

        State vector composition:
          [0:52]    obs_dim    — Raw 52-dim market observation snapshot.
                                Redis key: 'obs:current' (JSON list, published by market_data_svc)
          [52:68]   latent_dim — 16-dim Mamba-KAN regime posterior z_mu.
                                Passed directly from the Kafka regime-posterior message.
          [68:192]  alpha_dim  — 124-dim GATv2 alpha signal vector (5 features x 25 assets
                                or a flattened 124-dim projection from alpha_engine_svc).
                                Redis key: 'alpha:scores' (JSON list, published by alpha_engine_svc)

        Returns:
            (state_vector, is_complete): The assembled state and a flag indicating
            whether all upstream components were available. If False, the caller
            should log a warning — the EDT is operating on partially stale inputs.
        """
        is_complete = True

        # ── Component 1: Market Observation (obs_dim = 52) ────────────────────────
        obs_raw = await self.redis_client.get("obs:current")
        if obs_raw:
            obs_vec = np.array(json.loads(obs_raw), dtype=np.float32)
            if len(obs_vec) != _OBS_DIM:
                logger.warning(
                    f"obs:current has {len(obs_vec)} dims, expected {_OBS_DIM}. Zero-padding."
                )
                obs_vec = np.zeros(_OBS_DIM, dtype=np.float32)
                is_complete = False
        else:
            logger.warning("Redis key 'obs:current' not found. Using zeros for obs component.")
            obs_vec = np.zeros(_OBS_DIM, dtype=np.float32)
            is_complete = False

        # ── Component 2: Regime Posterior (latent_dim = 16) ──────────────────────
        # z_mu arrives directly in the Kafka message payload from regime_encoder_svc.
        if len(z_mu) != _LATENT_DIM:
            logger.warning(
                f"z_mu has {len(z_mu)} dims, expected {_LATENT_DIM}. Zero-padding."
            )
            z_mu = np.zeros(_LATENT_DIM, dtype=np.float32)
            is_complete = False
        latent_vec = z_mu[:_LATENT_DIM].astype(np.float32)

        # ── Component 3: GATv2 Alpha Scores (alpha_dim = 124) ────────────────────
        alpha_raw = await self.redis_client.get("alpha:scores")
        if alpha_raw:
            alpha_vec = np.array(json.loads(alpha_raw), dtype=np.float32)
            if len(alpha_vec) != _ALPHA_DIM:
                logger.warning(
                    f"alpha:scores has {len(alpha_vec)} dims, expected {_ALPHA_DIM}. "
                    "Zero-padding. Check that alpha_engine_svc is publishing."
                )
                alpha_vec = np.zeros(_ALPHA_DIM, dtype=np.float32)
                is_complete = False
        else:
            logger.warning(
                "Redis key 'alpha:scores' not found. "
                "EDT is blind to GATv2 signals. Check alpha_engine_svc."
            )
            alpha_vec = np.zeros(_ALPHA_DIM, dtype=np.float32)
            is_complete = False

        # ── Concatenate all components into the full state vector ─────────────────
        # Expected layout: [obs(52) | z_mu(16) | alpha(124)] = 192 dims
        full_state = np.concatenate([obs_vec, latent_vec, alpha_vec])

        assert full_state.shape == (_STATE_DIM,), (
            f"State assembly error: expected ({_STATE_DIM},), got {full_state.shape}. "
            "Check _OBS_DIM, _LATENT_DIM, _ALPHA_DIM constants against hyperparams.yaml."
        )

        return full_state, is_complete

    async def _process_allocation(self, regime_payload: Dict):
        """
        Calculates base weights via EDT, overlays Deep Hedging constraints,
        validates gross leverage, and publishes final target weights for execution.
        """
        z_mu = np.array(regime_payload.get("z_mu", []), dtype=np.float32)
        tda_alert = bool(regime_payload.get("tda_alert", 0))

        # ── FIX #5: Assemble the real 192-dim state from Redis ───────────────────
        full_state, state_is_complete = await self._assemble_state_vector(z_mu)

        if not state_is_complete:
            logger.warning(
                "State vector is incomplete (upstream service(s) not yet publishing). "
                "EDT is running on partially stale/zeroed components."
            )

        # ── 1. Base Portfolio Allocation (Elastic Decision Transformer) ──────────
        # The target return is regime-adaptive in production via EDT.compute_regime_target_return().
        # Using a static 10% annualised prompt here pending full z_t -> target_return mapping.
        target_return: float = 0.10
        mean_weights, std_weights = self.edt.get_weights(
            full_state, target_return, device=self.device
        )
        base_allocation = {
            ticker: float(weight)
            for ticker, weight in zip(self.universe_tickers, mean_weights)
        }

        # ── 2. Live Portfolio State for Deep Hedging ─────────────────────────────
        # Fetch [leverage, drawdown_pct, unrealized_pnl_pct] from Redis.
        leverage_raw = await self.redis_client.get("portfolio:leverage")
        drawdown_raw = await self.redis_client.get("portfolio:drawdown")
        pnl_raw = await self.redis_client.get("portfolio:unrealized_pnl")

        leverage = float(leverage_raw) if leverage_raw else 1.0
        drawdown = float(drawdown_raw) if drawdown_raw else 0.0
        unrealized_pnl = float(pnl_raw) if pnl_raw else 0.0

        port_state = np.array([leverage, drawdown, unrealized_pnl], dtype=np.float32)
        ltc_urgency = float(await self.redis_client.get("regime:ltc_urgency") or 0.0)

        # Fetch crash probability from World Model output (key published by world_model_svc).
        crash_prob_raw = await self.redis_client.get("world_model:crash_prob")
        crash_prob = float(crash_prob_raw) if crash_prob_raw else 0.0

        # ── 3. Deep Hedging Overlay ───────────────────────────────────────────────
        hedge_overlay = self.hedger.get_hedge_overlay(
            z_t=z_mu,
            portfolio_state=port_state,
            tda_alert=tda_alert,
            ltc_urgency=ltc_urgency,
            crash_probability=crash_prob,
        )

        # ── 4. Merge Base Allocation + Hedge Overlay ─────────────────────────────
        final_allocation = self._merge_allocations(base_allocation, hedge_overlay)

        # ── 5. FIX #14: Gross Leverage Guard ─────────────────────────────────────
        # Validate that sum(|w_i|) <= max_gross_leverage before submission.
        gross_leverage = sum(abs(w) for w in final_allocation.values())
        if gross_leverage > _MAX_GROSS_LEVERAGE:
            logger.error(
                f"Gross leverage {gross_leverage:.3f} exceeds limit {_MAX_GROSS_LEVERAGE}. "
                "Allocation REJECTED. Holding previous weights."
            )
            if self._last_valid_allocation is not None:
                final_allocation = self._last_valid_allocation
            else:
                # No prior valid allocation to fall back to — go to cash.
                logger.critical(
                    "No prior valid allocation available. Defaulting to 100% SHV (cash)."
                )
                final_allocation = {"SHV": 1.0}

        else:
            self._last_valid_allocation = final_allocation

        # ── 6. Broadcast to Execution Router ─────────────────────────────────────
        out_msg = {
            "timestamp": regime_payload.get("timestamp"),
            "weights": final_allocation,
        }
        await self.producer.send_and_wait(
            "target-weights", json.dumps(out_msg).encode("utf-8")
        )
        logger.info(
            f"Target weights broadcast. Gross leverage: {gross_leverage:.3f}. "
            f"Top holdings: {self._get_top_holdings(final_allocation)}"
        )

    def _merge_allocations(
        self, base: Dict[str, float], hedge: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Blends EDT base allocation with Deep Hedging overlay.
        Scales down base positions proportionately to make room for hedge positions,
        then re-normalises to ensure sum of weights == 1.0.
        """
        combined = base.copy()
        hedge_total = sum(hedge.values())

        if hedge_total > 0:
            scale_factor = 1.0 - min(hedge_total, 1.0)
            for k in combined:
                combined[k] *= scale_factor
            for k, v in hedge.items():
                combined[k] = combined.get(k, 0.0) + v

        # Normalise — prevents floating point drift from compounding across rebalances.
        total = sum(combined.values())
        if abs(total) > 1e-8:
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