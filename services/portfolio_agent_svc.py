"""
FORTRESS v5 - portfolio_agent_svc.py
Path: services/portfolio_agent_svc.py

Async Microservice: The Capital Allocator.
Listens to Kafka for regime updates, triggers the Elastic Decision Transformer
(EDT) and Deep Hedging network, and broadcasts validated target portfolio weights
to the execution router.

FIXES APPLIED:
  - BUG #5 (CRITICAL): The 192-dim EDT state vector was `np.random.randn(192)`.
    Random noise was submitted to the broker as a capital allocation signal.
    The state is now assembled from its three real components via Redis.
    [obs(52) | z_mu(16) | alpha(124)] — with graceful zero fallback per component.

  - BUG #14: No gross-leverage guard before broadcasting weights.
    A runaway EDT output could submit >150% gross leverage to execution.
    Added explicit validation: if sum(|weights|) > _MAX_GROSS_LEVERAGE,
    the allocation is rejected and previous valid weights are held.

  - BUG #PA-1 (NEW): EDT Return-To-Go (RTG) target was a static 0.10 (10%).
    The EDT is architecturally designed to be regime-conditional — receiving
    a static target completely defeats the purpose. The target is now computed
    by `edt.get_regime_return_target(z_t, volatility_targets)` which reads
    the Mamba-KAN latent vector and maps it to a regime-appropriate RTG.

  - BUG #PA-2 (NEW): Deep Hedger portfolio_state was `np.array([1.0, -0.02, 0.05])`
    (hardcoded). The hedger was making overlay decisions on fictional drawdown data.
    Portfolio state is now fetched from Redis keys:
      portfolio:leverage    → current gross leverage
      portfolio:drawdown    → current drawdown from ATH
      portfolio:unrealized_pnl → unrealized PnL fraction

  - BUG #PA-3 (NEW): Model weights were never loaded from disk.
    EDT and DeepHedging models ran with randomly initialised PyTorch weights.
    Added `_load_model_weights()` with existence checks and fallback logging.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import asyncio
import numpy as np
import torch
import yaml

logger = logging.getLogger("PortfolioAgentSvc")

# ── Dimensionality contract (must match hyperparams.yaml) ─────────────────────
_OBS_DIM:    int   = 52
_LATENT_DIM: int   = 16
_ALPHA_DIM:  int   = 124   # 5 per-asset features × 25 assets − 1 summary col
_STATE_DIM:  int   = _OBS_DIM + _LATENT_DIM + _ALPHA_DIM   # = 192

# Risk guard — must match config/risk_limits.yaml::portfolio_limits::max_gross_leverage
_MAX_GROSS_LEVERAGE: float = 1.50

# Model weight paths
_EDT_WEIGHTS:     str = "models/weights/edt_latest.pt"
_HEDGER_WEIGHTS:  str = "models/weights/hedger_latest.pt"

# Regime-conditional volatility targets for RTG mapping
# Maps regime label → annualised vol target → EDT return target
_REGIME_VOL_TARGETS: Dict[str, float] = {
    "bull_low_vol":    0.08,   # Low-vol bull → conservative RTG
    "bull_high_vol":   0.14,
    "bear_low_vol":    0.05,   # Defensive regime
    "bear_high_vol":   0.03,
    "crisis":          0.02,   # Capital preservation
    "recovery":        0.12,
    "flat_deflation":  0.06,
    "stagflation":     0.04,
    "rate_shock":      0.05,
    "credit_stress":   0.03,
    "momentum_bull":   0.15,
    "momentum_bear":   0.04,
    "liquidity_crunch": 0.02,
    "risk_on_EM":      0.13,
    "risk_off_DM":     0.04,
    "unknown":         0.08,   # Conservative default
}


class PortfolioAgentService:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ── Redis client ─────────────────────────────────────────────────────
        try:
            import redis.asyncio as redis
            self._redis = redis.Redis.from_url(
                os.getenv("REDIS_URL", "redis://localhost:6379")
            )
        except ImportError as exc:
            raise ImportError("redis[asyncio] required") from exc

        # ── Load models (BUG #PA-3 FIX) ─────────────────────────────────────
        from models.portfolio.edt_agent import ElasticDecisionTransformer
        from models.hedging.deep_hedging import DeepHedgingNetwork

        logger.info("Loading EDT and DeepHedging models...")
        self.edt    = ElasticDecisionTransformer(config.get("edt", {}))
        self.hedger = DeepHedgingNetwork(config.get("hedging", {}))

        self._load_model_weights()

        self.edt.to(self.device).eval()
        self.hedger.to(self.device).eval()

        # ── Asset universe ────────────────────────────────────────────────────
        with open("config/universe.yaml", "r") as f:
            univ = yaml.safe_load(f)
            self.universe_tickers: List[str] = [
                asset["ticker"] for asset in univ.get("assets", [])
            ]

        # ── State: last valid allocation (for fallback on validation failure) ─
        self._last_valid_allocation: Optional[Dict[str, float]] = None
        self._last_allocation_ts: float = 0.0

        # ── Kafka handles ─────────────────────────────────────────────────────
        self.consumer = None
        self.producer = None

    def _load_model_weights(self) -> None:
        """BUG #PA-3 FIX: Load trained weights from disk with graceful fallback."""
        for path, model, name in [
            (_EDT_WEIGHTS,    self.edt,    "EDT"),
            (_HEDGER_WEIGHTS, self.hedger, "DeepHedger"),
        ]:
            if os.path.isfile(path):
                try:
                    state_dict = torch.load(path, map_location=self.device)
                    model.load_state_dict(state_dict)
                    logger.info(f"✅ {name} weights loaded from '{path}'.")
                except Exception as exc:
                    logger.error(
                        f"❌ {name} weight load failed from '{path}': {exc}. "
                        "Using random weights."
                    )
            else:
                logger.warning(
                    f"⚠️  {name} weight file '{path}' not found. "
                    "Running with RANDOMLY INITIALISED WEIGHTS."
                )

    # ── Kafka setup ───────────────────────────────────────────────────────────

    async def setup_kafka(self) -> None:
        try:
            from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
        except ImportError as exc:
            raise ImportError("aiokafka required") from exc

        kafka_url = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

        self.consumer = AIOKafkaConsumer(
            "regime-posterior",
            bootstrap_servers=kafka_url,
            group_id="portfolio_agent_group",
            auto_offset_reset="latest",
        )
        self.producer = AIOKafkaProducer(bootstrap_servers=kafka_url)

        await self.consumer.start()
        await self.producer.start()
        logger.info("PortfolioAgentSvc: Kafka consumer/producer connected.")

    # ── Main loop ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        await self.setup_kafka()
        try:
            async for msg in self.consumer:
                payload = json.loads(msg.value.decode("utf-8"))
                await self._process_allocation(payload)
        except asyncio.CancelledError:
            logger.info("PortfolioAgentSvc: shutting down.")
        except Exception as exc:
            logger.critical(f"Fatal error in portfolio agent: {exc}", exc_info=True)
            raise
        finally:
            if self.consumer:
                await self.consumer.stop()
            if self.producer:
                await self.producer.stop()
            await self._redis.aclose()

    # ── Core allocation logic ─────────────────────────────────────────────────

    async def _process_allocation(self, regime_payload: Dict[str, Any]) -> None:
        """
        Full allocation pipeline:
          1. Assemble real 192-dim state vector from Redis (BUG #5 FIX)
          2. Compute regime-conditional RTG target (BUG #PA-1 FIX)
          3. EDT forward pass → base allocation
          4. Fetch real portfolio state from Redis (BUG #PA-2 FIX)
          5. Deep Hedging overlay
          6. Gross leverage validation (BUG #14 FIX)
          7. Publish to Kafka + Redis
        """
        z_mu        = np.array(regime_payload.get("z_mu",         []), dtype=np.float32)
        tda_alert   = bool(regime_payload.get("tda_alert",         0))
        ltc_urgency = float(regime_payload.get("ltc_urgency",      0.0))
        regime_label = str(regime_payload.get("regime_label",      "unknown"))

        # ── 1. Assemble real state vector ────────────────────────────────────
        full_state, is_complete = await self._assemble_state_vector(z_mu)
        if not is_complete:
            logger.warning(
                "State vector incomplete (upstream services still starting). "
                "EDT operating on partial state."
            )

        # ── 2. Regime-conditional RTG target (BUG #PA-1 FIX) ────────────────
        vol_target    = _REGIME_VOL_TARGETS.get(regime_label, 0.08)
        target_return = self.edt.get_regime_return_target(
            z_t=z_mu,
            volatility_targets={regime_label: vol_target},
        )
        logger.info(
            f"Regime='{regime_label}' | VolTarget={vol_target:.2%} | "
            f"RTG={target_return:.2%}"
        )

        # ── 3. EDT base allocation ───────────────────────────────────────────
        mean_weights, std_weights = self.edt.get_weights(
            state=full_state,
            target_return=target_return,
            device=self.device,
        )

        base_allocation = {
            ticker: float(w)
            for ticker, w in zip(self.universe_tickers, mean_weights)
        }

        # ── 4. Real portfolio state for Deep Hedging (BUG #PA-2 FIX) ────────
        port_state = await self._fetch_portfolio_state()

        # Fetch crash probability from SDE World Model (via Redis)
        crash_prob_raw = await self._redis.get("sde:crash_probability")
        crash_prob = float(crash_prob_raw) if crash_prob_raw else 0.15

        # ── 5. Deep Hedging overlay ──────────────────────────────────────────
        hedge_overlay = self.hedger.get_hedge_overlay(
            z_t=z_mu,
            portfolio_state=port_state,
            tda_alert=tda_alert,
            ltc_urgency=ltc_urgency,
            crash_probability=crash_prob,
        )

        # ── 6. Merge and validate gross leverage ─────────────────────────────
        final_allocation = self._merge_allocations(base_allocation, hedge_overlay)

        gross_leverage = sum(abs(w) for w in final_allocation.values())
        if gross_leverage > _MAX_GROSS_LEVERAGE:
            logger.warning(
                f"❌ Gross leverage {gross_leverage:.3f} > {_MAX_GROSS_LEVERAGE}. "
                f"Allocation REJECTED. Holding previous weights."
            )
            if self._last_valid_allocation:
                await self._publish_allocation(
                    self._last_valid_allocation, hedge_overlay, regime_label
                )
            return

        # Validate no single position exceeds 30% (hard position limit)
        for ticker, w in final_allocation.items():
            if abs(w) > 0.30:
                logger.warning(
                    f"Position limit: {ticker}={w:.2%} > 30%. Clipping."
                )
                final_allocation[ticker] = 0.30 * np.sign(w)

        # Renormalise after clipping
        total = sum(abs(w) for w in final_allocation.values())
        if total > 0:
            scale = min(1.0, _MAX_GROSS_LEVERAGE / total)
            final_allocation = {k: v * scale for k, v in final_allocation.items()}

        self._last_valid_allocation = final_allocation
        self._last_allocation_ts    = time.time()

        # Cache EDT uncertainty for monitoring dashboard
        avg_uncertainty = float(std_weights.mean())
        await self._redis.set("portfolio:edt_uncertainty", avg_uncertainty, ex=3600)

        # ── 7. Publish ───────────────────────────────────────────────────────
        await self._publish_allocation(final_allocation, hedge_overlay, regime_label)

    # ── State assembly (BUG #5 FIX) ──────────────────────────────────────────

    async def _assemble_state_vector(
        self, z_mu: np.ndarray
    ) -> Tuple[np.ndarray, bool]:
        """
        Assembles the real 192-dim EDT state from three Redis components:
          [obs(52) | z_mu(16) | alpha(124)]

        Returns (state_vector, is_complete).
        is_complete=False signals that one or more upstream components are
        unavailable (startup lag). The EDT will operate but with zeroed component(s).
        """
        is_complete = True

        # Component 1: Market observation (obs_dim=52)
        obs_raw = await self._redis.get("obs:current")
        if obs_raw:
            obs_vec = np.array(json.loads(obs_raw), dtype=np.float32)
            if len(obs_vec) != _OBS_DIM:
                obs_vec = np.zeros(_OBS_DIM, dtype=np.float32)
                is_complete = False
        else:
            logger.warning("'obs:current' not in Redis — obs component zeroed.")
            obs_vec = np.zeros(_OBS_DIM, dtype=np.float32)
            is_complete = False

        # Component 2: Latent regime z_mu (latent_dim=16) — from Kafka payload
        if len(z_mu) != _LATENT_DIM:
            logger.warning(f"z_mu dim={len(z_mu)}, expected {_LATENT_DIM}. Zeroing.")
            z_mu_vec = np.zeros(_LATENT_DIM, dtype=np.float32)
            is_complete = False
        else:
            z_mu_vec = z_mu[:_LATENT_DIM].astype(np.float32)

        # Component 3: GATv2 alpha scores (alpha_dim=124)
        alpha_raw = await self._redis.get("alpha:scores")
        if alpha_raw:
            alpha_vec = np.array(json.loads(alpha_raw), dtype=np.float32)
            if len(alpha_vec) != _ALPHA_DIM:
                logger.warning(
                    f"'alpha:scores' has {len(alpha_vec)} dims, expected {_ALPHA_DIM}. "
                    "Zeroing. Is alpha_engine_svc running?"
                )
                alpha_vec = np.zeros(_ALPHA_DIM, dtype=np.float32)
                is_complete = False
        else:
            logger.warning("'alpha:scores' not in Redis — alpha component zeroed.")
            alpha_vec = np.zeros(_ALPHA_DIM, dtype=np.float32)
            is_complete = False

        full_state = np.concatenate([obs_vec, z_mu_vec, alpha_vec])
        assert full_state.shape == (_STATE_DIM,), (
            f"State shape mismatch: {full_state.shape} != ({_STATE_DIM},)"
        )
        return full_state, is_complete

    # ── Portfolio state fetch (BUG #PA-2 FIX) ────────────────────────────────

    async def _fetch_portfolio_state(self) -> np.ndarray:
        """
        Fetches real-time portfolio metrics from Redis for the Deep Hedger.
        Keys published by execution_svc and order_manager.

        Returns:
            port_state: (3,) float32 [leverage, drawdown_pct, unrealized_pnl_pct]
        """
        try:
            leverage_raw = await self._redis.get("portfolio:leverage")
            drawdown_raw = await self._redis.get("portfolio:drawdown")
            pnl_raw      = await self._redis.get("portfolio:unrealized_pnl")

            leverage      = float(leverage_raw)  if leverage_raw else 1.0
            drawdown      = float(drawdown_raw)  if drawdown_raw else 0.0
            unrealized_pnl = float(pnl_raw)      if pnl_raw      else 0.0

            return np.array([leverage, drawdown, unrealized_pnl], dtype=np.float32)

        except Exception as exc:
            logger.debug(f"Portfolio state fetch failed ({exc}) — using neutral defaults.")
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)

    # ── Allocation helpers ────────────────────────────────────────────────────

    def _merge_allocations(
        self,
        base: Dict[str, float],
        hedge: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Overlays hedge positions on top of base allocations.

        Strategy:
          - Hedge tickers (VIXY, GLD, TLT) receive their hedge weight directly.
          - Equity tickers are proportionally scaled down to accommodate the hedge.
        """
        hedge_notional = sum(abs(w) for w in hedge.values())
        scale_factor   = max(0.0, 1.0 - hedge_notional)

        merged: Dict[str, float] = {}
        for ticker, w in base.items():
            if ticker in hedge:
                merged[ticker] = hedge[ticker]
            else:
                merged[ticker] = w * scale_factor

        # Add any hedge tickers not in base
        for ticker, w in hedge.items():
            if ticker not in merged:
                merged[ticker] = w

        return merged

    async def _publish_allocation(
        self,
        allocation: Dict[str, float],
        hedge_overlay: Dict[str, float],
        regime_label: str,
    ) -> None:
        """Publishes the final allocation to Kafka and Redis."""
        gross_leverage = sum(abs(w) for w in allocation.values())

        # Determine execution urgency hint for the MARL router
        ltc_urgency_raw = await self._redis.get("regime:ltc_urgency")
        ltc_urgency     = float(ltc_urgency_raw) if ltc_urgency_raw else 0.0

        if ltc_urgency > 0.85:
            agent_hint = "urgent_ddpg"
        elif ltc_urgency < 0.30:
            agent_hint = "opportunistic_sac"
        else:
            agent_hint = "stealth_ppo"

        payload = {
            "timestamp":     time.time(),
            "weights":       allocation,
            "gross_leverage": round(gross_leverage, 4),
            "hedge_overlay": hedge_overlay,
            "agent_hint":    agent_hint,
            "regime_label":  regime_label,
        }

        # Publish to Kafka for execution_svc
        await self.producer.send_and_wait(
            "target-weights",
            json.dumps(payload).encode("utf-8"),
        )

        # Cache in Redis for monitoring dashboard
        await self._redis.set(
            "portfolio:target_weights",
            json.dumps(payload),
            ex=3600,
        )

        logger.info(
            f"Allocation published: leverage={gross_leverage:.3f}, "
            f"agent_hint='{agent_hint}', "
            f"top3={sorted(allocation.items(), key=lambda x: -abs(x[1]))[:3]}"
        )


if __name__ == "__main__":
    import yaml
    with open("config/hyperparams.yaml", "r") as f:
        config = yaml.safe_load(f)
    svc = PortfolioAgentService(config)
    asyncio.run(svc.run())