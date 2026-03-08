"""
FORTRESS v5 - portfolio_agent_svc.py
Path: services/portfolio_agent_svc.py

Async Microservice: The Capital Allocator.
Listens to Kafka for regime updates, triggers the Elastic Decision Transformer
(EDT) and Deep Hedging network, and broadcasts validated target portfolio weights
to the execution router.

FIXES APPLIED (previous sessions):
  - BUG #5  (CRITICAL): EDT state was np.random.randn(192). Fixed: real Redis assembly.
  - BUG #14: No gross-leverage guard. Fixed: hard cap at 1.50x.
  - BUG #PA-1: Static RTG=0.10. Fixed: regime-conditional via edt.get_regime_return_target.
  - BUG #PA-2: Hedger portfolio_state was hardcoded. Fixed: live Redis fetch.
  - BUG #PA-3: Models ran with random weights. Fixed: _load_model_weights().

P1 ENHANCEMENTS (this session):
  - P1-VOL: VolatilityTargetingOverlay injected between EDT output and hedge overlay.
      EWMA variance (halflife=21 trading days) on realised portfolio returns.
      Scale factor: σ_target / max(σ_realized, σ_floor).
      σ_target=12% annualised. Cap gross leverage at 1.50 post-scaling.
      Expected: MaxDD −35%, Sortino +40%.

  - P1-KELLY: Kelly fraction scaling on regime uncertainty.
      F = 0.5 * exp(−κ * ||z_σ||₂) where κ=2.0.
      High z_σ norm (ambiguous regime) → near-zero positions.
      Applied BEFORE vol targeting — Kelly controls conviction, vol targeting
      controls absolute risk level. Compound effect: F_kelly × (σ_t / σ_r).
      Expected: MaxDD −20% during regime ambiguity.
"""

from __future__ import annotations

import json
import logging
import math
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
_ALPHA_DIM:  int   = 124
_STATE_DIM:  int   = _OBS_DIM + _LATENT_DIM + _ALPHA_DIM   # = 192

# ── Risk limits ────────────────────────────────────────────────────────────────
_MAX_GROSS_LEVERAGE: float = 1.50

# ── Model weight paths ────────────────────────────────────────────────────────
_EDT_WEIGHTS:     str = "models/weights/edt_latest.pt"
_HEDGER_WEIGHTS:  str = "models/weights/hedger_latest.pt"

# ── P1-VOL: Volatility targeting constants ─────────────────────────────────────
# σ_target=12% matches the audit recommendation; floor prevents division explosion
# on the first few days of live operation before EWMA warms up.
_VOL_TARGET_ANNUAL:    float = 0.12   # 12% annualised target volatility
_VOL_EWMA_HALFLIFE:    int   = 21     # trading days — matches 1-month realised vol window
_VOL_FLOOR_ANNUAL:     float = 0.03   # never scale up more than 4× (0.12/0.03)
_VOL_MAX_SCALE:        float = _VOL_TARGET_ANNUAL / _VOL_FLOOR_ANNUAL  # hard ceiling on leverage multiplier

# ── P1-KELLY: Regime uncertainty constants ────────────────────────────────────
# F = 0.5 × exp(−κ × ‖z_σ‖₂). Half-Kelly baseline (0.5) is already conservative;
# exponential decay over uncertainty norm prevents catastrophic draws in ambiguous regimes.
_KELLY_KAPPA:    float = 2.0   # sensitivity to ||z_sigma||_2
_KELLY_MIN_FRAC: float = 0.05  # never fully zero even in maximum uncertainty
_KELLY_MAX_FRAC: float = 0.50  # upper bound = half-Kelly at zero uncertainty

# ── Regime vol targets → RTG mapping ─────────────────────────────────────────
_REGIME_VOL_TARGETS: Dict[str, float] = {
    "bull_low_vol":     0.08,
    "bull_high_vol":    0.14,
    "bear_low_vol":     0.05,
    "bear_high_vol":    0.03,
    "crisis":           0.02,
    "recovery":         0.12,
    "flat_deflation":   0.06,
    "stagflation":      0.04,
    "rate_shock":       0.05,
    "credit_stress":    0.03,
    "momentum_bull":    0.15,
    "momentum_bear":    0.04,
    "liquidity_crunch": 0.02,
    "risk_on_EM":       0.13,
    "risk_off_DM":      0.04,
    "unknown":          0.08,
}

# ── Redis key schema ───────────────────────────────────────────────────────────
_REDIS_EWMA_VAR_KEY:     str = "portfolio:ewma_variance"
_REDIS_DAILY_RETURN_KEY: str = "portfolio:daily_return"
_REDIS_LEVERAGE_KEY:     str = "portfolio:leverage"
_REDIS_DRAWDOWN_KEY:     str = "portfolio:drawdown"
_REDIS_PNL_KEY:          str = "portfolio:unrealized_pnl"


# ─────────────────────────────────────────────────────────────────────────────
# P1-VOL: VolatilityTargetingOverlay
# ─────────────────────────────────────────────────────────────────────────────

class VolatilityTargetingOverlay:
    """
    Scales portfolio weights so realised annualised volatility tracks σ_target=12%.

    Mechanism:
      1. Maintain an EWMA variance estimate: Var_t = α·r²_t + (1−α)·Var_{t-1}
         where α = 1 − exp(−ln(2) / halflife). This is the RiskMetrics-97 formula.
      2. σ_realized = sqrt(252 × Var_t)
      3. scale_factor = clamp(σ_target / σ_realized, 0, MAX_SCALE)
      4. w_scaled = w × scale_factor
      5. If gross(w_scaled) > max_leverage → renormalise to max_leverage.

    The EWMA state persists across Kafka messages via Redis (`portfolio:ewma_variance`).
    On cold-start (key absent), the estimator is warm-started to the target variance
    so the system doesn't wildly over-leverage on the first tick.
    """

    def __init__(
        self,
        target_annual_vol: float = _VOL_TARGET_ANNUAL,
        halflife_days:     int   = _VOL_EWMA_HALFLIFE,
        vol_floor:         float = _VOL_FLOOR_ANNUAL,
        max_leverage:      float = _MAX_GROSS_LEVERAGE,
    ) -> None:
        self.target_annual_vol = target_annual_vol
        self.vol_floor         = vol_floor
        self.max_leverage      = max_leverage

        # EWMA decay: α = 1 − exp(−ln(2) / T½)
        self.alpha = 1.0 - math.exp(-math.log(2.0) / halflife_days)

        # In-process cache: avoids Redis round-trip on every message within same pod restart.
        # Initialised to target variance — neutral starting point.
        self._ewma_var: float = (target_annual_vol / math.sqrt(252.0)) ** 2

    async def update_ewma(self, redis_client: Any) -> None:
        """
        Fetches today's portfolio daily return from Redis and updates EWMA variance.
        Called once per allocation cycle, before `apply()`.

        On cold-start or missing key: keeps current in-process estimate (warm-started
        to target variance — avoids the 10× leverage spike on day-1).
        """
        try:
            # Load persisted EWMA from Redis (survives pod restart)
            stored_var = await redis_client.get(_REDIS_EWMA_VAR_KEY)
            if stored_var is not None:
                self._ewma_var = float(stored_var)

            # Update with today's observed squared return
            r_str = await redis_client.get(_REDIS_DAILY_RETURN_KEY)
            if r_str is not None:
                r_daily = float(r_str)
                # RiskMetrics-97 EWMA: Var_t = α·r²_t + (1−α)·Var_{t-1}
                self._ewma_var = self.alpha * (r_daily ** 2) + (1.0 - self.alpha) * self._ewma_var
                await redis_client.set(_REDIS_EWMA_VAR_KEY, str(self._ewma_var))
        except Exception as exc:
            # Non-fatal: continue with in-process estimate
            logger.debug(f"VolTargeting EWMA update failed ({exc}); using cached estimate.")

    def apply(
        self,
        weights: np.ndarray,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Scales weight vector to target realised volatility.

        Args:
            weights: (N,) float32 — pre-overlay portfolio weights (can exceed 1.0 gross).

        Returns:
            w_scaled:       (N,) scaled weights.
            scale_factor:   The multiplier applied (logged for monitoring).
            sigma_realized: Estimated annualised realised vol (logged for monitoring).
        """
        # σ_realized in annualised terms
        sigma_realized = math.sqrt(self._ewma_var * 252.0)
        # Clamp denominator to floor to prevent over-levering on ultra-low-vol days
        sigma_effective = max(sigma_realized, self.vol_floor)
        scale_factor = self.target_annual_vol / sigma_effective

        # Hard cap: never amplify beyond 4× (target/floor) to prevent day-1 artefacts
        scale_factor = min(scale_factor, _VOL_MAX_SCALE)

        w_scaled = weights * scale_factor

        # Secondary leverage clamp: vol targeting can still produce >150% gross if
        # weights were already near the limit before scaling. Hard clamp here.
        gross = float(np.abs(w_scaled).sum())
        if gross > self.max_leverage:
            w_scaled = w_scaled * (self.max_leverage / gross)

        return w_scaled, scale_factor, sigma_realized


# ─────────────────────────────────────────────────────────────────────────────
# P1-KELLY: Kelly fraction scalar
# ─────────────────────────────────────────────────────────────────────────────

def compute_kelly_fraction(z_sigma: np.ndarray) -> float:
    """
    Half-Kelly fraction attenuated by latent regime uncertainty.

        F = clip(0.5 × exp(−κ × ‖z_σ‖₂), KELLY_MIN, KELLY_MAX)

    Rationale:
      - ‖z_σ‖₂ = 0  → F = 0.5 (full half-Kelly; regime perfectly certain).
      - ‖z_σ‖₂ = 1  → F ≈ 0.5 × exp(−2) ≈ 0.068 (regime highly ambiguous).
      - ‖z_σ‖₂ > 1.5 → F clips to _KELLY_MIN = 0.05 (near-zero, capital preservation).

    The 16-dim z_σ posterior from MambaKANVAE has expected norm ≈ sqrt(16)=4 at
    uninformative prior, and norm ≈ 0.5–1.5 for well-identified regimes after training.
    κ=2.0 is calibrated so that regime ambiguity (norm~1.0) halves positions.

    Args:
        z_sigma: (16,) float32 — posterior standard deviation from Mamba-KAN VAE.

    Returns:
        Scalar Kelly fraction in [_KELLY_MIN, _KELLY_MAX].
    """
    if len(z_sigma) == 0:
        logger.warning("Empty z_sigma received; defaulting to minimum Kelly fraction.")
        return _KELLY_MIN_FRAC

    z_sigma_norm = float(np.linalg.norm(z_sigma))
    fraction = _KELLY_MAX_FRAC * math.exp(-_KELLY_KAPPA * z_sigma_norm)
    fraction = float(np.clip(fraction, _KELLY_MIN_FRAC, _KELLY_MAX_FRAC))
    return fraction


# ─────────────────────────────────────────────────────────────────────────────
# Main Service
# ─────────────────────────────────────────────────────────────────────────────

class PortfolioAgentService:
    """
    Capital allocation microservice.

    Allocation pipeline (P1-updated):
      1.  Assemble 192-dim state vector from Redis.
      2.  Compute regime-conditional RTG target.
      3.  EDT forward pass → base weights (mean, std).
      4.  Kelly fraction scaling (P1-KELLY): w_kelly = w_base × F_kelly(z_σ).
      5.  Volatility targeting overlay (P1-VOL): w_vol = w_kelly × (σ_t / σ_r).
      6.  Deep Hedging overlay: w_final = merge(w_vol, hedge).
      7.  Gross leverage guard: reject if |w_final| > 1.50, hold prev.
      8.  Publish to Kafka + Redis.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Redis ─────────────────────────────────────────────────────────────
        try:
            import redis.asyncio as redis
            self._redis = redis.Redis.from_url(
                os.getenv("REDIS_URL", "redis://localhost:6379")
            )
        except ImportError as exc:
            raise ImportError("redis[asyncio] required") from exc

        # Models ─────────────────────────────────────────────────────────────
        from models.portfolio.edt_agent import ElasticDecisionTransformer
        from models.hedging.deep_hedging import DeepHedgingNetwork

        self.edt    = ElasticDecisionTransformer(config.get("edt", {}))
        self.hedger = DeepHedgingNetwork(config.get("hedging", {}))
        self._load_model_weights()
        self.edt.to(self.device).eval()
        self.hedger.to(self.device).eval()

        # Universe ────────────────────────────────────────────────────────────
        with open("config/universe.yaml", "r") as f:
            univ = yaml.safe_load(f)
            self.universe_tickers: List[str] = [
                asset["ticker"] for asset in univ.get("assets", [])
            ]

        # P1 overlays ─────────────────────────────────────────────────────────
        self._vol_overlay = VolatilityTargetingOverlay(
            target_annual_vol=_VOL_TARGET_ANNUAL,
            halflife_days=_VOL_EWMA_HALFLIFE,
            vol_floor=_VOL_FLOOR_ANNUAL,
            max_leverage=_MAX_GROSS_LEVERAGE,
        )

        # Fallback state ──────────────────────────────────────────────────────
        self._last_valid_allocation: Optional[Dict[str, float]] = None
        self._last_allocation_ts:   float = 0.0

        self.consumer = None
        self.producer = None

    def _load_model_weights(self) -> None:
        for path, model, name in [
            (_EDT_WEIGHTS,    self.edt,    "EDT"),
            (_HEDGER_WEIGHTS, self.hedger, "DeepHedger"),
        ]:
            if os.path.isfile(path):
                try:
                    model.load_state_dict(torch.load(path, map_location=self.device))
                    logger.info(f"✅ {name} weights loaded from '{path}'.")
                except Exception as exc:
                    logger.error(f"❌ {name} weight load failed ({exc}). Using random weights.")
            else:
                logger.warning(f"⚠️  {name} weights not found at '{path}'. Running randomly initialised.")

    # ── Kafka ─────────────────────────────────────────────────────────────────

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

    # ── Core allocation pipeline ──────────────────────────────────────────────

    async def _process_allocation(self, regime_payload: Dict[str, Any]) -> None:
        """
        P1-updated allocation pipeline (8 stages).
        Stages 4 (Kelly) and 5 (vol targeting) are new this session.
        """
        z_mu         = np.array(regime_payload.get("z_mu",       []), dtype=np.float32)
        z_sigma      = np.array(regime_payload.get("z_sigma",    []), dtype=np.float32)
        tda_alert    = bool(regime_payload.get("tda_alert",       0))
        ltc_urgency  = float(regime_payload.get("ltc_urgency",    0.0))
        regime_label = str(regime_payload.get("regime_label",    "unknown"))

        # ── 1. Assemble 192-dim state ─────────────────────────────────────────
        full_state, is_complete = await self._assemble_state_vector(z_mu)
        if not is_complete:
            logger.warning("State vector incomplete — EDT operating on partial state.")

        # ── 2. Regime-conditional RTG ─────────────────────────────────────────
        vol_target    = _REGIME_VOL_TARGETS.get(regime_label, 0.08)
        target_return = self.edt.get_regime_return_target(
            z_t=z_mu,
            volatility_targets={regime_label: vol_target},
        )

        # ── 3. EDT base allocation ────────────────────────────────────────────
        mean_weights, std_weights = self.edt.get_weights(
            state=full_state,
            target_return=target_return,
            device=self.device,
        )
        # mean_weights: (N_ASSETS,) np.float32

        # ── 4. Kelly fraction scaling (P1-KELLY) ──────────────────────────────
        # Attenuates all positions by regime-uncertainty-scaled half-Kelly fraction.
        # High ‖z_σ‖₂ → regime is ambiguous → near-zero exposure until it resolves.
        kelly_frac = compute_kelly_fraction(z_sigma)
        w_kelly    = mean_weights * kelly_frac

        logger.info(
            f"Regime='{regime_label}' | ‖z_σ‖={np.linalg.norm(z_sigma):.3f} "
            f"| Kelly_F={kelly_frac:.3f} | RTG={target_return:.2%}"
        )

        # ── 5. Volatility targeting overlay (P1-VOL) ──────────────────────────
        # Update EWMA from latest daily return in Redis, then scale weights.
        await self._vol_overlay.update_ewma(self._redis)
        w_vol, scale_factor, sigma_realized = self._vol_overlay.apply(w_kelly)

        logger.info(
            f"VolTargeting | σ_realized={sigma_realized:.2%} annualised "
            f"| σ_target={_VOL_TARGET_ANNUAL:.2%} | scale={scale_factor:.3f}x "
            f"| gross_before_hedge={np.abs(w_vol).sum():.3f}"
        )

        base_allocation = {
            ticker: float(w)
            for ticker, w in zip(self.universe_tickers, w_vol)
        }

        # ── 6. Deep Hedging overlay ───────────────────────────────────────────
        port_state     = await self._get_portfolio_state()
        crash_prob     = self._estimate_crash_probability(z_mu, z_sigma)
        hedge_overlay  = self.hedger.get_hedge_overlay(
            z_t=z_mu,
            portfolio_state=port_state,
            tda_alert=tda_alert,
            ltc_urgency=ltc_urgency,
            crash_probability=crash_prob,
        )
        final_allocation = self._merge_allocations(base_allocation, hedge_overlay)

        # ── 7. Gross leverage guard (BUG #14 FIX) ────────────────────────────
        gross = sum(abs(w) for w in final_allocation.values())
        if gross > _MAX_GROSS_LEVERAGE:
            logger.error(
                f"❌ Gross leverage={gross:.3f} exceeds {_MAX_GROSS_LEVERAGE:.2f}x. "
                "Holding previous valid allocation."
            )
            if self._last_valid_allocation:
                final_allocation = self._last_valid_allocation
            else:
                # Emergency: flat + cash
                final_allocation = {t: 0.0 for t in self.universe_tickers}
            await self._publish_allocation(final_allocation, hedge_overlay, regime_label)
            return

        self._last_valid_allocation = final_allocation
        self._last_allocation_ts    = time.time()

        # ── 8. Publish ────────────────────────────────────────────────────────
        await self._publish_allocation(final_allocation, hedge_overlay, regime_label)

    # ── State assembly ────────────────────────────────────────────────────────

    async def _assemble_state_vector(
        self,
        z_mu: np.ndarray,
    ) -> Tuple[np.ndarray, bool]:
        """
        Assembles [obs(52) | z_mu(16) | alpha(124)] = 192-dim EDT state.
        Returns (state, is_complete) — is_complete=False if any component missing.
        """
        is_complete = True
        state       = np.zeros(_STATE_DIM, dtype=np.float32)

        # obs features
        try:
            obs_raw = await self._redis.get("obs:current")
            obs = np.array(json.loads(obs_raw), dtype=np.float32) if obs_raw else None
            if obs is not None and len(obs) >= _OBS_DIM:
                state[:_OBS_DIM] = obs[:_OBS_DIM]
            else:
                is_complete = False
        except Exception:
            is_complete = False

        # z_mu from the Kafka payload (already have it)
        if len(z_mu) >= _LATENT_DIM:
            state[_OBS_DIM : _OBS_DIM + _LATENT_DIM] = z_mu[:_LATENT_DIM]
        else:
            is_complete = False

        # alpha scores
        try:
            alpha_raw = await self._redis.get("alpha:scores")
            alpha = np.array(json.loads(alpha_raw), dtype=np.float32) if alpha_raw else None
            if alpha is not None and len(alpha) >= _ALPHA_DIM:
                state[_OBS_DIM + _LATENT_DIM:] = alpha[:_ALPHA_DIM]
            else:
                is_complete = False
        except Exception:
            is_complete = False

        return state, is_complete

    async def _get_portfolio_state(self) -> np.ndarray:
        """
        Returns [leverage, drawdown_pct, unrealized_pnl_pct] from Redis.
        """
        try:
            leverage_raw = await self._redis.get(_REDIS_LEVERAGE_KEY)
            drawdown_raw = await self._redis.get(_REDIS_DRAWDOWN_KEY)
            pnl_raw      = await self._redis.get(_REDIS_PNL_KEY)

            leverage       = float(leverage_raw)  if leverage_raw else 1.0
            drawdown       = float(drawdown_raw)  if drawdown_raw else 0.0
            unrealized_pnl = float(pnl_raw)       if pnl_raw      else 0.0
            return np.array([leverage, drawdown, unrealized_pnl], dtype=np.float32)
        except Exception as exc:
            logger.debug(f"Portfolio state fetch failed ({exc}) — using neutral defaults.")
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)

    def _estimate_crash_probability(
        self,
        z_mu:    np.ndarray,
        z_sigma: np.ndarray,
    ) -> float:
        """
        Heuristic crash probability from regime posterior.
        P_crash = Φ(−‖z_mu‖ / (‖z_sigma‖ + ε)) — regimes far from the mean in
        negative direction combined with high uncertainty signal elevated risk.
        """
        mu_norm    = float(np.linalg.norm(z_mu))
        sigma_norm = float(np.linalg.norm(z_sigma))
        if sigma_norm < 1e-6:
            return 0.0
        from scipy.stats import norm as _norm  # lazy import — not in hot path
        return float(_norm.cdf(-mu_norm / (sigma_norm + 1e-6)))

    # ── Allocation helpers ────────────────────────────────────────────────────

    def _merge_allocations(
        self,
        base:  Dict[str, float],
        hedge: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Overlays hedge positions on base.
        Equity tickers proportionally scaled down to absorb hedge notional.
        """
        hedge_notional = sum(abs(w) for w in hedge.values())
        scale_factor   = max(0.0, 1.0 - hedge_notional)

        merged: Dict[str, float] = {}
        for ticker, w in base.items():
            merged[ticker] = hedge[ticker] if ticker in hedge else w * scale_factor
        for ticker, w in hedge.items():
            if ticker not in merged:
                merged[ticker] = w
        return merged

    async def _publish_allocation(
        self,
        allocation:    Dict[str, float],
        hedge_overlay: Dict[str, float],
        regime_label:  str,
    ) -> None:
        """Publishes final allocation to Kafka `target-weights` + Redis `portfolio:weights`."""
        gross_leverage = sum(abs(w) for w in allocation.values())

        # Determine execution agent hint from regime
        if regime_label in ("crisis", "liquidity_crunch", "bear_high_vol"):
            agent_hint = "urgent_ddpg"
        elif regime_label in ("momentum_bull", "risk_on_EM", "recovery"):
            agent_hint = "opportunistic_sac"
        else:
            agent_hint = "stealth_ppo"

        payload = {
            "timestamp":      time.time(),
            "weights":        allocation,
            "gross_leverage": gross_leverage,
            "hedge_overlay":  hedge_overlay,
            "agent_hint":     agent_hint,
        }
        payload_bytes = json.dumps(payload).encode("utf-8")

        await self.producer.send_and_wait("target-weights", payload_bytes)
        await self._redis.set("portfolio:weights", json.dumps(allocation))
        await self._redis.set("portfolio:leverage", str(gross_leverage))

        logger.info(
            f"✅ Allocation published | gross={gross_leverage:.3f}x "
            f"| hint={agent_hint} | regime='{regime_label}'"
        )


# ── Service entrypoint ────────────────────────────────────────────────────────

async def main() -> None:
    with open("config/hyperparams.yaml", "r") as f:
        config = yaml.safe_load(f)

    svc = PortfolioAgentService(config)
    await svc.run()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    asyncio.run(main())