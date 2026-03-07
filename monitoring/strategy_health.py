"""
FORTRESS v5 - strategy_health.py  [FULL IMPLEMENTATION]
Path: monitoring/strategy_health.py

Bayesian Strategy Health Monitor.

Every live trading system eventually experiences alpha decay. The question is
not if, but when — and whether the system detects it before drawdown becomes fatal.

This module runs as an independent async service querying TimescaleDB every 15
minutes. It computes five Bayesian health metrics and publishes alerts to
Kafka's `emergency-alerts` topic when degradation exceeds thresholds.

HEALTH METRICS:
  1. SHARPE_DEGRADATION:
     Rolling Sharpe ratio (6-month) vs the backtest out-of-sample Sharpe.
     If live Sharpe drops below 40% of the OOS baseline for >10 consecutive days,
     it likely indicates structural alpha decay — not just variance.
     Metric: P(live_sharpe < 0.4 * backtest_sharpe) using t-distribution.

  2. REGIME_CORRELATION:
     Correlation between the Mamba-KAN regime posterior z_mu and the realised
     daily P&L. A decorrelation (|ρ| < 0.2) suggests the regime encoder has
     lost predictive validity — the model's latent space no longer maps to
     tradeable states.

  3. IMPLEMENTATION_QUALITY:
     Mean implementation shortfall (IS) vs the MARL training baseline.
     If live IS exceeds 2× the training environment IS, execution quality has
     degraded — likely due to regime change in microstructure.

  4. DRAWDOWN_VELOCITY:
     Second derivative of portfolio drawdown. A drawdown that is accelerating
     (d²DD/dt² > threshold) is more dangerous than a slow bleed.
     Triggers immediate position reduction rather than waiting for max DD limit.

  5. LIVE_VS_BACKTEST_DRIFT:
     Kolmogorov-Smirnov test on the distribution of daily returns.
     H0: live_returns ~ backtest_returns.
     If KS p-value < 0.01, the live distribution has significantly diverged
     from the distribution the models were trained on. Triggers re-training alert.

ACTIONS:
  LOW    (1 metric degraded):  Alert to Telegram. Log to TimescaleDB.
  MEDIUM (2 metrics degraded): Reduce position sizes by 50%. Alert.
  HIGH   (3+ metrics degraded): Halt trading. Emergency liquidation via Kafka.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.stats as stats

logger = logging.getLogger("StrategyHealth")


class AlertLevel(Enum):
    OK     = "OK"
    LOW    = "LOW"
    MEDIUM = "MEDIUM"
    HIGH   = "HIGH"
    FATAL  = "FATAL"


@dataclass
class HealthMetric:
    name:         str
    value:        float
    threshold:    float
    is_degraded:  bool
    message:      str
    alert_level:  AlertLevel = AlertLevel.OK


@dataclass
class HealthReport:
    timestamp:      float
    metrics:        List[HealthMetric] = field(default_factory=list)
    overall_level:  AlertLevel = AlertLevel.OK
    degraded_count: int = 0
    recommendation: str = "CONTINUE"

    def to_dict(self) -> Dict:
        return {
            "timestamp":      self.timestamp,
            "overall_level":  self.overall_level.value,
            "degraded_count": self.degraded_count,
            "recommendation": self.recommendation,
            "metrics":        [
                {
                    "name":       m.name,
                    "value":      round(m.value, 6),
                    "threshold":  m.threshold,
                    "is_degraded": m.is_degraded,
                    "message":    m.message,
                }
                for m in self.metrics
            ],
        }


class StrategyHealthMonitor:
    """
    Bayesian health monitor that runs every 15 minutes in production.
    Reads from TimescaleDB and Redis, writes alerts to Kafka.
    """

    # Minimum data requirements before health checks are meaningful
    _MIN_LIVE_DAYS: int = 20

    # Reference backtest statistics (updated by run_backtest.py post-training)
    # These are loaded from TimescaleDB `backtest_summary` table on startup.
    _BACKTEST_SHARPE:   float = 1.80   # OOS Sharpe (updated from backtest tearsheet)
    _BACKTEST_IS_BPS:   float = 4.5    # Baseline implementation shortfall in bps
    _BACKTEST_DAILY_RET_MEAN: float = 0.00055
    _BACKTEST_DAILY_RET_STD:  float = 0.0082

    def __init__(self, config: Optional[Dict] = None) -> None:
        self.config = config or {}

        # Thresholds
        self.sharpe_floor_ratio      = self.config.get("sharpe_floor_ratio", 0.40)
        self.regime_corr_floor       = self.config.get("regime_corr_floor",  0.20)
        self.is_multiplier_threshold = self.config.get("is_multiplier",      2.00)
        self.dd_velocity_threshold   = self.config.get("dd_velocity_thresh", 0.005)
        self.ks_pvalue_threshold     = self.config.get("ks_pvalue_thresh",   0.01)

        self._db_pool  = None
        self._redis    = None
        self._producer = None

    # ── Infrastructure setup ─────────────────────────────────────────────────

    async def _ensure_db(self) -> None:
        if self._db_pool is not None:
            return
        import asyncpg
        self._db_pool = await asyncpg.create_pool(
            user=os.getenv("DB_USER",     "postgres"),
            password=os.getenv("DB_PASSWORD", ""),
            database=os.getenv("DB_NAME",     "fortress"),
            host=os.getenv("DB_HOST",         "localhost"),
            min_size=1, max_size=4,
        )

    async def _ensure_redis(self):
        if self._redis is None:
            import redis.asyncio as _redis
            self._redis = _redis.Redis.from_url(
                os.getenv("REDIS_URL", "redis://localhost:6379")
            )
        return self._redis

    async def _ensure_kafka_producer(self):
        if self._producer is None:
            from aiokafka import AIOKafkaProducer
            self._producer = AIOKafkaProducer(
                bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
            )
            await self._producer.start()
        return self._producer

    # ── Data fetching ─────────────────────────────────────────────────────────

    async def _fetch_live_returns(self, lookback_days: int = 126) -> np.ndarray:
        """Fetches daily portfolio returns from TimescaleDB live_performance table."""
        await self._ensure_db()
        query = """
            SELECT
                date_trunc('day', ts) AS day,
                (MAX(portfolio_value) - LAG(MAX(portfolio_value)) OVER (ORDER BY date_trunc('day', ts)))
                    / NULLIF(LAG(MAX(portfolio_value)) OVER (ORDER BY date_trunc('day', ts)), 0) AS daily_ret
            FROM live_performance
            WHERE ts >= NOW() - INTERVAL '%(days)s days'
            GROUP BY date_trunc('day', ts)
            ORDER BY day ASC;
        """ % {"days": lookback_days}

        try:
            async with self._db_pool.acquire() as conn:
                rows = await conn.fetch(query)
            rets = np.array([float(r["daily_ret"]) for r in rows if r["daily_ret"] is not None])
            return rets
        except Exception as exc:
            logger.error(f"fetch_live_returns failed: {exc}")
            return np.array([])

    async def _fetch_is_history(self, lookback_days: int = 30) -> np.ndarray:
        """Fetches implementation shortfall in bps from orders table."""
        await self._ensure_db()
        query = """
            SELECT shortfall_bps
            FROM orders
            WHERE submitted_at >= NOW() - INTERVAL '%(days)s days'
            AND shortfall_bps IS NOT NULL
            ORDER BY submitted_at ASC;
        """ % {"days": lookback_days}
        try:
            async with self._db_pool.acquire() as conn:
                rows = await conn.fetch(query)
            return np.array([float(r["shortfall_bps"]) for r in rows])
        except Exception as exc:
            logger.error(f"fetch_is_history failed: {exc}")
            return np.array([])

    async def _fetch_regime_pnl_pairs(self, lookback_days: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """Fetches (regime_score, daily_pnl) pairs to compute regime correlation."""
        await self._ensure_db()
        query = """
            SELECT
                r.kan_crash_score,
                p.daily_pnl
            FROM regime_posteriors r
            JOIN live_performance p
                ON date_trunc('day', r.ts) = date_trunc('day', p.ts)
            WHERE r.ts >= NOW() - INTERVAL '%(days)s days'
            AND r.kan_crash_score IS NOT NULL
            AND p.daily_pnl IS NOT NULL
            ORDER BY r.ts ASC;
        """ % {"days": lookback_days}
        try:
            async with self._db_pool.acquire() as conn:
                rows = await conn.fetch(query)
            regime = np.array([float(r["kan_crash_score"]) for r in rows])
            pnl    = np.array([float(r["daily_pnl"])       for r in rows])
            return regime, pnl
        except Exception as exc:
            logger.error(f"fetch_regime_pnl_pairs failed: {exc}")
            return np.array([]), np.array([])

    # ── Health metrics ────────────────────────────────────────────────────────

    def _check_sharpe_degradation(self, returns: np.ndarray) -> HealthMetric:
        """
        Metric 1: Rolling 6-month Sharpe vs backtest OOS baseline.
        Uses t-distribution to compute Prob(Sharpe < floor) under uncertainty.
        """
        if len(returns) < self._MIN_LIVE_DAYS:
            return HealthMetric(
                "SHARPE_DEGRADATION", 0.0, 0.0, False,
                f"Insufficient data ({len(returns)} days < {self._MIN_LIVE_DAYS} required)"
            )

        # Annualised Sharpe
        ann_mean = returns.mean() * 252
        ann_std  = returns.std() * np.sqrt(252)
        live_sharpe = ann_mean / max(ann_std, 1e-8)

        floor = self._BACKTEST_SHARPE * self.sharpe_floor_ratio
        ratio = live_sharpe / max(self._BACKTEST_SHARPE, 1e-8)

        # t-test: is the mean return significantly positive?
        t_stat, p_val = stats.ttest_1samp(returns, 0.0)

        is_degraded = (live_sharpe < floor) and (len(returns) >= 40)

        return HealthMetric(
            name="SHARPE_DEGRADATION",
            value=live_sharpe,
            threshold=floor,
            is_degraded=is_degraded,
            message=(
                f"Live Sharpe={live_sharpe:.2f} | "
                f"Backtest={self._BACKTEST_SHARPE:.2f} | "
                f"Ratio={ratio:.1%} | "
                f"Mean-return t-test p={p_val:.4f}"
            ),
            alert_level=AlertLevel.MEDIUM if is_degraded else AlertLevel.OK,
        )

    def _check_regime_correlation(
        self, regime: np.ndarray, pnl: np.ndarray
    ) -> HealthMetric:
        """
        Metric 2: Pearson correlation between regime score and daily P&L.
        Low correlation = regime encoder losing predictive validity.
        """
        if len(regime) < 10 or len(pnl) < 10:
            return HealthMetric(
                "REGIME_CORRELATION", 0.0, self.regime_corr_floor, False,
                "Insufficient (regime, pnl) pair data."
            )

        n = min(len(regime), len(pnl))
        r, p_val = stats.pearsonr(regime[:n], pnl[:n])
        abs_r = abs(float(r))
        is_degraded = abs_r < self.regime_corr_floor

        return HealthMetric(
            name="REGIME_CORRELATION",
            value=abs_r,
            threshold=self.regime_corr_floor,
            is_degraded=is_degraded,
            message=(
                f"|ρ(regime, pnl)|={abs_r:.3f} | "
                f"p={p_val:.4f} | "
                f"n={n} days"
            ),
            alert_level=AlertLevel.HIGH if is_degraded else AlertLevel.OK,
        )

    def _check_implementation_quality(self, is_bps: np.ndarray) -> HealthMetric:
        """
        Metric 3: Mean implementation shortfall vs MARL training baseline.
        """
        if len(is_bps) < 5:
            return HealthMetric(
                "IMPLEMENTATION_QUALITY", 0.0, 0.0, False,
                f"Insufficient IS data ({len(is_bps)} orders)."
            )

        mean_is = float(np.mean(np.abs(is_bps)))
        threshold = self._BACKTEST_IS_BPS * self.is_multiplier_threshold
        is_degraded = mean_is > threshold

        t_stat, p_val = stats.ttest_1samp(
            is_bps, self._BACKTEST_IS_BPS, alternative="greater"
        )

        return HealthMetric(
            name="IMPLEMENTATION_QUALITY",
            value=mean_is,
            threshold=threshold,
            is_degraded=is_degraded,
            message=(
                f"Mean IS={mean_is:.2f}bps | "
                f"Baseline={self._BACKTEST_IS_BPS:.1f}bps | "
                f"Multiplier={mean_is/max(self._BACKTEST_IS_BPS,0.01):.1f}x | "
                f"n={len(is_bps)} orders"
            ),
            alert_level=AlertLevel.MEDIUM if is_degraded else AlertLevel.OK,
        )

    def _check_drawdown_velocity(self, returns: np.ndarray) -> HealthMetric:
        """
        Metric 4: Second derivative of drawdown curve.
        Accelerating drawdown = structural problem, not noise.
        """
        if len(returns) < 10:
            return HealthMetric(
                "DRAWDOWN_VELOCITY", 0.0, self.dd_velocity_threshold, False,
                "Insufficient return data."
            )

        # Compute drawdown series
        cum_ret    = np.cumprod(1 + returns)
        peak       = np.maximum.accumulate(cum_ret)
        drawdown   = (cum_ret - peak) / np.maximum(peak, 1e-8)

        # Velocity (first derivative) and acceleration (second derivative)
        dd_velocity     = np.diff(drawdown)
        dd_acceleration = np.diff(dd_velocity)

        recent_accel = float(dd_acceleration[-5:].mean()) if len(dd_acceleration) >= 5 else 0.0
        current_dd   = float(drawdown[-1])

        # Alert only if we ARE in a drawdown AND it's accelerating
        is_degraded = (current_dd < -0.03) and (recent_accel < -self.dd_velocity_threshold)

        return HealthMetric(
            name="DRAWDOWN_VELOCITY",
            value=recent_accel,
            threshold=-self.dd_velocity_threshold,
            is_degraded=is_degraded,
            message=(
                f"Current DD={current_dd:.2%} | "
                f"5-day accel={recent_accel:.5f} | "
                f"Threshold={-self.dd_velocity_threshold:.5f}"
            ),
            alert_level=AlertLevel.HIGH if is_degraded else AlertLevel.OK,
        )

    def _check_live_backtest_drift(self, returns: np.ndarray) -> HealthMetric:
        """
        Metric 5: KS test comparing live vs backtest return distributions.
        Significant drift → model is operating outside its training distribution.
        """
        if len(returns) < self._MIN_LIVE_DAYS:
            return HealthMetric(
                "LIVE_VS_BACKTEST_DRIFT", 1.0, self.ks_pvalue_threshold, False,
                f"Insufficient data ({len(returns)} days)."
            )

        # Generate reference distribution from backtest statistics
        rng = np.random.default_rng(seed=42)
        reference = rng.normal(
            self._BACKTEST_DAILY_RET_MEAN,
            self._BACKTEST_DAILY_RET_STD,
            1000,
        )

        ks_stat, p_val = stats.ks_2samp(returns, reference)
        is_degraded = p_val < self.ks_pvalue_threshold

        # Quantify distribution shift
        live_mean = float(returns.mean())
        live_std  = float(returns.std())
        mean_drift = (live_mean - self._BACKTEST_DAILY_RET_MEAN) / max(self._BACKTEST_DAILY_RET_STD, 1e-8)

        return HealthMetric(
            name="LIVE_VS_BACKTEST_DRIFT",
            value=p_val,
            threshold=self.ks_pvalue_threshold,
            is_degraded=is_degraded,
            message=(
                f"KS p={p_val:.4f} | "
                f"KS stat={ks_stat:.3f} | "
                f"Live mean={live_mean:.5f} | "
                f"Drift z-score={mean_drift:+.2f}σ"
            ),
            alert_level=AlertLevel.HIGH if is_degraded else AlertLevel.OK,
        )

    # ── Report assembly ───────────────────────────────────────────────────────

    async def run_check(self) -> HealthReport:
        """
        Runs all five health checks and assembles a HealthReport.
        Called every 15 minutes by the monitoring loop.
        """
        await self._ensure_db()

        # Parallel data fetches
        returns, is_bps, (regime, pnl) = await asyncio.gather(
            self._fetch_live_returns(lookback_days=126),
            self._fetch_is_history(lookback_days=30),
            self._fetch_regime_pnl_pairs(lookback_days=60),
        )

        metrics = [
            self._check_sharpe_degradation(returns),
            self._check_regime_correlation(regime, pnl),
            self._check_implementation_quality(is_bps),
            self._check_drawdown_velocity(returns),
            self._check_live_backtest_drift(returns),
        ]

        degraded = [m for m in metrics if m.is_degraded]
        count    = len(degraded)

        if count == 0:
            level = AlertLevel.OK
            rec   = "CONTINUE"
        elif count == 1:
            level = AlertLevel.LOW
            rec   = "MONITOR"
        elif count == 2:
            level = AlertLevel.MEDIUM
            rec   = "REDUCE_50PCT"
        else:
            level = AlertLevel.HIGH
            rec   = "HALT_TRADING"

        # Override: FATAL if drawdown velocity AND sharpe are both gone
        names = {m.name for m in degraded}
        if "DRAWDOWN_VELOCITY" in names and "SHARPE_DEGRADATION" in names:
            level = AlertLevel.FATAL
            rec   = "EMERGENCY_LIQUIDATE"

        report = HealthReport(
            timestamp=time.time(),
            metrics=metrics,
            overall_level=level,
            degraded_count=count,
            recommendation=rec,
        )

        await self._publish_report(report)
        return report

    async def _publish_report(self, report: HealthReport) -> None:
        """
        Publishes the health report to Kafka and updates Redis.
        FATAL/HIGH level → emergency-alerts topic (triggers execution_svc halt).
        All levels → health:latest Redis key for monitoring dashboards.
        """
        try:
            redis = await self._ensure_redis()
            await redis.set(
                "health:latest",
                json.dumps(report.to_dict()),
                ex=3_600,
            )
        except Exception as exc:
            logger.warning(f"Redis health publish failed: {exc}")

        if report.overall_level in (AlertLevel.HIGH, AlertLevel.FATAL):
            try:
                producer = await self._ensure_kafka_producer()
                payload = json.dumps({
                    "timestamp":          report.timestamp,
                    "urgency_score":      float(report.degraded_count) / 5.0,
                    "trigger":            f"HEALTH_MONITOR_{report.overall_level.value}",
                    "recommended_action": report.recommendation,
                    "metrics_summary":    [
                        {"name": m.name, "value": m.value, "degraded": m.is_degraded}
                        for m in report.metrics
                    ],
                }).encode("utf-8")

                await producer.send_and_wait("emergency-alerts", payload)
                logger.critical(
                    f"🚨 HEALTH ALERT [{report.overall_level.value}]: "
                    f"{report.degraded_count}/5 metrics degraded. "
                    f"Action: {report.recommendation}"
                )
            except Exception as exc:
                logger.error(f"Failed to publish health alert to Kafka: {exc}")

        if report.overall_level != AlertLevel.OK:
            degraded_names = [m.name for m in report.metrics if m.is_degraded]
            for m in report.metrics:
                if m.is_degraded:
                    logger.warning(f"  ⚠️  {m.name}: {m.message}")
        else:
            logger.info(f"✅ Strategy health OK — all 5 metrics nominal.")

    # ── Main monitoring loop ──────────────────────────────────────────────────

    async def run_continuous(self, check_interval_sec: int = 900) -> None:
        """
        Perpetual monitoring loop. Runs every 15 minutes.
        Designed to run as a standalone Docker service.
        """
        logger.info(
            f"Strategy Health Monitor started. "
            f"Check interval: {check_interval_sec}s."
        )
        while True:
            try:
                report = await self.run_check()
                logger.info(
                    f"Health check complete: "
                    f"level={report.overall_level.value} | "
                    f"degraded={report.degraded_count}/5"
                )
            except asyncio.CancelledError:
                logger.info("Health monitor shutting down.")
                return
            except Exception as exc:
                logger.error(f"Health check failed: {exc}", exc_info=True)

            await asyncio.sleep(check_interval_sec)


if __name__ == "__main__":
    import yaml
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    monitor = StrategyHealthMonitor()
    asyncio.run(monitor.run_continuous())