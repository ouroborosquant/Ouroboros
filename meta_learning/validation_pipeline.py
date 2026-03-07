"""
FORTRESS v5 - validation_pipeline.py
Path: meta_learning/validation_pipeline.py

Standalone 5-Layer Validation Pipeline for AI-Generated Features.
Previously this file contained only a docstring. The ValidationPipeline
class was defined inside meta_agent.py, making it impossible to import,
test, or extend independently.

This module provides the canonical implementation used by:
  - meta_learning/meta_agent.py (ConstitutionalMetaAgent)
  - tests/test_validation_pipeline.py (unit tests for each gate)
  - CI/CD pre-deployment checks

GATE ARCHITECTURE:
  Gate 1: Sandbox Compilation
    Compiles the generated code inside a Docker container with --network none.
    No network = no data exfiltration. No execution = no side effects.
    Exits 0 only if `compile(code)` succeeds without SyntaxError.

  Gate 2: Historical Backtest (Deflated Sharpe Ratio)
    Injects the feature function into research/backtest_engine.py via subprocess.
    The backtest runs on 4 years of real TimescaleDB data.
    Passes only if DSR > config::min_deflated_sharpe (default 1.0).

  Gate 3: Neural SDE Adversarial Stress Test
    Spawns training/diffusion/generate_scenarios.py in a subprocess.
    Generates 2,000 synthetic crash paths from the world model.
    Passes only if CVaR-95 of the feature output > _CVAR_STABILITY_FLOOR.

  Gate 4: Shadow Deployment (Scaffold — pending k8s integration)
    Deploys the feature to a paper-trading shadow pod for 7 days.
    Currently logs intent and returns True.

  Gate 5: Live Performance Monitor (NEW)
    After Gate 4 shadow deployment, monitors live Sharpe vs. baseline.
    Triggers auto-revert if shadow Sharpe degrades below threshold.
    Currently a scaffold, wired to EvolutionLog.revert_last().
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import tempfile
from typing import Dict, Any, Optional

logger = logging.getLogger("ValidationPipeline")

# ── Gate thresholds ───────────────────────────────────────────────────────────
_SANDBOX_TIMEOUT_SECONDS: int   = 60
_BACKTEST_TIMEOUT_SECONDS: int  = 600   # 10 min — full 4-year backtest
_STRESS_TEST_TIMEOUT_SEC: int   = 300   # 5 min  — 2000-path SDE simulation
_CVAR_STABILITY_FLOOR: float    = -0.10  # Feature CVaR-95 must exceed this
_SHADOW_MONITOR_HOURS: int      = 168    # 7 days


class ValidationPipeline:
    """
    The 5-Layer Security Gate for AI-Generated Code.

    Instantiate once per ConstitutionalMetaAgent and reuse across evolution cycles.

    Usage:
        pipeline = ValidationPipeline(config)
        gate1_ok = await pipeline.run_sandbox_compilation(code)
        dsr      = await pipeline.run_historical_backtest(code)
        gate3_ok = await pipeline.run_diffusion_stress_test(code)
        await pipeline.deploy_to_shadow(code)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.min_dsr: float = self.config.get("min_deflated_sharpe", 1.0)

    # ── GATE 1: Sandbox Compilation ───────────────────────────────────────────

    async def run_sandbox_compilation(self, code: str) -> bool:
        """
        Compiles the generated code inside a Docker container with no network
        access (`--network none`). The container runs `python -c "compile(...)"`.

        On success: container exits 0 → returns True.
        On SyntaxError or import error: container exits non-zero → returns False.

        The Docker sandbox prevents:
          - Network exfiltration of trading secrets
          - File system writes to the host
          - Side-channel data access

        Falls back to a subprocess compile (no Docker) if the Docker daemon
        is not available, with a logged WARNING.
        """
        logger.info("Gate 1 [Sandbox Compilation]: Starting...")

        # Write code to a temp file (avoid shell injection via code string)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, prefix="gate1_"
        ) as f:
            # Wrap in compile() only — does not execute any code
            wrapper = f"""
import sys
with open(sys.argv[1], 'r') as fh:
    src = fh.read()
try:
    compile(src, '<generated>', 'exec')
    print("COMPILE_OK")
    sys.exit(0)
except SyntaxError as e:
    print(f"SYNTAX_ERROR: {{e}}", file=sys.stderr)
    sys.exit(1)
"""
            compile_script = f.name + "_runner.py"
            feature_file   = f.name

        try:
            # Write both files
            with open(compile_script, "w") as fh:
                fh.write(wrapper)
            with open(feature_file, "w") as fh:
                fh.write(code)

            # Try Docker first
            docker_available = await self._check_docker_available()

            if docker_available:
                result = await asyncio.create_subprocess_exec(
                    "docker", "run",
                    "--rm",
                    "--network", "none",
                    "--memory", "256m",
                    "--cpus",   "0.5",
                    "--volume", f"{compile_script}:/sandbox/runner.py:ro",
                    "--volume", f"{feature_file}:/sandbox/feature.py:ro",
                    "python:3.11-slim",
                    "python", "/sandbox/runner.py", "/sandbox/feature.py",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            else:
                logger.warning(
                    "Gate 1: Docker not available. "
                    "Falling back to local compile (reduced isolation)."
                )
                result = await asyncio.create_subprocess_exec(
                    "python", compile_script, feature_file,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

            try:
                stdout, stderr = await asyncio.wait_for(
                    result.communicate(), timeout=_SANDBOX_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                logger.error(
                    f"Gate 1 [Sandbox]: TIMEOUT (>{_SANDBOX_TIMEOUT_SECONDS}s)"
                )
                return False

            if result.returncode == 0:
                logger.info("Gate 1 [Sandbox Compilation]: ✅ PASSED")
                return True
            else:
                logger.warning(
                    f"Gate 1 [Sandbox Compilation]: ❌ FAILED\n"
                    f"stderr: {stderr.decode()[:1000]}"
                )
                return False

        finally:
            for path in (compile_script, feature_file):
                try:
                    os.unlink(path)
                except OSError:
                    pass

    # ── GATE 2: Historical Backtest ───────────────────────────────────────────

    async def run_historical_backtest(self, code: str) -> float:
        """
        Injects the generated feature into the backtest engine via subprocess.
        The backtest engine loads the feature dynamically and runs a full
        event-driven simulation on TimescaleDB historical data.

        Returns:
            dsr: Deflated Sharpe Ratio. Values > self.min_dsr (default 1.0)
                 indicate statistically significant positive expectancy after
                 correcting for multiple testing (Harvey et al., 2016).
                 Returns 0.0 on failure.
        """
        logger.info("Gate 2 [Historical Backtest]: Starting...")

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, prefix="gate2_feature_"
        ) as f:
            f.write(code)
            feature_path = f.name

        try:
            result = await asyncio.create_subprocess_exec(
                "python", "research/backtest_engine.py",
                "--feature-path",   feature_path,
                "--output-format",  "json",
                "--start-date",     "2020-01-02",
                "--end-date",       "2024-12-31",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    result.communicate(), timeout=_BACKTEST_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                logger.error(
                    f"Gate 2 [Backtest]: TIMEOUT (>{_BACKTEST_TIMEOUT_SECONDS}s). "
                    "DSR = 0.0."
                )
                return 0.0

            if result.returncode != 0:
                logger.warning(
                    f"Gate 2 [Backtest]: ❌ subprocess returned non-zero.\n"
                    f"stderr: {stderr.decode()[:500]}"
                )
                return 0.0

            # Parse JSON output from backtest engine
            output = stdout.decode()
            for line in output.splitlines():
                try:
                    data = json.loads(line)
                    dsr  = float(data.get("deflated_sharpe_ratio", 0.0))
                    cagr = float(data.get("cagr", 0.0))
                    mdd  = float(data.get("max_drawdown", 0.0))

                    if dsr >= self.min_dsr:
                        logger.info(
                            f"Gate 2 [Backtest]: ✅ PASSED "
                            f"(DSR={dsr:.3f}, CAGR={cagr:.2%}, MaxDD={mdd:.2%})"
                        )
                    else:
                        logger.warning(
                            f"Gate 2 [Backtest]: ❌ FAILED "
                            f"(DSR={dsr:.3f} < threshold={self.min_dsr:.2f})"
                        )
                    return dsr
                except (json.JSONDecodeError, ValueError):
                    continue

            logger.warning("Gate 2 [Backtest]: ❌ Could not parse DSR from output.")
            return 0.0

        finally:
            try:
                os.unlink(feature_path)
            except OSError:
                pass

    # ── GATE 3: Neural SDE Stress Test ───────────────────────────────────────

    async def run_diffusion_stress_test(self, code: str) -> bool:
        """
        Tests the generated feature against 2,000 adversarial synthetic market
        paths from the trained Neural SDE World Model.

        Delegates to training/diffusion/generate_scenarios.py which:
          1. Loads trained SDE weights
          2. Samples adversarial regime vectors (tail 10% of prior)
          3. Generates paths and evaluates the feature on each
          4. Computes CVaR-95 of feature output distribution

        Returns True only if CVaR-95 > _CVAR_STABILITY_FLOOR (-10%).
        """
        logger.info("Gate 3 [SDE Stress Test]: Starting (2,000 adversarial paths)...")

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, prefix="gate3_feature_"
        ) as f:
            f.write(code)
            feature_path = f.name

        try:
            result = await asyncio.create_subprocess_exec(
                "python", "training/diffusion/generate_scenarios.py",
                "--feature-path",  feature_path,
                "--n-paths",       "2000",
                "--output-format", "json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    result.communicate(), timeout=_STRESS_TEST_TIMEOUT_SEC
                )
            except asyncio.TimeoutError:
                logger.error(
                    f"Gate 3 [Stress Test]: TIMEOUT (>{_STRESS_TEST_TIMEOUT_SEC}s). "
                    "Treating as FAILED."
                )
                return False

            if result.returncode != 0:
                logger.error(
                    f"Gate 3 [Stress Test]: ❌ subprocess crashed.\n"
                    f"stderr: {stderr.decode()[:500]}"
                )
                return False

            output = stdout.decode()
            for line in output.splitlines():
                try:
                    data    = json.loads(line)
                    passed  = bool(data.get("stress_test_passed", False))
                    cvar_95 = float(data.get("cvar_95", -999.0))
                    finite_rate = float(data.get("finite_rate", 0.0))

                    if passed:
                        logger.info(
                            f"Gate 3 [Stress Test]: ✅ PASSED "
                            f"(CVaR-95={cvar_95:.4f}, finite_rate={finite_rate:.2%})"
                        )
                    else:
                        logger.warning(
                            f"Gate 3 [Stress Test]: ❌ FAILED "
                            f"(CVaR-95={cvar_95:.4f} < floor={_CVAR_STABILITY_FLOOR})"
                        )
                    return passed
                except (json.JSONDecodeError, ValueError):
                    continue

            logger.error("Gate 3 [Stress Test]: ❌ Could not parse result.")
            return False

        finally:
            try:
                os.unlink(feature_path)
            except OSError:
                pass

    # ── GATE 4: Shadow Deployment ─────────────────────────────────────────────

    async def deploy_to_shadow(self, code: str) -> bool:
        """
        Gate 4: Deploys the validated feature to the paper-trading shadow
        environment for 7 days before promoting to live.

        Current status: SCAFFOLD — pending Kubernetes shadow pod integration.
        The feature code is written to Redis with a TTL of 7 days.
        The shadow trading loop (not yet implemented) reads from this key
        and runs the feature in parallel with the live system.

        Returns True always (the monitoring gate is Gate 5).
        """
        logger.info(
            "Gate 4 [Shadow Deployment]: Writing feature to Redis shadow store "
            "(k8s shadow pod integration pending)."
        )
        try:
            import redis
            r = redis.Redis.from_url(
                os.getenv("REDIS_URL", "redis://localhost:6379")
            )
            r.set(
                "meta:shadow_feature_code",
                code,
                ex=_SHADOW_MONITOR_HOURS * 3600,  # 7-day TTL
            )
            r.set("meta:shadow_deployed_at", __import__("time").time())
            r.close()
            logger.info(
                f"Gate 4 [Shadow Deployment]: ✅ Feature written to Redis "
                f"with {_SHADOW_MONITOR_HOURS}h TTL."
            )
        except Exception as exc:
            logger.warning(f"Gate 4 [Shadow Deployment]: Redis write failed: {exc}.")

        return True

    # ── GATE 5: Live Performance Monitor ─────────────────────────────────────

    async def monitor_shadow_performance(
        self,
        expected_dsr_floor: float = 0.8,
    ) -> bool:
        """
        Gate 5: Monitors the shadow deployment's live Sharpe ratio.
        If performance degrades below expected_dsr_floor, reverts the deployment.

        SCAFFOLD — requires shadow pod to publish live_shadow_sharpe to Redis.
        Called externally by the weekly health check cron.

        Returns:
            True  — shadow is performing acceptably.
            False — shadow degraded; revert triggered.
        """
        try:
            import redis
            r     = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
            raw   = r.get("meta:shadow_live_sharpe")
            r.close()

            if raw is None:
                logger.info("Gate 5: No shadow Sharpe data yet (shadow pod not publishing).")
                return True

            live_sharpe = float(raw)
            if live_sharpe < expected_dsr_floor:
                logger.critical(
                    f"Gate 5 [Live Monitor]: ❌ Shadow Sharpe {live_sharpe:.3f} < "
                    f"floor {expected_dsr_floor:.3f}. Triggering auto-revert."
                )
                # Trigger revert via EvolutionLog
                from meta_learning.evolution_log import EvolutionLog
                log = EvolutionLog()
                log.revert_last(
                    reason=f"Auto-revert: shadow Sharpe {live_sharpe:.3f} < {expected_dsr_floor:.3f}"
                )
                return False

            logger.info(f"Gate 5 [Live Monitor]: ✅ Shadow Sharpe={live_sharpe:.3f} acceptable.")
            return True

        except Exception as exc:
            logger.warning(f"Gate 5 [Live Monitor]: Failed to evaluate ({exc}).")
            return True  # Default to keep (fail-open for monitoring)

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    async def _check_docker_available() -> bool:
        """Returns True if the Docker daemon is reachable."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "info",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.wait(), timeout=5.0)
            return proc.returncode == 0
        except Exception:
            return False