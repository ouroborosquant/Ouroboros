"""
FORTRESS v5 - evolution_log.py  [FULL IMPLEMENTATION]
Path: meta_learning/evolution_log.py

Persistent Evolution Audit Log for the Constitutional Meta-Learning Agent.

Previously an empty file. meta_agent.py called `_log_evolution_event()` which
would attempt to use this module, causing an AttributeError on the first
successful mutation deployment.

Design:
  - Append-only JSON-Lines format (.jsonl) — one record per line.
    This is the simplest possible audit trail: no DB dependency, no schema
    migrations, human-readable, trivially parseable by pandas.
  - Each record captures the full context needed for human review:
    the generated code, the gate scores, the DSR, the deployment timestamp,
    and whether the organism is currently running the mutation live.
  - Redis caching: the last N records are cached in Redis for fast read
    by the monitoring dashboard without filesystem I/O on every tick.
  - Immutable: records are never modified after writing. A `reverted` flag
    is appended as a new record, not by mutating the original.

USAGE (from meta_agent.py):
    log = EvolutionLog()
    log.append(
        generated_code=new_code,
        dsr=dsr,
        sandbox_passed=True,
        stress_test_passed=True,
        deployed=True,
    )

    # Read the full audit trail
    history = log.read_all()

    # Revert the last deployment
    log.revert_last(reason="Live performance degraded after 24h")
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger("EvolutionLog")

_DEFAULT_LOG_PATH = "meta_learning/evolution_audit.jsonl"
_REDIS_LOG_KEY    = "meta:evolution_log"
_REDIS_MAX_CACHED = 50   # Cache the last 50 records in Redis


class EvolutionRecord:
    """
    Immutable data class representing one constitutional mutation event.
    """
    __slots__ = (
        "event_id", "timestamp", "generated_code",
        "dsr", "cvar_95", "sandbox_passed", "stress_test_passed",
        "deployed", "reverted", "revert_reason",
        "code_hash", "code_length_lines",
    )

    def __init__(
        self,
        event_id:            str,
        timestamp:           str,
        generated_code:      str,
        dsr:                 float,
        cvar_95:             float,
        sandbox_passed:      bool,
        stress_test_passed:  bool,
        deployed:            bool,
        reverted:            bool  = False,
        revert_reason:       str   = "",
    ):
        self.event_id           = event_id
        self.timestamp          = timestamp
        self.generated_code     = generated_code
        self.dsr                = dsr
        self.cvar_95            = cvar_95
        self.sandbox_passed     = sandbox_passed
        self.stress_test_passed = stress_test_passed
        self.deployed           = deployed
        self.reverted           = reverted
        self.revert_reason      = revert_reason
        # Derived fields — not stored in DB, computed for readability
        import hashlib
        self.code_hash          = hashlib.sha256(generated_code.encode()).hexdigest()[:12]
        self.code_length_lines  = len(generated_code.splitlines())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id":           self.event_id,
            "timestamp":          self.timestamp,
            "code_hash":          self.code_hash,
            "code_length_lines":  self.code_length_lines,
            "dsr":                self.dsr,
            "cvar_95":            self.cvar_95,
            "sandbox_passed":     self.sandbox_passed,
            "stress_test_passed": self.stress_test_passed,
            "deployed":           self.deployed,
            "reverted":           self.reverted,
            "revert_reason":      self.revert_reason,
            # Store the full code for audit — redact in public dashboards
            "generated_code":     self.generated_code,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvolutionRecord":
        return cls(
            event_id=d.get("event_id", ""),
            timestamp=d.get("timestamp", ""),
            generated_code=d.get("generated_code", ""),
            dsr=float(d.get("dsr", 0.0)),
            cvar_95=float(d.get("cvar_95", float("nan"))),
            sandbox_passed=bool(d.get("sandbox_passed", False)),
            stress_test_passed=bool(d.get("stress_test_passed", False)),
            deployed=bool(d.get("deployed", False)),
            reverted=bool(d.get("reverted", False)),
            revert_reason=str(d.get("revert_reason", "")),
        )


class EvolutionLog:
    """
    Append-only JSON-Lines audit log for all constitutional mutations.

    Thread-safe for single-process use. For multi-process safety,
    the log file should be stored on a distributed filesystem or in
    TimescaleDB in production.

    Args:
        log_path:     Path to the .jsonl audit file.
        redis_url:    Optional Redis URL for caching recent records.
    """

    def __init__(
        self,
        log_path: str = _DEFAULT_LOG_PATH,
        redis_url: Optional[str] = None,
    ):
        self.log_path  = log_path
        self._redis    = None
        self._redis_url = redis_url or os.getenv("REDIS_URL")

        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)

    def _get_redis(self):
        """Lazy Redis connection — only initialised if REDIS_URL is available."""
        if self._redis is not None:
            return self._redis
        if not self._redis_url:
            return None
        try:
            import redis
            self._redis = redis.Redis.from_url(self._redis_url, decode_responses=True)
            self._redis.ping()
            return self._redis
        except Exception as exc:
            logger.warning(f"Redis unavailable for evolution log caching: {exc}")
            return None

    def append(
        self,
        generated_code:     str,
        dsr:                float,
        cvar_95:            float  = float("nan"),
        sandbox_passed:     bool   = False,
        stress_test_passed: bool   = False,
        deployed:           bool   = False,
    ) -> EvolutionRecord:
        """
        Appends a new mutation event to the audit log.

        Args:
            generated_code:     The full Python source code produced by the LLM.
            dsr:                Deflated Sharpe Ratio from Gate 2 backtest.
            cvar_95:            CVaR-95 from Gate 3 stress test (NaN if not run).
            sandbox_passed:     Whether Gate 1 sandbox compilation passed.
            stress_test_passed: Whether Gate 3 diffusion stress test passed.
            deployed:           Whether the code reached Gate 4 shadow deployment.

        Returns:
            The EvolutionRecord that was written.
        """
        import uuid
        record = EvolutionRecord(
            event_id=str(uuid.uuid4())[:8],
            timestamp=datetime.utcnow().isoformat(),
            generated_code=generated_code,
            dsr=dsr,
            cvar_95=cvar_95,
            sandbox_passed=sandbox_passed,
            stress_test_passed=stress_test_passed,
            deployed=deployed,
        )

        # Write to JSONL file (atomic append)
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record.to_dict()) + "\n")
            logger.info(
                f"Evolution event {record.event_id} logged "
                f"(deployed={deployed}, DSR={dsr:.4f}, code_hash={record.code_hash})"
            )
        except OSError as exc:
            logger.error(f"Failed to write evolution log: {exc}")
            raise

        # Update Redis cache (best-effort)
        r = self._get_redis()
        if r is not None:
            try:
                r.lpush(_REDIS_LOG_KEY, json.dumps(record.to_dict()))
                r.ltrim(_REDIS_LOG_KEY, 0, _REDIS_MAX_CACHED - 1)
            except Exception as exc:
                logger.warning(f"Redis cache update failed: {exc}")

        return record

    def revert_last(self, reason: str = "") -> Optional[EvolutionRecord]:
        """
        Marks the most recently deployed (and not yet reverted) mutation as reverted.
        Appends a NEW record with reverted=True — does not modify existing records.

        Args:
            reason: Human-readable reason for the revert.

        Returns:
            The revert record, or None if no deployable record was found.
        """
        records = self.read_all()
        last_deployed = None
        for rec in reversed(records):
            if rec.deployed and not rec.reverted:
                last_deployed = rec
                break

        if last_deployed is None:
            logger.warning("No deployed, non-reverted record found to revert.")
            return None

        revert_record = EvolutionRecord(
            event_id=f"REVERT-{last_deployed.event_id}",
            timestamp=datetime.utcnow().isoformat(),
            generated_code=last_deployed.generated_code,
            dsr=last_deployed.dsr,
            cvar_95=last_deployed.cvar_95,
            sandbox_passed=last_deployed.sandbox_passed,
            stress_test_passed=last_deployed.stress_test_passed,
            deployed=False,
            reverted=True,
            revert_reason=reason,
        )

        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(revert_record.to_dict()) + "\n")
            logger.warning(
                f"Reverted evolution {last_deployed.event_id}. Reason: {reason}"
            )
        except OSError as exc:
            logger.error(f"Failed to write revert record: {exc}")
            raise

        return revert_record

    def read_all(self) -> List[EvolutionRecord]:
        """
        Reads and parses the entire audit log.

        Returns:
            List of EvolutionRecord objects in chronological order.
        """
        if not os.path.exists(self.log_path):
            return []

        records: List[EvolutionRecord] = []
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        records.append(EvolutionRecord.from_dict(d))
                    except (json.JSONDecodeError, KeyError) as exc:
                        logger.warning(f"Corrupted log line {line_no}: {exc}")
        except OSError as exc:
            logger.error(f"Could not read evolution log: {exc}")

        return records

    def summary(self) -> Dict[str, Any]:
        """
        Returns a summary dict for the monitoring dashboard.

        Keys: total_evolutions, deployed_count, reverted_count,
              best_dsr, worst_dsr, last_deployed_at.
        """
        records = self.read_all()
        deployed = [r for r in records if r.deployed and not r.reverted]
        reverted = [r for r in records if r.reverted]

        dsrs = [r.dsr for r in records if np.isfinite(r.dsr) if not (r.dsr != r.dsr)]

        return {
            "total_evolutions":    len(records),
            "deployed_count":      len(deployed),
            "reverted_count":      len(reverted),
            "best_dsr":            max(dsrs) if dsrs else None,
            "worst_dsr":           min(dsrs) if dsrs else None,
            "last_deployed_at":    deployed[-1].timestamp if deployed else None,
            "last_event_id":       records[-1].event_id if records else None,
        }


# Import guard for numpy — needed by summary()
import numpy as np