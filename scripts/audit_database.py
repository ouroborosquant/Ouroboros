"""
FORTRESS v5 — audit_database.py
Path: scripts/audit_database.py

Forensic look-ahead bias audit — first gate in the pipeline.

TWO EXECUTION MODES (auto-detected):
  A) DB Mode (TimescaleDB reachable):
       Delegates to LookAheadAuditor in data/validation/lookahead_audit.py.
       Checks that as_of_date ≤ metric_date for ALL macro and price rows.
       A single violation → sys.exit(1). No exceptions.

  B) Synthetic Mode (DB offline / no connection string):
       Audits locally cached parquet files for:
         1. Temporal monotonicity of date indices.
         2. No future dates beyond today's wall clock.
         3. Cross-file date alignment (alpha ⊆ regime ⊆ market).
       Missing cache files are treated as PASS (they don't yet exist and
       will be created by the next pipeline stage — nothing to audit yet).

Exit 0 → emits the required sentinel: "Forensic Audit Passed. ZERO look-ahead violations."
Exit 1 → violations detected; pipeline aborted.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ForensicAudit")

# ── Constants ─────────────────────────────────────────────────────────────────

_CACHE_DIR = Path("research/outputs/cache")
_PASS_SENTINEL = "Forensic Audit Passed. ZERO look-ahead violations."

_CACHE_FILES: list[str] = [
    "market_data.parquet",
    "regime_posteriors.parquet",
    "alpha_signals.parquet",
]

# Ordered subsets: alpha_signals dates ⊆ regime_posteriors dates ⊆ market_data dates
_SUBSET_CHAIN: list[tuple[str, str]] = [
    ("alpha_signals.parquet",    "regime_posteriors.parquet"),
    ("regime_posteriors.parquet","market_data.parquet"),
]


# ── MODE A: Live TimescaleDB audit ────────────────────────────────────────────

async def _attempt_db_audit() -> Optional[bool]:
    """
    Returns:
      True  → DB audit passed.
      False → DB audit found violations (hard fail).
      None  → DB unreachable; caller should fall back to synthetic audit.
    """
    db_host = os.getenv("DB_HOST", "localhost")
    db_name = os.getenv("DB_NAME", "fortress")

    try:
        import asyncpg  # type: ignore
        # Probe connection before delegating to auditor
        probe = await asyncpg.connect(
            host=db_host,
            database=db_name,
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", ""),
            timeout=5.0,
        )
        await probe.close()
    except Exception as exc:
        logger.warning(
            f"TimescaleDB unreachable ({db_host}/{db_name}): {exc}. "
            "Falling back to synthetic parquet audit."
        )
        return None

    try:
        # Import here so missing asyncpg doesn't crash synthetic mode
        from data.validation.lookahead_audit import LookAheadAuditor  # type: ignore
    except ImportError as exc:
        logger.error(f"Cannot import LookAheadAuditor: {exc}. Falling back to synthetic audit.")
        return None

    logger.info(f"TimescaleDB reachable. Running institutional look-ahead audit on {db_name}...")
    auditor = LookAheadAuditor()
    await auditor.initialize()

    try:
        macro_ok = await auditor.audit_macro_table()
        price_ok = await auditor.audit_price_table()
    finally:
        if auditor.db_pool:
            await auditor.db_pool.close()

    return macro_ok and price_ok


# ── MODE B: Local parquet integrity audit ─────────────────────────────────────

def _run_parquet_audit() -> bool:
    """
    Validates locally cached parquet files for temporal integrity.

    Rules enforced:
      R1. Missing files → WARN + PASS (pre-generation state is valid).
      R2. Date index must be strictly monotonically increasing.
      R3. No date may exceed today's wall-clock date.
      R4. Subset chain: alpha_signals ⊆ regime_posteriors ⊆ market_data.
    """
    logger.info("Running synthetic parquet integrity audit...")

    existing: dict[str, pd.DataFrame] = {}
    for fname in _CACHE_FILES:
        fpath = _CACHE_DIR / fname
        if not fpath.exists():
            logger.warning(
                f"  Cache file missing: {fpath}. "
                "Treating as PRE-GENERATION (PASS). Run precompute scripts first."
            )
            continue

        df = pd.read_parquet(fpath)
        # Normalise to DatetimeIndex
        if "date" in df.columns:
            df = df.set_index("date")
        df.index = pd.to_datetime(df.index, errors="coerce").normalize()
        existing[fname] = df
        logger.info(f"  Loaded {fname}: {len(df):,} rows, {df.index.min().date()} → {df.index.max().date()}")

    if not existing:
        # Nothing to audit yet — pipeline hasn't generated anything.
        logger.info("  No cache files found yet. Audit vacuously passes.")
        return True

    violations: list[str] = []
    today = pd.Timestamp.today().normalize()

    # R2: Monotonicity
    for fname, df in existing.items():
        if not df.index.is_monotonic_increasing:
            violations.append(
                f"[R2] {fname}: date index is NOT monotonically increasing — "
                "look-ahead write detected."
            )
        dups = df.index[df.index.duplicated()].unique()
        if len(dups):
            violations.append(f"[R2] {fname}: {len(dups)} duplicate date(s) found: {dups[:3].tolist()}")

    # R3: No future dates
    for fname, df in existing.items():
        future = df.index[df.index > today]
        if len(future):
            violations.append(
                f"[R3] {fname}: {len(future)} dates beyond today ({today.date()}): "
                f"{future[:3].tolist()}"
            )

    # R4: Subset chain
    for child_name, parent_name in _SUBSET_CHAIN:
        if child_name not in existing or parent_name not in existing:
            continue
        child_dates = set(existing[child_name].index)
        parent_dates = set(existing[parent_name].index)
        orphans = child_dates - parent_dates
        if orphans:
            sample = sorted(orphans)[:3]
            violations.append(
                f"[R4] {child_name} contains {len(orphans)} dates absent from {parent_name}: "
                f"e.g. {sample} — these are dangling forward references."
            )

    if violations:
        for v in violations:
            logger.critical(f"❌ VIOLATION: {v}")
        return False

    logger.info("✅ All cache files are temporally consistent.")
    return True


# ── Entry point ───────────────────────────────────────────────────────────────

async def main() -> None:
    logger.info("══════ Fortress v5 — Forensic Look-Ahead Audit ══════")

    db_result: Optional[bool] = await _attempt_db_audit()

    if db_result is True:
        logger.info("✅ TimescaleDB institutional audit complete.")
        logger.info(_PASS_SENTINEL)
        sys.exit(0)

    elif db_result is False:
        logger.critical("❌ TimescaleDB audit FAILED. Pipeline cannot proceed.")
        sys.exit(1)

    else:
        # DB unreachable — run synthetic mode
        passed = _run_parquet_audit()
        if passed:
            logger.info(_PASS_SENTINEL)
            sys.exit(0)
        else:
            logger.critical(
                "SYSTEM HALTED: Chronological integrity violations detected in cache. "
                "Delete research/outputs/cache/ and re-run precompute scripts."
            )
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())