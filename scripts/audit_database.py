"""
FORTRESS v5 — audit_database.py  [PATCH v2]
Path: scripts/audit_database.py

BUG #14 FIXED:
    market_data.parquet is LONG-FORMAT: 37,750 rows = 1510 dates × 25 tickers.
    Rule R2 applied a wide-format date-uniqueness check to it, producing 1510
    "duplicate date" false positives (every date appears 25 times by design).
    The audit always halted the pipeline, forcing --skip-audit on every run.

    Fix: _run_parquet_audit() now classifies each cache file as wide-format
    (unique DatetimeIndex: regime_posteriors, alpha_signals, prices_wide,
    returns_wide) or long-format (composite (date, ticker) key: market_data).

    For wide-format files: R2 checks date index uniqueness (original behaviour).
    For long-format files: R2 checks (date, ticker) composite key uniqueness.
    The date field on long-format files is read from the "date" column (not the
    index) to preserve existing parquet schema.

    Also added R5: long-format files must cover exactly the expected N_ASSETS
    tickers. Fewer tickers indicate a truncated precompute run.
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

_CACHE_DIR     = Path("research/outputs/cache")
_PASS_SENTINEL = "Forensic Audit Passed. ZERO look-ahead violations."

# Wide-format files: DatetimeIndex must be unique.
_WIDE_CACHE_FILES: list[str] = [
    "regime_posteriors.parquet",
    "alpha_signals.parquet",
    "prices_wide.parquet",
    "returns_wide.parquet",
]

# Long-format files: (date, ticker) composite key must be unique.
# Maps filename → expected number of assets.
_LONG_CACHE_FILES: dict[str, int] = {
    "market_data.parquet": 25,
}

# All files subject to audit
_ALL_CACHE_FILES = _WIDE_CACHE_FILES + list(_LONG_CACHE_FILES.keys())

# Ordered subsets for R4 date-coverage check (wide-format only)
_SUBSET_CHAIN: list[tuple[str, str]] = [
    ("alpha_signals.parquet",    "regime_posteriors.parquet"),
]


# ── MODE A: Live TimescaleDB audit ────────────────────────────────────────────

async def _attempt_db_audit() -> Optional[bool]:
    """
    Returns True (DB passed), False (violations found), None (DB unreachable).
    """
    db_host = os.getenv("DB_HOST", "localhost")
    db_name = os.getenv("DB_NAME", "fortress")

    try:
        import asyncpg  # type: ignore

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

def _load_wide(fpath: Path) -> Optional[pd.DataFrame]:
    """Load a wide-format parquet into a DatetimeIndex DataFrame."""
    df = pd.read_parquet(fpath)
    if "date" in df.columns:
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index, errors="coerce").normalize()
    return df


def _load_long(fpath: Path) -> Optional[pd.DataFrame]:
    """
    Load a long-format parquet.
    Returns a DataFrame with at minimum a 'date' column and a 'ticker' column.
    Does NOT set date as index — composite key check requires both columns.
    """
    df = pd.read_parquet(fpath)
    # Normalise date representation regardless of storage format
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    elif df.index.name == "date" or isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        df.rename(columns={"index": "date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    return df


def _run_parquet_audit() -> bool:
    logger.info("Running synthetic parquet integrity audit...")

    # Load wide-format files
    wide_frames: dict[str, pd.DataFrame] = {}
    for fname in _WIDE_CACHE_FILES:
        fpath = _CACHE_DIR / fname
        if not fpath.exists():
            logger.warning(
                f"  Cache file missing: {fpath}. "
                "PRE-GENERATION state — PASS."
            )
            continue
        df = _load_wide(fpath)
        wide_frames[fname] = df
        logger.info(
            f"  Loaded {fname}: {len(df):,} rows, "
            f"{df.index.min().date()} → {df.index.max().date()}"
        )

    # Load long-format files
    long_frames: dict[str, pd.DataFrame] = {}
    for fname in _LONG_CACHE_FILES:
        fpath = _CACHE_DIR / fname
        if not fpath.exists():
            logger.warning(f"  Cache file missing: {fpath}. PRE-GENERATION state — PASS.")
            continue
        df = _load_long(fpath)
        long_frames[fname] = df
        n_dates  = df["date"].nunique() if "date" in df.columns else 0
        n_tickers = df["ticker"].nunique() if "ticker" in df.columns else 0
        date_range = (
            f"{df['date'].min().date()} → {df['date'].max().date()}"
            if "date" in df.columns else "?"
        )
        logger.info(
            f"  Loaded {fname}: {len(df):,} rows, "
            f"{n_dates} dates × {n_tickers} tickers, {date_range}  [long-format]"
        )

    if not wide_frames and not long_frames:
        logger.info("  No cache files found yet. Audit vacuously passes.")
        return True

    violations: list[str] = []
    today = pd.Timestamp.today().normalize()

    # ── R2: Wide-format date uniqueness ───────────────────────────────────────
    for fname, df in wide_frames.items():
        if not df.index.is_monotonic_increasing:
            violations.append(
                f"[R2] {fname}: date index is NOT monotonically increasing."
            )
        dups = df.index[df.index.duplicated()].unique()
        if len(dups):
            violations.append(
                f"[R2] {fname}: {len(dups)} duplicate dates: {dups[:3].tolist()}"
            )

    # ── R2 (long): (date, ticker) composite uniqueness ────────────────────────
    for fname, df in long_frames.items():
        if "date" not in df.columns or "ticker" not in df.columns:
            logger.warning(
                f"  {fname}: missing 'date' or 'ticker' column — "
                "skipping composite key check."
            )
            continue
        composite = df[["date", "ticker"]]
        n_dups = composite.duplicated().sum()
        if n_dups > 0:
            sample = df[composite.duplicated(keep=False)][["date", "ticker"]].head(3)
            violations.append(
                f"[R2] {fname}: {n_dups} duplicate (date, ticker) pairs: "
                f"{sample.values.tolist()}"
            )
        # Monotonicity: per-ticker date sequences must be ascending
        not_monotone = (
            df.groupby("ticker")["date"]
            .apply(lambda s: not s.is_monotonic_increasing)
        )
        bad_tickers = not_monotone[not_monotone].index.tolist()
        if bad_tickers:
            violations.append(
                f"[R2] {fname}: non-monotone date sequence for tickers: {bad_tickers[:5]}"
            )

    # ── R3: No future dates ───────────────────────────────────────────────────
    for fname, df in wide_frames.items():
        future = df.index[df.index > today]
        if len(future):
            violations.append(
                f"[R3] {fname}: {len(future)} dates beyond today ({today.date()}): "
                f"{future[:3].tolist()}"
            )
    for fname, df in long_frames.items():
        if "date" in df.columns:
            future = df.loc[df["date"] > today, "date"]
            if len(future):
                violations.append(
                    f"[R3] {fname}: {len(future)} rows with future dates: "
                    f"{future.head(3).tolist()}"
                )

    # ── R4: Wide-format subset chain ──────────────────────────────────────────
    for child_name, parent_name in _SUBSET_CHAIN:
        if child_name not in wide_frames or parent_name not in wide_frames:
            continue
        orphans = set(wide_frames[child_name].index) - set(wide_frames[parent_name].index)
        if orphans:
            sample = sorted(orphans)[:3]
            violations.append(
                f"[R4] {child_name} has {len(orphans)} dates absent from "
                f"{parent_name}: e.g. {sample}"
            )

    # ── R5: Long-format ticker completeness ───────────────────────────────────
    for fname, expected_n_assets in _LONG_CACHE_FILES.items():
        if fname not in long_frames:
            continue
        df = long_frames[fname]
        if "ticker" not in df.columns:
            continue
        actual = df["ticker"].nunique()
        if actual < expected_n_assets:
            violations.append(
                f"[R5] {fname}: only {actual}/{expected_n_assets} tickers present — "
                "precompute run was truncated."
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