"""
FORTRESS v5 — scripts/bootstrap_market_data.py  [v2.0 — OHLC EXPANSION]

v2.0 Changes
------------
Saves ohlc_wide.parquet in addition to prices/returns/volumes.
ohlc_wide uses MultiIndex columns (price_type, ticker) so slicing is:
    ohlc_df["Open"][ticker]  or  ohlc_df.xs("High", axis=1, level=0)

Required by NightEffectEngine which needs adjusted Open, High, Low, Close
aligned to the same index as Close. yfinance auto_adjust=True ensures all
OHLC columns are adjusted consistently — critical for eliminating phantom
gap artefacts on dividend/split ex-dates.
"""
from __future__ import annotations

import logging
import sys
import yaml
from pathlib import Path
from typing import List

import pandas as pd
import yfinance as yf

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Bootstrap")

_CACHE_DIR     = Path("research/outputs/cache")
_UNIVERSE_FILE = Path("config/universe.yaml")
_START_DATE    = "2018-01-01"   # 252-day rolling window buffer


def main() -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    with open(_UNIVERSE_FILE, "r") as f:
        config = yaml.safe_load(f)

    tickers: List[str] = [asset["ticker"] for asset in config["assets"]]

    # yfinance v0.2+ uses hyphen notation for share classes.
    # BRK.B in universe.yaml must become BRK-B for the download request,
    # then the result column is renamed back to match the universe definition.
    _YF_REMAP = {"BRK.B": "BRK-B"}
    yf_tickers = [_YF_REMAP.get(t, t) for t in tickers]
    _YF_REVERSE = {v: k for k, v in _YF_REMAP.items()}

    logger.info(f"Bootstrapping {len(tickers)} assets from {_START_DATE}...")

    # ── Download all OHLCV in a single request ────────────────────────────────
    raw = yf.download(
        yf_tickers,
        start=_START_DATE,
        auto_adjust=True,   # CRITICAL: consistent adjustment across O/H/L/C
        progress=False,
        threads=True,
    )

    if raw.empty:
        logger.error("yfinance returned empty DataFrame. Check network / ticker list.")
        sys.exit(1)

    def _extract(price_type: str) -> pd.DataFrame:
        """Extract one price type, rename yf columns back to universe names, ffill."""
        try:
            df = raw[price_type] if price_type in raw.columns else raw.xs(
                price_type, axis=1, level=0
            )
        except KeyError:
            logger.warning(f"  {price_type} not found — skipping")
            return pd.DataFrame(index=raw.index, columns=tickers, dtype="float32")
        # Rename BRK-B → BRK.B (and any other remaps) before reindex
        df = df.rename(columns=_YF_REVERSE)
        return df.reindex(columns=tickers).ffill().astype("float32")

    open_df   = _extract("Open")
    high_df   = _extract("High")
    low_df    = _extract("Low")
    close_df  = _extract("Close")
    volume_df = _extract("Volume")

    # ── Individual wide parquets (Close / Returns / Volume) ───────────────────
    returns_df = close_df.pct_change().fillna(0.0)

    close_df.to_parquet(_CACHE_DIR / "prices_wide.parquet")
    returns_df.to_parquet(_CACHE_DIR / "returns_wide.parquet")
    volume_df.to_parquet(_CACHE_DIR / "volumes_wide.parquet")

    # ── Multi-level OHLC parquet ──────────────────────────────────────────────
    # Schema: MultiIndex columns (price_type, ticker) — e.g. ("Open", "AAPL")
    # Access pattern: ohlc_df["Open"]  →  (T, N) DataFrame of open prices
    ohlc_df = pd.concat(
        {
            "Open":  open_df,
            "High":  high_df,
            "Low":   low_df,
            "Close": close_df,
        },
        axis=1,
    )  # columns = MultiIndex [("Open","AAPL"), ("High","AAPL"), ...]
    ohlc_df.to_parquet(_CACHE_DIR / "ohlc_wide.parquet")

    # ── Diagnostics ───────────────────────────────────────────────────────────
    n_days = len(close_df)
    logger.info(f"✅ prices_wide.parquet  → {close_df.shape}")
    logger.info(f"✅ returns_wide.parquet → {returns_df.shape}")
    logger.info(f"✅ volumes_wide.parquet → {volume_df.shape}")
    logger.info(f"✅ ohlc_wide.parquet    → {ohlc_df.shape} (MultiIndex cols)")

    # Sanity-check: overnight gap on ex-dates should be <1% for adjusted prices
    _spot_check_adjustment(open_df, close_df, tickers[:5])


def _spot_check_adjustment(
    open_df: pd.DataFrame,
    close_df: pd.DataFrame,
    sample_tickers: List[str],
    gap_warn_threshold: float = 0.05,
) -> None:
    """
    Warn if any adjusted overnight gap exceeds `gap_warn_threshold` (5%).
    Persistent large gaps after adjustment indicate a data quality issue
    that will corrupt the NightEffectEngine signal.
    """
    import numpy as np

    for ticker in sample_tickers:
        if ticker not in open_df.columns:
            continue
        o = open_df[ticker].values
        c = close_df[ticker].values
        c_prev = np.concatenate([[np.nan], c[:-1]])
        with np.errstate(divide="ignore", invalid="ignore"):
            gaps = np.abs(np.log(o / c_prev))
        extreme = np.nansum(gaps > gap_warn_threshold)
        if extreme > 0:
            logger.warning(
                f"  ⚠️  {ticker}: {extreme} days with |overnight gap| > "
                f"{gap_warn_threshold*100:.0f}% — verify auto_adjust=True"
            )


if __name__ == "__main__":
    main()