"""
FORTRESS v5 — scripts/bootstrap_market_data.py  [v3.0 — VIX TERM STRUCTURE]

v3.0 Changes
------------
Dynamically appends "^VIX" and "^VIX3M" to the yfinance download list so that
both series are persisted inside prices_wide.parquet and returns_wide.parquet.
These indices are required by VTSLeadEngine (models/alpha/vts_lead.py) to
compute the daily VIX term-structure innovation shock.

Design decision: VIX/VIX3M are appended to the *download* list only — they do
NOT appear in the universe YAML config and are never passed to the portfolio
optimizer. Their presence as extra columns in the parquet files is intentional:
VTSLeadEngine reads them by name (^VIX, ^VIX3M) without the optimizer ever
seeing them.

Note: ^VIX and ^VIX3M have no Volume data — the volumes_wide.parquet and
ohlc_wide.parquet saves use `_AUX_INDICES` as the exclusion list to avoid
NaN-poisoned volume/OHLC columns for the universe assets.

v2.0 Changes
------------
Saves ohlc_wide.parquet in addition to prices/returns/volumes.
ohlc_wide uses MultiIndex columns (price_type, ticker) so slicing is:
    ohlc_df["Open"][ticker]  or  ohlc_df.xs("High", axis=1, level=0)
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

# VIX term-structure indices required by VTSLeadEngine.
# These are appended to the download request but kept out of the YAML universe.
_AUX_INDICES: List[str] = ["^VIX", "^VIX3M"]


def main() -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    with open(_UNIVERSE_FILE, "r") as f:
        config = yaml.safe_load(f)

    tickers: List[str] = [asset["ticker"] for asset in config["assets"]]

    # yfinance v0.2+ uses hyphen notation for share classes.
    _YF_REMAP = {"BRK.B": "BRK-B"}
    yf_tickers: List[str] = [_YF_REMAP.get(t, t) for t in tickers]
    _YF_REVERSE: dict = {v: k for k, v in _YF_REMAP.items()}

    # ── Phase 1 directive: append VIX term-structure indices ─────────────────
    # Appended AFTER the universe tickers so column ordering in the parquet
    # always places universe assets first (preserves existing column-index
    # assumptions in downstream loaders).
    yf_tickers_extended: List[str] = yf_tickers + _AUX_INDICES
    logger.info(
        f"Bootstrapping {len(tickers)} universe assets + "
        f"{len(_AUX_INDICES)} VIX indices from {_START_DATE}..."
    )

    # ── Download all OHLCV in a single request ────────────────────────────────
    raw = yf.download(
        yf_tickers_extended,
        start=_START_DATE,
        auto_adjust=True,   # consistent OHLC adjustment across splits/dividends
        progress=False,
        threads=True,
    )

    if raw.empty:
        logger.error("yfinance returned empty DataFrame. Check network / ticker list.")
        sys.exit(1)

    def _extract(price_type: str, columns: List[str]) -> pd.DataFrame:
        """
        Extract `price_type` from the MultiIndex raw download, rename
        BRK-B → BRK.B, reindex to `columns`, and forward-fill gaps.
        `columns` parameterised so callers can pass either the extended
        list (for prices/returns) or the universe-only list (for OHLC/volumes).
        """
        try:
            df = (
                raw[price_type]
                if isinstance(raw.columns, pd.MultiIndex)
                else raw.xs(price_type, axis=1, level=0)
            )
        except KeyError:
            # Some price types absent for index tickers (e.g. Volume for ^VIX)
            return pd.DataFrame(index=raw.index, columns=columns, dtype="float64")

        df = df.rename(columns=_YF_REVERSE)
        return df.reindex(columns=columns).ffill()

    # ── Close prices: full extended universe including VIX indices ────────────
    # Rename extended list: ^VIX and ^VIX3M don't need remapping (no BRK-B equiv)
    extended_with_remap = [_YF_REVERSE.get(t, t) for t in yf_tickers_extended]
    # De-duplicate: _YF_REVERSE maps BRK-B → BRK.B; VIX tickers pass through
    extended_out_cols: List[str] = tickers + _AUX_INDICES   # canonical names

    close_df   = _extract("Close",  extended_out_cols)
    # Verify we have both VIX series non-trivially populated
    for aux in _AUX_INDICES:
        if aux in close_df.columns:
            non_null = close_df[aux].notna().sum()
            logger.info(f"  {aux}: {non_null} non-null close prices")
        else:
            logger.error(f"  {aux} NOT in downloaded close prices — VTSLead will fail!")

    # Returns: extended (includes VIX pct_change for completeness)
    returns_df = close_df.pct_change()

    prices_path  = _CACHE_DIR / "prices_wide.parquet"
    returns_path = _CACHE_DIR / "returns_wide.parquet"
    close_df.to_parquet(prices_path)
    returns_df.to_parquet(returns_path)
    logger.info(f"prices_wide   → {prices_path}  {close_df.shape}")
    logger.info(f"returns_wide  → {returns_path} {returns_df.shape}")

    # ── Volume: universe only (^VIX/^VIX3M have no volume data) ─────────────
    vol_df = _extract("Volume", tickers)
    vol_path = _CACHE_DIR / "volumes_wide.parquet"
    vol_df.to_parquet(vol_path)
    logger.info(f"volumes_wide  → {vol_path}  {vol_df.shape}")

    # ── OHLC: universe only (index tickers lack meaningful OHLC) ─────────────
    ohlc_dict = {
        price_type: _extract(price_type, tickers)
        for price_type in ("Open", "High", "Low", "Close")
    }
    ohlc_wide = pd.concat(ohlc_dict, axis=1)   # MultiIndex: (price_type, ticker)
    ohlc_path = _CACHE_DIR / "ohlc_wide.parquet"
    ohlc_wide.to_parquet(ohlc_path)
    logger.info(f"ohlc_wide     → {ohlc_path}  {ohlc_wide.shape}")

    # ── Final integrity check ─────────────────────────────────────────────────
    n_dates = len(close_df)
    n_total = len(extended_out_cols)
    null_counts = close_df.isnull().sum()
    heavy_nulls = null_counts[null_counts > n_dates * 0.10]
    if not heavy_nulls.empty:
        logger.warning(f"Assets with >10% NaN closes: {heavy_nulls.to_dict()}")
    else:
        logger.info(f"✅ Bootstrap complete: {n_dates} dates × {n_total} series (no heavy NaN)")


if __name__ == "__main__":
    main()