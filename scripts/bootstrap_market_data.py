"""
FORTRESS v5 - scripts/bootstrap_market_data.py
Rebuilds the foundational price and return matrices for the expanded 100-asset universe.
"""
import logging
import yaml
import pandas as pd
import yfinance as yf
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("Bootstrap")

_CACHE_DIR = Path("research/outputs/cache")
_UNIVERSE_FILE = Path("config/universe.yaml")

def main():
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    with open(_UNIVERSE_FILE, "r") as f:
        config = yaml.safe_load(f)
    
    tickers = [asset["ticker"] for asset in config["assets"]]
    logger.info(f"Bootstrapping market data for {len(tickers)} assets from 2018-01-01...")

    # Download prices with a wide buffer for 252d rolling windows
    raw = yf.download(tickers, start="2018-01-01", auto_adjust=True, progress=False)
    
    if raw.empty:
        logger.error("Download failed.")
        return

    # Handle multi-index columns from yfinance
    prices = raw["Close"] if "Close" in raw.columns else raw.xs("Close", axis=1, level=0)
    
    # Strictly align to our universe definition and ffill missing data
    prices = prices.reindex(columns=tickers).ffill()
    
    # Compute returns (drop fill_method to comply with pandas 2.1+ deprecation)
    returns = prices.pct_change().fillna(0.0)

    prices_path = _CACHE_DIR / "prices_wide.parquet"
    returns_path = _CACHE_DIR / "returns_wide.parquet"

    prices.to_parquet(prices_path)
    returns.to_parquet(returns_path)

    logger.info(f"✅ Saved prices: {prices_path} {prices.shape}")
    logger.info(f"✅ Saved returns: {returns_path} {returns.shape}")

if __name__ == "__main__":
    main()