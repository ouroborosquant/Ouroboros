"""
FORTRESS v5 - options_data.py
Path: data/ingestion/options_data.py

Options Surface & Gamma Exposure (GEX) Ingestion.
Calculates the aggregate dealer hedging flow to predict intraday volatility clusters.
"""

import os
import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any

# External dependencies
try:
    import asyncpg
    import aiohttp
except ImportError:
    raise ImportError("Requires asyncpg and aiohttp.")

logger = logging.getLogger("OptionsPipeline")

class OptionsSurfaceIngestion:
    def __init__(self, db_pool: asyncpg.Pool, config: Dict[str, Any]):
        self.db_pool = db_pool
        # Usually, Polygon.io is the gold standard for institutional options data
        self.polygon_api_key = os.getenv("POLYGON_API_KEY")
        
        # We focus options tracking purely on the macro indices
        self.primary_underlyings = ['SPY', 'QQQ', 'IWM']

    async def fetch_option_chain(self, session: aiohttp.ClientSession, ticker: str) -> pd.DataFrame:
        """
        Fetches the complete active option chain for the underlying asset.
        """
        if not self.polygon_api_key:
            logger.warning("POLYGON_API_KEY missing. Returning dummy options surface.")
            return self._generate_dummy_chain(ticker)

        # Polygon REST API endpoint for fetching all active contracts
        url = f"https://api.polygon.io/v3/snapshot/options/{ticker}?apiKey={self.polygon_api_key}"
        
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    results = data.get('results', [])
                    
                    # Flatten into a DataFrame for vectorized Greek math
                    df = pd.json_normalize(results)
                    return df
                else:
                    logger.error(f"Polygon API Error {response.status} for {ticker}")
                    return pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to fetch option chain for {ticker}: {e}")
            return pd.DataFrame()

    def calculate_net_gex(self, chain_df: pd.DataFrame, current_price: float) -> float:
        """
        Calculates Market Maker Net Gamma Exposure (GEX).
        
        Formula Approximation:
        Call Gamma = + (Open Interest * Gamma * 100 * Spot Price)
        Put Gamma  = - (Open Interest * Gamma * 100 * Spot Price)
        
        Market makers are assumed to be short the calls (investors buy them) 
        and short the puts (investors buy them for protection).
        """
        if chain_df.empty or 'gamma' not in chain_df.columns:
            return 0.0

        # Filter out deep out-of-the-money options to save compute
        chain_df = chain_df[(chain_df['strike'] > current_price * 0.8) & 
                            (chain_df['strike'] < current_price * 1.2)]
        
        # Isolate calls and puts
        calls = chain_df[chain_df['contract_type'] == 'call']
        puts = chain_df[chain_df['contract_type'] == 'put']
        
        # Calculate Call GEX (Positive Gamma)
        call_gex = (calls['open_interest'] * calls['gamma'] * 100 * current_price).sum()
        
        # Calculate Put GEX (Negative Gamma)
        put_gex = (puts['open_interest'] * puts['gamma'] * 100 * current_price).sum()
        
        # Total Dealer Net GEX
        net_gex = call_gex - put_gex
        
        # Scale down to billions for interpretability
        return net_gex / 1_000_000_000.0

    async def run_intraday_sweep(self):
        """
        Executes an update of the volatility surface. Called hourly by DataPipeline.
        """
        logger.info("Initiating Options Surface & GEX Calculation...")
        
        async with aiohttp.ClientSession() as session:
            for ticker in self.primary_underlyings:
                # 1. Fetch current underlying spot price (Mocked to 500 for SPY)
                spot_price = 500.0 
                
                # 2. Fetch the chain
                chain_df = await self.fetch_option_chain(session, ticker)
                
                if not chain_df.empty:
                    # 3. Calculate metrics
                    net_gex_bn = self.calculate_net_gex(chain_df, spot_price)
                    
                    logger.info(f"[{ticker}] Total Net GEX: ${net_gex_bn:.2f} Billion")
                    
                    # 4. Save to TimescaleDB
                    await self._save_to_db(ticker, net_gex_bn, spot_price)

    async def _save_to_db(self, ticker: str, net_gex: float, spot: float):
        query = """
            INSERT INTO options_features (metric_date, timestamp, ticker, net_gex_bn, spot_price)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT DO NOTHING;
        """
        now = datetime.utcnow()
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute(query, now.date(), now, ticker, float(net_gex), float(spot))
        except Exception as e:
            logger.error(f"DB Insert failed for Options GEX: {e}")

    def _generate_dummy_chain(self, ticker: str) -> pd.DataFrame:
        """Scaffold method for testing without an API key."""
        n_contracts = 200
        spot = 500.0
        strikes = np.linspace(spot * 0.8, spot * 1.2, n_contracts)
        
        # Distribute gamma in a bell curve around the money
        gamma = np.exp(-0.5 * ((strikes - spot) / 10.0)**2) * 0.05
        oi = np.random.randint(100, 10000, n_contracts)
        
        types = ['call'] * (n_contracts // 2) + ['put'] * (n_contracts // 2)
        
        return pd.DataFrame({
            'strike': strikes,
            'gamma': gamma,
            'open_interest': oi,
            'contract_type': types
        })