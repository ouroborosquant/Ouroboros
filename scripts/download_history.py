"""
FORTRESS v5 - download_history.py
Path: scripts/download_history.py

Historical Data Bootstrap Pipeline.
Seeds the TimescaleDB hypertables with decades of price and vintage macro data.
Must be run once before initiating any training loops.
"""

import os
import yaml
import asyncio
import logging
import asyncpg
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict

# External API clients
try:
    from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
except ImportError:
    raise ImportError("alpaca-py is required. Install via: pip install alpaca-py")

from data.ingestion.macro_ingestion import MacroIngestionService

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("HistoryBootstrap")

class HistoricalSeeder:
    def __init__(self, config_path: str = 'config/data_sources.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.alpaca_client = StockHistoricalDataClient(
            api_key=os.getenv('ALPACA_API_KEY'),
            secret_key=os.getenv('ALPACA_SECRET_KEY')
        )
        self.db_pool = None

    async def init_db(self):
        """Initializes the async connection pool to TimescaleDB."""
        self.db_pool = await asyncpg.create_pool(
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD'),
            database=os.getenv('DB_NAME', 'fortress'),
            host=os.getenv('DB_HOST', 'localhost'),
            port=os.getenv('DB_PORT', '5432')
        )
        logger.info("TimescaleDB connection pool established.")

    async def seed_price_history(self, start_date: datetime, end_date: datetime):
        """
        Downloads Daily OHLCV data for the ETF universe and inserts it into the DB.
        Because price data is immediately public, metric_date == as_of_date.
        """
        universe = self.config.get('universe', ['SPY', 'QQQ', 'TLT', 'GLD', 'HYG'])
        logger.info(f"Fetching price history for {len(universe)} assets from {start_date.date()} to {end_date.date()}...")

        request_params = StockBarsRequest(
            symbol_or_symbols=universe,
            timeframe=TimeFrame.Day,
            start=start_date,
            end=end_date
        )

        bars = self.alpaca_client.get_stock_bars(request_params)
        df = bars.df.reset_index()
        
        # Prepare data for bulk insert
        records = []
        for _, row in df.iterrows():
            ticker = row['symbol']
            # Convert timestamp to date
            metric_date = row['timestamp'].date() 
            # For prices, the date it happened is the date it became known
            as_of_date = metric_date 
            
            records.append((
                metric_date, as_of_date, ticker, 
                float(row['open']), float(row['high']), 
                float(row['low']), float(row['close']), 
                int(row['volume']), float(row['vwap'])
            ))

        query = """
            INSERT INTO prices (metric_date, as_of_date, ticker, open, high, low, close, volume, vwap)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (metric_date, ticker, as_of_date) DO NOTHING;
        """

        async with self.db_pool.acquire() as conn:
            async with conn.transaction():
                await conn.executemany(query, records)
                
        logger.info(f"Successfully inserted {len(records)} daily price records.")

    async def seed_macro_history(self, start_date: str):
        """Utilizes the MacroIngestionService to pull vintage ALFRED data."""
        macro_svc = MacroIngestionService(self.db_pool, self.config['macro_sources'])
        await macro_svc.seed_historical_data(start_date=start_date)

    async def run(self):
        await self.init_db()
        
        # Define historical window (e.g., last 15 years for robust deep learning)
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=365 * 15)
        
        try:
            # Run ingestion sequentially to respect system memory and DB locks
            await self.seed_price_history(start_date, end_date)
            await self.seed_macro_history(start_date.strftime('%Y-%m-%d'))
            logger.info("Historical bootstrap complete. The Organism now has memory.")
        finally:
            await self.db_pool.close()

if __name__ == "__main__":
    seeder = HistoricalSeeder()
    asyncio.run(seeder.run())