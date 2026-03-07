"""
FORTRESS v5 - macro_ingestion.py
Path: data/ingestion/macro_ingestion.py

Ingests macroeconomic data from the FRED API.
CRITICAL: Uses the ALFRED vintage API to capture initially reported values
to mathematically guarantee zero look-ahead bias in the TimescaleDB.
"""

import os
import asyncio
import logging
import aiohttp
import asyncpg
from datetime import datetime, date, timedelta
from typing import Dict, Any, List

logger = logging.getLogger("MacroIngestion")

class MacroIngestionService:
    # FRED API endpoint for vintage observations
    FRED_OBS_URL = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(self, db_pool: asyncpg.Pool, macro_config: Dict[str, Any]):
        """
        Initializes the FRED ingestion service.
        macro_config should contain a list of series_ids and their publication lags.
        """
        self.db_pool = db_pool
        self.config = macro_config
        self.api_key = os.getenv("FRED_API_KEY")
        if not self.api_key:
            raise ValueError("FRED_API_KEY environment variable is not set.")
        
        # We process a maximum of 120 requests per minute to respect FRED rate limits
        self.rate_limit_semaphore = asyncio.Semaphore(2) 

    async def run(self):
        """
        The continuous async loop called by the master DataPipeline orchestrator.
        Runs once daily to check for new macroeconomic releases.
        """
        series_list = self.config.get('series_ids', ['T10Y2Y', 'NFCI', 'CPIAUCSL', 'UNRATE', 'WALCL'])
        
        while True:
            logger.info("Checking for daily FRED macroeconomic updates...")
            today_str = datetime.utcnow().strftime('%Y-%m-%d')
            
            tasks = [self.fetch_and_store_vintage(series, today_str, today_str) for series in series_list]
            await asyncio.gather(*tasks)
            
            logger.info("Macro update complete. Sleeping for 24 hours.")
            await asyncio.sleep(86400) # Sleep for 24 hours

    async def fetch_and_store_vintage(self, series_id: str, start_date: str, end_date: str):
        """
        Fetches data from FRED using the realtime_start and realtime_end parameters.
        This guarantees we pull the exact vintage of data available ON that date.
        """
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": start_date,
            "observation_end": end_date,
            # The 'realtime' parameters are the secret to preventing look-ahead bias
            "realtime_start": start_date, 
            "realtime_end": end_date
        }

        async with self.rate_limit_semaphore:
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(self.FRED_OBS_URL, params=params) as response:
                        if response.status != 200:
                            logger.error(f"FRED API Error {response.status} for {series_id}")
                            return
                        
                        data = await response.json()
                        observations = data.get('observations', [])
                        
                        if observations:
                            await self._insert_into_db(series_id, observations)
                            
                except Exception as e:
                    logger.error(f"Failed to fetch {series_id}: {str(e)}")
            
            # 1-second delay between requests to ensure we stay well below the 120/min limit
            await asyncio.sleep(1.0) 

    async def _insert_into_db(self, series_id: str, observations: List[Dict]):
        """
        Safely upserts the vintage data into the TimescaleDB hypertable.
        """
        query = """
            INSERT INTO macro_indicators (metric_date, as_of_date, series_id, value)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (metric_date, as_of_date, series_id) 
            DO UPDATE SET value = EXCLUDED.value;
        """
        
        # Prepare bulk insertion data
        records = []
        for obs in observations:
            if obs['value'] == '.':  # FRED uses '.' for null/missing values
                continue
                
            metric_date = datetime.strptime(obs['date'], '%Y-%m-%d').date()
            # The realtime_start from ALFRED is the exact date this specific value became public
            as_of_date = datetime.strptime(obs['realtime_start'], '%Y-%m-%d').date()
            value = float(obs['value'])
            
            records.append((metric_date, as_of_date, series_id, value))

        if not records:
            return

        async with self.db_pool.acquire() as conn:
            async with conn.transaction():
                await conn.executemany(query, records)
                
        logger.debug(f"Inserted {len(records)} vintage records for {series_id}.")

    async def seed_historical_data(self, start_date: str = "2000-01-01"):
        """
        Utility method used ONLY during initial setup (scripts/download_history.py).
        Downloads the entire vintage history for all configured series.
        Because it uses ALFRED, the as_of_date correctly preserves the exact timeline
        of historical revisions.
        """
        series_list = self.config.get('series_ids', ['T10Y2Y', 'NFCI', 'CPIAUCSL', 'UNRATE', 'WALCL'])
        today_str = datetime.utcnow().strftime('%Y-%m-%d')
        
        logger.info(f"Seeding historical ALFRED vintage data from {start_date}...")
        
        # For historical seeding, we expand the realtime window to capture all past revisions
        for series_id in series_list:
            await self.fetch_and_store_vintage(series_id, start_date, today_str)
            
        logger.info("Historical macro seeding complete.")