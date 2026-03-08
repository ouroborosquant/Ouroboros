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

    def __init__(self, db_pool: asyncpg.Pool, macro_config: Any):
        """
        Initializes the FRED ingestion service.
        macro_config can be a list of dicts (from data_sources.yaml) 
        or a dict containing series_ids.
        """
        self.db_pool = db_pool
        self.config = macro_config
        self.api_key = os.getenv("FRED_API_KEY")
        if not self.api_key:
            raise ValueError("FRED_API_KEY environment variable is not set.")
        
        # We process a maximum of 120 requests per minute to respect FRED rate limits
        self.rate_limit_semaphore = asyncio.Semaphore(2) 

    def _get_series_list(self) -> List[str]:
        """Helper to extract series IDs from various config formats."""
        if isinstance(self.config, list):
            # Handles the list of dicts format in config/data_sources.yaml
            return [item['id'] for item in self.config if 'id' in item]
        if isinstance(self.config, dict):
            return self.config.get('series_ids', ['T10Y2Y', 'NFCI', 'CPIAUCSL', 'UNRATE', 'WALCL'])
        return ['T10Y2Y', 'NFCI', 'CPIAUCSL', 'UNRATE', 'WALCL']

    async def run(self):
        """
        The continuous async loop called by the master DataPipeline orchestrator.
        Runs once daily to check for new macroeconomic releases.
        """
        series_list = self._get_series_list()
        
        while True:
            logger.info("Checking for daily FRED macroeconomic updates...")
            today_str = datetime.utcnow().strftime('%Y-%m-%d')
            
            tasks = [self.fetch_and_store_vintage(series, today_str, today_str) for series in series_list]
            await asyncio.gather(*tasks)
            
            logger.info("Macro update complete. Sleeping for 24 hours.")
            await asyncio.sleep(86400) # Sleep for 24 hours

    async def fetch_and_store_vintage(self, series_id: str, start_date: Any, end_date: Any):
        s_str = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date).strip()
        e_str = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date).strip()

        params = {
            "series_id": series_id,
            "api_key": self.api_key.strip(),
            "file_type": "json",
            "observation_start": s_str,
            "observation_end": e_str,
            "realtime_start": s_str, 
            "realtime_end": e_str  # <--- CRITICAL: This must match the chunk end
        }
        # ... rest of the method remains the same ...

        async with self.rate_limit_semaphore:
            async with aiohttp.ClientSession() as session:
                try:
                    # Debug: Print the URL once to see exactly what is being sent
                    # logger.info(f"Requesting FRED: {self.FRED_OBS_URL}?series_id={series_id}&observation_start={s_str}")
                    
                    async with session.get(self.FRED_OBS_URL, params=params) as response:
                        if response.status != 200:
                            # Capture the error message from FRED to see WHY it's 400
                            err_body = await response.text()
                            logger.error(f"FRED API Error {response.status} for {series_id}: {err_body}")
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
                
            try:
                metric_date = datetime.strptime(obs['date'], '%Y-%m-%d').date()
                # The realtime_start from ALFRED is the exact date this specific value became public
                as_of_date = datetime.strptime(obs['realtime_start'], '%Y-%m-%d').date()
                value = float(obs['value'])
                records.append((metric_date, as_of_date, series_id, value))
            except (ValueError, KeyError):
                continue

        if not records:
            return

        async with self.db_pool.acquire() as conn:
            async with conn.transaction():
                await conn.executemany(query, records)
                
        logger.debug(f"Inserted {len(records)} vintage records for {series_id}.")

    async def seed_historical_data(self, start_date: Any = "2011-03-12"):
        series_list = self._get_series_list()
        current_start = datetime.strptime(start_date, '%Y-%m-%d') if isinstance(start_date, str) else start_date
        final_end = datetime.utcnow()

        for series_id in series_list:
            chunk_start = current_start
            while chunk_start < final_end:
                # 3-year chunks (1095 days) to stay well under the 2000-vintage limit
                chunk_end = min(chunk_start + timedelta(days=3*365), final_end)
                
                logger.info(f" -> Fetching {series_id} chunk: {chunk_start.date()} to {chunk_end.date()}")
                await self.fetch_and_store_vintage(series_id, chunk_start, chunk_end)
                
                chunk_start = chunk_end + timedelta(days=1)
            
        logger.info("Historical macro seeding complete.")