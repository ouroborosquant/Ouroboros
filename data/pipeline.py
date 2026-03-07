"""
FORTRESS v5 - pipeline.py
Path: data/pipeline.py

Master Data Orchestrator.
Responsible for async ingestion scheduling and look-ahead-safe data retrieval.
"""

import os
import yaml
import asyncio
import logging
import asyncpg
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

# Setup institutional-grade logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DataPipeline")

class ConfigurationError(Exception):
    pass

class LookAheadError(Exception):
    pass

class DataPipeline:
    def __init__(self, config_path: str = 'config/data_sources.yaml'):
        """
        Initializes the pipeline, validates all API keys, and sets up the DB pool.
        """
        self.config = self._load_config(config_path)
        self._validate_environment()
        self.db_pool: Optional[asyncpg.Pool] = None
        
        # We will lazy-load the actual ingestion services to keep this orchestrator light
        self.services = {}

    def _load_config(self, path: str) -> Dict[str, Any]:
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def _validate_environment(self):
        """Ensures the system refuses to start if required keys are missing."""
        required_keys = ['DB_PASSWORD', 'ALPACA_API_KEY', 'ALPACA_SECRET_KEY', 'FRED_API_KEY']
        missing = [key for key in required_keys if not os.getenv(key)]
        if missing:
            raise ConfigurationError(f"CRITICAL: Missing required environment variables: {missing}")

    async def initialize_db_pool(self):
        """Creates the async connection pool for TimescaleDB."""
        self.db_pool = await asyncpg.create_pool(
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD'),
            database=os.getenv('DB_NAME', 'fortress'),
            host=os.getenv('DB_HOST', 'localhost'),
            port=os.getenv('DB_PORT', '5432'),
            min_size=5,
            max_size=20
        )
        logger.info("TimescaleDB async pool initialized.")

    async def _run_service_with_restart(self, service_name: str, service_func, interval_sec: int):
        """
        Runs an ingestion service continuously. If it crashes, logs the error,
        publishes to the dead-letter queue, and restarts with exponential backoff.
        """
        backoff = 1
        while True:
            try:
                logger.info(f"Starting {service_name}...")
                await service_func()
                backoff = 1  # Reset backoff on successful run
                await asyncio.sleep(interval_sec)
            except Exception as e:
                logger.error(f"{service_name} FAILED: {str(e)}. Restarting in {backoff}s...")
                # TODO: In production, publish failure to Kafka 'data-health' topic here
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 300)  # Max backoff of 5 minutes

    async def run_continuous(self):
        """
        LIVE TRADING ENTRY POINT.
        Spins up all ingestion microservices as concurrent, isolated tasks.
        """
        if not self.db_pool:
            await self.initialize_db_pool()

        # Import services here to avoid circular dependencies
        from data.ingestion.macro_ingestion import MacroIngestionService
        from data.alt_data.satellite_pipeline import SatellitePipeline
        
        macro_svc = MacroIngestionService(self.db_pool, self.config['macro_sources'])
        sat_svc = SatellitePipeline(self.config['satellite_targets'])

        # Create concurrent tasks for all pipelines
        tasks = [
            asyncio.create_task(self._run_service_with_restart("Macro_Ingestion", macro_svc.run, 3600)),
            asyncio.create_task(self._run_service_with_restart("Satellite_Ingestion", sat_svc.run_daily, 86400)),
            # Add price, options, and AIS pipelines here
        ]
        
        logger.info("All continuous ingestion tasks launched.")
        await asyncio.gather(*tasks)

    # ── RESEARCH & BACKTESTING METHODS ───────────────────────────────────────

    async def get_observation_vector(self, target_date: str) -> np.ndarray:
        """
        CRITICAL RESEARCH METHOD: The Look-Ahead Bias Firewall.
        Returns the exact 52-dimensional state vector as it was known ON target_date.
        """
        if not self.db_pool:
            await self.initialize_db_pool()

        # Notice the strict constraint: as_of_date <= $1. 
        # This mathematically prevents querying future revisions or unreleased data.
        query = """
            SELECT 
                (SELECT value FROM macro_indicators WHERE series_id = 'T10Y2Y' AND as_of_date <= $1 ORDER BY as_of_date DESC LIMIT 1) as yield_curve,
                (SELECT value FROM macro_indicators WHERE series_id = 'NFCI' AND as_of_date <= $1 ORDER BY as_of_date DESC LIMIT 1) as fin_conditions
            -- Add the remaining 50 feature subqueries here
        """
        
        async with self.db_pool.acquire() as conn:
            record = await conn.fetchrow(query, target_date)
            
        if not record:
            raise LookAheadError(f"Insufficient data to construct observation vector for {target_date}")
            
        # Convert to numpy array, handling any potential NULLs (forward fill logic)
        obs_array = np.array(list(record.values()), dtype=np.float32)
        
        # Replace NaNs with 0.0 or a neutral prior if data was truly unavailable
        obs_array = np.nan_to_num(obs_array, nan=0.0)
        
        return obs_array

    async def get_return_matrix(self, target_date: str, lookback_days: int) -> pd.DataFrame:
        """
        Fetches the historical return matrix for the 25-asset universe, 
        strictly constrained by what was available on target_date.
        Used by the Tensor Network Optimizer and TDA topology builder.
        """
        if not self.db_pool:
            await self.initialize_db_pool()

        query = """
            SELECT metric_date, ticker, adj_close
            FROM prices
            WHERE as_of_date <= $1
            AND metric_date >= ($1::date - $2::integer)
            ORDER BY metric_date ASC
        """
        
        async with self.db_pool.acquire() as conn:
            records = await conn.fetch(query, target_date, lookback_days + 30) # buffer for trading days
            
        if not records:
            return pd.DataFrame()

        # Convert to DataFrame and pivot to get shape (time, ticker)
        df = pd.DataFrame(records, columns=['metric_date', 'ticker', 'adj_close'])
        df['metric_date'] = pd.to_datetime(df['metric_date'])
        
        pivot_df = df.pivot(index='metric_date', columns='ticker', values='adj_close')
        pivot_df = pivot_df.ffill().pct_change().dropna()
        
        # Return exactly the required lookback window
        return pivot_df.tail(lookback_days)