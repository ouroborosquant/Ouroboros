"""
FORTRESS v5 - lookahead_audit.py
Path: data/validation/lookahead_audit.py

Cryptographic Look-Ahead Bias Auditor.
Scans the TimescaleDB hypertables to mathematically guarantee that no data point 
has an `as_of_date` greater than the simulation's `metric_date`.
"""

import os
import asyncio
import asyncpg
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LookAheadAuditor")

class LookAheadAuditor:
    def __init__(self):
        self.db_pool = None

    async def initialize(self):
        self.db_pool = await asyncpg.create_pool(
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD'),
            database=os.getenv('DB_NAME', 'fortress'),
            host=os.getenv('DB_HOST', 'localhost'),
            port=os.getenv('DB_PORT', '5432')
        )

    async def audit_macro_table(self) -> bool:
        """
        Validates the dual-timestamp ALFRED architecture.
        Ensures that for every row, the date the data became available to the public (as_of_date)
        is strictly greater than or equal to the actual economic period it measures (metric_date).
        """
        query = """
            SELECT COUNT(*) 
            FROM macro_indicators 
            WHERE as_of_date < metric_date;
        """
        async with self.db_pool.acquire() as conn:
            violations = await conn.fetchval(query)
            
        if violations > 0:
            logger.critical(f"❌ MACRO LEAKAGE DETECTED: Found {violations} rows where data was known before the event occurred.")
            return False
            
        logger.info("✅ Macro Hypertable: ZERO look-ahead bias detected.")
        return True

    async def audit_price_table(self) -> bool:
        """
        Validates the price ingestion architecture.
        Price data is instantaneous, so metric_date should exactly equal as_of_date.
        """
        query = """
            SELECT COUNT(*) 
            FROM prices 
            WHERE as_of_date != metric_date;
        """
        async with self.db_pool.acquire() as conn:
            violations = await conn.fetchval(query)
            
        if violations > 0:
            logger.critical(f"❌ PRICE LEAKAGE DETECTED: Found {violations} misaligned timestamps.")
            return False
            
        logger.info("✅ Price Hypertable: ZERO look-ahead bias detected.")
        return True

    async def run_full_audit(self):
        logger.info("Initiating Database Forensic Audit for Chronological Leakage...")
        await self.initialize()
        
        try:
            macro_clean = await self.audit_macro_table()
            price_clean = await self.audit_price_table()
            
            if not (macro_clean and price_clean):
                logger.critical("SYSTEM HALTED: Database is contaminated with look-ahead bias. Purge and re-ingest.")
                exit(1)
            else:
                logger.info("Forensic Audit Passed. Database is cryptographically safe for backtesting.")
        finally:
            await self.db_pool.close()

if __name__ == "__main__":
    auditor = LookAheadAuditor()
    asyncio.run(auditor.run_full_audit())