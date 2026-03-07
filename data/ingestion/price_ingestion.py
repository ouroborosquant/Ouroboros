"""
FORTRESS v5 - price_ingestion.py
Path: data/ingestion/price_ingestion.py

Live Microstructure Ingestion Pipeline.
Consumes real-time WebSockets from Alpaca, computes Order Book Imbalance (OBI),
and streams the 15-dim intraday feature vector to the Kafka event bus for the LTC network.
"""

import os
import json
import logging
import pandas as pd
import asyncio
import numpy as np
from typing import Dict, Any

# External dependencies
try:
    from alpaca.data.live import StockDataStream
    from alpaca.data.models import Trade, Quote
    from aiokafka import AIOKafkaProducer
    import redis.asyncio as redis
except ImportError:
    raise ImportError("Requires alpaca-py, aiokafka, and redis.")

logger = logging.getLogger("PriceIngestion")

class RealTimeMicrostructure:
    def __init__(self, config: Dict[str, Any]):
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.secret_key = os.getenv('ALPACA_SECRET_KEY')
        
        # Load the universe from config
        self.universe = config.get('universe', ['SPY', 'QQQ', 'TLT', 'GLD', 'VIXY'])
        
        # In-memory state for computing high-frequency derivatives
        self.state = {ticker: {'last_price': 0.0, 'bid': 0.0, 'ask': 0.0, 'bid_size': 0.0, 'ask_size': 0.0, 'vol_1m': 0.0} for ticker in self.universe}
        
        self.kafka_producer = None
        self.redis_client = None
        self.stream = StockDataStream(self.api_key, self.secret_key)

    async def setup_infrastructure(self):
        kafka_url = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        self.kafka_producer = AIOKafkaProducer(bootstrap_servers=kafka_url)
        await self.kafka_producer.start()
        
        self.redis_client = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
        logger.info("Kafka and Redis connected for Price Ingestion.")

    async def trade_callback(self, trade: Trade):
        """Processes executed trades to calculate volume pace and momentum."""
        ticker = trade.symbol
        if ticker in self.state:
            self.state[ticker]['last_price'] = float(trade.price)
            self.state[ticker]['vol_1m'] += float(trade.size)
            
            # Periodically emit the feature vector to Kafka for the LTC monitor
            await self._emit_tick_features(ticker)

    async def quote_callback(self, quote: Quote):
        """Processes L1/L2 quotes to calculate the Order Book Imbalance (OBI)."""
        ticker = quote.symbol
        if ticker in self.state:
            self.state[ticker]['bid'] = float(quote.bid_price)
            self.state[ticker]['ask'] = float(quote.ask_price)
            self.state[ticker]['bid_size'] = float(quote.bid_size)
            self.state[ticker]['ask_size'] = float(quote.ask_size)

    async def _emit_tick_features(self, ticker: str):
        """
        Constructs the 15-dimensional intraday feature vector required by the
        Liquid Time-Constant (LTC) network.
        """
        s = self.state[ticker]
        
        # 1. Micro-Price (Volume weighted midpoint)
        total_size = s['bid_size'] + s['ask_size']
        if total_size > 0:
            micro_price = (s['bid'] * s['ask_size'] + s['ask'] * s['bid_size']) / total_size
        else:
            micro_price = s['last_price']
            
        # 2. Order Book Imbalance [-1.0 to 1.0]
        # +1.0 means heavy bid pressure (bullish), -1.0 means heavy ask pressure (bearish)
        obi = (s['bid_size'] - s['ask_size']) / total_size if total_size > 0 else 0.0
        
        # 3. Spread (Liquidity proxy)
        spread = s['ask'] - s['bid']
        
        # 4. Construct the dummy 15-dim vector (expanded for the full LTC input)
        # In production, this includes cross-asset correlations, tick-momentum, etc.
        features = np.zeros(15, dtype=np.float32)
        features[0] = obi
        features[1] = spread
        features[2] = micro_price
        features[3] = s['vol_1m']
        # ... (fill remaining features)
        
        payload = {
            "ticker": ticker,
            "timestamp": pd.Timestamp.utcnow().timestamp(),
            "features": features.tolist()
        }
        
        # Send to the Regime Encoder microservice for continuous-time evaluation
        await self.kafka_producer.send_and_wait('market-data-ticks', json.dumps(payload).encode('utf-8'))
        
        # Update Redis for the Execution Router to access instantly
        await self.redis_client.hset(f"l2_state:{ticker}", mapping={
            "obi": obi,
            "spread": spread,
            "micro_price": micro_price
        })

    async def run(self):
        await self.setup_infrastructure()
        
        # Subscribe the callbacks to the Alpaca WebSocket
        self.stream.subscribe_trades(self.trade_callback, *self.universe)
        self.stream.subscribe_quotes(self.quote_callback, *self.universe)
        
        logger.info(f"Subscribed to real-time WebSockets for {len(self.universe)} assets.")
        
        # Run the WebSocket loop (this blocks forever)
        await self.stream._run_forever()

if __name__ == "__main__":
    import yaml
    with open("config/universe.yaml", "r") as f:
        univ = yaml.safe_load(f)
        tickers = [a['ticker'] for a in univ.get('assets', [])]
        
    ingester = RealTimeMicrostructure({'universe': tickers})
    asyncio.run(ingester.run())