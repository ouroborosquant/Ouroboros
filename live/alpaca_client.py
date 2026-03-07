"""
FORTRESS v5 - alpaca_client.py
Path: live/alpaca_client.py

Resilient Broker API Wrapper.
Shields the execution layer from transient network failures, 429 rate limits, 
and 50x server errors using exponential backoff and jitter.
"""

import os
import time
import logging
from typing import List, Any, Optional

# External Dependencies
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    import requests
except ImportError:
    raise ImportError("Requires alpaca-py and tenacity.")

logger = logging.getLogger("ResilientAlpaca")

class BrokerAPIError(Exception):
    """Custom exception for persistent broker failures."""
    pass

class ResilientAlpacaClient:
    def __init__(self, paper: bool = True):
        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        
        if not api_key or not secret_key:
            logger.warning("Alpaca keys missing. Client will fail on network calls.")
            
        self.client = TradingClient(api_key, secret_key, paper=paper)
        logger.info(f"Initialized Resilient Alpaca Wrapper (Paper={paper}).")

    # The @retry decorator automatically handles transient HTTP errors
    # It waits 2^x * 1 second between each retry, up to 5 attempts.
    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type((requests.exceptions.ConnectionError, requests.exceptions.Timeout)),
        reraise=True
    )
    def submit_order(self, order_request: Any) -> Any:
        """Submits an order with exponential backoff for network timeouts."""
        try:
            return self.client.submit_order(order_data=order_request)
        except Exception as e:
            error_msg = str(e).lower()
            if "insufficient buying power" in error_msg:
                logger.error(f"Order rejected: Insufficient Buying Power. {e}")
                raise BrokerAPIError("INSUFFICIENT_FUNDS")
            elif "rate limit" in error_msg or "429" in error_msg:
                logger.warning("Alpaca Rate Limit hit. Forcing strict backoff.")
                time.sleep(2) # Hard pause to let the bucket refill
                raise requests.exceptions.ConnectionError("Rate Limit")
            else:
                logger.error(f"Alpaca API Exception: {e}")
                raise BrokerAPIError(f"API_ERROR: {e}")

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=5),
        stop=stop_after_attempt(3),
        reraise=True
    )
    def cancel_all_orders(self) -> List[Any]:
        """Emergency method: Cancels all open orders."""
        logger.info("Issuing global cancel_orders command to broker...")
        return self.client.cancel_orders()

    @retry(wait=wait_exponential(min=1, max=3), stop=stop_after_attempt(3))
    def get_all_positions(self) -> List[Any]:
        """Fetches current portfolio positions safely."""
        return self.client.get_all_positions()

    @retry(wait=wait_exponential(min=1, max=3), stop=stop_after_attempt(3))
    def get_account_value(self) -> float:
        """Safely retrieves the total net liquidation value of the account."""
        account = self.client.get_account()
        return float(account.portfolio_value)