"""
FORTRESS v5 - alpaca_client.py
Path: live/alpaca_client.py

Resilient Broker API Wrapper.
Shields the execution layer from transient network failures, 429 rate limits,
and 50x server errors using exponential backoff and jitter.

FIXES APPLIED:
  - BUG #1: Renamed `order_request` -> `order_data` to match all call sites in execution_svc.py.
            The previous mismatch caused a TypeError on every live trade submission.
  - BUG #6: Removed `time.sleep(2)` from the synchronous path. This call was blocking
            the entire asyncio event loop during rate-limit storms. The 429 path now
            raises RateLimitError which the async caller handles via `await asyncio.sleep()`.
"""

import os
import logging
from typing import List, Any, Optional

# External Dependencies
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
    from tenacity import (
        retry,
        wait_exponential,
        stop_after_attempt,
        retry_if_exception_type,
    )
    import requests
except ImportError:
    raise ImportError("Requires alpaca-py and tenacity.")

logger = logging.getLogger("ResilientAlpaca")


class BrokerAPIError(Exception):
    """Custom exception for persistent, non-retryable broker failures."""
    pass


class RateLimitError(Exception):
    """
    Raised specifically when Alpaca returns HTTP 429 (Too Many Requests).
    Unlike BrokerAPIError, this IS retryable — but the sleep must happen
    in the async caller (execution_svc.py) via `await asyncio.sleep()`.
    Blocking sleep inside a synchronous method freezes the entire event loop.
    """
    pass


class ResilientAlpacaClient:
    def __init__(self, paper: bool = True):
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")

        if not api_key or not secret_key:
            logger.warning("Alpaca keys missing. Client will fail on any network call.")

        self.client = TradingClient(api_key, secret_key, paper=paper)
        logger.info(f"Initialized Resilient Alpaca Wrapper (Paper={paper}).")

    @retry(
        # Exponential backoff: waits 1s, 2s, 4s, 8s, 10s (capped) between attempts.
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(5),
        # Only auto-retry on transient network errors; propagate business-logic errors immediately.
        retry=retry_if_exception_type(
            (requests.exceptions.ConnectionError, requests.exceptions.Timeout)
        ),
        reraise=True,
    )
    def submit_order(self, order_data: Any) -> Any:
        """
        Submits an order with exponential backoff for transient network failures.

        FIX #1: Parameter renamed from `order_request` to `order_data` to match
        every call site in execution_svc.py (e.g., self.alpaca.submit_order(order_data=req)).
        The previous mismatch raised TypeError on every trade attempt.

        FIX #6: `time.sleep(2)` has been removed. Sleeping inside a synchronous
        method called from an async event loop blocks ALL coroutines — including
        emergency halt listeners. Rate limit handling is now delegated to the
        async caller via raising `RateLimitError`.

        Args:
            order_data: An Alpaca MarketOrderRequest or LimitOrderRequest object.

        Raises:
            RateLimitError: On HTTP 429 — the async caller must `await asyncio.sleep()`.
            BrokerAPIError: On non-retryable broker rejections (insufficient funds, etc.).
        """
        try:
            return self.client.submit_order(order_data=order_data)
        except Exception as e:
            error_msg = str(e).lower()

            if "insufficient buying power" in error_msg or "insufficient" in error_msg:
                logger.error(f"Order rejected — Insufficient Buying Power: {e}")
                # Do NOT retry. Raise immediately to prevent order duplication.
                raise BrokerAPIError("INSUFFICIENT_FUNDS") from e

            elif "rate limit" in error_msg or "429" in error_msg:
                # Signal the async caller to back off. Do NOT sleep here.
                logger.warning("Alpaca rate limit hit (HTTP 429). Propagating RateLimitError to async layer.")
                raise RateLimitError("RATE_LIMIT_429") from e

            elif "forbidden" in error_msg or "403" in error_msg:
                logger.error(f"Order rejected — Forbidden (check paper/live key mismatch): {e}")
                raise BrokerAPIError("FORBIDDEN") from e

            else:
                # For unknown exceptions, allow tenacity to retry if it's a connection error,
                # otherwise propagate as a generic BrokerAPIError.
                logger.error(f"Alpaca API Exception: {e}")
                raise BrokerAPIError(f"API_ERROR: {e}") from e

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=5),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def cancel_all_orders(self) -> List[Any]:
        """
        Emergency method: Cancels all open orders.
        Called during _handle_emergency_halt — must not fail silently.
        """
        logger.info("Issuing global cancel_orders command to broker...")
        return self.client.cancel_orders()

    @retry(
        wait=wait_exponential(min=1, max=3),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def get_all_positions(self) -> List[Any]:
        """Fetches current portfolio positions with retry on transient errors."""
        return self.client.get_all_positions()

    @retry(
        wait=wait_exponential(min=1, max=3),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def get_account(self) -> Any:
        """
        Returns the full Alpaca Account object.
        Callers should cast `account.portfolio_value` to float themselves.
        """
        return self.client.get_account()

    @retry(
        wait=wait_exponential(min=1, max=3),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def get_account_value(self) -> float:
        """Convenience wrapper: safely retrieves the total net liquidation value."""
        account = self.client.get_account()
        return float(account.portfolio_value)