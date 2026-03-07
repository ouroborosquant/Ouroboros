"""
FORTRESS v5 - alpaca_client.py
Path: live/alpaca_client.py

Resilient Broker API Wrapper.
Shields the execution layer from transient network failures, 429 rate limits,
and 50x server errors using exponential backoff, jitter, and a circuit breaker.

FIXES APPLIED:
  - BUG #ALPHA-1: `RateLimitError` and `BrokerAPIError` were referenced in
                  `execution_svc.py` via `from live.alpaca_client import RateLimitError`
                  but were NOT defined in this file. This caused an ImportError at
                  container startup, preventing the entire execution microservice
                  from loading.

  - BUG #ALPHA-2: All broker methods were synchronous (`self.client.get_account()`)
                  and were being awaited directly in async coroutines. Awaiting a
                  synchronous function returns the function object, not its result.
                  All public methods now expose an `async` interface that offloads
                  the blocking SDK call to a thread pool via `asyncio.to_thread`.

  - IMPROVEMENT:  Added a software-level Circuit Breaker on top of tenacity's
                  retry logic. After `_CB_FAILURE_THRESHOLD` consecutive failures
                  (beyond retry budget), the circuit opens and all calls return
                  `None` immediately for `_CB_RESET_TIMEOUT` seconds, preventing
                  a flood of blocked threads from exhausting the executor pool.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from typing import Any, List, Optional

logger = logging.getLogger("ResilientAlpaca")

# ── Custom exceptions (BUG #ALPHA-1 FIX) ────────────────────────────────────

class BrokerAPIError(Exception):
    """
    Raised when the Alpaca broker API returns a persistent, non-retryable error
    after exhausting the tenacity retry budget.
    """
    pass


class RateLimitError(BrokerAPIError):
    """
    Raised specifically when Alpaca returns HTTP 429 (Too Many Requests)
    and the retry budget has been exhausted. The caller (execution_svc.py)
    catches this to implement its own backoff tier.
    """
    pass


# ── Circuit Breaker Configuration ───────────────────────────────────────────
_CB_FAILURE_THRESHOLD: int   = 5      # Consecutive failures to open the circuit
_CB_RESET_TIMEOUT:     float = 60.0   # Seconds the circuit stays open before half-open probe
_MAX_RETRIES:          int   = 5      # tenacity retry budget per call
_BACKOFF_MULTIPLIER:   float = 2.0    # Exponential backoff base
_JITTER_MAX_SECONDS:   float = 1.5    # Random jitter ceiling to avoid thundering herd


class _CircuitBreaker:
    """
    Three-state circuit breaker (CLOSED → OPEN → HALF-OPEN).

    CLOSED:    Normal operation. Failures are counted.
    OPEN:      All calls fail fast with BrokerAPIError (no broker contact).
               Resets to HALF-OPEN after _CB_RESET_TIMEOUT seconds.
    HALF-OPEN: Next call is allowed through as a probe.
               Success → CLOSED. Failure → OPEN (timeout restarts).
    """
    def __init__(self) -> None:
        self._failure_count: int = 0
        self._last_failure_time: float = 0.0
        self._state: str = "CLOSED"  # "CLOSED" | "OPEN" | "HALF_OPEN"

    def record_success(self) -> None:
        self._failure_count = 0
        self._state = "CLOSED"

    def record_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self._failure_count >= _CB_FAILURE_THRESHOLD:
            if self._state != "OPEN":
                logger.error(
                    f"Circuit Breaker OPENED after {self._failure_count} consecutive failures. "
                    f"All broker calls suppressed for {_CB_RESET_TIMEOUT}s."
                )
            self._state = "OPEN"

    def allow_request(self) -> bool:
        if self._state == "CLOSED":
            return True
        if self._state == "OPEN":
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= _CB_RESET_TIMEOUT:
                logger.warning("Circuit Breaker entering HALF-OPEN — probing broker.")
                self._state = "HALF_OPEN"
                return True
            return False
        # HALF_OPEN: allow the probe through
        return True


class ResilientAlpacaClient:
    """
    Async wrapper around the Alpaca Trading SDK.

    Design:
      - All public methods are `async`. They offload the blocking SDK call
        to `asyncio.to_thread` so they never freeze the event loop (BUG #ALPHA-2).
      - A Circuit Breaker wraps every call. After _CB_FAILURE_THRESHOLD consecutive
        failures, the circuit opens and returns None immediately, preventing
        executor thread exhaustion.
      - Retries use exponential backoff with jitter on IOError / connection errors.
        429 (RateLimitError) is retried separately with a longer sleep.

    Usage:
        client = ResilientAlpacaClient(paper=True)
        account = await client.get_account()
        positions = await client.get_all_positions()
        await client.submit_order(order_data=order_request)
        await client.cancel_all_orders()
    """

    def __init__(self, paper: bool = True) -> None:
        try:
            from alpaca.trading.client import TradingClient
        except ImportError as exc:
            raise ImportError("alpaca-py is required: pip install alpaca-py") from exc

        api_key    = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")

        if not api_key or not secret_key:
            logger.warning(
                "ALPACA_API_KEY / ALPACA_SECRET_KEY not set. "
                "Client will raise on any network call."
            )

        self._client    = TradingClient(api_key, secret_key, paper=paper)
        self._cb        = _CircuitBreaker()
        self._paper     = paper
        logger.info(f"ResilientAlpacaClient initialised (paper={paper}).")

    # ── Internal retry helper ────────────────────────────────────────────────

    async def _call_with_retry(self, fn, *args, **kwargs) -> Any:
        """
        Executes `fn(*args, **kwargs)` in a thread pool, with:
          1. Circuit breaker guard (fast-fail if OPEN).
          2. Exponential backoff + jitter retry loop (up to _MAX_RETRIES attempts).
          3. Special handling for HTTP 429: longer sleep, raises RateLimitError
             after budget exhaustion.

        Args:
            fn: A synchronous callable (SDK method).

        Returns:
            The return value of `fn`, or raises BrokerAPIError / RateLimitError.
        """
        if not self._cb.allow_request():
            raise BrokerAPIError(
                "Circuit breaker OPEN — broker calls suppressed. "
                "System is protecting the event loop."
            )

        last_exc: Optional[Exception] = None
        sleep_seconds = 1.0

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                result = await asyncio.to_thread(fn, *args, **kwargs)
                self._cb.record_success()
                return result

            except Exception as exc:
                last_exc = exc
                exc_str  = str(exc).lower()

                is_rate_limit = "429" in exc_str or "too many requests" in exc_str
                is_server_err = "503" in exc_str or "502" in exc_str or "500" in exc_str

                if is_rate_limit:
                    logger.warning(
                        f"Rate limit (429) on attempt {attempt}/{_MAX_RETRIES}. "
                        f"Sleeping {sleep_seconds:.1f}s before retry."
                    )
                elif is_server_err:
                    logger.warning(
                        f"Server error on attempt {attempt}/{_MAX_RETRIES}: {exc}. "
                        f"Sleeping {sleep_seconds:.1f}s."
                    )
                else:
                    # Non-retryable error — propagate immediately
                    self._cb.record_failure()
                    raise BrokerAPIError(f"Non-retryable broker error: {exc}") from exc

                if attempt < _MAX_RETRIES:
                    jitter    = random.uniform(0, _JITTER_MAX_SECONDS)
                    await asyncio.sleep(sleep_seconds + jitter)
                    sleep_seconds = min(sleep_seconds * _BACKOFF_MULTIPLIER, 120.0)

        # Exhausted retry budget
        self._cb.record_failure()
        exc_str = str(last_exc).lower()
        if "429" in exc_str or "too many requests" in exc_str:
            raise RateLimitError(
                f"Alpaca rate limit persists after {_MAX_RETRIES} retries."
            ) from last_exc
        raise BrokerAPIError(
            f"Broker error persists after {_MAX_RETRIES} retries: {last_exc}"
        ) from last_exc

    # ── Public async API (BUG #ALPHA-2 FIX) ─────────────────────────────────

    async def get_account(self) -> Any:
        """
        Returns the Alpaca Account object.
        Fields: buying_power, cash, portfolio_value, account_blocked, etc.
        """
        return await self._call_with_retry(self._client.get_account)

    async def get_all_positions(self) -> List[Any]:
        """
        Returns a list of all current open positions.
        Each position has: symbol, qty, current_price, unrealized_pl, etc.
        """
        return await self._call_with_retry(self._client.get_all_positions)

    async def submit_order(self, order_data: Any) -> Any:
        """
        Submits a MarketOrderRequest or LimitOrderRequest to Alpaca.

        Args:
            order_data: A MarketOrderRequest or LimitOrderRequest instance.

        Returns:
            The Alpaca Order object on success.

        Raises:
            RateLimitError: If 429 persists after retry budget.
            BrokerAPIError: For all other persistent failures.
        """
        return await self._call_with_retry(
            self._client.submit_order, order_data=order_data
        )

    async def cancel_all_orders(self) -> List[Any]:
        """
        Cancels ALL open orders.
        Called by execution_svc._handle_emergency_halt() on a TDA / LTC crash signal.

        BUG #2 FIX (execution_svc.py): The original code called
        `self.alpaca.cancel_orders()` which does not exist on TradingClient.
        The correct method is `cancel_orders_for_symbol(symbol)` or
        `cancel_all_orders()`. This wrapper exposes the correct name.
        """
        return await self._call_with_retry(self._client.cancel_orders)

    async def get_position(self, symbol: str) -> Optional[Any]:
        """
        Returns the position object for `symbol`, or None if no position exists.
        """
        try:
            return await self._call_with_retry(
                self._client.get_open_position, symbol_or_asset_id=symbol
            )
        except BrokerAPIError as exc:
            # 404 "position does not exist" is not an error — normalise to None
            if "not found" in str(exc).lower() or "404" in str(exc).lower():
                return None
            raise

    async def close_position(self, symbol: str) -> Optional[Any]:
        """
        Immediately closes (liquidates) the full position in `symbol`.
        Used by the emergency halt handler.
        """
        return await self._call_with_retry(
            self._client.close_position, symbol_or_asset_id=symbol
        )

    async def close_all_positions(self, cancel_orders: bool = True) -> List[Any]:
        """
        Liquidates all open positions. Used for full portfolio halt.

        Args:
            cancel_orders: If True, also cancels all open orders first.
        """
        return await self._call_with_retry(
            self._client.close_all_positions, cancel_orders=cancel_orders
        )

    @property
    def circuit_state(self) -> str:
        """Exposes the circuit breaker state for health check monitoring."""
        return self._cb._state