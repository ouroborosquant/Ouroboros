"""
FORTRESS v5 - telegram_bot.py
Path: monitoring/telegram_bot.py

Asynchronous Telegram Alert Service.
Pushes real-time execution logs, regime shifts, and critical hardware alerts
to a private Telegram channel.

FIXES APPLIED:
  - BUG #11: Previously opened a new `aiohttp.ClientSession()` per `send_alert` call.
             Under a flash crash, dozens of concurrent alerts each created a new TCP
             session — hammering the Telegram API, triggering its own rate limits,
             and leaking file descriptors on failure.
             The session is now created once during `initialize()` and reused for the
             entire process lifetime. `close()` must be called on shutdown.

ADDITIONAL IMPROVEMENTS:
  - Added an internal asyncio.Queue to serialise alert delivery. This prevents
    concurrent `send_alert` calls from racing each other during high-alert periods
    (e.g., simultaneous LTC urgency + TDA topology alert).
  - Added Telegram rate-limit backoff (HTTP 429 from Telegram's own API).
  - Added `initialize()` / `close()` lifecycle methods for use by FortressOrganism.
"""

import os
import asyncio
import logging
from typing import Optional

try:
    import aiohttp
except ImportError:
    raise ImportError("Requires aiohttp.")

logger = logging.getLogger("TelegramAlerts")

# Telegram's Bot API rate limit is ~30 messages/second per bot.
# We conservatively cap at 1 message per 0.5s to avoid triggering it.
_INTER_MESSAGE_DELAY_SECONDS: float = 0.5
_TELEGRAM_RATE_LIMIT_BACKOFF_SECONDS: float = 10.0


class TelegramAlertService:
    def __init__(
        self,
        token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ):
        """
        Constructs the alert service. Call `await initialize()` before sending alerts.

        Args:
            token:   Telegram Bot API token. Falls back to TELEGRAM_BOT_TOKEN env var.
            chat_id: Target channel/chat ID. Falls back to TELEGRAM_CHAT_ID env var.
        """
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")

        if self.token:
            self.base_url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        else:
            self.base_url = None
            logger.warning(
                "No Telegram token found. TelegramAlertService running in silent mock mode."
            )

        # FIX #11: Session is initialised once, not per-call.
        self._session: Optional[aiohttp.ClientSession] = None

        # Internal queue serialises outbound messages to avoid racing during alert storms.
        self._queue: asyncio.Queue = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None

    async def initialize(self) -> None:
        """
        Must be called once before `send_alert`. Creates the persistent HTTP session
        and starts the background queue worker.

        FIX #11: Session is created here and reused — not opened/closed per message.
        """
        if self.base_url:
            # connector=aiohttp.TCPConnector(limit=5) caps concurrent connections
            # to Telegram, providing a natural rate-limit buffer.
            self._session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=5),
                timeout=aiohttp.ClientTimeout(total=10),
            )
            logger.info("TelegramAlertService HTTP session initialized.")

        # Start the background worker that drains the alert queue.
        self._worker_task = asyncio.create_task(
            self._queue_worker(), name="TelegramQueueWorker"
        )

    async def close(self) -> None:
        """
        Graceful shutdown. Drains remaining queued alerts, cancels the worker,
        and closes the HTTP session.
        Call this from FortressOrganism.shutdown().
        """
        if self._worker_task:
            # Allow the worker to drain in-flight items before cancelling.
            await self._queue.join()
            self._worker_task.cancel()

        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("TelegramAlertService HTTP session closed.")

    async def send_alert(self, message: str, tier: str = "INFO") -> None:
        """
        Enqueues an alert for asynchronous delivery.
        Returns immediately — delivery is handled by the background worker.

        Tiers: INFO (🟢), WARNING (🟡), CRITICAL (🔴)

        Args:
            message: Human-readable alert body.
            tier:    Severity level — "INFO", "WARNING", or "CRITICAL".
        """
        prefix = {"INFO": "🟢", "WARNING": "🟡", "CRITICAL": "🔴"}.get(tier, "⚪")
        formatted_msg = f"{prefix} [{tier}] FORTRESS v5\n\n{message}"

        if not self.base_url or not self.chat_id:
            # Mock mode: log to stdout only.
            logger.info(
                f"MOCK TELEGRAM [{tier}]: {formatted_msg.replace(chr(10), ' | ')}"
            )
            return

        await self._queue.put((formatted_msg, tier))

    async def _queue_worker(self) -> None:
        """
        Background coroutine that drains the alert queue.
        Serialises delivery and handles Telegram's own rate limits (HTTP 429).
        """
        while True:
            try:
                formatted_msg, tier = await self._queue.get()
                await self._deliver(formatted_msg)
                self._queue.task_done()
                # FIX #11: Brief delay between messages prevents Telegram 429s.
                await asyncio.sleep(_INTER_MESSAGE_DELAY_SECONDS)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"TelegramQueueWorker unhandled exception: {e}")

    async def _deliver(self, formatted_msg: str) -> None:
        """
        Sends a single message to Telegram using the shared persistent session.

        FIX #11: Uses `self._session` (persistent) instead of creating a new
        `aiohttp.ClientSession()` per delivery.
        """
        if not self._session or self._session.closed:
            logger.error(
                "Telegram session is closed. Call initialize() before sending alerts."
            )
            return

        payload = {
            "chat_id": self.chat_id,
            "text": formatted_msg,
            "parse_mode": "HTML",
        }

        try:
            async with self._session.post(self.base_url, json=payload) as response:
                if response.status == 200:
                    return
                elif response.status == 429:
                    # Telegram is rate-limiting us — back off and re-enqueue.
                    retry_after = float(
                        (await response.json()).get("parameters", {}).get(
                            "retry_after", _TELEGRAM_RATE_LIMIT_BACKOFF_SECONDS
                        )
                    )
                    logger.warning(
                        f"Telegram rate limit hit (HTTP 429). "
                        f"Backing off {retry_after}s..."
                    )
                    await asyncio.sleep(retry_after)
                    # Re-enqueue for retry.
                    await self._queue.put((formatted_msg, "RETRY"))
                else:
                    error_text = await response.text()
                    logger.error(
                        f"Telegram API rejected message (HTTP {response.status}): {error_text}"
                    )
        except aiohttp.ClientConnectionError as e:
            logger.error(f"Telegram connection error: {e}")
        except asyncio.TimeoutError:
            logger.error("Telegram request timed out.")
        except Exception as e:
            logger.error(f"Unexpected Telegram delivery error: {e}")