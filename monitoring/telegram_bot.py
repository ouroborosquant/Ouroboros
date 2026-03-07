"""
FORTRESS v5 - telegram_bot.py
Path: monitoring/telegram_bot.py

Asynchronous Telegram Alert Service.
Pushes real-time execution logs, regime shifts, and critical hardware alerts to a private channel.
"""

import os
import aiohttp
import logging
from typing import Optional

logger = logging.getLogger("TelegramAlerts")

class TelegramAlertService:
    def __init__(self, token: Optional[str] = None, chat_id: Optional[str] = None):
        """
        Initializes the webhook client. If keys are missing, gracefully defaults to 
        printing to the standard terminal output (mock mode).
        """
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        
        if self.token:
            self.base_url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        else:
            self.base_url = None
            logger.warning("No Telegram credentials found. Running in silent mock mode.")

    async def send_alert(self, message: str, tier: str = "INFO"):
        """
        Fires an async HTTP POST request to the Telegram API.
        Tiers: INFO (🟢), WARNING (🟡), CRITICAL (🔴)
        """
        prefix = "🟢" if tier == "INFO" else "🟡" if tier == "WARNING" else "🔴"
        formatted_msg = f"{prefix} [{tier}] FORTRESS v5\n\n{message}"

        # If no API key, just log it locally
        if not self.base_url or not self.chat_id:
            logger.info(f"MOCK TELEGRAM ALERT: {formatted_msg.replace(chr(10), ' | ')}")
            return

        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "chat_id": self.chat_id,
                    "text": formatted_msg,
                    "parse_mode": "HTML"
                }
                async with session.post(self.base_url, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Telegram API rejected message: {error_text}")
        except Exception as e:
            logger.error(f"Network error sending Telegram alert: {e}")