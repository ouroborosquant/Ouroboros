import math
from datetime import date

from src.alpha.base_alpha import BaseAlpha
from src.core.constants import Direction, SessionPhase
from src.core.events import BarEvent, SignalEvent
from src.core.time_utils import get_session_phase, to_ny_time
from src.features.opening_range import OpeningRangeTracker
from src.features.technicals import IntradayVWAP, RelativeVolumeTracker, RollingATR


class ORBBreakoutStrategy(BaseAlpha):
    def __init__(
        self,
        symbol: str = "MNQ",
        buffer_ticks: int = 2,
        tick_size: float = 0.25,
        min_rvol_threshold: float = 1.0,
        min_range_atr_ratio: float = 0.70,
        max_range_atr_ratio: float = 2.20,
        require_vwap_alignment: bool = True,
        risk_reward_ratio: float = 2.50,
        atr_multiplier: float = 1.25,
        hard_stop_max_ticks: int = 40,
    ) -> None:
        self.symbol = symbol
        self.buffer_ticks = buffer_ticks
        self.tick_size = tick_size
        self.min_rvol_threshold = min_rvol_threshold
        self.min_range_atr_ratio = min_range_atr_ratio
        self.max_range_atr_ratio = max_range_atr_ratio
        self.require_vwap_alignment = require_vwap_alignment
        self.risk_reward_ratio = risk_reward_ratio
        self.atr_multiplier = atr_multiplier
        self.hard_stop_max_ticks = hard_stop_max_ticks

        self.orb_tracker = OpeningRangeTracker(tick_size=self.tick_size)
        self.vwap_tracker = IntradayVWAP()
        self.atr_tracker = RollingATR(period=14)
        self.rvol_tracker = RelativeVolumeTracker()

        self.current_date: date | None = None
        self.has_signaled_today: bool = False

    def reset_session(self) -> None:
        self.has_signaled_today = False

    def on_bar(self, bar: BarEvent) -> SignalEvent | None:
        if bar.symbol != self.symbol:
            return None

        bar_date = to_ny_time(bar.timestamp).date()
        if self.current_date != bar_date:
            self.current_date = bar_date
            self.reset_session()

        orb = self.orb_tracker.update(bar)
        vwap_data = self.vwap_tracker.update(bar)
        current_atr = self.atr_tracker.update(bar.high, bar.low, bar.close)

        phase = get_session_phase(bar.timestamp)
        if phase != SessionPhase.ACTIVE_TRADING:
            return None

        if self.has_signaled_today or not self.orb_tracker.is_ready or orb is None:
            return None

        if vwap_data is None or current_atr <= 0:
            return None

        # Normalización temporal sqrt(15) para comparar rango de 15m con ATR de 1m
        expected_15m_atr = current_atr * math.sqrt(15)
        range_atr_ratio = orb.range_size / expected_15m_atr
        if not (self.min_range_atr_ratio <= range_atr_ratio <= self.max_range_atr_ratio):
            return None

        rvol = self.rvol_tracker.calculate_rvol(orb.total_volume)
        if rvol < self.min_rvol_threshold:
            return None

        buffer_offset = self.buffer_ticks * self.tick_size
        breakout_long = bar.close > (orb.high + buffer_offset)
        breakout_short = bar.close < (orb.low - buffer_offset)

        if breakout_long:
            if self.require_vwap_alignment and bar.close <= vwap_data.vwap:
                return None

            entry_price = bar.close
            stop_distance = min(
                current_atr * self.atr_multiplier,
                self.hard_stop_max_ticks * self.tick_size,
            )
            stop_loss = round(entry_price - stop_distance, 2)
            take_profit = round(entry_price + (stop_distance * self.risk_reward_ratio), 2)

            self.has_signaled_today = True
            return SignalEvent(
                timestamp=bar.timestamp,
                symbol=self.symbol,
                direction=Direction.LONG,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence_score=min(1.0, 0.5 + (rvol / 10.0)),
                metadata={
                    "strategy": "ORB_Breakout_Long",
                    "orb_high": orb.high,
                    "orb_low": orb.low,
                    "rvol": rvol,
                    "atr": current_atr,
                    "vwap": vwap_data.vwap,
                },
            )

        if breakout_short:
            if self.require_vwap_alignment and bar.close >= vwap_data.vwap:
                return None

            entry_price = bar.close
            stop_distance = min(
                current_atr * self.atr_multiplier,
                self.hard_stop_max_ticks * self.tick_size,
            )
            stop_loss = round(entry_price + stop_distance, 2)
            take_profit = round(entry_price - (stop_distance * self.risk_reward_ratio), 2)

            self.has_signaled_today = True
            return SignalEvent(
                timestamp=bar.timestamp,
                symbol=self.symbol,
                direction=Direction.SHORT,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence_score=min(1.0, 0.5 + (rvol / 10.0)),
                metadata={
                    "strategy": "ORB_Breakout_Short",
                    "orb_high": orb.high,
                    "orb_low": orb.low,
                    "rvol": rvol,
                    "atr": current_atr,
                    "vwap": vwap_data.vwap,
                },
            )

        return None
