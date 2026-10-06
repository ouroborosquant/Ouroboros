from dataclasses import dataclass
from datetime import date
from enum import Enum

from src.alpha.base_alpha import BaseAlpha
from src.core.constants import Direction, SessionPhase
from src.core.events import BarEvent, SignalEvent
from src.core.time_utils import get_session_phase, to_ny_time
from src.features.opening_range import OpeningRangeTracker
from src.features.technicals import IntradayVWAP, RelativeVolumeTracker, RollingATR


class RetestState(str, Enum):
    IDLE = "IDLE"
    WAITING_RETEST_LONG = "WAITING_RETEST_LONG"
    WAITING_RETEST_SHORT = "WAITING_RETEST_SHORT"
    COMPLETED = "COMPLETED"


class ORBTrendContinuationStrategy(BaseAlpha):
    def __init__(
        self,
        symbol: str = "MNQ",
        buffer_ticks: int = 2,
        retest_tolerance_ticks: int = 12,
        tick_size: float = 0.25,
        min_rvol_threshold: float = 1.0,
        min_range_atr_ratio: float = 0.60,
        max_range_atr_ratio: float = 2.50,
        stop_atr_multiplier: float = 1.25,
        risk_reward_ratio: float = 2.20,
        target_atr_multiplier: float | None = None,
        hard_stop_max_ticks: int = 100,
    ) -> None:
        self.symbol = symbol
        self.buffer_ticks = buffer_ticks
        self.retest_tolerance = retest_tolerance_ticks * tick_size
        self.tick_size = tick_size
        self.min_rvol_threshold = min_rvol_threshold
        self.min_range_atr_ratio = min_range_atr_ratio
        self.max_range_atr_ratio = max_range_atr_ratio
        self.stop_atr_multiplier = stop_atr_multiplier
        self.risk_reward_ratio = target_atr_multiplier if target_atr_multiplier is not None else risk_reward_ratio
        self.hard_stop_max_ticks = hard_stop_max_ticks

        self.orb_tracker = OpeningRangeTracker(tick_size=self.tick_size)
        self.vwap_tracker = IntradayVWAP()
        self.atr_tracker = RollingATR(period=14)
        self.rvol_tracker = RelativeVolumeTracker()

        self.current_date: date | None = None
        self.state = RetestState.IDLE
        self.retest_pivot_extreme: float = 0.0

    def reset_session(self) -> None:
        self.state = RetestState.IDLE
        self.retest_pivot_extreme = 0.0

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

        if self.state == RetestState.COMPLETED or not self.orb_tracker.is_ready or orb is None:
            return None

        if vwap_data is None or current_atr <= 0:
            return None

        expected_15m_atr = current_atr * (15 ** 0.5)
        range_atr_ratio = orb.range_size / expected_15m_atr
        if not (self.min_range_atr_ratio <= range_atr_ratio <= self.max_range_atr_ratio):
            return None

        rvol = self.rvol_tracker.calculate_rvol(orb.total_volume)
        if rvol < self.min_rvol_threshold:
            return None

        buffer_offset = self.buffer_ticks * self.tick_size

        # FASE 1: Detección de la rotura inicial
        if self.state == RetestState.IDLE:
            if bar.close > (orb.high + buffer_offset) and bar.close > vwap_data.vwap:
                self.state = RetestState.WAITING_RETEST_LONG
                self.retest_pivot_extreme = bar.high
            elif bar.close < (orb.low - buffer_offset) and bar.close < vwap_data.vwap:
                self.state = RetestState.WAITING_RETEST_SHORT
                self.retest_pivot_extreme = bar.low
            return None

        # FASE 2: Retesteo para Largos
        if self.state == RetestState.WAITING_RETEST_LONG:
            self.retest_pivot_extreme = max(self.retest_pivot_extreme, bar.high)
            in_retest_zone = (orb.high - self.retest_tolerance) <= bar.low <= (orb.high + self.retest_tolerance * 2) or abs(bar.low - vwap_data.vwap) <= self.retest_tolerance
            is_bullish_turn = bar.close > bar.open and bar.close > vwap_data.vwap

            if in_retest_zone and is_bullish_turn:
                entry_price = bar.close
                stop_dist = min(
                    current_atr * self.stop_atr_multiplier,
                    self.hard_stop_max_ticks * self.tick_size,
                )
                stop_loss = round(entry_price - stop_dist, 2)
                take_profit = round(entry_price + (stop_dist * self.risk_reward_ratio), 2)
                self.state = RetestState.COMPLETED
                return SignalEvent(
                    timestamp=bar.timestamp,
                    symbol=self.symbol,
                    direction=Direction.LONG,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    confidence_score=min(1.0, 0.5 + (rvol / 10.0)),
                    metadata={
                        "strategy": "ORB_Retest_Continuation_Long",
                        "orb_high": orb.high,
                        "orb_low": orb.low,
                        "rvol": rvol,
                        "atr": current_atr,
                        "vwap": vwap_data.vwap,
                    },
                )

        # FASE 2: Retesteo para Cortos
        if self.state == RetestState.WAITING_RETEST_SHORT:
            self.retest_pivot_extreme = min(self.retest_pivot_extreme, bar.low)
            in_retest_zone = (orb.low - self.retest_tolerance * 2) <= bar.high <= (orb.low + self.retest_tolerance) or abs(bar.high - vwap_data.vwap) <= self.retest_tolerance
            is_bearish_turn = bar.close < bar.open and bar.close < vwap_data.vwap

            if in_retest_zone and is_bearish_turn:
                entry_price = bar.close
                stop_dist = min(
                    current_atr * self.stop_atr_multiplier,
                    self.hard_stop_max_ticks * self.tick_size,
                )
                stop_loss = round(entry_price + stop_dist, 2)
                take_profit = round(entry_price - (stop_dist * self.risk_reward_ratio), 2)
                self.state = RetestState.COMPLETED
                return SignalEvent(
                    timestamp=bar.timestamp,
                    symbol=self.symbol,
                    direction=Direction.SHORT,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    confidence_score=min(1.0, 0.5 + (rvol / 10.0)),
                    metadata={
                        "strategy": "ORB_Retest_Continuation_Short",
                        "orb_high": orb.high,
                        "orb_low": orb.low,
                        "rvol": rvol,
                        "atr": current_atr,
                        "vwap": vwap_data.vwap,
                    },
                )

        return None
