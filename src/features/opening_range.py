from dataclasses import dataclass
from datetime import date, datetime

from src.core.constants import SessionPhase
from src.core.events import BarEvent
from src.core.time_utils import get_session_phase, to_ny_time


@dataclass(frozen=True)
class OpeningRange:
    trading_date: date
    high: float
    low: float
    midpoint: float
    range_size: float
    total_volume: float
    bar_count: int


class OpeningRangeTracker:
    def __init__(self, tick_size: float = 0.25) -> None:
        self.tick_size = tick_size
        self.current_date: date | None = None
        self._raw_high: float = float("-inf")
        self._raw_low: float = float("inf")
        self._accumulated_volume: float = 0.0
        self._bar_count: int = 0
        self._is_finalized: bool = False
        self._final_range: OpeningRange | None = None

    def reset(self, new_date: date) -> None:
        self.current_date = new_date
        self._raw_high = float("-inf")
        self._raw_low = float("inf")
        self._accumulated_volume = 0.0
        self._bar_count = 0
        self._is_finalized = False
        self._final_range = None

    def update(self, bar: BarEvent) -> OpeningRange | None:
        bar_ny = to_ny_time(bar.timestamp)
        bar_date = bar_ny.date()

        if self.current_date != bar_date:
            self.reset(bar_date)

        phase = get_session_phase(bar.timestamp)

        if phase == SessionPhase.ORB_BUILDING:
            self._raw_high = max(self._raw_high, bar.high)
            self._raw_low = min(self._raw_low, bar.low)
            self._accumulated_volume += bar.volume
            self._bar_count += 1
            return None

        if not self._is_finalized and self._bar_count > 0:
            range_size = round(self._raw_high - self._raw_low, 4)
            midpoint = round((self._raw_high + self._raw_low) / 2.0, 4)

            self._final_range = OpeningRange(
                trading_date=self.current_date,  # type: ignore[arg-type]
                high=self._raw_high,
                low=self._raw_low,
                midpoint=midpoint,
                range_size=range_size,
                total_volume=self._accumulated_volume,
                bar_count=self._bar_count,
            )
            self._is_finalized = True

        return self._final_range

    @property
    def is_ready(self) -> bool:
        return self._is_finalized and self._final_range is not None

    @property
    def current_range(self) -> OpeningRange | None:
        return self._final_range
