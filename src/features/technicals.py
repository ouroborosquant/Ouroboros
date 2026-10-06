import math
from dataclasses import dataclass
from datetime import date

from src.core.events import BarEvent
from src.core.time_utils import to_ny_time


@dataclass(frozen=True)
class VWAPSnapshot:
    vwap: float
    std_dev: float
    upper_band_1: float
    lower_band_1: float
    upper_band_2: float
    lower_band_2: float


class RollingATR:
    def __init__(self, period: int = 14) -> None:
        self.period = period
        self.prev_close: float | None = None
        self.atr: float | None = None
        self._tr_history: list[float] = []

    def update(self, high: float, low: float, close: float) -> float:
        if self.prev_close is None:
            tr = high - low
        else:
            tr = max(
                high - low,
                abs(high - self.prev_close),
                abs(low - self.prev_close),
            )

        self.prev_close = close

        if self.atr is None:
            self._tr_history.append(tr)
            if len(self._tr_history) >= self.period:
                self.atr = sum(self._tr_history) / self.period
        else:
            self.atr = (self.atr * (self.period - 1) + tr) / self.period

        return self.atr if self.atr is not None else tr


class IntradayVWAP:
    def __init__(self) -> None:
        self.current_date: date | None = None
        self.cum_volume: float = 0.0
        self.cum_pv: float = 0.0
        self.cum_pv2: float = 0.0

    def reset(self, new_date: date) -> None:
        self.current_date = new_date
        self.cum_volume = 0.0
        self.cum_pv = 0.0
        self.cum_pv2 = 0.0

    def update(self, bar: BarEvent) -> VWAPSnapshot | None:
        bar_date = to_ny_time(bar.timestamp).date()
        if self.current_date != bar_date:
            self.reset(bar_date)

        if bar.volume <= 0:
            return None

        typical_price = (bar.high + bar.low + bar.close) / 3.0

        self.cum_volume += bar.volume
        self.cum_pv += typical_price * bar.volume
        self.cum_pv2 += (typical_price**2) * bar.volume

        vwap = self.cum_pv / self.cum_volume
        variance = max(0.0, (self.cum_pv2 / self.cum_volume) - (vwap**2))
        std_dev = math.sqrt(variance)

        return VWAPSnapshot(
            vwap=round(vwap, 4),
            std_dev=round(std_dev, 4),
            upper_band_1=round(vwap + std_dev, 4),
            lower_band_1=round(vwap - std_dev, 4),
            upper_band_2=round(vwap + 2.0 * std_dev, 4),
            lower_band_2=round(vwap - 2.0 * std_dev, 4),
        )


class RelativeVolumeTracker:
    def __init__(self, default_benchmark_volume: float = 18500.0) -> None:
        self.benchmark_volume = default_benchmark_volume

    def calculate_rvol(self, orb_volume: float) -> float:
        if self.benchmark_volume <= 0:
            return 1.0
        return round(orb_volume / self.benchmark_volume, 2)
