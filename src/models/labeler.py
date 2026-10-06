from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd

from src.core.constants import Direction


@dataclass(frozen=True)
class BarrierLabel:
    entry_time: datetime
    exit_time: datetime
    direction: Direction
    entry_price: float
    exit_price: float
    realized_return: float
    meta_label: int
    barrier_touched: str


class TripleBarrierLabeler:
    def __init__(
        self,
        pt_multiplier: float = 2.50,
        sl_multiplier: float = 1.25,
        max_holding_minutes: int = 120,
    ) -> None:
        self.pt_multiplier = pt_multiplier
        self.sl_multiplier = sl_multiplier
        self.max_holding_delta = timedelta(minutes=max_holding_minutes)

    def label_signal(
        self,
        entry_time: datetime,
        entry_price: float,
        direction: Direction,
        atr: float,
        future_bars: pd.DataFrame,
    ) -> BarrierLabel | None:
        if future_bars.empty or atr <= 0:
            return None

        vertical_barrier = entry_time + self.max_holding_delta

        if direction == Direction.LONG:
            take_profit = entry_price + (self.pt_multiplier * atr)
            stop_loss = entry_price - (self.sl_multiplier * atr)
        else:
            take_profit = entry_price - (self.pt_multiplier * atr)
            stop_loss = entry_price + (self.sl_multiplier * atr)

        for _, bar in future_bars.iterrows():
            bar_time = bar["timestamp"]
            high = bar["high"]
            low = bar["low"]
            close = bar["close"]

            if bar_time >= vertical_barrier:
                ret = ((close - entry_price) / entry_price) * direction.value
                return BarrierLabel(
                    entry_time=entry_time,
                    exit_time=bar_time,
                    direction=direction,
                    entry_price=entry_price,
                    exit_price=close,
                    realized_return=ret,
                    meta_label=1 if ret > 0.002 else 0,
                    barrier_touched="VERTICAL_EXPIRY",
                )

            if direction == Direction.LONG:
                if low <= stop_loss:
                    return BarrierLabel(
                        entry_time=entry_time,
                        exit_time=bar_time,
                        direction=direction,
                        entry_price=entry_price,
                        exit_price=stop_loss,
                        realized_return=(stop_loss - entry_price) / entry_price,
                        meta_label=0,
                        barrier_touched="STOP_LOSS",
                    )
                if high >= take_profit:
                    return BarrierLabel(
                        entry_time=entry_time,
                        exit_time=bar_time,
                        direction=direction,
                        entry_price=entry_price,
                        exit_price=take_profit,
                        realized_return=(take_profit - entry_price) / entry_price,
                        meta_label=1,
                        barrier_touched="TAKE_PROFIT",
                    )

            elif direction == Direction.SHORT:
                if high >= stop_loss:
                    return BarrierLabel(
                        entry_time=entry_time,
                        exit_time=bar_time,
                        direction=direction,
                        entry_price=entry_price,
                        exit_price=stop_loss,
                        realized_return=(entry_price - stop_loss) / entry_price,
                        meta_label=0,
                        barrier_touched="STOP_LOSS",
                    )
                if low <= take_profit:
                    return BarrierLabel(
                        entry_time=entry_time,
                        exit_time=bar_time,
                        direction=direction,
                        entry_price=entry_price,
                        exit_price=take_profit,
                        realized_return=(entry_price - take_profit) / entry_price,
                        meta_label=1,
                        barrier_touched="TAKE_PROFIT",
                    )

        return None
