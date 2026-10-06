from dataclasses import dataclass
from datetime import datetime

from src.core.constants import Direction
from src.core.events import BarEvent


@dataclass
class ActivePosition:
    entry_time: datetime
    symbol: str
    direction: Direction
    contracts: int
    entry_price: float
    stop_loss: float
    take_profit: float


@dataclass
class ClosedTrade:
    entry_time: datetime
    exit_time: datetime
    symbol: str
    direction: Direction
    contracts: int
    entry_price: float
    exit_price: float
    gross_pnl: float
    net_pnl: float
    commission: float
    slippage: float
    exit_reason: str


class FillSimulator:
    def __init__(self, point_value: float, tick_size: float, tick_value: float) -> None:
        self.point_value = point_value
        self.tick_size = tick_size
        self.tick_value = tick_value

    def evaluate_bar(
        self,
        position: ActivePosition,
        bar: BarEvent,
        commission_per_side: float,
        slippage_ticks: float,
        is_force_flatten: bool = False,
    ) -> ClosedTrade | None:
        exit_price: float | None = None
        exit_reason: str = ""

        if is_force_flatten:
            exit_price = bar.close
            exit_reason = "SESSION_FLATTEN"
        elif position.direction == Direction.LONG:
            hit_sl = bar.low <= position.stop_loss
            hit_tp = bar.high >= position.take_profit

            if hit_sl and hit_tp:
                exit_price = position.stop_loss - (slippage_ticks * self.tick_size)
                exit_reason = "STOP_LOSS_AMBIGUOUS"
            elif hit_sl:
                exit_price = position.stop_loss - (slippage_ticks * self.tick_size)
                exit_reason = "STOP_LOSS"
            elif hit_tp:
                exit_price = position.take_profit
                exit_reason = "TAKE_PROFIT"
        elif position.direction == Direction.SHORT:
            hit_sl = bar.high >= position.stop_loss
            hit_tp = bar.low <= position.take_profit

            if hit_sl and hit_tp:
                exit_price = position.stop_loss + (slippage_ticks * self.tick_size)
                exit_reason = "STOP_LOSS_AMBIGUOUS"
            elif hit_sl:
                exit_price = position.stop_loss + (slippage_ticks * self.tick_size)
                exit_reason = "STOP_LOSS"
            elif hit_tp:
                exit_price = position.take_profit
                exit_reason = "TAKE_PROFIT"

        if exit_price is None:
            return None

        price_diff = (exit_price - position.entry_price) * position.direction.value
        gross_pnl = price_diff * position.contracts * self.point_value
        commissions = 2 * commission_per_side * position.contracts
        slippage_cost = slippage_ticks * self.tick_value * position.contracts
        net_pnl = gross_pnl - commissions - slippage_cost

        return ClosedTrade(
            entry_time=position.entry_time,
            exit_time=bar.timestamp,
            symbol=position.symbol,
            direction=position.direction,
            contracts=position.contracts,
            entry_price=position.entry_price,
            exit_price=round(exit_price, 2),
            gross_pnl=round(gross_pnl, 2),
            net_pnl=round(net_pnl, 2),
            commission=round(commissions, 2),
            slippage=round(slippage_cost, 2),
            exit_reason=exit_reason,
        )
