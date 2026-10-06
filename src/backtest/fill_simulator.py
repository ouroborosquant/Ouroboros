"""Simulador de ejecución intrabarra: fills conservadores, fricción contabilizada una sola vez y gestión dinámica."""

import math
from dataclasses import dataclass, field
from datetime import datetime

from src.core.constants import Direction
from src.core.events import BarEvent


@dataclass(frozen=True)
class ExecutionConfig:
    commission_per_side: float = 0.62
    entry_slip_ticks: float = 1.0          # entrada a mercado
    stop_slip_ticks: float = 1.0           # el stop es una orden a mercado
    flatten_slip_ticks: float = 1.0
    limit_fill_slip_ticks: float = 0.0     # límites (TP, parciales, entrada límite) llenan al precio
    limit_trade_through_ticks: float = 1.0  # un límite sólo llena si el precio lo atraviesa N ticks


@dataclass(frozen=True)
class ManagementConfig:
    scale_out_enabled: bool = True
    scale_out_rr: float = 1.0
    scale_out_fraction: float = 0.5
    breakeven_enabled: bool = True
    breakeven_rr: float = 1.2
    trail_enabled: bool = True
    trail_trigger_rr: float = 1.8
    trail_atr_multiplier: float = 2.0
    trail_atr_time_scale: float = math.sqrt(5.0)  # ATR 1m -> ATR 5m equivalente (difusión browniana)
    time_stop_bars: int | None = 45
    time_stop_min_mfe_rr: float = 0.5


@dataclass
class PendingEntry:
    symbol: str
    direction: Direction
    contracts: int
    limit_price: float
    stop_loss: float
    take_profit: float
    ttl_bars: int
    created_at: datetime
    bars_waiting: int = 0


@dataclass
class ActivePosition:
    entry_time: datetime
    symbol: str
    direction: Direction
    contracts: int
    entry_price: float              # precio ideal de la señal (referencia para slippage)
    stop_loss: float                # stop vigente (se endurece con BE / trailing)
    take_profit: float
    entry_fill: float | None = None
    partial_level: float | None = None
    partial_qty: int = 0
    commission: float = 0.0
    mfe_points: float = 0.0
    mae_points: float = 0.0
    bars_held: int = 0
    scaled_out: bool = False
    breakeven_set: bool = False
    trailing_active: bool = False
    ideal_gross: float = 0.0
    actual_gross: float = 0.0
    exit_notional: float = 0.0
    initial_stop: float = field(init=False)
    remaining: int = field(init=False)
    risk_points: float = field(init=False)
    extreme_price: float = field(init=False)

    def __post_init__(self) -> None:
        if self.entry_fill is None:
            self.entry_fill = self.entry_price
        self.initial_stop = self.stop_loss
        self.remaining = self.contracts
        self.risk_points = abs(self.entry_fill - self.stop_loss)
        self.extreme_price = self.entry_fill


@dataclass
class ClosedTrade:
    entry_time: datetime
    exit_time: datetime
    symbol: str
    direction: Direction
    contracts: int
    entry_price: float
    exit_price: float
    gross_pnl: float        # PnL ideal (sin slippage ni comisiones)
    net_pnl: float          # gross - slippage - commission
    commission: float
    slippage: float
    exit_reason: str
    r_multiple: float = 0.0
    mfe_r: float = 0.0
    mae_r: float = 0.0
    bars_held: int = 0
    ambiguous: bool = False
    partial_exit: bool = False


class FillSimulator:
    """
    Convenciones (conservadoras):
    - Intrabarra el adverso se evalúa antes que el favorable: stop gana a TP/parcial en la misma barra.
    - Gap a través del stop llena en la apertura. Los límites exigen trade-through.
    - BE / trailing / parciales se calculan al cierre de la barra y rigen desde la siguiente.
    - La fricción se mide una sola vez: precios deslizados + comisiones; `slippage` = ideal - deslizado.
    """

    def __init__(
        self,
        point_value: float,
        tick_size: float,
        tick_value: float,
        execution: ExecutionConfig | None = None,
        management: ManagementConfig | None = None,
    ) -> None:
        self.point_value = point_value
        self.tick_size = tick_size
        self.tick_value = tick_value
        self.execution = execution or ExecutionConfig()
        self.management = management or ManagementConfig()
        cfg = self.execution
        # BE "con costes": cubre comisión ida y vuelta y el slippage del stop
        self._be_offset = (2.0 * cfg.commission_per_side / tick_value + cfg.stop_slip_ticks) * tick_size

    def _snap(self, price: float) -> float:
        return round(round(price / self.tick_size) * self.tick_size, 4)

    def round_trip_cost_ticks(self, order_type: str = "MARKET") -> float:
        cfg = self.execution
        entry_slip = cfg.entry_slip_ticks if order_type == "MARKET" else cfg.limit_fill_slip_ticks
        return 2.0 * cfg.commission_per_side / self.tick_value + entry_slip + cfg.stop_slip_ticks

    def open_position(
        self,
        entry_time: datetime,
        symbol: str,
        direction: Direction,
        contracts: int,
        signal_entry: float,
        stop_loss: float,
        take_profit: float,
        order_type: str = "MARKET",
    ) -> ActivePosition:
        d = direction.value
        slip = self.execution.entry_slip_ticks if order_type == "MARKET" else self.execution.limit_fill_slip_ticks
        fill = signal_entry + d * slip * self.tick_size
        pos = ActivePosition(
            entry_time=entry_time,
            symbol=symbol,
            direction=direction,
            contracts=contracts,
            entry_price=signal_entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_fill=fill,
            commission=contracts * self.execution.commission_per_side,
        )
        m = self.management
        if m.scale_out_enabled and contracts >= 2:
            pos.partial_qty = min(contracts - 1, max(1, int(contracts * m.scale_out_fraction)))
            pos.partial_level = self._snap(fill + d * m.scale_out_rr * pos.risk_points)
        return pos

    def evaluate_pending(
        self, order: PendingEntry, bar: BarEvent
    ) -> tuple[ActivePosition | None, str | None]:
        """Devuelve (posición, estado). Estado: FILLED | CANCEL_INVALID | CANCEL_RAN_AWAY | CANCEL_TTL | None."""
        order.bars_waiting += 1
        d = order.direction.value
        tt = self.execution.limit_trade_through_ticks * self.tick_size
        adverse_px = bar.low if d == 1 else bar.high
        favorable_px = bar.high if d == 1 else bar.low

        if (order.limit_price - adverse_px) * d >= tt:
            pos = self.open_position(
                entry_time=bar.timestamp,
                symbol=order.symbol,
                direction=order.direction,
                contracts=order.contracts,
                signal_entry=order.limit_price,
                stop_loss=order.stop_loss,
                take_profit=order.take_profit,
                order_type="LIMIT",
            )
            return pos, "FILLED"

        if (bar.close - order.stop_loss) * d <= 0:
            return None, "CANCEL_INVALID"
        if (favorable_px - order.take_profit) * d >= 0:
            return None, "CANCEL_RAN_AWAY"
        if order.bars_waiting >= order.ttl_bars:
            return None, "CANCEL_TTL"
        return None, None

    def open_pnl_extremes(self, position: ActivePosition, bar: BarEvent) -> tuple[float, float]:
        """(mejor, peor) PnL de la posición en la barra, neto de comisiones y parciales ya realizados."""
        d = position.direction.value
        fill = position.entry_fill or position.entry_price
        adverse_px = bar.low if d == 1 else bar.high
        favorable_px = bar.high if d == 1 else bar.low
        realized = position.actual_gross - position.commission
        rem = position.remaining
        best = realized + rem * (favorable_px - fill) * d * self.point_value
        worst = (
            realized
            + rem * (adverse_px - fill) * d * self.point_value
            - rem * self.execution.commission_per_side
        )
        return best, worst

    def _tighten_stop(self, position: ActivePosition, candidate: float, ref: float) -> bool:
        d = position.direction.value
        # un stop no puede colocarse al otro lado del mercado
        candidate = min(candidate, ref - self.tick_size) if d == 1 else max(candidate, ref + self.tick_size)
        candidate = self._snap(candidate)
        if (candidate - position.stop_loss) * d > 0:
            position.stop_loss = candidate
            return True
        return False

    def _realize(self, position: ActivePosition, qty: int, ideal: float, slip_ticks: float) -> None:
        d = position.direction.value
        fill = ideal - d * slip_ticks * self.tick_size
        entry_fill = position.entry_fill or position.entry_price
        position.ideal_gross += (ideal - position.entry_price) * d * qty * self.point_value
        position.actual_gross += (fill - entry_fill) * d * qty * self.point_value
        position.commission += qty * self.execution.commission_per_side
        position.exit_notional += ideal * qty
        position.remaining -= qty

    def _exit_all(
        self, position: ActivePosition, bar: BarEvent, ideal: float, slip_ticks: float, reason: str
    ) -> ClosedTrade:
        d = position.direction.value
        entry_fill = position.entry_fill or position.entry_price
        self._realize(position, position.remaining, ideal, slip_ticks)
        position.mfe_points = max(position.mfe_points, (ideal - entry_fill) * d)
        position.mae_points = max(position.mae_points, (entry_fill - ideal) * d)

        net = position.actual_gross - position.commission
        slippage = position.ideal_gross - position.actual_gross
        risk_dollars = position.risk_points * self.point_value * position.contracts
        rp = position.risk_points

        return ClosedTrade(
            entry_time=position.entry_time,
            exit_time=bar.timestamp,
            symbol=position.symbol,
            direction=position.direction,
            contracts=position.contracts,
            entry_price=position.entry_price,
            exit_price=round(position.exit_notional / position.contracts, 2),
            gross_pnl=round(position.ideal_gross, 2),
            net_pnl=round(net, 2),
            commission=round(position.commission, 2),
            slippage=round(slippage, 2),
            exit_reason=reason,
            r_multiple=round(net / risk_dollars, 3) if risk_dollars > 0 else 0.0,
            mfe_r=round(position.mfe_points / rp, 3) if rp > 0 else 0.0,
            mae_r=round(position.mae_points / rp, 3) if rp > 0 else 0.0,
            bars_held=position.bars_held,
            ambiguous=reason == "STOP_LOSS_AMBIGUOUS",
            partial_exit=position.scaled_out,
        )

    def evaluate_bar(
        self,
        position: ActivePosition,
        bar: BarEvent,
        atr: float | None = None,
        is_force_flatten: bool = False,
        flatten_reason: str = "SESSION_FLATTEN",
        flatten_price: float | None = None,
        entry_bar: bool = False,
    ) -> ClosedTrade | None:
        """
        Procesa una barra. `entry_bar=True` (barra de relleno de un límite): sólo se evalúa el stop,
        porque el orden intrabarra entre relleno y objetivos es desconocido.
        """
        cfg = self.execution
        m = self.management
        d = position.direction.value
        tick = self.tick_size
        position.bars_held += 1

        if is_force_flatten:
            px = bar.open if flatten_price is None else flatten_price
            return self._exit_all(position, bar, px, cfg.flatten_slip_ticks, flatten_reason)

        tt = cfg.limit_trade_through_ticks * tick
        adverse_px = bar.low if d == 1 else bar.high
        favorable_px = bar.high if d == 1 else bar.low
        entry_fill = position.entry_fill or position.entry_price

        stop = position.stop_loss
        stop_hit = (stop - adverse_px) * d >= 0
        tp_hit = (favorable_px - (position.take_profit + d * tt)) * d >= 0
        partial_hit = (
            position.partial_level is not None
            and not position.scaled_out
            and (favorable_px - (position.partial_level + d * tt)) * d >= 0
        )

        if stop_hit:
            gap = (stop - bar.open) * d >= 0
            ideal = bar.open if gap else stop
            if tp_hit or partial_hit:
                reason = "STOP_LOSS_AMBIGUOUS"
            elif (stop - position.initial_stop) * d > 0:
                reason = "TRAIL_STOP" if position.trailing_active else "BREAKEVEN_STOP"
            else:
                reason = "STOP_LOSS"
            return self._exit_all(position, bar, ideal, cfg.stop_slip_ticks, reason)

        if entry_bar:
            return None

        if partial_hit and position.partial_level is not None:
            self._realize(position, position.partial_qty, position.partial_level, cfg.limit_fill_slip_ticks)
            position.scaled_out = True
            if self._tighten_stop(position, entry_fill + d * self._be_offset, ref=bar.close):
                position.breakeven_set = True

        if tp_hit:
            return self._exit_all(position, bar, position.take_profit, cfg.limit_fill_slip_ticks, "TAKE_PROFIT")

        position.extreme_price = (
            max(position.extreme_price, bar.high) if d == 1 else min(position.extreme_price, bar.low)
        )
        position.mfe_points = max(position.mfe_points, (favorable_px - entry_fill) * d)
        position.mae_points = max(position.mae_points, (entry_fill - adverse_px) * d)
        mfe_rr = position.mfe_points / position.risk_points if position.risk_points > 0 else 0.0

        if m.breakeven_enabled and not position.breakeven_set and mfe_rr >= m.breakeven_rr:
            if self._tighten_stop(position, entry_fill + d * self._be_offset, ref=bar.close):
                position.breakeven_set = True

        if m.trail_enabled and atr is not None and atr > 0 and mfe_rr >= m.trail_trigger_rr:
            position.trailing_active = True
            trail = position.extreme_price - d * m.trail_atr_multiplier * atr * m.trail_atr_time_scale
            self._tighten_stop(position, trail, ref=bar.close)

        if (
            m.time_stop_bars is not None
            and position.bars_held >= m.time_stop_bars
            and mfe_rr < m.time_stop_min_mfe_rr
        ):
            return self._exit_all(position, bar, bar.close, cfg.flatten_slip_ticks, "TIME_STOP")

        return None