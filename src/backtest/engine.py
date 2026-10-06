"""Motor de backtest bar a bar: drawdown MTM intradía, órdenes límite pendientes y gestión dinámica."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from typing import TYPE_CHECKING, Literal

import pandas as pd

from src.alpha.base_alpha import BaseAlpha
from src.backtest.fee_models import CMEFeeModel
from src.backtest.fill_simulator import (
    ActivePosition,
    ClosedTrade,
    ExecutionConfig,
    FillSimulator,
    ManagementConfig,
    PendingEntry,
)
from src.backtest.metrics import MetricsCalculator, PerformanceMetrics
from src.core.constants import CircuitBreakerStatus, SessionPhase
from src.core.events import BarEvent, SignalEvent
from src.core.time_utils import get_session_phase, to_ny_time
from src.features.technicals import RollingATR
from src.risk.circuit_breakers import CircuitBreaker
from src.risk.drawdown_tracker import DrawdownTracker
from src.risk.position_sizer import InstrumentSpecs, PositionSizer
from src.risk.prop_compliance import PropComplianceValidator

if TYPE_CHECKING:
    from src.models.meta_classifier import MetaClassifier


@dataclass
class BacktestResult:
    metrics: PerformanceMetrics
    trades: list[ClosedTrade]
    equity_curve: pd.Series
    breached: bool = False
    breach_time: datetime | None = None
    diagnostics: dict[str, object] = field(default_factory=dict)


class BacktestEngine:
    """
    Supuestos: `BarEvent.timestamp` = apertura de la barra; la señal se evalúa al cierre y la entrada
    a mercado ocurre a ese cierre (+slippage). Flatten de sesión al open de la primera barra >= 15:50.
    El colchón se mide con equity MTM intrabarra (favorable antes que adverso) y suelo con bloqueo.
    """

    def __init__(
        self,
        strategy: BaseAlpha,
        specs: InstrumentSpecs,
        fee_model: CMEFeeModel,
        initial_capital: float = 50000.0,
        max_trailing_drawdown: float = 2000.0,
        max_daily_loss: float = 1000.0,
        meta_classifier: MetaClassifier | None = None,
        debug: bool = True,
        *,
        management: ManagementConfig | None = None,
        execution: ExecutionConfig | None = None,
        max_cost_ratio: float | None = 0.12,
        drawdown_mode: Literal["peak_unrealized", "closed_balance"] = "peak_unrealized",
        lock_offset: float = 100.0,
        stop_on_breach: bool = True,
    ) -> None:
        self.strategy = strategy
        self.specs = specs
        self.fee_model = fee_model
        self.initial_capital = initial_capital
        self.meta_classifier = meta_classifier
        self.debug = debug
        self.max_cost_ratio = max_cost_ratio
        self.stop_on_breach = stop_on_breach

        execution = execution or ExecutionConfig(
            commission_per_side=fee_model.commission_per_contract,
            entry_slip_ticks=fee_model.slippage_ticks,
            stop_slip_ticks=fee_model.slippage_ticks,
            flatten_slip_ticks=fee_model.slippage_ticks,
        )

        self.circuit_breaker = CircuitBreaker(max_daily_loss=max_daily_loss)
        self.drawdown_tracker = DrawdownTracker(
            initial_balance=initial_capital,
            max_trailing_drawdown=max_trailing_drawdown,
            mode=drawdown_mode,
            lock_offset=lock_offset,
        )
        self.position_sizer = PositionSizer(
            max_trailing_drawdown=max_trailing_drawdown,
            dd_gamma=1.5,
        )
        self.compliance_validator = PropComplianceValidator(
            circuit_breaker=self.circuit_breaker,
            drawdown_tracker=self.drawdown_tracker,
            position_sizer=self.position_sizer,
            specs={specs.symbol: specs},
        )
        self.fill_simulator = FillSimulator(
            point_value=specs.point_value,
            tick_size=specs.tick_size,
            tick_value=specs.tick_size * specs.point_value,
            execution=execution,
            management=management,
        )
        self.atr_tracker = RollingATR(period=14)

        self.active_position: ActivePosition | None = None
        self.pending_entry: PendingEntry | None = None
        self.closed_trades: list[ClosedTrade] = []
        self.equity_history: dict[date, float] = {}
        self.current_balance = initial_capital
        self.current_date: date | None = None
        self.day_closed_pnl = 0.0

        self.signals_seen = 0
        self.signals_passed_meta = 0
        self.rejects: Counter[str] = Counter()
        self.exit_reasons: Counter[str] = Counter()
        self.pending_stats: Counter[str] = Counter()
        self.dll_flattens = 0
        self.peak_dd_pct = 0.0
        self.min_buffer = max_trailing_drawdown
        self.breached = False
        self.breach_time: datetime | None = None
        self._risk_flatten_pending = False

    # ------------------------------------------------------------------ run
    def run(self, bars: list[BarEvent]) -> BacktestResult:
        last_bar: BarEvent | None = None

        for bar in bars:
            bar_date = to_ny_time(bar.timestamp).date()
            if self.current_date != bar_date:
                self._on_new_day(bar_date, last_bar)
            last_bar = bar

            phase = get_session_phase(bar.timestamp)
            atr_1m = self.atr_tracker.update(bar.high, bar.low, bar.close)

            entry_bar = False
            if self.pending_entry is not None:
                if phase != SessionPhase.ACTIVE_TRADING:
                    self.pending_entry = None
                    self.pending_stats["limit_cancelled_phase"] += 1
                else:
                    position, status = self.fill_simulator.evaluate_pending(self.pending_entry, bar)
                    if position is not None:
                        self.active_position = position
                        self.pending_entry = None
                        self.pending_stats["limit_filled"] += 1
                        entry_bar = True
                    elif status is not None:
                        self.pending_entry = None
                        self.pending_stats[f"limit_{status.lower()}"] += 1

            if self.active_position is not None and self._manage_position(bar, phase, atr_1m, entry_bar):
                break

            signal = self.strategy.on_bar(bar)
            if signal is not None:
                self._handle_signal(signal, bar)

        if self.active_position is not None and last_bar is not None:
            trade = self.fill_simulator.evaluate_bar(
                self.active_position,
                last_bar,
                is_force_flatten=True,
                flatten_reason="END_OF_DATA",
                flatten_price=last_bar.close,
            )
            if trade is not None:
                self._book_trade(trade)

        if self.current_date is not None:
            self.equity_history[self.current_date] = self.current_balance

        return self._build_result()

    # ------------------------------------------------------------------ internals
    def _on_new_day(self, bar_date: date, last_bar: BarEvent | None) -> None:
        if self.current_date is not None:
            if self.active_position is not None and last_bar is not None:
                trade = self.fill_simulator.evaluate_bar(
                    self.active_position,
                    last_bar,
                    is_force_flatten=True,
                    flatten_reason="DATA_GAP_FLATTEN",
                    flatten_price=last_bar.close,
                )
                if trade is not None:
                    self._book_trade(trade)
            self.equity_history[self.current_date] = self.current_balance

        self.pending_entry = None
        self.current_date = bar_date
        self.day_closed_pnl = 0.0
        self._risk_flatten_pending = False
        self.circuit_breaker.reset_daily_session()
        self.strategy.reset_session()

    def _book_trade(self, trade: ClosedTrade) -> None:
        self.closed_trades.append(trade)
        self.exit_reasons[trade.exit_reason] += 1
        self.current_balance += trade.net_pnl
        # deshace el PnL MTM escrito por update_intraday_pnl antes de que el breaker sume el cierre
        self.circuit_breaker.daily_pnl = self.day_closed_pnl
        self.day_closed_pnl += trade.net_pnl
        self.circuit_breaker.register_trade_closed(trade.net_pnl)
        self.drawdown_tracker.update(closed_balance=self.current_balance)
        self.active_position = None
        self._risk_flatten_pending = False

    def _manage_position(self, bar: BarEvent, phase: SessionPhase, atr_1m: float, entry_bar: bool) -> bool:
        """Evalúa la posición, actualiza drawdown MTM y breakers. Devuelve True si debe abortarse el run."""
        pos = self.active_position
        if pos is None:
            return False

        balance_before = self.current_balance
        best, worst = self.fill_simulator.open_pnl_extremes(pos, bar)

        session_flatten = phase == SessionPhase.MANDATORY_FLATTEN
        trade = self.fill_simulator.evaluate_bar(
            pos,
            bar,
            atr=atr_1m,
            is_force_flatten=session_flatten or self._risk_flatten_pending,
            flatten_reason="SESSION_FLATTEN" if session_flatten else "RISK_FLATTEN",
            entry_bar=entry_bar,
        )
        if trade is not None:
            best = max(best, trade.net_pnl)
            worst = min(worst, trade.net_pnl)

        state = self.drawdown_tracker.update_intrabar(balance_before, best, worst)
        self.peak_dd_pct = max(self.peak_dd_pct, state.drawdown_pct)
        self.min_buffer = min(self.min_buffer, state.remaining_buffer)

        if trade is None:
            status = self.circuit_breaker.update_intraday_pnl(self.day_closed_pnl + worst)
            if status == CircuitBreakerStatus.EMERGENCY_SHUTDOWN and not self._risk_flatten_pending:
                self._risk_flatten_pending = True
                self.dll_flattens += 1
        else:
            self._book_trade(trade)

        if state.breached and not self.breached:
            self.breached = True
            self.breach_time = bar.timestamp
            if self.active_position is not None:
                forced = self.fill_simulator.evaluate_bar(
                    self.active_position,
                    bar,
                    is_force_flatten=True,
                    flatten_reason="ACCOUNT_BREACH",
                    flatten_price=bar.close,
                )
                if forced is not None:
                    self._book_trade(forced)

        return state.breached and self.stop_on_breach

    def _meta_gate(self, signal: SignalEvent) -> tuple[bool, float]:
        if self.meta_classifier is None:
            return True, 1.0
        atr = signal.metadata.get("atr", 15.0)
        norm_range = (signal.metadata["orb_high"] - signal.metadata["orb_low"]) / (atr * math.sqrt(15))
        features = {
            "rvol_opening_range": signal.metadata.get("rvol", 1.0),
            "range_to_atr_ratio": round(norm_range, 4),
            "distance_to_vwap_zscore": round(abs(signal.entry_price - signal.metadata["vwap"]) / atr, 4),
            "confidence_score": signal.confidence_score,
        }
        return self.meta_classifier.should_execute(features)

    def _handle_signal(self, signal: SignalEvent, bar: BarEvent) -> None:
        self.signals_seen += 1
        if self.active_position is not None or self.pending_entry is not None:
            self.rejects["busy"] += 1
            return

        execute, meta_prob = self._meta_gate(signal)
        if not execute:
            self.rejects["meta"] += 1
            if self.debug and self.rejects["meta"] <= 5:
                print(f"[-] Señal #{self.signals_seen} ({signal.direction.label}) rechazada por Meta-Labeling: P={meta_prob:.3f}")
            return
        self.signals_passed_meta += 1

        order_type = str(signal.metadata.get("order_type", "MARKET"))
        tick = self.specs.tick_size

        stop_ticks = abs(signal.entry_price - signal.stop_loss) / tick
        if self.max_cost_ratio is not None:
            cost_ticks = self.fill_simulator.round_trip_cost_ticks(order_type)
            if stop_ticks <= 0 or cost_ticks / stop_ticks > self.max_cost_ratio:
                self.rejects["cost_gate"] += 1
                return

        valid, reason, contracts = self.compliance_validator.validate_signal(signal, bar.timestamp)
        if not valid or contracts <= 0:
            self.rejects["compliance"] += 1
            if self.debug and self.rejects["compliance"] <= 5:
                print(f"[-] Señal rechazada por PropCompliance: {reason}")
            return

        if order_type == "LIMIT":
            self.pending_entry = PendingEntry(
                symbol=signal.symbol,
                direction=signal.direction,
                contracts=contracts,
                limit_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                ttl_bars=int(signal.metadata.get("ttl_bars", 3)),
                created_at=bar.timestamp,
            )
            self.pending_stats["limit_placed"] += 1
        else:
            self.active_position = self.fill_simulator.open_position(
                entry_time=bar.timestamp,
                symbol=signal.symbol,
                direction=signal.direction,
                contracts=contracts,
                signal_entry=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                order_type="MARKET",
            )
            if self.debug and len(self.closed_trades) < 3:
                print(f"[+] Orden MARKET: {signal.direction.label} {contracts} @ {signal.entry_price} (P_meta={meta_prob:.2f})")

    def _build_result(self) -> BacktestResult:
        equity_series = pd.Series(self.equity_history, dtype=float)
        metrics = MetricsCalculator.calculate(
            trades=[t.__dict__ for t in self.closed_trades],
            daily_equity_series=equity_series,
            initial_capital=self.initial_capital,
        )
        if self.closed_trades:
            # el DD de cierre diario subestima el trailing intradía: se reporta el MTM
            metrics = replace(
                metrics,
                max_drawdown_dollars=round(max(metrics.max_drawdown_dollars, self.drawdown_tracker.peak_drawdown), 2),
                max_drawdown_pct=round(max(metrics.max_drawdown_pct, self.peak_dd_pct), 4),
            )

        placed = self.pending_stats["limit_placed"]
        diagnostics: dict[str, object] = {
            "signals_seen": self.signals_seen,
            "signals_passed_meta": self.signals_passed_meta,
            "rejects": dict(self.rejects),
            "exit_reasons": dict(self.exit_reasons),
            "limit_orders": dict(self.pending_stats),
            "limit_fill_rate": round(self.pending_stats["limit_filled"] / placed, 3) if placed else None,
            "peak_trailing_dd": round(self.drawdown_tracker.peak_drawdown, 2),
            "min_buffer": round(self.min_buffer, 2),
            "hwm": round(self.drawdown_tracker.high_water_mark, 2),
            "floor": round(self.drawdown_tracker.floor, 2),
            "floor_locked": self.drawdown_tracker.locked,
            "dll_flattens": self.dll_flattens,
            "ambiguous_stops": self.exit_reasons["STOP_LOSS_AMBIGUOUS"],
        }
        if self.debug:
            print(f"[*] Diagnóstico: {diagnostics}")

        return BacktestResult(
            metrics=metrics,
            trades=self.closed_trades,
            equity_curve=equity_series,
            breached=self.breached,
            breach_time=self.breach_time,
            diagnostics=diagnostics,
        )