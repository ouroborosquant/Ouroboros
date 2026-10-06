from dataclasses import dataclass
from datetime import date
import math
import numpy as np
import pandas as pd

from src.alpha.orb_breakout import ORBBreakoutStrategy
from src.backtest.fee_models import CMEFeeModel
from src.backtest.fill_simulator import ActivePosition, ClosedTrade, FillSimulator
from src.backtest.metrics import MetricsCalculator, PerformanceMetrics
from src.core.constants import SessionPhase
from src.core.events import BarEvent
from src.core.time_utils import get_session_phase, to_ny_time
from src.models.meta_classifier import MetaClassifier
from src.risk.circuit_breakers import CircuitBreaker
from src.risk.drawdown_tracker import DrawdownTracker
from src.risk.position_sizer import InstrumentSpecs, PositionSizer
from src.risk.prop_compliance import PropComplianceValidator


@dataclass
class BacktestResult:
    metrics: PerformanceMetrics
    trades: list[ClosedTrade]
    equity_curve: pd.Series


class BacktestEngine:
    def __init__(
        self,
        strategy: ORBBreakoutStrategy,
        specs: InstrumentSpecs,
        fee_model: CMEFeeModel,
        initial_capital: float = 50000.0,
        max_trailing_drawdown: float = 2000.0,
        max_daily_loss: float = 1000.0,
        meta_classifier: MetaClassifier | None = None,
        debug: bool = True,
    ) -> None:
        self.strategy = strategy
        self.specs = specs
        self.fee_model = fee_model
        self.initial_capital = initial_capital
        self.meta_classifier = meta_classifier
        self.debug = debug

        self.circuit_breaker = CircuitBreaker(max_daily_loss=max_daily_loss)
        self.drawdown_tracker = DrawdownTracker(
            initial_balance=initial_capital,
            max_trailing_drawdown=max_trailing_drawdown,
            mode="peak_unrealized",
        )
        self.position_sizer = PositionSizer()
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
        )

        self.active_position: ActivePosition | None = None
        self.closed_trades: list[ClosedTrade] = []
        self.equity_history: dict[date, float] = {}
        self.current_balance = initial_capital
        self.current_date: date | None = None
        self.signals_seen = 0
        self.signals_passed_meta = 0

    def run(self, bars: list[BarEvent]) -> BacktestResult:
        daily_closed_balance = self.initial_capital

        for bar in bars:
            bar_date = to_ny_time(bar.timestamp).date()

            if self.current_date != bar_date:
                if self.current_date is not None:
                    self.equity_history[self.current_date] = daily_closed_balance
                self.current_date = bar_date
                self.circuit_breaker.reset_daily_session()
                self.strategy.reset_session()

            phase = get_session_phase(bar.timestamp)
            is_flatten_window = phase == SessionPhase.MANDATORY_FLATTEN

            # 1. Gestionar posición activa si existe
            if self.active_position is not None:
                trade_result = self.fill_simulator.evaluate_bar(
                    position=self.active_position,
                    bar=bar,
                    commission_per_side=self.fee_model.commission_per_contract,
                    slippage_ticks=self.fee_model.slippage_ticks,
                    is_force_flatten=is_flatten_window,
                )
                if trade_result is not None:
                    self.closed_trades.append(trade_result)
                    self.current_balance += trade_result.net_pnl
                    daily_closed_balance = self.current_balance
                    self.circuit_breaker.register_trade_closed(trade_result.net_pnl)
                    self.drawdown_tracker.update(closed_balance=self.current_balance)
                    self.active_position = None

            # 2. Alimentar SIEMPRE la barra a la estrategia para actualizar indicadores
            signal = self.strategy.on_bar(bar)

            # 3. Si se genera señal, procesarla
            if signal is not None:
                self.signals_seen += 1
                execute_signal = True
                meta_prob = 1.0

                if self.meta_classifier is not None:
                    current_atr = signal.metadata.get("atr", 15.0)
                    norm_range = (signal.metadata["orb_high"] - signal.metadata["orb_low"]) / (current_atr * math.sqrt(15))
                    features = {
                        "rvol_opening_range": signal.metadata.get("rvol", 1.0),
                        "range_to_atr_ratio": round(norm_range, 4),
                        "distance_to_vwap_zscore": round(abs(signal.entry_price - signal.metadata["vwap"]) / current_atr, 4),
                        "confidence_score": signal.confidence_score,
                    }
                    execute_signal, meta_prob = self.meta_classifier.should_execute(features)

                if not execute_signal:
                    if self.debug and self.signals_seen <= 5:
                        print(f"[-] Señal #{self.signals_seen} ({signal.direction.label}) rechazada por Meta-Labeling: P={meta_prob:.3f} < cutoff")
                    continue

                self.signals_passed_meta += 1

                if self.active_position is None:
                    valid, reason, contracts = self.compliance_validator.validate_signal(
                        signal, bar.timestamp
                    )
                    if valid and contracts > 0:
                        self.active_position = ActivePosition(
                            entry_time=bar.timestamp,
                            symbol=signal.symbol,
                            direction=signal.direction,
                            contracts=contracts,
                            entry_price=signal.entry_price,
                            stop_loss=signal.stop_loss,
                            take_profit=signal.take_profit,
                        )
                        if self.debug and len(self.closed_trades) < 3:
                            print(f"[+] Orden EJECUTADA #{len(self.closed_trades)+1}: {signal.direction.label} {contracts} contratos @ {signal.entry_price} (P_meta={meta_prob:.2f})")
                    else:
                        if self.debug and self.signals_passed_meta <= 5:
                            print(f"[-] Señal rechazada por PropCompliance: {reason} (contratos={contracts})")

        if self.debug:
            print(f"\n[*] Diagnóstico de señales: Total detectadas={self.signals_seen} | Aprobadas por Meta-Model={self.signals_passed_meta} | Trades tomados={len(self.closed_trades)}")

        if self.current_date is not None:
            self.equity_history[self.current_date] = daily_closed_balance

        equity_series = pd.Series(self.equity_history)
        trades_dict = [t.__dict__ for t in self.closed_trades]
        metrics = MetricsCalculator.calculate(
            trades=trades_dict,
            daily_equity_series=equity_series,
            initial_capital=self.initial_capital,
        )

        return BacktestResult(
            metrics=metrics,
            trades=self.closed_trades,
            equity_curve=equity_series,
        )
