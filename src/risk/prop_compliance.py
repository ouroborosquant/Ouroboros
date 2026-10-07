"""Validador global de cumplimiento normativo para cuentas fondeadas."""

from datetime import datetime

from src.core.constants import CircuitBreakerStatus, SessionPhase
from src.core.events import OrderEvent, SignalEvent
from src.core.time_utils import get_session_phase
from src.risk.circuit_breakers import CircuitBreaker
from src.risk.drawdown_tracker import DrawdownTracker
from src.risk.position_sizer import InstrumentSpecs, PositionSizer


class PropComplianceValidator:
    """Validador central de seguridad y restricciones de cuenta fondeada."""

    def __init__(
        self,
        circuit_breaker: CircuitBreaker,
        drawdown_tracker: DrawdownTracker,
        position_sizer: PositionSizer,
        specs: dict[str, InstrumentSpecs],
    ) -> None:
        self.circuit_breaker = circuit_breaker
        self.drawdown_tracker = drawdown_tracker
        self.position_sizer = position_sizer
        self.specs = specs

    def validate_signal(self, signal: SignalEvent, current_time: datetime) -> tuple[bool, str, int]:
        """
        Evalúa una señal de entrada. Devuelve (aprobado: bool, motivo: str, contratos: int).
        """
        # 1. Validación de fase de mercado
        phase = get_session_phase(current_time)
        if phase != SessionPhase.ACTIVE_TRADING:
            return False, f"Fase de mercado no válida para entrar: {phase.value}", 0

        # 2. Validación de disyuntores de circuito
        can_trade, reason = self.circuit_breaker.can_open_new_trade()
        if not can_trade:
            return False, reason, 0

        # 3. Verificación de especificaciones de instrumento
        instrument = self.specs.get(signal.symbol)
        if not instrument:
            return False, f"Instrumento {signal.symbol} no configurado", 0

        # 4. Cálculo de contratos con el colchón disponible
        state = self.drawdown_tracker.update(
            closed_balance=self.drawdown_tracker.current_balance,
            unrealized_pnl=self.drawdown_tracker.unrealized_pnl,
        )

        contracts = self.position_sizer.calculate_contracts(
            remaining_buffer=state.remaining_buffer,
            entry_price=signal.entry_price,
            stop_loss_price=signal.stop_loss,
            specs=instrument,
        )

        # Reducir contratos a la mitad si estamos en SOFT_ALERT (nunca promover 0 -> 1)
        if contracts > 0 and self.circuit_breaker.status == CircuitBreakerStatus.SOFT_ALERT:
            contracts = max(1, contracts // 2)

        if contracts <= 0:
            return False, "Riesgo supera el colchón disponible de la cuenta (0 contratos calculados)", 0

        return True, "Señal conforme a normativas de riesgo", contracts