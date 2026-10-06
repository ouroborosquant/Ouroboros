"""Disyuntores deterministas para protección contra pérdidas intradía."""

from src.core.constants import CircuitBreakerStatus


class CircuitBreaker:
    """
    Monitor de seguridad que transiciona entre estados:
    NORMAL -> SOFT_ALERT (50% daily limit) -> TRADING_HALTED (85% o 2 pérdidas) -> EMERGENCY (95%)
    """

    def __init__(
        self,
        max_daily_loss: float,
        max_consecutive_losses: int = 2,
        max_trades_per_day: int = 3,
    ) -> None:
        self.max_daily_loss = abs(max_daily_loss)
        self.max_consecutive_losses = max_consecutive_losses
        self.max_trades_per_day = max_trades_per_day

        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.trades_executed_today = 0
        self.status = CircuitBreakerStatus.NORMAL

    def register_trade_closed(self, pnl: float) -> CircuitBreakerStatus:
        """Actualiza estadísticas tras el cierre de un trade."""
        self.daily_pnl += pnl
        self.trades_executed_today += 1

        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        return self.evaluate_status()

    def update_intraday_pnl(self, current_daily_equity_pnl: float) -> CircuitBreakerStatus:
        """Evalúa las pérdidas en tiempo real (cerradas + flotantes)."""
        self.daily_pnl = current_daily_equity_pnl
        return self.evaluate_status()

    def evaluate_status(self) -> CircuitBreakerStatus:
        """Determina el estado del disyuntor según las reglas configuradas."""
        # Pérdida acumulada en el día (en positivo para evaluar umbrales)
        loss = abs(self.daily_pnl) if self.daily_pnl < 0 else 0.0

        # Nivel 3: Liquidación total de emergencia (95% del límite diario)
        if loss >= self.max_daily_loss * 0.95:
            self.status = CircuitBreakerStatus.EMERGENCY_SHUTDOWN
            return self.status

        # Nivel 2: Parada completa de nuevas entradas (85% de pérdida, límite de trades o 2 pérdidas seguidas)
        if (
            loss >= self.max_daily_loss * 0.85
            or self.consecutive_losses >= self.max_consecutive_losses
            or self.trades_executed_today >= self.max_trades_per_day
        ):
            self.status = CircuitBreakerStatus.TRADING_HALTED
            return self.status

        # Nivel 1: Alerta suave (50% del límite diario) -> Reducción de riesgo a la mitad
        if loss >= self.max_daily_loss * 0.50:
            self.status = CircuitBreakerStatus.SOFT_ALERT
            return self.status

        self.status = CircuitBreakerStatus.NORMAL
        return self.status

    def can_open_new_trade(self) -> tuple[bool, str]:
        """Verifica si se permite una nueva orden."""
        if self.status in (CircuitBreakerStatus.TRADING_HALTED, CircuitBreakerStatus.EMERGENCY_SHUTDOWN):
            return False, f"Trading bloqueado por Circuit Breaker ({self.status.value})"
        if self.trades_executed_today >= self.max_trades_per_day:
            return False, f"Límite de operaciones diarias alcanzado ({self.trades_executed_today})"
        if self.consecutive_losses >= self.max_consecutive_losses:
            return False, f"Límite de pérdidas consecutivas alcanzado ({self.consecutive_losses})"
        return True, "Operación autorizada"

    def reset_daily_session(self) -> None:
        """Reinicia los contadores al inicio de una nueva sesión bursátil."""
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.trades_executed_today = 0
        self.status = CircuitBreakerStatus.NORMAL