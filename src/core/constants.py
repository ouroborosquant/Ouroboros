"""Constantes globales y enumeraciones tipadas del sistema."""

from enum import Enum, auto


class Direction(int, Enum):
    """Dirección de la posición o señal en el mercado."""

    SHORT = -1
    FLAT = 0
    LONG = 1

    @property
    def label(self) -> str:
        return { -1: "SHORT", 0: "FLAT", 1: "LONG" }[self.value]


class OrderType(str, Enum):
    """Tipos de orden soportados por el broker."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(str, Enum):
    """Ciclo de vida de una orden."""

    PENDING_SUBMIT = "PENDING_SUBMIT"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class SessionPhase(str, Enum):
    """Fases de la sesión operativa (horario RTH de Nueva York)."""

    PRE_MARKET = "PRE_MARKET"                  # Antes de las 09:30 EST
    ORB_BUILDING = "ORB_BUILDING"              # 09:30 a 09:45 EST (Construcción del rango)
    ACTIVE_TRADING = "ACTIVE_TRADING"          # 09:45 a 11:30 EST (Ventana óptima de entrada)
    POST_ENTRY_CUTOFF = "POST_ENTRY_CUTOFF"    # 11:30 a 15:50 EST (Solo gestión de posiciones)
    MANDATORY_FLATTEN = "MANDATORY_FLATTEN"    # 15:50 a 16:00 EST (Liquidación forzada)
    CLOSED = "CLOSED"                          # Fuera de mercado


class CircuitBreakerStatus(str, Enum):
    """Nivel de alerta y disyunción del motor de riesgo."""

    NORMAL = "NORMAL"                          # Operativa estándar
    SOFT_ALERT = "SOFT_ALERT"                  # Riesgo reducido al 50%
    TRADING_HALTED = "TRADING_HALTED"          # Bloqueo de nuevas entradas por hoy
    EMERGENCY_SHUTDOWN = "EMERGENCY_SHUTDOWN"  # Cierre forzoso de todo y desconexión