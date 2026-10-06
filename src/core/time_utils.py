"""Manejo de zonas horarias y fases de sesión para CME RTH."""

from datetime import datetime, time
from zoneinfo import ZoneInfo

from src.core.constants import SessionPhase

NY_TZ = ZoneInfo("America/New_York")
UTC_TZ = ZoneInfo("UTC")

# Horarios clave RTH (America/New_York)
TIME_RTH_OPEN = time(9, 30, 0)
TIME_ORB_END = time(9, 45, 0)
TIME_ENTRY_CUTOFF = time(11, 30, 0)
TIME_FLATTEN_START = time(15, 50, 0)
TIME_RTH_CLOSE = time(16, 0, 0)


def to_ny_time(dt: datetime) -> datetime:
    """Convierte cualquier datetime con o sin zona a America/New_York."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC_TZ).astimezone(NY_TZ)
    return dt.astimezone(NY_TZ)


def get_current_ny_time() -> datetime:
    """Devuelve la fecha/hora actual en America/New_York."""
    return datetime.now(NY_TZ)


def get_session_phase(dt: datetime) -> SessionPhase:
    """Determina la fase operativa de la sesión según el horario de Nueva York."""
    ny_dt = to_ny_time(dt)
    current_t = ny_dt.time()

    # Días no operativos (Fin de semana)
    if ny_dt.weekday() >= 5:
        return SessionPhase.CLOSED

    if current_t < TIME_RTH_OPEN:
        return SessionPhase.PRE_MARKET
    if current_t < TIME_ORB_END:
        return SessionPhase.ORB_BUILDING
    if current_t < TIME_ENTRY_CUTOFF:
        return SessionPhase.ACTIVE_TRADING
    if current_t < TIME_FLATTEN_START:
        return SessionPhase.POST_ENTRY_CUTOFF
    if current_t < TIME_RTH_CLOSE:
        return SessionPhase.MANDATORY_FLATTEN
    return SessionPhase.CLOSED