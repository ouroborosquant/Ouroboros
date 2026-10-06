"""Definición de eventos para la arquitectura orientada a eventos."""

from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field

from src.core.constants import CircuitBreakerStatus, Direction, OrderStatus, OrderType


class BaseEvent(BaseModel):
    """Evento base con marca temporal."""

    timestamp: datetime = Field(description="Momento en que ocurre el evento")

    model_config = {"frozen": True}


class BarEvent(BaseEvent):
    """Barra OHLCV agregada recibida del feed de datos."""

    symbol: str
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float | None = None
    delta: float | None = None


class SignalEvent(BaseEvent):
    """Señal cuantitativa emitida por la estrategia alpha tras pasar filtros."""

    symbol: str
    direction: Direction
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence_score: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class OrderEvent(BaseEvent):
    """Orden generada y validada por el módulo de riesgo hacia el broker."""

    order_id: str
    symbol: str
    direction: Direction
    order_type: OrderType
    quantity: int = Field(gt=0)
    limit_price: float | None = None
    stop_price: float | None = None
    tag: str = ""


class FillEvent(BaseEvent):
    """Confirmación de ejecución de orden emitida por el broker."""

    fill_id: str
    order_id: str
    symbol: str
    direction: Direction
    filled_quantity: int
    fill_price: float
    commission: float
    slippage_ticks: float = 0.0


class CircuitBreakerEvent(BaseEvent):
    """Evento crítico emitido por el sistema de corte de pérdidas."""

    status: CircuitBreakerStatus
    reason: str
    current_drawdown: float
    daily_pnl: float