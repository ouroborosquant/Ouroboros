"""Dimensionamiento de posición con fórmula de supervivencia ante la ruina."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class InstrumentSpecs:
    symbol: str
    tick_size: float
    point_value: float
    max_contracts: int


class PositionSizer:
    """
    Calcula el tamaño de la orden (en contratos enteros) en función del
    colchón de pérdida disponible y la distancia al Stop Loss.
    """

    def __init__(self, risk_divisor_k: float = 14.0, max_buffer_risk_pct: float = 0.07) -> None:
        self.risk_divisor_k = risk_divisor_k
        self.max_buffer_risk_pct = max_buffer_risk_pct

    def calculate_contracts(
        self,
        remaining_buffer: float,
        entry_price: float,
        stop_loss_price: float,
        specs: InstrumentSpecs,
    ) -> int:
        """
        Calcula contratos enteros permitidos.
        Si el stop loss es demasiado amplio para el colchón, devuelve 0 (orden rechazada).
        """
        if remaining_buffer <= 0:
            return 0

        # Máximo capital a arriesgar en este trade específico ($)
        risk_by_k = remaining_buffer / self.risk_divisor_k
        risk_by_pct = remaining_buffer * self.max_buffer_risk_pct
        risk_dollars = min(risk_by_k, risk_by_pct)

        # Distancia en puntos
        stop_distance_points = abs(entry_price - stop_loss_price)
        if stop_distance_points <= 0:
            return 0

        # Coste del riesgo por contrato individual
        risk_per_contract = stop_distance_points * specs.point_value

        raw_contracts = risk_dollars / risk_per_contract
        contracts = math.floor(raw_contracts)

        # Respetar límites máximos del contrato / firma
        return min(max(0, contracts), specs.max_contracts)