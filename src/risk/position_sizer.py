"""Dimensionamiento de posición con fórmula de supervivencia ante la ruina."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class InstrumentSpecs:
    symbol: str
    tick_size: float
    point_value: float
    max_contracts: int


def lcb_half_kelly(r_multiples: list[float], z: float = 1.64, min_trades: int = 60) -> float:
    """Risk fraction of effective capital from the lower confidence bound of mean R (half-Kelly).
    Returns 0.0 until the edge is statistically distinguishable from zero."""
    n = len(r_multiples)
    if n < min_trades:
        return 0.0
    mean = sum(r_multiples) / n
    var = sum((x - mean) ** 2 for x in r_multiples) / (n - 1)
    if var <= 0:
        return 0.0
    mu_lcb = mean - z * math.sqrt(var / n)
    return max(0.0, 0.5 * mu_lcb / var)


class PositionSizer:
    """
    risk_$ = min( base(buffer) * shrink * edge_scale , caps )
      base   = min(buffer/K, buffer*max_buffer_risk_pct)
      shrink = (buffer / D)^gamma   (convex de-risking after giving back buffer)
      caps   = hard_risk_cap_pct*buffer, 50% of remaining daily loss limit
    Defaults reproduce the previous behaviour exactly.
    """

    def __init__(
        self,
        risk_divisor_k: float = 14.0,
        max_buffer_risk_pct: float = 0.07,
        *,
        max_trailing_drawdown: float | None = None,
        dd_gamma: float = 1.0,
        hard_risk_cap_pct: float = 0.10,
        dll_fraction: float = 0.5,
        one_lot_threshold: float | None = None,
    ) -> None:
        self.risk_divisor_k = risk_divisor_k
        self.max_buffer_risk_pct = max_buffer_risk_pct
        self.max_trailing_drawdown = max_trailing_drawdown
        self.dd_gamma = dd_gamma
        self.hard_risk_cap_pct = hard_risk_cap_pct
        self.dll_fraction = dll_fraction
        self.one_lot_threshold = one_lot_threshold

    def risk_budget(
        self,
        remaining_buffer: float,
        edge_scale: float = 1.0,
        daily_loss_remaining: float | None = None,
    ) -> float:
        if remaining_buffer <= 0 or edge_scale <= 0:
            return 0.0
        base = min(remaining_buffer / self.risk_divisor_k, remaining_buffer * self.max_buffer_risk_pct)
        if self.max_trailing_drawdown:
            ratio = min(1.0, remaining_buffer / self.max_trailing_drawdown)
            base *= ratio ** self.dd_gamma
        risk = base * edge_scale
        risk = min(risk, remaining_buffer * self.hard_risk_cap_pct)
        if daily_loss_remaining is not None:
            risk = min(risk, max(0.0, daily_loss_remaining) * self.dll_fraction)
        return risk

    def calculate_contracts(
        self,
        remaining_buffer: float,
        entry_price: float,
        stop_loss_price: float,
        specs: InstrumentSpecs,
        *,
        edge_scale: float = 1.0,
        daily_loss_remaining: float | None = None,
    ) -> int:
        risk_dollars = self.risk_budget(remaining_buffer, edge_scale, daily_loss_remaining)
        stop_distance_points = abs(entry_price - stop_loss_price)
        if risk_dollars <= 0 or stop_distance_points <= 0:
            return 0
        one_lot_risk = stop_distance_points * specs.point_value
        contracts = math.floor(risk_dollars / one_lot_risk)
        if (
            contracts == 0
            and self.one_lot_threshold is not None
            and risk_dollars >= self.one_lot_threshold * one_lot_risk
            and one_lot_risk <= self.hard_risk_cap_pct * remaining_buffer
            and (
                daily_loss_remaining is None
                or one_lot_risk <= max(0.0, daily_loss_remaining) * self.dll_fraction
            )
        ):
            contracts = 1
        return min(max(0, contracts), specs.max_contracts)