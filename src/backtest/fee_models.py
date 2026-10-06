from dataclasses import dataclass


@dataclass(frozen=True)
class TransactionCost:
    commission: float
    slippage_cost: float
    total_cost: float


class CMEFeeModel:
    def __init__(self, commission_per_contract: float = 0.62, slippage_ticks: float = 1.0) -> None:
        self.commission_per_contract = commission_per_contract
        self.slippage_ticks = slippage_ticks

    def calculate_cost(self, contracts: int, tick_value: float) -> TransactionCost:
        if contracts <= 0:
            return TransactionCost(0.0, 0.0, 0.0)

        comm = contracts * self.commission_per_contract
        slip = contracts * (self.slippage_ticks * tick_value)
        return TransactionCost(
            commission=round(comm, 2),
            slippage_cost=round(slip, 2),
            total_cost=round(comm + slip, 2),
        )
