"""Monitor continuo de Drawdown flotante y cerrado para prop firms."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class DrawdownState:
    current_balance: float
    unrealized_pnl: float
    equity: float
    high_water_mark: float
    current_drawdown: float
    drawdown_pct: float
    remaining_buffer: float


class DrawdownTracker:
    """
    Rastrea el trailing drawdown según la regla de la firma:
    - peak_unrealized (Apex, Topstep): el trailing sube con las ganancias flotantes intradía.
    - closed_balance (FTMO, FundedNext): el trailing se ancla al balance al cierre del día/trade.
    """

    def __init__(
        self,
        initial_balance: float,
        max_trailing_drawdown: float,
        mode: Literal["peak_unrealized", "closed_balance"] = "peak_unrealized",
    ) -> None:
        self.initial_balance = initial_balance
        self.max_trailing_drawdown = max_trailing_drawdown
        self.mode = mode

        self.current_balance = initial_balance
        self.unrealized_pnl = 0.0
        self.high_water_mark = initial_balance
        self.lock_hwm_threshold = initial_balance + max_trailing_drawdown

    def update(self, closed_balance: float, unrealized_pnl: float = 0.0) -> DrawdownState:
        self.current_balance = closed_balance
        self.unrealized_pnl = unrealized_pnl
        equity = closed_balance + unrealized_pnl

        # Actualización del High-Water Mark según modalidad
        reference_equity = equity if self.mode == "peak_unrealized" else closed_balance
        if reference_equity > self.high_water_mark:
            self.high_water_mark = reference_equity

        # Nivel de corte mínimo donde la cuenta se declara perdida
        # En muchas firmas el trailing stop se congela cuando alcanza el balance inicial + $100
        fail_level = self.high_water_mark - self.max_trailing_drawdown

        current_dd = max(0.0, self.high_water_mark - equity)
        dd_pct = (current_dd / self.high_water_mark) if self.high_water_mark > 0 else 0.0
        remaining_buffer = max(0.0, equity - fail_level)

        return DrawdownState(
            current_balance=self.current_balance,
            unrealized_pnl=self.unrealized_pnl,
            equity=equity,
            high_water_mark=self.high_water_mark,
            current_drawdown=current_dd,
            drawdown_pct=dd_pct,
            remaining_buffer=remaining_buffer,
        )