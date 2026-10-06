"""Monitor de trailing drawdown intradía (mark-to-market) con bloqueo del suelo para prop firms."""

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
    floor: float = 0.0
    locked: bool = False
    breached: bool = False


class DrawdownTracker:
    """
    Trailing drawdown con suelo F = min(HWM - D, balance_inicial + lock_offset).

    - peak_unrealized (Apex/Topstep): el HWM sube con el equity flotante máximo intrabarra.
    - closed_balance: el HWM sólo sube con balance cerrado.
    En ambos modos la brecha se evalúa contra el equity más adverso de la barra.

    Convención conservadora intrabarra: la excursión favorable ocurre ANTES que la adversa
    (eleva el suelo primero), de modo que MFE + pérdida final consumen colchón.
    """

    def __init__(
        self,
        initial_balance: float,
        max_trailing_drawdown: float,
        mode: Literal["peak_unrealized", "closed_balance"] = "peak_unrealized",
        lock_offset: float = 100.0,
        lock_enabled: bool = True,
    ) -> None:
        self.initial_balance = initial_balance
        self.max_trailing_drawdown = max_trailing_drawdown
        self.mode = mode
        self.lock_offset = lock_offset
        self.lock_enabled = lock_enabled

        self.current_balance = initial_balance
        self.unrealized_pnl = 0.0
        self.high_water_mark = initial_balance
        self.floor = initial_balance - max_trailing_drawdown
        self.peak_drawdown = 0.0
        self.breached = False

    @property
    def locked(self) -> bool:
        return (
            self.lock_enabled
            and self.high_water_mark - self.max_trailing_drawdown
            >= self.initial_balance + self.lock_offset
        )

    def _apply(self, closed_balance: float, best_equity: float, worst_equity: float) -> None:
        reference = best_equity if self.mode == "peak_unrealized" else closed_balance
        if reference > self.high_water_mark:
            self.high_water_mark = reference

        trailing = self.high_water_mark - self.max_trailing_drawdown
        # min() hace el suelo monótono no decreciente: HWM sólo sube y el tope es constante
        self.floor = (
            min(trailing, self.initial_balance + self.lock_offset)
            if self.lock_enabled
            else trailing
        )

        if worst_equity <= self.floor:
            self.breached = True
        self.peak_drawdown = max(self.peak_drawdown, self.high_water_mark - worst_equity)

    def _state(self, closed_balance: float, unrealized: float, equity: float) -> DrawdownState:
        current_dd = max(0.0, self.high_water_mark - equity)
        return DrawdownState(
            current_balance=closed_balance,
            unrealized_pnl=unrealized,
            equity=equity,
            high_water_mark=self.high_water_mark,
            current_drawdown=current_dd,
            drawdown_pct=(current_dd / self.high_water_mark) if self.high_water_mark > 0 else 0.0,
            remaining_buffer=max(0.0, equity - self.floor),
            floor=self.floor,
            locked=self.locked,
            breached=self.breached,
        )

    def update(self, closed_balance: float, unrealized_pnl: float = 0.0) -> DrawdownState:
        """Actualización puntual (cierre de trade o validación de señal)."""
        self.current_balance = closed_balance
        self.unrealized_pnl = unrealized_pnl
        equity = closed_balance + unrealized_pnl
        self._apply(closed_balance, equity, equity)
        return self._state(closed_balance, unrealized_pnl, equity)

    def update_intrabar(
        self,
        closed_balance: float,
        best_open_pnl: float,
        worst_open_pnl: float,
    ) -> DrawdownState:
        """
        Actualización por barra con PnL abierto (incluye parciales ya realizados de la posición)
        en sus extremos favorable/adverso. `closed_balance` excluye la posición en curso.
        """
        self.current_balance = closed_balance
        self.unrealized_pnl = worst_open_pnl
        self._apply(
            closed_balance,
            best_equity=closed_balance + best_open_pnl,
            worst_equity=closed_balance + worst_open_pnl,
        )
        return self._state(closed_balance, worst_open_pnl, closed_balance + worst_open_pnl)