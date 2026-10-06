"""ORB retest + pullback: máquina de estados con invalidación, timeout, stop estructural y re-armado."""

import math
import statistics
from collections import deque
from datetime import date
from enum import Enum

from src.alpha.base_alpha import BaseAlpha
from src.core.constants import Direction, SessionPhase
from src.core.events import BarEvent, SignalEvent
from src.core.time_utils import get_session_phase, to_ny_time
from src.features.opening_range import OpeningRange, OpeningRangeTracker
from src.features.technicals import IntradayVWAP, RelativeVolumeTracker, RollingATR

_SQRT_15 = math.sqrt(15.0)
_SQRT_5 = math.sqrt(5.0)


class RetestState(str, Enum):
    IDLE = "IDLE"
    WAITING_RETEST_LONG = "WAITING_RETEST_LONG"
    WAITING_RETEST_SHORT = "WAITING_RETEST_SHORT"
    COMPLETED = "COMPLETED"


class ORBTrendContinuationStrategy(BaseAlpha):
    """
    IDLE -> WAITING_RETEST_{side} -> (señal) -> IDLE | COMPLETED

    Transiciones de WAITING_RETEST:
    - invalidación: cierre dentro del OR por más de `invalidation_atr`·ATR  -> IDLE
    - timeout: más de `max_wait_bars` barras                                -> IDLE
    - confirmación: toque de zona + barra de reversión + (opcional) flujo   -> señal
    Tras una señal, el mismo lado sólo se rearma si el precio supera la extensión previa
    (máx. `max_entries_per_side` por lado y día). Se cuentan señales emitidas, no fills.
    """

    def __init__(
        self,
        symbol: str = "MNQ",
        buffer_ticks: int = 2,
        retest_tolerance_ticks: int = 12,
        tick_size: float = 0.25,
        min_rvol_threshold: float = 1.0,
        min_range_atr_ratio: float = 0.60,
        max_range_atr_ratio: float = 2.50,
        stop_atr_multiplier: float = 1.25,
        risk_reward_ratio: float = 2.20,
        target_atr_multiplier: float | None = None,
        hard_stop_max_ticks: int = 100,
        *,
        stop_min_ticks: int = 40,
        stop_buffer_ticks: int = 2,
        atr_floor_mult: float = 0.8,
        min_rr: float = 1.5,
        target_range_mult: float = 1.0,
        entry_mode: str = "limit_pullback",
        limit_pullback_frac: float = 0.382,
        limit_ttl_bars: int = 3,
        max_wait_bars: int = 25,
        invalidation_atr: float = 0.5,
        min_close_location: float = 0.60,
        max_entries_per_side: int = 2,
        allow_vwap_retest: bool = False,
        require_flow: bool = False,
        rvol_history_days: int = 20,
        rvol_min_history: int = 5,
    ) -> None:
        if entry_mode not in ("market_close", "limit_pullback"):
            raise ValueError(f"entry_mode inválido: {entry_mode}")

        self.symbol = symbol
        self.buffer_ticks = buffer_ticks
        self.retest_tolerance = retest_tolerance_ticks * tick_size
        self.tick_size = tick_size
        self.min_rvol_threshold = min_rvol_threshold
        self.min_range_atr_ratio = min_range_atr_ratio
        self.max_range_atr_ratio = max_range_atr_ratio
        self.stop_atr_multiplier = stop_atr_multiplier
        self.risk_reward_ratio = (
            target_atr_multiplier if target_atr_multiplier is not None else risk_reward_ratio
        )
        self.hard_stop_max_ticks = hard_stop_max_ticks
        self.stop_min_ticks = stop_min_ticks
        self.stop_buffer_ticks = stop_buffer_ticks
        self.atr_floor_mult = atr_floor_mult
        self.min_rr = min_rr
        self.target_range_mult = target_range_mult
        self.entry_mode = entry_mode
        self.limit_pullback_frac = limit_pullback_frac
        self.limit_ttl_bars = limit_ttl_bars
        self.max_wait_bars = max_wait_bars
        self.invalidation_atr = invalidation_atr
        self.min_close_location = min_close_location
        self.max_entries_per_side = max_entries_per_side
        self.allow_vwap_retest = allow_vwap_retest
        self.require_flow = require_flow
        self.rvol_min_history = rvol_min_history

        self.orb_tracker = OpeningRangeTracker(tick_size=tick_size)
        self.vwap_tracker = IntradayVWAP()
        self.atr_tracker = RollingATR(period=14)
        self.rvol_tracker = RelativeVolumeTracker()

        # estado multi-día
        self._orb_volumes: deque[float] = deque(maxlen=rvol_history_days)
        self._orb_recorded_date: date | None = None
        self._effort: deque[float] = deque(maxlen=60)
        self._effort_pending: float | None = None
        self.current_date: date | None = None

        self.reset_session()

    def reset_session(self) -> None:
        self.state = RetestState.IDLE
        self.retest_pivot_extreme = 0.0
        self._direction = 0
        self._bars_waiting = 0
        self._pullback_extreme = 0.0
        self._pb_delta = 0.0
        self._pb_volume = 0.0
        self._entries = {1: 0, -1: 0}
        self._extension: dict[int, float | None] = {1: None, -1: None}
        self._filters_ok: bool | None = None
        self._rvol: float | None = None
        self._range_ratio = 0.0

    # ------------------------------------------------------------------ utils
    def _round_tick(self, price: float) -> float:
        return round(round(price / self.tick_size) * self.tick_size, 4)

    def _effort_z(self, bar: BarEvent) -> float:
        """z-score de volumen/rango (esfuerzo vs resultado) frente a la ventana previa."""
        if len(self._effort) < 20:
            return 0.0
        x = bar.volume / ((bar.high - bar.low) + self.tick_size)
        sd = statistics.pstdev(self._effort)
        return (x - statistics.fmean(self._effort)) / sd if sd > 0 else 0.0

    def _push_effort(self, bar: BarEvent) -> None:
        # la barra actual entra en la ventana en la siguiente llamada: z se mide contra barras previas
        if self._effort_pending is not None:
            self._effort.append(self._effort_pending)
        self._effort_pending = bar.volume / ((bar.high - bar.low) + self.tick_size)

    def _evaluate_day_filters(self, orb: OpeningRange, atr: float) -> bool:
        if self._filters_ok is not None:
            return self._filters_ok

        self._range_ratio = orb.range_size / (atr * _SQRT_15)
        n = len(self._orb_volumes)
        median_vol = statistics.median(self._orb_volumes) if n else 0.0
        if n >= max(1, self.rvol_min_history) and median_vol > 0:
            self._rvol = orb.total_volume / median_vol
        elif self.rvol_min_history == 0:
            self._rvol = self.rvol_tracker.calculate_rvol(orb.total_volume)
        else:
            self._rvol = None  # warm-up: sin benchmark fiable no se opera

        if self._orb_recorded_date != self.current_date:
            self._orb_volumes.append(orb.total_volume)
            self._orb_recorded_date = self.current_date

        self._filters_ok = (
            self._rvol is not None
            and self._rvol >= self.min_rvol_threshold
            and self.min_range_atr_ratio <= self._range_ratio <= self.max_range_atr_ratio
        )
        return self._filters_ok

    # ------------------------------------------------------------------ main
    def on_bar(self, bar: BarEvent) -> SignalEvent | None:
        if bar.symbol != self.symbol:
            return None

        bar_date = to_ny_time(bar.timestamp).date()
        if self.current_date != bar_date:
            self.current_date = bar_date
            self.reset_session()

        orb = self.orb_tracker.update(bar)
        vwap_data = self.vwap_tracker.update(bar)
        atr = self.atr_tracker.update(bar.high, bar.low, bar.close)
        absorption_z = self._effort_z(bar)
        self._push_effort(bar)

        if get_session_phase(bar.timestamp) != SessionPhase.ACTIVE_TRADING:
            return None
        if self.state == RetestState.COMPLETED or orb is None or not self.orb_tracker.is_ready:
            return None
        if vwap_data is None or atr <= 0:
            return None
        if not self._evaluate_day_filters(orb, atr):
            return None

        if self.state == RetestState.IDLE:
            self._detect_breakout(bar, orb, vwap_data.vwap)
            return None
        return self._process_retest(bar, orb, vwap_data.vwap, atr, absorption_z)

    def _detect_breakout(self, bar: BarEvent, orb: OpeningRange, vwap: float) -> None:
        buf = self.buffer_ticks * self.tick_size
        for d in (1, -1):
            if self._entries[d] >= self.max_entries_per_side:
                continue
            if d == 1:
                broke = bar.close > orb.high + buf and bar.close > vwap
            else:
                broke = bar.close < orb.low - buf and bar.close < vwap
            if not broke:
                continue
            ext = self._extension[d]
            if ext is not None and (bar.close - ext) * d <= 0:
                continue  # re-armado exige nueva extensión

            self.state = RetestState.WAITING_RETEST_LONG if d == 1 else RetestState.WAITING_RETEST_SHORT
            self._direction = d
            self._bars_waiting = 0
            self._pullback_extreme = math.inf if d == 1 else -math.inf
            self.retest_pivot_extreme = bar.high if d == 1 else bar.low
            self._pb_delta = 0.0
            self._pb_volume = 0.0
            return

    def _abort_setup(self) -> None:
        self.state = RetestState.IDLE
        self._direction = 0
        self._bars_waiting = 0
        self._pb_delta = 0.0
        self._pb_volume = 0.0

    def _process_retest(
        self, bar: BarEvent, orb: OpeningRange, vwap: float, atr: float, absorption_z: float
    ) -> SignalEvent | None:
        d = self._direction
        level = orb.high if d == 1 else orb.low
        self._bars_waiting += 1

        if d == 1:
            self.retest_pivot_extreme = max(self.retest_pivot_extreme, bar.high)
            self._pullback_extreme = min(self._pullback_extreme, bar.low)
        else:
            self.retest_pivot_extreme = min(self.retest_pivot_extreme, bar.low)
            self._pullback_extreme = max(self._pullback_extreme, bar.high)

        invalidated = (bar.close - level) * d < -self.invalidation_atr * atr
        if invalidated or self._bars_waiting > self.max_wait_bars:
            self._abort_setup()
            return None

        signal = self._try_confirm(bar, orb, vwap, atr, absorption_z)

        # el flujo de la barra de confirmación no forma parte del pullback previo
        if bar.delta is not None:
            self._pb_delta += bar.delta
            self._pb_volume += bar.volume
        return signal

    def _try_confirm(
        self, bar: BarEvent, orb: OpeningRange, vwap: float, atr: float, absorption_z: float
    ) -> SignalEvent | None:
        d = self._direction
        tick = self.tick_size
        tol = self.retest_tolerance
        level = orb.high if d == 1 else orb.low
        adverse = bar.low if d == 1 else bar.high

        in_zone = -tol <= (adverse - level) * d <= 2.0 * tol
        if not in_zone and self.allow_vwap_retest:
            in_zone = abs(adverse - vwap) <= tol
        if not in_zone:
            return None

        if (bar.close - bar.open) * d <= 0 or (bar.close - vwap) * d <= 0:
            return None
        rng = bar.high - bar.low
        if rng <= 0:
            return None
        close_loc = (bar.close - bar.low) / rng if d == 1 else (bar.high - bar.close) / rng
        if close_loc < self.min_close_location:
            return None

        delta_confirm: float | None = None
        if bar.delta is not None and bar.volume > 0:
            delta_confirm = d * bar.delta / bar.volume
        if self.require_flow and (delta_confirm is None or delta_confirm <= 0):
            return None

        # --- geometría de la orden ---
        if self.entry_mode == "market_close":
            entry = bar.close
            order_type = "MARKET"
        else:
            entry = self._round_tick(bar.close - d * self.limit_pullback_frac * abs(bar.close - adverse))
            order_type = "LIMIT"

        if d == 1:
            inv_level = min(self._pullback_extreme, orb.high - 0.25 * orb.range_size)
        else:
            inv_level = max(self._pullback_extreme, orb.low + 0.25 * orb.range_size)
        structural_stop = inv_level - d * self.stop_buffer_ticks * tick

        risk = (entry - structural_stop) * d
        floor = max(
            self.stop_min_ticks * tick,
            self.atr_floor_mult * atr * _SQRT_5,
            self.stop_atr_multiplier * atr,
        )
        risk = max(risk, floor)
        risk = math.ceil(risk / tick - 1e-9) * tick
        if risk > self.hard_stop_max_ticks * tick:
            return None  # stop estructural demasiado ancho para el presupuesto de riesgo

        stop = round(entry - d * risk, 2)
        target_dist = min(
            max(self.target_range_mult * orb.range_size, self.min_rr * risk),
            self.risk_reward_ratio * risk,
        )
        take_profit = self._round_tick(entry + d * target_dist)

        rvol = self._rvol if self._rvol is not None else 0.0
        pb_ratio = (self._pb_delta / self._pb_volume * d) if self._pb_volume > 0 else None
        depth_ticks = (level - self._pullback_extreme) * d / tick

        direction = Direction.LONG if d == 1 else Direction.SHORT
        self._entries[d] += 1
        self._extension[d] = self.retest_pivot_extreme
        bars_since_breakout = self._bars_waiting
        self._abort_setup()
        if all(self._entries[s] >= self.max_entries_per_side for s in (1, -1)):
            self.state = RetestState.COMPLETED

        return SignalEvent(
            timestamp=bar.timestamp,
            symbol=self.symbol,
            direction=direction,
            entry_price=entry,
            stop_loss=stop,
            take_profit=take_profit,
            confidence_score=min(1.0, 0.5 + rvol / 10.0),  # legado: determinista en RVOL, no usar como feature
            metadata={
                "strategy": f"ORB_Retest_Continuation_{direction.label.title()}",
                "orb_high": orb.high,
                "orb_low": orb.low,
                "rvol": round(rvol, 3),
                "atr": atr,
                "vwap": vwap,
                "order_type": order_type,
                "ttl_bars": self.limit_ttl_bars,
                "stop_ticks": risk / tick,
                "target_ticks": target_dist / tick,
                "rr": round(target_dist / risk, 3),
                "bars_since_breakout": bars_since_breakout,
                "retest_depth_ticks": round(depth_ticks, 2),
                "close_location": round(close_loc, 3),
                "delta_confirm": None if delta_confirm is None else round(delta_confirm, 4),
                "pullback_delta_ratio": None if pb_ratio is None else round(pb_ratio, 4),
                "absorption_z": round(absorption_z, 3),
                "range_atr_ratio": round(self._range_ratio, 3),
                "entry_number": self._entries[d],
            },
        )