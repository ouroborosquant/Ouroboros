from abc import ABC, abstractmethod

from src.core.events import BarEvent, SignalEvent


class BaseAlpha(ABC):
    @abstractmethod
    def on_bar(self, bar: BarEvent) -> SignalEvent | None:
        pass

    @abstractmethod
    def reset_session(self) -> None:
        pass
