"""Abstract base class for all trading strategies."""

from abc import ABC, abstractmethod

from bot.events.bus import EventBus
from bot.events.market import BarEvent, QuoteEvent


class Strategy(ABC):
    """Abstract base for all trading strategies.

    The constructor automatically subscribes :meth:`on_quote` and
    :meth:`on_bar` to the provided *event_bus*.
    """

    def __init__(self, event_bus: EventBus, name: str) -> None:
        self._event_bus = event_bus
        self._name = name
        event_bus.subscribe(QuoteEvent, self.on_quote)
        event_bus.subscribe(BarEvent, self.on_bar)

    @property
    def name(self) -> str:
        """Return the strategy name."""
        return self._name

    @abstractmethod
    def on_quote(self, event: QuoteEvent) -> None:  # pragma: no cover
        """Handle an incoming quote update."""
        ...

    @abstractmethod
    def on_bar(self, event: BarEvent) -> None:  # pragma: no cover
        """Handle an incoming OHLCV bar."""
        ...
