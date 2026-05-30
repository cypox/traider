"""Signal and forecast events."""

from dataclasses import dataclass

from bot.core.signals import Forecast, Signal
from bot.events.base import BaseEvent


@dataclass(frozen=True, slots=True)
class SignalEvent(BaseEvent):
    """Wraps a Signal emitted by a strategy."""

    signal: Signal


@dataclass(frozen=True, slots=True)
class ForecastEvent(BaseEvent):
    """Wraps a combined Forecast produced by portfolio construction."""

    forecast: Forecast
