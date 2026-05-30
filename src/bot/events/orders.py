"""Order lifecycle events."""

from dataclasses import dataclass

from bot.core.execution import ExecutionIntent
from bot.events.base import BaseEvent


@dataclass(frozen=True, slots=True)
class OrderEvent(BaseEvent):
    """An execution intent dispatched to the execution provider."""

    intent: ExecutionIntent
    order_id: str
