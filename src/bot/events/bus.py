"""Synchronous, deterministic event bus."""

import contextlib
from collections import defaultdict
from collections.abc import Callable
from typing import TypeVar, cast

import structlog

from bot.events.base import BaseEvent

T = TypeVar("T", bound=BaseEvent)

_logger = structlog.get_logger(__name__)


class EventBus:
    """Synchronous publish/subscribe event bus.

    Components subscribe handlers for specific event types.  When an event is
    published, all handlers whose registered type is a superclass (or the exact
    class) of the event are called in registration order.  A failing handler is
    logged at ERROR level and never blocks remaining handlers.

    EventBus is NOT a singleton — inject one instance per component tree.
    """

    def __init__(self) -> None:
        self._handlers: dict[type[BaseEvent], list[Callable[[BaseEvent], None]]] = defaultdict(list)

    def subscribe(
        self,
        event_type: type[T],
        handler: Callable[[T], None],
    ) -> None:
        """Register *handler* to receive events of *event_type* (and subtypes)."""
        self._handlers[event_type].append(cast(Callable[[BaseEvent], None], handler))

    def unsubscribe(
        self,
        event_type: type[T],
        handler: Callable[[T], None],
    ) -> None:
        """Remove *handler* from *event_type*.  No-op if not registered."""
        handlers = self._handlers.get(event_type)
        if handlers is not None:
            handler_typed = cast(Callable[[BaseEvent], None], handler)
            with contextlib.suppress(ValueError):
                handlers.remove(handler_typed)

    def publish(self, event: BaseEvent) -> None:
        """Deliver *event* synchronously to all matching handlers.

        Handlers are matched if the event is an instance of the handler's
        registered type (subtype matching included).  If a handler raises,
        the exception is logged and delivery continues to remaining handlers.
        """
        for event_type, handlers in list(self._handlers.items()):
            if isinstance(event, event_type):
                for handler in list(handlers):
                    try:
                        handler(event)
                    except Exception as exc:  # noqa: BLE001
                        _logger.error(
                            "event handler failed",
                            handler_name=getattr(handler, "__name__", repr(handler)),
                            exception=str(exc),
                        )

    def subscriber_count(self, event_type: type[BaseEvent]) -> int:
        """Return the count of handlers registered for exactly *event_type*."""
        return len(self._handlers.get(event_type, []))

    def clear(self) -> None:
        """Remove all subscriptions.  Useful for test teardown."""
        self._handlers.clear()
