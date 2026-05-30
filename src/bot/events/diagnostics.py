"""Diagnostics event for structured observability payloads."""

from dataclasses import dataclass

from bot.events.base import BaseEvent

# NOTE: super().__post_init__() is broken with frozen+slots dataclass inheritance
# in Python 3.13 (CPython bug). Use BaseEvent.__post_init__(self) explicitly.


@dataclass(frozen=True, slots=True)
class DiagnosticsEvent(BaseEvent):
    """A structured diagnostics payload with primitive values only.

    All values in ``payload`` must be ``str``, ``int``, ``float``, or ``bool``.
    Nested dicts, lists, and other complex types are rejected at construction.
    """

    payload: dict[str, str | int | float | bool]

    def __post_init__(self) -> None:
        BaseEvent.__post_init__(self)
        for key, value in self.payload.items():
            if not isinstance(value, str | int | float | bool):
                raise ValueError(
                    f"payload values must be str, int, float, or bool; "
                    f"got {type(value).__name__!r} for key {key!r}"
                )
