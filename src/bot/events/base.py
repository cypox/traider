"""Base event type for the trading framework event bus."""

from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class BaseEvent:
    """Base class for all domain events.

    All events carry a UTC-aware timestamp, a non-empty source identifier, and
    a UUID-based event_id that is auto-generated when not supplied.
    """

    timestamp: datetime
    source: str
    event_id: str = field(default_factory=lambda: str(uuid4()), kw_only=True)

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware UTC")
        if not self.source:
            raise ValueError("source must be non-empty")
