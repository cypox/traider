"""Execution intent model."""

from dataclasses import dataclass
from decimal import Decimal

from bot.core.errors import InvalidIntentError
from bot.core.instruments import Instrument
from bot.core.positions import ApprovedPosition
from bot.core.signals import Direction


@dataclass(frozen=True, slots=True)
class ExecutionIntent:
    """An instruction sent to the execution engine to trade a specific quantity.

    ``side`` must be LONG or SHORT (never FLAT), ``quantity`` must be strictly
    positive, and ``reason`` must be non-empty.
    """

    instrument: Instrument
    side: Direction
    quantity: Decimal
    reason: str
    source_approved: ApprovedPosition

    def __post_init__(self) -> None:
        if self.side == Direction.FLAT:
            raise InvalidIntentError("ExecutionIntent side must be LONG or SHORT, not FLAT")
        if self.quantity <= Decimal(0):
            raise InvalidIntentError(
                f"ExecutionIntent quantity must be positive; got {self.quantity}"
            )
        if not self.reason:
            raise InvalidIntentError("ExecutionIntent reason must be non-empty")
