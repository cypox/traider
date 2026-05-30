"""Immutable monetary value type."""

from dataclasses import dataclass
from decimal import Decimal

from bot.core.errors import CurrencyMismatchError


@dataclass(frozen=True, slots=True)
class Money:
    """An immutable monetary value with an ISO 4217 currency code.

    All arithmetic operations return new Money instances. Addition and
    subtraction require matching currencies; multiplication accepts a plain
    Decimal scalar.
    """

    amount: Decimal
    currency: str

    def __add__(self, other: "Money") -> "Money":
        if self.currency != other.currency:
            raise CurrencyMismatchError(f"Cannot add {self.currency} and {other.currency}")
        return Money(self.amount + other.amount, self.currency)

    def __sub__(self, other: "Money") -> "Money":
        if self.currency != other.currency:
            raise CurrencyMismatchError(f"Cannot subtract {other.currency} from {self.currency}")
        return Money(self.amount - other.amount, self.currency)

    def __mul__(self, scalar: Decimal) -> "Money":
        return Money(self.amount * scalar, self.currency)

    def __neg__(self) -> "Money":
        return Money(-self.amount, self.currency)

    def __abs__(self) -> "Money":
        return Money(abs(self.amount), self.currency)

    @property
    def is_positive(self) -> bool:
        """Return True when amount > 0."""
        return self.amount > Decimal(0)

    @property
    def is_negative(self) -> bool:
        """Return True when amount < 0."""
        return self.amount < Decimal(0)

    @property
    def is_zero(self) -> bool:
        """Return True when amount == 0."""
        return self.amount == Decimal(0)

    @classmethod
    def zero(cls, currency: str) -> "Money":
        """Return a zero-valued Money for the given currency."""
        return cls(Decimal(0), currency)
