"""Financial instrument definitions."""

from dataclasses import dataclass
from enum import Enum

from bot.core.errors import InvalidInstrumentError


class AssetClass(Enum):
    """The asset class of a financial instrument."""

    EQUITY = "EQUITY"
    BOND = "BOND"
    FUTURE = "FUTURE"
    CRYPTO = "CRYPTO"
    FX = "FX"


@dataclass(frozen=True, slots=True, eq=False)
class Instrument:
    """An identifiable financial instrument traded on a specific exchange.

    Equality and hashing are based solely on (symbol, exchange) so that the same
    ticker on two different exchanges is treated as a distinct instrument.
    """

    symbol: str
    asset_class: AssetClass
    currency: str
    exchange: str

    def __post_init__(self) -> None:
        if not self.symbol or not self.currency or not self.exchange:
            raise InvalidInstrumentError(
                f"symbol, currency, and exchange must be non-empty; "
                f"got symbol={self.symbol!r}, currency={self.currency!r}, "
                f"exchange={self.exchange!r}"
            )

    def __str__(self) -> str:
        return f"{self.symbol}@{self.exchange}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Instrument):
            return NotImplemented
        return (self.symbol, self.exchange) == (other.symbol, other.exchange)

    def __hash__(self) -> int:
        return hash((self.symbol, self.exchange))
