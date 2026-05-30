"""Value objects used by provider interfaces."""

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal

from bot.core.instruments import Instrument
from bot.core.money import Money


@dataclass(frozen=True, slots=True)
class ContractDetails:
    """Full contract specification for a financial instrument.

    ``coupon``, ``maturity_date``, and ``face_value`` are ``None`` for non-bonds.
    """

    instrument: Instrument
    full_name: str
    coupon: Decimal | None
    maturity_date: date | None
    face_value: Money | None
    tick_size: Decimal
    multiplier: Decimal


@dataclass(frozen=True, slots=True)
class YieldCurve:
    """A risk-free yield curve defined by tenor/rate pairs.

    ``tenors_days`` must be strictly increasing.
    ``rates`` are annualized decimal fractions (0.052 == 5.2%).
    Both sequences must have the same length.
    """

    as_of: datetime
    tenors_days: tuple[int, ...]
    rates: tuple[Decimal, ...]

    def __post_init__(self) -> None:
        if len(self.tenors_days) != len(self.rates):
            raise ValueError(
                f"tenors_days and rates must have the same length; "
                f"got {len(self.tenors_days)} and {len(self.rates)}"
            )
        for i in range(1, len(self.tenors_days)):
            if self.tenors_days[i] <= self.tenors_days[i - 1]:
                raise ValueError(
                    f"tenors_days must be strictly increasing; "
                    f"got {self.tenors_days[i - 1]} then {self.tenors_days[i]}"
                )
        for rate in self.rates:
            if rate < Decimal(0):
                raise ValueError(f"rates must be non-negative; got {rate}")

    def interpolate(self, days: int) -> Decimal:
        """Return the interpolated rate for the given number of days.

        Uses linear interpolation between adjacent tenors.
        Applies flat extrapolation beyond the endpoints (no exception raised).
        If only one tenor is present, that rate is returned for any ``days``.
        """
        if len(self.tenors_days) == 1:
            return self.rates[0]
        if days <= self.tenors_days[0]:
            return self.rates[0]
        if days >= self.tenors_days[-1]:
            return self.rates[-1]
        for i in range(len(self.tenors_days) - 1):
            t0, t1 = self.tenors_days[i], self.tenors_days[i + 1]
            if days <= t1:
                r0, r1 = self.rates[i], self.rates[i + 1]
                fraction = Decimal(days - t0) / Decimal(t1 - t0)
                return r0 + fraction * (r1 - r0)
        return self.rates[-1]  # pragma: no cover  # unreachable; satisfies type checker
