"""Fixed income analytics: YTM, duration, DV01, and convexity."""

import math
from decimal import Decimal

from bot.analytics.errors import ConvergenceError, InvalidInputError

_MAX_ITERATIONS = 200
_TOLERANCE = Decimal("1e-10")


def _decimal_pow(base: Decimal, exponent: Decimal) -> Decimal:
    """Compute base^exponent via exp(exponent * ln(base))."""
    if exponent == Decimal(0):
        return Decimal(1)
    return (exponent * base.ln()).exp()


def _build_schedule(
    face_value: Decimal,
    coupon_rate: Decimal,
    days_to_maturity: int,
    coupon_frequency: int,
) -> list[tuple[Decimal, Decimal]]:
    """Return list of (d_i, CF_i): periods-to-cash-flow and cash-flow amount.

    d_i is expressed in coupon periods. The i-th cash flow occurs at
    d_i / coupon_frequency years from today.
    """
    period_length = Decimal("365") / Decimal(coupon_frequency)
    days_dec = Decimal(days_to_maturity)
    n_periods = math.ceil(days_dec / period_length)
    remainder = days_dec % period_length
    first_frac = remainder / period_length if remainder > Decimal(0) else Decimal("1")
    coupon_payment = coupon_rate * face_value / Decimal(coupon_frequency)

    schedule: list[tuple[Decimal, Decimal]] = []
    for i in range(1, n_periods + 1):
        d_i = first_frac + Decimal(i - 1)
        cf_i = coupon_payment + face_value if i == n_periods else coupon_payment
        schedule.append((d_i, cf_i))

    return schedule


def compute_ytm(
    face_value: Decimal,
    coupon_rate: Decimal,
    price_pct: Decimal,
    days_to_maturity: int,
    coupon_frequency: int = 2,
) -> Decimal:
    """Compute yield to maturity via Newton-Raphson iteration.

    Parameters
    ----------
    face_value:
        Par value of the bond (positive).
    coupon_rate:
        Annual coupon rate as a decimal fraction, e.g. ``Decimal("0.05")`` for 5 %.
    price_pct:
        Clean price as percentage of face value, e.g. ``Decimal("99.5")``.
    days_to_maturity:
        Calendar days until the bond matures (must be > 0).
    coupon_frequency:
        Number of coupon payments per year (default 2 for semi-annual).

    Returns
    -------
    Decimal
        Annual YTM as a decimal fraction, e.g. ``Decimal("0.05")`` for 5 %.
    """
    if days_to_maturity <= 0:
        raise InvalidInputError(f"days_to_maturity must be positive, got {days_to_maturity}")
    if price_pct <= Decimal(0):
        raise InvalidInputError(f"price_pct must be positive, got {price_pct}")
    if face_value <= Decimal(0):
        raise InvalidInputError(f"face_value must be positive, got {face_value}")

    cf = Decimal(coupon_frequency)
    actual_price = price_pct / Decimal("100") * face_value
    schedule = _build_schedule(face_value, coupon_rate, days_to_maturity, coupon_frequency)

    # Initial guess: standard YTM approximation formula
    years = Decimal(days_to_maturity) / Decimal("365")
    annual_coupon = coupon_rate * face_value
    avg_price = (face_value + actual_price) / Decimal("2")
    ytm = (annual_coupon + (face_value - actual_price) / years) / avg_price

    for _iteration in range(_MAX_ITERATIONS):
        base = Decimal("1") + ytm / cf
        pv = Decimal(0)
        dpv_dy = Decimal(0)

        for d_i, cashflow in schedule:
            discount = _decimal_pow(base, -d_i)
            pv += cashflow * discount
            # d/dy [(1+y/cf)^{-d_i}] = -(d_i/cf) * (1+y/cf)^{-d_i - 1}
            dpv_dy += -(d_i / cf) * cashflow * discount / base

        delta = (pv - actual_price) / dpv_dy
        ytm -= delta

        if abs(delta) < _TOLERANCE:
            return ytm

    raise ConvergenceError(f"YTM did not converge in {_MAX_ITERATIONS} iterations")


def compute_discount_factor(ytm: Decimal, days_to_maturity: int) -> Decimal:
    """Compute df = 1 / (1 + ytm)^(days / 365) using annual compounding.

    Parameters
    ----------
    ytm:
        Annual yield (non-negative).
    days_to_maturity:
        Days to maturity (non-negative).  Returns ``Decimal("1")`` when 0.
    """
    if ytm < Decimal(0):
        raise InvalidInputError(f"ytm must be non-negative, got {ytm}")
    if days_to_maturity < 0:
        raise InvalidInputError(f"days_to_maturity must be non-negative, got {days_to_maturity}")

    exponent = Decimal(days_to_maturity) / Decimal("365")
    base = Decimal("1") + ytm
    return _decimal_pow(base, -exponent)


def compute_macaulay_duration(
    face_value: Decimal,
    coupon_rate: Decimal,
    ytm: Decimal,
    days_to_maturity: int,
    coupon_frequency: int = 2,
) -> Decimal:
    """Compute Macaulay duration in years.

    Returns the present-value-weighted average time to cash flows.
    """
    if days_to_maturity <= 0:
        raise InvalidInputError(f"days_to_maturity must be positive, got {days_to_maturity}")

    cf = Decimal(coupon_frequency)
    base = Decimal("1") + ytm / cf
    schedule = _build_schedule(face_value, coupon_rate, days_to_maturity, coupon_frequency)

    total_pv = Decimal(0)
    weighted_pv = Decimal(0)

    for d_i, cashflow in schedule:
        discount = _decimal_pow(base, -d_i)
        pv_i = cashflow * discount
        t_i = d_i / cf  # time in years
        total_pv += pv_i
        weighted_pv += t_i * pv_i

    return weighted_pv / total_pv


def compute_modified_duration(
    macaulay_duration: Decimal,
    ytm: Decimal,
    coupon_frequency: int = 2,
) -> Decimal:
    """Compute modified duration from Macaulay duration.

    ``modified = macaulay / (1 + ytm / coupon_frequency)``
    """
    return macaulay_duration / (Decimal("1") + ytm / Decimal(coupon_frequency))


def compute_dv01(
    face_value: Decimal,
    modified_duration: Decimal,
    price_pct: Decimal,
) -> Decimal:
    """Compute DV01: dollar change in bond price for a 1 bp rise in yield.

    ``DV01 = modified_duration * (price_pct / 100) * face_value * 0.0001``
    """
    return modified_duration * (price_pct / Decimal("100")) * face_value * Decimal("0.0001")


def compute_convexity(
    face_value: Decimal,
    coupon_rate: Decimal,
    ytm: Decimal,
    days_to_maturity: int,
    coupon_frequency: int = 2,
) -> Decimal:
    """Compute convexity in years².

    ``C = sum(t * (t + 1/cf) * PV(CF_t)) / (Price * (1 + ytm/cf)^2)``
    """
    if days_to_maturity <= 0:
        raise InvalidInputError(f"days_to_maturity must be positive, got {days_to_maturity}")

    cf = Decimal(coupon_frequency)
    base = Decimal("1") + ytm / cf
    schedule = _build_schedule(face_value, coupon_rate, days_to_maturity, coupon_frequency)

    total_pv = Decimal(0)
    weighted_sum = Decimal(0)

    for d_i, cashflow in schedule:
        discount = _decimal_pow(base, -d_i)
        pv_i = cashflow * discount
        t_i = d_i / cf  # time in years
        total_pv += pv_i
        weighted_sum += t_i * (t_i + Decimal("1") / cf) * pv_i

    return weighted_sum / (total_pv * base**2)
