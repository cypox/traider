"""Unit tests for src/bot/core/ domain models."""

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from bot.core.errors import (
    ConfigError,
    CurrencyMismatchError,
    InvalidForecastError,
    InvalidInstrumentError,
    InvalidIntentError,
    InvalidSignalError,
    TradingError,
)
from bot.core.execution import ExecutionIntent
from bot.core.instruments import AssetClass, Instrument
from bot.core.money import Money
from bot.core.positions import ApprovedPosition, Position, TargetPosition
from bot.core.signals import Direction, Forecast, Signal

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_UTC_NOW = datetime.now(tz=UTC)


def _instr(
    symbol: str = "SPY",
    exchange: str = "SMART",
    asset_class: AssetClass = AssetClass.EQUITY,
    currency: str = "USD",
) -> Instrument:
    return Instrument(symbol=symbol, asset_class=asset_class, currency=currency, exchange=exchange)


def _forecast(instrument: Instrument | None = None, value: float = 0.5) -> Forecast:
    return Forecast(
        instrument=instrument or _instr(),
        timestamp=_UTC_NOW,
        value=value,
        source="test_strategy",
    )


def _target(instrument: Instrument | None = None) -> TargetPosition:
    instr = instrument or _instr()
    return TargetPosition(
        instrument=instr,
        target_quantity=Decimal("10"),
        target_notional=Money(Decimal("1000"), "USD"),
        source_forecast=_forecast(instr),
    )


def _approved(instrument: Instrument | None = None) -> ApprovedPosition:
    instr = instrument or _instr()
    return ApprovedPosition(
        instrument=instr,
        approved_quantity=Decimal("10"),
        original_target=_target(instr),
        risk_notes="approved",
    )


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------


def test_trading_error_is_exception() -> None:
    assert issubclass(TradingError, Exception)


def test_currency_mismatch_error_is_trading_error() -> None:
    assert issubclass(CurrencyMismatchError, TradingError)


def test_invalid_instrument_error_is_trading_error() -> None:
    assert issubclass(InvalidInstrumentError, TradingError)


def test_invalid_signal_error_is_trading_error() -> None:
    assert issubclass(InvalidSignalError, TradingError)


def test_invalid_forecast_error_is_trading_error() -> None:
    assert issubclass(InvalidForecastError, TradingError)


def test_invalid_intent_error_is_trading_error() -> None:
    assert issubclass(InvalidIntentError, TradingError)


def test_config_error_is_trading_error() -> None:
    assert issubclass(ConfigError, TradingError)


# ---------------------------------------------------------------------------
# Instrument
# ---------------------------------------------------------------------------


def test_instrument_str() -> None:
    assert str(_instr()) == "SPY@SMART"


def test_instrument_empty_symbol_raises() -> None:
    with pytest.raises(InvalidInstrumentError):
        Instrument(symbol="", asset_class=AssetClass.EQUITY, currency="USD", exchange="SMART")


def test_instrument_empty_currency_raises() -> None:
    with pytest.raises(InvalidInstrumentError):
        Instrument(symbol="SPY", asset_class=AssetClass.EQUITY, currency="", exchange="SMART")


def test_instrument_empty_exchange_raises() -> None:
    with pytest.raises(InvalidInstrumentError):
        Instrument(symbol="SPY", asset_class=AssetClass.EQUITY, currency="USD", exchange="")


def test_instrument_eq_same_symbol_same_exchange() -> None:
    # currency differs but (symbol, exchange) matches → equal
    a = Instrument("SPY", AssetClass.EQUITY, "USD", "SMART")
    b = Instrument("SPY", AssetClass.BOND, "EUR", "SMART")
    assert a == b


def test_instrument_eq_same_symbol_different_exchange() -> None:
    a = Instrument("SPY", AssetClass.EQUITY, "USD", "SMART")
    b = Instrument("SPY", AssetClass.EQUITY, "USD", "NYSE")
    assert a != b


def test_instrument_hash_equal_instruments() -> None:
    a = Instrument("SPY", AssetClass.EQUITY, "USD", "SMART")
    b = Instrument("SPY", AssetClass.BOND, "EUR", "SMART")
    assert hash(a) == hash(b)
    assert len({a, b}) == 1


def test_instrument_hash_different_instruments() -> None:
    a = _instr(exchange="SMART")
    b = _instr(exchange="NYSE")
    assert hash(a) != hash(b)


def test_instrument_eq_non_instrument_returns_not_implemented() -> None:
    instr = _instr()
    result = instr.__eq__("not_an_instrument")
    assert result is NotImplemented


def test_asset_class_values() -> None:
    assert AssetClass.EQUITY.value == "EQUITY"
    assert AssetClass.BOND.value == "BOND"
    assert AssetClass.FUTURE.value == "FUTURE"
    assert AssetClass.CRYPTO.value == "CRYPTO"
    assert AssetClass.FX.value == "FX"


# ---------------------------------------------------------------------------
# Money
# ---------------------------------------------------------------------------


def test_money_add_same_currency() -> None:
    assert Money(Decimal("1"), "USD") + Money(Decimal("2"), "USD") == Money(Decimal("3"), "USD")


def test_money_add_currency_mismatch_raises() -> None:
    with pytest.raises(CurrencyMismatchError):
        Money(Decimal("1"), "USD") + Money(Decimal("1"), "EUR")


def test_money_sub_same_currency() -> None:
    assert Money(Decimal("5"), "USD") - Money(Decimal("3"), "USD") == Money(Decimal("2"), "USD")


def test_money_sub_currency_mismatch_raises() -> None:
    with pytest.raises(CurrencyMismatchError):
        Money(Decimal("5"), "USD") - Money(Decimal("3"), "EUR")


def test_money_mul_scalar() -> None:
    assert Money(Decimal("10"), "USD") * Decimal("3") == Money(Decimal("30"), "USD")


def test_money_neg() -> None:
    assert -Money(Decimal("10"), "USD") == Money(Decimal("-10"), "USD")


def test_money_abs_negative() -> None:
    assert abs(Money(Decimal("-10"), "USD")) == Money(Decimal("10"), "USD")


def test_money_abs_positive() -> None:
    assert abs(Money(Decimal("10"), "USD")) == Money(Decimal("10"), "USD")


def test_money_is_positive() -> None:
    assert Money(Decimal("1"), "USD").is_positive is True
    assert Money(Decimal("-1"), "USD").is_positive is False
    assert Money(Decimal("0"), "USD").is_positive is False


def test_money_is_negative() -> None:
    assert Money(Decimal("-1"), "USD").is_negative is True
    assert Money(Decimal("1"), "USD").is_negative is False
    assert Money(Decimal("0"), "USD").is_negative is False


def test_money_is_zero() -> None:
    assert Money(Decimal("0"), "USD").is_zero is True
    assert Money(Decimal("1"), "USD").is_zero is False


def test_money_zero_classmethod() -> None:
    z = Money.zero("EUR")
    assert z.amount == Decimal(0)
    assert z.currency == "EUR"
    assert z.is_zero


@given(
    a=st.decimals(allow_nan=False, allow_infinity=False),
    b=st.decimals(allow_nan=False, allow_infinity=False),
)
def test_money_addition_commutativity(a: Decimal, b: Decimal) -> None:
    assert Money(a, "USD") + Money(b, "USD") == Money(a + b, "USD")


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------


def test_signal_valid_construction() -> None:
    sig = Signal(
        instrument=_instr(),
        timestamp=_UTC_NOW,
        direction=Direction.LONG,
        confidence=0.8,
        reason="momentum breakout",
    )
    assert sig.direction == Direction.LONG
    assert sig.confidence == 0.8


def test_signal_confidence_zero_raises() -> None:
    with pytest.raises(InvalidSignalError):
        Signal(
            instrument=_instr(),
            timestamp=_UTC_NOW,
            direction=Direction.LONG,
            confidence=0.0,
            reason="test",
        )


def test_signal_confidence_above_one_raises() -> None:
    with pytest.raises(InvalidSignalError):
        Signal(
            instrument=_instr(),
            timestamp=_UTC_NOW,
            direction=Direction.LONG,
            confidence=1.01,
            reason="test",
        )


def test_signal_confidence_one_is_valid() -> None:
    sig = Signal(
        instrument=_instr(),
        timestamp=_UTC_NOW,
        direction=Direction.SHORT,
        confidence=1.0,
        reason="max conviction",
    )
    assert sig.confidence == 1.0


def test_signal_flat_direction_raises() -> None:
    with pytest.raises(InvalidSignalError):
        Signal(
            instrument=_instr(),
            timestamp=_UTC_NOW,
            direction=Direction.FLAT,
            confidence=0.5,
            reason="test",
        )


def test_signal_empty_reason_raises() -> None:
    with pytest.raises(InvalidSignalError):
        Signal(
            instrument=_instr(),
            timestamp=_UTC_NOW,
            direction=Direction.LONG,
            confidence=0.5,
            reason="",
        )


def test_signal_naive_timestamp_raises() -> None:
    with pytest.raises(InvalidSignalError):
        Signal(
            instrument=_instr(),
            timestamp=datetime.now(),  # naive
            direction=Direction.LONG,
            confidence=0.5,
            reason="test",
        )


# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------


def test_forecast_valid_extremes() -> None:
    f_plus = _forecast(value=1.0)
    f_minus = _forecast(value=-1.0)
    assert f_plus.value == 1.0
    assert f_minus.value == -1.0


def test_forecast_zero_value_valid() -> None:
    f = _forecast(value=0.0)
    assert f.value == 0.0


def test_forecast_above_one_raises() -> None:
    with pytest.raises(InvalidForecastError):
        _forecast(value=1.01)


def test_forecast_below_minus_one_raises() -> None:
    with pytest.raises(InvalidForecastError):
        _forecast(value=-1.01)


# ---------------------------------------------------------------------------
# Position
# ---------------------------------------------------------------------------


def test_position_direction_long() -> None:
    pos = Position(
        instrument=_instr(),
        quantity=Decimal("10"),
        average_price=Money(Decimal("100"), "USD"),
    )
    assert pos.direction == Direction.LONG
    assert not pos.is_flat


def test_position_direction_short() -> None:
    pos = Position(
        instrument=_instr(),
        quantity=Decimal("-5"),
        average_price=Money(Decimal("200"), "USD"),
    )
    assert pos.direction == Direction.SHORT
    assert not pos.is_flat


def test_position_direction_flat() -> None:
    pos = Position(
        instrument=_instr(),
        quantity=Decimal("0"),
        average_price=Money(Decimal("100"), "USD"),
    )
    assert pos.direction == Direction.FLAT
    assert pos.is_flat


def test_position_market_value() -> None:
    pos = Position(
        instrument=_instr(),
        quantity=Decimal("10"),
        average_price=Money(Decimal("100"), "USD"),
    )
    mv = pos.market_value(Money(Decimal("110"), "USD"))
    assert mv == Money(Decimal("1100"), "USD")


def test_position_unrealized_pnl() -> None:
    pos = Position(
        instrument=_instr(),
        quantity=Decimal("10"),
        average_price=Money(Decimal("100"), "USD"),
    )
    pnl = pos.unrealized_pnl(Money(Decimal("110"), "USD"))
    assert pnl == Money(Decimal("100"), "USD")


# ---------------------------------------------------------------------------
# TargetPosition
# ---------------------------------------------------------------------------


def test_target_position_construction() -> None:
    instr = _instr()
    fc = _forecast(instr)
    target = TargetPosition(
        instrument=instr,
        target_quantity=Decimal("5"),
        target_notional=Money(Decimal("500"), "USD"),
        source_forecast=fc,
    )
    assert target.target_quantity == Decimal("5")
    assert target.target_notional.currency == instr.currency


# ---------------------------------------------------------------------------
# ApprovedPosition
# ---------------------------------------------------------------------------


def test_approved_position_quantity_less_than_target() -> None:
    instr = _instr()
    tgt = _target(instr)
    approved = ApprovedPosition(
        instrument=instr,
        approved_quantity=Decimal("3"),
        original_target=tgt,
        risk_notes="reduced due to concentration limit",
    )
    assert approved.approved_quantity < tgt.target_quantity


# ---------------------------------------------------------------------------
# ExecutionIntent
# ---------------------------------------------------------------------------


def test_execution_intent_valid() -> None:
    instr = _instr()
    intent = ExecutionIntent(
        instrument=instr,
        side=Direction.LONG,
        quantity=Decimal("10"),
        reason="entry signal",
        source_approved=_approved(instr),
    )
    assert intent.side == Direction.LONG
    assert intent.quantity == Decimal("10")


def test_execution_intent_flat_side_raises() -> None:
    instr = _instr()
    with pytest.raises(InvalidIntentError):
        ExecutionIntent(
            instrument=instr,
            side=Direction.FLAT,
            quantity=Decimal("10"),
            reason="test",
            source_approved=_approved(instr),
        )


def test_execution_intent_zero_quantity_raises() -> None:
    instr = _instr()
    with pytest.raises(InvalidIntentError):
        ExecutionIntent(
            instrument=instr,
            side=Direction.LONG,
            quantity=Decimal("0"),
            reason="test",
            source_approved=_approved(instr),
        )


def test_execution_intent_negative_quantity_raises() -> None:
    instr = _instr()
    with pytest.raises(InvalidIntentError):
        ExecutionIntent(
            instrument=instr,
            side=Direction.SHORT,
            quantity=Decimal("-1"),
            reason="test",
            source_approved=_approved(instr),
        )


def test_execution_intent_empty_reason_raises() -> None:
    instr = _instr()
    with pytest.raises(InvalidIntentError):
        ExecutionIntent(
            instrument=instr,
            side=Direction.LONG,
            quantity=Decimal("5"),
            reason="",
            source_approved=_approved(instr),
        )
