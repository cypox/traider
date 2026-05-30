"""Shared contract conversion utilities for IBKR providers."""

from __future__ import annotations

from ib_async import Bond, Contract, Forex, Future, Stock

from bot.core.instruments import AssetClass, Instrument
from bot.providers.errors import UnsupportedAssetClassError


def instrument_to_contract(instrument: Instrument) -> Contract:
    """Convert a domain ``Instrument`` to an ib_async ``Contract``."""
    if instrument.asset_class == AssetClass.EQUITY:
        return Stock(instrument.symbol, "SMART", instrument.currency)
    if instrument.asset_class == AssetClass.FUTURE:
        return Future(
            symbol=instrument.symbol,
            exchange=instrument.exchange,
            currency=instrument.currency,
        )
    if instrument.asset_class == AssetClass.BOND:
        return Bond(symbol=instrument.symbol)  # type: ignore[no-untyped-call]
    if instrument.asset_class == AssetClass.FX:
        return Forex(instrument.symbol)
    raise UnsupportedAssetClassError(
        f"unsupported asset class: {instrument.asset_class!r} "
        "(CRYPTO is handled by the Hyperliquid provider)"
    )


def contract_to_instrument(contract: Contract) -> Instrument | None:
    """Best-effort reverse mapping from an ib_async ``Contract`` to an ``Instrument``.

    Returns ``None`` for contract types that cannot be mapped (e.g. options).
    """
    sec_type = contract.secType
    if sec_type == "STK":
        asset_class = AssetClass.EQUITY
    elif sec_type == "FUT":
        asset_class = AssetClass.FUTURE
    elif sec_type == "BOND":
        asset_class = AssetClass.BOND
    elif sec_type == "CASH":
        asset_class = AssetClass.FX
    else:
        return None

    exchange = contract.exchange if contract.exchange else "UNKNOWN"
    return Instrument(
        symbol=contract.symbol,
        asset_class=asset_class,
        currency=contract.currency,
        exchange=exchange,
    )
