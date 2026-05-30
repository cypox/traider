"""IBKR metadata provider via ib_async."""

from datetime import date, datetime
from decimal import Decimal

import structlog
from ib_async import IB
from ib_async import ContractDetails as IbContractDetails
from ib_async import Stock as IbStock

from bot.core.instruments import AssetClass, Instrument
from bot.core.money import Money
from bot.events.bus import EventBus
from bot.providers.base import MetadataProvider
from bot.providers.errors import InstrumentNotFoundError
from bot.providers.ibkr.market_data import _instrument_to_contract
from bot.providers.models import ContractDetails

_logger = structlog.get_logger(__name__)


def _parse_maturity(maturity_str: str) -> date:
    """Parse a maturity date string in YYYYMMDD or YYYY-MM-DD format."""
    try:
        return date.fromisoformat(maturity_str)
    except ValueError:
        return datetime.strptime(maturity_str, "%Y%m%d").date()


class IBKRMetadataProvider(MetadataProvider):
    """Retrieves instrument and contract metadata from IBKR via ib_async."""

    def __init__(self, ib: IB, event_bus: EventBus) -> None:
        self._ib = ib
        self._event_bus = event_bus

    async def get_instrument(self, symbol: str, exchange: str) -> Instrument:
        """Look up an equity instrument by symbol and exchange."""
        results: list[IbContractDetails] = await self._ib.reqContractDetailsAsync(
            IbStock(symbol, exchange)
        )
        if not results:
            raise InstrumentNotFoundError(f"instrument not found: {symbol!r} on {exchange!r}")
        cd = results[0]
        contract = cd.contract
        if contract is None:
            raise InstrumentNotFoundError(f"instrument not found: {symbol!r} on {exchange!r}")
        return Instrument(
            symbol=contract.symbol,
            asset_class=AssetClass.EQUITY,
            currency=contract.currency,
            exchange=contract.exchange or exchange,
        )

    async def get_contract_details(self, instrument: Instrument) -> ContractDetails:
        """Retrieve full contract details for an instrument."""
        ib_contract = _instrument_to_contract(instrument)
        results: list[IbContractDetails] = await self._ib.reqContractDetailsAsync(ib_contract)
        if not results:
            raise InstrumentNotFoundError(f"contract details not found for {instrument.symbol!r}")

        # Deduplicate by conId to avoid processing the same contract twice.
        seen_con_ids: set[int] = set()
        unique_results: list[IbContractDetails] = []
        for cd in results:
            if cd.contract is None:
                continue
            con_id = cd.contract.conId
            if con_id not in seen_con_ids:
                seen_con_ids.add(con_id)
                unique_results.append(cd)

        if not unique_results:
            raise InstrumentNotFoundError(f"contract details not found for {instrument.symbol!r}")

        # For bonds (or any multi-result case), pick nearest future maturity.
        if len(unique_results) > 1:
            today = date.today()
            future_results = [
                cd for cd in unique_results if cd.maturity and _parse_maturity(cd.maturity) >= today
            ]
            pool = future_results if future_results else unique_results
            selected = min(
                pool,
                key=lambda cd: _parse_maturity(cd.maturity) if cd.maturity else date.max,
            )
        else:
            selected = unique_results[0]

        # Parse the contract multiplier (empty string means no multiplier → 1).
        raw_contract = selected.contract
        multiplier_str = raw_contract.multiplier if raw_contract is not None else ""
        multiplier = Decimal(str(float(multiplier_str))) if multiplier_str else Decimal("1")

        # Bond-specific fields.
        if instrument.asset_class == AssetClass.BOND:
            coupon: Decimal | None = Decimal(str(selected.coupon))
            maturity_date: date | None = (
                _parse_maturity(selected.maturity) if selected.maturity else date.max
            )
            face_value: Money | None = Money(Decimal("1000"), instrument.currency)
        else:
            coupon = None
            maturity_date = None
            face_value = None

        return ContractDetails(
            instrument=instrument,
            full_name=selected.longName,
            coupon=coupon,
            maturity_date=maturity_date,
            face_value=face_value,
            tick_size=Decimal(str(selected.minTick)),
            multiplier=multiplier,
        )
