"""CLI entry point for the systematic trading bot."""

from __future__ import annotations

import argparse
import asyncio
import signal
from decimal import Decimal
from pathlib import Path
from typing import Any

import structlog

from bot.config.loader import load_config
from bot.core.instruments import AssetClass, Instrument
from bot.core.money import Money
from bot.events.bus import EventBus
from bot.execution.engine import ExecutionEngine
from bot.portfolio.construction import (
    PortfolioConstruction,
    PortfolioConstructionConfig,
    SignalCombinationMethod,
)
from bot.portfolio.state import PortfolioState
from bot.providers.ibkr.execution import IBKRExecutionProvider
from bot.providers.ibkr.market_data import IBKRMarketDataProvider
from bot.providers.ibkr.metadata import IBKRMetadataProvider
from bot.risk.config import RiskConfig
from bot.risk.engine import RiskEngine
from bot.strategies.diagnostic.strategy import DiagnosticStrategy

_logger = structlog.get_logger(__name__)

# Starter instrument list — configurable in a future prompt.
_STARTER_INSTRUMENTS: list[Instrument] = [
    Instrument(symbol="SPY", asset_class=AssetClass.EQUITY, currency="USD", exchange="SMART"),
    Instrument(symbol="ZN", asset_class=AssetClass.FUTURE, currency="USD", exchange="CME"),
    Instrument(symbol="ZT", asset_class=AssetClass.FUTURE, currency="USD", exchange="CME"),
]


def _setup_logging(config: dict[str, Any]) -> None:
    """Configure structlog based on the logging section of the config."""
    fmt: str = config.get("logging", {}).get("format", "human")

    renderer: Any
    if fmt == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            renderer,
        ],
    )


async def run(config_path: Path | None = None) -> None:
    """Wire all components and run the trading bot until interrupted."""
    config = load_config(config_path)
    _setup_logging(config)

    _logger.info(
        "Starting trading runner",
        ibkr_host=config["ibkr"]["host"],
        ibkr_port=config["ibkr"]["port"],
    )

    event_bus = EventBus()
    portfolio_state = PortfolioState(
        initial_cash=Money(Decimal("100000"), "USD"),
        event_bus=event_bus,
    )

    risk_engine = RiskEngine(
        config=RiskConfig(
            max_position_usd=Decimal(str(config["risk"]["max_position_usd"])),
            max_gross_exposure_usd=Decimal("500000"),
            max_drawdown_pct=Decimal(str(config["risk"]["max_drawdown_pct"])),
            max_concentration_pct=Decimal("0.20"),
            daily_loss_limit_usd=Decimal("5000"),
        ),
        event_bus=event_bus,
        portfolio_state=portfolio_state,
    )

    portfolio_construction = PortfolioConstruction(
        config=PortfolioConstructionConfig(
            target_annual_vol=0.15,
            vol_lookback_days=60,
            min_forecast_for_trade=0.01,
            combination_method=SignalCombinationMethod.SIMPLE_AVERAGE,
        ),
        event_bus=event_bus,
        portfolio_state=portfolio_state,
    )

    # IBKRMarketDataProvider owns the IB instance — no IB object anywhere else.
    market_data_provider = IBKRMarketDataProvider(
        host=str(config["ibkr"]["host"]),
        port=int(config["ibkr"]["port"]),
        client_id=int(config["ibkr"]["client_id"]),
        event_bus=event_bus,
    )

    # Sibling providers share the IB instance via market_data_provider.ib.
    metadata_provider = IBKRMetadataProvider(
        market_data_provider=market_data_provider,
        event_bus=event_bus,
    )

    execution_provider = IBKRExecutionProvider(
        market_data_provider=market_data_provider,
        event_bus=event_bus,
    )

    diagnostic_strategy = DiagnosticStrategy(
        event_bus=event_bus,
        metadata_provider=metadata_provider,
        instruments=list(_STARTER_INSTRUMENTS),
    )

    # Keep references so subscriptions stay alive.
    _keep_alive = (risk_engine, portfolio_construction)

    execution_engine = ExecutionEngine(
        event_bus=event_bus,
        execution_provider=execution_provider,
        portfolio_state=portfolio_state,
    )

    await market_data_provider.connect()
    await execution_provider.connect()
    await diagnostic_strategy.refresh_metadata()
    await market_data_provider.subscribe_quotes(list(_STARTER_INSTRUMENTS))
    await market_data_provider.subscribe_bars(list(_STARTER_INSTRUMENTS), interval_seconds=5)

    stop_event = asyncio.Event()

    def _request_stop() -> None:
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _request_stop)

    _logger.info("Runner ready, waiting for market data")
    await stop_event.wait()

    await execution_provider.disconnect()
    await market_data_provider.disconnect()
    _logger.info("Runner stopped cleanly")

    # Satisfy type checker — held to prevent GC of subscribed components.
    del _keep_alive, execution_engine, diagnostic_strategy


def main() -> None:
    """CLI entry point — parse arguments and start the event loop."""
    parser = argparse.ArgumentParser(
        prog="trading-bot",
        description="Systematic trading bot — connects to IBKR and runs strategies.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to JSON config file (default: config/default.json)",
    )
    args = parser.parse_args()
    config_path: Path | None = args.config

    try:
        asyncio.run(run(config_path=config_path))
    except KeyboardInterrupt:
        pass
