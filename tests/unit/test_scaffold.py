"""Structural tests verifying the repository scaffold is correctly assembled."""

import importlib
from pathlib import Path

import pytest

from bot.config.errors import ConfigError
from bot.config.loader import load_config

_PACKAGES = [
    "bot",
    "bot.core",
    "bot.events",
    "bot.providers",
    "bot.providers.ibkr",
    "bot.providers.hyperliquid",
    "bot.providers.mock",
    "bot.portfolio",
    "bot.risk",
    "bot.signals",
    "bot.strategies",
    "bot.strategies.diagnostic",
    "bot.strategies.trend",
    "bot.strategies.carry",
    "bot.strategies.momentum",
    "bot.strategies.crypto",
    "bot.execution",
    "bot.backtest",
    "bot.analytics",
    "bot.monitoring",
    "bot.storage",
    "bot.cli",
    "bot.config",
]


@pytest.mark.parametrize("package_name", _PACKAGES)
def test_all_packages_importable(package_name: str) -> None:
    importlib.import_module(package_name)


def test_default_json_parses() -> None:
    config = load_config()
    assert isinstance(config, dict)


def test_ibkr_port_is_integer() -> None:
    config = load_config()
    assert isinstance(config["ibkr"]["port"], int)


def test_load_config_returns_required_keys() -> None:
    config = load_config()
    assert {"ibkr", "logging", "risk", "backtest"}.issubset(config.keys())


def test_load_config_nonexistent_raises_config_error() -> None:
    with pytest.raises(ConfigError):
        load_config(Path("/nonexistent/path/config.toml"))
