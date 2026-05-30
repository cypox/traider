"""Provider-specific exception hierarchy."""

from bot.core.errors import TradingError


class ProviderError(TradingError):
    """Base class for all provider errors."""


class InstrumentNotFoundError(ProviderError):
    """Requested instrument was not found in the provider catalog."""


class ConnectionError(ProviderError):  # noqa: A001
    """Provider failed to connect or the connection was lost."""


class UnsupportedAssetClassError(ProviderError):
    """Provider does not support the requested asset class."""


class OrderRejectedError(ProviderError):
    """Broker rejected the submitted order."""
