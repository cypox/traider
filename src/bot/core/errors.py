"""Core exception hierarchy for the trading framework."""


class TradingError(Exception):
    """Base exception for all trading framework errors."""


class CurrencyMismatchError(TradingError):
    """Raised when arithmetic is attempted on Money values with different currencies."""


class InvalidInstrumentError(TradingError):
    """Raised when an Instrument is constructed with invalid field values."""


class InvalidSignalError(TradingError):
    """Raised when a Signal is constructed with invalid field values."""


class InvalidForecastError(TradingError):
    """Raised when a Forecast is constructed with invalid field values."""


class InvalidIntentError(TradingError):
    """Raised when an ExecutionIntent is constructed with invalid field values."""


class ConfigError(TradingError):
    """Raised when configuration loading or parsing fails."""
