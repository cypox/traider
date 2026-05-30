import tomllib
from pathlib import Path
from typing import Any

from bot.config.errors import ConfigError

_DEFAULT_CONFIG_PATH = Path(__file__).parents[3] / "config" / "default.toml"


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load configuration from a TOML file.

    Args:
        path: Path to the TOML config file. Defaults to config/default.toml
              relative to the repository root.

    Returns:
        Parsed configuration as a nested dictionary.

    Raises:
        ConfigError: If the file is not found or cannot be parsed.

    """
    resolved = path if path is not None else _DEFAULT_CONFIG_PATH
    try:
        with resolved.open("rb") as fh:
            return tomllib.load(fh)
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file not found: {resolved}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Failed to parse config file: {resolved}") from exc
