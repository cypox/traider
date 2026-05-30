import json
from pathlib import Path
from typing import Any

from bot.config.errors import ConfigError

_DEFAULT_CONFIG_PATH = Path(__file__).parents[3] / "config" / "default.json"


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load configuration from a JSON file.

    Args:
        path: Path to the JSON config file. Defaults to config/default.json
              relative to the repository root.

    Returns:
        Parsed configuration as a nested dictionary.

    Raises:
        ConfigError: If the file is not found or cannot be parsed.

    """
    resolved = path if path is not None else _DEFAULT_CONFIG_PATH
    try:
        with resolved.open(encoding="utf-8") as fh:
            config: dict[str, Any] = json.load(fh)
        return config
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file not found: {resolved}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Failed to parse config file: {resolved}") from exc
