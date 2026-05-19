from pathlib import Path

import yaml


def load_config(config_path: str | Path = "configs/config.yaml") -> dict:
    """Load project configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Parsed configuration dictionary.
    """
    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)
