"""Helper and utility functions for configuration loading and spatial calculations."""

from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except ImportError:
    yaml = None

from ..core.enums import ObstacleZone


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Loads configuration parameters from a YAML file.

    Args:
        config_path: Relative or absolute path to the config YAML file.

    Returns:
        Dictionary containing configuration properties.
    """
    path = Path(config_path)
    if not path.is_absolute():
        base_dir = Path(__file__).resolve().parent.parent.parent
        path = base_dir / config_path

    if not path.exists():
        return {}

    if yaml is None:
        # Previously a hard-coded fallback with different risk thresholds was returned silently
        raise ImportError(
            "PyYAML is required to read the configuration. Install dependencies with: "
            "venvv\\Scripts\\python.exe -m pip install -r requirements.txt"
        )

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config or {}


def determine_zone(
    center_x_norm: float,
    left_boundary: float = 0.33,
    right_boundary: float = 0.66,
) -> ObstacleZone:
    """Determines spatial obstacle zone (LEFT, CENTER, RIGHT) based on normalized X coordinate.

    Args:
        center_x_norm: Normalized horizontal center coordinate (0.0 to 1.0).
        left_boundary: Upper bound ratio for LEFT zone.
        right_boundary: Lower bound ratio for RIGHT zone.

    Returns:
        ObstacleZone enum value.
    """
    if center_x_norm < left_boundary:
        return ObstacleZone.LEFT
    elif center_x_norm <= right_boundary:
        return ObstacleZone.CENTER
    else:
        return ObstacleZone.RIGHT
