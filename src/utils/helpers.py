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

    if yaml is not None:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config or {}

    # Fallback basic YAML parser if PyYAML is not installed yet
    fallback_config: Dict[str, Any] = {
        "system": {"mode": "simulation", "environment": "windows_laptop"},
        "camera": {"type": "webcam", "device_id": 0, "width": 640, "height": 480, "fps": 30},
        "detection": {"model_path": "models/yolov8n.pt", "confidence_threshold": 0.5, "device": "cpu"},
        "sensors": {"distance_sensor": {"type": "simulated", "min_distance_m": 0.2, "max_distance_m": 4.0, "simulated_default_m": 2.5}},
        "risk_analysis": {"critical_distance_m": 0.8, "warning_distance_m": 1.8, "safe_distance_m": 3.0},
        "direction": {"clear_path_threshold": 2.0},
        "alerts": {"voice": {"enabled": True}, "vibration": {"type": "simulated", "enabled": True}},
        "logging": {"level": "INFO", "format": "%(asctime)s - [%(name)s] - %(levelname)s - %(message)s"},
    }
    return fallback_config


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
