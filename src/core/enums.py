"""Enumerations used across the Smart Navigation Shoe system."""

from enum import Enum


class ObstacleZone(str, Enum):
    """Spatial zone relative to the user's direction of movement."""
    LEFT = "LEFT"
    CENTER = "CENTER"
    RIGHT = "RIGHT"
    UNKNOWN = "UNKNOWN"


class RiskLevel(str, Enum):
    """Risk severity evaluation level."""
    SAFE = "SAFE"
    CAUTION = "CAUTION"
    WARNING = "WARNING"
    DANGER = "DANGER"

    # Additional severity levels for architectural compatibility
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AlertPattern(str, Enum):
    """Haptic/Vibration alert pattern types."""
    NONE = "NONE"
    LOW_PULSE = "LOW_PULSE"
    MEDIUM_PULSE = "MEDIUM_PULSE"
    CONTINUOUS_HIGH = "CONTINUOUS_HIGH"


class CameraStatus(str, Enum):
    """Camera operational states."""
    UNINITIALIZED = "UNINITIALIZED"
    CONNECTED = "CONNECTED"
    DISCONNECTED = "DISCONNECTED"
    ERROR = "ERROR"


class SensorStatus(str, Enum):
    """Operational status of a distance sensor."""
    HEALTHY = "HEALTHY"
    INVALID = "INVALID"
    OUT_OF_RANGE = "OUT_OF_RANGE"


class DirectionCommand(str, Enum):
    """Recommended navigation directions for the visually impaired user."""
    MOVE_FORWARD = "MOVE_FORWARD"
    SLIGHT_LEFT = "SLIGHT_LEFT"
    SLIGHT_RIGHT = "SLIGHT_RIGHT"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    STOP = "STOP"
