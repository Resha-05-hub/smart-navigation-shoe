"""Data model definitions for system objects, sensors, and analysis."""

from dataclasses import dataclass, field
from typing import List, Optional
import time

from .enums import ObstacleZone, RiskLevel, DirectionCommand, SensorStatus


@dataclass
class BoundingBox:
    """Bounding box coordinates (normalized 0.0 to 1.0 or pixel space)."""
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def center_x(self) -> float:
        """Calculate center X coordinate."""
        return (self.xmin + self.xmax) / 2.0

    @property
    def center_y(self) -> float:
        """Calculate center Y coordinate."""
        return (self.ymin + self.ymax) / 2.0

    @property
    def width(self) -> float:
        """Calculate bounding box width."""
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        """Calculate bounding box height."""
        return self.ymax - self.ymin


@dataclass
class DetectionItem:
    """Single object detection output from vision model."""
    label: str
    confidence: float
    bbox: BoundingBox
    zone: ObstacleZone = ObstacleZone.UNKNOWN
    estimated_distance_m: float = 0.0
    track_id: int = -1


@dataclass
class SensorReading:
    """Reading output from a hardware or simulated distance sensor."""
    distance_m: float
    sensor_id: str = "primary_distance"
    position: ObstacleZone = ObstacleZone.CENTER
    status: SensorStatus = SensorStatus.HEALTHY
    unit: str = "m"
    timestamp: float = field(default_factory=time.time)
    is_valid: bool = True

    @property
    def distance_cm(self) -> float:
        """Calculates distance in centimeters."""
        return round(self.distance_m * 100.0, 2)


@dataclass
class FusedObstacle:
    """Obstacle object combining computer vision detection and range sensor readings."""
    object_id: str
    label: str
    confidence: float
    bbox: Optional[BoundingBox]
    distance_m: float
    zone: ObstacleZone
    risk_level: RiskLevel = RiskLevel.SAFE
    timestamp: float = field(default_factory=time.time)

    @property
    def object_name(self) -> str:
        """Alias for label."""
        return self.label

    @property
    def sensor_distance_cm(self) -> float:
        """Sensor distance in centimeters."""
        return round(self.distance_m * 100.0, 2)

    def format_display(self) -> str:
        """Formats single fused obstacle summary."""
        if self.confidence <= 0.0:
            conf_pct = "N/A (sensor only)"
        else:
            conf_pct = f"{int(self.confidence * 100)}%" if self.confidence <= 1.0 else f"{self.confidence}%"
        return (
            "----------------------------------------\n"
            "SENSOR FUSION RESULT\n"
            "----------------------------------------\n"
            f"Object      : {self.label}\n"
            f"Confidence  : {conf_pct}\n"
            f"Zone        : {self.zone.value}\n"
            f"Distance    : {int(self.sensor_distance_cm)} cm ({self.distance_m:.2f} m)\n"
            f"Risk        : {self.risk_level.value}\n"
            "----------------------------------------"
        )


@dataclass
class RiskAssessment:
    """System-wide risk assessment summary."""
    overall_risk_level: RiskLevel
    critical_obstacles: List[FusedObstacle] = field(default_factory=list)
    recommended_action: str = ""
    timestamp: float = field(default_factory=time.time)
    faulty_zones: List[ObstacleZone] = field(default_factory=list)  # Zones whose distance sensor failed


@dataclass
class DirectionGuidance:
    """Spatial directional advice for user navigation."""
    recommended_direction: DirectionCommand
    clear_path_score: float = 1.0
    safety_notes: str = ""
    timestamp: float = field(default_factory=time.time)
