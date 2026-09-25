"""Data structures for dashboard state representation."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Union, Dict
import time

from src.core.models import FusedObstacle, SensorReading
from src.core.enums import RiskLevel, ObstacleZone, SensorStatus


@dataclass
class DetectionSummary:
    """Summary of a single detected and fused obstacle."""
    object_name: str
    confidence: float
    zone: str
    distance_cm: float
    risk_level: str

    @property
    def confidence_pct_str(self) -> str:
        """Formatted percentage string, e.g., '87%'."""
        if self.confidence <= 1.0:
            return f"{int(round(self.confidence * 100))}%"
        return f"{int(round(self.confidence))}%"


@dataclass
class SensorStatusSummary:
    """Summary of a physical or simulated distance sensor."""
    name: str  # "LEFT", "CENTER", "RIGHT"
    distance_cm: float
    health_status: str  # "HEALTHY", "INVALID"
    is_simulated: bool = True


@dataclass
class RecentEvent:
    """Summary of a historical logged detection event."""
    timestamp_str: str  # "10:21:04"
    object_name: str
    zone: str
    distance_cm: float
    risk_level: str


@dataclass
class SystemStatus:
    """System component status metrics."""
    camera_status: str = "CONNECTED"
    yolo_status: str = "ACTIVE"
    sensors_status: str = "ACTIVE (SIMULATED)"


@dataclass
class AlertStatusSummary:
    """Current alert system state summary."""
    vibration_action: str = "OFF"
    voice_message: str = "Path clear. Safe to proceed."


@dataclass
class DashboardSnapshot:
    """Complete snapshot of the system state for rendering."""
    system_status: SystemStatus = field(default_factory=SystemStatus)
    detections: List[DetectionSummary] = field(default_factory=list)
    alert_status: AlertStatusSummary = field(default_factory=AlertStatusSummary)
    sensors: List[SensorStatusSummary] = field(default_factory=list)
    recent_events: List[RecentEvent] = field(default_factory=list)
    overall_risk: str = "SAFE"
    timestamp: datetime = field(default_factory=datetime.now)

    @classmethod
    def from_pipeline_results(
        cls,
        fused_obstacles: List[FusedObstacle],
        sensor_readings: Union[List[SensorReading], Dict[str, SensorReading]],
        overall_risk: Union[str, RiskLevel],
        vibration_action: str = "OFF",
        voice_message: str = "Path clear.",
        recent_events: Optional[List[RecentEvent]] = None,
        camera_status: str = "CONNECTED",
        yolo_status: str = "ACTIVE",
        sensors_status: str = "ACTIVE (SIMULATED)",
    ) -> "DashboardSnapshot":
        """Factory method to construct snapshot from raw pipeline outputs."""
        risk_str = overall_risk.value if isinstance(overall_risk, RiskLevel) else str(overall_risk)

        # Build Detections List
        detections_list: List[DetectionSummary] = []
        for obs in fused_obstacles:
            z_str = obs.zone.value if isinstance(obs.zone, ObstacleZone) else str(obs.zone)
            r_str = (
                obs.risk_level.value
                if isinstance(obs.risk_level, RiskLevel)
                else str(obs.risk_level)
            )
            dist_cm = round(obs.distance_m * 100.0, 1)
            detections_list.append(
                DetectionSummary(
                    object_name=obs.label,
                    confidence=obs.confidence,
                    zone=z_str,
                    distance_cm=dist_cm,
                    risk_level=r_str,
                )
            )

        # Build Sensors List
        sensors_list: List[SensorStatusSummary] = []
        readings_iter = (
            sensor_readings.values() if isinstance(sensor_readings, dict) else sensor_readings
        )
        for r in readings_iter:
            pos_str = r.position.value if isinstance(r.position, ObstacleZone) else str(r.position)
            stat_str = "HEALTHY" if r.is_valid else "INVALID"
            dist_cm = round(r.distance_m * 100.0, 1) if r.is_valid else 0.0
            sensors_list.append(
                SensorStatusSummary(
                    name=pos_str,
                    distance_cm=dist_cm,
                    health_status=stat_str,
                    is_simulated=True,
                )
            )

        sys_status = SystemStatus(
            camera_status=camera_status,
            yolo_status=yolo_status,
            sensors_status=sensors_status,
        )

        alert_summary = AlertStatusSummary(
            vibration_action=vibration_action,
            voice_message=voice_message,
        )

        return cls(
            system_status=sys_status,
            detections=detections_list,
            alert_status=alert_summary,
            sensors=sensors_list,
            recent_events=recent_events or [],
            overall_risk=risk_str,
            timestamp=datetime.now(),
        )
