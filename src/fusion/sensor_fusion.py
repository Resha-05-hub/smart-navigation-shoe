"""Sensor fusion logic combining vision object detection and distance sensor measurements."""

from typing import List, Dict, Optional
import uuid

from ..core.models import DetectionItem, SensorReading, FusedObstacle
from ..core.enums import ObstacleZone, RiskLevel
from ..utils.logger import get_logger

logger = get_logger("sensor_fusion")


class SensorFusionEngine:
    """Combines camera bounding box detections with distance sensor readings for LEFT, CENTER, and RIGHT zones.

    Prototype Principle:
    - YOLO provides object identification and image bounding box spatial zone (LEFT, CENTER, RIGHT).
    - Distance sensors supply distance measurements for each spatial zone.
    - Sensor fusion maps visual object detections to their corresponding zone's distance reading.
    - Note: YOLO does NOT compute distance values.
    """

    def __init__(self, default_range_m: float = 2.5) -> None:
        self.default_range_m = default_range_m

    def fuse(
        self,
        detections: List[DetectionItem],
        sensor_readings: List[SensorReading],
    ) -> List[FusedObstacle]:
        """Fuses visual detections and spatial zone sensor readings into FusedObstacle objects.

        Args:
            detections: List of DetectionItem objects from vision pipeline.
            sensor_readings: List of SensorReading objects from range sensors.

        Returns:
            List of FusedObstacle objects with integrated zone distance info.
        """
        fused_obstacles: List[FusedObstacle] = []

        if not detections:
            return fused_obstacles

        # Build map of spatial position -> SensorReading
        zone_sensor_map: Dict[ObstacleZone, SensorReading] = {}
        for r in sensor_readings:
            if hasattr(r, "position") and r.position:
                zone_sensor_map[r.position] = r

        # Primary fall-back distance if zone reading is missing/invalid
        primary_fallback_m = self.default_range_m
        valid_readings = [r for r in sensor_readings if r.is_valid and r.distance_m > 0]
        if valid_readings:
            primary_fallback_m = valid_readings[0].distance_m

        for idx, item in enumerate(detections):
            target_zone = item.zone
            if target_zone == ObstacleZone.UNKNOWN and item.bbox:
                # Determine zone from normalized center X if UNKNOWN
                center_x_norm = item.bbox.center_x / 640.0 if item.bbox.xmax > 1.0 else item.bbox.center_x
                if center_x_norm < 0.33:
                    target_zone = ObstacleZone.LEFT
                elif center_x_norm <= 0.66:
                    target_zone = ObstacleZone.CENTER
                else:
                    target_zone = ObstacleZone.RIGHT

            # Retrieve corresponding distance sensor reading for object's spatial zone
            sensor_reading = zone_sensor_map.get(target_zone)

            if sensor_reading and sensor_reading.is_valid and sensor_reading.distance_m > 0:
                distance_m = sensor_reading.distance_m
            else:
                distance_m = primary_fallback_m

            fused_obstacle = FusedObstacle(
                object_id=f"fused_{idx}_{uuid.uuid4().hex[:6]}",
                label=item.label,
                confidence=item.confidence,
                bbox=item.bbox,
                distance_m=round(distance_m, 2),
                zone=target_zone,
                risk_level=RiskLevel.SAFE,  # Will be assigned by RiskAnalyzer
            )
            fused_obstacles.append(fused_obstacle)

        return fused_obstacles
