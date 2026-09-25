"""Sensor fusion logic combining vision object detection and distance sensor measurements."""

from typing import List, Dict, Optional, Any
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
    - Distance sensors that report a close obstacle in a zone where the camera sees nothing
      (e.g. low ground obstacles below the camera's view) produce a sensor-only obstacle.
    - Note: YOLO does NOT compute distance values.
    """

    SENSOR_ONLY_LABEL = "obstacle"

    def __init__(
        self,
        default_range_m: float = 2.5,
        sensor_only_max_m: float = 2.0,
        hysteresis_m: float = 0.0,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.default_range_m = default_range_m
        self.sensor_only_max_m = sensor_only_max_m
        self.hysteresis_m = hysteresis_m
        self._sensor_only_zones: set = set()  # Zones that reported a sensor-only obstacle last frame

        if config:
            sensor_cfg = config.get("sensors", {}).get("distance_sensor", {})
            self.default_range_m = sensor_cfg.get("simulated_default_m", default_range_m)
            # Sensor-only obstacles are reported out to the CAUTION boundary
            risk_cfg = config.get("risk_analysis", {})
            thresh_cm = risk_cfg.get("thresholds_cm", {})
            if "caution_cm" in thresh_cm:
                self.sensor_only_max_m = thresh_cm["caution_cm"] / 100.0
            self.hysteresis_m = risk_cfg.get("hysteresis_cm", hysteresis_m * 100.0) / 100.0

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

        # Build map of spatial position -> SensorReading
        zone_sensor_map: Dict[ObstacleZone, SensorReading] = {}
        for r in sensor_readings:
            if hasattr(r, "position") and r.position:
                zone_sensor_map[r.position] = r

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

            # A missing/invalid zone sensor must not borrow another zone's distance
            if sensor_reading and sensor_reading.is_valid and sensor_reading.distance_m > 0:
                distance_m = sensor_reading.distance_m
            else:
                distance_m = self.default_range_m

            object_id = f"track_{item.track_id}" if item.track_id >= 0 else f"fused_{idx}_{uuid.uuid4().hex[:6]}"
            fused_obstacle = FusedObstacle(
                object_id=object_id,
                label=item.label,
                confidence=item.confidence,
                bbox=item.bbox,
                distance_m=round(distance_m, 2),
                zone=target_zone,
                risk_level=RiskLevel.SAFE,  # Will be assigned by RiskAnalyzer
            )
            fused_obstacles.append(fused_obstacle)

        # Close sensor readings in zones with no visual detection become unidentified obstacles.
        # An obstacle already reported stays until it is hysteresis_m beyond the range, so noise
        # around the boundary does not make it blink in and out.
        visual_zones = {obs.zone for obs in fused_obstacles}
        sensor_only_zones = set()
        for zone in (ObstacleZone.LEFT, ObstacleZone.CENTER, ObstacleZone.RIGHT):
            reading = zone_sensor_map.get(zone)
            if zone in visual_zones or reading is None or not reading.is_valid:
                continue
            max_m = self.sensor_only_max_m + (self.hysteresis_m if zone in self._sensor_only_zones else 0.0)
            if 0 < reading.distance_m <= max_m:
                sensor_only_zones.add(zone)
                fused_obstacles.append(
                    FusedObstacle(
                        object_id=f"sensor_{zone.value.lower()}_{uuid.uuid4().hex[:6]}",
                        label=self.SENSOR_ONLY_LABEL,
                        confidence=0.0,  # Not visually identified
                        bbox=None,
                        distance_m=round(reading.distance_m, 2),
                        zone=zone,
                        risk_level=RiskLevel.SAFE,  # Will be assigned by RiskAnalyzer
                    )
                )

        self._sensor_only_zones = sensor_only_zones
        return fused_obstacles
