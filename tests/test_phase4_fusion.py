"""Unit tests for Phase 4 Sensor Fusion and Risk Analysis."""

import unittest

from src.fusion.sensor_fusion import SensorFusionEngine
from src.decision.risk_analyzer import RiskAnalyzer
from src.sensors.simulated_sensor import SimulatedDistanceSensor
from src.sensors.sensor_manager import SensorManager
from src.core.models import DetectionItem, BoundingBox, SensorReading, FusedObstacle
from src.core.enums import ObstacleZone, RiskLevel, SensorStatus


class TestPhase4SensorFusionAndRisk(unittest.TestCase):
    """Test suite for Phase 4 Sensor Fusion and Risk Analysis logic."""

    def setUp(self):
        self.fusion = SensorFusionEngine(default_range_m=2.5)
        self.risk_analyzer = RiskAnalyzer(
            danger_distance_m=0.5,    # < 50 cm = DANGER
            warning_distance_m=1.0,   # 50 - 100 cm = WARNING
            caution_distance_m=2.0,   # 100 - 200 cm = CAUTION
        )

    def test_left_object_mapped_to_left_sensor(self):
        """Verify LEFT zone visual object maps to LEFT distance sensor."""
        bbox_left = BoundingBox(xmin=10, ymin=10, xmax=100, ymax=200)  # Center X = 55 (< 0.33 of 640)
        item_left = DetectionItem(label="dog", confidence=0.85, bbox=bbox_left, zone=ObstacleZone.LEFT)
        sensor_left = SensorReading(distance_m=1.5, position=ObstacleZone.LEFT, is_valid=True)

        fused = self.fusion.fuse(detections=[item_left], sensor_readings=[sensor_left])

        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0].label, "dog")
        self.assertEqual(fused[0].zone, ObstacleZone.LEFT)
        self.assertEqual(fused[0].sensor_distance_cm, 150.0)

    def test_center_object_mapped_to_center_sensor(self):
        """Verify CENTER zone visual object maps to CENTER distance sensor."""
        bbox_center = BoundingBox(xmin=250, ymin=10, xmax=350, ymax=200)  # Center X = 300
        item_center = DetectionItem(label="person", confidence=0.91, bbox=bbox_center, zone=ObstacleZone.CENTER)
        sensor_center = SensorReading(distance_m=0.8, position=ObstacleZone.CENTER, is_valid=True)

        fused = self.fusion.fuse(detections=[item_center], sensor_readings=[sensor_center])

        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0].label, "person")
        self.assertEqual(fused[0].zone, ObstacleZone.CENTER)
        self.assertEqual(fused[0].sensor_distance_cm, 80.0)

    def test_right_object_mapped_to_right_sensor(self):
        """Verify RIGHT zone visual object maps to RIGHT distance sensor."""
        bbox_right = BoundingBox(xmin=500, ymin=10, xmax=600, ymax=200)  # Center X = 550
        item_right = DetectionItem(label="chair", confidence=0.88, bbox=bbox_right, zone=ObstacleZone.RIGHT)
        sensor_right = SensorReading(distance_m=2.2, position=ObstacleZone.RIGHT, is_valid=True)

        fused = self.fusion.fuse(detections=[item_right], sensor_readings=[sensor_right])

        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0].label, "chair")
        self.assertEqual(fused[0].zone, ObstacleZone.RIGHT)
        self.assertEqual(fused[0].sensor_distance_cm, 220.0)

    def test_risk_classification_ranges(self):
        """Verify risk levels for DANGER (<50cm), WARNING (50-100cm), CAUTION (100-200cm), SAFE (>200cm)."""
        bbox = BoundingBox(xmin=0.1, ymin=0.1, xmax=0.3, ymax=0.5)

        obs_danger = FusedObstacle("1", "obstacle", 0.9, bbox, distance_m=0.4, zone=ObstacleZone.CENTER)   # 40 cm
        obs_warning = FusedObstacle("2", "person", 0.91, bbox, distance_m=0.8, zone=ObstacleZone.CENTER)   # 80 cm
        obs_caution = FusedObstacle("3", "dog", 0.85, bbox, distance_m=1.5, zone=ObstacleZone.LEFT)        # 150 cm
        obs_safe = FusedObstacle("4", "chair", 0.88, bbox, distance_m=2.2, zone=ObstacleZone.RIGHT)       # 220 cm

        self.assertEqual(self.risk_analyzer.classify_obstacle_risk(obs_danger), RiskLevel.DANGER)
        self.assertEqual(self.risk_analyzer.classify_obstacle_risk(obs_warning), RiskLevel.WARNING)
        self.assertEqual(self.risk_analyzer.classify_obstacle_risk(obs_caution), RiskLevel.CAUTION)
        self.assertEqual(self.risk_analyzer.classify_obstacle_risk(obs_safe), RiskLevel.SAFE)

    def test_multiple_detections_different_zones(self):
        """Verify handling multiple detections in different spatial zones."""
        bbox_l = BoundingBox(xmin=10, ymin=10, xmax=100, ymax=200)
        bbox_c = BoundingBox(xmin=250, ymin=10, xmax=350, ymax=200)
        bbox_r = BoundingBox(xmin=500, ymin=10, xmax=600, ymax=200)

        item_l = DetectionItem("dog", 0.85, bbox_l, ObstacleZone.LEFT)
        item_c = DetectionItem("person", 0.91, bbox_c, ObstacleZone.CENTER)
        item_r = DetectionItem("chair", 0.88, bbox_r, ObstacleZone.RIGHT)

        readings = [
            SensorReading(1.5, position=ObstacleZone.LEFT, is_valid=True),
            SensorReading(0.8, position=ObstacleZone.CENTER, is_valid=True),
            SensorReading(2.2, position=ObstacleZone.RIGHT, is_valid=True),
        ]

        fused = self.fusion.fuse([item_l, item_c, item_r], readings)
        assessment = self.risk_analyzer.evaluate(fused)

        self.assertEqual(len(fused), 3)
        self.assertEqual(assessment.overall_risk_level, RiskLevel.WARNING)

    def test_multiple_objects_same_zone(self):
        """Verify multiple visual objects falling in the same spatial zone."""
        bbox1 = BoundingBox(xmin=250, ymin=10, xmax=300, ymax=200)
        bbox2 = BoundingBox(xmin=310, ymin=10, xmax=350, ymax=200)

        item1 = DetectionItem("person", 0.90, bbox1, ObstacleZone.CENTER)
        item2 = DetectionItem("bicycle", 0.82, bbox2, ObstacleZone.CENTER)

        sensor_c = SensorReading(0.8, position=ObstacleZone.CENTER, is_valid=True)

        fused = self.fusion.fuse([item1, item2], [sensor_c])
        self.assertEqual(len(fused), 2)
        self.assertEqual(fused[0].sensor_distance_cm, 80.0)
        self.assertEqual(fused[1].sensor_distance_cm, 80.0)

    def test_invalid_sensor_distance_handling(self):
        """Verify fallback behavior when a sensor reading is invalid."""
        bbox = BoundingBox(xmin=250, ymin=10, xmax=350, ymax=200)
        item = DetectionItem("person", 0.9, bbox, ObstacleZone.CENTER)
        invalid_reading = SensorReading(-1.0, position=ObstacleZone.CENTER, status=SensorStatus.INVALID, is_valid=False)

        fused = self.fusion.fuse([item], [invalid_reading])
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0].distance_m, self.fusion.default_range_m)

    def test_empty_detection_list(self):
        """Verify empty detection list with clear sensors returns empty fused obstacle list."""
        readings = [SensorReading(3.0, position=ObstacleZone.CENTER, is_valid=True)]
        fused = self.fusion.fuse([], readings)
        self.assertEqual(len(fused), 0)

        assessment = self.risk_analyzer.evaluate(fused)
        self.assertEqual(assessment.overall_risk_level, RiskLevel.SAFE)

    def test_sensor_only_obstacle_when_camera_sees_nothing(self):
        """Verify a close sensor reading with no visual detection still raises risk."""
        readings = [
            SensorReading(3.0, position=ObstacleZone.LEFT, is_valid=True),
            SensorReading(0.3, position=ObstacleZone.CENTER, is_valid=True),
            SensorReading(3.0, position=ObstacleZone.RIGHT, is_valid=True),
        ]
        fused = self.fusion.fuse([], readings)

        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0].label, SensorFusionEngine.SENSOR_ONLY_LABEL)
        self.assertEqual(fused[0].zone, ObstacleZone.CENTER)
        self.assertEqual(fused[0].confidence, 0.0)
        self.assertIsNone(fused[0].bbox)
        self.assertEqual(self.risk_analyzer.evaluate(fused).overall_risk_level, RiskLevel.DANGER)

    def test_sensor_only_obstacle_not_added_when_zone_has_visual_detection(self):
        """Verify a zone with a YOLO detection is not duplicated as a sensor-only obstacle."""
        bbox_c = BoundingBox(xmin=250, ymin=10, xmax=350, ymax=200)
        item_c = DetectionItem("person", 0.91, bbox_c, ObstacleZone.CENTER)
        readings = [SensorReading(0.8, position=ObstacleZone.CENTER, is_valid=True)]

        fused = self.fusion.fuse([item_c], readings)
        self.assertEqual([obs.label for obs in fused], ["person"])

    def test_sensor_only_threshold_from_config(self):
        """Verify sensor-only obstacles are reported out to the configured CAUTION distance."""
        fusion = SensorFusionEngine(config={"risk_analysis": {"thresholds_cm": {"caution_cm": 120.0}}})
        self.assertEqual(fusion.sensor_only_max_m, 1.2)

        near = fusion.fuse([], [SensorReading(1.1, position=ObstacleZone.LEFT, is_valid=True)])
        far = fusion.fuse([], [SensorReading(1.5, position=ObstacleZone.LEFT, is_valid=True)])
        self.assertEqual(len(near), 1)
        self.assertEqual(len(far), 0)

    def test_invalid_zone_sensor_does_not_borrow_other_zone(self):
        """Verify an invalid CENTER sensor falls back to default range, not the LEFT sensor's distance."""
        bbox_c = BoundingBox(xmin=250, ymin=10, xmax=350, ymax=200)
        item_c = DetectionItem("person", 0.9, bbox_c, ObstacleZone.CENTER)
        readings = [
            SensorReading(2.5, position=ObstacleZone.LEFT, is_valid=True),
            SensorReading(-1.0, position=ObstacleZone.CENTER, status=SensorStatus.INVALID, is_valid=False),
        ]
        # LEFT at 2.5 m is outside the sensor-only range, so only the person is fused
        fusion = SensorFusionEngine(default_range_m=3.0)
        fused = fusion.fuse([item_c], readings)

        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0].distance_m, 3.0)

    def test_preservation_of_confidence_and_label(self):
        """Verify object label, confidence %, and bbox are preserved exactly through fusion."""
        bbox = BoundingBox(xmin=100, ymin=100, xmax=200, ymax=200)
        item = DetectionItem(label="potted plant", confidence=0.77, bbox=bbox, zone=ObstacleZone.LEFT)
        reading = SensorReading(1.5, position=ObstacleZone.LEFT, is_valid=True)

        fused = self.fusion.fuse([item], [reading])
        self.assertEqual(fused[0].object_name, "potted plant")
        self.assertEqual(fused[0].confidence, 0.77)
        self.assertEqual(fused[0].bbox, bbox)


if __name__ == "__main__":
    unittest.main()
