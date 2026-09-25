"""Unit tests for RiskAnalyzer decision component."""

import unittest

from src.decision.risk_analyzer import RiskAnalyzer
from src.core.models import FusedObstacle, BoundingBox, SensorReading
from src.core.enums import RiskLevel, ObstacleZone, SensorStatus


class TestRiskAnalysisArchitecture(unittest.TestCase):
    """Test suite for risk analyzer component."""

    def test_risk_analyzer_classification(self):
        """Verify obstacle risk level classification based on distance thresholds."""
        analyzer = RiskAnalyzer(critical_distance_m=0.8, warning_distance_m=1.8, safe_distance_m=3.0)

        bbox = BoundingBox(xmin=0.1, ymin=0.1, xmax=0.3, ymax=0.5)

        obs_critical = FusedObstacle(
            object_id="1", label="person", confidence=0.9, bbox=bbox, distance_m=0.5, zone=ObstacleZone.CENTER
        )
        obs_high = FusedObstacle(
            object_id="2", label="car", confidence=0.9, bbox=bbox, distance_m=1.2, zone=ObstacleZone.CENTER
        )
        obs_medium = FusedObstacle(
            object_id="3", label="dog", confidence=0.8, bbox=bbox, distance_m=1.5, zone=ObstacleZone.LEFT
        )
        obs_safe = FusedObstacle(
            object_id="4", label="chair", confidence=0.8, bbox=bbox, distance_m=4.0, zone=ObstacleZone.RIGHT
        )

        self.assertEqual(analyzer.classify_obstacle_risk(obs_critical), RiskLevel.CRITICAL)
        self.assertEqual(analyzer.classify_obstacle_risk(obs_high), RiskLevel.HIGH)
        self.assertEqual(analyzer.classify_obstacle_risk(obs_medium), RiskLevel.MEDIUM)
        self.assertEqual(analyzer.classify_obstacle_risk(obs_safe), RiskLevel.SAFE)

    def test_risk_assessment_evaluation(self):
        """Verify system-wide RiskAssessment generation."""
        analyzer = RiskAnalyzer(critical_distance_m=0.8, warning_distance_m=1.8, safe_distance_m=3.0)

        bbox = BoundingBox(xmin=0.1, ymin=0.1, xmax=0.3, ymax=0.5)
        obs = FusedObstacle(
            object_id="1", label="stairs", confidence=0.95, bbox=bbox, distance_m=0.6, zone=ObstacleZone.CENTER
        )

        assessment = analyzer.evaluate([obs])

        self.assertEqual(assessment.overall_risk_level, RiskLevel.CRITICAL)
        self.assertEqual(len(assessment.critical_obstacles), 1)
        self.assertIn("Immediate danger", assessment.recommended_action)


class TestSensorFaultRisk(unittest.TestCase):
    """A failed sensor means unknown space, never a silent SAFE."""

    def test_faulty_sensor_raises_risk_to_caution(self):
        readings = [
            SensorReading(3.0, position=ObstacleZone.LEFT),
            SensorReading(-1.0, position=ObstacleZone.CENTER, status=SensorStatus.INVALID, is_valid=False),
        ]
        assessment = RiskAnalyzer().evaluate([], readings)
        self.assertEqual(assessment.overall_risk_level, RiskLevel.CAUTION)
        self.assertEqual(assessment.faulty_zones, [ObstacleZone.CENTER])
        self.assertIn("Sensor fault: CENTER", assessment.recommended_action)

    def test_fault_does_not_lower_higher_risk(self):
        obs = FusedObstacle("1", "person", 0.9, None, 0.4, ObstacleZone.CENTER)
        readings = [SensorReading(-1.0, position=ObstacleZone.LEFT, status=SensorStatus.INVALID, is_valid=False)]
        self.assertEqual(RiskAnalyzer().evaluate([obs], readings).overall_risk_level, RiskLevel.DANGER)

    def test_out_of_range_is_not_a_fault(self):
        readings = [SensorReading(5.0, position=ObstacleZone.RIGHT, status=SensorStatus.OUT_OF_RANGE, is_valid=False)]
        assessment = RiskAnalyzer().evaluate([], readings)
        self.assertEqual(assessment.overall_risk_level, RiskLevel.SAFE)
        self.assertEqual(assessment.faulty_zones, [])


if __name__ == "__main__":
    unittest.main()
