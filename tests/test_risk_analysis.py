"""Unit tests for RiskAnalyzer decision component."""

import unittest

from src.decision.risk_analyzer import RiskAnalyzer
from src.core.models import FusedObstacle, BoundingBox
from src.core.enums import RiskLevel, ObstacleZone


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


if __name__ == "__main__":
    unittest.main()
