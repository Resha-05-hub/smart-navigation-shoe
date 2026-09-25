"""Unit tests for DirectionAnalyzer decision component."""

import unittest

from src.decision.direction_analyzer import DirectionAnalyzer
from src.decision.risk_analyzer import RiskAnalyzer
from src.core.models import FusedObstacle, BoundingBox
from src.core.enums import DirectionCommand, ObstacleZone, RiskLevel


class TestDirectionArchitecture(unittest.TestCase):
    """Test suite for direction analyzer component."""

    def test_direction_analyzer_forward(self):
        """Verify forward movement recommendation when path is clear."""
        analyzer = DirectionAnalyzer(clear_path_threshold_m=2.0)
        risk_analyzer = RiskAnalyzer()

        assessment = risk_analyzer.evaluate([])
        guidance = analyzer.analyze_path([], assessment)

        self.assertEqual(guidance.recommended_direction, DirectionCommand.MOVE_FORWARD)
        self.assertEqual(guidance.clear_path_score, 1.0)

    def test_direction_analyzer_stop_on_critical(self):
        """Verify STOP command recommendation when critical risk is present."""
        analyzer = DirectionAnalyzer(clear_path_threshold_m=2.0)
        risk_analyzer = RiskAnalyzer()

        bbox = BoundingBox(xmin=0.4, ymin=0.4, xmax=0.6, ymax=0.8)
        obs = FusedObstacle(
            object_id="1", label="pole", confidence=0.9, bbox=bbox, distance_m=0.5, zone=ObstacleZone.CENTER
        )

        assessment = risk_analyzer.evaluate([obs])
        guidance = analyzer.analyze_path([obs], assessment)

        self.assertEqual(guidance.recommended_direction, DirectionCommand.STOP)
        self.assertEqual(guidance.clear_path_score, 0.0)

    def test_direction_analyzer_avoidance_steering(self):
        """Verify steering recommendation when center zone is obstructed."""
        analyzer = DirectionAnalyzer(clear_path_threshold_m=2.0)
        risk_analyzer = RiskAnalyzer()

        bbox = BoundingBox(xmin=0.4, ymin=0.4, xmax=0.6, ymax=0.8)
        obs_center = FusedObstacle(
            object_id="1", label="box", confidence=0.8, bbox=bbox, distance_m=1.2, zone=ObstacleZone.CENTER
        )
        obs_right = FusedObstacle(
            object_id="2", label="trash_can", confidence=0.8, bbox=bbox, distance_m=1.0, zone=ObstacleZone.RIGHT
        )

        assessment = risk_analyzer.evaluate([obs_center, obs_right])
        guidance = analyzer.analyze_path([obs_center, obs_right], assessment)

        self.assertEqual(guidance.recommended_direction, DirectionCommand.SLIGHT_LEFT)


if __name__ == "__main__":
    unittest.main()
