"""Unit tests for DirectionAnalyzer decision component."""

import unittest

from src.decision.direction_analyzer import DirectionAnalyzer
from src.decision.risk_analyzer import RiskAnalyzer
from src.core.models import FusedObstacle, BoundingBox, SensorReading
from src.core.enums import DirectionCommand, ObstacleZone, RiskLevel, SensorStatus


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


def obstacle(zone: ObstacleZone, distance_m: float, label: str = "obstacle") -> FusedObstacle:
    return FusedObstacle(label, label, 0.9, None, distance_m, zone)


class TestDirectionGuidanceRules(unittest.TestCase):
    """Step 5 guidance rules: side hazards, faulty sensors, blocked paths, and hysteresis."""

    def setUp(self):
        self.analyzer = DirectionAnalyzer(clear_path_threshold_m=2.0, side_hysteresis_m=0.3)
        self.risk = RiskAnalyzer()

    def guide(self, obstacles, readings=None):
        assessment = self.risk.evaluate(obstacles)
        return self.analyzer.analyze_path(obstacles, assessment, readings)

    def test_side_danger_with_clear_center_steers_away_instead_of_stop(self):
        guidance = self.guide([obstacle(ObstacleZone.LEFT, 0.4)])
        self.assertEqual(guidance.recommended_direction, DirectionCommand.SLIGHT_RIGHT)

        guidance = self.guide([obstacle(ObstacleZone.RIGHT, 0.8)])
        self.assertEqual(guidance.recommended_direction, DirectionCommand.SLIGHT_LEFT)

    def test_side_caution_keeps_straight(self):
        guidance = self.guide([obstacle(ObstacleZone.LEFT, 1.5)])
        self.assertEqual(guidance.recommended_direction, DirectionCommand.MOVE_FORWARD)

    def test_corridor_keeps_straight(self):
        guidance = self.guide([obstacle(ObstacleZone.LEFT, 0.7), obstacle(ObstacleZone.RIGHT, 0.8)])
        self.assertEqual(guidance.recommended_direction, DirectionCommand.MOVE_FORWARD)
        self.assertIn("both sides", guidance.safety_notes)

    def test_center_blocked_turns_when_no_side_is_fully_clear(self):
        guidance = self.guide([
            obstacle(ObstacleZone.CENTER, 1.2),
            obstacle(ObstacleZone.LEFT, 1.6),
            obstacle(ObstacleZone.RIGHT, 1.1),
        ])
        self.assertEqual(guidance.recommended_direction, DirectionCommand.TURN_LEFT)

    def test_all_directions_dangerous_stops(self):
        guidance = self.guide([
            obstacle(ObstacleZone.CENTER, 1.2),
            obstacle(ObstacleZone.LEFT, 0.3),
            obstacle(ObstacleZone.RIGHT, 0.4),
        ])
        self.assertEqual(guidance.recommended_direction, DirectionCommand.STOP)
        self.assertIn("No safe direction", guidance.safety_notes)

    def test_faulty_side_sensor_is_never_the_escape_route(self):
        readings = [
            SensorReading(-1.0, position=ObstacleZone.LEFT, status=SensorStatus.INVALID, is_valid=False),
            SensorReading(0.9, position=ObstacleZone.CENTER),
            SensorReading(1.6, position=ObstacleZone.RIGHT),
        ]
        guidance = self.guide([obstacle(ObstacleZone.CENTER, 0.9), obstacle(ObstacleZone.RIGHT, 1.6)], readings)
        # Without the fault, the empty LEFT zone would look clearest
        self.assertEqual(guidance.recommended_direction, DirectionCommand.TURN_RIGHT)

    def test_faulty_center_sensor_is_not_clear_ahead(self):
        readings = [
            SensorReading(3.0, position=ObstacleZone.LEFT),
            SensorReading(-1.0, position=ObstacleZone.CENTER, status=SensorStatus.INVALID, is_valid=False),
            SensorReading(3.0, position=ObstacleZone.RIGHT),
        ]
        guidance = self.guide([], readings)
        self.assertIn(guidance.recommended_direction, (DirectionCommand.SLIGHT_LEFT, DirectionCommand.SLIGHT_RIGHT))
        self.assertTrue(guidance.safety_notes.startswith("Center sensor fault."))

    def test_out_of_range_side_is_clear_not_faulty(self):
        readings = [
            SensorReading(5.0, position=ObstacleZone.LEFT, status=SensorStatus.OUT_OF_RANGE, is_valid=False),
            SensorReading(0.9, position=ObstacleZone.CENTER),
            SensorReading(1.6, position=ObstacleZone.RIGHT),
        ]
        guidance = self.guide([obstacle(ObstacleZone.CENTER, 0.9), obstacle(ObstacleZone.RIGHT, 1.6)], readings)
        self.assertEqual(guidance.recommended_direction, DirectionCommand.SLIGHT_LEFT)

    def test_hysteresis_keeps_previous_side_on_small_differences(self):
        center = obstacle(ObstacleZone.CENTER, 1.0)
        first = self.guide([center, obstacle(ObstacleZone.LEFT, 1.5), obstacle(ObstacleZone.RIGHT, 1.4)])
        self.assertEqual(first.recommended_direction, DirectionCommand.TURN_LEFT)

        # RIGHT now slightly clearer (by 0.2 m < 0.3 m hysteresis): stay LEFT
        second = self.guide([center, obstacle(ObstacleZone.LEFT, 1.4), obstacle(ObstacleZone.RIGHT, 1.6)])
        self.assertEqual(second.recommended_direction, DirectionCommand.TURN_LEFT)

        # RIGHT clearly better: switch
        third = self.guide([center, obstacle(ObstacleZone.LEFT, 1.1), obstacle(ObstacleZone.RIGHT, 1.8)])
        self.assertEqual(third.recommended_direction, DirectionCommand.TURN_RIGHT)

    def test_config_values(self):
        analyzer = DirectionAnalyzer(config={"direction": {"clear_path_threshold": 1.5, "side_hysteresis_m": 0.5}})
        self.assertEqual(analyzer.clear_path_threshold_m, 1.5)
        self.assertEqual(analyzer.side_hysteresis_m, 0.5)


if __name__ == "__main__":
    unittest.main()
