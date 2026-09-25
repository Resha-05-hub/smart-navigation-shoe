"""Unit tests for Phase 5 Alert Manager debouncing, vibration simulation, and TTS worker."""

import unittest
import time
from unittest.mock import MagicMock

from src.alerts.simulated_vibration import SimulatedVibration
from src.alerts.voice_alert import VoiceAlertManager
from src.alerts.alert_manager import AlertManager
from src.core.models import RiskAssessment, FusedObstacle, BoundingBox, DirectionGuidance
from src.core.enums import RiskLevel, ObstacleZone, DirectionCommand


class TestPhase5AlertDebouncing(unittest.TestCase):
    """Test suite for Phase 5 alert debouncing, cooldown, and status logging."""

    def setUp(self):
        self.vibration = SimulatedVibration(enabled=True)
        self.mock_voice = MagicMock()
        self.alert_manager = AlertManager(
            vibration=self.vibration,
            voice=self.mock_voice,
            cooldown_seconds=1.0,
        )

    def test_same_alert_repeated_immediately_suppressed(self):
        """Verify identical alert repeated immediately is suppressed by cooldown."""
        risk = RiskAssessment(overall_risk_level=RiskLevel.WARNING)
        obs = FusedObstacle("1", "person", 0.91, BoundingBox(250, 10, 350, 50), 0.8, ObstacleZone.CENTER)

        # Call 1 -> Triggered
        vib1, status1 = self.alert_manager.evaluate_and_trigger(risk, [obs])
        self.assertIn("[ALERT TRIGGERED]", status1)
        self.assertEqual(self.mock_voice.speak.call_count, 1)

        # Call 2 (immediate) -> Suppressed
        vib2, status2 = self.alert_manager.evaluate_and_trigger(risk, [obs])
        self.assertIn("[ALERT SUPPRESSED]", status2)
        self.assertIn("Cooldown active", status2)
        self.assertEqual(self.mock_voice.speak.call_count, 1)

    def test_same_alert_after_cooldown_allowed(self):
        """Verify identical alert repeated after cooldown duration is allowed."""
        risk = RiskAssessment(overall_risk_level=RiskLevel.WARNING)
        obs = FusedObstacle("1", "person", 0.91, BoundingBox(250, 10, 350, 50), 0.8, ObstacleZone.CENTER)

        # Call 1 -> Triggered
        _, status1 = self.alert_manager.evaluate_and_trigger(risk, [obs])
        self.assertIn("[ALERT TRIGGERED]", status1)
        self.assertEqual(self.mock_voice.speak.call_count, 1)

        # Simulate time passage past cooldown threshold (1.0s)
        self.alert_manager.last_triggered_time = time.time() - 1.5

        # Call 2 -> Triggered after cooldown
        _, status2 = self.alert_manager.evaluate_and_trigger(risk, [obs])
        self.assertIn("[ALERT TRIGGERED]", status2)
        self.assertEqual(self.mock_voice.speak.call_count, 2)

    def test_changed_risk_level_allowed_immediately(self):
        """Verify risk level change (WARNING -> DANGER) bypasses cooldown immediately."""
        risk_warning = RiskAssessment(overall_risk_level=RiskLevel.WARNING)
        obs_warning = FusedObstacle("1", "person", 0.91, BoundingBox(250, 10, 350, 50), 0.8, ObstacleZone.CENTER)

        _, status1 = self.alert_manager.evaluate_and_trigger(risk_warning, [obs_warning])
        self.assertIn("[ALERT TRIGGERED]", status1)
        self.assertEqual(self.mock_voice.speak.call_count, 1)

        # Immediate risk change to DANGER
        risk_danger = RiskAssessment(overall_risk_level=RiskLevel.DANGER)
        obs_danger = FusedObstacle("2", "car", 0.95, BoundingBox(250, 10, 350, 50), 0.4, ObstacleZone.CENTER)

        _, status2 = self.alert_manager.evaluate_and_trigger(risk_danger, [obs_danger])
        self.assertIn("[ALERT TRIGGERED]", status2)
        self.assertIn("Danger. Obstacle very close.", status2)
        self.assertEqual(self.mock_voice.speak.call_count, 2)

    def test_changed_direction_allowed_immediately(self):
        """Verify spatial direction change (WARNING CENTER -> WARNING LEFT) bypasses cooldown immediately."""
        risk = RiskAssessment(overall_risk_level=RiskLevel.WARNING)
        obs_center = FusedObstacle("1", "person", 0.91, BoundingBox(250, 10, 350, 50), 0.8, ObstacleZone.CENTER)

        _, status1 = self.alert_manager.evaluate_and_trigger(risk, [obs_center])
        self.assertIn("[ALERT TRIGGERED]", status1)
        self.assertEqual(self.mock_voice.speak.call_count, 1)

        # Immediate zone change to LEFT
        obs_left = FusedObstacle("2", "chair", 0.88, BoundingBox(10, 10, 50, 50), 0.8, ObstacleZone.LEFT)

        _, status2 = self.alert_manager.evaluate_and_trigger(risk, [obs_left])
        self.assertIn("[ALERT TRIGGERED]", status2)
        self.assertIn("Warning. Obstacle on the left. Slow down.", status2)
        self.assertEqual(self.mock_voice.speak.call_count, 2)

    def test_safe_state_repeated_suppressed(self):
        """Verify repeated SAFE state does not continuously trigger voice alerts."""
        risk_safe = RiskAssessment(overall_risk_level=RiskLevel.SAFE)

        # Frame 1 SAFE -> Suppressed (initial baseline)
        _, status1 = self.alert_manager.evaluate_and_trigger(risk_safe, [])
        self.assertIn("[ALERT SUPPRESSED]", status1)

        # Frame 2 SAFE (immediate) -> Suppressed
        _, status2 = self.alert_manager.evaluate_and_trigger(risk_safe, [])
        self.assertIn("[ALERT SUPPRESSED]", status2)

        # Simulate time passage past cooldown threshold
        self.alert_manager.last_triggered_time = time.time() - 5.0

        # Frame 3 SAFE (after cooldown) -> Still Suppressed (SAFE never repeats)
        _, status3 = self.alert_manager.evaluate_and_trigger(risk_safe, [])
        self.assertIn("[ALERT SUPPRESSED]", status3)
        self.assertEqual(self.mock_voice.speak.call_count, 0)

    def test_primary_zone_uses_highest_risk_obstacle(self):
        """Verify alerts describe the most dangerous obstacle, not whichever YOLO listed first."""
        risk = RiskAssessment(overall_risk_level=RiskLevel.WARNING)
        chair_right_safe = FusedObstacle(
            "1", "chair", 0.9, BoundingBox(500, 10, 600, 50), 2.2, ObstacleZone.RIGHT, RiskLevel.SAFE
        )
        person_center_warning = FusedObstacle(
            "2", "person", 0.9, BoundingBox(250, 10, 350, 50), 0.8, ObstacleZone.CENTER, RiskLevel.WARNING
        )

        vib, status = self.alert_manager.evaluate_and_trigger(risk, [chair_right_safe, person_center_warning])
        self.assertEqual(vib, "VIBRATION: BOTH - MEDIUM PULSE")
        self.assertIn("Warning. Obstacle ahead. Slow down.", status)

    def test_primary_obstacle_ties_broken_by_distance(self):
        """Verify the closer obstacle wins when two share the same risk level."""
        far_left = FusedObstacle("1", "chair", 0.9, None, 0.9, ObstacleZone.LEFT, RiskLevel.WARNING)
        near_right = FusedObstacle("2", "dog", 0.9, None, 0.6, ObstacleZone.RIGHT, RiskLevel.WARNING)

        primary = AlertManager.select_primary_obstacle([far_left, near_right])
        self.assertIs(primary, near_right)
        self.assertIsNone(AlertManager.select_primary_obstacle([]))

    def test_last_alert_triggered_flag(self):
        """Verify last_alert_triggered reports whether the latest call actually spoke."""
        risk = RiskAssessment(overall_risk_level=RiskLevel.WARNING)
        obs = FusedObstacle("1", "person", 0.91, None, 0.8, ObstacleZone.CENTER, RiskLevel.WARNING)

        self.alert_manager.evaluate_and_trigger(risk, [obs])
        self.assertTrue(self.alert_manager.last_alert_triggered)
        self.assertEqual(self.alert_manager.last_triggered_text, "Warning. Obstacle ahead. Slow down.")

        self.alert_manager.evaluate_and_trigger(risk, [obs])
        self.assertFalse(self.alert_manager.last_alert_triggered)

    def test_voice_message_includes_direction(self):
        risk = RiskAssessment(overall_risk_level=RiskLevel.WARNING)
        obs = FusedObstacle("1", "person", 0.91, None, 0.8, ObstacleZone.CENTER, RiskLevel.WARNING)
        guidance = DirectionGuidance(DirectionCommand.SLIGHT_LEFT, 1.0, "Center blocked. Veer slight left.")

        _, status = self.alert_manager.evaluate_and_trigger(risk, [obs], guidance)
        self.assertEqual(
            self.alert_manager.last_triggered_text,
            "Warning. Obstacle ahead. Slow down. Move slightly left.",
        )
        self.mock_voice.speak.assert_called_with(
            "Warning. Obstacle ahead. Slow down. Move slightly left.", non_blocking=True
        )

    def test_safe_message_has_no_direction(self):
        self.assertEqual(
            self.alert_manager.determine_voice_message(RiskLevel.SAFE, ObstacleZone.CENTER, DirectionCommand.MOVE_FORWARD),
            "Path clear.",
        )
        self.assertEqual(
            self.alert_manager.determine_voice_message(RiskLevel.DANGER, ObstacleZone.CENTER, DirectionCommand.STOP),
            "Danger. Obstacle very close. Stop.",
        )

    def test_direction_change_bypasses_cooldown(self):
        risk = RiskAssessment(overall_risk_level=RiskLevel.WARNING)
        obs = FusedObstacle("1", "person", 0.91, None, 0.8, ObstacleZone.CENTER, RiskLevel.WARNING)
        left = DirectionGuidance(DirectionCommand.SLIGHT_LEFT)
        right = DirectionGuidance(DirectionCommand.SLIGHT_RIGHT)

        self.alert_manager.evaluate_and_trigger(risk, [obs], left)
        _, same = self.alert_manager.evaluate_and_trigger(risk, [obs], left)
        self.assertIn("[ALERT SUPPRESSED]", same)

        _, changed = self.alert_manager.evaluate_and_trigger(risk, [obs], right)
        self.assertIn("[ALERT TRIGGERED]", changed)
        self.assertIn("Move slightly right.", changed)

    def test_flapping_back_to_previous_state_suppressed(self):
        """A -> B -> A within the cooldown is threshold noise: the return to A is not re-announced."""
        risk = RiskAssessment(overall_risk_level=RiskLevel.WARNING)
        left = FusedObstacle("1", "wall", 0.9, None, 0.8, ObstacleZone.LEFT, RiskLevel.WARNING)
        right = FusedObstacle("2", "wall", 0.9, None, 0.8, ObstacleZone.RIGHT, RiskLevel.WARNING)
        straight = DirectionGuidance(DirectionCommand.MOVE_FORWARD)

        _, a = self.alert_manager.evaluate_and_trigger(risk, [left], straight)
        _, b = self.alert_manager.evaluate_and_trigger(risk, [right], straight)
        _, back = self.alert_manager.evaluate_and_trigger(risk, [left], straight)
        self.assertIn("[ALERT TRIGGERED]", a)
        self.assertIn("[ALERT TRIGGERED]", b)
        self.assertIn("Flapping", back)

        # After the cooldown the current state is announced normally
        self.alert_manager.last_triggered_time = time.time() - 1.5
        _, later = self.alert_manager.evaluate_and_trigger(risk, [left], straight)
        self.assertIn("[ALERT TRIGGERED]", later)

    def test_escalation_is_never_treated_as_flapping(self):
        warning = RiskAssessment(overall_risk_level=RiskLevel.WARNING)
        danger = RiskAssessment(overall_risk_level=RiskLevel.DANGER)
        near = FusedObstacle("1", "person", 0.9, None, 0.4, ObstacleZone.CENTER, RiskLevel.DANGER)
        mid = FusedObstacle("1", "person", 0.9, None, 0.6, ObstacleZone.CENTER, RiskLevel.WARNING)

        self.alert_manager.evaluate_and_trigger(danger, [near])
        self.alert_manager.evaluate_and_trigger(warning, [mid])
        _, again = self.alert_manager.evaluate_and_trigger(danger, [near])
        self.assertIn("[ALERT TRIGGERED]", again)

    def test_sensor_fault_announced_instead_of_path_clear(self):
        fault = RiskAssessment(overall_risk_level=RiskLevel.CAUTION, faulty_zones=[ObstacleZone.CENTER])
        guidance = DirectionGuidance(DirectionCommand.SLIGHT_RIGHT)

        vib, status = self.alert_manager.evaluate_and_trigger(fault, [], guidance)
        self.assertIn("[ALERT TRIGGERED]", status)
        self.assertEqual(
            self.alert_manager.last_triggered_text,
            "Caution. Center sensor not responding. Move slightly right.",
        )
        self.assertEqual(vib, "VIBRATION: BOTH - SHORT PULSE")

    def test_obstacle_more_severe_than_fault_takes_priority(self):
        risk = RiskAssessment(overall_risk_level=RiskLevel.DANGER, faulty_zones=[ObstacleZone.LEFT])
        obs = FusedObstacle("1", "person", 0.9, None, 0.4, ObstacleZone.CENTER, RiskLevel.DANGER)
        self.alert_manager.evaluate_and_trigger(risk, [obs], DirectionGuidance(DirectionCommand.STOP))
        self.assertEqual(self.alert_manager.last_triggered_text, "Danger. Obstacle very close. Stop.")

    def test_no_overlapping_tts_calls_worker(self):
        """Verify thread-safe VoiceAlertManager queue processes rapid calls without overlapping loops."""
        voice = VoiceAlertManager(enabled=True)
        try:
            for i in range(15):
                voice.speak(f"Rapid test call {i}")
            time.sleep(0.1)
        finally:
            voice.stop()


if __name__ == "__main__":
    unittest.main()
