"""Unit tests for Phase 5 Alert Manager debouncing, vibration simulation, and TTS worker."""

import unittest
import time
from unittest.mock import MagicMock

from src.alerts.simulated_vibration import SimulatedVibration
from src.alerts.voice_alert import VoiceAlertManager
from src.alerts.alert_manager import AlertManager
from src.core.models import RiskAssessment, FusedObstacle, BoundingBox
from src.core.enums import RiskLevel, ObstacleZone


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
