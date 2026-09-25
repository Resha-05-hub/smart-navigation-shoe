"""Coordinating alert manager unifying voice speech and haptic vibration feedback."""

import time
from typing import Optional, List, Tuple, Dict, Any

from .vibration_interface import VibrationInterface
from .simulated_vibration import SimulatedVibration
from .voice_alert import VoiceAlertManager
from .audio_manager import AudioManager
from ..core.models import RiskAssessment, DirectionGuidance, FusedObstacle
from ..core.enums import RiskLevel, AlertPattern, ObstacleZone
from ..decision.risk_analyzer import RISK_PRIORITY
from ..utils.logger import get_logger

logger = get_logger("alert_manager")


class AlertManager:
    """Dispatches appropriate haptic vibration patterns and voice prompts based on system risk and direction guidance.

    Features:
    - Cooldown / debouncing mechanism to prevent continuous voice spam.
    - Directional vibration patterns (LEFT, CENTER, RIGHT).
    - Unified single voice alert for multi-object scenes.
    - Clear distinction between triggered vs. suppressed alerts.
    """

    def __init__(
        self,
        vibration: VibrationInterface,
        voice: VoiceAlertManager,
        audio: Optional[AudioManager] = None,
        cooldown_seconds: float = 2.0,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.vibration = vibration
        self.voice = voice
        self.audio = audio or AudioManager()
        self.cooldown_seconds = cooldown_seconds

        if config:
            alert_cfg = config.get("alerts", {})
            voice_cfg = alert_cfg.get("voice", {})
            self.cooldown_seconds = voice_cfg.get("cooldown_seconds", cooldown_seconds)

        self.last_triggered_time: float = 0.0
        self.last_triggered_risk: Optional[RiskLevel] = None
        self.last_triggered_zone: Optional[ObstacleZone] = None
        self.last_triggered_text: str = ""
        self.last_vibration_text: str = "VIBRATION: OFF"
        self.last_alert_triggered: bool = False

    @staticmethod
    def select_primary_obstacle(fused_obstacles: List[FusedObstacle]) -> Optional[FusedObstacle]:
        """Returns the obstacle that alerts should describe: highest risk first, then closest distance."""
        if not fused_obstacles:
            return None
        return max(
            fused_obstacles,
            key=lambda obs: (RISK_PRIORITY.get(obs.risk_level, 0), -obs.distance_m),
        )

    def determine_voice_message(self, risk_level: RiskLevel, zone: ObstacleZone) -> str:
        """Maps overall risk level and primary obstacle zone to prototype voice messages."""
        if risk_level == RiskLevel.SAFE:
            return "Path clear."

        if risk_level in (RiskLevel.DANGER, RiskLevel.CRITICAL):
            if zone == ObstacleZone.LEFT:
                return "Danger. Obstacle on the left."
            elif zone == ObstacleZone.RIGHT:
                return "Danger. Obstacle on the right."
            else:
                return "Danger. Obstacle very close."
        elif risk_level in (RiskLevel.WARNING, RiskLevel.HIGH):
            if zone == ObstacleZone.LEFT:
                return "Warning. Obstacle on the left. Slow down."
            elif zone == ObstacleZone.RIGHT:
                return "Warning. Obstacle on the right. Slow down."
            else:
                return "Warning. Obstacle ahead. Slow down."
        else:  # CAUTION / LOW / MEDIUM
            if zone == ObstacleZone.LEFT:
                return "Caution. Obstacle on the left."
            elif zone == ObstacleZone.RIGHT:
                return "Caution. Obstacle on the right."
            else:
                return "Caution. Obstacle nearby."

    def evaluate_and_trigger(
        self,
        risk_assessment: RiskAssessment,
        fused_obstacles: List[FusedObstacle],
    ) -> Tuple[str, str]:
        """Evaluates fused obstacles & risk level to trigger vibration and debounced voice alerts.

        Returns:
            Tuple of (vibration_status_string, alert_status_string).
        """
        overall_risk = risk_assessment.overall_risk_level

        # Determine primary target zone from highest risk obstacle
        primary_zone = ObstacleZone.CENTER
        primary_obstacle = self.select_primary_obstacle(fused_obstacles)
        if primary_obstacle is not None:
            primary_zone = primary_obstacle.zone

        # 1. Trigger Vibration Feedback
        if isinstance(self.vibration, SimulatedVibration):
            vib_text = self.vibration.trigger_directional_vibration(overall_risk, primary_zone)
        else:
            vib_text = "VIBRATION: ACTIVE"
            if overall_risk in (RiskLevel.DANGER, RiskLevel.CRITICAL):
                self.vibration.trigger_vibration(AlertPattern.CONTINUOUS_HIGH, primary_zone)
            elif overall_risk in (RiskLevel.WARNING, RiskLevel.HIGH):
                self.vibration.trigger_vibration(AlertPattern.MEDIUM_PULSE, primary_zone)
            elif overall_risk in (RiskLevel.CAUTION, RiskLevel.MEDIUM, RiskLevel.LOW):
                self.vibration.trigger_vibration(AlertPattern.LOW_PULSE, primary_zone)
            else:
                self.vibration.stop_all()

        self.last_vibration_text = vib_text

        # 2. Voice Alert Cooldown & Debouncing Evaluation
        voice_message = self.determine_voice_message(overall_risk, primary_zone)
        current_time = time.time()

        state_changed = (
            (self.last_triggered_risk is None)
            or (overall_risk != self.last_triggered_risk)
            or (primary_zone != self.last_triggered_zone)
        )
        cooldown_elapsed = (current_time - self.last_triggered_time) >= self.cooldown_seconds

        alert_triggered = False
        suppress_reason = ""

        if overall_risk == RiskLevel.SAFE:
            # SAFE state handling: trigger once if transitioning from a non-SAFE risk state, but never repeat continuously
            if state_changed and self.last_triggered_risk is not None and self.last_triggered_risk != RiskLevel.SAFE:
                alert_triggered = True
            else:
                alert_triggered = False
                suppress_reason = "SAFE state active"
        elif state_changed:
            # Immediate trigger on significant risk level or directional zone change
            alert_triggered = True
        elif cooldown_elapsed:
            # Cooldown period elapsed for active hazard warning
            alert_triggered = True
        else:
            # Alert suppressed by active cooldown
            alert_triggered = False
            suppress_reason = "Cooldown active"

        self.last_alert_triggered = alert_triggered

        if alert_triggered:
            self.voice.speak(voice_message, non_blocking=True)
            self.last_triggered_time = current_time
            self.last_triggered_risk = overall_risk
            self.last_triggered_zone = primary_zone
            self.last_triggered_text = voice_message
            alert_status_text = f"[ALERT TRIGGERED] {overall_risk.value} | {primary_zone.value} | {voice_message}"
            logger.info(alert_status_text)
        else:
            if self.last_triggered_risk is None:
                self.last_triggered_risk = overall_risk
                self.last_triggered_zone = primary_zone
                self.last_triggered_time = current_time
            alert_status_text = f"[ALERT SUPPRESSED] {overall_risk.value} | {primary_zone.value} | {suppress_reason}"
            logger.debug(alert_status_text)

        return vib_text, alert_status_text

    def process_alerts(
        self,
        risk: RiskAssessment,
        direction: DirectionGuidance,
    ) -> None:
        """Processes risk level and direction guidance for backwards compatibility."""
        fused_obstacles = risk.critical_obstacles
        self.evaluate_and_trigger(risk, fused_obstacles)
