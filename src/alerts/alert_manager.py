"""Coordinating alert manager unifying voice speech and haptic vibration feedback."""

import time
from typing import Optional, List, Tuple, Dict, Any

from .vibration_interface import VibrationInterface
from .simulated_vibration import SimulatedVibration
from .voice_alert import VoiceAlertManager
from ..core.models import RiskAssessment, DirectionGuidance, FusedObstacle
from ..core.enums import RiskLevel, AlertPattern, ObstacleZone, DirectionCommand
from ..decision.risk_analyzer import RISK_PRIORITY
from ..decision.direction_analyzer import DIRECTION_VOICE
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
        cooldown_seconds: float = 2.0,
        clear_confirm_seconds: float = 0.0,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.vibration = vibration
        self.voice = voice
        self.cooldown_seconds = cooldown_seconds
        self.clear_confirm_seconds = clear_confirm_seconds  # "Path clear" only after this long without hazards

        if config:
            alert_cfg = config.get("alerts", {})
            voice_cfg = alert_cfg.get("voice", {})
            self.cooldown_seconds = voice_cfg.get("cooldown_seconds", cooldown_seconds)
            self.clear_confirm_seconds = voice_cfg.get("clear_confirm_seconds", clear_confirm_seconds)

        self._safe_since: Optional[float] = None

        self.last_triggered_time: float = 0.0
        self.last_triggered_risk: Optional[RiskLevel] = None
        self.last_triggered_zone: Optional[ObstacleZone] = None
        self.last_triggered_direction: Optional[DirectionCommand] = None
        self.last_triggered_fault: Optional[ObstacleZone] = None
        self._previous_triggered_state: Optional[tuple] = None  # State announced before the last one
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

    def determine_voice_message(
        self,
        risk_level: RiskLevel,
        zone: ObstacleZone,
        direction: Optional[DirectionCommand] = None,
    ) -> str:
        """Maps risk level and primary obstacle zone to a voice message, plus the direction to take."""
        message = self._hazard_message(risk_level, zone)
        if direction is not None and risk_level != RiskLevel.SAFE:
            message = f"{message} {DIRECTION_VOICE[direction]}"
        return message

    @staticmethod
    def _hazard_message(risk_level: RiskLevel, zone: ObstacleZone) -> str:
        """Describes the hazard: severity and where it is."""
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
        direction: Optional[DirectionGuidance] = None,
    ) -> Tuple[str, str]:
        """Evaluates fused obstacles & risk level to trigger vibration and debounced voice alerts.

        Args:
            direction: Optional guidance; its instruction is spoken with hazard alerts, and a
                changed instruction is announced immediately (bypassing the cooldown).

        Returns:
            Tuple of (vibration_status_string, alert_status_string).
        """
        overall_risk = risk_assessment.overall_risk_level
        direction_cmd = direction.recommended_direction if direction is not None else None

        # Determine primary target zone from highest risk obstacle
        primary_zone = ObstacleZone.CENTER
        primary_obstacle = self.select_primary_obstacle(fused_obstacles)
        if primary_obstacle is not None:
            primary_zone = primary_obstacle.zone

        # A failed sensor drives the alert when no obstacle is as severe as the fault itself
        fault_zone: Optional[ObstacleZone] = None
        if risk_assessment.faulty_zones and (
            primary_obstacle is None
            or RISK_PRIORITY.get(primary_obstacle.risk_level, 0) < RISK_PRIORITY.get(overall_risk, 0)
        ):
            fault_zone = risk_assessment.faulty_zones[0]
            primary_zone = fault_zone

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
        if fault_zone is not None:
            voice_message = f"Caution. {fault_zone.value.capitalize()} sensor not responding."
            if direction_cmd is not None:
                voice_message += f" {DIRECTION_VOICE[direction_cmd]}"
        else:
            voice_message = self.determine_voice_message(overall_risk, primary_zone, direction_cmd)
        current_time = time.time()

        state = (overall_risk, primary_zone, direction_cmd, fault_zone)
        last_state = (
            self.last_triggered_risk,
            self.last_triggered_zone,
            self.last_triggered_direction,
            self.last_triggered_fault,
        )
        state_changed = (self.last_triggered_risk is None) or (state != last_state)
        cooldown_elapsed = (current_time - self.last_triggered_time) >= self.cooldown_seconds

        # Flapping: returning to the state announced just before the last alert (A -> B -> A) within
        # the cooldown is noise near a threshold, not news. Escalating risk is always announced.
        escalated = self.last_triggered_risk is not None and (
            RISK_PRIORITY.get(overall_risk, 0) > RISK_PRIORITY.get(self.last_triggered_risk, 0)
        )
        flapping = (
            state_changed
            and not escalated
            and not cooldown_elapsed
            and state == self._previous_triggered_state
        )

        alert_triggered = False
        suppress_reason = ""

        if overall_risk == RiskLevel.SAFE:
            self._safe_since = current_time if self._safe_since is None else self._safe_since
        else:
            self._safe_since = None

        if overall_risk == RiskLevel.SAFE:
            # SAFE state handling: announce once after leaving a hazard, only when the path has stayed
            # clear for clear_confirm_seconds (a detection dropping out for a frame is not "clear")
            leaving_hazard = (
                state_changed
                and self.last_triggered_risk is not None
                and self.last_triggered_risk != RiskLevel.SAFE
            )
            confirmed = current_time - self._safe_since >= self.clear_confirm_seconds
            if leaving_hazard and confirmed:
                alert_triggered = True
            else:
                alert_triggered = False
                suppress_reason = "SAFE state active" if not leaving_hazard else "Confirming path clear"
        elif flapping:
            alert_triggered = False
            suppress_reason = "Flapping between recent states"
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
            if state != last_state:
                self._previous_triggered_state = last_state
            self.last_triggered_time = current_time
            self.last_triggered_risk = overall_risk
            self.last_triggered_zone = primary_zone
            self.last_triggered_direction = direction_cmd
            self.last_triggered_fault = fault_zone
            self.last_triggered_text = voice_message
            alert_status_text = f"[ALERT TRIGGERED] {overall_risk.value} | {primary_zone.value} | {voice_message}"
            logger.info(alert_status_text)
        else:
            if self.last_triggered_risk is None:
                self.last_triggered_risk = overall_risk
                self.last_triggered_zone = primary_zone
                self.last_triggered_direction = direction_cmd
                self.last_triggered_fault = fault_zone
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
        self.evaluate_and_trigger(risk, fused_obstacles, direction)
