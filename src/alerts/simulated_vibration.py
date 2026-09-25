"""Simulated vibration interface for PC/laptop execution."""

from .vibration_interface import VibrationInterface
from ..core.enums import AlertPattern, ObstacleZone, RiskLevel
from ..utils.logger import get_logger

logger = get_logger("simulated_vibration")


class SimulatedVibration(VibrationInterface):
    """Simulated haptic vibration actuator for software testing and debugging.

    Note:
    Logs haptic signals to console/loggers without sending physical electric pulses.
    Used for laptop development. Physical motors will be integrated in Phase 8 on Raspberry Pi.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.active_pattern = AlertPattern.NONE
        self.last_vibration_status: str = "VIBRATION: OFF"

    def get_vibration_text(self, risk_level: RiskLevel, zone: ObstacleZone) -> str:
        """Determines simulated directional vibration output string.

        Mapping:
        - SAFE: OFF
        - CAUTION LEFT: LEFT - SHORT PULSE
        - CAUTION CENTER: BOTH - SHORT PULSE
        - CAUTION RIGHT: RIGHT - SHORT PULSE
        - WARNING LEFT: LEFT - MEDIUM PULSE
        - WARNING CENTER: BOTH - MEDIUM PULSE
        - WARNING RIGHT: RIGHT - MEDIUM PULSE
        - DANGER LEFT: LEFT - RAPID PULSES
        - DANGER CENTER: BOTH - RAPID PULSES
        - DANGER RIGHT: RIGHT - RAPID PULSES
        """
        if risk_level == RiskLevel.SAFE or not self.enabled:
            return "VIBRATION: OFF"

        motor_str = "BOTH"
        if zone == ObstacleZone.LEFT:
            motor_str = "LEFT"
        elif zone == ObstacleZone.RIGHT:
            motor_str = "RIGHT"

        if risk_level in (RiskLevel.DANGER, RiskLevel.CRITICAL):
            pulse_str = "RAPID PULSES"
        elif risk_level in (RiskLevel.WARNING, RiskLevel.HIGH):
            pulse_str = "MEDIUM PULSE"
        else:  # CAUTION / LOW / MEDIUM
            pulse_str = "SHORT PULSE"

        return f"VIBRATION: {motor_str} - {pulse_str}"

    def trigger_directional_vibration(
        self,
        risk_level: RiskLevel,
        zone: ObstacleZone = ObstacleZone.CENTER,
    ) -> str:
        """Triggers simulated directional vibration motors based on risk level and spatial zone."""
        status_text = self.get_vibration_text(risk_level, zone)
        if self.enabled and status_text != self.last_vibration_status:
            logger.info(f"[SIMULATED HAPTIC] {status_text}")  # Log changes only, not every frame
        self.last_vibration_status = status_text

        return status_text

    def trigger_vibration(
        self,
        pattern: AlertPattern,
        zone: ObstacleZone = ObstacleZone.CENTER,
        duration_sec: float = 0.5,
    ) -> None:
        """Logs simulated vibration pulse (backwards compatibility)."""
        if not self.enabled:
            return

        self.active_pattern = pattern
        logger.info(
            f"[SIMULATED VIBRATION] Pattern: '{pattern.value}' | Zone: '{zone.value}' | Duration: {duration_sec}s"
        )

    def stop_all(self) -> None:
        """Resets simulated vibration state."""
        self.active_pattern = AlertPattern.NONE
        self.last_vibration_status = "VIBRATION: OFF"
        logger.info("[SIMULATED VIBRATION] Stopped all vibration actuators.")
