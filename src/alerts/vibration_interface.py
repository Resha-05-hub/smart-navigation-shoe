"""Abstract Base Class for haptic vibration actuators."""

from abc import ABC, abstractmethod

from ..core.enums import AlertPattern, ObstacleZone


class VibrationInterface(ABC):
    """Interface defining operations for haptic vibration motors."""

    @abstractmethod
    def trigger_vibration(
        self,
        pattern: AlertPattern,
        zone: ObstacleZone = ObstacleZone.CENTER,
        duration_sec: float = 0.5,
    ) -> None:
        """Triggers vibration feedback pattern on target directional zone.

        Args:
            pattern: AlertPattern enum value.
            zone: Target directional vibration motor (LEFT, CENTER, RIGHT).
            duration_sec: Duration of vibration burst in seconds.
        """
        pass

    @abstractmethod
    def stop_all(self) -> None:
        """Stops all active vibration motors immediately."""
        pass
