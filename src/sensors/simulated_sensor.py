"""Simulated distance sensor implementation for PC/laptop testing."""

import random
import time
from typing import Optional, Tuple

from .distance_sensor_interface import DistanceSensorInterface
from ..core.models import SensorReading
from ..core.enums import ObstacleZone, SensorStatus
from ..utils.logger import get_logger

logger = get_logger("simulated_sensor")


class SimulatedDistanceSensor(DistanceSensorInterface):
    """Simulated distance sensor generating controlled range values for LEFT, CENTER, or RIGHT shoe positions.

    Disclaimer:
    This class generates SIMULATED range data for PC software development and testing.
    It does NOT read from physical GPIO hardware pins.
    """

    def __init__(
        self,
        position: ObstacleZone = ObstacleZone.CENTER,
        min_distance_m: float = 0.2,
        max_distance_m: float = 4.0,
        default_distance_m: float = 2.5,
        sensor_id: Optional[str] = None,
    ) -> None:
        self.position = position
        self.min_distance_m = min_distance_m
        self.max_distance_m = max_distance_m
        self.current_distance_m = default_distance_m
        self.sensor_id = sensor_id or f"simulated_sensor_{position.value.lower()}"

    def set_simulated_distance(self, distance_m: float) -> None:
        """Sets simulated distance in meters."""
        self.current_distance_m = distance_m

    def set_simulated_distance_cm(self, distance_cm: float) -> None:
        """Sets simulated distance in centimeters."""
        self.current_distance_m = distance_cm / 100.0

    def evaluate_status(self, distance_m: float) -> Tuple[SensorStatus, bool]:
        """Evaluates sensor reading status based on configured range boundaries.

        Returns:
            Tuple of (SensorStatus enum, is_valid boolean).
        """
        if distance_m < 0.0 or distance_m == 0.0:
            return SensorStatus.INVALID, False
        elif distance_m < self.min_distance_m or distance_m > self.max_distance_m:
            return SensorStatus.OUT_OF_RANGE, False
        else:
            return SensorStatus.HEALTHY, True

    def read_distance(self) -> SensorReading:
        """Returns current simulated distance reading with validation status."""
        status, is_valid = self.evaluate_status(self.current_distance_m)

        return SensorReading(
            distance_m=round(self.current_distance_m, 3),
            sensor_id=self.sensor_id,
            position=self.position,
            status=status,
            unit="m",
            timestamp=time.time(),
            is_valid=is_valid,
        )

    def read_random_distance(self) -> SensorReading:
        """Generates a random valid distance within configured bounds for testing."""
        distance = random.uniform(self.min_distance_m, self.max_distance_m)
        self.set_simulated_distance(distance)
        return self.read_distance()

    def is_healthy(self) -> bool:
        """Checks if current simulated distance is within healthy operational range."""
        status, _ = self.evaluate_status(self.current_distance_m)
        return status == SensorStatus.HEALTHY
