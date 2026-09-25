"""Ultrasonic distance sensor implementation reserved for future Raspberry Pi hardware phase."""

import time
from typing import Optional

from .distance_sensor_interface import DistanceSensorInterface
from ..core.models import SensorReading
from ..core.enums import ObstacleZone, SensorStatus
from ..utils.logger import get_logger

logger = get_logger("ultrasonic_sensor")


class UltrasonicSensor(DistanceSensorInterface):
    """HC-SR04 Ultrasonic Distance Sensor driver (Future Raspberry Pi Hardware Implementation).

    Note:
    Reserved for physical hardware integration phase on Raspberry Pi OS via GPIO pins.
    Does NOT access GPIO hardware during PC/Windows simulation phases.
    """

    def __init__(
        self,
        trigger_pin: int = 23,
        echo_pin: int = 24,
        position: ObstacleZone = ObstacleZone.CENTER,
        sensor_id: Optional[str] = None,
    ) -> None:
        self.trigger_pin = trigger_pin
        self.echo_pin = echo_pin
        self.position = position
        self.sensor_id = sensor_id or f"pi_ultrasonic_{position.value.lower()}"
        self._gpio_initialized = False

    def initialize_gpio(self) -> bool:
        """Initializes RPi.GPIO or gpiozero pin configurations for HC-SR04.

        Returns:
            False when running on non-Raspberry Pi platforms or without GPIO permissions.
        """
        logger.info("Initializing HC-SR04 ultrasonic sensor GPIO pins...")
        try:
            # Future GPIO library import (e.g. import RPi.GPIO as GPIO or from gpiozero import DistanceSensor)
            raise NotImplementedError(
                "Physical GPIO ultrasonic sensor interface is reserved for future hardware phase. "
                "Use SimulatedDistanceSensor for current laptop execution."
            )
        except Exception as e:
            logger.warning(f"Ultrasonic sensor GPIO unavailable in Phase 3: {e}")
            self._gpio_initialized = False
            return False

    def read_distance(self) -> SensorReading:
        """Reads pulse timing from HC-SR04 sensor to calculate obstacle distance."""
        if not self._gpio_initialized:
            logger.warning(f"UltrasonicSensor ({self.position.value}) read attempted without active GPIO hardware.")
            return SensorReading(
                distance_m=-1.0,
                sensor_id=self.sensor_id,
                position=self.position,
                status=SensorStatus.INVALID,
                unit="m",
                timestamp=time.time(),
                is_valid=False,
            )

        # Future hardware pulse measurement logic:
        # pulse_start -> pulse_end -> distance_m = (duration * 34300) / 2 / 100
        return SensorReading(
            distance_m=-1.0,
            sensor_id=self.sensor_id,
            position=self.position,
            status=SensorStatus.INVALID,
            unit="m",
            timestamp=time.time(),
            is_valid=False,
        )

    def is_healthy(self) -> bool:
        """Checks if GPIO pins and pulse timing are functioning."""
        return self._gpio_initialized
