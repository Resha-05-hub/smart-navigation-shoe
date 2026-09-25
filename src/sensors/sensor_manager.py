"""Sensor manager coordinating LEFT, CENTER, and RIGHT distance sensors."""

from typing import Dict, List, Optional, Any
from .distance_sensor_interface import DistanceSensorInterface
from .simulated_sensor import SimulatedDistanceSensor
from .ultrasonic_sensor import UltrasonicSensor
from ..core.models import SensorReading
from ..core.enums import ObstacleZone, SensorStatus
from ..utils.logger import get_logger

logger = get_logger("sensor_manager")


class SensorManager:
    """Manages spatial distance sensors mounted on LEFT, CENTER, and RIGHT positions of the shoe.

    Note:
    Supports software simulation (Phase 3) and future physical HC-SR04 ultrasonic sensors (Phase 4).
    """

    def __init__(
        self,
        sensor_type: str = "simulated",
        min_distance_m: float = 0.2,
        max_distance_m: float = 4.0,
        default_left_m: float = 1.5,
        default_center_m: float = 0.8,
        default_right_m: float = 2.2,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.sensor_type = sensor_type
        self.min_distance_m = min_distance_m
        self.max_distance_m = max_distance_m
        self.sensors: Dict[ObstacleZone, DistanceSensorInterface] = {}
        trigger_pins = {"left": 23, "center": 17, "right": 22}
        echo_pins = {"left": 24, "center": 27, "right": 10}

        if config:
            sensor_cfg = config.get("sensors", {}).get("distance_sensor", {})
            self.sensor_type = sensor_cfg.get("type", sensor_type)
            self.min_distance_m = sensor_cfg.get("min_distance_m", min_distance_m)
            self.max_distance_m = sensor_cfg.get("max_distance_m", max_distance_m)

            pos_cfg = sensor_cfg.get("simulated_positions", {})
            if "left_cm" in pos_cfg:
                default_left_m = pos_cfg["left_cm"] / 100.0
            if "center_cm" in pos_cfg:
                default_center_m = pos_cfg["center_cm"] / 100.0
            if "right_cm" in pos_cfg:
                default_right_m = pos_cfg["right_cm"] / 100.0
            trigger_pins.update(sensor_cfg.get("gpio_trigger_pins", {}) or {})
            echo_pins.update(sensor_cfg.get("gpio_echo_pins", {}) or {})

        if self.sensor_type == "simulated":
            self.sensors = {
                ObstacleZone.LEFT: SimulatedDistanceSensor(
                    position=ObstacleZone.LEFT,
                    min_distance_m=self.min_distance_m,
                    max_distance_m=self.max_distance_m,
                    default_distance_m=default_left_m,
                    sensor_id="simulated_left_sensor",
                ),
                ObstacleZone.CENTER: SimulatedDistanceSensor(
                    position=ObstacleZone.CENTER,
                    min_distance_m=self.min_distance_m,
                    max_distance_m=self.max_distance_m,
                    default_distance_m=default_center_m,
                    sensor_id="simulated_center_sensor",
                ),
                ObstacleZone.RIGHT: SimulatedDistanceSensor(
                    position=ObstacleZone.RIGHT,
                    min_distance_m=self.min_distance_m,
                    max_distance_m=self.max_distance_m,
                    default_distance_m=default_right_m,
                    sensor_id="simulated_right_sensor",
                ),
            }
        else:
            # Ultrasonic hardware integration placeholder (pins from sensors.distance_sensor.gpio_*_pins)
            self.sensors = {
                zone: UltrasonicSensor(
                    trigger_pin=trigger_pins[zone.value.lower()],
                    echo_pin=echo_pins[zone.value.lower()],
                    position=zone,
                    sensor_id=f"pi_{zone.value.lower()}_ultrasonic",
                )
                for zone in (ObstacleZone.LEFT, ObstacleZone.CENTER, ObstacleZone.RIGHT)
            }

    def get_sensor(self, position: ObstacleZone) -> Optional[DistanceSensorInterface]:
        """Retrieves individual sensor instance for a specific zone."""
        return self.sensors.get(position)

    def read_all_sensors(self) -> Dict[ObstacleZone, SensorReading]:
        """Reads distance values across all spatial zones (LEFT, CENTER, RIGHT)."""
        readings = {}
        for zone in [ObstacleZone.LEFT, ObstacleZone.CENTER, ObstacleZone.RIGHT]:
            sensor = self.sensors.get(zone)
            if sensor:
                readings[zone] = sensor.read_distance()
            else:
                readings[zone] = SensorReading(
                    distance_m=-1.0,
                    position=zone,
                    status=SensorStatus.INVALID,
                    is_valid=False,
                )
        return readings

    def get_readings_list(self) -> List[SensorReading]:
        """Returns a list of all current sensor readings."""
        return list(self.read_all_sensors().values())

    def set_simulated_distances(self, left_m: float, center_m: float, right_m: float) -> None:
        """Sets distance values in meters for all three simulated sensors."""
        if self.sensor_type == "simulated":
            cast_left: Any = self.sensors[ObstacleZone.LEFT]
            cast_center: Any = self.sensors[ObstacleZone.CENTER]
            cast_right: Any = self.sensors[ObstacleZone.RIGHT]

            cast_left.set_simulated_distance(left_m)
            cast_center.set_simulated_distance(center_m)
            cast_right.set_simulated_distance(right_m)

    def set_simulated_distances_cm(self, left_cm: float, center_cm: float, right_cm: float) -> None:
        """Sets distance values in centimeters for all three simulated sensors."""
        self.set_simulated_distances(left_cm / 100.0, center_cm / 100.0, right_cm / 100.0)

    def format_sensor_display(self, display_unit: str = "cm") -> str:
        """Formats a human-readable display string clearly labeled as SIMULATED SENSOR DATA."""
        readings = self.read_all_sensors()
        lines = [
            "======================================================================",
            "  SIMULATED SENSOR DATA (Phase 3 Software Simulation)",
            "======================================================================",
        ]

        for zone in [ObstacleZone.LEFT, ObstacleZone.CENTER, ObstacleZone.RIGHT]:
            r = readings[zone]
            if display_unit.lower() == "cm":
                dist_str = f"{int(r.distance_cm)} cm ({r.distance_m:.2f} m)"
            else:
                dist_str = f"{r.distance_m:.2f} m ({int(r.distance_cm)} cm)"

            status_str = r.status.value
            lines.append(f"  {zone.value:<7} : {dist_str:<18} | Status: {status_str}")

        lines.append("======================================================================")
        return "\n".join(lines)
