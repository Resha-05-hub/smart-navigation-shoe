"""Unit tests for Phase 3 simulated distance sensors and SensorManager."""

import unittest
from src.sensors.simulated_sensor import SimulatedDistanceSensor
from src.sensors.sensor_manager import SensorManager
from src.core.models import SensorReading
from src.core.enums import ObstacleZone, SensorStatus


class TestSimulatedSensorsPhase3(unittest.TestCase):
    """Test suite for Phase 3 simulated distance sensors."""

    def test_valid_distance_reading(self):
        """Verify healthy distance reading in meters and centimeters."""
        sensor = SimulatedDistanceSensor(
            position=ObstacleZone.LEFT,
            min_distance_m=0.2,
            max_distance_m=4.0,
            default_distance_m=1.5,
        )
        reading = sensor.read_distance()

        self.assertEqual(reading.position, ObstacleZone.LEFT)
        self.assertEqual(reading.distance_m, 1.5)
        self.assertEqual(reading.distance_cm, 150.0)
        self.assertEqual(reading.status, SensorStatus.HEALTHY)
        self.assertTrue(reading.is_valid)

    def test_invalid_negative_distance_reading(self):
        """Verify negative distance results in INVALID status."""
        sensor = SimulatedDistanceSensor(position=ObstacleZone.CENTER)
        sensor.set_simulated_distance(-1.0)
        reading = sensor.read_distance()

        self.assertEqual(reading.status, SensorStatus.INVALID)
        self.assertFalse(reading.is_valid)

    def test_invalid_zero_distance_reading(self):
        """Verify zero distance results in INVALID status."""
        sensor = SimulatedDistanceSensor(position=ObstacleZone.CENTER)
        sensor.set_simulated_distance(0.0)
        reading = sensor.read_distance()

        self.assertEqual(reading.status, SensorStatus.INVALID)
        self.assertFalse(reading.is_valid)

    def test_out_of_range_distance_reading(self):
        """Verify distance exceeding maximum limit results in OUT_OF_RANGE status."""
        sensor = SimulatedDistanceSensor(
            position=ObstacleZone.RIGHT,
            min_distance_m=0.2,
            max_distance_m=4.0,
            default_distance_m=2.2,
        )
        sensor.set_simulated_distance_cm(500.0)  # 5.0 m > 4.0 m
        reading = sensor.read_distance()

        self.assertEqual(reading.status, SensorStatus.OUT_OF_RANGE)
        self.assertFalse(reading.is_valid)

    def test_sensor_manager_three_positions(self):
        """Verify SensorManager manages LEFT, CENTER, and RIGHT sensors."""
        manager = SensorManager(
            default_left_m=1.5,
            default_center_m=0.8,
            default_right_m=2.2,
        )
        readings = manager.read_all_sensors()

        self.assertEqual(len(readings), 3)
        self.assertIn(ObstacleZone.LEFT, readings)
        self.assertIn(ObstacleZone.CENTER, readings)
        self.assertIn(ObstacleZone.RIGHT, readings)

        self.assertEqual(readings[ObstacleZone.LEFT].distance_cm, 150.0)
        self.assertEqual(readings[ObstacleZone.CENTER].distance_cm, 80.0)
        self.assertEqual(readings[ObstacleZone.RIGHT].distance_cm, 220.0)

    def test_sensor_manager_set_distances_cm(self):
        """Verify setting simulated distance in centimeters across sensors."""
        manager = SensorManager()
        manager.set_simulated_distances_cm(120.0, 90.0, 310.0)
        readings = manager.read_all_sensors()

        self.assertEqual(readings[ObstacleZone.LEFT].distance_cm, 120.0)
        self.assertEqual(readings[ObstacleZone.CENTER].distance_cm, 90.0)
        self.assertEqual(readings[ObstacleZone.RIGHT].distance_cm, 310.0)

    def test_format_sensor_display_output(self):
        """Verify format_sensor_display output string contains expected labels."""
        manager = SensorManager(default_left_m=1.5, default_center_m=0.8, default_right_m=2.2)
        display_str = manager.format_sensor_display(display_unit="cm")

        self.assertIn("SIMULATED SENSOR DATA", display_str)
        self.assertIn("LEFT", display_str)
        self.assertIn("150 cm", display_str)
        self.assertIn("CENTER", display_str)
        self.assertIn("80 cm", display_str)
        self.assertIn("RIGHT", display_str)
        self.assertIn("220 cm", display_str)


if __name__ == "__main__":
    unittest.main()
