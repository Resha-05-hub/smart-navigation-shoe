"""Unit tests for sensor fusion engine and distance sensor implementations."""

import unittest

from src.sensors.simulated_sensor import SimulatedDistanceSensor
from src.sensors.ultrasonic_sensor import UltrasonicSensor
from src.fusion.sensor_fusion import SensorFusionEngine
from src.core.models import DetectionItem, BoundingBox, SensorReading
from src.core.enums import ObstacleZone, RiskLevel


class TestSensorFusionArchitecture(unittest.TestCase):
    """Test suite for distance sensors and sensor fusion engine."""

    def test_simulated_distance_sensor(self):
        """Verify SimulatedDistanceSensor reading and value setting."""
        sensor = SimulatedDistanceSensor(min_distance_m=0.2, max_distance_m=4.0, default_distance_m=2.5)
        self.assertTrue(sensor.is_healthy())

        reading = sensor.read_distance()
        self.assertIsInstance(reading, SensorReading)
        self.assertEqual(reading.distance_m, 2.5)
        self.assertTrue(reading.is_valid)

        sensor.set_simulated_distance(1.2)
        self.assertEqual(sensor.read_distance().distance_m, 1.2)

    def test_ultrasonic_sensor_phase1_behavior(self):
        """Verify UltrasonicSensor returns invalid reading when GPIO is absent."""
        sensor = UltrasonicSensor(trigger_pin=23, echo_pin=24)
        self.assertFalse(sensor.is_healthy())

        reading = sensor.read_distance()
        self.assertFalse(reading.is_valid)
        self.assertEqual(reading.distance_m, -1.0)

    def test_sensor_fusion_combines_vision_and_distance(self):
        """Verify SensorFusionEngine combines DetectionItem and SensorReading."""
        fusion = SensorFusionEngine(default_range_m=2.5)
        bbox = BoundingBox(xmin=0.4, ymin=0.3, xmax=0.6, ymax=0.9)
        detection = DetectionItem(label="chair", confidence=0.85, bbox=bbox, zone=ObstacleZone.CENTER)
        sensor_reading = SensorReading(distance_m=1.5, sensor_id="simulated", is_valid=True)

        fused = fusion.fuse(detections=[detection], sensor_readings=[sensor_reading])

        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0].label, "chair")
        self.assertEqual(fused[0].distance_m, 1.5)
        self.assertEqual(fused[0].zone, ObstacleZone.CENTER)


if __name__ == "__main__":
    unittest.main()
