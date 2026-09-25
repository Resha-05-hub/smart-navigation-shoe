"""Unit tests for the simulated sensor scenario engine (software-only demonstration)."""

import unittest

from src.sensors.sensor_manager import SensorManager
from src.sensors.scenario_engine import SensorScenarioEngine, Scenario, Keyframe
from src.fusion.sensor_fusion import SensorFusionEngine
from src.decision.risk_analyzer import RiskAnalyzer
from src.core.enums import ObstacleZone, RiskLevel, SensorStatus
from src.utils.helpers import load_config


class FakeClock:
    """Manually advanced clock for deterministic timing tests."""

    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def approach_scenario() -> Scenario:
    """CENTER obstacle closes from 300 cm to 100 cm over 10 s."""
    return Scenario(
        name="approach",
        keyframes=[Keyframe(0, 300, 300, 300), Keyframe(10, 300, 100, 300)],
        description="test approach",
    )


class TestScenarioInterpolation(unittest.TestCase):
    """Scenario keyframe parsing and interpolation."""

    def test_linear_interpolation_between_keyframes(self):
        scenario = approach_scenario()
        self.assertEqual(scenario.distances_at(0)[ObstacleZone.CENTER], 300)
        self.assertAlmostEqual(scenario.distances_at(5)[ObstacleZone.CENTER], 200)
        self.assertEqual(scenario.distances_at(10)[ObstacleZone.CENTER], 100)
        self.assertEqual(scenario.distances_at(99)[ObstacleZone.CENTER], 100)  # Clamped past the end

    def test_fault_holds_until_next_keyframe(self):
        scenario = Scenario.from_config("fault", {"keyframes": [
            [0, 300, 120, 300], [3, 300, None, 300], [7, 300, 120, 300],
        ]})
        self.assertEqual(scenario.distances_at(2.9)[ObstacleZone.CENTER], 120)
        self.assertIsNone(scenario.distances_at(3)[ObstacleZone.CENTER])
        self.assertIsNone(scenario.distances_at(6.9)[ObstacleZone.CENTER])
        self.assertEqual(scenario.distances_at(7)[ObstacleZone.CENTER], 120)

    def test_from_config_sorts_and_validates(self):
        scenario = Scenario.from_config("s", {"keyframes": [[5, 1, 2, 3], [0, 4, 5, 6]]})
        self.assertEqual([k.time_s for k in scenario.keyframes], [0.0, 5.0])

        with self.assertRaises(ValueError):
            Scenario.from_config("bad", {"keyframes": [[0, 1, 2]]})
        with self.assertRaises(ValueError):
            Scenario.from_config("empty", {"keyframes": []})


class TestScenarioEngine(unittest.TestCase):
    """Driving SensorManager readings over time, by script and by keyboard."""

    def setUp(self):
        self.clock = FakeClock()
        self.manager = SensorManager(default_left_m=3.0, default_center_m=3.0, default_right_m=3.0)
        self.engine = SensorScenarioEngine(
            self.manager,
            scenarios={"approach": approach_scenario()},
            clock=self.clock,
        )

    def center_cm(self) -> float:
        return self.manager.read_all_sensors()[ObstacleZone.CENTER].distance_cm

    def test_starts_in_manual_mode_with_sensor_distances(self):
        self.engine.update()
        self.assertEqual(self.engine.mode, SensorScenarioEngine.MANUAL)
        self.assertEqual(self.engine.status_text, "MANUAL")
        self.assertEqual(self.center_cm(), 300.0)

    def test_scripted_scenario_moves_sensor_over_time(self):
        self.engine.start_scenario("approach", loop=False)
        self.clock.advance(5)
        self.engine.update()
        self.assertEqual(self.center_cm(), 200.0)
        self.assertIn("SCENARIO approach 5.0s", self.engine.status_text)

        self.clock.advance(6)
        self.engine.update()
        self.assertEqual(self.center_cm(), 100.0)
        self.assertTrue(self.engine.finished)

    def test_looping_scenario_restarts(self):
        self.engine.start_scenario("approach", loop=True)
        self.clock.advance(12.5)  # 2.5 s into the second pass
        self.engine.update()
        self.assertEqual(self.center_cm(), 250.0)
        self.assertFalse(self.engine.finished)

    def test_pause_freezes_scenario_time(self):
        self.engine.start_scenario("approach", loop=False)
        self.clock.advance(2)
        self.engine.toggle_pause()
        self.clock.advance(100)
        self.engine.update()
        self.assertEqual(self.center_cm(), 260.0)
        self.assertIn("PAUSED", self.engine.status_text)

        self.engine.toggle_pause()
        self.clock.advance(3)
        self.engine.update()
        self.assertEqual(self.center_cm(), 200.0)

    def test_keyboard_moves_zones_closer_and_farther(self):
        for _ in range(3):
            self.assertTrue(self.engine.handle_key(ord("s")))  # CENTER closer
        self.assertTrue(self.engine.handle_key(ord("a")))      # LEFT closer
        self.assertTrue(self.engine.handle_key(ord("c")))      # RIGHT farther
        self.engine.update()

        readings = self.manager.read_all_sensors()
        self.assertEqual(readings[ObstacleZone.CENTER].distance_cm, 270.0)
        self.assertEqual(readings[ObstacleZone.LEFT].distance_cm, 290.0)
        self.assertEqual(readings[ObstacleZone.RIGHT].distance_cm, 310.0)

    def test_manual_adjustment_clamped_to_sensor_range(self):
        self.engine.set_distances_cm(25, 300, 395)
        self.engine.handle_key(ord("a"))
        self.engine.handle_key(ord("c"))
        self.engine.update()
        readings = self.manager.read_all_sensors()
        self.assertEqual(readings[ObstacleZone.LEFT].distance_cm, 20.0)    # min_distance_m 0.2
        self.assertEqual(readings[ObstacleZone.RIGHT].distance_cm, 400.0)  # max_distance_m 4.0

    def test_manual_key_during_scenario_keeps_current_distance(self):
        self.engine.start_scenario("approach")
        self.clock.advance(5)
        self.engine.update()  # CENTER at 200 cm

        self.engine.handle_key(ord("s"))
        self.engine.update()
        self.assertEqual(self.engine.mode, SensorScenarioEngine.MANUAL)
        self.assertEqual(self.center_cm(), 190.0)

    def test_clear_reset_and_unhandled_keys(self):
        self.engine.set_distances_cm(50, 50, 50)
        self.engine.handle_key(ord("0"))
        self.engine.update()
        self.assertEqual(self.center_cm(), 400.0)

        self.engine.handle_key(ord("r"))
        self.engine.update()
        self.assertEqual(self.center_cm(), 300.0)

        self.assertFalse(self.engine.handle_key(ord("q")))  # Quit stays with the caller
        self.assertFalse(self.engine.handle_key(255))       # cv2.waitKey "no key"

    def test_next_scenario_cycles(self):
        self.engine.scenarios["second"] = Scenario("second", [Keyframe(0, 100, 100, 100)])
        self.assertEqual(self.engine.next_scenario(), "approach")
        self.assertEqual(self.engine.next_scenario(), "second")
        self.assertEqual(self.engine.next_scenario(), "approach")

    def test_unknown_scenario_raises(self):
        with self.assertRaises(KeyError):
            self.engine.start_scenario("missing")

    def test_fault_marks_sensor_invalid(self):
        self.engine.set_distances_cm(300, None, 300)
        self.engine.update()
        reading = self.manager.read_all_sensors()[ObstacleZone.CENTER]
        self.assertFalse(reading.is_valid)
        self.assertEqual(reading.status, SensorStatus.INVALID)
        self.assertEqual(self.engine.distances_text, "L 300cm | C FAULT | R 300cm")

    def test_noise_is_bounded_and_seeded(self):
        values = []
        for _ in range(2):
            manager = SensorManager(default_left_m=3.0, default_center_m=1.0, default_right_m=3.0)
            engine = SensorScenarioEngine(manager, noise_cm=1.0, seed=7)
            values.append([engine.update()[ObstacleZone.CENTER] for _ in range(50)])

        self.assertEqual(values[0], values[1])
        self.assertTrue(all(90.0 < v < 110.0 for v in values[0]))
        self.assertGreater(len(set(values[0])), 1)

    def test_disabled_for_non_simulated_sensors(self):
        manager = SensorManager(sensor_type="ultrasonic")
        engine = SensorScenarioEngine(manager, initial_cm=(300, 300, 300))
        self.assertFalse(engine.enabled)
        engine.update()  # Must not touch hardware sensors


class TestScenarioEngineConfigAndPipeline(unittest.TestCase):
    """Scenarios from config.yaml driving fusion and risk end to end."""

    def setUp(self):
        self.config = load_config()
        self.clock = FakeClock()
        self.manager = SensorManager(config=self.config)
        self.engine = SensorScenarioEngine(self.manager, config=self.config, clock=self.clock)
        self.fusion = SensorFusionEngine(config=self.config)
        self.risk = RiskAnalyzer(config=self.config)
        self.engine.noise_cm = 0.0

    def overall_risk_at(self, seconds: float) -> RiskLevel:
        self.clock.now = self.engine._start_time + seconds
        self.engine.update()
        fused = self.fusion.fuse([], self.manager.get_readings_list())
        return self.risk.evaluate(fused).overall_risk_level

    def test_config_scenarios_loaded_and_startup_is_clear(self):
        for name in ("approaching_obstacle", "passing_left_then_right", "narrow_corridor",
                     "mixed_zones", "sensor_fault"):
            self.assertIn(name, self.engine.scenario_names)

        self.assertEqual(self.engine.mode, SensorScenarioEngine.MANUAL)
        self.engine.update()
        fused = self.fusion.fuse([], self.manager.get_readings_list())
        self.assertEqual(self.risk.evaluate(fused).overall_risk_level, RiskLevel.SAFE)

    def test_approaching_obstacle_escalates_risk(self):
        self.engine.start_scenario("approaching_obstacle", loop=False)
        # CENTER: 300 cm (0 s) -> ~184 cm (5 s) -> ~69 cm (8 s) -> 30 cm (10 s) -> 300 cm (14 s)
        observed = [self.overall_risk_at(t) for t in (0, 5, 8, 10, 14)]
        self.assertEqual(
            observed,
            [RiskLevel.SAFE, RiskLevel.CAUTION, RiskLevel.WARNING, RiskLevel.DANGER, RiskLevel.SAFE],
        )

    def test_sensor_fault_scenario_reports_invalid_center(self):
        self.engine.start_scenario("sensor_fault", loop=False)
        self.clock.now = self.engine._start_time + 4
        self.engine.update()
        self.assertFalse(self.manager.read_all_sensors()[ObstacleZone.CENTER].is_valid)


if __name__ == "__main__":
    unittest.main()
