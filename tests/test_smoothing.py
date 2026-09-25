"""Unit tests for step 7 flicker reduction: risk hysteresis, path-clear confirmation, and object tracking."""

import unittest
from unittest.mock import MagicMock, patch

from src.alerts.alert_manager import AlertManager
from src.alerts.simulated_vibration import SimulatedVibration
from src.core.enums import ObstacleZone, RiskLevel
from src.core.models import BoundingBox, DetectionItem, FusedObstacle, RiskAssessment, SensorReading
from src.decision.risk_analyzer import RiskAnalyzer
from src.detection.object_tracker import ObjectTracker, iou
from src.fusion.sensor_fusion import SensorFusionEngine
from src.sensors.scenario_engine import SensorScenarioEngine
from src.sensors.sensor_manager import SensorManager
from src.utils.helpers import load_config


def obstacle(distance_m: float, zone: ObstacleZone = ObstacleZone.CENTER) -> FusedObstacle:
    return FusedObstacle("o", "obstacle", 0.0, None, distance_m, zone)


class TestRiskHysteresis(unittest.TestCase):

    def setUp(self):
        self.analyzer = RiskAnalyzer(hysteresis_m=0.1)

    def level(self, distance_m, zone=ObstacleZone.CENTER):
        return self.analyzer.evaluate([obstacle(distance_m, zone)]).overall_risk_level

    def test_escalates_immediately(self):
        self.assertEqual(self.level(2.5), RiskLevel.SAFE)
        self.assertEqual(self.level(1.99), RiskLevel.CAUTION)
        self.assertEqual(self.level(0.49), RiskLevel.DANGER)

    def test_deescalates_only_past_threshold_plus_margin(self):
        self.assertEqual(self.level(0.95), RiskLevel.WARNING)
        self.assertEqual(self.level(1.05), RiskLevel.WARNING)  # Within 10 cm of the 1.0 m boundary
        self.assertEqual(self.level(1.09), RiskLevel.WARNING)
        self.assertEqual(self.level(1.11), RiskLevel.CAUTION)

    def test_zone_state_resets_when_zone_is_empty(self):
        self.assertEqual(self.level(0.95), RiskLevel.WARNING)
        self.analyzer.evaluate([])
        self.assertEqual(self.level(1.05), RiskLevel.CAUTION)

    def test_zones_are_independent(self):
        self.analyzer.evaluate([obstacle(0.95, ObstacleZone.LEFT)])
        left, right = obstacle(0.95, ObstacleZone.LEFT), obstacle(1.05, ObstacleZone.RIGHT)
        self.analyzer.evaluate([left, right])
        self.assertEqual(left.risk_level, RiskLevel.WARNING)
        self.assertEqual(right.risk_level, RiskLevel.CAUTION)  # RIGHT had no WARNING history to hold

        right_only = self.analyzer.evaluate([obstacle(1.05, ObstacleZone.RIGHT)])
        self.assertEqual(right_only.overall_risk_level, RiskLevel.CAUTION)

    def test_classify_obstacle_risk_stays_stateless(self):
        self.analyzer.evaluate([obstacle(0.95)])
        self.assertEqual(self.analyzer.classify_obstacle_risk(obstacle(1.05)), RiskLevel.CAUTION)

    def test_hysteresis_from_config(self):
        analyzer = RiskAnalyzer(config={"risk_analysis": {"hysteresis_cm": 15.0}})
        self.assertAlmostEqual(analyzer.hysteresis_m, 0.15)
        self.assertEqual(RiskAnalyzer().hysteresis_m, 0.0)


class TestSensorOnlyHysteresis(unittest.TestCase):

    def test_sensor_only_obstacle_held_within_margin(self):
        fusion = SensorFusionEngine(sensor_only_max_m=2.0, hysteresis_m=0.1)

        def fuse(distance_m):
            return fusion.fuse([], [SensorReading(distance_m, position=ObstacleZone.CENTER)])

        self.assertEqual(len(fuse(2.05)), 0)  # Not yet reported: normal 2.0 m limit applies
        self.assertEqual(len(fuse(1.99)), 1)
        self.assertEqual(len(fuse(2.05)), 1)  # Already reported: held until 2.1 m
        self.assertEqual(len(fuse(2.11)), 0)


class TestStaticObstacleNearThreshold(unittest.TestCase):
    """Regression: a stationary obstacle at a boundary with sensor noise must not flicker."""

    def test_static_obstacle_at_boundary_is_stable(self):
        config = load_config()
        for distance_cm in (50, 100, 200):
            manager = SensorManager(config=config)
            engine = SensorScenarioEngine(manager, config=config, seed=3)
            engine.noise_cm = 1.0
            engine.set_distances_cm(400, distance_cm, 400)
            fusion, risk = SensorFusionEngine(config=config), RiskAnalyzer(config=config)

            levels = []
            for _ in range(100):
                engine.update()
                readings = manager.get_readings_list()
                levels.append(risk.evaluate(fusion.fuse([], readings), readings).overall_risk_level)

            changes = sum(1 for a, b in zip(levels, levels[1:]) if a != b)
            self.assertLessEqual(changes, 1, f"{distance_cm} cm flickered {changes} times")


class TestPathClearConfirmation(unittest.TestCase):

    def setUp(self):
        self.now = [100.0]
        self.voice = MagicMock()
        self.manager = AlertManager(SimulatedVibration(), self.voice, clear_confirm_seconds=1.0)
        self.patcher = patch("src.alerts.alert_manager.time.time", lambda: self.now[0])
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def step(self, level, dt=0.1):
        self.now[0] += dt
        obstacles = [] if level == RiskLevel.SAFE else [
            FusedObstacle("1", "person", 0.9, None, 0.8, ObstacleZone.CENTER, level)
        ]
        _, status = self.manager.evaluate_and_trigger(RiskAssessment(overall_risk_level=level), obstacles)
        return status

    def test_path_clear_waits_for_confirmation(self):
        self.assertIn("[ALERT TRIGGERED]", self.step(RiskLevel.WARNING))
        self.assertIn("Confirming path clear", self.step(RiskLevel.SAFE))
        self.assertIn("Confirming path clear", self.step(RiskLevel.SAFE, dt=0.5))
        self.assertIn("Path clear.", self.step(RiskLevel.SAFE, dt=0.6))

    def test_one_frame_dropout_never_says_clear(self):
        self.step(RiskLevel.WARNING)
        self.step(RiskLevel.SAFE)  # Detection dropped for one frame
        self.step(RiskLevel.WARNING)
        spoken = [c.args[0] for c in self.voice.speak.call_args_list]
        self.assertNotIn("Path clear.", spoken)


def box(x, y=100, w=100, h=200) -> BoundingBox:
    return BoundingBox(x, y, x + w, y + h)


def det(x, label="person", **kw) -> DetectionItem:
    return DetectionItem(label=label, confidence=0.9, bbox=box(x, **kw))


class TestObjectTracker(unittest.TestCase):

    def setUp(self):
        self.tracker = ObjectTracker(iou_threshold=0.3, max_missed=2, min_hits=2, box_smoothing=1.0)

    def test_iou(self):
        self.assertAlmostEqual(iou(box(0), box(0)), 1.0)
        self.assertEqual(iou(box(0), box(500)), 0.0)
        self.assertAlmostEqual(iou(box(0), box(50)), 1 / 3)

    def test_new_object_reported_after_min_hits_with_stable_id(self):
        self.assertEqual(self.tracker.update([det(100)], 640), [])  # First sighting: not yet confirmed
        first = self.tracker.update([det(105)], 640)
        second = self.tracker.update([det(110)], 640)
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0].track_id, second[0].track_id)
        self.assertGreaterEqual(first[0].track_id, 1)

    def test_one_frame_ghost_is_never_reported(self):
        self.assertEqual(self.tracker.update([det(100)], 640), [])
        self.assertEqual(self.tracker.update([], 640), [])

    def test_missed_frames_coast_then_drop(self):
        self.tracker.update([det(100)], 640)
        self.tracker.update([det(100)], 640)
        self.assertEqual(len(self.tracker.update([], 640)), 1)  # Miss 1: coasting
        self.assertEqual(len(self.tracker.update([], 640)), 1)  # Miss 2: coasting
        self.assertEqual(self.tracker.update([], 640), [])      # Miss 3 > max_missed: dropped

    def test_matching_is_class_aware(self):
        self.tracker.update([det(100)], 640)
        self.tracker.update([det(100)], 640)
        out = self.tracker.update([det(100, label="chair")], 640)
        self.assertEqual([d.label for d in out], ["person"])  # Chair is a new, unconfirmed track

    def test_two_objects_keep_their_own_ids(self):
        self.tracker.update([det(50), det(400)], 640)
        a = self.tracker.update([det(55), det(395)], 640)
        b = self.tracker.update([det(390), det(60)], 640)  # Detection order swapped
        ids_a = {round(d.bbox.xmin): d.track_id for d in a}
        ids_b = {round(d.bbox.xmin): d.track_id for d in b}
        self.assertEqual(ids_a[55], ids_b[60])
        self.assertEqual(ids_a[395], ids_b[390])

    def test_box_smoothing_and_zone_from_smoothed_box(self):
        tracker = ObjectTracker(min_hits=1, box_smoothing=0.5)
        tracker.update([det(100)], 640)
        out = tracker.update([det(140)], 640)
        self.assertAlmostEqual(out[0].bbox.xmin, 120.0)
        self.assertEqual(out[0].zone, ObstacleZone.LEFT)  # Center x 170 / 640 < 0.33

    def test_config_and_reset(self):
        tracker = ObjectTracker(config={"tracking": {"min_hits": 3, "max_missed_frames": 5}})
        self.assertEqual((tracker.min_hits, tracker.max_missed), (3, 5))
        tracker.update([det(100)], 640)
        tracker.reset()
        self.assertEqual(tracker.tracks, [])


class TestTrackIdsInFusion(unittest.TestCase):

    def test_fused_obstacle_uses_track_id(self):
        item = DetectionItem("person", 0.9, box(250), ObstacleZone.CENTER, track_id=7)
        fused = SensorFusionEngine().fuse([item], [SensorReading(0.8, position=ObstacleZone.CENTER)])
        self.assertEqual(fused[0].object_id, "track_7")


if __name__ == "__main__":
    unittest.main()
