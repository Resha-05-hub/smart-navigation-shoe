"""Unit tests for camera-linked simulated sensors (monocular distance estimation)."""

import math
import unittest

from src.sensors.sensor_manager import SensorManager
from src.sensors.scenario_engine import SensorScenarioEngine
from src.sensors.vision_distance_estimator import VisionDistanceEstimator
from src.fusion.sensor_fusion import SensorFusionEngine
from src.decision.risk_analyzer import RiskAnalyzer
from src.core.models import BoundingBox, DetectionItem
from src.core.enums import ObstacleZone, RiskLevel

FRAME_W, FRAME_H = 640, 480
FOCAL_60 = 320 / math.tan(math.radians(30))  # ~554.3 px for a 60 deg horizontal FOV


def person_at(distance_m: float, center_x: float = 320.0, label: str = "person",
              zone: ObstacleZone = ObstacleZone.CENTER) -> DetectionItem:
    """Fully visible box sized as a 1.7 m x 0.45 m person would appear at distance_m."""
    height_px = 1.7 * FOCAL_60 / distance_m
    width_px = 0.45 * FOCAL_60 / distance_m
    top = 240 - height_px / 2
    return DetectionItem(
        label=label,
        confidence=0.9,
        bbox=BoundingBox(center_x - width_px / 2, top, center_x + width_px / 2, top + height_px),
        zone=zone,
    )


class FakeClock:
    def __init__(self) -> None:
        self.now = 50.0

    def __call__(self) -> float:
        return self.now


class TestVisionDistanceEstimator(unittest.TestCase):
    """Pinhole-model distance estimation."""

    def setUp(self):
        self.estimator = VisionDistanceEstimator(horizontal_fov_deg=60.0)

    def test_focal_length_from_fov(self):
        self.assertAlmostEqual(self.estimator.focal_length_px(FRAME_W), FOCAL_60, places=3)

    def test_fully_visible_person_distance(self):
        for distance in (1.5, 3.0):
            estimate = self.estimator.estimate_distance_m(person_at(distance), FRAME_W, FRAME_H)
            self.assertAlmostEqual(estimate, distance, places=3)

    def test_truncated_person_uses_width(self):
        """A person close to a laptop webcam: box cut off at the bottom, shoulders ~350 px wide."""
        seated = DetectionItem("person", 0.9, BoundingBox(145, 80, 495, 480), ObstacleZone.CENTER)
        estimate = self.estimator.estimate_distance_m(seated, FRAME_W, FRAME_H)
        # Height alone (400 px) would say ~2.4 m; width gives the realistic ~0.7 m
        self.assertAlmostEqual(estimate, 0.45 * FOCAL_60 / 350, places=3)
        self.assertLess(estimate, 1.0)

    def test_unknown_class_uses_default_size_and_config_overrides(self):
        box = DetectionItem("fire hydrant", 0.9, BoundingBox(300, 200, 355.43, 310.86), ObstacleZone.CENTER)
        self.assertAlmostEqual(self.estimator.estimate_distance_m(box, FRAME_W, FRAME_H), 5.0, places=2)

        estimator = VisionDistanceEstimator(config={"sensors": {"simulation": {"camera_linked": {
            "horizontal_fov_deg": 70.0,
            "object_sizes_m": {"fire hydrant": [0.8, 0.4]},
        }}}})
        self.assertEqual(estimator.horizontal_fov_deg, 70.0)
        self.assertEqual(estimator.object_sizes_m["fire hydrant"], (0.8, 0.4))
        self.assertEqual(estimator.object_sizes_m["person"], (1.70, 0.45))  # Defaults kept

    def test_normalized_bbox_supported(self):
        pixel = person_at(2.0)
        b = pixel.bbox
        normalized = DetectionItem("person", 0.9, BoundingBox(
            b.xmin / FRAME_W, b.ymin / FRAME_H, b.xmax / FRAME_W, b.ymax / FRAME_H), ObstacleZone.CENTER)
        self.assertAlmostEqual(self.estimator.estimate_distance_m(normalized, FRAME_W, FRAME_H), 2.0, places=3)

    def test_degenerate_boxes_ignored(self):
        flat = DetectionItem("person", 0.9, BoundingBox(10, 10, 10.5, 200), ObstacleZone.LEFT)
        no_box = DetectionItem("person", 0.9, None, ObstacleZone.LEFT)  # type: ignore[arg-type]
        self.assertIsNone(self.estimator.estimate_distance_m(flat, FRAME_W, FRAME_H))
        self.assertIsNone(self.estimator.estimate_distance_m(no_box, FRAME_W, FRAME_H))

    def test_zone_distances_report_nearest_object(self):
        detections = [
            person_at(3.0, center_x=320),
            person_at(1.2, center_x=330),
            person_at(2.0, center_x=100, zone=ObstacleZone.LEFT),
        ]
        zones = self.estimator.zone_distances_cm(detections, FRAME_W, FRAME_H)
        self.assertAlmostEqual(zones[ObstacleZone.CENTER], 120.0, places=1)
        self.assertAlmostEqual(zones[ObstacleZone.LEFT], 200.0, places=1)
        self.assertIsNone(zones[ObstacleZone.RIGHT])


class TestCameraLinkedEngine(unittest.TestCase):
    """Scenario engine in camera-linked mode."""

    def setUp(self):
        self.clock = FakeClock()
        self.manager = SensorManager(default_left_m=3.0, default_center_m=3.0, default_right_m=3.0)
        self.engine = SensorScenarioEngine(
            self.manager,
            estimator=VisionDistanceEstimator(horizontal_fov_deg=60.0),
            smoothing=0.5,
            hold_s=0.5,
            clock=self.clock,
        )
        self.engine.set_camera_linked()

    def readings_cm(self):
        return {zone: r.distance_cm for zone, r in self.manager.read_all_sensors().items()}

    def step(self, detections, dt: float = 0.1):
        self.clock.now += dt
        self.engine.update(detections, (FRAME_W, FRAME_H))
        return self.readings_cm()

    def test_detected_zone_reads_estimate_and_others_clear(self):
        readings = self.step([person_at(1.5)])
        self.assertAlmostEqual(readings[ObstacleZone.CENTER], 150.0, places=0)
        self.assertEqual(readings[ObstacleZone.LEFT], 400.0)
        self.assertEqual(readings[ObstacleZone.RIGHT], 400.0)
        self.assertEqual(self.engine.status_text, "CAMERA-LINKED (estimated)")

    def test_estimates_are_smoothed_while_tracking(self):
        self.step([person_at(2.0)])
        readings = self.step([person_at(1.0)])
        self.assertAlmostEqual(readings[ObstacleZone.CENTER], 150.0, places=0)  # 0.5*100 + 0.5*200

    def test_flicker_held_then_cleared(self):
        self.step([person_at(1.0)])
        held = self.step([], dt=0.3)
        self.assertAlmostEqual(held[ObstacleZone.CENTER], 100.0, places=0)

        cleared = self.step([], dt=0.3)  # 0.6 s since last detection > hold_s
        self.assertEqual(cleared[ObstacleZone.CENTER], 400.0)

        reappeared = self.step([person_at(0.8)])  # New object reported without smoothing lag
        self.assertAlmostEqual(reappeared[ObstacleZone.CENTER], 80.0, places=0)

    def test_estimates_clamped_to_sensor_range(self):
        far = self.step([person_at(8.0)])
        self.assertEqual(far[ObstacleZone.CENTER], 400.0)

    def test_update_without_frame_size_keeps_last_distances(self):
        self.step([person_at(1.0)])
        self.clock.now += 0.1
        self.engine.update()
        self.assertAlmostEqual(self.readings_cm()[ObstacleZone.CENTER], 100.0, places=0)

    def test_v_toggles_and_manual_keys_take_over_from_camera(self):
        self.step([person_at(1.0)])
        self.assertTrue(self.engine.handle_key(ord("s")))  # Manual key: keep 100 cm, then 10 cm closer
        self.assertEqual(self.engine.mode, SensorScenarioEngine.MANUAL)
        self.engine.update()
        self.assertAlmostEqual(self.readings_cm()[ObstacleZone.CENTER], 90.0, places=0)

        self.assertTrue(self.engine.handle_key(ord("v")))
        self.assertEqual(self.engine.mode, SensorScenarioEngine.CAMERA)
        self.assertTrue(self.engine.handle_key(ord("v")))
        self.assertEqual(self.engine.mode, SensorScenarioEngine.MANUAL)

    def test_reset_returns_to_camera_startup_mode(self):
        engine = SensorScenarioEngine(
            self.manager,
            config={"sensors": {"simulation": {"startup": "camera"}}},
            clock=self.clock,
        )
        self.assertEqual(engine.mode, SensorScenarioEngine.CAMERA)
        engine.handle_key(ord("s"))
        self.assertEqual(engine.mode, SensorScenarioEngine.MANUAL)
        engine.handle_key(ord("r"))
        self.assertEqual(engine.mode, SensorScenarioEngine.CAMERA)

    def test_person_walking_toward_camera_escalates_risk(self):
        fusion = SensorFusionEngine(default_range_m=2.5)
        risk = RiskAnalyzer()
        observed = []
        for distance in (3.0, 3.0, 1.6, 1.6, 0.8, 0.8, 0.4, 0.4, 0.4):
            detections = [person_at(distance)]
            self.step(detections)
            fused = fusion.fuse(detections, self.manager.get_readings_list())
            observed.append(risk.evaluate(fused).overall_risk_level)

        self.assertEqual(observed[0], RiskLevel.SAFE)
        self.assertIn(RiskLevel.CAUTION, observed)
        self.assertIn(RiskLevel.WARNING, observed)
        self.assertEqual(observed[-1], RiskLevel.DANGER)


class TestConfiguredObjectSizes(unittest.TestCase):
    """bottle, laptop, and cell phone use their configured sizes, not the 1.0 m x 0.5 m default."""

    CLASSES = ("bottle", "laptop", "cell phone")

    def setUp(self):
        from src.utils.helpers import load_config
        self.config = load_config()
        self.configured = self.config["sensors"]["simulation"]["camera_linked"]["object_sizes_m"]
        self.estimator = VisionDistanceEstimator(config=self.config)

    def box_at(self, label: str, size_m, distance_m: float) -> DetectionItem:
        """Fully visible box of a real (height, width) object at distance_m, 60 deg FOV camera."""
        height_px = size_m[0] * FOCAL_60 / distance_m
        width_px = size_m[1] * FOCAL_60 / distance_m
        return DetectionItem(
            label, 0.9,
            BoundingBox(320 - width_px / 2, 240 - height_px / 2, 320 + width_px / 2, 240 + height_px / 2),
            ObstacleZone.CENTER,
        )

    def test_sizes_loaded_from_config(self):
        for label in self.CLASSES:
            self.assertIn(label, self.configured)
            self.assertEqual(self.estimator.object_sizes_m[label], tuple(self.configured[label]))
            self.assertNotEqual(self.estimator.object_sizes_m[label], self.estimator.default_size_m)

    def test_estimate_uses_configured_size(self):
        for label in self.CLASSES:
            size = self.configured[label]
            for true_m in (0.5, 1.0, 2.0):
                estimate = self.estimator.estimate_distance_m(self.box_at(label, size, true_m), FRAME_W, FRAME_H)
                self.assertAlmostEqual(estimate, true_m, places=2, msg=f"{label} at {true_m} m")

    def test_default_size_would_misjudge_small_objects(self):
        """Without the configured sizes these objects read far away (e.g. a phone at 0.5 m as ~3.3 m)."""
        fallback = VisionDistanceEstimator(horizontal_fov_deg=60.0)
        for label in ("bottle", "cell phone"):
            det = self.box_at(label, self.configured[label], 0.5)
            self.assertGreater(fallback.estimate_distance_m(det, FRAME_W, FRAME_H), 1.9)
            self.assertAlmostEqual(self.estimator.estimate_distance_m(det, FRAME_W, FRAME_H), 0.5, places=2)


if __name__ == "__main__":
    unittest.main()
