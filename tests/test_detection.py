"""Unit tests for detection interface and camera module architecture."""

import unittest
from src.detection.yolo_detector import YoloDetector
from src.detection.detection_result import DetectionResult
from src.camera.webcam_camera import WebcamCamera
from src.camera.raspberry_pi_camera import RaspberryPiCamera
from src.core.models import BoundingBox, DetectionItem
from src.core.enums import ObstacleZone


class TestDetectionArchitecture(unittest.TestCase):
    """Test suite for detection and camera modules."""

    def test_yolo_detector_interface_instantiation(self):
        """Verify YoloDetector can be instantiated with default parameters."""
        detector = YoloDetector(model_path="models/yolov8n.pt", confidence_threshold=0.5)
        self.assertIsNotNone(detector)
        self.assertEqual(detector.confidence_threshold, 0.5)
        self.assertFalse(detector.is_loaded())

    def test_detection_result_structure(self):
        """Verify DetectionResult dataclass structure and properties."""
        bbox = BoundingBox(xmin=0.1, ymin=0.2, xmax=0.4, ymax=0.8)
        item = DetectionItem(label="person", confidence=0.92, bbox=bbox, zone=ObstacleZone.CENTER)
        result = DetectionResult(detections=[item], frame_width=640, frame_height=480)

        self.assertEqual(result.count, 1)
        self.assertEqual(result.detections[0].label, "person")
        self.assertAlmostEqual(result.detections[0].bbox.center_x, 0.25, places=5)

    def test_webcam_camera_simulation(self):
        """Verify WebcamCamera fallback simulation mode and is_connected property."""
        camera = WebcamCamera(device_id=999, width=640, height=480, simulation_fallback=True)
        connected = camera.connect()
        self.assertTrue(connected)
        self.assertTrue(camera.is_opened())
        self.assertTrue(camera.is_connected)

        success, frame = camera.read_frame()
        self.assertTrue(success)
        self.assertIsNotNone(frame)

        camera.release()
        self.assertFalse(camera.is_opened())
        self.assertFalse(camera.is_connected)

    def test_raspberry_pi_camera_phase1_behavior(self):
        """Verify RaspberryPiCamera handles non-Pi platform in Phase 1 cleanly."""
        pi_cam = RaspberryPiCamera()
        connected = pi_cam.connect()
        self.assertFalse(connected)
        self.assertFalse(pi_cam.is_opened())
        self.assertFalse(pi_cam.is_connected)



if __name__ == "__main__":
    unittest.main()
