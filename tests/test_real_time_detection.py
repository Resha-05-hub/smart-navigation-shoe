"""Unit tests for Phase 2 real-time object detection logic."""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from src.detection.yolo_detector import YoloDetector
from src.detection.detection_result import DetectionResult
from src.core.models import DetectionItem, BoundingBox
from src.core.enums import ObstacleZone


class TestPhase2Detection(unittest.TestCase):
    """Test suite for Phase 2 real-time detection components."""

    def test_draw_detections(self):
        """Verify draw_detections annotates frame without modifying dimensions or throwing errors."""
        detector = YoloDetector()
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        bbox = BoundingBox(xmin=100, ymin=100, xmax=200, ymax=300)
        item = DetectionItem(label="person", confidence=0.91, bbox=bbox, zone=ObstacleZone.CENTER)
        result = DetectionResult(detections=[item], frame_width=640, frame_height=480)

        annotated = detector.draw_detections(dummy_frame, result, show_confidence=True, show_box=True)

        self.assertIsNotNone(annotated)
        self.assertEqual(annotated.shape, (480, 640, 3))
        # Ensure distance is 0.0 (no fake distance created)
        self.assertEqual(item.estimated_distance_m, 0.0)

    @patch("src.detection.yolo_detector.YoloDetector.load_model")
    def test_mock_yolo_detection_parsing(self, mock_load):
        """Verify detection result parsing when YOLO model produces mock boxes."""
        mock_load.return_value = True
        detector = YoloDetector()
        detector._is_loaded = True

        # Mock YOLO model call output
        mock_box = MagicMock()
        mock_box.xyxy = [MagicMock(tolist=lambda: [50.0, 60.0, 150.0, 200.0])]
        mock_box.conf = [MagicMock(item=lambda: 0.88)]
        mock_box.cls = [MagicMock(item=lambda: 0)]

        mock_result_item = MagicMock()
        mock_result_item.boxes = [mock_box]
        mock_result_item.names = {0: "person"}

        detector._model = MagicMock(return_value=[mock_result_item])

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(dummy_frame)

        self.assertEqual(result.count, 1)
        det = result.detections[0]
        self.assertEqual(det.label, "person")
        self.assertEqual(det.confidence, 0.88)
        self.assertEqual(det.estimated_distance_m, 0.0)  # No fake distance
        self.assertEqual(det.bbox.xmin, 50.0)

    def test_target_classes_resolved_to_model_ids(self):
        """Verify configured class names become YOLO class IDs, ignoring names the model lacks."""
        detector = YoloDetector(target_classes=["person", "chair", "stairs"])
        detector._model = MagicMock(names={0: "person", 27: "tie", 56: "chair"})

        with self.assertLogs("yolo_detector", level="WARNING") as logs:
            detector._resolve_target_class_ids()

        self.assertEqual(detector._target_class_ids, [0, 56])
        self.assertIn("stairs", logs.output[0])

    def test_target_classes_passed_to_model_call(self):
        """Verify detect() restricts YOLO inference to the resolved class IDs."""
        detector = YoloDetector(target_classes=["person"])
        detector._model = MagicMock(names={0: "person", 27: "tie"})
        detector._model.return_value = []
        detector._resolve_target_class_ids()
        detector._is_loaded = True

        detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        self.assertEqual(detector._model.call_args.kwargs["classes"], [0])

    def test_no_target_classes_detects_everything(self):
        """Verify an empty target list leaves YOLO unfiltered."""
        detector = YoloDetector(target_classes=[])
        detector._model = MagicMock(names={0: "person"})
        detector._resolve_target_class_ids()
        self.assertIsNone(detector._target_class_ids)


if __name__ == "__main__":
    unittest.main()
