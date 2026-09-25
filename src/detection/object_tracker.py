"""Lightweight IoU tracker that stabilizes YOLO detections across frames."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..core.models import BoundingBox, DetectionItem
from ..utils.helpers import determine_zone


def iou(a: BoundingBox, b: BoundingBox) -> float:
    """Intersection over union of two boxes."""
    ix = max(0.0, min(a.xmax, b.xmax) - max(a.xmin, b.xmin))
    iy = max(0.0, min(a.ymax, b.ymax) - max(a.ymin, b.ymin))
    inter = ix * iy
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    """One tracked object."""
    track_id: int
    label: str
    bbox: BoundingBox
    confidence: float
    hits: int = 1
    misses: int = 0
    confirmed: bool = False


class ObjectTracker:
    """Keeps object identities across frames and smooths YOLO flicker.

    - Detections match existing tracks of the same class by bounding-box overlap (IoU).
    - A new object is reported only after min_hits detections, filtering one-frame false positives.
    - A tracked object that YOLO misses is kept ("coasts") for up to max_missed frames.
    - Boxes are smoothed with an exponential moving average to reduce jitter.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_missed: int = 3,
        min_hits: int = 2,
        box_smoothing: float = 0.7,
        zone_boundaries: Tuple[float, float] = (0.33, 0.66),
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.min_hits = min_hits
        self.box_smoothing = box_smoothing  # Weight of the newest box (1.0 = no smoothing)
        self.zone_boundaries = zone_boundaries

        if config:
            track_cfg = config.get("tracking", {})
            self.iou_threshold = track_cfg.get("iou_threshold", iou_threshold)
            self.max_missed = track_cfg.get("max_missed_frames", max_missed)
            self.min_hits = track_cfg.get("min_hits", min_hits)
            self.box_smoothing = track_cfg.get("box_smoothing", box_smoothing)
            zones_cfg = config.get("risk_analysis", {}).get("zones", {})
            self.zone_boundaries = (
                zones_cfg.get("left_boundary", zone_boundaries[0]),
                zones_cfg.get("right_boundary", zone_boundaries[1]),
            )

        self.tracks: List[Track] = []
        self._next_id = 1

    def reset(self) -> None:
        """Forgets all tracks (e.g. when the video source restarts)."""
        self.tracks = []
        self._next_id = 1

    def update(self, detections: List[DetectionItem], frame_width: int) -> List[DetectionItem]:
        """Matches this frame's detections to tracks and returns the confirmed, stabilized objects."""
        candidates = sorted(
            (
                (iou(track.bbox, det.bbox), t_idx, d_idx)
                for t_idx, track in enumerate(self.tracks)
                for d_idx, det in enumerate(detections)
                if det.bbox is not None and det.label == track.label
            ),
            reverse=True,
        )

        matched_tracks, matched_dets = set(), set()
        for overlap, t_idx, d_idx in candidates:
            if overlap < self.iou_threshold:
                break
            if t_idx in matched_tracks or d_idx in matched_dets:
                continue
            matched_tracks.add(t_idx)
            matched_dets.add(d_idx)
            self._absorb(self.tracks[t_idx], detections[d_idx])

        for t_idx, track in enumerate(self.tracks):
            if t_idx not in matched_tracks:
                track.misses += 1
        self.tracks = [t for t in self.tracks if t.misses <= self.max_missed]

        for d_idx, det in enumerate(detections):
            if d_idx in matched_dets or det.bbox is None:
                continue
            track = Track(self._next_id, det.label, det.bbox, det.confidence)
            track.confirmed = track.hits >= self.min_hits
            self.tracks.append(track)
            self._next_id += 1

        return [self._to_detection(t, frame_width) for t in self.tracks if t.confirmed]

    def _absorb(self, track: Track, det: DetectionItem) -> None:
        s = self.box_smoothing
        old, new = track.bbox, det.bbox
        track.bbox = BoundingBox(
            xmin=s * new.xmin + (1 - s) * old.xmin,
            ymin=s * new.ymin + (1 - s) * old.ymin,
            xmax=s * new.xmax + (1 - s) * old.xmax,
            ymax=s * new.ymax + (1 - s) * old.ymax,
        )
        track.confidence = det.confidence
        track.hits += 1
        track.misses = 0
        track.confirmed = track.confirmed or track.hits >= self.min_hits

    def _to_detection(self, track: Track, frame_width: int) -> DetectionItem:
        center_norm = track.bbox.center_x / frame_width if frame_width > 0 else 0.5
        return DetectionItem(
            label=track.label,
            confidence=track.confidence,
            bbox=track.bbox,
            zone=determine_zone(center_norm, *self.zone_boundaries),
            track_id=track.track_id,
        )
