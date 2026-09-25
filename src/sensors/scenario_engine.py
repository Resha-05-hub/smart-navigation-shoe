"""Scenario engine that drives the simulated LEFT, CENTER, and RIGHT distance sensors over time.

Disclaimer:
This module generates SIMULATED range data for the software-only demonstration.
It replaces physical HC-SR04 readings with distances controlled by the presenter
(keyboard) or by scripted, time-based scenarios defined in config.yaml.
"""

import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from .sensor_manager import SensorManager
from .vision_distance_estimator import VisionDistanceEstimator
from ..core.enums import ObstacleZone
from ..core.models import DetectionItem
from ..utils.logger import get_logger

logger = get_logger("scenario_engine")

ZONES: Tuple[ObstacleZone, ObstacleZone, ObstacleZone] = (
    ObstacleZone.LEFT,
    ObstacleZone.CENTER,
    ObstacleZone.RIGHT,
)

# A keyframe distance of None simulates a faulty sensor (reported as INVALID)
ZoneDistances = Dict[ObstacleZone, Optional[float]]

CONTROLS_HELP = (
    "Keys: a/z LEFT  s/x CENTER  d/c RIGHT (closer/farther) | "
    "v camera-linked | 0 clear | r reset | n next scenario | p pause | q quit"
)


@dataclass
class Keyframe:
    """Sensor distances (cm) at a point in time (seconds from scenario start)."""
    time_s: float
    left_cm: Optional[float]
    center_cm: Optional[float]
    right_cm: Optional[float]

    def distance(self, zone: ObstacleZone) -> Optional[float]:
        """Returns this keyframe's distance for a zone."""
        return {
            ObstacleZone.LEFT: self.left_cm,
            ObstacleZone.CENTER: self.center_cm,
            ObstacleZone.RIGHT: self.right_cm,
        }[zone]


@dataclass
class Scenario:
    """Named timeline of keyframes, linearly interpolated between frames."""
    name: str
    keyframes: List[Keyframe]
    description: str = ""

    @property
    def duration_s(self) -> float:
        """Time of the last keyframe."""
        return self.keyframes[-1].time_s

    def distances_at(self, t: float) -> ZoneDistances:
        """Interpolates each zone's distance at time t (clamped to the timeline).

        A faulty (None) value on either side of a segment holds the earlier keyframe's value,
        so a fault starts exactly at its keyframe and lasts until the next one.
        """
        frames = self.keyframes
        if t <= frames[0].time_s:
            return {zone: frames[0].distance(zone) for zone in ZONES}
        if t >= frames[-1].time_s:
            return {zone: frames[-1].distance(zone) for zone in ZONES}

        for start, end in zip(frames, frames[1:]):
            if start.time_s <= t < end.time_s:
                span = end.time_s - start.time_s
                ratio = (t - start.time_s) / span if span > 0 else 0.0
                result: ZoneDistances = {}
                for zone in ZONES:
                    a, b = start.distance(zone), end.distance(zone)
                    result[zone] = a if a is None or b is None else a + (b - a) * ratio
                return result

        return {zone: frames[-1].distance(zone) for zone in ZONES}

    @classmethod
    def from_config(cls, name: str, cfg: Dict[str, Any]) -> "Scenario":
        """Builds a scenario from a config entry with 'keyframes: [[t, left, center, right], ...]'."""
        raw_frames = cfg.get("keyframes") or []
        keyframes: List[Keyframe] = []
        for raw in raw_frames:
            if len(raw) != 4:
                raise ValueError(f"Scenario '{name}': keyframe {raw} must be [time_s, left_cm, center_cm, right_cm]")
            t, left, center, right = raw
            keyframes.append(Keyframe(float(t), _opt_float(left), _opt_float(center), _opt_float(right)))

        if not keyframes:
            raise ValueError(f"Scenario '{name}' has no keyframes")
        keyframes.sort(key=lambda k: k.time_s)
        return cls(name=name, keyframes=keyframes, description=cfg.get("description", ""))


def _opt_float(value: Any) -> Optional[float]:
    return None if value is None else float(value)


class SensorScenarioEngine:
    """Moves simulated sensor distances by keyboard (manual), scripted scenarios, or the camera view.

    Modes:
    - manual: the presenter moves each zone's obstacle with the keyboard.
    - scenario name: a scripted keyframe timeline plays.
    - camera: each zone reads the nearest YOLO object's estimated distance (camera-linked).

    Call update() once per frame before reading the SensorManager (passing detections and frame
    size for camera mode), and pass OpenCV key codes to handle_key() for live control.
    """

    MANUAL = "manual"
    CAMERA = "camera"

    def __init__(
        self,
        sensor_manager: SensorManager,
        scenarios: Optional[Dict[str, Scenario]] = None,
        initial_cm: Optional[Tuple[float, float, float]] = None,
        step_cm: float = 10.0,
        noise_cm: float = 0.0,
        loop: bool = True,
        seed: Optional[int] = None,
        estimator: Optional[VisionDistanceEstimator] = None,
        smoothing: float = 0.5,
        hold_s: float = 0.5,
        config: Optional[Dict[str, Any]] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.sensor_manager = sensor_manager
        self.scenarios: Dict[str, Scenario] = dict(scenarios or {})
        self.step_cm = step_cm
        self.noise_cm = noise_cm
        self.loop = loop
        self.estimator = estimator or VisionDistanceEstimator(config=config)
        self.smoothing = smoothing  # Weight of the newest camera estimate (1.0 = no smoothing)
        self.hold_s = hold_s        # Keep a camera distance this long after its detection flickers out
        self._clock = clock
        self._rng = random.Random(seed)
        startup = self.MANUAL

        if initial_cm is None:
            initial_cm = tuple(
                round(sensor_manager.get_sensor(zone).read_distance().distance_cm, 1)  # type: ignore[union-attr]
                if sensor_manager.get_sensor(zone) else 300.0
                for zone in ZONES
            )

        if config:
            sim_cfg = config.get("sensors", {}).get("simulation", {})
            self.step_cm = sim_cfg.get("step_cm", step_cm)
            self.noise_cm = sim_cfg.get("noise_cm", noise_cm)
            self.loop = sim_cfg.get("loop", loop)
            startup = sim_cfg.get("startup", self.MANUAL) or self.MANUAL
            for name, scenario_cfg in (sim_cfg.get("scenarios") or {}).items():
                self.scenarios[name] = Scenario.from_config(name, scenario_cfg)
            cam_cfg = sim_cfg.get("camera_linked", {})
            self.smoothing = cam_cfg.get("smoothing", smoothing)
            self.hold_s = cam_cfg.get("hold_s", hold_s)

        self.min_cm = sensor_manager.min_distance_m * 100.0
        self.max_cm = sensor_manager.max_distance_m * 100.0
        self.enabled = sensor_manager.sensor_type == "simulated"
        if not self.enabled:
            logger.warning("Scenario engine disabled: sensors are not simulated.")

        self._initial: ZoneDistances = dict(zip(ZONES, initial_cm))
        self._manual: ZoneDistances = dict(self._initial)
        self._current: ZoneDistances = dict(self._initial)
        self.mode: str = self.MANUAL
        self._active: Optional[Scenario] = None
        self._active_loop: bool = self.loop
        self._start_time: float = 0.0
        self._paused_at: Optional[float] = None
        self.finished: bool = False
        self._camera: ZoneDistances = {zone: self.max_cm for zone in ZONES}
        self._camera_last_seen: Dict[ObstacleZone, Optional[float]] = {zone: None for zone in ZONES}

        self.startup_mode = startup
        if startup not in (self.MANUAL, self.CAMERA) and startup not in self.scenarios:
            logger.warning(f"Unknown startup mode '{startup}'; starting in manual mode.")
            self.startup_mode = self.MANUAL
        self._enter_startup_mode()

    def _enter_startup_mode(self) -> None:
        if self.startup_mode == self.CAMERA:
            self.set_camera_linked()
        elif self.startup_mode in self.scenarios:
            self.start_scenario(self.startup_mode)

    # ------------------------------------------------------------------ control

    @property
    def scenario_names(self) -> List[str]:
        """Names of all available scripted scenarios, in definition order."""
        return list(self.scenarios.keys())

    @property
    def is_paused(self) -> bool:
        """True while a scripted scenario is paused."""
        return self._paused_at is not None

    def start_scenario(self, name: str, loop: Optional[bool] = None) -> None:
        """Starts a scripted scenario from its beginning."""
        if name not in self.scenarios:
            raise KeyError(f"Unknown scenario '{name}'. Available: {self.scenario_names}")
        self._active = self.scenarios[name]
        self._active_loop = self.loop if loop is None else loop
        self.mode = name
        self._start_time = self._clock()
        self._paused_at = None
        self.finished = False
        logger.info(f"Scenario started: {name} - {self._active.description}")

    def next_scenario(self) -> Optional[str]:
        """Starts the scenario after the current one (or the first), returning its name."""
        names = self.scenario_names
        if not names:
            return None
        index = names.index(self.mode) + 1 if self.mode in names else 0
        name = names[index % len(names)]
        self.start_scenario(name)
        return name

    def set_manual(self) -> None:
        """Switches to manual control, keeping the current distances so nothing jumps."""
        if self.mode != self.MANUAL:
            self._manual = dict(self._current)
        self.mode = self.MANUAL
        self._active = None
        self._paused_at = None
        self.finished = False

    def set_distances_cm(self, left_cm: Optional[float], center_cm: Optional[float], right_cm: Optional[float]) -> None:
        """Sets all three distances in manual mode (None simulates a sensor fault)."""
        self.set_manual()
        for zone, value in zip(ZONES, (left_cm, center_cm, right_cm)):
            self._manual[zone] = None if value is None else self._clamp(value)

    def adjust(self, zone: ObstacleZone, delta_cm: float) -> None:
        """Moves one zone's obstacle closer (negative delta) or farther (positive delta)."""
        self.set_manual()
        base = self._manual.get(zone)
        self._manual[zone] = self._clamp((self.max_cm if base is None else base) + delta_cm)

    def set_camera_linked(self) -> None:
        """Drives each zone's sensor from the nearest detected object's estimated distance."""
        self.mode = self.CAMERA
        self._active = None
        self._paused_at = None
        self.finished = False
        self._camera = {zone: self.max_cm for zone in ZONES}
        self._camera_last_seen = {zone: None for zone in ZONES}
        logger.info("Camera-linked simulated sensors enabled (distances estimated from the camera view).")

    def clear_all(self) -> None:
        """Clears the path: every sensor reads its maximum range."""
        self.set_distances_cm(self.max_cm, self.max_cm, self.max_cm)

    def reset(self) -> None:
        """Returns to the configured startup mode (manual resets to the startup distances)."""
        self.set_manual()
        self._manual = dict(self._initial)
        self._enter_startup_mode()

    def toggle_pause(self) -> None:
        """Pauses or resumes the running scripted scenario."""
        if self._active is None:
            return
        now = self._clock()
        if self._paused_at is None:
            self._paused_at = now
        else:
            self._start_time += now - self._paused_at
            self._paused_at = None

    def handle_key(self, key: int) -> bool:
        """Applies an OpenCV key code (cv2.waitKey() & 0xFF). Returns True if the key was used."""
        if key in (255, -1):
            return False

        char = chr(key).lower() if 0 <= key < 256 else ""
        zone_keys = {
            "a": (ObstacleZone.LEFT, -1), "z": (ObstacleZone.LEFT, 1),
            "s": (ObstacleZone.CENTER, -1), "x": (ObstacleZone.CENTER, 1),
            "d": (ObstacleZone.RIGHT, -1), "c": (ObstacleZone.RIGHT, 1),
        }
        if char in zone_keys:
            zone, direction = zone_keys[char]
            self.adjust(zone, direction * self.step_cm)
        elif char == "0":
            self.clear_all()
        elif char == "r":
            self.reset()
        elif char == "n":
            self.next_scenario()
        elif char == "p":
            self.toggle_pause()
        elif char == "v":
            if self.mode == self.CAMERA:
                self.set_manual()
            else:
                self.set_camera_linked()
        else:
            return False
        return True

    # ------------------------------------------------------------------ update

    def elapsed_s(self) -> float:
        """Seconds into the running scenario (0 in manual mode)."""
        if self._active is None:
            return 0.0
        now = self._paused_at if self._paused_at is not None else self._clock()
        return max(0.0, now - self._start_time)

    def target_distances(self) -> ZoneDistances:
        """Noise-free distances the simulation is currently aiming for."""
        if self.mode == self.CAMERA:
            return dict(self._camera)
        if self._active is None:
            return dict(self._manual)

        elapsed = self.elapsed_s()
        duration = self._active.duration_s
        if self._active_loop and duration > 0:
            elapsed %= duration
        elif elapsed >= duration:
            self.finished = True
        return self._active.distances_at(elapsed)

    def _update_camera_distances(
        self,
        detections: List[DetectionItem],
        frame_size: Tuple[int, int],
    ) -> None:
        """Smooths per-zone camera estimates; a zone clears only after hold_s without detections."""
        now = self._clock()
        estimates = self.estimator.zone_distances_cm(detections, frame_size[0], frame_size[1])

        for zone in ZONES:
            estimate = estimates[zone]
            last_seen = self._camera_last_seen[zone]
            tracking = last_seen is not None and now - last_seen <= self.hold_s

            if estimate is not None:
                estimate = self._clamp(estimate)
                previous = self._camera[zone]
                if tracking and previous is not None:
                    estimate = self.smoothing * estimate + (1.0 - self.smoothing) * previous
                self._camera[zone] = estimate  # A newly appearing object is reported without lag
                self._camera_last_seen[zone] = now
            elif not tracking:
                self._camera[zone] = self.max_cm
                self._camera_last_seen[zone] = None

    def update(
        self,
        detections: Optional[List[DetectionItem]] = None,
        frame_size: Optional[Tuple[int, int]] = None,
    ) -> ZoneDistances:
        """Pushes the current distances (plus sensor noise) into the simulated sensors.

        Args:
            detections: YOLO detections for this frame (used in camera-linked mode).
            frame_size: (width, height) of the analyzed frame (required for camera-linked mode).
        """
        if self.mode == self.CAMERA and frame_size is not None:
            self._update_camera_distances(detections or [], frame_size)

        targets = self.target_distances()
        self._current = targets
        if not self.enabled:
            return targets

        applied: ZoneDistances = {}
        for zone in ZONES:
            sensor: Any = self.sensor_manager.get_sensor(zone)
            value = targets[zone]
            if value is None:
                sensor.set_simulated_distance_cm(-1.0)  # Reported as INVALID
                applied[zone] = None
                continue
            if self.noise_cm > 0:
                value = self._clamp(value + self._rng.gauss(0.0, self.noise_cm))
            sensor.set_simulated_distance_cm(value)
            applied[zone] = value
        return applied

    @property
    def status_text(self) -> str:
        """Short mode description for dashboards, e.g. 'SCENARIO approaching_obstacle 3.2s'."""
        if self.mode == self.CAMERA:
            return "CAMERA-LINKED (estimated)"
        if self._active is None:
            return "MANUAL"

        elapsed = self.elapsed_s()
        duration = self._active.duration_s
        if self._active_loop and duration > 0:
            elapsed %= duration
        else:
            elapsed = min(elapsed, duration)

        state = " PAUSED" if self.is_paused else (" DONE" if self.finished else "")
        return f"SCENARIO {self.mode} {elapsed:.1f}s{state}"

    @property
    def distances_text(self) -> str:
        """Noise-free target distances, e.g. 'L 300cm | C 45cm | R FAULT'."""
        parts = []
        for zone, value in self.target_distances().items():
            parts.append(f"{zone.value[0]} {'FAULT' if value is None else f'{int(round(value))}cm'}")
        return " | ".join(parts)

    def _clamp(self, value_cm: float) -> float:
        return max(self.min_cm, min(self.max_cm, value_cm))
