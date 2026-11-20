"""Local webcam computer vision sensor powered by MediaPipe Tasks and OpenCV."""

from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import math
import statistics
import threading
import time
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import core as mp_core
from mediapipe.tasks.python import vision as mp_vision
from PIL import Image, ImageDraw

from focus_guard.core.config import AppConfig, save_user_config
from focus_guard.core.event_bus import EventBus
from focus_guard.core.events import PhoneBox, VisionSignalEvent

logger = logging.getLogger(__name__)

_BLINK_WINDOW_SIZE = 20
_BLINK_CLOSE_THRESHOLD = 0.50
_POSE_HISTORY_SIZE = 30
_PHONE_MAX_RESULTS = 5
_PHONE_CATEGORIES = frozenset({"", "cell phone"})
_PERSON_SCORE_THRESHOLD = 0.40
_MAX_PHONE_BOX_AREA = 0.22
_CAMERA_RETRY_INTERVAL_SEC = 15.0
_MIN_FPS = 0.1
_FACE_LANDMARKER_FILENAME = "face_landmarker.task"
_PHONE_DETECTOR_FILENAME = "efficientdet_lite0.tflite"


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _clean_zero(value: float) -> float:
    return 0.0 if abs(value) < 1e-12 else float(value)


def _is_frame_obstructed(rgb_frame: np.ndarray) -> bool:
    """Return True when frame is covered/pitch-black (very low luminance and variance)."""
    mean_luma = float(rgb_frame.mean())
    if mean_luma >= 12.0:
        return False
    std_luma = float(rgb_frame.std())
    return std_luma < 8.0


def compute_perclos(blink_history: Sequence[bool]) -> float:
    """Compute PERCLOS (fraction of frames with closed eyes) in [0.0, 1.0]."""
    total = len(blink_history)
    if total == 0:
        return 0.0
    closed_frames = sum(1 for closed in blink_history if closed)
    return _clamp01(closed_frames / total)


def rotation_matrix_to_euler_deg(mat: np.ndarray) -> tuple[float, float, float]:
    """Extract (yaw_deg, pitch_deg, roll_deg) from a 3x3 or 4x4 transformation matrix."""
    arr = np.asarray(mat, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] < 3:
        raise ValueError(f"Expected 2D matrix with shape >= (3, 3), got {arr.shape}")

    rot = arr[:3, :3]
    if not np.isfinite(rot).all():
        return (0.0, 0.0, 0.0)

    sy = math.hypot(rot[0, 0], rot[1, 0])
    if sy > 1e-6:
        pitch = math.atan2(rot[2, 1], rot[2, 2])
        yaw = math.atan2(-rot[2, 0], sy)
        roll = math.atan2(rot[1, 0], rot[0, 0])
    else:
        pitch = math.atan2(-rot[1, 2], rot[1, 1])
        yaw = math.atan2(-rot[2, 0], sy)
        roll = 0.0

    return (
        _clean_zero(math.degrees(yaw)),
        _clean_zero(math.degrees(pitch)),
        _clean_zero(math.degrees(roll)),
    )


def _landmark_xy(lm: Any) -> tuple[float, float]:
    x = float(getattr(lm, "x", 0.0) or 0.0)
    y = float(getattr(lm, "y", 0.0) or 0.0)
    return x, y


def _estimate_pose_from_landmarks(landmarks: Sequence[Any]) -> tuple[float, float, float]:
    """Geometric fallback for (yaw_deg, pitch_deg, roll_deg) using FaceMesh landmarks."""
    if len(landmarks) < 468:
        return (0.0, 0.0, 0.0)

    nose_x, nose_y = _landmark_xy(landmarks[1])
    left_eye_x, left_eye_y = _landmark_xy(landmarks[33])
    right_eye_x, right_eye_y = _landmark_xy(landmarks[263])
    left_cheek_x, _ = _landmark_xy(landmarks[234])
    right_cheek_x, _ = _landmark_xy(landmarks[454])
    _, forehead_y = _landmark_xy(landmarks[10])
    _, chin_y = _landmark_xy(landmarks[152])

    half_width = (right_cheek_x - left_cheek_x) / 2.0
    cheek_mid_x = (left_cheek_x + right_cheek_x) / 2.0
    if abs(half_width) < 1e-6:
        yaw_deg = 0.0
    else:
        yaw_deg = max(-90.0, min(90.0, ((nose_x - cheek_mid_x) / half_width) * 45.0))

    eye_mid_y = (left_eye_y + right_eye_y) / 2.0
    face_height = chin_y - forehead_y
    if abs(face_height) < 1e-6:
        pitch_deg = 0.0
    else:
        rel_nose_drop = ((nose_y - eye_mid_y) / face_height) - 0.22
        pitch_deg = max(-90.0, min(90.0, rel_nose_drop * 120.0))

    dx = right_eye_x - left_eye_x
    dy = right_eye_y - left_eye_y
    roll_deg = 0.0 if abs(dx) < 1e-6 and abs(dy) < 1e-6 else math.degrees(math.atan2(dy, dx))

    return (_clean_zero(yaw_deg), _clean_zero(pitch_deg), _clean_zero(roll_deg))


def _extract_blendshapes(blendshapes: Sequence[Any]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for category in blendshapes:
        name = getattr(category, "category_name", None)
        score = getattr(category, "score", None)
        if name and score is not None:
            scores[str(name)] = float(score)
    return scores


class MediaPipeVisionSensor:
    """Privacy-first local webcam vision sensor implementing VisionSensorPort."""

    def __init__(
        self,
        config: AppConfig,
        event_bus: EventBus,
        *,
        retry_interval_sec: float = _CAMERA_RETRY_INTERVAL_SEC,
    ) -> None:
        self._config = config
        self._event_bus = event_bus
        self._retry_interval_sec = max(0.05, float(retry_interval_sec))

        self._face_model_path = config.models_dir / _FACE_LANDMARKER_FILENAME
        self._phone_model_path = config.models_dir / _PHONE_DETECTOR_FILENAME

        self._face_landmarker: Any | None = None
        self._object_detector: Any | None = None
        self._face_model_error: str | None = None
        self._capture: cv2.VideoCapture | None = None

        self._available: bool = False
        self._latest_signal: VisionSignalEvent | None = None
        self._latest_rgb_frame: np.ndarray | None = None
        self._blink_history: deque[bool] = deque(maxlen=_BLINK_WINDOW_SIZE)
        self._pose_history: deque[tuple[int, float, float, float]] = deque(
            maxlen=_POSE_HISTORY_SIZE
        )
        self._pose_seq: int = 0
        self._last_calibration_seq: int = 0

        self._last_phone_check_ts: float | None = None
        self._last_phone_detected: bool = False
        self._last_phone_confidence: float = 0.0
        self._last_phone_boxes: list[PhoneBox] = []
        self._last_person_detected: bool = False

        self._resource_lock = threading.Lock()
        self._frame_lock = threading.Lock()
        self._task: asyncio.Task[None] | None = None
        self._stop_event: asyncio.Event | None = None
        self._stop_requested: bool = False
        self._running: bool = False
        self._camera_warned: bool = False

        if self._config.vision.enabled:
            self._init_detectors()
        else:
            self._face_model_error = "vision disabled"

    @staticmethod
    def _build_base_options(model_path: Path) -> Any:
        # MediaPipe C++ loader can fail on non-ASCII Windows paths; buffer bypasses path encoding.
        if str(model_path).isascii():
            return mp_core.base_options.BaseOptions(model_asset_path=str(model_path))
        return mp_core.base_options.BaseOptions(model_asset_buffer=model_path.read_bytes())

    def _init_detectors(self) -> None:
        self._face_model_error = None

        if not self._face_model_path.is_file():
            self._face_model_error = f"missing face landmarker model: {self._face_model_path}"
            logger.warning("%s", self._face_model_error)
            self._face_landmarker = None
        else:
            try:
                face_opts = mp_vision.FaceLandmarkerOptions(
                    base_options=self._build_base_options(self._face_model_path),
                    running_mode=mp_vision.RunningMode.IMAGE,
                    num_faces=1,
                    output_face_blendshapes=True,
                    output_facial_transformation_matrixes=True,
                )
                self._face_landmarker = mp_vision.FaceLandmarker.create_from_options(face_opts)
            except Exception as exc:
                self._face_model_error = f"failed to initialize FaceLandmarker: {exc}"
                logger.warning("%s", self._face_model_error)
                self._face_landmarker = None

        want_person = bool(self._config.vision.enable_person_detector)
        want_phone = bool(self._config.vision.enable_phone_detector)
        if not want_person and not want_phone:
            self._object_detector = None
            return

        if not self._phone_model_path.is_file():
            logger.warning(
                "Object detector model missing at %s; person/phone detection disabled.",
                self._phone_model_path,
            )
            self._object_detector = None
            return

        try:
            allowlist: list[str] = []
            if want_person:
                allowlist.append("person")
            if want_phone:
                allowlist.append("cell phone")
            phone_threshold = _clamp01(float(self._config.vision.phone_score_threshold))
            det_threshold = (
                min(_PERSON_SCORE_THRESHOLD, phone_threshold)
                if (want_person and want_phone)
                else (_PERSON_SCORE_THRESHOLD if want_person else phone_threshold)
            )
            obj_opts = mp_vision.ObjectDetectorOptions(
                base_options=self._build_base_options(self._phone_model_path),
                running_mode=mp_vision.RunningMode.IMAGE,
                score_threshold=det_threshold,
                category_allowlist=allowlist,
                max_results=_PHONE_MAX_RESULTS,
            )
            self._object_detector = mp_vision.ObjectDetector.create_from_options(obj_opts)
        except Exception as exc:
            logger.warning(
                "Failed to initialize ObjectDetector (%s); person/phone detection disabled.",
                exc,
            )
            self._object_detector = None

    def _evaluate_phone_detector(
        self, mp_image: Any, now_ts: float
    ) -> tuple[bool, float, list[PhoneBox]]:
        if self._object_detector is None:
            self._last_person_detected = False
            return (False, 0.0, [])

        interval = max(0.0, float(self._config.vision.phone_detector_interval_sec))
        should_run = (
            self._last_phone_check_ts is None
            or (now_ts - self._last_phone_check_ts) >= interval
            or now_ts < self._last_phone_check_ts
        )
        if not should_run:
            return (
                self._last_phone_detected,
                self._last_phone_confidence,
                list(self._last_phone_boxes),
            )

        want_phone = bool(self._config.vision.enable_phone_detector)
        want_person = bool(self._config.vision.enable_person_detector)
        threshold = _clamp01(float(self._config.vision.phone_score_threshold))
        img_w = max(1, int(getattr(mp_image, "width", 1) or 1))
        img_h = max(1, int(getattr(mp_image, "height", 1) or 1))

        try:
            det_result = self._object_detector.detect(mp_image)
            best_score = 0.0
            found_match = False
            found_person = False
            boxes: list[PhoneBox] = []

            for det in getattr(det_result, "detections", None) or []:
                det_best_score = -1.0
                det_best_label = "cell phone"
                for cat in getattr(det, "categories", None) or []:
                    raw_name = (
                        getattr(cat, "category_name", None)
                        or getattr(cat, "display_name", None)
                        or ""
                    )
                    norm_name = str(raw_name).strip().lower()
                    score = getattr(cat, "score", None)
                    if score is None:
                        continue
                    fscore = float(score)
                    if want_person and norm_name == "person" and fscore >= _PERSON_SCORE_THRESHOLD:
                        found_person = True
                        continue
                    if not want_phone or norm_name not in _PHONE_CATEGORIES:
                        continue
                    if fscore >= threshold and fscore > det_best_score:
                        det_best_score = fscore
                        det_best_label = str(raw_name).strip() or "cell phone"

                if want_phone and det_best_score >= threshold:
                    bbox = getattr(det, "bounding_box", None)
                    if bbox is not None:
                        ox = float(getattr(bbox, "origin_x", 0.0) or 0.0)
                        oy = float(getattr(bbox, "origin_y", 0.0) or 0.0)
                        bw = float(getattr(bbox, "width", 0.0) or 0.0)
                        bh = float(getattr(bbox, "height", 0.0) or 0.0)
                        xmin = _clamp01(ox / img_w)
                        ymin = _clamp01(oy / img_h)
                        xmax = _clamp01((ox + max(0.0, bw)) / img_w)
                        ymax = _clamp01((oy + max(0.0, bh)) / img_h)
                        box_area = max(0.0, xmax - xmin) * max(0.0, ymax - ymin)
                        # Reject large static/background false positives (e.g. chair headrest)
                        if box_area > _MAX_PHONE_BOX_AREA:
                            continue
                        boxes.append(
                            PhoneBox(
                                xmin=xmin,
                                ymin=ymin,
                                xmax=xmax,
                                ymax=ymax,
                                score=_clamp01(det_best_score),
                                label=det_best_label,
                            )
                        )
                    found_match = True
                    best_score = max(best_score, det_best_score)

            self._last_person_detected = found_person
            if found_match:
                self._last_phone_detected = True
                self._last_phone_confidence = _clamp01(best_score)
                self._last_phone_boxes = boxes
            else:
                self._last_phone_detected = False
                self._last_phone_confidence = 0.0
                self._last_phone_boxes = []
        except Exception as exc:
            logger.warning("ObjectDetector inference failed: %s", exc)
        finally:
            self._last_phone_check_ts = now_ts

        return (
            self._last_phone_detected,
            self._last_phone_confidence,
            list(self._last_phone_boxes),
        )

    def _extract_head_pose(
        self,
        face_result: Any,
        landmarks: Sequence[Any],
    ) -> tuple[float, float, float]:
        matrices = getattr(face_result, "facial_transformation_matrixes", None) or []
        if len(matrices) > 0 and matrices[0] is not None:
            try:
                return rotation_matrix_to_euler_deg(matrices[0])
            except (ValueError, TypeError):
                pass
        return _estimate_pose_from_landmarks(landmarks)

    def _signal_from_face_result(
        self,
        face_result: Any,
        phone_detected: bool,
        phone_confidence: float,
        phone_boxes: Sequence[PhoneBox],
    ) -> VisionSignalEvent:
        vcfg = self._config.vision
        camera_position = str(vcfg.camera_position)
        calibrated = bool(vcfg.calibrated)

        face_landmarks_list = getattr(face_result, "face_landmarks", None) or []
        if not face_landmarks_list:
            if self._last_person_detected:
                # Side-profile (65-85 deg) or head tilted down on tablet near keyboard:
                # person is clearly present at the desk even when FaceLandmarker misses profile landmarks.
                return VisionSignalEvent(
                    camera_available=True,
                    face_present=True,
                    gaze_on_screen_score=0.85,
                    looking_at_screen=True,
                    looking_down_score=0.0,
                    phone_detected=phone_detected,
                    phone_confidence=phone_confidence,
                    camera_position=camera_position,
                    calibrated=calibrated,
                    phone_boxes=list(phone_boxes),
                )
            return VisionSignalEvent(
                camera_available=True,
                face_present=False,
                gaze_on_screen_score=0.0,
                looking_at_screen=False,
                phone_detected=phone_detected,
                phone_confidence=phone_confidence,
                camera_position=camera_position,
                calibrated=calibrated,
                phone_boxes=list(phone_boxes),
            )

        landmarks = face_landmarks_list[0]
        yaw_deg, pitch_deg, roll_deg = self._extract_head_pose(face_result, landmarks)

        blendshapes_list = getattr(face_result, "face_blendshapes", None) or []
        bs = _extract_blendshapes(blendshapes_list[0]) if blendshapes_list else {}

        blink_left = _clamp01(bs.get("eyeBlinkLeft", 0.0))
        blink_right = _clamp01(bs.get("eyeBlinkRight", 0.0))
        yawn_score = _clamp01(bs.get("jawOpen", 0.0))

        look_down_left = _clamp01(bs.get("eyeLookDownLeft", 0.0))
        look_down_right = _clamp01(bs.get("eyeLookDownRight", 0.0))
        eye_look_down = (look_down_left + look_down_right) / 2.0

        with self._frame_lock:
            self._pose_seq += 1
            self._pose_history.append((self._pose_seq, yaw_deg, pitch_deg, eye_look_down))

        neutral_yaw_deg = float(vcfg.neutral_yaw_deg)
        neutral_pitch_deg = float(vcfg.neutral_pitch_deg)
        neutral_eye_look_down = float(vcfg.neutral_eye_look_down)
        yaw_min = min(float(vcfg.screen_yaw_min_deg), float(vcfg.screen_yaw_max_deg))
        yaw_max = max(float(vcfg.screen_yaw_min_deg), float(vcfg.screen_yaw_max_deg))

        if yaw_min <= yaw_deg <= yaw_max:
            yaw_outside_deg = 0.0
        elif yaw_deg < yaw_min:
            yaw_outside_deg = yaw_min - yaw_deg
        else:
            yaw_outside_deg = yaw_deg - yaw_max

        calibrated_yaw_delta_deg = yaw_deg - neutral_yaw_deg
        calibrated_pitch_delta_deg = pitch_deg - neutral_pitch_deg

        eye_delta = _clamp01((eye_look_down - neutral_eye_look_down * 0.75) / 0.45)

        phone_pitch_delta = vcfg.phone_pitch_delta_deg
        if phone_pitch_delta is not None and abs(float(phone_pitch_delta)) >= 4.0:
            pitch_ratio = calibrated_pitch_delta_deg / float(phone_pitch_delta)
            pitch_down_component = _clamp01(pitch_ratio)
        elif calibrated:
            if eye_look_down >= neutral_eye_look_down * 0.85:
                pitch_down_component = _clamp01(
                    max(0.0, abs(calibrated_pitch_delta_deg) - 6.0) / 22.0
                )
            else:
                pitch_down_component = _clamp01(
                    max(0.0, abs(calibrated_pitch_delta_deg) - 10.0) / 25.0
                )
        else:
            pitch_down_component = _clamp01(max(0.0, calibrated_pitch_delta_deg) / 30.0)

        looking_down_score = _clamp01(0.45 * eye_delta + 0.55 * pitch_down_component)

        yaw_pen = yaw_outside_deg / max(10.0, float(vcfg.yaw_tolerance_deg))
        pitch_pen = max(0.0, abs(calibrated_pitch_delta_deg) - 12.0) / max(
            10.0, float(vcfg.pitch_tolerance_deg)
        )
        gaze_on_screen_score = _clamp01(1.0 - min(1.0, 0.65 * yaw_pen + 0.35 * pitch_pen))
        allow_down = bool(self._config.behavior.allow_looking_down_for_notes)
        looking_at_screen = (
            (yaw_outside_deg <= 6.0)
            and (allow_down or looking_down_score < 0.60)
            and (gaze_on_screen_score >= 0.45)
        )

        # Looking down to write/read notes naturally lowers upper eyelids and inflates MediaPipe
        # eyeBlink scores (especially on lateral webcams) — never treat note-taking posture as sleep.
        is_looking_down_for_notes = allow_down and (
            looking_down_score >= 0.30
            or eye_look_down >= max(0.35, neutral_eye_look_down + 0.12)
            or (calibrated and abs(calibrated_pitch_delta_deg) >= 8.0)
            or (not calibrated and pitch_deg >= 10.0)
        )
        eyes_closed = (
            not is_looking_down_for_notes
            and ((blink_left + blink_right) / 2.0) > _BLINK_CLOSE_THRESHOLD
            and min(blink_left, blink_right) > 0.35
        )
        self._blink_history.append(eyes_closed)
        drowsiness_perclos = compute_perclos(self._blink_history)

        return VisionSignalEvent(
            camera_available=True,
            face_present=True,
            gaze_on_screen_score=gaze_on_screen_score,
            head_yaw_deg=yaw_deg,
            head_pitch_deg=pitch_deg,
            head_roll_deg=roll_deg,
            drowsiness_perclos=drowsiness_perclos,
            yawn_score=yawn_score,
            looking_down_score=looking_down_score,
            phone_detected=phone_detected,
            phone_confidence=phone_confidence,
            camera_position=camera_position,
            calibrated=calibrated,
            looking_at_screen=looking_at_screen,
            calibrated_yaw_delta_deg=calibrated_yaw_delta_deg,
            calibrated_pitch_delta_deg=calibrated_pitch_delta_deg,
            phone_boxes=list(phone_boxes),
        )

    def _set_latest_signal(self, sig: VisionSignalEvent) -> VisionSignalEvent:
        with self._frame_lock:
            self._latest_signal = sig
        return sig

    def analyze_rgb_frame(
        self,
        rgb_frame: np.ndarray,
        now_ts: float | None = None,
    ) -> VisionSignalEvent:
        """Analyze an in-memory RGB numpy frame synchronously without touching disk."""
        vcfg = self._config.vision
        if (
            rgb_frame is None
            or not isinstance(rgb_frame, np.ndarray)
            or rgb_frame.ndim != 3
            or rgb_frame.shape[2] != 3
            or rgb_frame.size == 0
        ):
            return self._set_latest_signal(
                VisionSignalEvent(
                    camera_available=self._face_landmarker is not None,
                    face_present=False,
                    gaze_on_screen_score=0.0,
                    looking_at_screen=False,
                    camera_position=str(vcfg.camera_position),
                    calibrated=bool(vcfg.calibrated),
                    error="invalid rgb frame",
                )
            )

        contiguous_rgb = np.array(rgb_frame, dtype=np.uint8, copy=True, order="C")
        with self._frame_lock:
            self._latest_rgb_frame = contiguous_rgb

        if self._face_landmarker is None:
            return self._set_latest_signal(
                VisionSignalEvent(
                    camera_available=False,
                    face_present=False,
                    gaze_on_screen_score=0.0,
                    looking_at_screen=False,
                    camera_position=str(vcfg.camera_position),
                    calibrated=bool(vcfg.calibrated),
                    error=self._face_model_error or "face landmarker unavailable",
                )
            )

        ts = time.monotonic() if now_ts is None else float(now_ts)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=contiguous_rgb)

        phone_detected, phone_confidence, phone_boxes = self._evaluate_phone_detector(mp_image, ts)

        try:
            face_result = self._face_landmarker.detect(mp_image)
        except Exception as exc:
            logger.warning("FaceLandmarker inference failed: %s", exc)
            return self._set_latest_signal(
                VisionSignalEvent(
                    camera_available=True,
                    face_present=False,
                    gaze_on_screen_score=0.0,
                    looking_at_screen=False,
                    phone_detected=phone_detected,
                    phone_confidence=phone_confidence,
                    camera_position=str(vcfg.camera_position),
                    calibrated=bool(vcfg.calibrated),
                    phone_boxes=list(phone_boxes),
                    error=f"face inference error: {exc}",
                )
            )

        sig = self._signal_from_face_result(
            face_result, phone_detected, phone_confidence, phone_boxes
        )
        if not sig.face_present and _is_frame_obstructed(contiguous_rgb):
            self._last_person_detected = False
            sig = sig.model_copy(
                update={
                    "camera_obstructed": True,
                    "phone_detected": False,
                    "phone_confidence": 0.0,
                    "phone_boxes": [],
                }
            )
        return self._set_latest_signal(sig)

    def latest_frame_jpeg(
        self, max_width: int = 480, quality: int = 65, annotate: bool = False
    ) -> bytes | None:
        """Encode the latest RAM-only webcam frame to JPEG bytes with optional diagnostics."""
        with self._frame_lock:
            frame = self._latest_rgb_frame
            signal = self._latest_signal

        if frame is None:
            return None

        try:
            img = Image.fromarray(frame, mode="RGB")
            target_max_w = int(max_width)
            if target_max_w > 0 and img.width > target_max_w:
                ratio = target_max_w / float(img.width)
                target_h = max(1, int(round(img.height * ratio)))
                img = img.resize((target_max_w, target_h), Image.Resampling.LANCZOS)

            if annotate and signal is not None:
                self._draw_diagnostic_overlay(img, signal)

            buf = io.BytesIO()
            clamped_quality = max(1, min(95, int(quality)))
            img.save(buf, format="JPEG", quality=clamped_quality)
            return buf.getvalue()
        except Exception as exc:
            logger.warning("Failed to encode webcam frame to JPEG: %s", exc)
            return None

    @staticmethod
    def _draw_diagnostic_overlay(img: Image.Image, signal: VisionSignalEvent) -> None:
        draw = ImageDraw.Draw(img)
        header_h = min(26, img.height)

        for box in signal.phone_boxes:
            x1 = max(0, min(img.width - 1, int(round(box.xmin * img.width))))
            y1 = max(0, min(img.height - 1, int(round(box.ymin * img.height))))
            x2 = max(0, min(img.width - 1, int(round(box.xmax * img.width))))
            y2 = max(0, min(img.height - 1, int(round(box.ymax * img.height))))
            if x2 <= x1:
                x2 = min(img.width - 1, x1 + 1)
            if y2 <= y1:
                y2 = min(img.height - 1, y1 + 1)

            box_color = (239, 68, 68) if box.score >= 0.45 else (245, 158, 11)
            draw.rectangle([(x1, y1), (x2, y2)], outline=box_color, width=3)

            label_text = f"CELULAR {int(round(box.score * 100))}%"
            text_y = y1 + 4 if y1 < header_h + 16 else y1 - 14
            text_x = min(x1 + 2, max(0, img.width - 90))
            draw.rectangle(
                [
                    (max(0, text_x - 2), max(0, text_y - 2)),
                    (min(img.width - 1, text_x + 86), min(img.height - 1, text_y + 12)),
                ],
                fill=box_color,
            )
            draw.text((text_x, text_y), label_text, fill=(255, 255, 255))

        draw.rectangle([(0, 0), (img.width, header_h)], fill=(15, 23, 42))
        if signal.camera_obstructed:
            status_text = "CAMERA ESCURA / TAMPADA"
            status_color = (239, 68, 68)
        elif not signal.face_present:
            status_text = "NAO ESTOU TE VENDO!"
            status_color = (239, 68, 68)
        elif signal.looking_at_screen and abs(signal.head_yaw_deg) < 0.01 and abs(signal.head_pitch_deg) < 0.01:
            status_text = "PRESENTE (PERFIL / TABLET) OK"
            status_color = (34, 197, 94)
        elif signal.looking_at_screen:
            status_text = "OLHANDO P/ TELA OK"
            status_color = (34, 197, 94)
        else:
            status_text = "FORA DA ZONA DA TELA"
            status_color = (239, 68, 68) if signal.phone_detected else (245, 158, 11)

        metrics_text = (
            f"Yaw:{signal.head_yaw_deg:+.0f}° Pitch:{signal.head_pitch_deg:+.0f}° "
            f"PERCLOS:{signal.drowsiness_perclos * 100:.0f}%"
        )
        draw.text((8, 7), status_text, fill=status_color)
        status_bbox = draw.textbbox((8, 7), status_text)
        metrics_x = min(img.width - 10, status_bbox[2] + 12)
        if metrics_x < img.width - 40:
            draw.text((metrics_x, 7), metrics_text, fill=(226, 232, 240))

    def _persist_user_config(self) -> None:
        try:
            save_user_config(self._config)
        except Exception as exc:
            logger.warning("Failed to persist vision calibration: %s", exc)

    def _calibration_summary(
        self,
        mode: str,
        *,
        ok: bool = True,
        samples: int = 0,
        error: str | None = None,
    ) -> dict[str, Any]:
        v = self._config.vision
        summary: dict[str, Any] = {
            "ok": ok,
            "mode": mode,
            "samples": samples,
            "camera_position": v.camera_position,
            "calibrated": v.calibrated,
            "neutral_yaw_deg": v.neutral_yaw_deg,
            "screen_yaw_min_deg": v.screen_yaw_min_deg,
            "screen_yaw_max_deg": v.screen_yaw_max_deg,
            "neutral_pitch_deg": v.neutral_pitch_deg,
            "neutral_eye_look_down": v.neutral_eye_look_down,
            "phone_pitch_delta_deg": v.phone_pitch_delta_deg,
            "yaw_tolerance_deg": v.yaw_tolerance_deg,
            "pitch_tolerance_deg": v.pitch_tolerance_deg,
        }
        if error is not None:
            summary["error"] = error
        return summary

    async def calibrate_pose(
        self, mode: str = "primary_screen", duration_sec: float = 2.0
    ) -> dict[str, Any]:
        """Calibrate lateral screen zone and phone-down pitch from live/recent pose samples."""
        if mode == "reset":
            v = self._config.vision
            v.calibrated = False
            v.neutral_yaw_deg = 0.0
            v.screen_yaw_min_deg = -55.0
            v.screen_yaw_max_deg = 25.0
            v.neutral_pitch_deg = 0.0
            v.neutral_eye_look_down = 0.22
            v.phone_pitch_delta_deg = None
            with self._frame_lock:
                self._last_calibration_seq = self._pose_seq
            self._persist_user_config()
            return self._calibration_summary(mode, ok=True, samples=0)

        if mode not in ("primary_screen", "secondary_screen", "phone_pose"):
            return self._calibration_summary(
                mode,
                ok=False,
                samples=0,
                error=f"unsupported calibration mode: {mode}",
            )

        with self._frame_lock:
            start_seq = self._pose_seq

        if self._running and duration_sec > 0.0:
            deadline = time.monotonic() + float(duration_sec)
            while time.monotonic() < deadline:
                with self._frame_lock:
                    fresh_count = sum(1 for seq, _, _, _ in self._pose_history if seq > start_seq)
                if fresh_count >= 3:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                await asyncio.sleep(min(0.2, remaining))

        with self._frame_lock:
            fresh = [(y, p, e) for seq, y, p, e in self._pose_history if seq > start_seq]
            since_last = [
                (y, p, e) for seq, y, p, e in self._pose_history if seq > self._last_calibration_seq
            ]
            recent = [(y, p, e) for _, y, p, e in self._pose_history]
            latest_sig = self._latest_signal
            end_seq = self._pose_seq

        samples = fresh or since_last or recent
        if not samples and latest_sig is not None and latest_sig.face_present:
            samples = [
                (
                    float(latest_sig.head_yaw_deg),
                    float(latest_sig.head_pitch_deg),
                    float(self._config.vision.neutral_eye_look_down),
                )
            ]

        if not samples:
            return self._calibration_summary(
                mode,
                ok=False,
                samples=0,
                error="no face detected for calibration",
            )

        med_yaw = float(statistics.median(s[0] for s in samples))
        med_pitch = float(statistics.median(s[1] for s in samples))
        med_eye = float(statistics.median(s[2] for s in samples))

        v = self._config.vision
        if mode == "primary_screen":
            v.calibrated = True
            v.neutral_yaw_deg = round(med_yaw, 1)
            v.neutral_pitch_deg = round(med_pitch, 1)
            v.neutral_eye_look_down = round(med_eye, 3)
            v.screen_yaw_min_deg = round(min(med_yaw - 20.0, -15.0), 1)
            v.screen_yaw_max_deg = round(max(med_yaw + 20.0, 15.0), 1)
        elif mode == "secondary_screen":
            v.calibrated = True
            v.screen_yaw_min_deg = round(min(v.screen_yaw_min_deg, med_yaw - 18.0), 1)
            v.screen_yaw_max_deg = round(max(v.screen_yaw_max_deg, med_yaw + 18.0), 1)
        elif mode == "phone_pose":
            v.phone_pitch_delta_deg = round(med_pitch - v.neutral_pitch_deg, 1)

        with self._frame_lock:
            self._last_calibration_seq = end_seq

        self._persist_user_config()
        return self._calibration_summary(mode, ok=True, samples=len(samples))

    def _capture_and_analyze_sync(self) -> tuple[bool, VisionSignalEvent | str] | None:
        with self._resource_lock:
            if self._stop_requested:
                return None

            if self._capture is None or not self._capture.isOpened():
                if self._capture is not None:
                    with contextlib.suppress(Exception):
                        self._capture.release()
                    self._capture = None

                try:
                    cap = cv2.VideoCapture(self._config.vision.camera_index)
                except Exception as exc:
                    return (False, f"camera open error: {exc}")

                if cap is None or not cap.isOpened():
                    if cap is not None:
                        with contextlib.suppress(Exception):
                            cap.release()
                    return (
                        False,
                        f"camera index {self._config.vision.camera_index} failed to open",
                    )

                with contextlib.suppress(Exception):
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                self._capture = cap

            try:
                ok, bgr_frame = self._capture.read()
            except Exception as exc:
                with contextlib.suppress(Exception):
                    self._capture.release()
                self._capture = None
                return (False, f"camera read error: {exc}")

            if not ok or bgr_frame is None or getattr(bgr_frame, "size", 0) == 0:
                with contextlib.suppress(Exception):
                    self._capture.release()
                self._capture = None
                return (False, "camera frame read failed")

            try:
                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            except Exception as exc:
                return (False, f"frame conversion error: {exc}")

            signal = self.analyze_rgb_frame(rgb_frame)
            return (True, signal)

    def _release_resources(self) -> None:
        with self._resource_lock:
            if self._capture is not None:
                with contextlib.suppress(Exception):
                    self._capture.release()
                self._capture = None

            if self._face_landmarker is not None:
                with contextlib.suppress(Exception):
                    self._face_landmarker.close()
                self._face_landmarker = None

            if self._object_detector is not None:
                with contextlib.suppress(Exception):
                    self._object_detector.close()
                self._object_detector = None

    async def _run_loop(self) -> None:
        frame_interval = 1.0 / max(_MIN_FPS, float(self._config.vision.target_fps))
        while not self._stop_requested:
            try:
                t0 = time.monotonic()
                result = await asyncio.to_thread(self._capture_and_analyze_sync)
                if result is None or self._stop_requested:
                    break

                ok, payload = result
                if ok and isinstance(payload, VisionSignalEvent):
                    self._available = True
                    self._camera_warned = False
                    self._set_latest_signal(payload)
                    try:
                        await asyncio.wait_for(self._event_bus.publish(payload), timeout=1.5)
                    except TimeoutError:
                        logger.debug("VisionSignalEvent publish timed out; keeping capture loop alive.")
                    elapsed = time.monotonic() - t0
                    delay = max(0.01, frame_interval - elapsed)
                else:
                    error_msg = str(payload)
                    self._available = False
                    if not self._camera_warned:
                        logger.warning(
                            "Webcam unavailable (camera_index=%s): %s",
                            self._config.vision.camera_index,
                            error_msg,
                        )
                        self._camera_warned = True
                    fail_event = VisionSignalEvent(
                        camera_available=False,
                        face_present=False,
                        gaze_on_screen_score=0.0,
                        looking_at_screen=False,
                        camera_position=str(self._config.vision.camera_position),
                        calibrated=bool(self._config.vision.calibrated),
                        error=error_msg,
                    )
                    self._set_latest_signal(fail_event)
                    try:
                        await asyncio.wait_for(self._event_bus.publish(fail_event), timeout=1.5)
                    except TimeoutError:
                        logger.debug("Vision failure event publish timed out; keeping capture loop alive.")
                    delay = self._retry_interval_sec
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Unexpected vision loop error: %s", exc)
                self._available = False
                delay = self._retry_interval_sec

            if self._stop_requested or self._stop_event is None:
                break
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=delay)
                break
            except TimeoutError:
                continue

    async def start(self) -> None:
        """Start the background webcam analysis loop."""
        if self._running:
            return

        self._stop_requested = False
        self._stop_event = asyncio.Event()
        self._camera_warned = False

        if not self._config.vision.enabled:
            self._available = False
            event = self._set_latest_signal(
                VisionSignalEvent(
                    camera_available=False,
                    face_present=False,
                    gaze_on_screen_score=0.0,
                    looking_at_screen=False,
                    camera_position=str(self._config.vision.camera_position),
                    calibrated=bool(self._config.vision.calibrated),
                    error="vision disabled",
                )
            )
            await self._event_bus.publish(event)
            return

        if self._face_landmarker is None:
            self._init_detectors()

        if self._face_landmarker is None:
            self._available = False
            event = self._set_latest_signal(
                VisionSignalEvent(
                    camera_available=False,
                    face_present=False,
                    gaze_on_screen_score=0.0,
                    looking_at_screen=False,
                    camera_position=str(self._config.vision.camera_position),
                    calibrated=bool(self._config.vision.calibrated),
                    error=self._face_model_error or "face landmarker unavailable",
                )
            )
            await self._event_bus.publish(event)
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop(), name="focus_guard_vision_loop")

    async def stop(self) -> None:
        """Stop the background loop and release camera and MediaPipe resources."""
        self._running = False
        self._available = False
        self._stop_requested = True
        if self._stop_event is not None:
            self._stop_event.set()

        if self._task is not None:
            if not self._task.done():
                self._task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._task
            self._task = None

        with self._frame_lock:
            self._latest_rgb_frame = None
            self._pose_history.clear()

        await asyncio.to_thread(self._release_resources)

    def latest_signal(self) -> VisionSignalEvent | None:
        """Return the most recent VisionSignalEvent produced by the sensor."""
        with self._frame_lock:
            return self._latest_signal

    def is_available(self) -> bool:
        """Return whether the webcam vision pipeline is currently available."""
        return self._available
