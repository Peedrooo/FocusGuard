"""Unit tests for the FocusGuard Webcam Computer Vision subsystem."""

from __future__ import annotations

import asyncio
import io
import logging
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

import focus_guard.sensors.vision.vision_sensor as vision_sensor_mod
from focus_guard.core.config import AppConfig, VisionConfig, save_user_config
from focus_guard.core.event_bus import EventBus
from focus_guard.core.events import PhoneBox, VisionSignalEvent
from focus_guard.core.ports import VisionSensorPort
from focus_guard.sensors.vision import (
    MediaPipeVisionSensor,
    compute_perclos,
    rotation_matrix_to_euler_deg,
)


def _rot_x(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, math.cos(a), -math.sin(a)], [0.0, math.sin(a), math.cos(a)]]
    )


def _rot_y(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array(
        [[math.cos(a), 0.0, math.sin(a)], [0.0, 1.0, 0.0], [-math.sin(a), 0.0, math.cos(a)]]
    )


def _rot_z(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array(
        [[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]]
    )


@dataclass
class _FakeCategory:
    category_name: str | None = None
    score: float | None = None
    display_name: str | None = None


@dataclass
class _FakeBoundingBox:
    origin_x: int = 0
    origin_y: int = 0
    width: int = 0
    height: int = 0


@dataclass
class _FakeLandmark:
    x: float
    y: float
    z: float = 0.0


@dataclass
class _FakeFaceResult:
    face_landmarks: list[list[Any]]
    face_blendshapes: list[list[_FakeCategory]]
    facial_transformation_matrixes: list[np.ndarray]


@dataclass
class _FakeDetection:
    categories: list[_FakeCategory]
    bounding_box: _FakeBoundingBox | None = None


@dataclass
class _FakeDetectionResult:
    detections: list[_FakeDetection]


def _make_neutral_landmarks() -> list[_FakeLandmark]:
    landmarks = [_FakeLandmark(0.5, 0.5, 0.0) for _ in range(468)]
    landmarks[1] = _FakeLandmark(0.50, 0.50)    # nose tip
    landmarks[33] = _FakeLandmark(0.42, 0.40)   # left eye outer
    landmarks[263] = _FakeLandmark(0.58, 0.40)  # right eye outer
    landmarks[234] = _FakeLandmark(0.30, 0.48)  # left cheek
    landmarks[454] = _FakeLandmark(0.70, 0.48)  # right cheek
    landmarks[10] = _FakeLandmark(0.50, 0.20)   # forehead
    landmarks[152] = _FakeLandmark(0.50, 0.80)  # chin
    return landmarks


@pytest.mark.asyncio
async def test_sensor_implements_vision_sensor_port() -> None:
    cfg = AppConfig(vision=VisionConfig(enabled=False))
    sensor = MediaPipeVisionSensor(cfg, EventBus())
    assert isinstance(sensor, VisionSensorPort)
    assert sensor.is_available() is False
    assert sensor.latest_signal() is None
    assert sensor.latest_frame_jpeg() is None


def test_compute_perclos_math() -> None:
    assert compute_perclos([]) == 0.0
    assert compute_perclos([False, False, False, False]) == pytest.approx(0.0)
    assert compute_perclos([True, False, True, False]) == pytest.approx(0.5)
    assert compute_perclos([True, True, True]) == pytest.approx(1.0)

    window: deque[bool] = deque([False] * 20, maxlen=20)
    for _ in range(5):
        window.append(True)
    assert compute_perclos(window) == pytest.approx(0.25)


def test_rotation_matrix_to_euler_deg() -> None:
    assert rotation_matrix_to_euler_deg(np.eye(3)) == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)
    assert rotation_matrix_to_euler_deg(np.eye(4)) == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)

    yaw, pitch, roll = rotation_matrix_to_euler_deg(_rot_y(30.0))
    assert (yaw, pitch, roll) == pytest.approx((30.0, 0.0, 0.0), abs=1e-4)

    yaw, pitch, roll = rotation_matrix_to_euler_deg(_rot_x(20.0))
    assert (yaw, pitch, roll) == pytest.approx((0.0, 20.0, 0.0), abs=1e-4)

    yaw, pitch, roll = rotation_matrix_to_euler_deg(_rot_z(-15.0))
    assert (yaw, pitch, roll) == pytest.approx((0.0, 0.0, -15.0), abs=1e-4)

    mat4 = np.eye(4)
    mat4[:3, :3] = _rot_y(-22.5)
    mat4[:3, 3] = [1.0, 2.0, -50.0]
    yaw4, pitch4, roll4 = rotation_matrix_to_euler_deg(mat4)
    assert (yaw4, pitch4, roll4) == pytest.approx((-22.5, 0.0, 0.0), abs=1e-4)

    # Gimbal lock at 90 degrees yaw must not raise
    g_yaw, g_pitch, g_roll = rotation_matrix_to_euler_deg(_rot_y(90.0))
    assert g_yaw == pytest.approx(90.0, abs=1e-3)
    assert math.isfinite(g_pitch) and math.isfinite(g_roll)

    # Non-finite matrix returns zero tuple gracefully
    nan_mat = np.eye(3)
    nan_mat[0, 0] = float("nan")
    assert rotation_matrix_to_euler_deg(nan_mat) == (0.0, 0.0, 0.0)

    with pytest.raises(ValueError):
        rotation_matrix_to_euler_deg(np.eye(2))


@pytest.mark.asyncio
async def test_analyze_rgb_frame_on_blank_and_noise_frames() -> None:
    cfg = AppConfig()
    sensor = MediaPipeVisionSensor(cfg, EventBus())
    try:
        blank = np.zeros((240, 320, 3), dtype=np.uint8)
        rng = np.random.default_rng(42)
        noise = rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8)

        for idx, frame in enumerate([blank, noise]):
            sig = sensor.analyze_rgb_frame(frame, now_ts=float(idx) * 3.0)
            assert isinstance(sig, VisionSignalEvent)
            assert sig.face_present is False
            assert sig.looking_at_screen is False
            assert sig.gaze_on_screen_score == 0.0
            assert 0.0 <= sig.drowsiness_perclos <= 1.0
            assert 0.0 <= sig.yawn_score <= 1.0
            assert 0.0 <= sig.looking_down_score <= 1.0
            assert 0.0 <= sig.phone_confidence <= 1.0
            assert sig.error is None
    finally:
        await sensor.stop()


@pytest.mark.asyncio
async def test_mocked_face_landmarker_blendshapes_and_pose(tmp_path: Path) -> None:
    cfg = AppConfig(data_dir=tmp_path)
    cfg.vision.enable_phone_detector = True
    cfg.behavior.allow_looking_down_for_notes = False
    sensor = MediaPipeVisionSensor(cfg, EventBus())

    mock_landmarker = MagicMock()
    mock_detector = MagicMock()
    sensor._face_landmarker = mock_landmarker
    sensor._object_detector = mock_detector
    sensor._face_model_error = None

    # Rotation: yaw = -18 deg (within default [-55, 25]), pitch = +21 deg
    rot_mat = np.eye(4)
    rot_mat[:3, :3] = _rot_y(-18.0) @ _rot_x(21.0)

    landmarks = [_make_neutral_landmarks()]
    blendshapes_closed = [
        [
            _FakeCategory("eyeBlinkLeft", 0.85),
            _FakeCategory("eyeBlinkRight", 0.75),
            _FakeCategory("jawOpen", 0.62),
            _FakeCategory("eyeLookDownLeft", 0.70),
            _FakeCategory("eyeLookDownRight", 0.90),
        ]
    ]
    blendshapes_open = [
        [
            _FakeCategory("eyeBlinkLeft", 0.10),
            _FakeCategory("eyeBlinkRight", 0.15),
            _FakeCategory("jawOpen", 0.05),
            _FakeCategory("eyeLookDownLeft", 0.10),
            _FakeCategory("eyeLookDownRight", 0.10),
        ]
    ]

    mock_landmarker.detect.side_effect = [
        _FakeFaceResult(landmarks, blendshapes_closed, [rot_mat]),
        _FakeFaceResult(landmarks, blendshapes_open, [np.eye(4)]),
    ]
    mock_detector.detect.return_value = _FakeDetectionResult(
        detections=[_FakeDetection(categories=[_FakeCategory("cell phone", 0.82)])]
    )

    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    sig1 = sensor.analyze_rgb_frame(frame, now_ts=10.0)
    assert sig1.camera_available is True
    assert sig1.face_present is True
    assert sig1.camera_position == "right"
    assert sig1.calibrated is False
    assert sig1.drowsiness_perclos == pytest.approx(1.0)
    assert sig1.yawn_score == pytest.approx(0.62)
    assert sig1.head_yaw_deg == pytest.approx(-18.0, abs=0.5)
    assert sig1.head_pitch_deg == pytest.approx(21.0, abs=0.5)

    eye_delta_1 = min(1.0, max(0.0, (0.80 - 0.22 * 0.75) / 0.45))
    pitch_comp_1 = max(0.0, sig1.head_pitch_deg) / 30.0
    expected_look_down = 0.45 * eye_delta_1 + 0.55 * pitch_comp_1
    assert sig1.looking_down_score == pytest.approx(expected_look_down, abs=1e-3)

    expected_pitch_pen = max(0.0, abs(sig1.head_pitch_deg) - 12.0) / 18.0
    expected_gaze = 1.0 - min(1.0, 0.35 * expected_pitch_pen)
    assert sig1.gaze_on_screen_score == pytest.approx(expected_gaze, abs=1e-4)
    assert sig1.looking_at_screen is False  # looking_down_score >= 0.60
    assert sig1.phone_detected is True
    assert sig1.phone_confidence == pytest.approx(0.82)
    assert mock_detector.detect.call_count == 1

    # Second frame within 1.2s interval: phone detector should use cache, PERCLOS drops to 0.5
    sig2 = sensor.analyze_rgb_frame(frame, now_ts=11.0)
    assert sig2.drowsiness_perclos == pytest.approx(0.5)
    assert sig2.yawn_score == pytest.approx(0.05)
    assert sig2.looking_down_score == pytest.approx(0.0)
    assert sig2.gaze_on_screen_score == pytest.approx(1.0)
    assert sig2.looking_at_screen is True
    assert sig2.phone_detected is True
    assert mock_detector.detect.call_count == 1

    # Third frame after 1.2s interval: phone detector re-evaluates and clears phone
    mock_landmarker.detect.side_effect = [
        _FakeFaceResult(landmarks, blendshapes_open, []),  # empty matrix tests landmark fallback
    ]
    mock_detector.detect.return_value = _FakeDetectionResult(detections=[])
    sig3 = sensor.analyze_rgb_frame(frame, now_ts=13.0)
    assert mock_detector.detect.call_count == 2
    assert sig3.phone_detected is False
    assert sig3.phone_confidence == 0.0
    assert sig3.phone_boxes == []
    assert sig3.head_yaw_deg == pytest.approx(0.0, abs=1e-3)
    assert sig3.head_roll_deg == pytest.approx(0.0, abs=1e-3)

    await sensor.stop()


@pytest.mark.asyncio
async def test_lateral_yaw_within_calibrated_zone_and_outside(tmp_path: Path) -> None:
    cfg = AppConfig(
        data_dir=tmp_path,
        vision=VisionConfig(
            camera_position="right",
            calibrated=True,
            neutral_yaw_deg=-35.0,
            screen_yaw_min_deg=-55.0,
            screen_yaw_max_deg=25.0,
            neutral_pitch_deg=0.0,
            neutral_eye_look_down=0.22,
        ),
    )
    sensor = MediaPipeVisionSensor(cfg, EventBus())
    mock_landmarker = MagicMock()
    sensor._face_landmarker = mock_landmarker
    sensor._face_model_error = None

    landmarks = [_make_neutral_landmarks()]
    neutral_bs = [
        [
            _FakeCategory("eyeBlinkLeft", 0.05),
            _FakeCategory("eyeBlinkRight", 0.05),
            _FakeCategory("jawOpen", 0.0),
            _FakeCategory("eyeLookDownLeft", 0.18),
            _FakeCategory("eyeLookDownRight", 0.20),
        ]
    ]

    rot_lateral = np.eye(4)
    rot_lateral[:3, :3] = _rot_y(-38.0)

    rot_far_away = np.eye(4)
    rot_far_away[:3, :3] = _rot_y(-78.0)

    mock_landmarker.detect.side_effect = [
        _FakeFaceResult(landmarks, neutral_bs, [rot_lateral]),
        _FakeFaceResult(landmarks, neutral_bs, [rot_far_away]),
    ]

    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    sig_lateral = sensor.analyze_rgb_frame(frame, now_ts=1.0)
    assert sig_lateral.head_yaw_deg == pytest.approx(-38.0, abs=0.5)
    assert sig_lateral.calibrated_yaw_delta_deg == pytest.approx(-3.0, abs=0.5)
    assert sig_lateral.looking_at_screen is True
    assert sig_lateral.gaze_on_screen_score >= 0.8

    sig_away = sensor.analyze_rgb_frame(frame, now_ts=2.0)
    assert sig_away.head_yaw_deg == pytest.approx(-78.0, abs=0.5)
    assert sig_away.looking_at_screen is False
    assert sig_away.gaze_on_screen_score < 0.45

    await sensor.stop()


@pytest.mark.asyncio
async def test_calibrated_looking_down_with_phone_pitch_delta(tmp_path: Path) -> None:
    cfg = AppConfig(
        data_dir=tmp_path,
        vision=VisionConfig(
            calibrated=True,
            neutral_yaw_deg=-30.0,
            screen_yaw_min_deg=-55.0,
            screen_yaw_max_deg=25.0,
            neutral_pitch_deg=2.0,
            neutral_eye_look_down=0.20,
            phone_pitch_delta_deg=-18.0,
        ),
    )
    cfg.behavior.allow_looking_down_for_notes = False
    sensor = MediaPipeVisionSensor(cfg, EventBus())
    mock_landmarker = MagicMock()
    sensor._face_landmarker = mock_landmarker
    sensor._face_model_error = None

    landmarks = [_make_neutral_landmarks()]
    bs_down = [[_FakeCategory("eyeLookDownLeft", 0.65), _FakeCategory("eyeLookDownRight", 0.65)]]
    bs_up = [[_FakeCategory("eyeLookDownLeft", 0.15), _FakeCategory("eyeLookDownRight", 0.15)]]

    rot_phone = np.eye(4)
    rot_phone[:3, :3] = _rot_y(-30.0) @ _rot_x(-16.0)  # delta = -18 deg matching phone_pitch_delta

    rot_up = np.eye(4)
    rot_up[:3, :3] = _rot_y(-30.0) @ _rot_x(14.0)  # opposite direction -> pitch_ratio < 0 -> 0

    mock_landmarker.detect.side_effect = [
        _FakeFaceResult(landmarks, bs_down, [rot_phone]),
        _FakeFaceResult(landmarks, bs_up, [rot_up]),
    ]

    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    sig_phone = sensor.analyze_rgb_frame(frame, now_ts=1.0)
    assert sig_phone.calibrated_pitch_delta_deg == pytest.approx(-18.0, abs=0.5)
    assert sig_phone.looking_down_score >= 0.9
    assert sig_phone.looking_at_screen is False

    sig_up = sensor.analyze_rgb_frame(frame, now_ts=2.0)
    assert sig_up.looking_down_score == pytest.approx(0.0)
    assert sig_up.looking_at_screen is True

    # Calibrated without phone_pitch_delta_deg falls back to calibrated pitch deadzone
    cfg.vision.phone_pitch_delta_deg = None
    mock_landmarker.detect.side_effect = [
        _FakeFaceResult(landmarks, bs_down, [rot_phone]),
    ]
    sig_no_delta = sensor.analyze_rgb_frame(frame, now_ts=3.0)
    expected_pitch_comp = min(1.0, max(0.0, abs(-18.0) - 6.0) / 22.0)
    assert sig_no_delta.looking_down_score == pytest.approx(
        0.45 * 1.0 + 0.55 * expected_pitch_comp, abs=1e-3
    )

    await sensor.stop()


@pytest.mark.asyncio
async def test_calibrate_pose_modes_and_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    user_yaml = tmp_path / "user_config.yaml"
    monkeypatch.setattr(
        vision_sensor_mod,
        "save_user_config",
        lambda cfg_obj: save_user_config(cfg_obj, path=user_yaml),
    )

    cfg = AppConfig(data_dir=tmp_path)
    sensor = MediaPipeVisionSensor(cfg, EventBus())
    mock_landmarker = MagicMock()
    sensor._face_landmarker = mock_landmarker
    sensor._face_model_error = None

    # No face seen yet -> returns ok=False gracefully
    empty_res = await sensor.calibrate_pose("primary_screen")
    assert empty_res["ok"] is False
    assert empty_res["samples"] == 0

    landmarks = [_make_neutral_landmarks()]
    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    def _push_pose(yaw: float, pitch: float, eye: float, count: int = 3) -> None:
        mat = np.eye(4)
        mat[:3, :3] = _rot_y(yaw) @ _rot_x(pitch)
        bs = [[_FakeCategory("eyeLookDownLeft", eye), _FakeCategory("eyeLookDownRight", eye)]]
        mock_landmarker.detect.side_effect = [_FakeFaceResult(landmarks, bs, [mat])] * count
        for i in range(count):
            sensor.analyze_rgb_frame(frame, now_ts=float(i) + 1.0)

    # 1. Primary screen calibration
    _push_pose(yaw=-34.0, pitch=5.0, eye=0.24)
    res_primary = await sensor.calibrate_pose("primary_screen")
    assert res_primary["ok"] is True
    assert res_primary["calibrated"] is True
    assert res_primary["neutral_yaw_deg"] == pytest.approx(-34.0, abs=0.2)
    assert res_primary["neutral_pitch_deg"] == pytest.approx(5.0, abs=0.2)
    assert res_primary["neutral_eye_look_down"] == pytest.approx(0.24, abs=0.01)
    assert res_primary["screen_yaw_min_deg"] == pytest.approx(-54.0, abs=0.2)
    assert res_primary["screen_yaw_max_deg"] == pytest.approx(15.0, abs=0.2)
    assert user_yaml.exists()

    # 2. Secondary screen expansion
    _push_pose(yaw=12.0, pitch=2.0, eye=0.20)
    res_secondary = await sensor.calibrate_pose("secondary_screen")
    assert res_secondary["ok"] is True
    assert res_secondary["screen_yaw_min_deg"] == pytest.approx(-54.0, abs=0.2)
    assert res_secondary["screen_yaw_max_deg"] == pytest.approx(30.0, abs=0.2)

    # 3. Phone pose delta
    _push_pose(yaw=-20.0, pitch=-15.0, eye=0.70)
    res_phone = await sensor.calibrate_pose("phone_pose")
    assert res_phone["ok"] is True
    assert res_phone["phone_pitch_delta_deg"] == pytest.approx(-20.0, abs=0.3)

    # 4. Reset defaults
    res_reset = await sensor.calibrate_pose("reset")
    assert res_reset["ok"] is True
    assert res_reset["calibrated"] is False
    assert res_reset["neutral_yaw_deg"] == 0.0
    assert res_reset["screen_yaw_min_deg"] == -55.0
    assert res_reset["screen_yaw_max_deg"] == 25.0
    assert res_reset["neutral_pitch_deg"] == 0.0
    assert res_reset["neutral_eye_look_down"] == 0.22
    assert res_reset["phone_pitch_delta_deg"] is None

    # 5. Read-only / failing save_user_config degrades gracefully without crashing
    def _fail_save(_cfg: AppConfig) -> Path:
        raise PermissionError("read-only disk")

    monkeypatch.setattr(vision_sensor_mod, "save_user_config", _fail_save)
    _push_pose(yaw=-30.0, pitch=2.0, eye=0.20)
    res_ro = await sensor.calibrate_pose("primary_screen")
    assert res_ro["ok"] is True
    assert res_ro["calibrated"] is True

    await sensor.stop()


@pytest.mark.asyncio
async def test_remote_and_chair_rejected_while_person_profile_and_valid_phone_work(
    tmp_path: Path,
) -> None:
    cfg = AppConfig(data_dir=tmp_path)
    cfg.vision.enable_phone_detector = True
    cfg.vision.enable_person_detector = True
    cfg.vision.phone_score_threshold = 0.60
    sensor = MediaPipeVisionSensor(cfg, EventBus())

    mock_landmarker = MagicMock()
    mock_detector = MagicMock()
    sensor._face_landmarker = mock_landmarker
    sensor._object_detector = mock_detector
    sensor._face_model_error = None

    landmarks = [_make_neutral_landmarks()]
    mock_landmarker.detect.side_effect = [
        # Frame 1: "remote" + huge chair box must BOTH be rejected
        _FakeFaceResult(landmarks, [[]], [np.eye(4)]),
        # Frame 2: genuine small hand-held cell phone above 0.60 accepted
        _FakeFaceResult(landmarks, [[]], [np.eye(4)]),
        # Frame 3: side-profile (0 face landmarks) + person detected -> face_present=True!
        _FakeFaceResult([], [], []),
    ]
    mock_detector.detect.side_effect = [
        _FakeDetectionResult(
            detections=[
                _FakeDetection(
                    categories=[_FakeCategory("remote", 0.78)],
                    bounding_box=_FakeBoundingBox(origin_x=40, origin_y=30, width=20, height=30),
                ),
                _FakeDetection(
                    categories=[_FakeCategory("cell phone", 0.75)],
                    # Giant 50% area box like black mesh chair headrest -> rejected
                    bounding_box=_FakeBoundingBox(origin_x=10, origin_y=10, width=100, height=100),
                ),
            ]
        ),
        _FakeDetectionResult(
            detections=[
                _FakeDetection(
                    categories=[_FakeCategory("cell phone", 0.74)],
                    bounding_box=_FakeBoundingBox(origin_x=40, origin_y=30, width=24, height=40),
                )
            ]
        ),
        _FakeDetectionResult(
            detections=[
                _FakeDetection(
                    categories=[_FakeCategory("person", 0.88)],
                    bounding_box=_FakeBoundingBox(origin_x=20, origin_y=10, width=120, height=110),
                )
            ]
        ),
    ]

    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    sig_reject = sensor.analyze_rgb_frame(frame, now_ts=1.0)
    assert sig_reject.phone_detected is False
    assert sig_reject.phone_boxes == []

    sig_valid = sensor.analyze_rgb_frame(frame, now_ts=3.0)
    assert sig_valid.phone_detected is True
    assert sig_valid.phone_confidence == pytest.approx(0.74)
    assert len(sig_valid.phone_boxes) == 1
    assert isinstance(sig_valid.phone_boxes[0], PhoneBox)
    assert sig_valid.phone_boxes[0].label == "cell phone"

    # Side-profile with no face landmarks but person detected stays PRESENT and looking_at_screen!
    sig_profile = sensor.analyze_rgb_frame(frame, now_ts=5.0)
    assert sig_profile.face_present is True
    assert sig_profile.looking_at_screen is True
    assert sig_profile.gaze_on_screen_score >= 0.8

    await sensor.stop()


@pytest.mark.asyncio
async def test_latest_frame_jpeg_in_ram_and_overlay(tmp_path: Path) -> None:
    cfg = AppConfig(data_dir=tmp_path)
    sensor = MediaPipeVisionSensor(cfg, EventBus())
    assert sensor.latest_frame_jpeg() is None

    mock_landmarker = MagicMock()
    mock_detector = MagicMock()
    sensor._face_landmarker = mock_landmarker
    sensor._object_detector = mock_detector
    sensor._face_model_error = None

    landmarks = [_make_neutral_landmarks()]
    mock_landmarker.detect.return_value = _FakeFaceResult(landmarks, [[]], [np.eye(4)])
    mock_detector.detect.return_value = _FakeDetectionResult(
        detections=[
            _FakeDetection(
                categories=[_FakeCategory("cell phone", 0.78)],
                bounding_box=_FakeBoundingBox(origin_x=160, origin_y=60, width=120, height=140),
            )
        ]
    )

    frame = np.full((240, 640, 3), 120, dtype=np.uint8)
    sensor.analyze_rgb_frame(frame, now_ts=1.0)

    raw_jpeg = sensor.latest_frame_jpeg(max_width=480, quality=65, annotate=False)
    assert raw_jpeg is not None
    assert raw_jpeg.startswith(b"\xff\xd8")
    with Image.open(io.BytesIO(raw_jpeg)) as img_raw:
        assert img_raw.size == (480, 180)

    ann_jpeg = sensor.latest_frame_jpeg(max_width=480, quality=65, annotate=True)
    assert ann_jpeg is not None
    assert ann_jpeg.startswith(b"\xff\xd8")
    assert ann_jpeg != raw_jpeg
    with Image.open(io.BytesIO(ann_jpeg)) as img_ann:
        assert img_ann.size == (480, 180)

    # Confirm no image files were written into tmp_path
    assert list(tmp_path.rglob("*.jpg")) == []
    assert list(tmp_path.rglob("*.jpeg")) == []
    assert list(tmp_path.rglob("*.png")) == []

    await sensor.stop()
    assert sensor.latest_frame_jpeg() is None


@pytest.mark.asyncio
async def test_missing_models_degrade_gracefully(tmp_path: Path) -> None:
    cfg = AppConfig(data_dir=tmp_path)
    bus = EventBus()
    received: list[VisionSignalEvent] = []
    bus.subscribe(VisionSignalEvent, received.append)

    sensor = MediaPipeVisionSensor(cfg, bus)
    assert sensor.is_available() is False

    sig = sensor.analyze_rgb_frame(np.zeros((64, 64, 3), dtype=np.uint8))
    assert sig.camera_available is False
    assert sig.face_present is False
    assert sig.looking_at_screen is False
    assert sig.error is not None

    await sensor.start()
    assert sensor.is_available() is False
    assert len(received) == 1
    assert received[0].camera_available is False
    assert received[0].looking_at_screen is False
    await sensor.stop()


@pytest.mark.asyncio
async def test_vision_disabled_emits_single_event_and_skips_camera(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vc_mock = MagicMock()
    monkeypatch.setattr("focus_guard.sensors.vision.vision_sensor.cv2.VideoCapture", vc_mock)

    cfg = AppConfig(vision=VisionConfig(enabled=False))
    bus = EventBus()
    events: list[VisionSignalEvent] = []
    bus.subscribe(VisionSignalEvent, events.append)

    sensor = MediaPipeVisionSensor(cfg, bus)
    await sensor.start()

    assert sensor.is_available() is False
    assert len(events) == 1
    assert events[0].camera_available is False
    assert events[0].looking_at_screen is False
    assert events[0].error == "vision disabled"
    vc_mock.assert_not_called()
    await sensor.stop()


@pytest.mark.asyncio
async def test_camera_open_failure_fallback_warns_once_and_recovers(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    cap_fail = MagicMock()
    cap_fail.isOpened.return_value = False

    cap_ok = MagicMock()
    cap_ok.isOpened.return_value = True
    cap_ok.read.return_value = (True, np.zeros((120, 160, 3), dtype=np.uint8))

    open_attempts = 0

    def fake_video_capture(_index: int) -> MagicMock:
        nonlocal open_attempts
        open_attempts += 1
        return cap_fail if open_attempts <= 2 else cap_ok

    monkeypatch.setattr(
        "focus_guard.sensors.vision.vision_sensor.cv2.VideoCapture", fake_video_capture
    )

    cfg = AppConfig(data_dir=tmp_path, vision=VisionConfig(target_fps=20.0))
    bus = EventBus()
    events: list[VisionSignalEvent] = []
    bus.subscribe(VisionSignalEvent, events.append)

    sensor = MediaPipeVisionSensor(cfg, bus, retry_interval_sec=0.05)
    mock_landmarker = MagicMock()
    mock_landmarker.detect.return_value = _FakeFaceResult([], [], [])
    sensor._face_landmarker = mock_landmarker
    sensor._face_model_error = None

    with caplog.at_level(logging.WARNING, logger="focus_guard.sensors.vision.vision_sensor"):
        await sensor.start()
        for _ in range(40):
            if any(ev.camera_available for ev in events):
                break
            await asyncio.sleep(0.03)
        await sensor.stop()

    unavailable_events = [ev for ev in events if not ev.camera_available]
    recovered_events = [ev for ev in events if ev.camera_available]

    assert len(unavailable_events) >= 2
    assert all(ev.face_present is False and ev.error for ev in unavailable_events)
    assert len(recovered_events) >= 1
    assert sensor.is_available() is False  # False after stop()
    assert sensor.latest_signal() is not None
    cap_fail.release.assert_called()
    cap_ok.release.assert_called_once()
    mock_landmarker.close.assert_called_once()

    warning_msgs = [r.message for r in caplog.records if "Webcam unavailable" in r.message]
    assert len(warning_msgs) == 1


@pytest.mark.asyncio
async def test_missing_phone_model_still_allows_face_landmarker(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True)
    real_face_model = Path(__file__).resolve().parents[2] / "data" / "models" / "face_landmarker.task"
    assert real_face_model.is_file()
    (models_dir / "face_landmarker.task").write_bytes(real_face_model.read_bytes())

    cfg = AppConfig(data_dir=tmp_path)
    sensor = MediaPipeVisionSensor(cfg, EventBus())
    try:
        assert sensor._face_landmarker is not None
        assert sensor._object_detector is None
        sig = sensor.analyze_rgb_frame(np.zeros((120, 160, 3), dtype=np.uint8), now_ts=1.0)
        assert sig.camera_available is True
        assert sig.face_present is False
        assert sig.phone_detected is False
        assert sig.error is None
    finally:
        await sensor.stop()


@pytest.mark.asyncio
async def test_corrupted_face_model_and_invalid_frame_degrade_safely(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True)
    (models_dir / "face_landmarker.task").write_bytes(b"not-a-valid-tflite-bundle")
    (models_dir / "efficientdet_lite0.tflite").write_bytes(b"not-a-valid-tflite")

    cfg = AppConfig(data_dir=tmp_path)
    sensor = MediaPipeVisionSensor(cfg, EventBus())
    assert sensor._face_landmarker is None
    assert sensor._object_detector is None

    bad_frame_sig = sensor.analyze_rgb_frame(np.zeros((10, 10), dtype=np.uint8))
    assert bad_frame_sig.face_present is False
    assert bad_frame_sig.looking_at_screen is False
    assert bad_frame_sig.error == "invalid rgb frame"

    await sensor.stop()
    await sensor.stop()  # idempotent stop


@pytest.mark.asyncio
async def test_obstructed_dark_camera_frame_detected_and_annotated(tmp_path: Path) -> None:
    cfg = AppConfig(data_dir=tmp_path)
    sensor = MediaPipeVisionSensor(cfg, EventBus())
    mock_landmarker = MagicMock()
    mock_landmarker.detect.return_value = _FakeFaceResult([], [], [])
    sensor._face_landmarker = mock_landmarker
    sensor._face_model_error = None
    try:
        dark_frame = np.full((120, 160, 3), 4, dtype=np.uint8)
        sig_dark = sensor.analyze_rgb_frame(dark_frame, now_ts=1.0)
        assert sig_dark.camera_available is True
        assert sig_dark.face_present is False
        assert sig_dark.camera_obstructed is True
        assert sig_dark.looking_at_screen is False

        raw_jpeg = sensor.latest_frame_jpeg(max_width=320, quality=70, annotate=False)
        ann_jpeg = sensor.latest_frame_jpeg(max_width=320, quality=70, annotate=True)
        assert raw_jpeg is not None and ann_jpeg is not None
        assert ann_jpeg != raw_jpeg

        # Bright frame with texture (no face) -> face_present=False, NOT camera_obstructed
        bright_wall = np.full((120, 160, 3), 180, dtype=np.uint8)
        sig_bright = sensor.analyze_rgb_frame(bright_wall, now_ts=3.0)
        assert sig_bright.camera_available is True
        assert sig_bright.face_present is False
        assert sig_bright.camera_obstructed is False
    finally:
        await sensor.stop()


@pytest.mark.asyncio
async def test_vision_loop_sets_buffersize_1_and_survives_slow_event_bus(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cap_ok = MagicMock()
    cap_ok.isOpened.return_value = True
    frame_idx = {"n": 0}

    def _read_frame() -> tuple[bool, np.ndarray]:
        frame_idx["n"] += 1
        val = min(255, 40 + frame_idx["n"] * 10)
        return True, np.full((120, 160, 3), val, dtype=np.uint8)

    cap_ok.read.side_effect = _read_frame
    monkeypatch.setattr(
        "focus_guard.sensors.vision.vision_sensor.cv2.VideoCapture", lambda _idx: cap_ok
    )

    cfg = AppConfig(data_dir=tmp_path, vision=VisionConfig(target_fps=20.0))
    bus = EventBus()
    received: list[VisionSignalEvent] = []
    bus.subscribe(VisionSignalEvent, received.append)

    sensor = MediaPipeVisionSensor(cfg, bus, retry_interval_sec=0.05)
    mock_landmarker = MagicMock()
    mock_landmarker.detect.return_value = _FakeFaceResult([], [], [])
    sensor._face_landmarker = mock_landmarker
    sensor._face_model_error = None

    await sensor.start()
    try:
        for _ in range(30):
            if frame_idx["n"] >= 2:
                break
            await asyncio.sleep(0.03)
    finally:
        await sensor.stop()

    assert frame_idx["n"] >= 2
    assert len(received) >= 2
    cap_ok.set.assert_any_call(vision_sensor_mod.cv2.CAP_PROP_BUFFERSIZE, 1)


@pytest.mark.asyncio
async def test_looking_down_for_notes_suppresses_false_perclos_fatigue(tmp_path: Path) -> None:
    cfg = AppConfig(
        data_dir=tmp_path,
        vision=VisionConfig(
            calibrated=True,
            neutral_yaw_deg=0.0,
            neutral_pitch_deg=0.0,
            neutral_eye_look_down=0.22,
            phone_pitch_delta_deg=11.2,
        ),
    )
    cfg.behavior.allow_looking_down_for_notes = True
    sensor = MediaPipeVisionSensor(cfg, EventBus())
    mock_landmarker = MagicMock()
    sensor._face_landmarker = mock_landmarker
    sensor._face_model_error = None

    landmarks = [_make_neutral_landmarks()]
    # Looking down to write notes: high eyeBlink caused by downward eyelids + pitch=+12 deg
    rot_down = np.eye(4)
    rot_down[:3, :3] = _rot_x(12.0)
    bs_writing_down = [
        [
            _FakeCategory("eyeBlinkLeft", 0.78),
            _FakeCategory("eyeBlinkRight", 0.72),
            _FakeCategory("eyeLookDownLeft", 0.60),
            _FakeCategory("eyeLookDownRight", 0.62),
        ]
    ]
    # Genuine closed eyes while looking straight at the monitor (pitch=0, eyeLookDown=0.15)
    bs_genuinely_drowsy = [
        [
            _FakeCategory("eyeBlinkLeft", 0.85),
            _FakeCategory("eyeBlinkRight", 0.82),
            _FakeCategory("eyeLookDownLeft", 0.15),
            _FakeCategory("eyeLookDownRight", 0.15),
        ]
    ]

    mock_landmarker.detect.side_effect = [
        _FakeFaceResult(landmarks, bs_writing_down, [rot_down]),
        _FakeFaceResult(landmarks, bs_genuinely_drowsy, [np.eye(4)]),
    ]

    frame = np.full((120, 160, 3), 120, dtype=np.uint8)
    try:
        sig_writing = sensor.analyze_rgb_frame(frame, now_ts=1.0)
        assert sig_writing.looking_down_score >= 0.30
        assert sig_writing.drowsiness_perclos == pytest.approx(0.0)

        sig_drowsy = sensor.analyze_rgb_frame(frame, now_ts=2.0)
        assert sig_drowsy.drowsiness_perclos == pytest.approx(0.5)
    finally:
        await sensor.stop()
