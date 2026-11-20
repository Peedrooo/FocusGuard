"""Unit tests for the FocusGuard OS telemetry & ephemeral screen capture subsystem."""

from __future__ import annotations

import asyncio
import io
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import psutil
import pytest
from PIL import Image, ImageDraw

from focus_guard.core.config import AIConfig, AppConfig, PrivacyConfig
from focus_guard.core.event_bus import EventBus
from focus_guard.core.events import (
    OSContextEvent,
    ProcessActivityInfo,
    ScreenSnapshot,
    VisibleWindowInfo,
)
from focus_guard.core.ports import OSSensorPort, ScreenCapturePort
from focus_guard.core.privacy import PrivacyShield
from focus_guard.sensors.os_monitor import (
    MSSScreenCapture,
    UsageBehaviorTracker,
    WindowsOSSensor,
    compute_dhash,
)
from focus_guard.sensors.os_monitor import os_sensor as os_sensor_mod
from focus_guard.sensors.os_monitor import screen_capture as screen_capture_mod


def _hamming_distance(hex_a: str, hex_b: str) -> int:
    return (int(hex_a, 16) ^ int(hex_b, 16)).bit_count()


def _make_config() -> AppConfig:
    return AppConfig(
        ai=AIConfig(screenshot_max_width=768, screenshot_jpeg_quality=70),
        privacy=PrivacyConfig(
            blocked_process_names=["bitwarden.exe", "keepassxc.exe"],
            blocked_title_keywords=["anonima", "nubank", "internet banking", "pix"],
        ),
    )


def _make_bgrx_frame(width: int, height: int) -> tuple[bytes, tuple[int, int]]:
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)
    for x in range(0, width, 32):
        color = (x % 255, (x * 2) % 255, 180)
        draw.rectangle([x, 0, min(width - 1, x + 31), height - 1], fill=color)
    rgba = img.convert("RGBA")
    r, g, b, a = rgba.split()
    bgra = Image.merge("RGBA", (b, g, r, a)).tobytes()
    return bgra, (width, height)


def test_exports_and_protocol_conformance() -> None:
    cfg = _make_config()
    bus = EventBus()
    sensor = WindowsOSSensor(cfg, bus)
    capture = MSSScreenCapture(cfg)

    assert isinstance(sensor, OSSensorPort)
    assert isinstance(capture, ScreenCapturePort)


def test_compute_dhash_determinism_similarity_and_difference() -> None:
    # Horizontal bright-to-dark gradient
    grad_lr = Image.new("RGB", (180, 120))
    for x in range(180):
        val = 255 - int((x / 179) * 255)
        for y in range(120):
            grad_lr.putpixel((x, y), (val, val, val))

    # Slightly modified version (small local change)
    grad_modified = grad_lr.copy()
    for x in range(10, 20):
        for y in range(10, 20):
            grad_modified.putpixel((x, y), (0, 0, 0))

    # Opposite gradient (dark-to-bright)
    grad_rl = Image.new("RGB", (180, 120))
    for x in range(180):
        val = int((x / 179) * 255)
        for y in range(120):
            grad_rl.putpixel((x, y), (val, val, val))

    h1 = compute_dhash(grad_lr)
    h1_again = compute_dhash(grad_lr)
    h_mod = compute_dhash(grad_modified)
    h_opp = compute_dhash(grad_rl)

    assert len(h1) == 16
    int(h1, 16)  # valid hex
    assert h1 == h1_again
    assert _hamming_distance(h1, h_mod) <= 8
    assert _hamming_distance(h1, h_opp) >= 48


def test_screen_capture_skips_sensitive_windows_without_calling_mss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _make_config()
    capturer = MSSScreenCapture(cfg)
    mss_mock = MagicMock(side_effect=AssertionError("mss.mss() must not be called"))
    monkeypatch.setattr(screen_capture_mod.mss, "mss", mss_mock)

    # 1) Already flagged sensitive on OSContextEvent
    sensitive_ev = OSContextEvent(
        process_name="chrome.exe",
        window_title="Nubank",
        is_sensitive_window=True,
        sensitive_reason="keyword:nubank",
    )
    assert capturer.capture(sensitive_ev) is None

    # 2) Even if caller forgot to set is_sensitive_window, PrivacyShield inside capture blocks it
    unflagged_sensitive_ev = OSContextEvent(
        process_name="Bitwarden.exe",
        window_title="Vault",
        is_sensitive_window=False,
    )
    assert capturer.capture(unflagged_sensitive_ev) is None
    assert mss_mock.call_count == 0


def test_screen_capture_resizes_and_encodes_jpeg_in_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _make_config()
    cfg.ai.capture_mode = "primary_monitor"
    capturer = MSSScreenCapture(cfg)
    bgra, size = _make_bgrx_frame(1920, 1080)

    fake_sct_img = SimpleNamespace(size=size, bgra=bgra)
    fake_sct = MagicMock()
    fake_sct.monitors = [
        {"left": 0, "top": 0, "width": 1920, "height": 1080},
        {"left": 0, "top": 0, "width": 1920, "height": 1080},
    ]
    fake_sct.grab.return_value = fake_sct_img
    fake_ctx = MagicMock()
    fake_ctx.__enter__.return_value = fake_sct
    fake_ctx.__exit__.return_value = False
    monkeypatch.setattr(screen_capture_mod.mss, "mss", MagicMock(return_value=fake_ctx))

    os_ev = OSContextEvent(
        process_name="chrome.exe",
        window_title="Tec Concursos - Direito Constitucional",
        browser_url="https://www.tecconcursos.com.br/questoes",
        is_sensitive_window=False,
    )
    snap = capturer.capture(os_ev)

    assert isinstance(snap, ScreenSnapshot)
    assert snap.width == 768
    assert snap.height == 432
    assert snap.monitor_index == 1
    assert len(snap.perceptual_hash) == 16
    int(snap.perceptual_hash, 16)
    assert snap.image_bytes_jpeg.startswith(b"\xff\xd8")

    decoded = Image.open(io.BytesIO(snap.image_bytes_jpeg))
    assert decoded.format == "JPEG"
    assert decoded.size == (768, 432)
    fake_sct.grab.assert_called_once_with(fake_sct.monitors[1])


def test_screen_capture_handles_mss_failure_gracefully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _make_config()
    capturer = MSSScreenCapture(cfg)
    monkeypatch.setattr(
        screen_capture_mod.mss,
        "mss",
        MagicMock(side_effect=RuntimeError("Display device unavailable")),
    )

    snap = capturer.capture(
        OSContextEvent(process_name="code.exe", window_title="editor.py")
    )
    assert snap is None


def test_os_sensor_browser_url_caching_audio_privacy_and_idle_wraparound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _make_config()
    bus = EventBus()
    sensor = WindowsOSSensor(cfg, bus, privacy_shield=PrivacyShield(cfg.privacy))

    state: dict[str, Any] = {
        "hwnd": 101,
        "title": "Tec Concursos - Google Chrome",
        "pid": 4242,
        "proc_name": "chrome.exe",
        "last_input": 0xFFFFFF00,  # 256 ms before 32-bit wrap
        "tick": 0x00000100,  # 256 ms after 32-bit wrap => 512 ms idle
    }
    uia_calls: list[int] = []
    urls_by_title: dict[str, str] = {
        "Tec Concursos - Google Chrome": "www.tecconcursos.com.br/questoes",
        "YouTube - Google Chrome": "https://www.youtube.com/watch?v=abc",
        "Internet Banking - Google Chrome": "https://internetbanking.nubank.com.br/app",
    }

    monkeypatch.setattr(os_sensor_mod.win32gui, "GetForegroundWindow", lambda: state["hwnd"])
    monkeypatch.setattr(
        os_sensor_mod.win32gui, "GetWindowText", lambda _h: state["title"]
    )
    monkeypatch.setattr(
        os_sensor_mod.win32process,
        "GetWindowThreadProcessId",
        lambda _h: (1, state["pid"]),
    )
    monkeypatch.setattr(
        os_sensor_mod.psutil,
        "Process",
        lambda _pid: SimpleNamespace(name=lambda: state["proc_name"]),
    )
    monkeypatch.setattr(
        os_sensor_mod.win32api, "GetLastInputInfo", lambda: state["last_input"]
    )
    monkeypatch.setattr(os_sensor_mod.win32api, "GetTickCount", lambda: state["tick"])

    def fake_uia_query(hwnd: int, _proc: str) -> str | None:
        uia_calls.append(hwnd)
        raw = urls_by_title[str(state["title"])]
        return os_sensor_mod._normalize_browser_url(raw)

    monkeypatch.setattr(sensor, "_query_uia_browser_url", fake_uia_query)

    # Audio sessions: 1 active via peak, 1 active via State==1, 1 silent, 1 system (Process=None), 1 broken
    silent_ctl = MagicMock()
    silent_ctl.QueryInterface.return_value.GetPeakValue.return_value = 0.0
    loud_ctl = MagicMock()
    loud_ctl.QueryInterface.return_value.GetPeakValue.return_value = 0.35
    broken_proc = MagicMock()
    broken_proc.name.side_effect = psutil.AccessDenied(pid=99)

    sessions = [
        SimpleNamespace(Process=None, State=1, _ctl=None),
        SimpleNamespace(
            Process=SimpleNamespace(name=lambda: "Spotify.exe"),
            State=0,
            _ctl=loud_ctl,
        ),
        SimpleNamespace(
            Process=SimpleNamespace(name=lambda: "Chrome.exe"),
            State=1,
            _ctl=silent_ctl,
        ),
        SimpleNamespace(
            Process=SimpleNamespace(name=lambda: "chrome.exe"),
            State=1,
            _ctl=loud_ctl,
        ),
        SimpleNamespace(
            Process=SimpleNamespace(name=lambda: "discord.exe"),
            State=0,
            _ctl=silent_ctl,
        ),
        SimpleNamespace(Process=broken_proc, State=1, _ctl=None),
    ]
    co_init = MagicMock()
    co_uninit = MagicMock()
    monkeypatch.setattr(os_sensor_mod.comtypes, "CoInitialize", co_init)
    monkeypatch.setattr(os_sensor_mod.comtypes, "CoUninitialize", co_uninit)
    monkeypatch.setattr(
        os_sensor_mod.pycaw.AudioUtilities, "GetAllSessions", lambda: sessions
    )

    # 1) Initial sample: browser URL extracted via UIA and normalized
    ev1 = sensor.sample_context()
    assert ev1.process_name == "chrome.exe"
    assert ev1.window_title == "Tec Concursos - Google Chrome"
    assert ev1.browser_url == "https://www.tecconcursos.com.br/questoes"
    assert ev1.idle_seconds == pytest.approx(0.512, abs=0.001)
    assert ev1.background_audio_apps == ["chrome.exe", "spotify.exe"]
    assert ev1.is_sensitive_window is False
    assert len(uia_calls) == 1
    assert co_init.call_count == 1
    assert co_uninit.call_count == 1

    # 2) Same (hwnd, window_title): served from cache without polling UIA again
    ev2 = sensor.sample_context()
    assert ev2.browser_url == "https://www.tecconcursos.com.br/questoes"
    assert len(uia_calls) == 1

    # 3) Tab switches -> UIA polled once for new title
    state["title"] = "YouTube - Google Chrome"
    ev3 = sensor.sample_context()
    assert ev3.browser_url == "https://www.youtube.com/watch?v=abc"
    assert len(uia_calls) == 2

    # 4) Tab switches back to cached title -> LRU hit, still 2 UIA calls
    state["title"] = "Tec Concursos - Google Chrome"
    ev4 = sensor.sample_context()
    assert ev4.browser_url == "https://www.tecconcursos.com.br/questoes"
    assert len(uia_calls) == 2

    # 5) Sensitive banking URL flags PrivacyShield
    state["title"] = "Internet Banking - Google Chrome"
    ev5 = sensor.sample_context()
    assert ev5.browser_url == "https://internetbanking.nubank.com.br/app"
    assert ev5.is_sensitive_window is True
    assert ev5.sensitive_reason is not None

    # 6) Non-browser window does not invoke UIA
    state["hwnd"] = 202
    state["proc_name"] = "Code.exe"
    state["title"] = "main.py - FocusGuard"
    ev6 = sensor.sample_context()
    assert ev6.browser_url is None
    assert len(uia_calls) == 3


def test_os_sensor_uia_mock_control_and_timeout_resilience(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _make_config()
    bus = EventBus()
    sensor = WindowsOSSensor(cfg, bus, uia_timeout_sec=0.05)

    monkeypatch.setattr(os_sensor_mod.win32gui, "GetForegroundWindow", lambda: 555)
    monkeypatch.setattr(
        os_sensor_mod.win32gui, "GetWindowText", lambda _h: "Docs - Edge"
    )
    monkeypatch.setattr(
        os_sensor_mod.win32process, "GetWindowThreadProcessId", lambda _h: (1, 900)
    )
    monkeypatch.setattr(
        os_sensor_mod.psutil,
        "Process",
        lambda _pid: SimpleNamespace(name=lambda: "msedge.exe"),
    )
    monkeypatch.setattr(os_sensor_mod.win32api, "GetLastInputInfo", lambda: 1000)
    monkeypatch.setattr(os_sensor_mod.win32api, "GetTickCount", lambda: 2000)
    monkeypatch.setattr(os_sensor_mod.pycaw.AudioUtilities, "GetAllSessions", list)

    # Test real UIAutomation initializer + ControlFromHandle path with mock EditControl
    fake_init = MagicMock()
    fake_init.return_value.__enter__.return_value = fake_init
    fake_init.return_value.__exit__.return_value = False

    fake_edit = MagicMock()
    fake_edit.Exists.return_value = True
    fake_edit.GetValuePattern.return_value = SimpleNamespace(
        Value="https://learn.microsoft.com/en-us/"
    )
    fake_ctrl = MagicMock()
    fake_ctrl.EditControl.return_value = fake_edit

    monkeypatch.setattr(os_sensor_mod.auto, "UIAutomationInitializerInThread", fake_init)
    monkeypatch.setattr(
        os_sensor_mod.auto, "ControlFromHandle", MagicMock(return_value=fake_ctrl)
    )

    ev = sensor.sample_context()
    assert ev.browser_url == "https://learn.microsoft.com/en-us/"

    # Now test UIA hanging past timeout -> returns None rapidly without blocking
    def slow_control(_hwnd: int) -> None:
        time.sleep(0.3)
        return None

    monkeypatch.setattr(os_sensor_mod.win32gui, "GetWindowText", lambda _h: "Hung Tab")
    monkeypatch.setattr(os_sensor_mod.auto, "ControlFromHandle", slow_control)

    start = time.monotonic()
    ev_timeout = sensor.sample_context()
    elapsed = time.monotonic() - start
    assert ev_timeout.browser_url is None
    assert elapsed < 0.25


def test_os_sensor_handles_win32_and_psutil_errors_gracefully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _make_config()
    bus = EventBus()
    sensor = WindowsOSSensor(cfg, bus)

    monkeypatch.setattr(os_sensor_mod.win32gui, "GetForegroundWindow", lambda: 999)
    monkeypatch.setattr(
        os_sensor_mod.win32gui, "GetWindowText", lambda _h: "Elevated Window"
    )
    monkeypatch.setattr(
        os_sensor_mod.win32process, "GetWindowThreadProcessId", lambda _h: (1, 4)
    )
    monkeypatch.setattr(
        os_sensor_mod.psutil,
        "Process",
        MagicMock(side_effect=psutil.AccessDenied(pid=4)),
    )
    monkeypatch.setattr(
        os_sensor_mod.win32api,
        "GetLastInputInfo",
        MagicMock(side_effect=OSError("Win32 error")),
    )
    monkeypatch.setattr(
        os_sensor_mod.pycaw.AudioUtilities,
        "GetAllSessions",
        MagicMock(side_effect=RuntimeError("Audio service stopped")),
    )

    ev = sensor.sample_context()
    assert ev.process_name == ""
    assert ev.window_title == "Elevated Window"
    assert ev.browser_url is None
    assert ev.background_audio_apps == []
    assert ev.idle_seconds == 0.0
    assert ev.is_sensitive_window is False


@pytest.mark.asyncio
async def test_os_sensor_async_polling_publishes_and_stops_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _make_config()
    bus = EventBus()
    received: list[OSContextEvent] = []

    async def on_event(ev: OSContextEvent) -> None:
        received.append(ev)

    bus.subscribe(OSContextEvent, on_event)
    sensor = WindowsOSSensor(cfg, bus, poll_interval_sec=0.02)

    monkeypatch.setattr(
        sensor,
        "sample_context",
        lambda: OSContextEvent(process_name="obsidian.exe", window_title="Resumo.md"),
    )

    await sensor.start()
    await asyncio.sleep(0.08)
    await sensor.stop()

    count_at_stop = len(received)
    assert count_at_stop >= 1
    assert all(ev.process_name == "obsidian.exe" for ev in received)

    # Confirm no more events arrive after stop
    await asyncio.sleep(0.05)
    assert len(received) == count_at_stop


def test_screen_capture_single_monitor_and_no_upscale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _make_config()
    capturer = MSSScreenCapture(cfg)
    bgra, size = _make_bgrx_frame(400, 300)

    fake_sct_img = SimpleNamespace(size=size, bgra=bgra)
    fake_sct = MagicMock()
    fake_sct.monitors = [{"left": 0, "top": 0, "width": 400, "height": 300}]
    fake_sct.grab.return_value = fake_sct_img
    fake_ctx = MagicMock()
    fake_ctx.__enter__.return_value = fake_sct
    fake_ctx.__exit__.return_value = False
    monkeypatch.setattr(screen_capture_mod.mss, "mss", MagicMock(return_value=fake_ctx))

    snap = capturer.capture()
    assert isinstance(snap, ScreenSnapshot)
    assert snap.monitor_index == 0
    assert snap.width == 400
    assert snap.height == 300
    fake_sct.grab.assert_called_once_with(fake_sct.monitors[0])


def test_os_sensor_lru_eviction_and_zero_hwnd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _make_config()
    bus = EventBus()
    sensor = WindowsOSSensor(cfg, bus, url_cache_maxsize=2)

    state: dict[str, Any] = {"hwnd": 0, "title": "Tab A"}
    uia_lookups: list[str] = []

    monkeypatch.setattr(os_sensor_mod.win32gui, "GetForegroundWindow", lambda: state["hwnd"])
    monkeypatch.setattr(os_sensor_mod.win32gui, "GetWindowText", lambda _h: state["title"])
    monkeypatch.setattr(
        os_sensor_mod.win32process, "GetWindowThreadProcessId", lambda _h: (1, 100)
    )
    monkeypatch.setattr(
        os_sensor_mod.psutil,
        "Process",
        lambda _pid: SimpleNamespace(name=lambda: "firefox.exe"),
    )
    monkeypatch.setattr(os_sensor_mod.win32api, "GetLastInputInfo", lambda: 100)
    monkeypatch.setattr(os_sensor_mod.win32api, "GetTickCount", lambda: 100)
    monkeypatch.setattr(os_sensor_mod.pycaw.AudioUtilities, "GetAllSessions", list)

    def fake_uia(_hwnd: int, _proc: str) -> str | None:
        title = str(state["title"])
        uia_lookups.append(title)
        return f"https://example.com/{title.replace(' ', '_')}"

    monkeypatch.setattr(sensor, "_query_uia_browser_url", fake_uia)

    # Zero hwnd degrades cleanly with no browser lookup
    ev0 = sensor.sample_context()
    assert ev0.process_name == ""
    assert ev0.window_title == ""
    assert ev0.browser_url is None
    assert uia_lookups == []

    # Populate 3 distinct titles into maxsize=2 cache -> Tab A is evicted
    state["hwnd"] = 77
    for title in ("Tab A", "Tab B", "Tab C"):
        state["title"] = title
        ev = sensor.sample_context()
        assert ev.browser_url == f"https://example.com/{title.replace(' ', '_')}"
    assert uia_lookups == ["Tab A", "Tab B", "Tab C"]

    # Tab B is still cached; Tab A was evicted and must re-query
    state["title"] = "Tab B"
    sensor.sample_context()
    assert uia_lookups == ["Tab A", "Tab B", "Tab C"]

    state["title"] = "Tab A"
    sensor.sample_context()
    assert uia_lookups == ["Tab A", "Tab B", "Tab C", "Tab A"]


def _make_solid_bgrx_frame(
    width: int, height: int, rgb: tuple[int, int, int]
) -> tuple[bytes, tuple[int, int]]:
    img = Image.new("RGBA", (width, height), (*rgb, 255))
    r, g, b, a = img.split()
    bgra = Image.merge("RGBA", (b, g, r, a)).tobytes()
    return bgra, (width, height)


def test_screen_capture_all_monitors_vs_primary_monitor_at_index_2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitors = [
        {"left": 0, "top": 0, "width": 4480, "height": 1080},
        {"left": 2560, "top": 0, "width": 1920, "height": 1080, "is_primary": False},
        {"left": 0, "top": 0, "width": 2560, "height": 1080, "is_primary": True},
    ]

    def _fake_grab(mon: dict[str, Any]) -> SimpleNamespace:
        bgra, size = _make_bgrx_frame(int(mon["width"]), int(mon["height"]))
        return SimpleNamespace(size=size, bgra=bgra)

    fake_sct = MagicMock()
    fake_sct.monitors = monitors
    fake_sct.grab.side_effect = _fake_grab
    fake_ctx = MagicMock()
    fake_ctx.__enter__.return_value = fake_sct
    fake_ctx.__exit__.return_value = False
    monkeypatch.setattr(screen_capture_mod.mss, "mss", MagicMock(return_value=fake_ctx))

    # 1) all_monitors -> monitors[0] (4480x1080 resized to 1400x338)
    cfg_all = _make_config()
    cfg_all.ai.capture_mode = "all_monitors"
    cfg_all.ai.screenshot_max_width = 1400
    snap_all = MSSScreenCapture(cfg_all).capture()
    assert snap_all is not None
    assert snap_all.monitor_index == 0
    assert snap_all.width == 1400
    assert snap_all.height == 338
    fake_sct.grab.assert_called_with(monitors[0])

    # 2) primary_monitor -> picks monitors[2] because is_primary=True is at index 2
    cfg_prim = _make_config()
    cfg_prim.ai.capture_mode = "primary_monitor"
    cfg_prim.ai.screenshot_max_width = 1400
    snap_prim = MSSScreenCapture(cfg_prim).capture()
    assert snap_prim is not None
    assert snap_prim.monitor_index == 2
    assert snap_prim.width == 1400
    fake_sct.grab.assert_called_with(monitors[2])

    # 3) primary_monitor without is_primary key -> matches (left==0, top==0) at index 2
    fake_sct.monitors = [
        {"left": 0, "top": 0, "width": 4480, "height": 1080},
        {"left": 2560, "top": 0, "width": 1920, "height": 1080},
        {"left": 0, "top": 0, "width": 2560, "height": 1080},
    ]
    snap_origin = MSSScreenCapture(cfg_prim).capture()
    assert snap_origin is not None
    assert snap_origin.monitor_index == 2

    # 3b) primary_monitor with neither is_primary nor (0,0) -> falls back to monitors[1]
    fake_sct.monitors = [
        {"left": 100, "top": 100, "width": 3840, "height": 1080},
        {"left": 100, "top": 100, "width": 1920, "height": 1080},
        {"left": 2020, "top": 100, "width": 1920, "height": 1080},
    ]
    snap_prim_fallback = MSSScreenCapture(cfg_prim).capture()
    assert snap_prim_fallback is not None
    assert snap_prim_fallback.monitor_index == 1

    # Restore dual-monitor setup with is_primary=True at index 2
    fake_sct.monitors = monitors

    # 4) active_monitor -> picks monitor containing foreground window center
    cfg_active = _make_config()
    cfg_active.ai.capture_mode = "active_monitor"
    cfg_active.ai.screenshot_max_width = 1400
    os_ev_mon1 = OSContextEvent(
        process_name="code.exe",
        window_title="editor.py",
        visible_windows=[
            VisibleWindowInfo(
                hwnd=10,
                process_name="code.exe",
                window_title="editor.py",
                rect=(2700, 100, 4100, 900),
                monitor_label="M2-Secundario",
                is_foreground=True,
                is_sensitive=False,
            )
        ],
    )
    snap_active = MSSScreenCapture(cfg_active).capture(os_ev_mon1)
    assert snap_active is not None
    assert snap_active.monitor_index == 1

    # 5) active_monitor without foreground window falls back to primary monitor (index 2)
    snap_fallback = MSSScreenCapture(cfg_active).capture(
        OSContextEvent(process_name="code.exe", window_title="editor.py")
    )
    assert snap_fallback is not None
    assert snap_fallback.monitor_index == 2


def test_screen_capture_rectangle_privacy_masking_in_ram(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _make_config()
    cfg.ai.capture_mode = "all_monitors"
    cfg.ai.screenshot_max_width = 1000
    cfg.ai.screenshot_jpeg_quality = 95
    capturer = MSSScreenCapture(cfg)

    # 2000x1000 bright orange frame -> resized to 1000x500 (scale = 0.5)
    bgra, size = _make_solid_bgrx_frame(2000, 1000, (240, 120, 40))
    fake_sct_img = SimpleNamespace(size=size, bgra=bgra)
    fake_sct = MagicMock()
    fake_sct.monitors = [
        {"left": 0, "top": 0, "width": 2000, "height": 1000},
        {"left": 0, "top": 0, "width": 1000, "height": 1000, "is_primary": True},
        {"left": 1000, "top": 0, "width": 1000, "height": 1000, "is_primary": False},
    ]
    fake_sct.grab.return_value = fake_sct_img
    fake_ctx = MagicMock()
    fake_ctx.__enter__.return_value = fake_sct
    fake_ctx.__exit__.return_value = False
    monkeypatch.setattr(screen_capture_mod.mss, "mss", MagicMock(return_value=fake_ctx))

    os_ev = OSContextEvent(
        process_name="obsidian.exe",
        window_title="Direito Constitucional.md",
        is_sensitive_window=False,
        visible_windows=[
            VisibleWindowInfo(
                hwnd=1,
                process_name="obsidian.exe",
                window_title="Direito Constitucional.md",
                rect=(50, 50, 900, 900),
                monitor_label="M1-Principal",
                is_foreground=True,
                is_sensitive=False,
            ),
            VisibleWindowInfo(
                hwnd=2,
                process_name="chrome.exe",
                window_title="[JANELA PROTEGIDA]",
                rect=(1100, 200, 1800, 800),
                monitor_label="M2-Secundario",
                is_foreground=False,
                is_sensitive=True,
            ),
        ],
    )

    snap = capturer.capture(os_ev)
    assert snap is not None
    assert snap.width == 1000
    assert snap.height == 500

    decoded = Image.open(io.BytesIO(snap.image_bytes_jpeg)).convert("RGB")
    # Inside sensitive box (1100..1800, 200..800) scaled by 0.5 -> (550..900, 100..400)
    masked_px = decoded.getpixel((725, 250))
    assert isinstance(masked_px, tuple)
    assert all(channel <= 15 for channel in masked_px)

    # Inside foreground non-sensitive box (50..900, 50..900) scaled by 0.5 -> (25..450, 25..450)
    intact_px = decoded.getpixel((200, 250))
    assert isinstance(intact_px, tuple)
    assert intact_px[0] > 180
    assert intact_px[1] > 80

    # Also test monitor coordinate translation when capturing secondary monitor (left=1000)
    cfg.ai.capture_mode = "active_monitor"
    cfg.ai.screenshot_max_width = 1000
    bgra_sec, size_sec = _make_solid_bgrx_frame(1000, 1000, (40, 180, 230))
    fake_sct.grab.return_value = SimpleNamespace(size=size_sec, bgra=bgra_sec)
    os_ev_sec = OSContextEvent(
        process_name="code.exe",
        window_title="main.py",
        is_sensitive_window=False,
        visible_windows=[
            VisibleWindowInfo(
                hwnd=3,
                process_name="code.exe",
                window_title="main.py",
                rect=(1020, 50, 1450, 900),
                monitor_label="M2-Secundario",
                is_foreground=True,
                is_sensitive=False,
            ),
            VisibleWindowInfo(
                hwnd=4,
                process_name="bitwarden.exe",
                window_title="[JANELA PROTEGIDA]",
                rect=(1500, 200, 1950, 700),
                monitor_label="M2-Secundario",
                is_foreground=False,
                is_sensitive=True,
            ),
        ],
    )
    snap_sec = capturer.capture(os_ev_sec)
    assert snap_sec is not None
    assert snap_sec.monitor_index == 2
    decoded_sec = Image.open(io.BytesIO(snap_sec.image_bytes_jpeg)).convert("RGB")
    # Translated: (1500-1000, 200, 1950-1000, 700) -> (500, 200, 950, 700)
    masked_sec_px = decoded_sec.getpixel((700, 450))
    intact_sec_px = decoded_sec.getpixel((200, 450))
    assert isinstance(masked_sec_px, tuple)
    assert isinstance(intact_sec_px, tuple)
    assert all(c <= 15 for c in masked_sec_px)
    assert intact_sec_px[2] > 180


def test_usage_behavior_tracker_modes_and_sliding_window() -> None:
    tracker = UsageBehaviorTracker(window_seconds=60.0)

    # 1) active_work: idle < 2.0 on steady screen increments continuous_same_screen_sec
    m1 = tracker.record("obsidian.exe|Resumo|", idle_seconds=0.3, now=100.0)
    assert m1.interaction_mode == "active_work"
    assert m1.active_input_ratio_60s == 1.0
    assert m1.window_switches_60s == 0
    assert m1.continuous_same_screen_sec == 0.0

    m2 = tracker.record("obsidian.exe|Resumo|", idle_seconds=0.8, now=112.5)
    assert m2.interaction_mode == "active_work"
    assert m2.continuous_same_screen_sec == 12.5

    # Signature change resets continuous_same_screen_sec
    m3 = tracker.record("chrome.exe|Tec Concursos|", idle_seconds=0.5, now=115.0)
    assert m3.window_switches_60s == 1
    assert m3.continuous_same_screen_sec == 0.0

    # 2) rapid_switching: >= 8 signature transitions in 60s
    rapid_tracker = UsageBehaviorTracker(window_seconds=60.0)
    for idx in range(9):
        sig = f"app_{idx % 2}.exe|Win {idx}|"
        metrics = rapid_tracker.record(sig, idle_seconds=0.2, now=200.0 + idx)
    assert metrics.window_switches_60s == 8
    assert metrics.interaction_mode == "rapid_switching"

    # After 60s window passes on same screen, old switches expire -> back to active_work
    settled = rapid_tracker.record("app_0.exe|Win 8|", idle_seconds=0.4, now=270.0)
    assert settled.window_switches_60s == 0
    assert settled.interaction_mode == "active_work"

    # 3) idle_with_media: idle >= 15s + active background audio
    media_tracker = UsageBehaviorTracker(window_seconds=60.0)
    m_media = media_tracker.record(
        "obsidian.exe|Resumo|",
        idle_seconds=22.0,
        background_audio_apps=["spotify.exe"],
        now=300.0,
    )
    assert m_media.interaction_mode == "idle_with_media"
    assert m_media.active_input_ratio_60s == 0.0

    # 4) reading_or_watching: 2s <= idle < 45s without audio
    m_read = media_tracker.record(
        "obsidian.exe|Resumo|",
        idle_seconds=25.0,
        background_audio_apps=[],
        now=305.0,
    )
    assert m_read.interaction_mode == "reading_or_watching"

    # 5) idle: idle >= 45s without audio
    idle_tracker = UsageBehaviorTracker(window_seconds=60.0)
    m_idle = idle_tracker.record(
        "obsidian.exe|Resumo|",
        idle_seconds=65.0,
        background_audio_apps=[],
        now=400.0,
    )
    assert m_idle.interaction_mode == "idle"
    assert m_idle.continuous_same_screen_sec == 0.0


def test_os_sensor_visible_windows_privacy_monitors_and_notable_processes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _make_config()
    bus = EventBus()
    sensor = WindowsOSSensor(cfg, bus, privacy_shield=PrivacyShield(cfg.privacy))

    # Secondary monitor returned first by Win32, primary second
    monkeypatch.setattr(
        os_sensor_mod.win32api,
        "EnumDisplayMonitors",
        lambda: [
            (1001, 0, (2560, 0, 4480, 1080)),
            (1002, 0, (0, 0, 2560, 1080)),
        ],
    )
    monkeypatch.setattr(
        os_sensor_mod.win32api,
        "GetMonitorInfo",
        lambda hmon: {
            "Monitor": (2560, 0, 4480, 1080) if hmon == 1001 else (0, 0, 2560, 1080),
            "Flags": 0 if hmon == 1001 else 1,
        },
    )

    fake_windows: dict[int, dict[str, Any]] = {
        10: {
            "visible": True,
            "iconic": False,
            "title": "Program Manager",
            "cls": "Progman",
            "rect": (0, 0, 4480, 1080),
            "pid": 100,
            "proc": "explorer.exe",
        },
        11: {
            "visible": True,
            "iconic": False,
            "title": "Toast",
            "cls": "ToastClass",
            "rect": (10, 10, 120, 80),  # too small (< 150x100)
            "pid": 101,
            "proc": "toast.exe",
        },
        12: {
            "visible": True,
            "iconic": True,  # minimized -> excluded
            "title": "Minimized App",
            "cls": "Chrome_WidgetWin_1",
            "rect": (0, 0, 1200, 800),
            "pid": 102,
            "proc": "chrome.exe",
        },
        13: {
            "visible": True,
            "iconic": False,
            "title": "Internet Banking - Nubank",
            "cls": "Chrome_WidgetWin_1",
            "rect": (2660, 100, 3660, 700),
            "pid": 103,
            "proc": "chrome.exe",
        },
        14: {
            "visible": True,
            "iconic": False,
            "title": "Direito Administrativo.md - Obsidian",
            "cls": "Chrome_WidgetWin_1",
            "rect": (0, 0, 1280, 1080),  # 50% of 2560x1080 primary monitor
            "pid": 104,
            "proc": "obsidian.exe",
        },
    }

    def fake_enum_windows(cb: Any, extra: Any) -> None:
        for hwnd in fake_windows:
            if cb(hwnd, extra) is False:
                break

    monkeypatch.setattr(os_sensor_mod.win32gui, "GetForegroundWindow", lambda: 14)
    monkeypatch.setattr(os_sensor_mod.win32gui, "EnumWindows", fake_enum_windows)
    monkeypatch.setattr(
        os_sensor_mod.win32gui,
        "IsWindowVisible",
        lambda h: bool(fake_windows.get(h, {}).get("visible", False)),
    )
    monkeypatch.setattr(
        os_sensor_mod.win32gui,
        "IsIconic",
        lambda h: bool(fake_windows.get(h, {}).get("iconic", False)),
    )
    monkeypatch.setattr(
        os_sensor_mod.win32gui,
        "GetWindowText",
        lambda h: str(fake_windows.get(h, {}).get("title", "")),
    )
    monkeypatch.setattr(
        os_sensor_mod.win32gui,
        "GetClassName",
        lambda h: str(fake_windows.get(h, {}).get("cls", "")),
    )
    monkeypatch.setattr(
        os_sensor_mod.win32gui,
        "GetWindowRect",
        lambda h: fake_windows[h]["rect"],
    )
    monkeypatch.setattr(
        os_sensor_mod.win32process,
        "GetWindowThreadProcessId",
        lambda h: (1, fake_windows[h]["pid"]),
    )
    pid_to_proc = {info["pid"]: info["proc"] for info in fake_windows.values()}
    monkeypatch.setattr(
        os_sensor_mod.psutil,
        "Process",
        lambda pid: SimpleNamespace(name=lambda: pid_to_proc[pid]),
    )
    monkeypatch.setattr(os_sensor_mod.win32api, "GetLastInputInfo", lambda: 900)
    monkeypatch.setattr(os_sensor_mod.win32api, "GetTickCount", lambda: 1400)

    loud_ctl = MagicMock()
    loud_ctl.QueryInterface.return_value.GetPeakValue.return_value = 0.4
    monkeypatch.setattr(
        os_sensor_mod.pycaw.AudioUtilities,
        "GetAllSessions",
        lambda: [
            SimpleNamespace(
                Process=SimpleNamespace(name=lambda: "Spotify.exe"),
                State=1,
                _ctl=loud_ctl,
            )
        ],
    )

    mb = 1024 * 1024
    proc_iter_calls = 0

    def fake_process_iter(_attrs: Any = None) -> list[SimpleNamespace]:
        nonlocal proc_iter_calls
        proc_iter_calls += 1
        return [
            SimpleNamespace(
                info={"pid": 1, "name": "svchost.exe", "memory_info": SimpleNamespace(rss=500 * mb)}
            ),
            SimpleNamespace(
                info={"pid": 2, "name": "dwm.exe", "memory_info": SimpleNamespace(rss=300 * mb)}
            ),
            SimpleNamespace(
                info={"pid": 3, "name": "chrome.exe", "memory_info": SimpleNamespace(rss=150 * mb)}
            ),
            SimpleNamespace(
                info={"pid": 4, "name": "Chrome.exe", "memory_info": SimpleNamespace(rss=250 * mb)}
            ),
            SimpleNamespace(
                info={"pid": 5, "name": "spotify.exe", "memory_info": SimpleNamespace(rss=80 * mb)}
            ),
            SimpleNamespace(
                info={"pid": 6, "name": "obsidian.exe", "memory_info": SimpleNamespace(rss=90 * mb)}
            ),
            SimpleNamespace(
                info={"pid": 7, "name": "anki.exe", "memory_info": SimpleNamespace(rss=40 * mb)}
            ),
            SimpleNamespace(
                info={"pid": 8, "name": "randomtiny.exe", "memory_info": SimpleNamespace(rss=15 * mb)}
            ),
            SimpleNamespace(
                info={"pid": 9, "name": "heavytool.exe", "memory_info": SimpleNamespace(rss=220 * mb)}
            ),
        ]

    monkeypatch.setattr(os_sensor_mod.psutil, "process_iter", fake_process_iter)

    ev = sensor.sample_context()

    assert ev.active_monitor_label == "M1-Principal"
    assert len(ev.visible_windows) == 2

    # Foreground window comes first in Z-order
    fg_win = ev.visible_windows[0]
    assert fg_win.hwnd == 14
    assert fg_win.is_foreground is True
    assert fg_win.is_sensitive is False
    assert fg_win.window_title == "Direito Administrativo.md - Obsidian"
    assert fg_win.monitor_label == "M1-Principal"
    assert fg_win.screen_area_pct == 50.0

    # Non-foreground sensitive window keeps rect & monitor_label, masks title
    sec_win = ev.visible_windows[1]
    assert sec_win.hwnd == 13
    assert sec_win.is_foreground is False
    assert sec_win.is_sensitive is True
    assert sec_win.window_title == "[JANELA PROTEGIDA]"
    assert sec_win.rect == (2660, 100, 3660, 700)
    assert sec_win.monitor_label == "M2-Secundario"

    # Notable running processes: excludes noise + small unrecognized app, aggregates chrome.exe
    proc_map: dict[str, ProcessActivityInfo] = {
        p.process_name: p for p in ev.notable_running_processes
    }
    assert "svchost.exe" not in proc_map
    assert "dwm.exe" not in proc_map
    assert "randomtiny.exe" not in proc_map
    assert set(proc_map) == {"spotify.exe", "obsidian.exe", "chrome.exe", "heavytool.exe", "anki.exe"}

    # Sorted by (has_active_audio, window_count, memory_mb)
    assert ev.notable_running_processes[0].process_name == "spotify.exe"
    assert ev.notable_running_processes[0].has_active_audio is True
    assert ev.notable_running_processes[0].category == "media"
    assert proc_map["chrome.exe"].memory_mb == 400.0
    assert proc_map["chrome.exe"].window_count == 1
    assert proc_map["chrome.exe"].category == "browser"
    assert proc_map["obsidian.exe"].category == "study"
    assert proc_map["anki.exe"].category == "study"
    assert proc_map["heavytool.exe"].category == "app"

    assert ev.usage_pattern.interaction_mode == "active_work"
    assert proc_iter_calls == 1

    # Second call within 3s reuses process cache (no extra process_iter call)
    monkeypatch.setattr(os_sensor_mod.win32gui, "GetForegroundWindow", lambda: 13)
    ev2 = sensor.sample_context()
    assert proc_iter_calls == 1
    assert ev2.active_monitor_label == "M2-Secundario"


def test_os_sensor_enum_windows_and_process_iter_resilience(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _make_config()
    bus = EventBus()
    sensor = WindowsOSSensor(cfg, bus)

    monkeypatch.setattr(os_sensor_mod.win32gui, "GetForegroundWindow", lambda: 0)
    monkeypatch.setattr(
        os_sensor_mod.win32api,
        "EnumDisplayMonitors",
        MagicMock(side_effect=OSError("Display error")),
    )
    monkeypatch.setattr(
        os_sensor_mod.win32gui,
        "EnumWindows",
        MagicMock(side_effect=RuntimeError("EnumWindows denied")),
    )
    monkeypatch.setattr(
        os_sensor_mod.psutil,
        "process_iter",
        MagicMock(side_effect=RuntimeError("psutil failure")),
    )
    monkeypatch.setattr(os_sensor_mod.pycaw.AudioUtilities, "GetAllSessions", list)

    ev = sensor.sample_context()
    assert ev.visible_windows == []
    assert ev.notable_running_processes == []
    assert ev.active_monitor_label == "M1-Principal"


def test_os_sensor_limits_visible_windows_and_notable_processes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _make_config()
    bus = EventBus()
    sensor = WindowsOSSensor(cfg, bus)

    monkeypatch.setattr(
        os_sensor_mod.win32api,
        "EnumDisplayMonitors",
        lambda: [(1, 0, (0, 0, 1920, 1080))],
    )
    monkeypatch.setattr(
        os_sensor_mod.win32api,
        "GetMonitorInfo",
        lambda _h: {"Monitor": (0, 0, 1920, 1080), "Flags": 1},
    )

    # 15 valid windows, with foreground hwnd=215 appearing last in EnumWindows order
    hwnds = list(range(201, 216))
    monkeypatch.setattr(os_sensor_mod.win32gui, "GetForegroundWindow", lambda: 215)
    monkeypatch.setattr(
        os_sensor_mod.win32gui,
        "EnumWindows",
        lambda cb, extra: [cb(h, extra) for h in hwnds],
    )
    monkeypatch.setattr(os_sensor_mod.win32gui, "IsWindowVisible", lambda _h: True)
    monkeypatch.setattr(os_sensor_mod.win32gui, "IsIconic", lambda _h: False)
    monkeypatch.setattr(os_sensor_mod.win32gui, "GetWindowText", lambda h: f"Window {h}")
    monkeypatch.setattr(os_sensor_mod.win32gui, "GetClassName", lambda _h: "AppClass")
    monkeypatch.setattr(os_sensor_mod.win32gui, "GetWindowRect", lambda _h: (0, 0, 800, 600))
    monkeypatch.setattr(
        os_sensor_mod.win32process, "GetWindowThreadProcessId", lambda h: (1, h)
    )
    monkeypatch.setattr(
        os_sensor_mod.psutil,
        "Process",
        lambda pid: SimpleNamespace(name=lambda: f"app_{pid}.exe"),
    )
    monkeypatch.setattr(os_sensor_mod.win32api, "GetLastInputInfo", lambda: 100)
    monkeypatch.setattr(os_sensor_mod.win32api, "GetTickCount", lambda: 200)
    monkeypatch.setattr(os_sensor_mod.pycaw.AudioUtilities, "GetAllSessions", list)

    mb = 1024 * 1024
    monkeypatch.setattr(
        os_sensor_mod.psutil,
        "process_iter",
        lambda _attrs=None: [
            SimpleNamespace(
                info={
                    "pid": 500 + idx,
                    "name": f"heavy_{idx:02d}.exe",
                    "memory_info": SimpleNamespace(rss=(150 + idx * 10) * mb),
                }
            )
            for idx in range(20)
        ],
    )

    ev = sensor.sample_context()
    assert len(ev.visible_windows) == 10
    assert ev.visible_windows[0].hwnd == 215
    assert ev.visible_windows[0].is_foreground is True
    assert len(ev.notable_running_processes) == 12
    sort_keys = [
        (int(p.has_active_audio), p.window_count, p.memory_mb)
        for p in ev.notable_running_processes
    ]
    assert sort_keys == sorted(sort_keys, reverse=True)


