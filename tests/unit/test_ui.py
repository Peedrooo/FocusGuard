"""Unit tests for FocusGuard UI subsystem (FastAPI, WebSocket, Tray, Interventions)."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from focus_guard.core.config import AppConfig, save_api_key_to_env, save_user_config
from focus_guard.core.event_bus import EventBus
from focus_guard.core.events import (
    ChatMessage,
    CoachPersona,
    FocusState,
    InterventionCommand,
    InterventionLevel,
    OSContextEvent,
)
from focus_guard.core.ports import InterventionDispatcherPort
from focus_guard.ui import (
    SystemTrayController,
    WindowsInterventionDispatcher,
    create_api_app,
    create_tray_icon_image,
    get_state_color,
)
from focus_guard.ui.tray import _parse_win32_hotkey
from tests.fakes import (
    FakeCoachAgent,
    FakeFocusEngine,
    FakeInterventionDispatcher,
    FakeMicRecorder,
    FakeSemanticJudge,
    FakeTTS,
    FakeVisionSensor,
)


def _build_test_stack(
    *,
    mic_available: bool = True,
    configured: bool = False,
    vision_sensor: FakeVisionSensor | None = None,
    tts: FakeTTS | None = None,
    mic: FakeMicRecorder | None = None,
    coach: FakeCoachAgent | None = None,
) -> tuple[
    TestClient,
    FakeFocusEngine,
    FakeCoachAgent,
    FakeSemanticJudge,
    FakeTTS,
    FakeMicRecorder,
    EventBus,
    AppConfig,
    FakeInterventionDispatcher,
]:
    config = AppConfig(gemini_api_key="initial-key" if configured else "")
    engine = FakeFocusEngine(config=config)
    coach_obj = coach or FakeCoachAgent(configured=configured)
    judge = FakeSemanticJudge(configured=configured)
    tts_obj = tts or FakeTTS()
    mic_obj = mic or FakeMicRecorder(available=mic_available)
    bus = EventBus()
    dispatcher = FakeInterventionDispatcher()
    app = create_api_app(
        engine=engine,
        coach=coach_obj,
        judge=judge,
        tts=tts_obj,
        mic=mic_obj,
        event_bus=bus,
        config=config,
        dispatcher=dispatcher,
        vision_sensor=vision_sensor,
    )
    client = TestClient(app)
    return client, engine, coach_obj, judge, tts_obj, mic_obj, bus, config, dispatcher


def test_dashboard_html_and_static_assets() -> None:
    client, *_ = _build_test_stack()
    with client:
        resp = client.get("/")
        assert resp.status_code == 200
        assert "FocusGuard" in resp.text
        assert "/static/app.js" in resp.text

        css_resp = client.get("/static/styles.css")
        assert css_resp.status_code == 200

        js_resp = client.get("/static/app.js")
        assert js_resp.status_code == 200
        assert "connectWebSocket" in js_resp.text


def test_status_and_session_lifecycle_endpoints() -> None:
    client, engine, _coach, _judge, _tts, _mic, _bus, _cfg, dispatcher = _build_test_stack()
    with client:
        status_resp = client.get("/api/status")
        assert status_resp.status_code == 200
        body = status_resp.json()
        assert "status" in body and "chat_history" in body and "notes" in body
        assert body["status"]["session"]["current_state"] == "PAUSED"

        start_empty = client.post("/api/session/start")
        assert start_empty.status_code == 200
        assert start_empty.json()["session"]["current_state"] == "FOCUSED"

        start_custom = client.post(
            "/api/session/start",
            json={"study_goal": "Receita Federal", "current_subject": "Contabilidade"},
        )
        assert start_custom.status_code == 200
        assert start_custom.json()["session"]["study_goal"] == "Receita Federal"
        assert start_custom.json()["session"]["current_subject"] == "Contabilidade"

        break_resp = client.post("/api/session/break", json={"minutes": 5})
        assert break_resp.status_code == 200
        assert break_resp.json()["session"]["current_state"] == "BREAK_MODE"

        end_break = client.post("/api/session/end-break")
        assert end_break.status_code == 200
        assert end_break.json()["session"]["current_state"] == "FOCUSED"

        engine.session.current_state = FocusState.INTERVENING
        ack_resp = client.post("/api/session/ack-focus")
        assert ack_resp.status_code == 200
        assert ack_resp.json()["status"]["session"]["current_state"] == "FOCUSED"
        assert dispatcher.overlay_dismissed >= 1

        pause_resp = client.post("/api/session/pause")
        assert pause_resp.status_code == 200
        assert pause_resp.json()["session"]["current_state"] == "PAUSED"


def test_privacy_goal_allowance_and_notes_endpoints() -> None:
    client, *_ = _build_test_stack()
    with client:
        priv_on = client.post("/api/privacy/toggle")
        assert priv_on.status_code == 200
        assert priv_on.json()["privacy_mode"] is True

        priv_off = client.post("/api/privacy/toggle", json={"enabled": False})
        assert priv_off.status_code == 200
        assert priv_off.json()["privacy_mode"] is False

        goal_resp = client.post(
            "/api/goal",
            json={"study_goal": "PF Agente", "current_subject": "Dir. Constitucional"},
        )
        assert goal_resp.status_code == 200
        assert goal_resp.json()["session"]["study_goal"] == "PF Agente"
        assert goal_resp.json()["session"]["current_subject"] == "Dir. Constitucional"

        allow_resp = client.post(
            "/api/allowance",
            json={"pattern": "youtube.com", "minutes": 10, "reason": "Aula de revisão"},
        )
        assert allow_resp.status_code == 200
        assert allow_resp.json()["allowance"]["pattern"] == "youtube.com"
        assert len(allow_resp.json()["status"]["active_allowances"]) == 1

        note_post = client.post(
            "/api/notes",
            json={"note": "Revisar remédios constitucionais", "subject": "Dir. Constitucional"},
        )
        assert note_post.status_code == 200
        assert note_post.json()["note"]["note"] == "Revisar remédios constitucionais"

        note_get = client.get("/api/notes")
        assert note_get.status_code == 200
        assert len(note_get.json()["notes"]) == 1


def test_chat_text_audio_and_ptt_endpoints() -> None:
    client, *_ = _build_test_stack()
    with client:
        text_resp = client.post(
            "/api/chat/text",
            json={"text": "Explica controle de constitucionalidade", "speak": False},
        )
        assert text_resp.status_code == 200
        assert text_resp.json()["reply"]["role"] == "coach"

        audio_b64 = base64.b64encode(b"RIFFfake-audio-bytes").decode("ascii")
        audio_json_resp = client.post(
            "/api/chat/audio",
            json={"audio_base64": audio_b64, "mime_type": "audio/wav"},
        )
        assert audio_json_resp.status_code == 200
        assert audio_json_resp.json()["transcript"]
        assert audio_json_resp.json()["reply"]["role"] == "coach"

        audio_mp_resp = client.post(
            "/api/chat/audio",
            files={"file": ("voice.webm", b"fake-webm-audio", "audio/webm")},
        )
        assert audio_mp_resp.status_code == 200
        assert audio_mp_resp.json()["reply"]["role"] == "coach"

        bad_audio = client.post("/api/chat/audio", json={"audio_base64": "!!!not-base64!!!"})
        assert bad_audio.status_code == 400

        ptt_resp = client.post("/api/chat/ptt")
        assert ptt_resp.status_code == 200
        ptt_data = ptt_resp.json()
        assert ptt_data["transcript"]
        assert ptt_data["reply"]["role"] == "coach"
        assert "audio_base64" in ptt_data

    # Unavailable mic returns clean 503
    client_no_mic, *_ = _build_test_stack(mic_available=False)
    with client_no_mic:
        no_mic_resp = client_no_mic.post("/api/chat/ptt")
        assert no_mic_resp.status_code == 503


def test_config_api_key_and_settings_endpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import focus_guard.ui.api_server as api_server_mod

    env_file = tmp_path / ".env"
    user_yaml = tmp_path / "user_config.yaml"
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(
        api_server_mod,
        "save_api_key_to_env",
        lambda key: save_api_key_to_env(key, env_path=env_file),
    )
    monkeypatch.setattr(
        api_server_mod,
        "save_user_config",
        lambda cfg: save_user_config(cfg, path=user_yaml),
    )

    client, engine, coach, judge, _tts, _mic, _bus, config, _disp = _build_test_stack()
    with client:
        key_resp = client.post("/api/config/api-key", json={"api_key": "AIza-fake-test-key"})
        assert key_resp.status_code == 200
        assert key_resp.json()["api_key_configured"] is True
        assert config.gemini_api_key == "AIza-fake-test-key"
        assert judge.is_configured() is True
        assert coach.is_configured() is True
        assert "GEMINI_API_KEY=AIza-fake-test-key" in env_file.read_text(encoding="utf-8")

        settings_resp = client.post(
            "/api/config/settings",
            json={
                "coach_persona": "sargento",
                "max_daily_usd": 0.5,
                "max_monthly_usd": 10.0,
                "youtube_music_grace_sec": 60,
            },
        )
        assert settings_resp.status_code == 200
        assert config.coach_persona == CoachPersona.SARGENTO
        assert config.ai.max_daily_usd == 0.5
        assert config.ai.max_monthly_usd == 10.0
        assert config.behavior.youtube_music_grace_sec == 60
        assert engine.budget.get_status().max_monthly_usd == 10.0
        assert user_yaml.exists()


def test_test_nudge_and_websocket_broadcast() -> None:
    client, engine, _coach, _judge, _tts, _mic, _bus, _cfg, dispatcher = _build_test_stack()
    engine.latest_os = OSContextEvent(
        process_name="chrome.exe",
        window_title="YouTube - Shorts",
        browser_url="https://youtube.com/shorts/abc",
    )

    with client, client.websocket_connect("/ws") as ws:
        initial = ws.receive_json()
        assert initial["type"] == "SystemStatusSnapshot"
        assert initial["data"]["session"]["current_state"] == "PAUSED"

        ws.send_text("ping")
        pong = ws.receive_json()
        assert pong["type"] == "pong"

        nudge_resp = client.post("/api/test-nudge", json={"level": 3})
        assert nudge_resp.status_code == 200
        cmd_json = nudge_resp.json()["command"]
        assert cmd_json["level"] == 3
        assert len(dispatcher.dispatched) == 1
        assert dispatcher.dispatched[0].level == InterventionLevel.FIRM_OVERLAY

        ws_event = ws.receive_json()
        assert ws_event["type"] == "InterventionCommand"
        assert ws_event["data"]["message"] == cmd_json["message"]


async def test_windows_intervention_dispatcher_levels() -> None:
    tts = FakeTTS()
    ack_called = False

    def _on_ack() -> None:
        nonlocal ack_called
        ack_called = True

    dispatcher = WindowsInterventionDispatcher(
        tts=tts,
        on_ack_focus=_on_ack,
        enable_ui=False,
    )
    assert isinstance(dispatcher, InterventionDispatcherPort)

    # Level 0 (NONE): does nothing
    await dispatcher.dispatch(
        InterventionCommand(
            level=InterventionLevel.NONE,
            title="Nenhum",
            message="Sem alerta",
        )
    )
    assert tts.spoken == []

    # Level 1 (SOFT, speak_aloud=False): toast/beep only (no TTS)
    await dispatcher.dispatch(
        InterventionCommand(
            level=InterventionLevel.SOFT,
            title="Aviso Leve",
            message="Ei, olha o foco!",
            speak_aloud=False,
        )
    )
    assert tts.spoken == []

    # Level 1 (SOFT, speak_aloud=True): toast/beep AND speaks via TTS!
    await dispatcher.dispatch(
        InterventionCommand(
            level=InterventionLevel.SOFT,
            title="Aviso Falado",
            message="Larga o celular agora, Pedro!",
            speak_aloud=True,
        )
    )
    assert tts.spoken == ["Larga o celular agora, Pedro!"]

    # Level 2 (VOICE): speaks message via TTS
    await dispatcher.dispatch(
        InterventionCommand(
            level=InterventionLevel.VOICE,
            title="Puxão de Orelha",
            message="Fecha esse vídeo agora, Pedro.",
            speak_aloud=True,
        )
    )
    assert tts.spoken == [
        "Larga o celular agora, Pedro!",
        "Fecha esse vídeo agora, Pedro.",
    ]

    # Level 3 (FIRM_OVERLAY): speaks message via TTS (and shows banner if enable_ui=True)
    await dispatcher.dispatch(
        InterventionCommand(
            level=InterventionLevel.FIRM_OVERLAY,
            title="Alerta Máximo",
            message="Volte imediatamente para Direito Administrativo.",
            speak_aloud=True,
        )
    )
    assert tts.spoken == [
        "Larga o celular agora, Pedro!",
        "Fecha esse vídeo agora, Pedro.",
        "Volte imediatamente para Direito Administrativo.",
    ]

    dispatcher.dismiss_overlay()

    # Resilient when TTS raises an exception
    class _BrokenTTS(FakeTTS):
        async def speak(self, text: str) -> bool:
            raise RuntimeError("TTS offline")

    safe_dispatcher = WindowsInterventionDispatcher(tts=_BrokenTTS(), enable_ui=False)
    await safe_dispatcher.dispatch(
        InterventionCommand(
            level=InterventionLevel.VOICE,
            title="Teste",
            message="Não deve quebrar",
        )
    )


@pytest.mark.asyncio
async def test_audio_only_dispatcher_skips_windows_toast_and_overlay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tts = FakeTTS()
    dispatcher = WindowsInterventionDispatcher(
        tts=tts,
        enable_ui=True,
        show_windows_toast=False,
        show_overlay=False,
    )
    toast_calls: list[str] = []
    overlay_calls: list[str] = []
    monkeypatch.setattr(dispatcher, "_show_toast_async", lambda t, m: toast_calls.append(t))
    monkeypatch.setattr(dispatcher, "_play_soft_beep_async", lambda: toast_calls.append("beep"))
    monkeypatch.setattr(dispatcher, "_show_or_update_overlay", lambda t, m: overlay_calls.append(t))

    await dispatcher.dispatch(
        InterventionCommand(
            level=InterventionLevel.SOFT,
            title="Alerta",
            message="Para de ver drone no YouTube e volta pra ITIL.",
            speak_aloud=True,
        )
    )
    await dispatcher.dispatch(
        InterventionCommand(
            level=InterventionLevel.FIRM_OVERLAY,
            title="Alerta Máximo",
            message="Fecha o YouTube agora, Pedro.",
            speak_aloud=True,
        )
    )
    assert toast_calls == []
    assert overlay_calls == []
    assert tts.spoken == [
        "Para de ver drone no YouTube e volta pra ITIL.",
        "Fecha o YouTube agora, Pedro.",
    ]


def test_system_tray_controller_icons_and_lifecycle() -> None:
    toggled: list[str] = []
    tray = SystemTrayController(
        on_toggle_session=lambda: toggled.append("session"),
        on_toggle_privacy=lambda: toggled.append("privacy"),
        on_talk_coach=lambda: toggled.append("talk"),
        enable_tray=False,
        enable_hotkeys=False,
    )

    assert get_state_color(FocusState.FOCUSED) == "#10b981"
    assert get_state_color(FocusState.GRACE_WINDOW) == "#f59e0b"
    assert get_state_color(FocusState.BREAK_MODE) == "#f59e0b"
    assert get_state_color(FocusState.INTERVENING) == "#ef4444"
    assert get_state_color(FocusState.DRIFT_SUSPECTED) == "#ef4444"
    assert get_state_color(FocusState.PAUSED) == "#64748b"
    assert get_state_color(FocusState.PRIVACY_MODE) == "#64748b"

    for st in (FocusState.FOCUSED, FocusState.BREAK_MODE, FocusState.INTERVENING, FocusState.PAUSED):
        img = create_tray_icon_image(st, size=32)
        assert img.size == (32, 32)
        assert img.mode == "RGBA"
        tray.update_state(st)

    tray.start()
    tray._handle_privacy_hotkey()
    tray._handle_talk_hotkey()
    assert toggled == ["privacy", "talk"]
    tray.stop()


def test_api_status_empty_chat_and_js_syntax_markers() -> None:
    client, *_ = _build_test_stack()
    with client:
        resp = client.get("/api/status")
        assert resp.status_code == 200
        data: dict[str, Any] = resp.json()
        assert data["chat_history"] == []
        assert data["notes"] == []


def test_audio_data_uri_ptt_empty_and_ws_resilience() -> None:
    client, _eng, _coach, _judge, _tts, mic, _bus, _cfg, _disp = _build_test_stack()
    mic.audio_bytes = b""

    with client:
        b64 = base64.b64encode(b"RIFFwave").decode("ascii")
        data_uri_resp = client.post(
            "/api/chat/audio",
            json={"audio_base64": f"data:audio/wav;base64,{b64}"},
        )
        assert data_uri_resp.status_code == 200
        assert data_uri_resp.json()["reply"]["role"] == "coach"

        empty_ptt = client.post("/api/chat/ptt")
        assert empty_ptt.status_code == 422

        assert client.post("/api/tts/stop").status_code == 200
        assert client.post("/api/ptt/trigger").status_code == 501

        with client.websocket_connect("/ws") as ws1:
            assert ws1.receive_json()["type"] == "SystemStatusSnapshot"
            with client.websocket_connect("/ws") as ws2:
                assert ws2.receive_json()["type"] == "SystemStatusSnapshot"
            # ws2 is now closed; broadcasting via test-nudge must still deliver to ws1 cleanly
            nudge_resp = client.post("/api/test-nudge", json={"level": 1})
            assert nudge_resp.status_code == 200
            ev = ws1.receive_json()
            assert ev["type"] == "InterventionCommand"
            assert ev["data"]["level"] == 1


def test_vision_preview_and_calibrate_endpoints() -> None:
    # Missing vision sensor -> 204 on preview, 503 on calibrate
    client_no_cam, *_ = _build_test_stack(vision_sensor=None)
    with client_no_cam:
        assert client_no_cam.get("/api/vision/preview.jpg").status_code == 204
        assert client_no_cam.post("/api/vision/calibrate", json={"mode": "primary_screen"}).status_code == 503

    # Active vision sensor -> 200 JPEG with Cache-Control: no-store
    vision = FakeVisionSensor(available=True)
    client, *_ = _build_test_stack(vision_sensor=vision)
    with client:
        prev_resp = client.get("/api/vision/preview.jpg?annotate=true")
        assert prev_resp.status_code == 200
        assert prev_resp.headers["content-type"] == "image/jpeg"
        assert prev_resp.headers["cache-control"] == "no-store"
        assert prev_resp.content == vision.latest_frame_jpeg()

        # Privacy mode blocks preview (204)
        client.post("/api/privacy/toggle", json={"enabled": True})
        assert client.get("/api/vision/preview.jpg").status_code == 204
        client.post("/api/privacy/toggle", json={"enabled": False})

        cal_resp = client.post(
            "/api/vision/calibrate",
            json={"mode": "secondary_screen", "duration_sec": 1.0},
        )
        assert cal_resp.status_code == 200
        cal_data = cal_resp.json()
        assert cal_data["ok"] is True
        assert cal_data["calibration"]["mode"] == "secondary_screen"
        assert vision.calibrations == ["secondary_screen"]


def test_audio_devices_test_ptt_stop_mime_and_extended_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import focus_guard.ui.api_server as api_server_mod

    user_yaml = tmp_path / "user_config.yaml"
    monkeypatch.setattr(
        api_server_mod,
        "save_user_config",
        lambda cfg: save_user_config(cfg, path=user_yaml),
    )

    class _DeviceTTS(FakeTTS):
        def __init__(self) -> None:
            super().__init__()
            self.selected_output: str | None = None

        def list_output_devices(self) -> list[str]:
            return ["Fone USB", "Alto-falantes"]

        def set_output_device(self, name: str | None) -> None:
            self.selected_output = name

    class _DeviceMic(FakeMicRecorder):
        def __init__(self) -> None:
            super().__init__(available=True)
            self.selected_input: str | None = None
            self.stopped = False

        def list_input_devices(self) -> list[str]:
            return ["Microfone Headset", "Webcam Mic"]

        def set_input_device(self, name: str | None) -> None:
            self.selected_input = name

        def stop_recording(self) -> None:
            self.stopped = True

    class _MimeCoach(FakeCoachAgent):
        def __init__(self) -> None:
            super().__init__(configured=True)
            self.last_mime: str | None = None

        async def reply_audio(
            self, audio_bytes: bytes, mime_type: str = "audio/wav", speak: bool = True
        ) -> tuple[str, ChatMessage]:
            self.last_mime = mime_type
            return await super().reply_audio(audio_bytes, mime_type=mime_type, speak=speak)

    tts = _DeviceTTS()
    mic = _DeviceMic()
    coach = _MimeCoach()
    client, _eng, _coach, _judge, _tts, _mic, _bus, config, _disp = _build_test_stack(
        tts=tts, mic=mic, coach=coach
    )

    with client:
        dev_get = client.get("/api/audio/devices")
        assert dev_get.status_code == 200
        assert dev_get.json()["output_devices"] == ["Fone USB", "Alto-falantes"]
        assert dev_get.json()["input_devices"] == ["Microfone Headset", "Webcam Mic"]

        dev_post = client.post(
            "/api/audio/devices",
            json={"output_device_name": "Fone USB", "input_device_name": "Microfone Headset"},
        )
        assert dev_post.status_code == 200
        assert config.voice.output_device_name == "Fone USB"
        assert config.voice.input_device_name == "Microfone Headset"
        assert tts.selected_output == "Fone USB"
        assert mic.selected_input == "Microfone Headset"
        assert user_yaml.exists()

        test_audio_resp = client.post("/api/audio/test", json={"text": "Alô fone!"})
        assert test_audio_resp.status_code == 200
        assert test_audio_resp.json()["ok"] is True
        assert "Alô fone!" in tts.spoken

        ptt_stop_resp = client.post("/api/chat/ptt/stop")
        assert ptt_stop_resp.status_code == 200
        assert ptt_stop_resp.json()["ok"] is True
        assert mic.stopped is True

        # MIME normalization: "audio/webm;codecs=opus" -> "audio/webm"
        mime_resp = client.post(
            "/api/chat/audio",
            files={"file": ("clip.webm", b"fake-opus-bytes", "audio/webm;codecs=opus")},
        )
        assert mime_resp.status_code == 200
        assert coach.last_mime == "audio/webm"

        # Extended settings: speak_on_all_interventions, phone_distraction_sec, capture_mode
        ext_resp = client.post(
            "/api/config/settings",
            json={
                "speak_on_all_interventions": False,
                "phone_distraction_sec": 20,
                "capture_mode": "active_monitor",
            },
        )
        assert ext_resp.status_code == 200
        assert config.behavior.speak_on_all_interventions is False
        assert config.behavior.phone_distraction_sec == 20
        assert config.ai.capture_mode == "active_monitor"

        get_set = client.get("/api/config/settings")
        assert get_set.status_code == 200
        assert get_set.json()["settings"]["capture_mode"] == "active_monitor"


def test_win32_hotkey_parser_and_fallback_to_pynput(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _parse_win32_hotkey("<ctrl>+<shift>+p") == (0x0002 | 0x0004, ord("P"))
    assert _parse_win32_hotkey("<ctrl>+<shift>+<space>") == (0x0002 | 0x0004, 0x20)
    assert _parse_win32_hotkey("") is None
    assert _parse_win32_hotkey("<ctrl>+") is None

    class _FailUser32:
        def PeekMessageW(self, *_args: Any) -> int:
            return 0

        def RegisterHotKey(self, *_args: Any) -> int:
            return 0  # Force fallback to pynput

        def UnregisterHotKey(self, *_args: Any) -> int:
            return 1

        def PostThreadMessageW(self, *_args: Any) -> int:
            return 1

    class _FakeKernel32:
        def GetCurrentThreadId(self) -> int:
            return 12345

    tray = SystemTrayController(enable_tray=False, enable_hotkeys=True)
    fallback_called = False
    monkeypatch.setattr(tray, "_get_win32_apis", lambda: (_FailUser32(), _FakeKernel32()))

    def _fake_pynput() -> None:
        nonlocal fallback_called
        fallback_called = True

    monkeypatch.setattr(tray, "_start_pynput_hotkeys", _fake_pynput)
    tray.start()
    assert fallback_called is True
    tray.stop()


def test_win32_hotkey_pump_dispatches_wm_hotkey_and_unregisters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ctypes

    calls: list[str] = []
    registered: list[tuple[int, int, int]] = []
    unregistered: list[int] = []
    quit_posted: list[int] = []

    class _HappyUser32:
        def __init__(self) -> None:
            self._step = 0

        def PeekMessageW(self, *_args: Any) -> int:
            return 0

        def RegisterHotKey(self, _hwnd: Any, hk_id: int, mods: int, vk: int) -> int:
            registered.append((hk_id, mods, vk))
            return 1

        def GetMessageW(self, msg_ptr: Any, *_args: Any) -> int:
            if self._step == 0:
                self._step += 1
                msg = ctypes.cast(msg_ptr, ctypes.POINTER(ctypes.wintypes.MSG)).contents  # type: ignore[attr-defined]
                msg.message = 0x0312  # WM_HOTKEY
                msg.wParam = 1
                return 1
            if self._step == 1:
                self._step += 1
                msg = ctypes.cast(msg_ptr, ctypes.POINTER(ctypes.wintypes.MSG)).contents  # type: ignore[attr-defined]
                msg.message = 0x0312  # WM_HOTKEY
                msg.wParam = 2
                return 1
            return 0  # WM_QUIT

        def UnregisterHotKey(self, _hwnd: Any, hk_id: int) -> int:
            unregistered.append(hk_id)
            return 1

        def PostThreadMessageW(self, tid: int, msg_id: int, *_args: Any) -> int:
            quit_posted.append(msg_id)
            return 1

    class _FakeKernel32:
        def GetCurrentThreadId(self) -> int:
            return 9999

    tray = SystemTrayController(
        on_toggle_privacy=lambda: calls.append("privacy"),
        on_talk_coach=lambda: calls.append("talk"),
        enable_tray=False,
        enable_hotkeys=True,
    )
    monkeypatch.setattr(tray, "_get_win32_apis", lambda: (_HappyUser32(), _FakeKernel32()))
    tray.start()
    if tray._win32_thread is not None:
        tray._win32_thread.join(timeout=1.5)
    tray.stop()

    assert len(registered) == 2
    assert calls == ["privacy", "talk"]
    assert set(unregistered) == {1, 2}


def test_dashboard_v2_ui_markers() -> None:
    client, *_ = _build_test_stack()
    with client:
        html = client.get("/").text
        assert "webcam-preview-img" in html
        assert "visible-windows-list" in html
        assert "setting-speak-all" in html
        assert "audio-output-select" in html

        js = client.get("/static/app.js").text
        assert "/api/vision/preview.jpg" in js
        assert "/api/vision/calibrate" in js
        assert "/api/audio/devices" in js
        assert "MicStateEvent" in js


def test_dashboard_v3_layout_swap_and_camera_unseen_setting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import focus_guard.ui.api_server as api_server_mod

    user_yaml = tmp_path / "user_config.yaml"
    monkeypatch.setattr(
        api_server_mod,
        "save_user_config",
        lambda cfg: save_user_config(cfg, path=user_yaml),
    )

    client, _eng, _coach, _judge, _tts, _mic, _bus, config, _disp = _build_test_stack()
    with client:
        html = client.get("/").text
        col_right_idx = html.index('class="col-right"')
        webcam_idx = html.index('id="webcam-card"')
        bottom_chat_idx = html.index('id="chat-bottom-card"')
        assert col_right_idx < webcam_idx < bottom_chat_idx
        assert 'id="btn-toggle-chat-panel"' in html
        assert 'id="vision-unseen-timer-badge"' in html
        assert 'id="setting-camera-unseen-sec"' in html

        resp = client.post(
            "/api/config/settings",
            json={
                "camera_not_seeing_user_sec": 15,
                "passive_freeze_alert_sec": 180,
            },
        )
        assert resp.status_code == 200
        assert config.behavior.camera_not_seeing_user_sec == 15
        assert config.behavior.passive_freeze_alert_sec == 180
        payload = resp.json()["settings"]
        assert payload["camera_not_seeing_user_sec"] == 15
        assert payload["passive_freeze_alert_sec"] == 180
        assert user_yaml.exists()


def test_subject_stats_endpoint_and_dashboard_card() -> None:
    client, engine, *_ = _build_test_stack()
    engine.session.current_subject = "ITIL v4"
    engine.session.focus_seconds = 320.0

    with client:
        html = client.get("/").text
        assert 'id="subject-stats-card"' in html
        assert 'id="subject-stats-list"' in html
        assert 'data-subject-period="today"' in html

        js = client.get("/static/app.js").text
        assert "/api/stats/subjects" in js
        assert "renderSubjectStats" in js

        status_resp = client.get("/api/status")
        assert status_resp.status_code == 200
        assert isinstance(status_resp.json().get("subject_stats"), list)
        assert status_resp.json()["subject_stats"][0]["subject"] == "ITIL v4"

        stats_resp = client.get("/api/stats/subjects")
        assert stats_resp.status_code == 200
        body = stats_resp.json()
        assert body["current_subject"] == "ITIL v4"
        assert body["total_focus_seconds"] == pytest.approx(320.0)
        assert body["subjects"][0]["is_current"] is True

        goal_resp = client.post(
            "/api/goal",
            json={"study_goal": "Concurso", "current_subject": "Português"},
        )
        assert goal_resp.status_code == 200
        assert goal_resp.json()["subject_stats"][0]["subject"] == "Português"

