"""Unit tests for FocusGuard AI subsystem (GeminiClient, Judge, Coach, Voice Pipeline)."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import soundfile as sf
from google.genai import types

from focus_guard.ai import (
    BudgetExceededError,
    EdgeAndOfflineTTS,
    FirmCoachAgent,
    GeminiClient,
    GeminiSemanticJudge,
    SoundDeviceMicRecorder,
)
from focus_guard.core import (
    ActivityVerdict,
    AIConfig,
    AppConfig,
    BudgetGuard,
    CoachPersona,
    EventBus,
    FocusState,
    HintsConfig,
    InterventionLevel,
    MicStateEvent,
    OSContextEvent,
    ProcessActivityInfo,
    ScreenSnapshot,
    UsagePatternMetrics,
    VisibleWindowInfo,
    VisionSignalEvent,
)
from focus_guard.core.ports import (
    CoachAgentPort,
    MicRecorderPort,
    SemanticJudgePort,
    TTSPort,
)
from tests.fakes import FakeFocusEngine, FakeStorage, FakeTTS


def _make_config(api_key: str = "test-gemini-key", **ai_overrides: Any) -> AppConfig:
    ai_defaults: dict[str, Any] = {
        "primary_model": "gemini-3.5-flash-lite",
        "fallback_models": ["gemini-3.1-flash-lite", "gemini-2.5-flash-lite"],
    }
    ai_defaults.update(ai_overrides)
    ai_cfg = AIConfig(**ai_defaults)
    return AppConfig(
        user_name="Pedro",
        study_goal="Concurso TRT Analista",
        current_subject="Direito Constitucional",
        coach_persona=CoachPersona.FIRME,
        gemini_api_key=api_key,
        ai=ai_cfg,
        hints=HintsConfig(
            likely_study_keywords=["tecconcursos", "questoes", "direito"],
            context_dependent_apps=["obsidian", "antigravity", "youtube", "chrome"],
            high_suspicion_patterns=["youtube.com/shorts", "instagram.com", "globoesporte"],
        ),
    )


def _fake_response(
    text: str = "",
    *,
    tokens_in: int | None = 120,
    tokens_out: int | None = 45,
    function_calls: list[Any] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        usage_metadata=SimpleNamespace(
            prompt_token_count=tokens_in,
            candidates_token_count=tokens_out,
            cached_content_token_count=0,
        ),
        function_calls=function_calls,
        candidates=[],
    )


# ----------------------------------------------------------------------
# Protocol conformance
# ----------------------------------------------------------------------


def test_ai_implementations_satisfy_protocols() -> None:
    cfg = _make_config()
    budget = BudgetGuard(cfg.ai)
    client = GeminiClient(cfg, budget)
    judge = GeminiSemanticJudge(client, cfg)
    engine = FakeFocusEngine(cfg)
    tts = EdgeAndOfflineTTS(cfg.voice)
    mic = SoundDeviceMicRecorder(cfg.voice)
    coach = FirmCoachAgent(client, engine, tts, cfg)

    assert isinstance(judge, SemanticJudgePort)
    assert isinstance(coach, CoachAgentPort)
    assert isinstance(tts, TTSPort)
    assert isinstance(mic, MicRecorderPort)


# ----------------------------------------------------------------------
# GeminiClient tests
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gemini_client_fallback_chain_and_storage() -> None:
    cfg = _make_config(
        fallback_models=["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-2.5-flash-lite"]
    )
    budget = BudgetGuard(cfg.ai)
    storage = FakeStorage()
    attempted_models: list[str] = []
    captured_configs: list[types.GenerateContentConfig] = []

    def raw_generate(*, model: str, contents: Any, config: types.GenerateContentConfig) -> Any:
        attempted_models.append(model)
        captured_configs.append(config)
        if model == "gemini-3.5-flash-lite":
            raise RuntimeError("503 Model overloaded")
        return _fake_response('{"ok": true}', tokens_in=300, tokens_out=100)

    client = GeminiClient(cfg, budget, storage=storage, _raw_generate=raw_generate)
    assert client.is_configured() is True

    resp, model_used, tok_in, tok_out, cost = await client.generate_content(
        contents="Olá",
        system_instruction="Seja direto",
        response_mime_type="application/json",
        temperature=0.2,
    )

    # Deduplicated fallback sequence: primary tried once, then second fallback
    assert attempted_models == ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
    assert model_used == "gemini-3.1-flash-lite"
    assert tok_in == 300
    assert tok_out == 100
    assert cost > 0.0
    assert resp.text == '{"ok": true}'
    assert captured_configs[0].system_instruction == "Seja direto"
    assert captured_configs[0].response_mime_type == "application/json"
    assert captured_configs[0].temperature == 0.2
    assert len(storage.usages) == 1
    assert storage.usages[0][0] == "gemini-3.1-flash-lite"


@pytest.mark.asyncio
async def test_gemini_client_injected_sdk_client_and_none_usage() -> None:
    cfg = _make_config(api_key="")
    budget = BudgetGuard(cfg.ai)

    class FakeModels:
        def generate_content(self, *, model: str, contents: Any, config: Any) -> Any:
            return _fake_response("ok", tokens_in=None, tokens_out=None)

    fake_sdk_client = SimpleNamespace(models=FakeModels())
    client = GeminiClient(cfg, budget, client=fake_sdk_client)
    assert client.is_configured() is True

    _, model_used, tok_in, tok_out, cost = await client.generate_content("ping")
    assert model_used == "gemini-3.5-flash-lite"
    assert tok_in == 0
    assert tok_out == 0
    assert cost == 0.0


@pytest.mark.asyncio
async def test_gemini_client_budget_throttling_and_reconfigure() -> None:
    cfg = _make_config(api_key="", max_daily_usd=0.005, max_monthly_usd=1.0)
    budget = BudgetGuard(cfg.ai)
    called = 0

    def raw_generate(**kwargs: Any) -> Any:
        nonlocal called
        called += 1
        return _fake_response("ok")

    client = GeminiClient(cfg, budget)
    assert client.is_configured() is False

    client.reconfigure_api_key("new-valid-key")
    assert client.is_configured() is True
    client._raw_generate = raw_generate

    budget.record_usage("gemini-3.5-flash-lite", tokens_in=100_000, tokens_out=5_000)
    assert budget.can_spend() is False

    with pytest.raises(BudgetExceededError):
        await client.generate_content("teste")
    assert called == 0

    client.reconfigure_api_key("")
    assert client.is_configured() is False


# ----------------------------------------------------------------------
# GeminiSemanticJudge tests
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_semantic_judge_privacy_shield_bypasses_gemini() -> None:
    cfg = _make_config()
    budget = BudgetGuard(cfg.ai)
    calls = 0

    def raw_generate(**kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return _fake_response("{}")

    client = GeminiClient(cfg, budget, _raw_generate=raw_generate)
    judge = GeminiSemanticJudge(client, cfg)

    sensitive_os = OSContextEvent(
        process_name="msedge.exe",
        window_title="Itaú Internet Banking",
        is_sensitive_window=True,
        sensitive_reason="keyword:itau",
    )
    verdict = await judge.evaluate(
        os_context=sensitive_os,
        vision_signal=None,
        study_goal=cfg.study_goal,
        current_subject=cfg.current_subject,
        screenshot=ScreenSnapshot(
            image_bytes_jpeg=b"\xff\xd8\xff\xd9",
            perceptual_hash="0" * 16,
            width=768,
            height=432,
        ),
    )

    assert calls == 0
    assert verdict.verdict == ActivityVerdict.UNCERTAIN
    assert verdict.confidence == 1.0
    assert verdict.category == "privacy_shield"
    assert verdict.used_vision is False
    assert "sensível" in verdict.reasoning.lower()


@pytest.mark.asyncio
async def test_semantic_judge_parses_markdown_json_and_attaches_screenshot() -> None:
    cfg = _make_config()
    budget = BudgetGuard(cfg.ai)
    captured_contents: list[Any] = []
    captured_system: list[str] = []

    def raw_generate(*, model: str, contents: Any, config: types.GenerateContentConfig) -> Any:
        captured_contents.append(contents)
        captured_system.append(str(config.system_instruction))
        return _fake_response(
            """```json
{
  "verdict": "DISTRACTED",
  "confidence": 0.93,
  "category": "social_video",
  "reasoning": "Usuário assistindo YouTube Shorts fora da matéria.",
  "suggested_message": "Pedro, fecha esse YouTube Shorts agora e volta para Direito Constitucional."
}
```""",
            tokens_in=450,
            tokens_out=75,
        )

    client = GeminiClient(cfg, budget, _raw_generate=raw_generate)
    judge = GeminiSemanticJudge(client, cfg)

    os_ctx = OSContextEvent(
        process_name="chrome.exe",
        window_title="Shorts - YouTube",
        browser_url="https://www.youtube.com/shorts/abc123",
    )
    vision = VisionSignalEvent(camera_available=True, face_present=True, phone_detected=False)
    snap = ScreenSnapshot(
        image_bytes_jpeg=b"\xff\xd8\xff\xe0FAKEJPEG\xff\xd9",
        perceptual_hash="abcdef0123456789",
        width=768,
        height=432,
    )

    verdict = await judge.evaluate(
        os_context=os_ctx,
        vision_signal=vision,
        study_goal=cfg.study_goal,
        current_subject=cfg.current_subject,
        screenshot=snap,
    )

    assert verdict.verdict == ActivityVerdict.DISTRACTED
    assert verdict.confidence == pytest.approx(0.93)
    assert verdict.category == "social_video"
    assert verdict.used_vision is True
    assert verdict.tokens_in == 450
    assert verdict.tokens_out == 75
    assert verdict.estimated_cost_usd > 0.0
    assert "Pedro" in verdict.suggested_message
    assert "Coach Firme e Direto" in captured_system[0]
    assert isinstance(captured_contents[0], list)
    assert len(captured_contents[0]) == 2
    assert isinstance(captured_contents[0][1], types.Part)


@pytest.mark.asyncio
async def test_semantic_judge_heuristic_fallback_zero_prison_and_signals() -> None:
    cfg = _make_config(api_key="")
    budget = BudgetGuard(cfg.ai)
    client = GeminiClient(cfg, budget)
    judge = GeminiSemanticJudge(client, cfg)

    # 1. Phone detected via vision
    v_phone = await judge.evaluate(
        OSContextEvent(process_name="obsidian.exe", window_title="Notas.md"),
        VisionSignalEvent(camera_available=True, face_present=True, phone_detected=True, phone_confidence=0.88),
        cfg.study_goal,
        "Direito Administrativo",
    )
    assert v_phone.verdict == ActivityVerdict.DISTRACTED
    assert "celular" in v_phone.suggested_message.lower()

    # 2. High suspicion pattern
    v_shorts = await judge.evaluate(
        OSContextEvent(
            process_name="chrome.exe",
            window_title="Shorts engraçados",
            browser_url="https://www.youtube.com/shorts/xyz",
        ),
        None,
        cfg.study_goal,
        "Português",
    )
    assert v_shorts.verdict == ActivityVerdict.DISTRACTED

    # 3. Study platform keyword
    v_tec = await judge.evaluate(
        OSContextEvent(
            process_name="chrome.exe",
            window_title="Tec Concursos - Questões de Direito",
            browser_url="https://www.tecconcursos.com.br/questoes",
        ),
        None,
        cfg.study_goal,
        "Direito Constitucional",
    )
    assert v_tec.verdict == ActivityVerdict.STUDYING

    # 4. Multi-purpose app (Antigravity / Obsidian) without distraction remains UNCERTAIN (Zero-Prison)
    v_obsidian = await judge.evaluate(
        OSContextEvent(process_name="Obsidian.exe", window_title="Vault Pessoal - Diario"),
        None,
        cfg.study_goal,
        "Direito Constitucional",
    )
    assert v_obsidian.verdict == ActivityVerdict.UNCERTAIN

    # 5. API failure fallback when configured
    cfg_err = _make_config(api_key="key")

    def boom(**kwargs: Any) -> Any:
        raise ConnectionError("offline")

    client_err = GeminiClient(cfg_err, BudgetGuard(cfg_err.ai), _raw_generate=boom)
    judge_err = GeminiSemanticJudge(client_err, cfg_err)
    v_music = await judge_err.evaluate(
        OSContextEvent(process_name="chrome.exe", window_title="Lo-fi Beats Study Music - YouTube"),
        None,
        cfg_err.study_goal,
        "Geral",
    )
    assert v_music.verdict == ActivityVerdict.MUSIC_ALLOWED


# ----------------------------------------------------------------------
# FirmCoachAgent tests
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_coach_agent_tool_calls_via_reply_text_and_tts() -> None:
    cfg = _make_config()
    budget = BudgetGuard(cfg.ai)
    engine = FakeFocusEngine(cfg)
    await engine.start_session()
    engine.latest_os = OSContextEvent(
        process_name="chrome.exe",
        window_title="Tec Concursos - Questão 42",
    )
    tts = FakeTTS()
    bus = EventBus()
    storage = FakeStorage()
    published: list[Any] = []
    bus.subscribe_all(lambda ev: published.append(ev))

    turn = 0

    def raw_generate(*, model: str, contents: Any, config: types.GenerateContentConfig) -> Any:
        nonlocal turn
        turn += 1
        if turn == 1:
            return _fake_response(
                function_calls=[
                    types.FunctionCall(
                        name="grant_temporary_allowance",
                        args={"pattern": "youtube", "minutes": 15, "reason": "Videoaula"},
                    ),
                    types.FunctionCall(
                        name="start_break",
                        args={"minutes": 5},
                    ),
                    types.FunctionCall(
                        name="set_study_goal",
                        args={"goal": "Simulado AGU", "subject": "Direito Administrativo"},
                    ),
                    types.FunctionCall(
                        name="record_study_note",
                        args={"note": "Revisar Súmula 473 STF", "subject": "Direito Administrativo"},
                    ),
                ],
                tokens_in=200,
                tokens_out=60,
            )
        return _fake_response(
            "Tudo certo, Pedro. YouTube liberado por 15 min, pausa de 5 min iniciada e meta atualizada para Direito Administrativo.",
            tokens_in=250,
            tokens_out=40,
        )

    client = GeminiClient(cfg, budget, _raw_generate=raw_generate)
    coach = FirmCoachAgent(client, engine, tts, cfg, event_bus=bus, storage=storage)

    reply = await coach.reply_text("Libera youtube 15 min, pausa de 5, muda pra Adm e anota Súmula 473", speak=True)

    assert "grant_temporary_allowance" in reply.tool_calls
    assert "start_break" in reply.tool_calls
    assert "set_study_goal" in reply.tool_calls
    assert "record_study_note" in reply.tool_calls
    assert len(engine.allowances) == 1
    assert engine.allowances[0].pattern == "youtube"
    assert engine.session.current_state == FocusState.BREAK_MODE
    assert engine.session.study_goal == "Simulado AGU"
    assert engine.session.current_subject == "Direito Administrativo"
    assert len(engine.notes) == 1
    assert "473" in engine.notes[0].note
    assert reply.audio_base64 is not None
    assert base64.b64decode(reply.audio_base64).startswith(b"FAKEAUDIO:")
    # local_playback=False by default avoids double echo in browser
    assert tts.spoken == []
    assert len(coach.get_history()) == 2
    assert len(storage.chats) == 2
    assert len(published) == 2


@pytest.mark.asyncio
async def test_coach_agent_inspect_current_screen_and_audio_reply() -> None:
    cfg = _make_config()
    budget = BudgetGuard(cfg.ai)
    engine = FakeFocusEngine(cfg)
    engine.latest_os = OSContextEvent(process_name="chrome.exe", window_title="Tec Concursos - Questão 10")
    tts = FakeTTS()

    calls: list[Any] = []

    def raw_generate(*, model: str, contents: Any, config: types.GenerateContentConfig) -> Any:
        calls.append(contents)
        if len(calls) == 1:
            # First call from reply_audio asks to inspect current screen via tool
            return _fake_response(
                function_calls=[
                    types.FunctionCall(
                        name="inspect_current_screen",
                        args={"question": "Explica essa questão na minha tela"},
                    )
                ]
            )
        return _fake_response(
            '{"transcript": "Coach, olha minha tela e explica essa questão.", '
            '"reply": "Na questão visível, a alternativa correta é a B por força do art. 37 da CF."}'
        )

    client = GeminiClient(cfg, budget, _raw_generate=raw_generate)
    coach = FirmCoachAgent(client, engine, tts, cfg)

    transcript, msg = await coach.reply_audio(
        b"RIFFfakewavbytes",
        mime_type="audio/wav",
        speak=True,
        local_playback=True,
    )

    assert transcript == "Coach, olha minha tela e explica essa questão."
    assert "art. 37" in msg.text
    assert "inspect_current_screen" in msg.tool_calls
    assert engine.screen_capture.calls == 1
    assert tts.spoken == [msg.text]
    assert msg.audio_base64 is None  # avoids double-echo in browser when spoken locally
    # Second turn contents should contain the attached JPEG Part from inspect_current_screen
    second_turn_last_parts = calls[1][-1].parts
    assert any(
        getattr(p, "inline_data", None) is not None and p.inline_data.mime_type == "image/jpeg"
        for p in second_turn_last_parts
    )


@pytest.mark.asyncio
async def test_coach_agent_offline_commands_nudge_and_history() -> None:
    cfg = _make_config(api_key="")
    budget = BudgetGuard(cfg.ai)
    engine = FakeFocusEngine(cfg)
    tts = FakeTTS()
    coach = FirmCoachAgent(GeminiClient(cfg, budget), engine, tts, cfg)

    assert coach.is_configured() is False

    msg_break = await coach.reply_text("pausa de 8 minutos", speak=False)
    assert "start_break" in msg_break.tool_calls
    assert engine.session.current_state == FocusState.BREAK_MODE
    assert engine.session.active_break_remaining_sec == 480

    msg_end = await coach.reply_text("encerrar pausa", speak=False)
    assert "end_break" in msg_end.tool_calls
    assert engine.session.current_state == FocusState.FOCUSED

    nudge = await coach.generate_nudge(
        level=InterventionLevel.FIRM_OVERLAY,
        reason="Instagram aberto há 4 minutos",
        os_context=OSContextEvent(process_name="chrome.exe", window_title="Instagram Reels"),
        vision_signal=VisionSignalEvent(camera_available=True, face_present=True, phone_detected=True),
        study_goal=cfg.study_goal,
        current_subject="Direito Penal",
    )
    assert "Pedro" in nudge
    assert "Direito Penal" in nudge
    assert "Instagram" in nudge

    assert len(coach.get_history()) == 4
    coach.clear_history()
    assert coach.get_history() == []


# ----------------------------------------------------------------------
# Voice Pipeline tests (EdgeAndOfflineTTS & SoundDeviceMicRecorder)
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_edge_tts_synthesize_and_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _make_config()

    class FakeCommunicateOK:
        def __init__(self, text: str, **kwargs: Any) -> None:
            self.text = text

        async def stream(self):
            yield {"type": "audio", "data": b"ID3chunk1"}
            yield {"type": "WordBoundary", "text": "Foco"}
            yield {"type": "audio", "data": b"chunk2"}

    tts = EdgeAndOfflineTTS(cfg.voice, communicate_factory=FakeCommunicateOK)
    audio, mime = await tts.synthesize("Foca nos estudos, Pedro.")
    assert audio == b"ID3chunk1chunk2"
    assert mime == "audio/mpeg"

    class FakeCommunicateFail:
        def __init__(self, text: str, **kwargs: Any) -> None:
            pass

        async def stream(self):
            if False:
                yield {}
            raise RuntimeError("network offline")

    spoken_offline: list[str] = []

    class FakePyttsx3Engine:
        def getProperty(self, name: str) -> list[Any]:
            return []

        def setProperty(self, name: str, val: Any) -> None:
            pass

        def say(self, text: str) -> None:
            spoken_offline.append(text)

        def runAndWait(self) -> None:
            pass

        def stop(self) -> None:
            pass

    tts_offline = EdgeAndOfflineTTS(
        cfg.voice,
        communicate_factory=FakeCommunicateFail,
        pyttsx3_factory=FakePyttsx3Engine,
    )
    audio_fail, mime_fail = await tts_offline.synthesize("Sem internet")
    assert audio_fail == b""
    assert mime_fail == "audio/wav"

    ok = await tts_offline.speak("Fallback offline funcionando.")
    assert ok is True
    assert spoken_offline == ["Fallback offline funcionando."]


@pytest.mark.asyncio
async def test_mic_recorder_availability_and_ram_wav_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _make_config()

    # 1. No input device available
    monkeypatch.setattr(
        "focus_guard.ai.voice_pipeline.sd.query_devices",
        lambda: [{"name": "Speaker", "max_input_channels": 0}],
    )
    mic_none = SoundDeviceMicRecorder(cfg.voice)
    assert mic_none.is_available() is False
    assert await mic_none.record_until_silence() is None

    # 2. Query failure handled gracefully
    monkeypatch.setattr(
        "focus_guard.ai.voice_pipeline.sd.query_devices",
        lambda: (_ for _ in ()).throw(OSError("No PortAudio")),
    )
    assert mic_none.is_available() is False

    # 3. Valid mic with speech onset followed by silence timeout -> WAV in RAM
    monkeypatch.setattr(
        "focus_guard.ai.voice_pipeline.sd.query_devices",
        lambda: [{"name": "USB Mic", "max_input_channels": 1}],
    )

    # Build deterministic chunks: 0.1s silence, 0.2s speech (sine), 0.5s silence (triggers 0.3s timeout)
    sr = 16000
    chunk_n = 1600  # 0.1s at 16kHz
    t = np.linspace(0, 0.1, chunk_n, endpoint=False, dtype=np.float32)
    speech_chunk = (0.3 * np.sin(2 * np.pi * 440.0 * t)).reshape(-1, 1)
    silence_chunk = np.zeros((chunk_n, 1), dtype=np.float32)
    scripted_chunks = [silence_chunk, speech_chunk, speech_chunk, silence_chunk, silence_chunk, silence_chunk, silence_chunk]

    class FakeInputStream:
        def __init__(self, **kwargs: Any) -> None:
            self.idx = 0

        def __enter__(self) -> FakeInputStream:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def read(self, n: int) -> tuple[np.ndarray, bool]:
            if self.idx < len(scripted_chunks):
                out = scripted_chunks[self.idx]
                self.idx += 1
                return out, False
            return silence_chunk, False

    mic = SoundDeviceMicRecorder(
        cfg.voice,
        rms_threshold=0.02,
        chunk_duration_sec=0.1,
        input_stream_factory=FakeInputStream,
    )
    assert mic.is_available() is True

    wav_bytes = await mic.record_until_silence(max_seconds=5.0, silence_timeout=0.3)
    assert wav_bytes is not None
    assert wav_bytes[:4] == b"RIFF"

    decoded, decoded_sr = sf.read(io.BytesIO(wav_bytes))
    assert decoded_sr == sr
    # 1 initial silence + 2 speech + 3 silence to hit 0.3s timeout = 6 chunks = 9600 samples
    assert len(decoded) == 6 * chunk_n

    # 4. Pure silence returns None
    class SilentStream:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> SilentStream:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def read(self, n: int) -> tuple[np.ndarray, bool]:
            return silence_chunk, False

    mic_silent = SoundDeviceMicRecorder(
        cfg.voice,
        rms_threshold=0.02,
        chunk_duration_sec=0.1,
        input_stream_factory=SilentStream,
    )
    assert await mic_silent.record_until_silence(max_seconds=0.3, silence_timeout=0.2) is None


@pytest.mark.asyncio
async def test_coach_agent_direct_screen_inspection_privacy_and_json_tool_protocol() -> None:
    cfg = _make_config()
    budget = BudgetGuard(cfg.ai)
    engine = FakeFocusEngine(cfg)
    tts = FakeTTS()

    calls: list[Any] = []

    def raw_generate(*, model: str, contents: Any, config: types.GenerateContentConfig) -> Any:
        calls.append(contents)
        if len(calls) == 1:
            return _fake_response("A alternativa correta é a C por causa da competência privativa.")
        if len(calls) == 2:
            return _fake_response(
                '{"transcript": "Anota isso e inicia pausa de 5 minutos", '
                '"reply": "Anotado e pausa iniciada.", '
                '"tool_calls": ['
                '{"name": "record_study_note", "args": {"note": "Art. 22 CF"}}, '
                '{"name": "start_break", "args": {"minutes": 5}}'
                ']}'
            )
        return _fake_response("Pedro, fecha essa aba e volta para Direito Constitucional agora.")

    client = GeminiClient(cfg, budget, _raw_generate=raw_generate)
    coach = FirmCoachAgent(client, engine, tts, cfg)

    # 1. Privacy mode blocks inspect_current_screen
    await engine.toggle_privacy(True)
    blocked = await coach.inspect_current_screen("Explica essa questão")
    assert blocked["captured"] is False
    assert blocked["privacy_protected"] is True

    # 2. Disabling privacy allows standalone inspect_current_screen
    await engine.toggle_privacy(False)
    ok_inspect = await coach.inspect_current_screen("Explica essa questão")
    assert ok_inspect["captured"] is True
    assert "alternativa correta" in ok_inspect["analysis"]

    # 3. JSON tool_calls protocol in reply_audio
    transcript, msg = await coach.reply_audio(b"RIFFaudio", speak=False)
    assert transcript == "Anota isso e inicia pausa de 5 minutos"
    assert "record_study_note" in msg.tool_calls
    assert "start_break" in msg.tool_calls
    assert len(engine.notes) == 1
    assert engine.session.current_state == FocusState.BREAK_MODE

    # 4. generate_nudge via Gemini shields sensitive window titles
    nudge = await coach.generate_nudge(
        level=InterventionLevel.VOICE,
        reason="Site suspeito",
        os_context=OSContextEvent(
            process_name="msedge.exe",
            window_title="Nubank Conta Secreta 1234",
            is_sensitive_window=True,
        ),
        vision_signal=None,
        study_goal=cfg.study_goal,
        current_subject="Direito Constitucional",
    )
    assert "Pedro" in nudge
    # Ensure sensitive title was NOT included in the prompt sent to Gemini
    assert "Nubank Conta Secreta" not in str(calls[2])


@pytest.mark.asyncio
async def test_edge_tts_pygame_playback_and_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _make_config()

    class FakeCommunicateOK:
        def __init__(self, text: str, **kwargs: Any) -> None:
            pass

        async def stream(self):
            yield {"type": "audio", "data": b"ID3FAKEMPEG"}

    loaded_streams: list[bytes] = []
    busy_calls = 0

    class FakeMusic:
        @staticmethod
        def load(fileobj: io.BytesIO, namehint: str = "") -> None:
            loaded_streams.append(fileobj.read())

        @staticmethod
        def play() -> None:
            pass

        @staticmethod
        def get_busy() -> bool:
            nonlocal busy_calls
            busy_calls += 1
            return busy_calls <= 1

        @staticmethod
        def stop() -> None:
            pass

    monkeypatch.setattr("focus_guard.ai.voice_pipeline.pygame.mixer.get_init", lambda: (44100, -16, 2))
    monkeypatch.setattr("focus_guard.ai.voice_pipeline.pygame.mixer.music", FakeMusic)

    tts = EdgeAndOfflineTTS(cfg.voice, communicate_factory=FakeCommunicateOK)
    played = await tts.speak("Bora estudar, Pedro.")
    assert played is True
    assert loaded_streams == [b"ID3FAKEMPEG"]
    tts.stop()


# ----------------------------------------------------------------------
# V2 Upgrade Tests (Judge Webcam/Multi-Monitor, Coach MIME/Windows, TTS & Mic V2)
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_semantic_judge_v2_webcam_lateral_prompt_and_phone_override() -> None:
    cfg = _make_config()
    cfg.vision.camera_position = "right"
    cfg.vision.screen_yaw_min_deg = -55.0
    cfg.vision.screen_yaw_max_deg = 25.0
    cfg.vision.calibrated = True
    budget = BudgetGuard(cfg.ai)

    captured_calls: list[dict[str, Any]] = []

    def raw_generate(*, model: str, contents: Any, config: types.GenerateContentConfig) -> Any:
        captured_calls.append(
            {
                "contents": contents,
                "system_instruction": str(config.system_instruction),
            }
        )
        # Deliberately test phone_in_use=True forcing DISTRACTED + fallback suggested_message
        return _fake_response(
            '{"verdict": "STUDYING", "confidence": 0.91, "category": "phone_check", '
            '"reasoning": "Tela mostra Obsidian no M1 e YouTube no M2, e webcam mostra celular na mão.", '
            '"suggested_message": "", "phone_in_use": true, "physical_state": "using_phone"}',
            tokens_in=520,
            tokens_out=90,
        )

    client = GeminiClient(cfg, budget, _raw_generate=raw_generate)
    judge = GeminiSemanticJudge(client, cfg)

    os_ctx = OSContextEvent(
        process_name="obsidian.exe",
        window_title="Direito Constitucional.md - Obsidian",
        active_monitor_label="M1",
        visible_windows=[
            VisibleWindowInfo(
                hwnd=1,
                process_name="obsidian.exe",
                window_title="Direito Constitucional.md",
                monitor_label="M1",
                screen_area_pct=55.0,
                is_foreground=True,
            ),
            VisibleWindowInfo(
                hwnd=2,
                process_name="chrome.exe",
                window_title="Melhores Gols do Brasileirão - YouTube",
                monitor_label="M2",
                screen_area_pct=45.0,
                is_foreground=False,
            ),
            VisibleWindowInfo(
                hwnd=3,
                process_name="chrome.exe",
                window_title="Itaú Internet Banking Segredo",
                monitor_label="M2",
                screen_area_pct=10.0,
                is_sensitive=True,
            ),
        ],
        notable_running_processes=[
            ProcessActivityInfo(
                process_name="chrome.exe",
                cpu_percent=7.4,
                memory_mb=850.0,
                has_active_audio=True,
                window_count=2,
                category="browser",
            )
        ],
        usage_pattern=UsagePatternMetrics(
            active_input_ratio_60s=0.15,
            window_switches_60s=11,
            continuous_same_screen_sec=40.0,
            interaction_mode="rapid_switching",
        ),
    )
    vision = VisionSignalEvent(
        camera_available=True,
        face_present=True,
        camera_position="right",
        calibrated=True,
        looking_at_screen=True,
        head_yaw_deg=-28.0,
        phone_detected=True,
        phone_confidence=0.82,
    )
    snap = ScreenSnapshot(
        image_bytes_jpeg=b"\xff\xd8\xff\xe0SCREENJPEG\xff\xd9",
        perceptual_hash="1234567890abcdef",
        width=1400,
        height=400,
    )
    webcam_jpeg = b"\xff\xd8\xff\xe0WEBCAMJPEG\xff\xd9"

    verdict = await judge.evaluate(
        os_context=os_ctx,
        vision_signal=vision,
        study_goal=cfg.study_goal,
        current_subject=cfg.current_subject,
        screenshot=snap,
        webcam_frame_jpeg=webcam_jpeg,
    )

    assert verdict.used_vision is True
    assert verdict.used_webcam is True
    assert verdict.phone_in_use is True
    assert verdict.physical_state == "using_phone"
    assert verdict.verdict == ActivityVerdict.DISTRACTED
    assert "celular" in verdict.suggested_message.lower()

    assert len(captured_calls) == 1
    contents = captured_calls[0]["contents"]
    sys_inst = captured_calls[0]["system_instruction"]
    assert isinstance(contents, list)
    assert len(contents) == 3
    prompt_str = contents[0]
    assert isinstance(contents[1], types.Part)
    assert isinstance(contents[2], types.Part)
    assert contents[1].inline_data is not None
    assert contents[1].inline_data.data == snap.image_bytes_jpeg
    assert contents[2].inline_data is not None
    assert contents[2].inline_data.data == webcam_jpeg

    # System instruction checks
    assert "camera_position='right'" in sys_inst
    assert "-55.0°..25.0°" in sys_inst
    assert "focused_on_screen" in sys_inst
    assert "visible_windows" in sys_inst
    assert "notable_running_processes" in sys_inst
    assert "usage_pattern" in sys_inst
    assert '"phone_in_use"' in sys_inst
    assert '"physical_state"' in sys_inst

    # User prompt checks (and sensitive window title redaction)
    assert "Monitor ativo: M1" in prompt_str
    assert "Melhores Gols do Brasileirão" in prompt_str
    assert "Itaú Internet Banking Segredo" not in prompt_str
    assert "janela sensível protegida" in prompt_str
    assert "rapid_switching" in prompt_str
    assert "window_switches_60s=11" in prompt_str
    assert "camera_position=right" in prompt_str
    assert "looking_at_screen=True" in prompt_str
    assert "calibrated=True" in prompt_str
    assert "Image 1 = Screen(s), Image 2 = Webcam" in prompt_str


@pytest.mark.asyncio
async def test_coach_agent_v2_mime_normalization_and_multimonitor_context() -> None:
    cfg = _make_config()
    budget = BudgetGuard(cfg.ai)
    engine = FakeFocusEngine(cfg)
    await engine.start_session()
    engine.latest_os = OSContextEvent(
        process_name="obsidian.exe",
        window_title="Resumo Adm.md",
        active_monitor_label="M1",
        visible_windows=[
            VisibleWindowInfo(
                process_name="obsidian.exe",
                window_title="Resumo Adm.md",
                monitor_label="M1",
                screen_area_pct=50.0,
                is_foreground=True,
            ),
            VisibleWindowInfo(
                process_name="chrome.exe",
                window_title="Nubank Saldo Secreto",
                monitor_label="M2",
                is_sensitive=True,
            ),
            VisibleWindowInfo(
                process_name="chrome.exe",
                window_title="Tec Concursos - Questões",
                monitor_label="M2",
                screen_area_pct=50.0,
            ),
            VisibleWindowInfo(process_name="app3.exe", window_title="Win 3", monitor_label="M1"),
            VisibleWindowInfo(process_name="app4.exe", window_title="Win 4", monitor_label="M2"),
            VisibleWindowInfo(process_name="app5.exe", window_title="Win 5", monitor_label="M1"),
            VisibleWindowInfo(process_name="app6.exe", window_title="Win 6 Ignored", monitor_label="M2"),
        ],
        usage_pattern=UsagePatternMetrics(
            interaction_mode="active_work",
            window_switches_60s=4,
            active_input_ratio_60s=0.72,
        ),
    )
    tts = FakeTTS()
    captured_contents: list[Any] = []

    def raw_generate(*, model: str, contents: Any, config: types.GenerateContentConfig) -> Any:
        captured_contents.append(contents)
        return _fake_response(
            '{"transcript": "Coach, estou focado?", "reply": "Sim, continua firme no Tec Concursos."}'
        )

    client = GeminiClient(cfg, budget, _raw_generate=raw_generate)
    coach = FirmCoachAgent(client, engine, tts, cfg)

    summary = await coach.get_session_summary()
    assert len(summary["visible_windows"]) == 5
    titles = [w["window_title"] for w in summary["visible_windows"]]
    assert "Nubank Saldo Secreto" not in titles
    assert "Win 6 Ignored" not in titles
    assert summary["usage_pattern"] == {
        "interaction_mode": "active_work",
        "window_switches_60s": 4,
        "active_input_ratio_60s": 0.72,
    }

    transcript, reply = await coach.reply_audio(
        b"WEBMFAKEBYTES",
        mime_type="audio/webm;codecs=opus",
        speak=False,
    )
    assert transcript == "Coach, estou focado?"
    assert "Tec Concursos" in reply.text

    sent_parts = captured_contents[0][-1].parts
    audio_parts = [
        p for p in sent_parts if getattr(p, "inline_data", None) is not None
    ]
    assert len(audio_parts) == 1
    assert audio_parts[0].inline_data.mime_type == "audio/webm"
    header_text = sent_parts[0].text
    assert "Tec Concursos - Questões" in header_text
    assert "Nubank Saldo Secreto" not in header_text
    assert "interaction_mode=active_work" in header_text
    assert "window_switches_60s=4" in header_text
    assert "active_input_ratio_60s=0.72" in header_text


@pytest.mark.asyncio
async def test_edge_tts_v2_device_routing_and_truncated_name_matching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _make_config()
    sdl2_devices = [
        "Alto-falantes (fifine  Microphone)",
        "Fones de ouvido (Soundcore Life Q30)",
        "Alto-falantes (Echo-0JN)",
    ]
    init_calls: list[str | None] = []
    quit_calls = 0
    mixer_initialized = False

    def fake_query_devices(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("kind") == "output":
            # Windows MME PortAudio truncates names at 31 chars
            return {
                "name": "Fones de ouvido (Soundcore Life",
                "max_output_channels": 2,
                "default_samplerate": 44100.0,
            }
        return [
            {"name": "Alto-falantes (fifine  Micropho", "max_output_channels": 2, "default_samplerate": 44100.0},
            {"name": "Fones de ouvido (Soundcore Life", "max_output_channels": 2, "default_samplerate": 44100.0},
            {"name": "Alto-falantes (Echo-0JN)", "max_output_channels": 2, "default_samplerate": 44100.0},
        ]

    def fake_get_init() -> tuple[int, int, int] | None:
        return (44100, -16, 2) if mixer_initialized else None

    def fake_init(*args: Any, **kwargs: Any) -> None:
        nonlocal mixer_initialized
        dev = kwargs.get("devicename")
        init_calls.append(dev)
        if dev == "Alto-falantes (Echo-0JN)":
            raise RuntimeError("Echo disconnected")
        mixer_initialized = True

    def fake_quit() -> None:
        nonlocal mixer_initialized, quit_calls
        mixer_initialized = False
        quit_calls += 1

    class FakeMusic:
        @staticmethod
        def load(fileobj: io.BytesIO, namehint: str = "") -> None:
            pass

        @staticmethod
        def play() -> None:
            pass

        @staticmethod
        def get_busy() -> bool:
            return False

        @staticmethod
        def stop() -> None:
            pass

    class FakeCommunicateOK:
        def __init__(self, text: str, **kwargs: Any) -> None:
            pass

        async def stream(self):
            yield {"type": "audio", "data": b"ID3FAKE"}

    monkeypatch.setattr("focus_guard.ai.voice_pipeline.sd.query_devices", fake_query_devices)
    monkeypatch.setattr(
        "focus_guard.ai.voice_pipeline.pygame._sdl2.audio.get_audio_device_names",
        lambda is_capture: sdl2_devices,
    )
    monkeypatch.setattr("focus_guard.ai.voice_pipeline.pygame.mixer.get_init", fake_get_init)
    monkeypatch.setattr("focus_guard.ai.voice_pipeline.pygame.mixer.init", fake_init)
    monkeypatch.setattr("focus_guard.ai.voice_pipeline.pygame.mixer.quit", fake_quit)
    monkeypatch.setattr("focus_guard.ai.voice_pipeline.pygame.mixer.music", FakeMusic)

    tts = EdgeAndOfflineTTS(cfg.voice, communicate_factory=FakeCommunicateOK)
    devices = tts.list_output_devices()
    assert [d["name"] for d in devices] == sdl2_devices
    default_entry = next(d for d in devices if d["is_default"])
    assert default_entry["name"] == "Fones de ouvido (Soundcore Life Q30)"

    # First playback should route to full SDL2 name (not index 0 fifine)
    ok = await tts.speak("Testando fone Bluetooth.")
    assert ok is True
    assert init_calls == ["Fones de ouvido (Soundcore Life Q30)"]

    # Second playback with same device reuses active mixer without re-init
    ok2 = await tts.speak("Segunda fala.")
    assert ok2 is True
    assert init_calls == ["Fones de ouvido (Soundcore Life Q30)"]

    # Switching to Echo (which fails devicename init) quits and falls back to pygame.mixer.init()
    tts.set_output_device("Echo-0JN")
    ok3 = await tts.speak("Testando fallback.")
    assert ok3 is True
    assert quit_calls >= 1
    assert init_calls[-2:] == ["Alto-falantes (Echo-0JN)", None]


@pytest.mark.asyncio
async def test_mic_recorder_v2_stereo_44k_resample_and_manual_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _make_config()
    bus = EventBus()
    mic_events: list[MicStateEvent] = []
    bus.subscribe(MicStateEvent, lambda ev: mic_events.append(ev))

    monkeypatch.setattr(
        "focus_guard.ai.voice_pipeline.sd.query_devices",
        lambda *args, **kwargs: (
            {
                "index": 1,
                "name": "Microfone (fifine  Microphone)",
                "max_input_channels": 2,
                "default_samplerate": 44100.0,
            }
            if kwargs.get("kind") == "input"
            else [
                {
                    "index": 0,
                    "name": "Headset (Soundcore Hands-Free)",
                    "max_input_channels": 1,
                    "default_samplerate": 16000.0,
                    "hostapi": 3,
                },
                {
                    "index": 1,
                    "name": "Microfone (fifine  Microphone)",
                    "max_input_channels": 2,
                    "default_samplerate": 44100.0,
                    "hostapi": 0,
                },
            ]
        ),
    )

    chunk_n = 4410  # 0.1s at 44.1kHz
    t = np.linspace(0, 0.1, chunk_n, endpoint=False, dtype=np.float32)
    sine = 0.28 * np.sin(2 * np.pi * 440.0 * t)
    stereo_speech = np.column_stack([sine, sine]).astype(np.float32)
    open_attempts: list[tuple[int, int]] = []
    recorder_ref: list[SoundDeviceMicRecorder] = []

    class FakeStereo44kStream:
        def __init__(self, *, samplerate: int, channels: int, **kwargs: Any) -> None:
            open_attempts.append((samplerate, channels))
            if channels == 1:
                raise RuntimeError("Device only supports stereo")
            self.reads = 0

        def __enter__(self) -> FakeStereo44kStream:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def read(self, n: int) -> tuple[np.ndarray, bool]:
            self.reads += 1
            time.sleep(0.01)
            if self.reads >= 5 and recorder_ref:
                recorder_ref[0].stop_recording()
            return stereo_speech, False

    mic = SoundDeviceMicRecorder(
        cfg.voice,
        event_bus=bus,
        rms_threshold=0.02,
        chunk_duration_sec=0.1,
        input_stream_factory=FakeStereo44kStream,
    )
    recorder_ref.append(mic)
    mic.set_input_device("fifine")

    listed = mic.list_input_devices()
    assert listed[0]["name"] == "Microfone (fifine  Microphone)"
    assert listed[0]["is_selected"] is True

    wav_bytes = await mic.record_until_silence(max_seconds=10.0, silence_timeout=2.0)
    assert wav_bytes is not None
    assert wav_bytes[:4] == b"RIFF"
    # Opened 44100Hz ch=1 (failed), then 44100Hz ch=2 (succeeded)
    assert open_attempts == [(44100, 1), (44100, 2)]

    decoded, decoded_sr = sf.read(io.BytesIO(wav_bytes))
    assert decoded_sr == 16000
    assert decoded.ndim == 1
    # 5 chunks * 0.1s = 0.5s -> 8000 samples at 16000Hz
    assert len(decoded) == 8000

    # Let any threadsafe queued tasks flush cleanly
    await asyncio.sleep(0.01)
    states = [ev.state for ev in mic_events]
    assert "listening" in states
    assert "processing" in states
    assert states[-1] == "idle"


@pytest.mark.asyncio
async def test_mic_recorder_v2_adaptive_vad_preroll_and_quiet_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _make_config()
    cfg.voice.play_mic_chime = False

    monkeypatch.setattr(
        "focus_guard.ai.voice_pipeline.sd.query_devices",
        lambda *args, **kwargs: (
            {"index": 0, "name": "Studio Mic", "max_input_channels": 1, "default_samplerate": 16000.0}
            if kwargs.get("kind") == "input"
            else [{"index": 0, "name": "Studio Mic", "max_input_channels": 1, "default_samplerate": 16000.0}]
        ),
    )

    chunk_n = 1600  # 0.1s at 16kHz
    t = np.linspace(0, 0.1, chunk_n, endpoint=False, dtype=np.float32)
    # Quiet ambient noise (rms ~0.001 -> adaptive_thresh = 0.0028)
    ambient = np.full((chunk_n, 1), 0.001, dtype=np.float32)
    # Quiet speech (rms ~0.0021 -> below 0.0028 threshold, but >= 0.0018 fallback!)
    quiet_speech = (0.003 * np.sin(2 * np.pi * 440.0 * t)).reshape(-1, 1)

    chunks = [ambient, ambient, ambient, quiet_speech, quiet_speech, quiet_speech]

    class FakeAdaptiveStream:
        def __init__(self, **kwargs: Any) -> None:
            self.idx = 0

        def __enter__(self) -> FakeAdaptiveStream:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def read(self, n: int) -> tuple[np.ndarray, bool]:
            if self.idx < len(chunks):
                out = chunks[self.idx]
                self.idx += 1
                return out, False
            mic_ref.stop_recording()
            return ambient, False

    monkeypatch.setattr("focus_guard.ai.voice_pipeline.sd.InputStream", FakeAdaptiveStream)
    mic_ref = SoundDeviceMicRecorder(cfg.voice, chunk_duration_sec=0.1)

    wav_bytes = await mic_ref.record_until_silence(max_seconds=5.0, silence_timeout=1.0)
    assert wav_bytes is not None
    decoded, decoded_sr = sf.read(io.BytesIO(wav_bytes))
    assert decoded_sr == 16000
    # Gentle normalization boosted peak (> 0.003)
    assert float(np.max(np.abs(decoded))) > 0.01


@pytest.mark.asyncio
async def test_semantic_judge_v2_1_conjunctive_anti_alibi_invariants() -> None:
    cfg = _make_config(api_key="live-key")
    cfg.current_subject = "itil v4"
    budget = BudgetGuard(cfg.ai)

    # Case 1: LLM says STUDYING, but matches_current_subject=False (background ITIL decoy tab)
    def _raw_alibi(*, model: str, contents: Any, config: Any) -> Any:
        return _fake_response(
            json.dumps(
                {
                    "foreground_topic": "Drone DJI Matrice 5TD review",
                    "matches_current_subject": False,
                    "secondary_monitor_contamination": False,
                    "is_foreground_music_only": False,
                    "verdict": "STUDYING",
                    "confidence": 0.95,
                    "category": "video",
                    "reasoning": "Há aba de ITIL no outro monitor.",
                    "suggested_message": "",
                }
            )
        )

    judge = GeminiSemanticJudge(GeminiClient(cfg, budget, _raw_generate=_raw_alibi), cfg)
    verdict = await judge.evaluate(
        OSContextEvent(
            process_name="chrome.exe",
            window_title="(4) O DJI Matrice 5TD é o drone MAIS AVANÇADO do MUNDO - YouTube",
        ),
        vision_signal=None,
        study_goal="Estudo para certificação ITIL",
        current_subject="itil v4",
    )
    assert verdict.verdict == ActivityVerdict.DISTRACTED
    assert verdict.suggested_message != ""

    # Case 2: LLM says STUDYING, but secondary_monitor_contamination=True -> DISTRACTED
    def _raw_contaminated(*, model: str, contents: Any, config: Any) -> Any:
        return _fake_response(
            json.dumps(
                {
                    "foreground_topic": "ITIL apostila PDF",
                    "matches_current_subject": True,
                    "secondary_monitor_contamination": True,
                    "is_foreground_music_only": False,
                    "verdict": "STUDYING",
                    "confidence": 0.90,
                    "category": "study",
                    "reasoning": "PDF na M1, mas vídeo de jogo na M2.",
                    "suggested_message": "",
                }
            )
        )

    judge2 = GeminiSemanticJudge(GeminiClient(cfg, budget, _raw_generate=_raw_contaminated), cfg)
    verdict2 = await judge2.evaluate(
        OSContextEvent(process_name="msedge.exe", window_title="ITIL v4.pdf"),
        vision_signal=None,
        study_goal="Estudo para certificação ITIL",
        current_subject="itil v4",
    )
    assert verdict2.verdict == ActivityVerdict.DISTRACTED

    # Case 3: Foreground FocusGuard dashboard is allowed when matches_current_subject=False
    verdict3 = await judge.evaluate(
        OSContextEvent(
            process_name="chrome.exe",
            window_title="FocusGuard — Monitor Multimodal de Estudos",
            browser_url="http://127.0.0.1:8765/",
        ),
        vision_signal=None,
        study_goal="Estudo para certificação ITIL",
        current_subject="itil v4",
    )
    assert verdict3.verdict == ActivityVerdict.STUDYING

    # Case 4: is_foreground_music_only=True downgrades STUDYING -> MUSIC_ALLOWED
    def _raw_music(*, model: str, contents: Any, config: Any) -> Any:
        return _fake_response(
            json.dumps(
                {
                    "foreground_topic": "X-COOL Tiktok Edit Audio",
                    "matches_current_subject": False,
                    "secondary_monitor_contamination": False,
                    "is_foreground_music_only": True,
                    "verdict": "STUDYING",
                    "confidence": 0.92,
                    "category": "music",
                    "reasoning": "Música no YouTube.",
                    "suggested_message": "",
                }
            )
        )

    judge4 = GeminiSemanticJudge(GeminiClient(cfg, budget, _raw_generate=_raw_music), cfg)
    verdict4 = await judge4.evaluate(
        OSContextEvent(process_name="chrome.exe", window_title="X-COOL! - YouTube"),
        vision_signal=None,
        study_goal="Estudo para certificação ITIL",
        current_subject="itil v4",
    )
    assert verdict4.verdict == ActivityVerdict.MUSIC_ALLOWED


@pytest.mark.asyncio
async def test_semantic_judge_v3_user_visible_in_webcam_false_forces_away_with_message() -> None:
    cfg = _make_config(api_key="live-key")
    cfg.current_subject = "itil v4"
    budget = BudgetGuard(cfg.ai)

    def _raw_not_visible(*, model: str, contents: Any, config: Any) -> Any:
        return _fake_response(
            json.dumps(
                {
                    "foreground_topic": "PDF de ITIL",
                    "matches_current_subject": True,
                    "secondary_monitor_contamination": False,
                    "is_foreground_music_only": False,
                    "user_visible_in_webcam": False,
                    "verdict": "STUDYING",
                    "confidence": 0.91,
                    "category": "study",
                    "reasoning": "Tela mostra PDF, mas a câmera não mostra ninguém na cadeira.",
                    "suggested_message": "",
                    "phone_in_use": False,
                    "physical_state": "focused_on_screen",
                }
            )
        )

    judge = GeminiSemanticJudge(GeminiClient(cfg, budget, _raw_generate=_raw_not_visible), cfg)
    verdict = await judge.evaluate(
        OSContextEvent(process_name="msedge.exe", window_title="ITIL v4 Apostila.pdf"),
        vision_signal=VisionSignalEvent(camera_available=True, face_present=True),
        study_goal="Certificação ITIL",
        current_subject="itil v4",
        webcam_frame_jpeg=b"\xff\xd8\xff\xe0WEBCAM\xff\xd9",
    )
    assert verdict.verdict == ActivityVerdict.AWAY
    assert verdict.physical_state == "away"
    assert verdict.used_webcam is True
    assert verdict.suggested_message != ""
    assert "Pedro" in verdict.suggested_message


@pytest.mark.asyncio
async def test_coach_agent_v3_anti_repetition_and_deterministic_variation() -> None:
    cfg = _make_config(api_key="live-key")
    cfg.current_subject = "ITIL v4"
    budget = BudgetGuard(cfg.ai)
    engine = FakeFocusEngine(cfg)
    await engine.start_session()
    tts = FakeTTS()

    prompts_seen: list[str] = []

    def _raw_nudge(*, model: str, contents: Any, config: Any) -> Any:
        prompts_seen.append(str(contents))
        idx = len(prompts_seen)
        return _fake_response(
            f"Pedro, cobrança única número {idx}: fecha esse YouTube e volta para ITIL v4!"
        )

    coach = FirmCoachAgent(GeminiClient(cfg, budget, _raw_generate=_raw_nudge), engine, tts, cfg)

    first = await coach.generate_nudge(
        level=InterventionLevel.SOFT,
        reason="YouTube fora de tema",
        os_context=OSContextEvent(process_name="chrome.exe", window_title="Drone DJI - YouTube"),
        vision_signal=VisionSignalEvent(camera_available=True, face_present=True),
        study_goal=cfg.study_goal,
        current_subject="ITIL v4",
    )
    assert "cobrança única número 1" in first
    assert len(coach._recent_nudges) == 1

    second = await coach.generate_nudge(
        level=InterventionLevel.VOICE,
        reason="PC em uso, mas não estou conseguindo te ver na câmera (fora de enquadramento)",
        os_context=OSContextEvent(process_name="chrome.exe", window_title="Drone DJI - YouTube"),
        vision_signal=VisionSignalEvent(camera_available=True, face_present=False),
        study_goal=cfg.study_goal,
        current_subject="ITIL v4",
    )
    assert "cobrança única número 2" in second
    assert "cobrança única número 1" in prompts_seen[1]
    assert "PROIBIDO repetir" in prompts_seen[1]

    # Deterministic variation rotates creatively without repeating consecutively
    n0 = coach._deterministic_nudge(
        "Pedro",
        CoachPersona.FIRME,
        InterventionLevel.SOFT,
        "PC em uso, mas não estou conseguindo te ver na câmera (fora de enquadramento)",
        OSContextEvent(process_name="chrome.exe", window_title="Simulado"),
        VisionSignalEvent(camera_available=True, face_present=False),
        "ITIL v4",
        variant_idx=0,
    )
    n1 = coach._deterministic_nudge(
        "Pedro",
        CoachPersona.FIRME,
        InterventionLevel.SOFT,
        "PC em uso, mas não estou conseguindo te ver na câmera (fora de enquadramento)",
        OSContextEvent(process_name="chrome.exe", window_title="Simulado"),
        VisionSignalEvent(camera_available=True, face_present=False),
        "ITIL v4",
        variant_idx=1,
    )
    n_obs = coach._deterministic_nudge(
        "Pedro",
        CoachPersona.FIRME,
        InterventionLevel.FIRM_OVERLAY,
        "Câmera tampada ou escura demais — não consigo te ver",
        None,
        VisionSignalEvent(camera_available=True, face_present=False, camera_obstructed=True),
        "ITIL v4",
        variant_idx=2,
    )
    assert n0 != n1
    assert "Pedro" in n0 and "ITIL v4" in n0
    assert "Pedro" in n_obs and "ITIL v4" in n_obs and "tampada" in n_obs.lower()


@pytest.mark.asyncio
async def test_edge_tts_interrupted_speech_never_falls_back_to_pyttsx3_or_plays_stale_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _make_config()
    synth_texts: list[str] = []
    played_payloads: list[bytes] = []
    pyttsx3_calls: list[str] = []

    release_first_synth = asyncio.Event()
    first_playing = asyncio.Event()
    loop = asyncio.get_running_loop()

    class ControlledCommunicate:
        def __init__(self, text: str, **kwargs: Any) -> None:
            self.text = text
            synth_texts.append(text)

        async def stream(self):
            if "primeira lenta" in self.text:
                await release_first_synth.wait()
            yield {"type": "audio", "data": f"MP3:{self.text}".encode()}

    class FakePyttsx3:
        def getProperty(self, name: str) -> list[Any]:
            return []

        def setProperty(self, name: str, val: Any) -> None:
            pass

        def say(self, text: str) -> None:
            pyttsx3_calls.append(text)

        def runAndWait(self) -> None:
            pass

        def stop(self) -> None:
            pass

    busy_state = {"playing": False, "ticks": 0}

    class FakeMusic:
        @staticmethod
        def load(fileobj: io.BytesIO, namehint: str = "") -> None:
            played_payloads.append(fileobj.read())

        @staticmethod
        def play() -> None:
            busy_state["playing"] = True
            busy_state["ticks"] = 0
            if played_payloads and b"longa" in played_payloads[-1]:
                loop.call_soon_threadsafe(first_playing.set)

        @staticmethod
        def get_busy() -> bool:
            if not busy_state["playing"]:
                return False
            busy_state["ticks"] += 1
            if played_payloads and b"longa" in played_payloads[-1]:
                return busy_state["ticks"] < 200
            return busy_state["ticks"] < 2

        @staticmethod
        def stop() -> None:
            busy_state["playing"] = False

        @staticmethod
        def unload() -> None:
            pass

    monkeypatch.setattr("focus_guard.ai.voice_pipeline.pygame.mixer.get_init", lambda: (44100, -16, 2))
    monkeypatch.setattr("focus_guard.ai.voice_pipeline.pygame.mixer.music", FakeMusic)

    tts = EdgeAndOfflineTTS(
        cfg.voice,
        communicate_factory=ControlledCommunicate,
        pyttsx3_factory=FakePyttsx3,
    )

    # 1. Markdown noise is stripped cleanly before TTS synthesis
    ok_md = await tts.speak("Pedro, sai do `chrome://history` e foca em *ITIL v4*!")
    assert ok_md is True
    assert synth_texts[-1] == "Pedro, sai do chrome://history e foca em ITIL v4!"

    # 2. Interrupting active playback must return False and NEVER fall back to pyttsx3
    long_task = asyncio.create_task(tts.speak("Fala longa antiga."))
    await asyncio.wait_for(first_playing.wait(), timeout=2.0)
    ok_new = await tts.speak("Fala nova urgente.")
    ok_old = await long_task
    assert ok_new is True
    assert ok_old is False
    assert pyttsx3_calls == []

    # 3. Stale synthesis that completes after a newer speak() must be discarded, never played
    played_payloads.clear()
    stale_task = asyncio.create_task(tts.speak("Fala primeira lenta do passado."))
    await asyncio.sleep(0.02)
    ok_fast = await tts.speak("Fala atual imediata.")
    release_first_synth.set()
    ok_stale = await stale_task
    assert ok_fast is True
    assert ok_stale is False
    assert played_payloads == [b"MP3:Fala atual imediata."]
    assert pyttsx3_calls == []




