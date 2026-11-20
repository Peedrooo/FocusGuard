"""Deterministic end-to-end Replay Simulator covering 10 real-world study scenarios.

Wires REAL FocusEngine + REAL SQLiteStorage + REAL EventBus + REAL BudgetGuard +
REAL PrivacyShield + REAL GeminiSemanticJudge + REAL FirmCoachAgent +
REAL WindowsInterventionDispatcher (enable_ui=False) with deterministic clock &
controlled Gemini responses.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from focus_guard.ai import FirmCoachAgent, GeminiClient, GeminiSemanticJudge
from focus_guard.core.budget import BudgetGuard
from focus_guard.core.config import AppConfig, load_config
from focus_guard.core.event_bus import EventBus
from focus_guard.core.events import (
    FocusState,
    InterventionCommand,
    InterventionLevel,
    OSContextEvent,
    ProactiveSuggestionEvent,
    ScreenSnapshot,
    UsagePatternMetrics,
    VisibleWindowInfo,
    VisionSignalEvent,
)
from focus_guard.core.privacy import PrivacyShield
from focus_guard.engine import FocusEngine, SQLiteStorage
from focus_guard.ui import WindowsInterventionDispatcher
from tests.fakes import FakeTTS, FakeVisionSensor


class ReplayClock:
    def __init__(self, start: datetime | None = None) -> None:
        self.now = start or datetime(2026, 11, 19, 14, 0, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> datetime:
        self.now += timedelta(seconds=seconds)
        return self.now


class ControlledScreenCapture:
    """Real PrivacyShield-enforced in-memory screen capture stub for deterministic replay."""

    def __init__(self, shield: PrivacyShield) -> None:
        self._shield = shield
        self.capture_count = 0

    def capture(self, os_context: OSContextEvent | None = None) -> ScreenSnapshot | None:
        if os_context is not None:
            if os_context.is_sensitive_window:
                return None
            sensitive, _ = self._shield.is_sensitive(
                os_context.process_name, os_context.window_title, os_context.browser_url
            )
            if sensitive:
                return None
        self.capture_count += 1
        return ScreenSnapshot(
            image_bytes_jpeg=b"\xff\xd8\xff\xe0FAKEJPEG\xff\xd9",
            perceptual_hash="0f1e2d3c4b5a6978",
            width=768,
            height=432,
        )


def _fake_gemini_response(payload: dict[str, Any], prompt_tok: int = 420, out_tok: int = 75) -> Any:
    return SimpleNamespace(
        text=json.dumps(payload, ensure_ascii=False),
        usage_metadata=SimpleNamespace(
            prompt_token_count=prompt_tok,
            candidates_token_count=out_tok,
        ),
        candidates=[],
    )


async def _step(engine: FocusEngine, clock: ReplayClock, total_sec: int, step_sec: int = 5) -> None:
    remaining = total_sec
    while remaining > 0:
        dt = min(step_sec, remaining)
        clock.advance(dt)
        await engine.tick(clock.now)
        remaining -= dt


async def _build_replay_rig(
    tmp_path: Path,
    judge_router: Callable[[str], dict[str, Any]],
) -> dict[str, Any]:
    cfg: AppConfig = load_config(user_config_path=tmp_path / "nonexistent_user_config.yaml")
    cfg.gemini_api_key = "replay-fake-key"
    cfg.data_dir = tmp_path / "data"
    cfg.study_goal = "Auditor Fiscal"
    cfg.current_subject = "Direito Constitucional"

    clock = ReplayClock()
    bus = EventBus()
    budget = BudgetGuard(cfg.ai)
    shield = PrivacyShield(cfg.privacy)
    storage = SQLiteStorage(cfg.db_path)
    await storage.init()

    judge_calls: list[str] = []

    def _raw_generate(*, model: str, contents: Any, config: Any) -> Any:
        if isinstance(contents, str):
            joined = contents
        elif isinstance(contents, (list, tuple)):
            text_parts = [str(p) for p in contents if isinstance(p, str)]
            joined = "\n".join(text_parts)
        else:
            joined = str(contents)
        sys_inst = str(getattr(config, "system_instruction", "") or "")
        # Differentiate judge vs coach nudge calls by prompt content
        if '"verdict"' in sys_inst or "Janela ativa" in joined or "Processo:" in joined:
            judge_calls.append(joined)
            payload = judge_router(joined)
            return _fake_gemini_response(payload)
        return _fake_gemini_response(
            {
                "reply": "Pedro, sem conversa fiada: volta para Direito Constitucional agora.",
                "nudge": "Pedro, para de enrolar e volta para Direito Constitucional.",
            },
            prompt_tok=200,
            out_tok=40,
        )

    gemini_client = GeminiClient(
        config=cfg,
        budget=budget,
        storage=storage,
        _raw_generate=_raw_generate,
    )
    judge = GeminiSemanticJudge(gemini_client=gemini_client, config=cfg)
    tts = FakeTTS()
    dispatcher = WindowsInterventionDispatcher(tts=tts, enable_ui=False)
    screen_capture = ControlledScreenCapture(shield=shield)
    vision_sensor = FakeVisionSensor()

    engine = FocusEngine(
        config=cfg,
        event_bus=bus,
        budget=budget,
        storage=storage,
        screen_capture=screen_capture,
        semantic_judge=judge,
        coach_agent=None,
        dispatcher=dispatcher,
        vision_sensor=vision_sensor,
        clock=clock,
    )
    coach = FirmCoachAgent(
        gemini_client=gemini_client,
        engine=engine,
        tts=tts,
        config=cfg,
        event_bus=bus,
        storage=storage,
    )
    engine.set_coach_agent(coach)
    # Bypass background sleep ticker; drive engine deterministically with clock.advance + tick()
    engine._subscribe_events()  # noqa: SLF001
    await engine.start_session()

    interventions: list[InterventionCommand] = []
    suggestions: list[ProactiveSuggestionEvent] = []

    async def _on_intervention(cmd: InterventionCommand) -> None:
        interventions.append(cmd)

    async def _on_suggestion(ev: ProactiveSuggestionEvent) -> None:
        suggestions.append(ev)

    bus.subscribe(InterventionCommand, _on_intervention)
    bus.subscribe(ProactiveSuggestionEvent, _on_suggestion)

    return {
        "cfg": cfg,
        "clock": clock,
        "bus": bus,
        "budget": budget,
        "shield": shield,
        "storage": storage,
        "judge_calls": judge_calls,
        "tts": tts,
        "dispatcher": dispatcher,
        "screen_capture": screen_capture,
        "vision_sensor": vision_sensor,
        "engine": engine,
        "coach": coach,
        "interventions": interventions,
        "suggestions": suggestions,
    }


@pytest.mark.asyncio
async def test_replay_01_tec_concursos_plus_youtube_music_background(tmp_path: Path) -> None:
    def _route(prompt: str) -> dict[str, Any]:
        return {
            "verdict": "STUDYING",
            "confidence": 0.96,
            "category": "question_bank",
            "reasoning": "Resolvendo questões no Tec Concursos.",
            "suggested_message": "",
        }

    rig = await _build_replay_rig(tmp_path, _route)
    engine: FocusEngine = rig["engine"]
    clock: ReplayClock = rig["clock"]
    bus: EventBus = rig["bus"]

    try:
        await bus.publish(
            OSContextEvent(
                timestamp=clock.now,
                process_name="chrome.exe",
                window_title="Tec Concursos - Questões de Direito Constitucional",
                browser_url="https://www.tecconcursos.com.br/questoes",
                background_audio_apps=["youtube"],
            )
        )
        await _step(engine, clock, 180, step_sec=10)

        status = engine.get_status()
        assert status.session.current_state == FocusState.FOCUSED
        assert status.session.interventions_count == 0
        assert len(rig["interventions"]) == 0
    finally:
        await rig["storage"].close()


@pytest.mark.asyncio
async def test_replay_02_quick_youtube_song_switch_grace_window(tmp_path: Path) -> None:
    rig = await _build_replay_rig(
        tmp_path,
        lambda _p: {
            "verdict": "STUDYING",
            "confidence": 0.95,
            "category": "study",
            "reasoning": "Estudo ativo.",
            "suggested_message": "",
        },
    )
    engine: FocusEngine = rig["engine"]
    clock: ReplayClock = rig["clock"]
    bus: EventBus = rig["bus"]

    try:
        # 1. Focused on PDF
        await bus.publish(
            OSContextEvent(
                timestamp=clock.now,
                process_name="msedge.exe",
                window_title="Direito Constitucional - PDF Esquematizado",
            )
        )
        await _step(engine, clock, 20, step_sec=5)
        assert engine.get_status().session.current_state == FocusState.FOCUSED

        calls_before_yt = len(rig["judge_calls"])

        # 2. Switch to YouTube for 25s to change song -> GRACE_WINDOW, 0 judge calls
        await bus.publish(
            OSContextEvent(
                timestamp=clock.now,
                process_name="chrome.exe",
                window_title="Lo-fi Hip Hop Radio - YouTube",
                browser_url="https://www.youtube.com/watch?v=jfKfPfyJRdk",
            )
        )
        await _step(engine, clock, 25, step_sec=5)
        assert engine.get_status().session.current_state == FocusState.GRACE_WINDOW
        assert len(rig["judge_calls"]) == calls_before_yt

        # 3. Return to PDF -> back to FOCUSED, 0 interventions
        await bus.publish(
            OSContextEvent(
                timestamp=clock.now,
                process_name="msedge.exe",
                window_title="Direito Constitucional - PDF Esquematizado",
                background_audio_apps=["chrome.exe"],
            )
        )
        await _step(engine, clock, 15, step_sec=5)
        assert engine.get_status().session.current_state == FocusState.FOCUSED
        assert len(rig["interventions"]) == 0
    finally:
        await rig["storage"].close()


@pytest.mark.asyncio
async def test_replay_03_youtube_shorts_triggers_drift_and_intervention(tmp_path: Path) -> None:
    rig = await _build_replay_rig(tmp_path, lambda _p: {"verdict": "DISTRACTED", "confidence": 0.95})
    engine: FocusEngine = rig["engine"]
    clock: ReplayClock = rig["clock"]
    bus: EventBus = rig["bus"]

    try:
        await bus.publish(
            OSContextEvent(
                timestamp=clock.now,
                process_name="chrome.exe",
                window_title="Shorts - YouTube",
                browser_url="https://www.youtube.com/shorts/abc123",
            )
        )
        # Immediate drift suspicion, then >= 45s triggers Level 1 intervention
        assert engine.get_status().session.current_state == FocusState.DRIFT_SUSPECTED
        await _step(engine, clock, 50, step_sec=5)

        assert engine.get_status().session.current_state == FocusState.INTERVENING
        assert len(rig["interventions"]) >= 1
        assert rig["interventions"][0].level == InterventionLevel.SOFT
    finally:
        await rig["storage"].close()


@pytest.mark.asyncio
async def test_replay_04_youtube_videoaula_evaluated_once_and_cached(tmp_path: Path) -> None:
    def _route(prompt: str) -> dict[str, Any]:
        return {
            "verdict": "STUDYING",
            "confidence": 0.94,
            "category": "videoaula",
            "reasoning": "Videoaula de Direito Constitucional alinhada ao concurso.",
            "suggested_message": "",
        }

    rig = await _build_replay_rig(tmp_path, _route)
    engine: FocusEngine = rig["engine"]
    clock: ReplayClock = rig["clock"]
    bus: EventBus = rig["bus"]

    try:
        yt_lecture = OSContextEvent(
            timestamp=clock.now,
            process_name="chrome.exe",
            window_title="Controle de Constitucionalidade - Aula Completa - YouTube",
            browser_url="https://www.youtube.com/watch?v=aula_constitucional",
        )
        await bus.publish(yt_lecture)
        # First 45s in grace window, then evaluated once at >=45s
        await _step(engine, clock, 55, step_sec=5)
        assert engine.get_status().session.current_state == FocusState.FOCUSED
        assert len(rig["judge_calls"]) == 1

        # Briefly alt-tab and return to exact same lecture -> uses session cache (0 new calls)
        await bus.publish(
            OSContextEvent(
                timestamp=clock.now,
                process_name="obsidian.exe",
                window_title="Resumo Constitucional - Obsidian",
            )
        )
        await _step(engine, clock, 5, step_sec=5)
        await bus.publish(
            yt_lecture.model_copy(update={"timestamp": clock.now})
        )
        await _step(engine, clock, 20, step_sec=5)

        assert engine.get_status().session.current_state == FocusState.FOCUSED
        assert len(rig["judge_calls"]) == 1
        assert len(rig["interventions"]) == 0
    finally:
        await rig["storage"].close()


@pytest.mark.asyncio
async def test_replay_05_multipurpose_apps_zero_prison_deduction(tmp_path: Path) -> None:
    def _route(prompt: str) -> dict[str, Any]:
        if "jogos" in prompt.lower() or "steam" in prompt.lower() or "futebol" in prompt.lower():
            return {
                "verdict": "DISTRACTED",
                "confidence": 0.92,
                "category": "entertainment",
                "reasoning": "Pedro abriu conteúdo de futebol fora do estudo.",
                "suggested_message": "Pedro, futebol não cai na prova. Fecha isso.",
            }
        return {
            "verdict": "STUDYING",
            "confidence": 0.93,
            "category": "study_tool",
            "reasoning": "Uso produtivo de Obsidian / Antigravity / NotebookLM.",
            "suggested_message": "",
        }

    rig = await _build_replay_rig(tmp_path, _route)
    engine: FocusEngine = rig["engine"]
    clock: ReplayClock = rig["clock"]
    bus: EventBus = rig["bus"]

    try:
        # Antigravity + Obsidian + NotebookLM should all be allowed when genuinely studying
        for proc, title in [
            ("antigravity.exe", "Antigravity - Organizando Cronograma Edital"),
            ("obsidian.exe", "Obsidian - Direito Administrativo - Atos"),
            ("chrome.exe", "NotebookLM - Caderno de Legislação"),
        ]:
            await bus.publish(OSContextEvent(timestamp=clock.now, process_name=proc, window_title=title))
            await _step(engine, clock, 15, step_sec=5)
            assert engine.get_status().session.current_state == FocusState.FOCUSED

        # Same Chrome app used for off-topic distraction gets caught semantically!
        await bus.publish(
            OSContextEvent(
                timestamp=clock.now,
                process_name="chrome.exe",
                window_title="Melhores gols do futebol ontem",
                browser_url="https://noticias.exemplo/futebol",
            )
        )
        await _step(engine, clock, 60, step_sec=5)
        assert engine.get_status().session.current_state == FocusState.INTERVENING
        assert len(rig["interventions"]) >= 1
    finally:
        await rig["storage"].close()


@pytest.mark.asyncio
async def test_replay_06_phone_detected_on_webcam_while_pdf_open(tmp_path: Path) -> None:
    phone_visible = {"active": False}

    def _route(_p: str) -> dict[str, Any]:
        if phone_visible["active"]:
            return {
                "verdict": "DISTRACTED",
                "confidence": 0.96,
                "category": "phone",
                "reasoning": "A foto da webcam mostra o usuário segurando o celular na mão.",
                "suggested_message": "Pedro, larga o celular agora e volta pro PDF.",
                "phone_in_use": True,
                "physical_state": "using_phone",
            }
        return {
            "verdict": "STUDYING",
            "confidence": 0.95,
            "category": "pdf",
            "reasoning": "PDF aberto na tela.",
            "suggested_message": "",
            "phone_in_use": False,
            "physical_state": "focused_on_screen",
        }

    rig = await _build_replay_rig(tmp_path, _route)
    engine: FocusEngine = rig["engine"]
    clock: ReplayClock = rig["clock"]
    bus: EventBus = rig["bus"]

    try:
        await bus.publish(
            OSContextEvent(
                timestamp=clock.now,
                process_name="msedge.exe",
                window_title="Lei 8112 Comentada.pdf",
            )
        )
        await _step(engine, clock, 15, step_sec=5)
        assert engine.get_status().session.current_state == FocusState.FOCUSED

        # User stops typing (idle=20s) and Gemini visually confirms phone in hand on webcam
        phone_visible["active"] = True
        await bus.publish(
            OSContextEvent(
                timestamp=clock.now,
                process_name="msedge.exe",
                window_title="Lei 8112 Comentada.pdf",
                idle_seconds=20.0,
            )
        )
        await bus.publish(
            VisionSignalEvent(
                timestamp=clock.now,
                camera_available=True,
                face_present=True,
                looking_down_score=0.85,
            )
        )
        await _step(engine, clock, 30, step_sec=5)

        status = engine.get_status()
        assert status.session.current_state == FocusState.INTERVENING
        assert status.session.phone_alerts_count >= 1
        assert any("celular" in cmd.reason.lower() or "celular" in cmd.message.lower() for cmd in rig["interventions"])
    finally:
        await rig["storage"].close()


@pytest.mark.asyncio
async def test_replay_07_escalation_ladder_1_2_3_and_reset(tmp_path: Path) -> None:
    rig = await _build_replay_rig(
        tmp_path,
        lambda _p: {
            "verdict": "DISTRACTED",
            "confidence": 0.96,
            "category": "social",
            "reasoning": "Distração contínua em rede social.",
            "suggested_message": "Pedro, fecha essa aba agora.",
        },
    )
    engine: FocusEngine = rig["engine"]
    clock: ReplayClock = rig["clock"]
    bus: EventBus = rig["bus"]

    try:
        await bus.publish(
            OSContextEvent(
                timestamp=clock.now,
                process_name="chrome.exe",
                window_title="Instagram",
                browser_url="https://www.instagram.com/",
            )
        )
        # Level 1 at >=45s
        await _step(engine, clock, 50, step_sec=5)
        assert any(c.level == InterventionLevel.SOFT for c in rig["interventions"])

        # Level 2 at >=120s (speaks aloud!)
        await _step(engine, clock, 80, step_sec=5)
        assert any(c.level == InterventionLevel.VOICE for c in rig["interventions"])
        assert len(rig["tts"].spoken) >= 1

        # Level 3 at >=240s
        await _step(engine, clock, 120, step_sec=5)
        assert any(c.level == InterventionLevel.FIRM_OVERLAY for c in rig["interventions"])

        # Return to study -> resets to FOCUSED
        await bus.publish(
            OSContextEvent(
                timestamp=clock.now,
                process_name="chrome.exe",
                window_title="Tec Concursos - Questões",
                browser_url="https://www.tecconcursos.com.br/questoes",
            )
        )
        # Switch out of high-suspicion clears drift once debounced/verdicted
        rig["judge_calls"].clear()
        await bus.publish(
            OSContextEvent(
                timestamp=clock.now,
                process_name="obsidian.exe",
                window_title="Obsidian - Notas",
            )
        )
        # Acknowledge via UI or wait for study verdict
        await engine.acknowledge_back_to_focus()
        assert engine.get_status().session.current_state == FocusState.FOCUSED
        assert engine.get_status().session.escalation_level == InterventionLevel.NONE
    finally:
        await rig["storage"].close()


@pytest.mark.asyncio
async def test_replay_08_drowsiness_perclos_emits_fatigue_suggestion(tmp_path: Path) -> None:
    rig = await _build_replay_rig(
        tmp_path,
        lambda _p: {
            "verdict": "STUDYING",
            "confidence": 0.9,
            "category": "pdf",
            "reasoning": "Estudando.",
            "suggested_message": "",
        },
    )
    engine: FocusEngine = rig["engine"]
    clock: ReplayClock = rig["clock"]
    bus: EventBus = rig["bus"]

    try:
        await bus.publish(
            OSContextEvent(
                timestamp=clock.now,
                process_name="msedge.exe",
                window_title="Direito Tributário - PDF",
            )
        )
        await bus.publish(
            VisionSignalEvent(
                timestamp=clock.now,
                camera_available=True,
                face_present=True,
                drowsiness_perclos=0.65,
                yawn_score=0.80,
            )
        )
        await _step(engine, clock, 30, step_sec=5)

        assert any(s.kind == "fatigue" and s.speak_aloud for s in rig["suggestions"])
        assert engine.get_status().session.drowsiness_alerts_count >= 1
    finally:
        await rig["storage"].close()


@pytest.mark.asyncio
async def test_replay_09_sensitive_bank_window_blocks_capture_and_cloud(tmp_path: Path) -> None:
    rig = await _build_replay_rig(tmp_path, lambda _p: {"verdict": "STUDYING", "confidence": 0.9})
    engine: FocusEngine = rig["engine"]
    clock: ReplayClock = rig["clock"]
    bus: EventBus = rig["bus"]
    shield: PrivacyShield = rig["shield"]
    capture: ControlledScreenCapture = rig["screen_capture"]

    try:
        sensitive, reason = shield.is_sensitive("chrome.exe", "Nubank - Internet Banking")
        assert sensitive is True

        await bus.publish(
            OSContextEvent(
                timestamp=clock.now,
                process_name="chrome.exe",
                window_title="Nubank - Internet Banking",
                is_sensitive_window=sensitive,
                sensitive_reason=reason,
            )
        )
        await _step(engine, clock, 45, step_sec=5)

        assert engine.capture_current_screen() is None
        assert capture.capture_count == 0
        assert len(rig["judge_calls"]) == 0
        assert len(rig["interventions"]) == 0
    finally:
        await rig["storage"].close()


@pytest.mark.asyncio
async def test_replay_10_temporary_allowance_granted_by_coach(tmp_path: Path) -> None:
    rig = await _build_replay_rig(tmp_path, lambda _p: {"verdict": "DISTRACTED", "confidence": 0.9})
    engine: FocusEngine = rig["engine"]
    clock: ReplayClock = rig["clock"]
    bus: EventBus = rig["bus"]

    try:
        await engine.grant_allowance("telegram", minutes=5, reason="Baixando PDF do cursinho")
        await bus.publish(
            OSContextEvent(
                timestamp=clock.now,
                process_name="telegram.exe",
                window_title="Telegram Desktop - Grupo Concurso",
            )
        )
        await _step(engine, clock, 90, step_sec=10)

        assert engine.get_status().session.current_state == FocusState.FOCUSED
        assert len(rig["interventions"]) == 0
    finally:
        await rig["storage"].close()


@pytest.mark.asyncio
async def test_replay_11_lateral_webcam_profile_is_focused_not_penalized(tmp_path: Path) -> None:
    rig = await _build_replay_rig(
        tmp_path,
        lambda _p: {
            "verdict": "STUDYING",
            "confidence": 0.95,
            "category": "study",
            "reasoning": "Estudando no monitor principal com webcam lateral.",
            "suggested_message": "",
            "phone_in_use": False,
            "physical_state": "focused_on_screen",
        },
    )
    engine: FocusEngine = rig["engine"]
    clock: ReplayClock = rig["clock"]
    bus: EventBus = rig["bus"]

    try:
        await bus.publish(
            OSContextEvent(
                timestamp=clock.now,
                process_name="chrome.exe",
                window_title="Tec Concursos - Questões",
                active_monitor_label="M1-Principal",
            )
        )
        # Sideways yaw (-38 deg toward primary LG monitor from right-side notebook cam)
        await bus.publish(
            VisionSignalEvent(
                timestamp=clock.now,
                camera_available=True,
                face_present=True,
                head_yaw_deg=-38.0,
                head_pitch_deg=2.0,
                gaze_on_screen_score=0.96,
                looking_down_score=0.05,
                camera_position="right",
                calibrated=True,
                looking_at_screen=True,
            )
        )
        await _step(engine, clock, 60, step_sec=5)

        assert engine.get_status().session.current_state == FocusState.FOCUSED
        assert len(rig["interventions"]) == 0
        assert any("lateral" in call.lower() or "looking_at_screen=true" in call.lower() for call in rig["judge_calls"])
    finally:
        await rig["storage"].close()


@pytest.mark.asyncio
async def test_replay_12_phone_leaky_bucket_and_express_spoken_intervention(tmp_path: Path) -> None:
    rig = await _build_replay_rig(
        tmp_path,
        lambda _p: {
            "verdict": "STUDYING",
            "confidence": 0.92,
            "category": "pdf",
            "reasoning": "PDF aberto na tela.",
            "suggested_message": "",
        },
    )
    rig["cfg"].behavior.require_ai_phone_confirmation = False
    rig["cfg"].behavior.allow_looking_down_for_notes = False
    engine: FocusEngine = rig["engine"]
    clock: ReplayClock = rig["clock"]
    bus: EventBus = rig["bus"]

    try:
        await bus.publish(
            OSContextEvent(
                timestamp=clock.now,
                process_name="msedge.exe",
                window_title="Direito Penal - PDF.pdf",
                idle_seconds=20.0,
            )
        )
        await _step(engine, clock, 12, step_sec=4)

        # Phone detected for 12s, flickers off for 2s (leaky bucket only decays 1s), then 6s more
        await bus.publish(
            VisionSignalEvent(
                timestamp=clock.now,
                camera_available=True,
                face_present=True,
                looking_at_screen=False,
                looking_down_score=0.75,
                phone_detected=True,
                phone_confidence=0.65,
            )
        )
        await _step(engine, clock, 12, step_sec=3)
        await bus.publish(
            VisionSignalEvent(
                timestamp=clock.now,
                camera_available=True,
                face_present=True,
                looking_at_screen=True,
                looking_down_score=0.10,
                phone_detected=False,
                phone_confidence=0.0,
            )
        )
        await _step(engine, clock, 2, step_sec=2)
        await bus.publish(
            VisionSignalEvent(
                timestamp=clock.now,
                camera_available=True,
                face_present=True,
                looking_at_screen=False,
                looking_down_score=0.80,
                phone_detected=True,
                phone_confidence=0.72,
            )
        )
        await _step(engine, clock, 6, step_sec=2)

        assert engine.get_status().session.current_state == FocusState.INTERVENING
        assert len(rig["interventions"]) >= 1
        assert rig["interventions"][0].speak_aloud is True
        assert len(rig["tts"].spoken) >= 1
    finally:
        await rig["storage"].close()


@pytest.mark.asyncio
async def test_replay_13_second_monitor_distraction_and_visible_windows_caught(tmp_path: Path) -> None:
    def _route(prompt: str) -> dict[str, Any]:
        if "youtube" in prompt.lower() or "futebol" in prompt.lower() or "m2-secundario" in prompt.lower():
            return {
                "verdict": "DISTRACTED",
                "confidence": 0.94,
                "category": "second_monitor_video",
                "reasoning": "Usuário clicou no Obsidian na M1, mas assiste futebol na M2 com teclado parado.",
                "suggested_message": "Pedro, você deixou o Obsidian clicado mas tá vendo futebol no segundo monitor. Fecha isso agora.",
                "phone_in_use": False,
                "physical_state": "focused_on_screen",
            }
        return {
            "verdict": "STUDYING",
            "confidence": 0.95,
            "category": "study",
            "reasoning": "Estudo ativo.",
            "suggested_message": "",
        }

    rig = await _build_replay_rig(tmp_path, _route)
    engine: FocusEngine = rig["engine"]
    clock: ReplayClock = rig["clock"]
    bus: EventBus = rig["bus"]

    try:
        await bus.publish(
            OSContextEvent(
                timestamp=clock.now,
                process_name="obsidian.exe",
                window_title="Obsidian - Direito Administrativo",
                active_monitor_label="M1-Principal",
                background_audio_apps=["chrome.exe"],
                idle_seconds=35.0,
                visible_windows=[
                    VisibleWindowInfo(
                        hwnd=1,
                        process_name="obsidian.exe",
                        window_title="Obsidian - Direito Administrativo",
                        monitor_label="M1-Principal",
                        screen_area_pct=60.0,
                        is_foreground=True,
                    ),
                    VisibleWindowInfo(
                        hwnd=2,
                        process_name="chrome.exe",
                        window_title="Melhores Momentos Futebol - YouTube",
                        monitor_label="M2-Secundario",
                        screen_area_pct=85.0,
                        is_foreground=False,
                    ),
                ],
                usage_pattern=UsagePatternMetrics(
                    active_input_ratio_60s=0.05,
                    window_switches_60s=1,
                    continuous_same_screen_sec=35.0,
                    interaction_mode="idle_with_media",
                ),
            )
        )
        await _step(engine, clock, 60, step_sec=5)

        assert engine.get_status().session.current_state == FocusState.INTERVENING
        assert len(rig["interventions"]) >= 1
        assert rig["interventions"][0].speak_aloud is True
        assert len(rig["tts"].spoken) >= 1
        assert any("M2-Secundario" in call for call in rig["judge_calls"])
    finally:
        await rig["storage"].close()


@pytest.mark.asyncio
async def test_replay_14_camera_not_seeing_user_while_mouse_active_speaks_immediately(
    tmp_path: Path,
) -> None:
    rig = await _build_replay_rig(
        tmp_path,
        lambda _p: {
            "foreground_topic": "Tec Concursos",
            "matches_current_subject": True,
            "secondary_monitor_contamination": False,
            "is_foreground_music_only": False,
            "user_visible_in_webcam": True,
            "verdict": "STUDYING",
            "confidence": 0.94,
            "category": "study",
            "reasoning": "Questões abertas na tela.",
            "suggested_message": "",
        },
    )
    engine: FocusEngine = rig["engine"]
    clock: ReplayClock = rig["clock"]
    bus: EventBus = rig["bus"]

    try:
        # User actively clicking/typing in study app (idle=1.5s), but webcam stops seeing user
        await bus.publish(
            OSContextEvent(
                timestamp=clock.now,
                process_name="chrome.exe",
                window_title="Tec Concursos - Questões",
                idle_seconds=1.5,
            )
        )
        await bus.publish(
            VisionSignalEvent(
                timestamp=clock.now,
                camera_available=True,
                face_present=False,
                looking_at_screen=False,
            )
        )
        await _step(engine, clock, 20, step_sec=5)

        status = engine.get_status()
        assert status.session.current_state == FocusState.INTERVENING
        assert status.session.camera_unseen_seconds >= 20.0
        assert len(rig["interventions"]) >= 1
        assert rig["interventions"][0].speak_aloud is True
        assert len(rig["tts"].spoken) >= 1
    finally:
        await rig["storage"].close()
