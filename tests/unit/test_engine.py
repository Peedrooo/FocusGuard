"""Comprehensive unit tests for SQLiteStorage and FocusEngine."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import aiosqlite
import pytest

from focus_guard.core.budget import BudgetGuard
from focus_guard.core.config import AIConfig, AppConfig, BehaviorConfig, HintsConfig, PrivacyConfig
from focus_guard.core.event_bus import EventBus
from focus_guard.core.events import (
    ActivityVerdict,
    ChatMessage,
    FocusState,
    FocusStateChangedEvent,
    InterventionCommand,
    InterventionLevel,
    OSContextEvent,
    ProactiveSuggestionEvent,
    SemanticVerdict,
    SessionStats,
    StudyNote,
    TemporaryAllowance,
    VisionSignalEvent,
)
from focus_guard.core.ports import FocusEnginePort, StoragePort
from focus_guard.engine import FocusEngine, SQLiteStorage
from tests.fakes import (
    FakeCoachAgent,
    FakeInterventionDispatcher,
    FakeScreenCapture,
    FakeSemanticJudge,
    FakeStorage,
    FakeVisionSensor,
)


class FakeClock:
    """Controllable UTC clock for deterministic engine & storage tests."""

    def __init__(self, initial: datetime | None = None) -> None:
        self.now = initial or datetime(2026, 11, 19, 12, 0, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> datetime:
        self.now = self.now + timedelta(seconds=seconds)
        return self.now


def _make_config() -> AppConfig:
    return AppConfig(
        user_name="Pedro",
        study_goal="Concurso Auditor Fiscal",
        current_subject="Direito Constitucional",
        ai=AIConfig(
            max_daily_usd=0.35,
            max_monthly_usd=8.00,
            focused_recheck_interval_sec=90,
            ambiguous_recheck_interval_sec=25,
            window_switch_debounce_sec=10,
        ),
        behavior=BehaviorConfig(
            youtube_music_grace_sec=45,
            distraction_level1_sec=45,
            distraction_level2_sec=120,
            distraction_level3_sec=240,
            intervention_cooldown_sec=75,
            absence_alert_sec=120,
            phone_distraction_sec=25,
            drowsiness_perclos_threshold=0.45,
            drowsiness_window_sec=25,
            pomodoro_focus_suggestion_min=50,
        ),
        hints=HintsConfig(
            likely_study_keywords=["tecconcursos", "tec concursos", "constituicao", "direito"],
            context_dependent_apps=["obsidian", "antigravity", "chrome", "youtube"],
            high_suspicion_patterns=["youtube.com/shorts", "instagram.com", "tiktok.com"],
        ),
        privacy=PrivacyConfig(
            blocked_process_names=["bitwarden.exe", "keepass.exe"],
            blocked_title_keywords=["nubank", "inprivate", "anonima"],
        ),
    )


def _make_engine(
    clock: FakeClock | None = None,
    config: AppConfig | None = None,
    judge: FakeSemanticJudge | None = None,
    capture: FakeScreenCapture | None = None,
    dispatcher: FakeInterventionDispatcher | None = None,
    storage: StoragePort | None = None,
    coach: FakeCoachAgent | None = None,
    vision_sensor: FakeVisionSensor | None = None,
) -> tuple[
    FocusEngine,
    FakeClock,
    EventBus,
    BudgetGuard,
    StoragePort,
    FakeScreenCapture,
    FakeSemanticJudge,
    FakeInterventionDispatcher,
]:
    clk = clock or FakeClock()
    cfg = config or _make_config()
    bus = EventBus()
    budget = BudgetGuard(cfg.ai)
    st = storage or FakeStorage()
    cap = capture or FakeScreenCapture()
    jdg = judge or FakeSemanticJudge(default_verdict=ActivityVerdict.STUDYING)
    disp = dispatcher or FakeInterventionDispatcher()
    eng = FocusEngine(
        config=cfg,
        event_bus=bus,
        budget=budget,
        storage=st,
        screen_capture=cap,
        semantic_judge=jdg,
        coach_agent=coach,
        dispatcher=disp,
        clock=clk,
        vision_sensor=vision_sensor,
    )
    return eng, clk, bus, budget, st, cap, jdg, disp


# ----------------------------------------------------------------------
# Protocol Compliance
# ----------------------------------------------------------------------


def test_real_implementations_satisfy_protocols(tmp_path: Path) -> None:
    storage = SQLiteStorage(tmp_path / "proto.db")
    engine, *_ = _make_engine()
    assert isinstance(storage, StoragePort)
    assert isinstance(engine, FocusEnginePort)


# ----------------------------------------------------------------------
# SQLiteStorage Tests
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sqlite_storage_init_wal_and_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "nested" / "dir" / "focus.db"
    storage = SQLiteStorage(db_path)
    await storage.init()
    try:
        assert db_path.exists()
        async with aiosqlite.connect(db_path) as conn:
            async with conn.execute("PRAGMA journal_mode;") as cur:
                mode_row = await cur.fetchone()
                assert mode_row is not None
                assert str(mode_row[0]).lower() == "wal"

            async with conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table';"
            ) as cur:
                tables = {row[0] for row in await cur.fetchall()}

        expected_tables = {
            "sessions",
            "verdicts",
            "interventions",
            "api_usage",
            "chat_messages",
            "study_notes",
            "allowances",
        }
        assert expected_tables.issubset(tables)
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_sqlite_storage_crud_and_usage_aggregation(tmp_path: Path) -> None:
    clock = FakeClock(datetime(2026, 11, 19, 10, 0, 0, tzinfo=UTC))
    storage = SQLiteStorage(tmp_path / "crud.db", clock=clock)
    await storage.init()
    try:
        # Session upsert
        stats = SessionStats(
            session_id="sess-1",
            study_goal="Receita Federal",
            current_subject="Tributário",
            current_state=FocusState.FOCUSED,
            focus_seconds=120.0,
        )
        await storage.save_session(stats)
        stats.focus_seconds = 300.0
        stats.interventions_count = 2
        await storage.save_session(stats)

        # Verdict & Intervention
        verdict = SemanticVerdict(
            verdict=ActivityVerdict.STUDYING,
            confidence=0.95,
            category="questoes",
            app_context="Tec Concursos",
            reasoning="Resolvendo questões de direito",
            used_vision=True,
            model_used="gemini-3.5-flash-lite",
            tokens_in=400,
            tokens_out=60,
            estimated_cost_usd=0.00027,
        )
        await storage.save_verdict("sess-1", verdict)

        cmd = InterventionCommand(
            level=InterventionLevel.SOFT,
            title="Alerta",
            message="Volta pro foco!",
            speak_aloud=False,
            reason="Instagram",
        )
        await storage.save_intervention("sess-1", cmd)

        # Chat messages (order & limit)
        m1 = ChatMessage(
            timestamp=clock(),
            role="user",
            text="Posso pausar?",
        )
        clock.advance(5)
        m2 = ChatMessage(
            timestamp=clock(),
            role="coach",
            text="Termina mais 10 minutos primeiro.",
            tool_calls=["add_note"],
            cost_usd=0.001,
        )
        clock.advance(5)
        m3 = ChatMessage(
            timestamp=clock(),
            role="user",
            text="Fechado.",
        )
        await storage.save_chat_message("sess-1", m1)
        await storage.save_chat_message("sess-1", m2)
        await storage.save_chat_message("sess-1", m3)

        recent_chats = await storage.list_chat_messages(limit=2)
        assert [c.text for c in recent_chats] == ["Termina mais 10 minutos primeiro.", "Fechado."]
        assert recent_chats[0].tool_calls == ["add_note"]
        assert recent_chats[0].cost_usd == pytest.approx(0.001)

        # Study notes
        n1 = StudyNote(timestamp=clock(), subject="Tributário", note="Art. 150 CF")
        clock.advance(5)
        n2 = StudyNote(timestamp=clock(), subject="Constitucional", note="Art. 5º CF")
        await storage.save_note(n1)
        await storage.save_note(n2)
        notes = await storage.list_notes(limit=10)
        assert [n.note for n in notes] == ["Art. 150 CF", "Art. 5º CF"]

        # Allowances (active vs expired)
        active_allow = TemporaryAllowance(
            pattern="youtube",
            reason="Aula extra",
            created_at=clock(),
            expires_at=clock() + timedelta(minutes=15),
        )
        expired_allow = TemporaryAllowance(
            pattern="twitter",
            reason="Expirada",
            created_at=clock() - timedelta(minutes=30),
            expires_at=clock() - timedelta(minutes=5),
        )
        await storage.save_allowance(active_allow)
        await storage.save_allowance(expired_allow)
        active_list = await storage.list_active_allowances()
        assert len(active_list) == 1
        assert active_list[0].pattern == "youtube"

        # Usage totals aggregated by UTC day & UTC month
        # Previous month (Oct 2026)
        clock.now = datetime(2026, 10, 31, 23, 0, 0, tzinfo=UTC)
        await storage.save_usage("gemini-3.5-flash-lite", 1000, 200, 0.05)

        # Same month, previous day (Nov 18 2026)
        clock.now = datetime(2026, 11, 18, 20, 0, 0, tzinfo=UTC)
        await storage.save_usage("gemini-3.5-flash-lite", 2000, 300, 0.10)

        # Today (Nov 19 2026) - two calls
        clock.now = datetime(2026, 11, 19, 10, 0, 0, tzinfo=UTC)
        await storage.save_usage("gemini-3.5-flash-lite", 500, 100, 0.02)
        await storage.save_usage("gemini-3.5-flash-lite", 700, 150, 0.03)

        daily_usd, monthly_usd, daily_in, daily_out, daily_calls = (
            await storage.get_usage_totals()
        )
        assert daily_usd == pytest.approx(0.05)
        assert monthly_usd == pytest.approx(0.15)
        assert daily_in == 1200
        assert daily_out == 250
        assert daily_calls == 2
    finally:
        await storage.close()


# ----------------------------------------------------------------------
# YouTube Paradox Tests
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_youtube_background_music_focused_zero_calls() -> None:
    engine, clock, _, _, _, cap, judge, _ = _make_engine()
    await engine.start_session()

    await engine.handle_os_event(
        OSContextEvent(
            process_name="obsidian.exe",
            window_title="Direito Administrativo - Resumo.md",
            background_audio_apps=["chrome.exe - YouTube"],
        )
    )
    for _ in range(30):
        clock.advance(1)
        await engine.tick()

    status = engine.get_status()
    assert status.session.current_state == FocusState.FOCUSED
    assert status.session.focus_seconds == pytest.approx(30.0)
    assert status.session.distraction_seconds == pytest.approx(0.0)
    assert len(judge.calls) == 0
    assert cap.calls == 0


@pytest.mark.asyncio
async def test_youtube_quick_song_switch_grace_then_focused_zero_calls() -> None:
    engine, clock, bus, _, _, _, judge, _ = _make_engine()
    state_events: list[FocusStateChangedEvent] = []
    bus.subscribe(FocusStateChangedEvent, state_events.append)

    await engine.start_session()
    await engine.handle_os_event(
        OSContextEvent(
            process_name="chrome.exe",
            window_title="YouTube - Lofi Beats Study Radio",
            browser_url="https://www.youtube.com/watch?v=jfKfPfyJRdk",
        )
    )
    assert engine.get_status().session.current_state == FocusState.GRACE_WINDOW

    clock.advance(25)
    await engine.tick()

    status_during = engine.get_status()
    assert status_during.session.current_state == FocusState.GRACE_WINDOW
    assert status_during.session.grace_seconds == pytest.approx(25.0)
    assert status_during.session.distraction_seconds == pytest.approx(0.0)
    assert len(judge.calls) == 0

    await engine.handle_os_event(
        OSContextEvent(
            process_name="chrome.exe",
            window_title="Tec Concursos - Questões",
            browser_url="https://www.tecconcursos.com.br/questoes",
        )
    )
    status_after = engine.get_status()
    assert status_after.session.current_state == FocusState.FOCUSED
    assert len(judge.calls) == 0
    assert any(
        ev.previous_state == FocusState.GRACE_WINDOW and ev.new_state == FocusState.FOCUSED
        for ev in state_events
    )


@pytest.mark.asyncio
async def test_youtube_shorts_immediate_drift_and_intervenes() -> None:
    engine, clock, _, _, _, _, judge, dispatcher = _make_engine()
    await engine.start_session()

    await engine.handle_os_event(
        OSContextEvent(
            process_name="chrome.exe",
            window_title="Shorts - YouTube",
            browser_url="https://www.youtube.com/shorts/dQw4w9WgXcQ",
        )
    )
    # No 45s grace window for Shorts!
    assert engine.get_status().session.current_state == FocusState.DRIFT_SUSPECTED
    assert len(judge.calls) == 0

    clock.advance(45)
    await engine.tick()

    status = engine.get_status()
    assert status.session.current_state == FocusState.INTERVENING
    assert status.session.escalation_level == InterventionLevel.SOFT
    assert status.session.distraction_seconds == pytest.approx(45.0)
    assert len(dispatcher.dispatched) == 1
    assert dispatcher.dispatched[0].level == InterventionLevel.SOFT
    assert dispatcher.dispatched[0].speak_aloud is True


@pytest.mark.asyncio
async def test_youtube_videoaula_caching_one_judge_call_then_cached() -> None:
    judge = FakeSemanticJudge(default_verdict=ActivityVerdict.STUDYING)
    engine, clock, _, _, _, _, _, _ = _make_engine(judge=judge)
    await engine.start_session()

    videoaula = OSContextEvent(
        process_name="chrome.exe",
        window_title="Aula 01 - Direito Constitucional - YouTube",
        browser_url="https://www.youtube.com/watch?v=aula_const_01",
    )
    await engine.handle_os_event(videoaula)
    assert engine.get_status().session.current_state == FocusState.GRACE_WINDOW

    # Advance 44s: still in grace, 0 calls
    clock.advance(44)
    await engine.tick()
    assert len(judge.calls) == 0

    # Cross 45s mark: triggers SemanticJudge with vision
    clock.advance(1)
    await engine.tick()
    assert len(judge.calls) == 1
    assert judge.calls[0]["used_vision"] is True
    assert engine.get_status().session.current_state == FocusState.FOCUSED

    # Switch to another app briefly (with background audio) and return to same videoaula
    await engine.handle_os_event(
        OSContextEvent(
            process_name="obsidian.exe",
            window_title="Caderno",
            background_audio_apps=["youtube"],
        )
    )
    clock.advance(5)
    await engine.tick()

    # Return to the exact same videoaula -> hits cache immediately, 0 extra judge calls!
    await engine.handle_os_event(videoaula)
    assert engine.get_status().session.current_state == FocusState.FOCUSED
    clock.advance(30)
    await engine.tick()
    assert len(judge.calls) == 1


# ----------------------------------------------------------------------
# Multi-Purpose Apps & Debounce Tests
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multipurpose_apps_semantic_evaluation_after_debounce() -> None:
    judge = FakeSemanticJudge(default_verdict=ActivityVerdict.STUDYING)
    engine, clock, _, _, _, _, _, _ = _make_engine(judge=judge)
    await engine.start_session()

    antigravity = OSContextEvent(
        process_name="antigravity.exe",
        window_title="Monitor - Antigravity",
    )
    await engine.handle_os_event(antigravity)

    # 9s < 10s debounce -> not evaluated yet
    clock.advance(9)
    await engine.tick()
    assert len(judge.calls) == 0

    # Quick switch to Obsidian before 10s resets debounce without calling judge on Antigravity
    obsidian = OSContextEvent(
        process_name="obsidian.exe",
        window_title="Resumo Tributário - Obsidian",
    )
    await engine.handle_os_event(obsidian)
    clock.advance(9)
    await engine.tick()
    assert len(judge.calls) == 0

    # At 10s stable on Obsidian -> evaluated once with screenshot
    clock.advance(1)
    await engine.tick()
    assert len(judge.calls) == 1
    assert judge.calls[0]["os_context"].process_name == "obsidian.exe"
    assert judge.calls[0]["used_vision"] is True
    assert engine.get_status().session.current_state == FocusState.FOCUSED

    # Switch to Tec Concursos -> judge returns DISTRACTED (e.g. off-topic forum) after 10s
    judge.scripted_verdicts.append(
        SemanticVerdict(
            verdict=ActivityVerdict.DISTRACTED,
            confidence=0.92,
            app_context="Fórum off-topic",
            reasoning="Lendo fofoca no fórum do Tec Concursos",
            suggested_message="Fecha o fórum e vai fazer questão!",
            used_vision=True,
            model_used="fake-gemini",
            tokens_in=300,
            tokens_out=50,
            estimated_cost_usd=0.0002,
        )
    )
    await engine.handle_os_event(
        OSContextEvent(
            process_name="chrome.exe",
            window_title="Tec Concursos - Comunidade",
            browser_url="https://www.tecconcursos.com.br/forum/offtopic",
        )
    )
    clock.advance(10)
    await engine.tick()
    assert len(judge.calls) == 2
    assert engine.get_status().session.current_state == FocusState.DRIFT_SUSPECTED


# ----------------------------------------------------------------------
# Escalation Ladder & Overlay Dismissal
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_escalation_ladder_levels_1_2_3_and_dismiss_on_return() -> None:
    coach = FakeCoachAgent(configured=True)
    judge = FakeSemanticJudge(default_verdict=ActivityVerdict.STUDYING)
    dispatcher = FakeInterventionDispatcher()
    fake_storage = FakeStorage()
    engine, clock, bus, _, _, _, _, _ = _make_engine(
        judge=judge, dispatcher=dispatcher, storage=fake_storage, coach=coach
    )

    commands: list[InterventionCommand] = []
    bus.subscribe(InterventionCommand, commands.append)

    await engine.start_session()

    # Cache Tec Concursos as STUDYING first
    tec = OSContextEvent(
        process_name="chrome.exe",
        window_title="Tec Concursos - Direito",
        browser_url="https://www.tecconcursos.com.br/questoes",
    )
    await engine.handle_os_event(tec)
    clock.advance(10)
    await engine.tick()
    assert engine.get_status().session.current_state == FocusState.FOCUSED

    # Switch to high-suspicion Instagram
    await engine.handle_os_event(
        OSContextEvent(
            process_name="chrome.exe",
            window_title="Instagram",
            browser_url="https://www.instagram.com/reels/",
        )
    )
    assert engine.get_status().session.current_state == FocusState.DRIFT_SUSPECTED

    # Level 1 at 45s
    clock.advance(45)
    await engine.tick()
    assert engine.get_status().session.current_state == FocusState.INTERVENING
    assert engine.get_status().session.escalation_level == InterventionLevel.SOFT
    assert len(dispatcher.dispatched) == 1
    assert dispatcher.dispatched[0].level == InterventionLevel.SOFT
    assert dispatcher.dispatched[0].speak_aloud is True
    assert "Pedro, para de enrolar" in dispatcher.dispatched[0].message

    # Level 2 at 120s (+75s)
    clock.advance(75)
    await engine.tick()
    assert engine.get_status().session.escalation_level == InterventionLevel.VOICE
    assert len(dispatcher.dispatched) == 2
    assert dispatcher.dispatched[1].level == InterventionLevel.VOICE
    assert dispatcher.dispatched[1].speak_aloud is True

    # Level 3 at 240s (+120s)
    clock.advance(120)
    await engine.tick()
    assert engine.get_status().session.escalation_level == InterventionLevel.FIRM_OVERLAY
    assert len(dispatcher.dispatched) == 3
    assert dispatcher.dispatched[2].level == InterventionLevel.FIRM_OVERLAY
    assert dispatcher.dispatched[2].speak_aloud is True
    assert len(commands) == 3
    assert len(fake_storage.interventions) == 3

    # Cooldown prevents duplicate Level 3 before 75s, then allows reminder at >= 75s
    clock.advance(50)
    await engine.tick()
    assert len(dispatcher.dispatched) == 3
    clock.advance(25)
    await engine.tick()
    assert len(dispatcher.dispatched) == 4

    # Return to cached Tec Concursos -> immediately clears drift & dismisses overlay
    await engine.handle_os_event(tec)
    status = engine.get_status()
    assert status.session.current_state == FocusState.FOCUSED
    assert status.session.escalation_level == InterventionLevel.NONE
    assert dispatcher.overlay_dismissed == 1


# ----------------------------------------------------------------------
# Vision Signals: Phone, Absence, Fatigue, Missing Hardware
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_phone_detection_distraction_and_recovery() -> None:
    cfg = _make_config()
    cfg.behavior.require_ai_phone_confirmation = False
    cfg.behavior.allow_looking_down_for_notes = False
    engine, clock, _, _, _, _, _, dispatcher = _make_engine(config=cfg)
    await engine.start_session()

    await engine.handle_vision_event(VisionSignalEvent(phone_detected=True, phone_confidence=0.9))

    # Below 25s threshold -> still FOCUSED
    clock.advance(20)
    await engine.tick()
    assert engine.get_status().session.current_state == FocusState.FOCUSED
    assert engine.get_status().session.phone_alerts_count == 0

    # Cross 25s -> express spoken phone lane escalates straight to INTERVENING Level 1
    clock.advance(5)
    await engine.tick()
    status = engine.get_status()
    assert status.session.current_state == FocusState.INTERVENING
    assert status.session.escalation_level == InterventionLevel.SOFT
    assert len(dispatcher.dispatched) == 1
    assert dispatcher.dispatched[0].speak_aloud is True
    assert "Uso de celular detectado pela câmera" in dispatcher.dispatched[0].reason
    assert status.session.phone_alerts_count == 1

    # Put phone down -> returns to FOCUSED & dismisses overlay
    await engine.handle_vision_event(VisionSignalEvent(phone_detected=False, face_present=True))
    await engine.tick()
    assert engine.get_status().session.current_state == FocusState.FOCUSED
    assert dispatcher.overlay_dismissed == 1


@pytest.mark.asyncio
async def test_fatigue_suggestion_and_missing_camera_resilience() -> None:
    engine, clock, bus, _, _, _, _, _ = _make_engine()
    suggestions: list[ProactiveSuggestionEvent] = []
    bus.subscribe(ProactiveSuggestionEvent, suggestions.append)

    await engine.start_session()

    # Missing camera must never cause absence drift
    await engine.handle_vision_event(
        VisionSignalEvent(camera_available=False, face_present=False)
    )
    clock.advance(200)
    await engine.tick()
    assert engine.get_status().session.current_state == FocusState.FOCUSED
    assert engine.get_status().camera_available is False

    # Drowsiness sustained for 25s -> fatigue ProactiveSuggestionEvent
    await engine.handle_vision_event(
        VisionSignalEvent(camera_available=True, face_present=True, drowsiness_perclos=0.60)
    )
    clock.advance(25)
    await engine.tick()

    fatigue_events = [s for s in suggestions if s.kind == "fatigue"]
    assert len(fatigue_events) == 1
    assert fatigue_events[0].speak_aloud is True
    assert engine.get_status().session.drowsiness_alerts_count == 1


# ----------------------------------------------------------------------
# Privacy Mode, Sensitive Windows, Allowances, Breaks, Budget
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_privacy_mode_and_sensitive_window_protection() -> None:
    engine, clock, _, _, _, cap, judge, _ = _make_engine()
    await engine.start_session()

    # Privacy Mode blocks capture & judge even on Shorts
    assert await engine.toggle_privacy(True) is True
    assert engine.get_status().session.current_state == FocusState.PRIVACY_MODE
    assert engine.capture_current_screen() is None

    await engine.handle_os_event(
        OSContextEvent(
            process_name="chrome.exe",
            window_title="Shorts - YouTube",
            browser_url="https://www.youtube.com/shorts/xyz",
        )
    )
    clock.advance(60)
    await engine.tick()
    assert cap.calls == 0
    assert len(judge.calls) == 0
    assert engine.get_status().session.distraction_seconds == pytest.approx(0.0)

    assert await engine.toggle_privacy(False) is False
    assert engine.get_status().session.current_state == FocusState.FOCUSED

    # Sensitive window (explicit flag OR PrivacyShield match) blocks capture & pauses drift
    await engine.handle_os_event(
        OSContextEvent(
            process_name="bitwarden.exe",
            window_title="Cofre de Senhas",
            is_sensitive_window=True,
        )
    )
    assert engine.capture_current_screen() is None
    clock.advance(60)
    await engine.tick()
    assert cap.calls == 0
    assert len(judge.calls) == 0
    assert engine.get_status().session.distraction_seconds == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_temporary_allowance_overrides_high_suspicion_until_expiry() -> None:
    engine, clock, _, _, _, _, judge, _ = _make_engine()
    await engine.start_session()

    await engine.grant_allowance(pattern="instagram", minutes=10, reason="Ver edital")
    await engine.handle_os_event(
        OSContextEvent(
            process_name="chrome.exe",
            window_title="Instagram",
            browser_url="https://www.instagram.com/concursos/",
        )
    )
    clock.advance(120)
    await engine.tick()
    assert engine.get_status().session.current_state == FocusState.FOCUSED
    assert len(judge.calls) == 0

    # Advance past 10 minutes -> allowance expires -> immediate high suspicion drift
    clock.advance(500)
    await engine.tick()
    assert engine.get_status().session.current_state == FocusState.DRIFT_SUSPECTED


@pytest.mark.asyncio
async def test_break_countdown_pomodoro_notes_and_budget_throttle(tmp_path: Path) -> None:
    clock = FakeClock()
    cfg = _make_config()
    cfg.ai.max_daily_usd = 0.00025  # Exhausts after 1 FakeSemanticJudge call ($0.0003)
    storage = SQLiteStorage(tmp_path / "engine.db", clock=clock)
    engine, _, bus, _, _, _, judge, _ = _make_engine(
        clock=clock, config=cfg, storage=storage
    )

    suggestions: list[ProactiveSuggestionEvent] = []
    bus.subscribe(ProactiveSuggestionEvent, suggestions.append)

    await engine.start()
    try:
        await engine.start_session(study_goal="Auditor", subject="Contabilidade")
        note = await engine.add_study_note("CPC 00 importante")
        assert note.subject == "Contabilidade"
        assert [n.note for n in await engine.list_study_notes()] == ["CPC 00 importante"]

        # Start break for 5m -> expires after 300s and emits info suggestion
        await engine.start_break(minutes=5)
        assert engine.get_status().session.current_state == FocusState.BREAK_MODE
        clock.advance(300)
        await engine.tick()
        assert engine.get_status().session.current_state == FocusState.FOCUSED
        assert any(s.kind == "info" and s.speak_aloud for s in suggestions)

        # Trigger 1 judge call, which exceeds $0.00025 daily cap
        await engine.handle_os_event(
            OSContextEvent(process_name="obsidian.exe", window_title="Contabilidade")
        )
        clock.advance(10)
        await engine.tick()
        assert len(judge.calls) == 1
        assert engine.get_status().budget.is_throttled is True

        # Switch to another window -> BudgetGuard blocks second judge call gracefully
        await engine.handle_os_event(
            OSContextEvent(process_name="antigravity.exe", window_title="Código")
        )
        clock.advance(15)
        await engine.tick()
        assert len(judge.calls) == 1

        # Continuous 50-min focus -> Pomodoro suggestion
        clock.advance(50 * 60)
        await engine.tick()
        assert any(s.kind == "pomodoro" and not s.speak_aloud for s in suggestions)
    finally:
        await engine.stop()
        await storage.close()


@pytest.mark.asyncio
async def test_absence_looking_down_late_coach_and_adaptive_rechecks() -> None:
    cfg = _make_config()
    cfg.behavior.require_ai_phone_confirmation = False
    cfg.behavior.allow_looking_down_for_notes = False
    judge = FakeSemanticJudge(default_verdict=ActivityVerdict.STUDYING)
    engine, clock, bus, _, _, _, _, dispatcher = _make_engine(config=cfg, judge=judge)
    await engine.start_session()

    # Late coach injection resolves circular DI
    coach = FakeCoachAgent(configured=True)
    engine.set_coach_agent(coach)

    # Publish OSContextEvent directly on EventBus (like OSSensor does)
    obsidian = OSContextEvent(process_name="obsidian.exe", window_title="Notas Gerais")
    await bus.publish(obsidian)
    clock.advance(10)
    await engine.tick()
    assert len(judge.calls) == 1

    # Focused adaptive recheck after 90s
    clock.advance(90)
    await engine.tick()
    assert len(judge.calls) == 2

    # set_study_goal invalidates cache so revisiting obsidian re-evaluates
    await engine.set_study_goal("Magistratura", "Penal")
    await bus.publish(obsidian)
    clock.advance(10)
    await engine.tick()
    assert len(judge.calls) == 3

    # looking_down_score > 0.60 for 25s triggers express spoken phone intervention
    await bus.publish(
        VisionSignalEvent(camera_available=True, face_present=True, looking_down_score=0.80)
    )
    clock.advance(25)
    await engine.tick()
    assert engine.get_status().session.current_state == FocusState.INTERVENING
    assert engine.get_status().session.phone_alerts_count == 1

    # Recover & test express camera-visibility absence alert (20s threshold)
    await bus.publish(
        VisionSignalEvent(camera_available=True, face_present=True, looking_down_score=0.0)
    )
    await engine.tick()
    assert engine.get_status().session.current_state == FocusState.FOCUSED

    await bus.publish(
        OSContextEvent(process_name="obsidian.exe", window_title="Notas Gerais", idle_seconds=60.0)
    )
    await bus.publish(
        VisionSignalEvent(camera_available=True, face_present=False, looking_down_score=0.0)
    )
    clock.advance(15)
    await engine.tick()
    assert engine.get_status().session.current_state == FocusState.FOCUSED

    # Cross 20s unseen -> express spoken lane jumps straight to INTERVENING with coach nudge
    clock.advance(5)
    await engine.tick()
    assert engine.get_status().session.current_state == FocusState.INTERVENING
    assert dispatcher.dispatched[-1].speak_aloud is True
    assert "Pedro, para de enrolar" in dispatcher.dispatched[-1].message

    await bus.publish(VisionSignalEvent(camera_available=True, face_present=True))
    await engine.acknowledge_back_to_focus()
    assert engine.get_status().session.current_state == FocusState.FOCUSED
    assert engine.get_status().session.camera_unseen_seconds == 0.0
    assert dispatcher.overlay_dismissed >= 1


@pytest.mark.asyncio
async def test_pause_end_break_unconfigured_judge_and_storage_hydration(tmp_path: Path) -> None:
    clock = FakeClock()
    db_path = tmp_path / "hydrate.db"
    pre_storage = SQLiteStorage(db_path, clock=clock)
    await pre_storage.init()
    await pre_storage.save_usage("gemini-3.5-flash-lite", 1000, 200, 0.12)
    await pre_storage.close()

    # Fresh storage & unconfigured judge
    storage = SQLiteStorage(db_path, clock=clock)
    judge = FakeSemanticJudge(configured=False)
    engine, _, _, _, _, _, _, _ = _make_engine(clock=clock, judge=judge, storage=storage)

    await engine.start()
    try:
        # BudgetGuard hydrated from SQLite on start()
        status = engine.get_status()
        assert status.budget.daily_usd == pytest.approx(0.12)
        assert status.budget.daily_tokens_in == 1000
        assert status.budget.daily_calls == 1

        await engine.start_session()
        await engine.handle_os_event(
            OSContextEvent(process_name="antigravity.exe", window_title="FocusGuard")
        )
        clock.advance(15)
        await engine.tick()
        # Unconfigured judge never fails and never calls evaluate()
        assert len(judge.calls) == 0

        # Pause stops time accounting & blocks screen capture
        paused = await engine.pause_session()
        assert paused.current_state == FocusState.PAUSED
        assert engine.capture_current_screen() is None
        focus_before = paused.focus_seconds
        clock.advance(120)
        await engine.tick()
        assert engine.get_status().session.focus_seconds == pytest.approx(focus_before)

        # Resume -> start_break -> end_break early
        await engine.start_session()
        await engine.start_break(minutes=5)
        assert engine.capture_current_screen() is None
        ended = await engine.end_break()
        assert ended.current_state == FocusState.FOCUSED
        assert ended.active_break_remaining_sec == 0.0
    finally:
        await engine.stop()
        await storage.close()


@pytest.mark.asyncio
async def test_speak_on_all_interventions_false_keeps_level1_silent() -> None:
    cfg = _make_config()
    cfg.behavior.speak_on_all_interventions = False
    engine, clock, _, _, _, _, _, dispatcher = _make_engine(config=cfg)
    await engine.start_session()

    await engine.handle_os_event(
        OSContextEvent(
            process_name="chrome.exe",
            window_title="Shorts - YouTube",
            browser_url="https://www.youtube.com/shorts/silent1",
        )
    )
    clock.advance(45)
    await engine.tick()
    assert len(dispatcher.dispatched) == 1
    assert dispatcher.dispatched[0].level == InterventionLevel.SOFT
    assert dispatcher.dispatched[0].speak_aloud is False

    clock.advance(75)
    await engine.tick()
    assert len(dispatcher.dispatched) == 2
    assert dispatcher.dispatched[1].level == InterventionLevel.VOICE
    assert dispatcher.dispatched[1].speak_aloud is True


@pytest.mark.asyncio
async def test_leaky_bucket_phone_hysteresis_and_express_15s_spoken_intervention() -> None:
    cfg = _make_config()
    cfg.behavior.phone_distraction_sec = 15
    cfg.behavior.speak_on_all_interventions = True
    cfg.behavior.require_ai_phone_confirmation = False
    cfg.behavior.allow_looking_down_for_notes = False
    engine, clock, _, _, _, _, _, dispatcher = _make_engine(config=cfg)
    await engine.start_session()

    # Off-screen + moderate downward gaze (looking_at_screen=False, looking_down_score=0.48 > 0.42)
    await engine.handle_vision_event(
        VisionSignalEvent(
            camera_available=True,
            face_present=True,
            looking_at_screen=False,
            looking_down_score=0.48,
        )
    )
    clock.advance(10)
    await engine.tick()
    assert engine.get_status().session.current_state == FocusState.FOCUSED
    assert engine._phone_continuous_sec == pytest.approx(10.0)

    # Brief 4s glance back at screen -> decays by 4 * 0.5 = 2.0s (to 8.0s), NOT snapped to 0!
    await engine.handle_vision_event(
        VisionSignalEvent(
            camera_available=True,
            face_present=True,
            looking_at_screen=True,
            looking_down_score=0.10,
            phone_detected=False,
        )
    )
    clock.advance(4)
    await engine.tick()
    assert engine._phone_continuous_sec == pytest.approx(8.0)
    assert engine.get_status().session.current_state == FocusState.FOCUSED

    # Resume looking down (>0.60) for 7s -> 8.0 + 7.0 = 15.0s -> express spoken Level 1!
    await engine.handle_vision_event(
        VisionSignalEvent(
            camera_available=True,
            face_present=True,
            looking_at_screen=True,
            looking_down_score=0.62,
        )
    )
    clock.advance(7)
    await engine.tick()
    status = engine.get_status()
    assert status.session.current_state == FocusState.INTERVENING
    assert status.session.escalation_level == InterventionLevel.SOFT
    assert status.session.phone_alerts_count == 1
    assert len(dispatcher.dispatched) == 1
    assert dispatcher.dispatched[0].speak_aloud is True
    assert "celular" in dispatcher.dispatched[0].reason.lower()

    # Put phone down for 4s -> returns to FOCUSED, bucket decays gradually (13.0s) keeping episode flag
    await engine.handle_vision_event(
        VisionSignalEvent(
            camera_available=True,
            face_present=True,
            looking_at_screen=True,
            looking_down_score=0.05,
        )
    )
    clock.advance(4)
    await engine.tick()
    assert engine.get_status().session.current_state == FocusState.FOCUSED
    assert engine._phone_continuous_sec == pytest.approx(13.0)
    assert engine._phone_alert_counted_in_episode is True

    # After full decay (26s * 0.5 = 13.0s) -> bucket reaches 0.0 and resets episode flag
    clock.advance(26)
    await engine.tick()
    assert engine._phone_continuous_sec == pytest.approx(0.0)
    assert engine._phone_alert_counted_in_episode is False


@pytest.mark.asyncio
async def test_dual_frame_webcam_attachment_on_idle_and_express_gemini_phone_verdict() -> None:
    cfg = _make_config()
    cfg.behavior.idle_webcam_check_sec = 12
    cfg.behavior.always_attach_webcam_to_judge = False
    cfg.ai.ambiguous_recheck_interval_sec = 25
    vision = FakeVisionSensor(available=True)
    judge = FakeSemanticJudge(default_verdict=ActivityVerdict.STUDYING)
    engine, clock, _, _, _, _, _, dispatcher = _make_engine(config=cfg, judge=judge)
    engine.set_vision_sensor(vision)
    await engine.start_session()

    await engine.handle_vision_event(
        VisionSignalEvent(
            camera_available=True,
            face_present=True,
            looking_at_screen=True,
            looking_down_score=0.0,
        )
    )
    await engine.handle_os_event(
        OSContextEvent(
            process_name="msedge.exe",
            window_title="Lei 8112 Comentada.pdf",
            idle_seconds=0.0,
        )
    )
    clock.advance(10)
    await engine.tick()
    assert len(judge.calls) == 1
    assert judge.calls[0]["used_webcam"] is False
    assert engine.get_status().session.current_state == FocusState.FOCUSED

    # User goes idle on the same PDF (>= 12s) -> rechecks at 25s & attaches webcam frame!
    judge.scripted_verdicts.append(
        SemanticVerdict(
            verdict=ActivityVerdict.DISTRACTED,
            confidence=0.95,
            category="phone",
            app_context="Lei 8112 Comentada.pdf",
            reasoning="Pedro está mexendo no celular com o PDF parado na tela.",
            suggested_message="Larga esse celular e resolve as questões do PDF!",
            used_vision=True,
            used_webcam=True,
            phone_in_use=True,
            physical_state="looking_at_phone",
            model_used="fake-gemini",
            tokens_in=400,
            tokens_out=60,
            estimated_cost_usd=0.0003,
        )
    )
    await engine.handle_os_event(
        OSContextEvent(
            process_name="msedge.exe",
            window_title="Lei 8112 Comentada.pdf",
            idle_seconds=15.0,
        )
    )
    clock.advance(25)
    await engine.tick()

    assert len(judge.calls) == 2
    assert judge.calls[1]["used_webcam"] is True
    status = engine.get_status()
    assert status.session.current_state == FocusState.INTERVENING
    assert status.session.escalation_level == InterventionLevel.SOFT
    assert status.session.phone_alerts_count == 1
    assert len(dispatcher.dispatched) == 1
    assert dispatcher.dispatched[0].speak_aloud is True
    assert "Larga esse celular" in dispatcher.dispatched[0].message


@pytest.mark.asyncio
async def test_chair_false_positive_and_tablet_notes_never_override_itil_studying() -> None:
    cfg = _make_config()
    cfg.study_goal = "Certificacao ITIL"
    cfg.current_subject = "ITIL v4"
    cfg.behavior.require_ai_phone_confirmation = True
    cfg.behavior.allow_looking_down_for_notes = True
    cfg.behavior.always_attach_webcam_to_judge = True

    vision = FakeVisionSensor(available=True)
    judge = FakeSemanticJudge(default_verdict=ActivityVerdict.STUDYING)
    engine, clock, _, _, _, _, _, dispatcher = _make_engine(config=cfg, judge=judge)
    engine.set_vision_sensor(vision)
    await engine.start_session()

    # Watching ITIL lecture on YouTube while looking down at tablet & chair triggers raw phone_detected
    await engine.handle_os_event(
        OSContextEvent(
            process_name="chrome.exe",
            window_title="ITIL 4 Foundation Full Course - YouTube",
            browser_url="https://www.youtube.com/watch?v=itil4",
        )
    )
    await engine.handle_vision_event(
        VisionSignalEvent(
            camera_available=True,
            face_present=True,
            looking_down_score=0.85,
            phone_detected=True,
            phone_confidence=0.28,
        )
    )
    clock.advance(60)
    await engine.tick()

    assert engine.get_status().session.current_state == FocusState.FOCUSED
    assert len(dispatcher.dispatched) == 0
    assert len(judge.calls) >= 1
    assert judge.calls[0]["used_webcam"] is True


@pytest.mark.asyncio
async def test_youtube_video_to_video_switch_resets_timer_and_foreground_music_parks_distracted() -> None:
    cfg = _make_config()
    cfg.behavior.youtube_music_grace_sec = 30
    cfg.ai.window_switch_debounce_sec = 6
    cfg.behavior.distraction_level1_sec = 25
    judge = FakeSemanticJudge(default_verdict=ActivityVerdict.STUDYING)
    engine, clock, _, _, _, _, _, dispatcher = _make_engine(config=cfg, judge=judge)
    await engine.start_session()

    # 1. Watch ITIL lecture for 40s (evaluated once at 30s)
    await engine.handle_os_event(
        OSContextEvent(
            process_name="chrome.exe",
            window_title="ITIL 4 Foundation - YouTube",
            browser_url="https://www.youtube.com/watch?v=itil_1",
        )
    )
    clock.advance(40)
    await engine.tick()
    assert len(judge.calls) == 1
    assert engine.get_status().session.current_state == FocusState.FOCUSED

    # 2. Switch to a DIFFERENT YouTube video (foreground music/random) -> MUST reset timer & wait grace!
    judge.scripted_verdicts.append(
        SemanticVerdict(
            verdict=ActivityVerdict.MUSIC_ALLOWED,
            confidence=0.95,
            category="music",
            app_context="X-COOL! - YouTube",
            reasoning="Música tocando no YouTube.",
            suggested_message="",
            used_vision=True,
            model_used="fake-gemini",
        )
    )
    await engine.handle_os_event(
        OSContextEvent(
            process_name="chrome.exe",
            window_title="X-COOL! - YouTube",
            browser_url="https://www.youtube.com/watch?v=xcool_music",
        )
    )
    # At t=0 on new video: must enter GRACE_WINDOW, NOT evaluate immediately on stale screenshot!
    assert engine.get_status().session.current_state == FocusState.GRACE_WINDOW
    assert len(judge.calls) == 1

    # 3. At t=25s (< 30s grace): still in GRACE_WINDOW
    clock.advance(25)
    await engine.tick()
    assert engine.get_status().session.current_state == FocusState.GRACE_WINDOW
    assert len(judge.calls) == 1

    # 4. Cross 30s grace while parked on foreground YouTube music -> converts MUSIC_ALLOWED -> DISTRACTED!
    clock.advance(5)
    await engine.tick()
    assert len(judge.calls) == 2
    assert engine.get_status().session.current_state == FocusState.DRIFT_SUSPECTED

    # 5. Continue parked 20s (5 + 20 = 25s) -> triggers spoken Level 1 intervention
    clock.advance(20)
    await engine.tick()
    assert engine.get_status().session.current_state == FocusState.INTERVENING
    assert len(dispatcher.dispatched) >= 1
    assert dispatcher.dispatched[0].speak_aloud is True


@pytest.mark.asyncio
async def test_v3_camera_unseen_while_pc_active_obstructed_gemini_away_and_passive_freeze() -> None:
    cfg = _make_config()
    cfg.behavior.camera_not_seeing_user_sec = 20
    cfg.behavior.distraction_level1_sec = 25
    cfg.behavior.intervention_cooldown_sec = 45
    cfg.behavior.passive_freeze_alert_sec = 240
    cfg.behavior.speak_on_all_interventions = True

    judge = FakeSemanticJudge(default_verdict=ActivityVerdict.STUDYING)
    engine, clock, bus, _, _, _, _, dispatcher = _make_engine(config=cfg, judge=judge)
    suggestions: list[ProactiveSuggestionEvent] = []
    bus.subscribe(ProactiveSuggestionEvent, suggestions.append)
    await engine.start_session()

    # (a) PC actively used (idle=2s) while camera stops seeing user -> alerts after 20s anyway!
    await bus.publish(
        OSContextEvent(
            process_name="chrome.exe",
            window_title="Tec Concursos - Questões",
            idle_seconds=2.0,
        )
    )
    await bus.publish(VisionSignalEvent(camera_available=True, face_present=False))
    clock.advance(20)
    await engine.tick()

    assert engine.get_status().session.current_state == FocusState.INTERVENING
    assert engine.get_status().session.camera_unseen_seconds >= 20.0
    assert len(dispatcher.dispatched) == 1
    assert dispatcher.dispatched[0].speak_aloud is True
    assert "fora de enquadramento" in dispatcher.dispatched[0].reason.lower()

    # Recover back to camera visible
    await bus.publish(VisionSignalEvent(camera_available=True, face_present=True))
    await engine.acknowledge_back_to_focus()
    assert engine.get_status().session.current_state == FocusState.FOCUSED

    # (b) Camera covered / dark (camera_obstructed=True) -> expresses obstructed reason after cooldown
    clock.advance(50)
    await engine.tick()
    await bus.publish(
        VisionSignalEvent(camera_available=True, face_present=False, camera_obstructed=True)
    )
    clock.advance(20)
    await engine.tick()
    assert engine.get_status().session.current_state == FocusState.INTERVENING
    assert "tampada" in dispatcher.dispatched[-1].reason.lower()
    assert dispatcher.dispatched[-1].speak_aloud is True

    # Recover and advance past cooldown
    await bus.publish(
        VisionSignalEvent(camera_available=True, face_present=True, camera_obstructed=False)
    )
    await engine.acknowledge_back_to_focus()
    clock.advance(50)
    await engine.tick()

    # (c) Gemini returning ActivityVerdict.AWAY triggers express spoken intervention
    judge.scripted_verdicts.append(
        SemanticVerdict(
            verdict=ActivityVerdict.AWAY,
            confidence=0.94,
            category="away",
            app_context="Tec Concursos",
            reasoning="Cadeira vazia na foto da webcam.",
            suggested_message="Pedro, cadeira vazia não resolve questão!",
            used_vision=True,
            used_webcam=True,
            physical_state="away",
            model_used="fake-gemini",
        )
    )
    await bus.publish(
        OSContextEvent(process_name="msedge.exe", window_title="Lei Seca PDF", idle_seconds=5.0)
    )
    clock.advance(10)
    await engine.tick()
    assert engine.get_status().session.current_state == FocusState.INTERVENING
    assert dispatcher.dispatched[-1].speak_aloud is True
    assert "cadeira vazia" in dispatcher.dispatched[-1].message.lower()

    # (d) Passive freeze check (idle >= 240s, no background audio, visible on cam)
    await engine.acknowledge_back_to_focus()
    await bus.publish(
        OSContextEvent(
            process_name="msedge.exe",
            window_title="Apostila ITIL.pdf",
            background_audio_apps=[],
            idle_seconds=245.0,
        )
    )
    clock.advance(10)
    await engine.tick()
    stuck_events = [s for s in suggestions if s.kind == "stuck"]
    assert len(stuck_events) == 1
    assert stuck_events[0].speak_aloud is True


@pytest.mark.asyncio
async def test_distracted_verdict_not_cached_and_secondary_monitor_change_clears_past_context() -> None:
    from focus_guard.core.events import VisibleWindowInfo

    cfg = _make_config()
    cfg.ai.window_switch_debounce_sec = 6
    judge = FakeSemanticJudge(default_verdict=ActivityVerdict.STUDYING)
    engine, clock, _, _, _, _, _, _ = _make_engine(config=cfg, judge=judge)
    await engine.start_session()

    # 1. Foreground = FocusGuard, M2 = Downdetector -> DISTRACTED
    judge.scripted_verdicts.append(
        SemanticVerdict(
            verdict=ActivityVerdict.DISTRACTED,
            confidence=0.98,
            category="distracao_monitor_secundario",
            app_context="FocusGuard",
            reasoning="Monitor secundário exibe o Downdetector.",
            suggested_message="Fecha o Downdetector!",
            used_vision=True,
            model_used="fake-gemini",
        )
    )
    fg_with_downdetector = OSContextEvent(
        process_name="chrome.exe",
        window_title="FocusGuard — Monitor Multimodal de Estudos",
        browser_url="http://127.0.0.1:8765/",
        visible_windows=[
            VisibleWindowInfo(
                hwnd=1,
                process_name="chrome.exe",
                window_title="FocusGuard — Monitor Multimodal de Estudos",
                monitor_label="M1-Principal",
                screen_area_pct=80.0,
                is_foreground=True,
            ),
            VisibleWindowInfo(
                hwnd=2,
                process_name="chrome.exe",
                window_title="Downdetector - Falhas de serviços",
                monitor_label="M2-Secundario",
                screen_area_pct=80.0,
                is_foreground=False,
            ),
        ],
    )
    await engine.handle_os_event(fg_with_downdetector)
    clock.advance(6)
    await engine.tick()
    assert len(judge.calls) == 1
    assert engine.get_status().session.current_state == FocusState.DRIFT_SUSPECTED
    # DISTRACTED verdict must NEVER be persisted in _session_verdict_cache
    assert engine._session_verdict_cache == {}

    # 2. User closes Downdetector on M2 and opens ITIL lecture while foreground stays on FocusGuard
    fg_with_itil = OSContextEvent(
        process_name="chrome.exe",
        window_title="FocusGuard — Monitor Multimodal de Estudos",
        browser_url="http://127.0.0.1:8765/",
        visible_windows=[
            VisibleWindowInfo(
                hwnd=1,
                process_name="chrome.exe",
                window_title="FocusGuard — Monitor Multimodal de Estudos",
                monitor_label="M1-Principal",
                screen_area_pct=80.0,
                is_foreground=True,
            ),
            VisibleWindowInfo(
                hwnd=3,
                process_name="chrome.exe",
                window_title="ITIL 4 Foundation Videoaula",
                monitor_label="M2-Secundario",
                screen_area_pct=80.0,
                is_foreground=False,
            ),
        ],
    )
    await engine.handle_os_event(fg_with_itil)
    clock.advance(6)
    await engine.tick()
    assert len(judge.calls) == 2
    assert engine.get_status().session.current_state == FocusState.FOCUSED


@pytest.mark.asyncio
async def test_slow_judge_and_nudge_never_block_vision_or_dispatch_stale_past_intervention() -> None:
    import asyncio

    cfg = _make_config()
    cfg.ai.window_switch_debounce_sec = 5
    cfg.behavior.distraction_level1_sec = 10

    judge_started = asyncio.Event()
    release_judge = asyncio.Event()

    class SlowJudge(FakeSemanticJudge):
        async def evaluate(self, *args, **kwargs) -> SemanticVerdict:
            judge_started.set()
            await release_judge.wait()
            return SemanticVerdict(
                verdict=ActivityVerdict.DISTRACTED,
                confidence=0.96,
                category="history",
                app_context="Histórico - Google Chrome",
                reasoning="Histórico do Chrome aberto (do passado).",
                suggested_message="Fecha o histórico do Chrome!",
                used_vision=True,
                model_used="fake-gemini",
            )

    nudge_started = asyncio.Event()
    release_nudge = asyncio.Event()

    class SlowCoach(FakeCoachAgent):
        async def generate_nudge(self, *args, **kwargs) -> str:
            nudge_started.set()
            await release_nudge.wait()
            return "Pedro, cobrança atrasada do passado!"

    judge = SlowJudge(default_verdict=ActivityVerdict.DISTRACTED)
    coach = SlowCoach(configured=True)
    dispatcher = FakeInterventionDispatcher()
    engine, clock, _, _, _, _, _, _ = _make_engine(
        config=cfg, judge=judge, coach=coach, dispatcher=dispatcher
    )
    await engine.start_session()

    # 1. Start slow evaluation on chrome://history
    await engine.handle_os_event(
        OSContextEvent(
            process_name="chrome.exe",
            window_title="Histórico - Google Chrome",
            browser_url="chrome://history",
        )
    )
    clock.advance(5)
    tick_task = asyncio.create_task(engine.tick())
    await asyncio.wait_for(judge_started.wait(), timeout=2.0)

    # While Gemini is blocked, vision & OS events MUST complete immediately (no camera freeze!)
    await asyncio.wait_for(
        engine.handle_vision_event(VisionSignalEvent(camera_available=True, face_present=True)),
        timeout=0.5,
    )
    await asyncio.wait_for(
        engine.handle_os_event(
            OSContextEvent(
                process_name="msedge.exe",
                window_title="ITIL v4 Apostila.pdf",
            )
        ),
        timeout=0.5,
    )

    # Release the slow judge for the OLD chrome://history window -> MUST be discarded!
    release_judge.set()
    await asyncio.wait_for(tick_task, timeout=2.0)
    assert engine.get_status().session.current_state == FocusState.FOCUSED
    assert len(dispatcher.dispatched) == 0

    # 2. Now trigger a high-suspicion Shorts intervention whose coach nudge is slow,
    # and verify acknowledge_back_to_focus() aborts the stale intervention without speaking!
    await engine.handle_os_event(
        OSContextEvent(
            process_name="chrome.exe",
            window_title="Shorts - YouTube",
            browser_url="https://www.youtube.com/shorts/old123",
        )
    )
    clock.advance(10)
    nudge_tick = asyncio.create_task(engine.tick())
    await asyncio.wait_for(nudge_started.wait(), timeout=2.0)

    # Vision and ack_focus MUST NOT block while nudge is being generated
    await asyncio.wait_for(
        engine.handle_vision_event(VisionSignalEvent(camera_available=True, face_present=True)),
        timeout=0.5,
    )
    await asyncio.wait_for(engine.acknowledge_back_to_focus(), timeout=0.5)

    release_nudge.set()
    await asyncio.wait_for(nudge_tick, timeout=2.0)
    assert engine.get_status().session.current_state == FocusState.FOCUSED
    assert len(dispatcher.dispatched) == 0


@pytest.mark.asyncio
async def test_looking_down_for_notes_does_not_trigger_fatigue_alert() -> None:
    cfg = _make_config()
    cfg.behavior.allow_looking_down_for_notes = True
    cfg.behavior.drowsiness_perclos_threshold = 0.45
    cfg.behavior.drowsiness_window_sec = 25

    engine, clock, bus, _, _, _, _, _ = _make_engine(config=cfg)
    suggestions: list[ProactiveSuggestionEvent] = []
    bus.subscribe(ProactiveSuggestionEvent, suggestions.append)
    await engine.start_session()

    # User looks down writing notes for 60s -> even if PERCLOS is elevated, no fatigue alert!
    await engine.handle_vision_event(
        VisionSignalEvent(
            camera_available=True,
            face_present=True,
            looking_down_score=0.55,
            drowsiness_perclos=0.75,
        )
    )
    clock.advance(60)
    await engine.tick()

    fatigue_events = [s for s in suggestions if s.kind == "fatigue"]
    assert fatigue_events == []
    assert engine.get_status().session.drowsiness_alerts_count == 0


@pytest.mark.asyncio
async def test_subject_study_summary_tracks_mid_session_switches_and_legacy_sessions(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "subjects.db"
    clock = FakeClock(datetime(2026, 11, 19, 10, 0, 0, tzinfo=UTC))
    storage = SQLiteStorage(db_path, clock=clock)
    await storage.init()

    try:
        # Legacy session from yesterday (only in `sessions`, no rows in `subject_study_logs`)
        legacy = SessionStats(
            session_id="legacy-sess-1",
            started_at=clock.now,
            study_goal="Certificação ITIL",
            current_subject="ITIL v4",
            current_state=FocusState.PAUSED,
            focus_seconds=600.0,
            distraction_seconds=60.0,
        )
        await storage.save_session(legacy)

        # Next day: live session studying "itil v4" (120s) then switching to "Português" (90s)
        clock.now = datetime(2026, 11, 20, 14, 0, 0, tzinfo=UTC)
        engine, _, _, _, _, _, _, _ = _make_engine(clock=clock, storage=storage)
        await engine.start()
        try:
            await engine.start_session(study_goal="Certificação ITIL", subject="itil v4")
            clock.advance(120)
            await engine.tick()

            await engine.set_study_goal("Concurso", "Português")
            clock.advance(90)
            await engine.tick()

            summary = await engine.get_subject_study_summary()
            by_name = {item["subject"].lower(): item for item in summary}
            assert "itil v4" in by_name
            assert "português" in by_name

            itil = by_name["itil v4"]
            pt = by_name["português"]

            assert itil["total_focus_seconds"] == pytest.approx(720.0)
            assert itil["today_focus_seconds"] == pytest.approx(120.0)
            assert itil["sessions_count"] == 2
            assert itil["is_current"] is False

            assert pt["total_focus_seconds"] == pytest.approx(90.0)
            assert pt["today_focus_seconds"] == pytest.approx(90.0)
            assert pt["sessions_count"] == 1
            assert pt["is_current"] is True
            assert itil["share_pct"] + pt["share_pct"] == pytest.approx(100.0, abs=0.2)
        finally:
            await engine.stop()
    finally:
        await storage.close()


