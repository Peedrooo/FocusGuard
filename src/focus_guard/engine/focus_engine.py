"""Core state machine and Zero-Prison Semantic-First orchestration engine."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from focus_guard.core.budget import BudgetGuard
from focus_guard.core.config import AppConfig
from focus_guard.core.event_bus import EventBus
from focus_guard.core.events import (
    ActivityVerdict,
    FocusState,
    FocusStateChangedEvent,
    InterventionCommand,
    InterventionLevel,
    OSContextEvent,
    ProactiveSuggestionEvent,
    ScreenSnapshot,
    SemanticVerdict,
    SessionStats,
    StudyNote,
    SystemStatusSnapshot,
    TemporaryAllowance,
    VisionSignalEvent,
)
from focus_guard.core.ports import (
    CoachAgentPort,
    InterventionDispatcherPort,
    ScreenCapturePort,
    SemanticJudgePort,
    StoragePort,
    VisionSensorPort,
)
from focus_guard.core.privacy import PrivacyShield

logger = logging.getLogger(__name__)

_FATIGUE_COOLDOWN_SEC = 300.0
_SESSION_PERSIST_INTERVAL_SEC = 30.0
_MAX_RECENT_VERDICTS = 20


def _default_clock() -> datetime:
    return datetime.now(UTC)


def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


class FocusEngine:
    """Zero-Prison Semantic-First Focus Engine orchestrating sensors, AI judge, and interventions."""

    def __init__(
        self,
        config: AppConfig,
        event_bus: EventBus,
        budget: BudgetGuard,
        storage: StoragePort,
        screen_capture: ScreenCapturePort,
        semantic_judge: SemanticJudgePort,
        coach_agent: CoachAgentPort | None = None,
        dispatcher: InterventionDispatcherPort | None = None,
        clock: Callable[[], datetime] | None = None,
        vision_sensor: VisionSensorPort | None = None,
    ) -> None:
        self._config = config
        self._event_bus = event_bus
        self._budget = budget
        self._storage = storage
        self._screen_capture = screen_capture
        self._semantic_judge = semantic_judge
        self._coach_agent = coach_agent
        self._dispatcher = dispatcher
        self._clock: Callable[[], datetime] = clock or _default_clock
        self._vision_sensor = vision_sensor
        self._privacy_shield = PrivacyShield(config.privacy)

        self._eval_lock = asyncio.Lock()
        self._running = False
        self._privacy_mode = False
        self._mic_available = False
        self._state_before_privacy: FocusState = FocusState.PAUSED
        self._ticker_task: asyncio.Task[None] | None = None

        now = self._now()
        self._session = SessionStats(
            started_at=now,
            study_goal=config.study_goal,
            current_subject=config.current_subject,
            current_state=FocusState.PAUSED,
        )

        self._latest_os: OSContextEvent | None = None
        self._latest_vision: VisionSignalEvent | None = None
        self._active_allowances: list[TemporaryAllowance] = []

        # Context & evaluation tracking
        self._session_verdict_cache: dict[str, SemanticVerdict] = {}
        self._current_sig: str | None = None
        self._context_epoch: int = 0
        self._judge_in_flight: bool = False
        self._intervention_in_flight: bool = False
        self._context_entered_at: datetime | None = None
        self._youtube_entered_at: datetime | None = None
        self._last_evaluated_sig: str | None = None
        self._last_evaluated_at: datetime | None = None
        self._awaiting_debounce: bool = False
        self._allowance_active_for_sig: bool = False
        self._active_allowance_expires_at: datetime | None = None
        self._pending_pre_expiry_focus_dt: float = 0.0
        self._last_verdict: SemanticVerdict | None = None
        self._last_verdict_value: ActivityVerdict | None = None
        self._last_verdict_reason: str = ""

        # Timers & escalation tracking
        self._last_tick_at: datetime | None = None
        self._last_session_saved_at: datetime | None = None
        self._current_drift_seconds: float = 0.0
        self._last_dispatched_level: InterventionLevel = InterventionLevel.NONE
        self._last_intervention_at: datetime | None = None

        # Vision continuous counters
        self._phone_continuous_sec: float = 0.0
        self._phone_alert_counted_in_episode: bool = False
        self._absence_continuous_sec: float = 0.0
        self._camera_unseen_continuous_sec: float = 0.0
        self._drowsiness_continuous_sec: float = 0.0
        self._last_fatigue_alert_at: datetime | None = None
        self._last_passive_freeze_alert_at: datetime | None = None

        # Pomodoro & per-subject time tracking
        self._current_focus_block_seconds: float = 0.0
        self._last_pomodoro_at: datetime | None = None
        self._subject_focus_pending: dict[str, float] = {}
        self._subject_distraction_pending: dict[str, float] = {}

        self._subscribe_events()

    def _now(self) -> datetime:
        return _to_utc(self._clock())

    def _subscribe_events(self) -> None:
        self._event_bus.subscribe(OSContextEvent, self.handle_os_event)
        self._event_bus.subscribe(VisionSignalEvent, self.handle_vision_event)

    def _unsubscribe_events(self) -> None:
        self._event_bus.unsubscribe(OSContextEvent, self.handle_os_event)
        self._event_bus.unsubscribe(VisionSignalEvent, self.handle_vision_event)

    def set_coach_agent(self, coach_agent: CoachAgentPort | None) -> None:
        """Attach or replace the CoachAgent (resolves circular DI)."""
        self._coach_agent = coach_agent

    def set_vision_sensor(self, vision_sensor: VisionSensorPort | None) -> None:
        """Attach or replace the VisionSensorPort for dual-frame webcam checks."""
        self._vision_sensor = vision_sensor

    def set_mic_available(self, available: bool) -> None:
        """Update microphone hardware availability status."""
        self._mic_available = bool(available)

    # ------------------------------------------------------------------
    # Lifecycle & Control (FocusEnginePort)
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Initialize storage, hydrate budget/allowances, subscribe, and start 1s ticker."""
        await self._storage.init()
        daily_usd, monthly_usd, daily_in, daily_out, daily_calls = (
            await self._storage.get_usage_totals()
        )
        if monthly_usd > 0 or daily_calls > 0:
            self._budget.hydrate(daily_usd, monthly_usd, daily_in, daily_out, daily_calls)

        now = self._now()
        stored_allowances = await self._storage.list_active_allowances()
        existing_ids = {a.id for a in self._active_allowances}
        for item in stored_allowances:
            if item.id not in existing_ids and item.is_active(now):
                self._active_allowances.append(item)

        self._subscribe_events()
        self._running = True
        if self._ticker_task is None or self._ticker_task.done():
            self._ticker_task = asyncio.create_task(self._ticker_loop())

    async def stop(self) -> None:
        """Stop background ticker, unsubscribe from bus, and persist final session snapshot."""
        self._running = False
        if self._ticker_task is not None and not self._ticker_task.done():
            self._ticker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._ticker_task
        self._ticker_task = None
        self._unsubscribe_events()
        await self._persist_session_safe()

    async def _ticker_loop(self) -> None:
        try:
            while self._running:
                await asyncio.sleep(1.0)
                if not self._running:
                    break
                try:
                    await self.tick()
                except Exception:
                    logger.exception("Error during periodic FocusEngine tick")
        except asyncio.CancelledError:
            pass

    async def start_session(
        self, study_goal: str | None = None, subject: str | None = None
    ) -> SessionStats:
        """Start or resume a focus study session."""
        async with self._eval_lock:
            now = self._now()
            if study_goal and study_goal.strip():
                self._session.study_goal = study_goal.strip()
            if subject and subject.strip():
                self._session.current_subject = subject.strip()

            self._context_epoch += 1
            self._last_tick_at = now
            self._context_entered_at = now
            self._last_evaluated_at = None
            self._last_evaluated_sig = None

            if not self._privacy_mode and self._session.current_state != FocusState.FOCUSED:
                await self._transition_state(
                    FocusState.FOCUSED, "Sessão de estudo iniciada", now
                )
            await self._persist_session_safe()
            return self._session.model_copy(deep=True)

    async def pause_session(self) -> SessionStats:
        """Pause monitoring and time accounting without closing the engine."""
        async with self._eval_lock:
            now = self._now()
            await self._advance_timers_only_locked(now)
            self._context_epoch += 1
            if self._session.current_state == FocusState.INTERVENING:
                self._dismiss_overlay()
            self._stop_speech()
            self._reset_drift()
            self._session.active_break_remaining_sec = 0.0
            await self._transition_state(FocusState.PAUSED, "Sessão pausada", now)
            await self._persist_session_safe()
            return self._session.model_copy(deep=True)

    async def toggle_privacy(self, enabled: bool | None = None) -> bool:
        """Enable or disable Privacy Mode (blocks screen capture and cloud evaluation)."""
        async with self._eval_lock:
            now = self._now()
            await self._advance_timers_only_locked(now)
            target = (not self._privacy_mode) if enabled is None else bool(enabled)
            if target == self._privacy_mode:
                return self._privacy_mode

            self._context_epoch += 1
            self._privacy_mode = target
            if self._privacy_mode:
                if self._session.current_state != FocusState.PRIVACY_MODE:
                    self._state_before_privacy = self._session.current_state
                if self._session.current_state in (
                    FocusState.INTERVENING,
                    FocusState.DRIFT_SUSPECTED,
                ):
                    self._dismiss_overlay()
                self._stop_speech()
                self._reset_drift()
                await self._transition_state(
                    FocusState.PRIVACY_MODE, "Modo Privacidade ativado", now
                )
            else:
                restore = self._state_before_privacy
                if restore == FocusState.BREAK_MODE and self._session.active_break_remaining_sec > 0:
                    target_state = FocusState.BREAK_MODE
                elif restore == FocusState.PAUSED:
                    target_state = FocusState.PAUSED
                else:
                    target_state = FocusState.FOCUSED
                self._last_tick_at = now
                self._context_entered_at = now
                self._last_evaluated_sig = None
                self._last_evaluated_at = None
                await self._transition_state(
                    target_state, "Modo Privacidade desativado", now
                )
            await self._persist_session_safe()
            return self._privacy_mode

    async def set_study_goal(self, goal: str, subject: str) -> SessionStats:
        """Update study goal & subject and invalidate cached verdicts for fresh judgment."""
        async with self._eval_lock:
            now = self._now()
            await self._advance_timers_only_locked(now)
            await self._flush_subject_time_safe()
            self._context_epoch += 1
            self._session.study_goal = goal
            self._session.current_subject = subject
            self._session_verdict_cache.clear()
            self._last_evaluated_at = None
            self._last_evaluated_sig = None
            await self._persist_session_safe()
            return self._session.model_copy(deep=True)

    async def start_break(self, minutes: int = 5) -> SessionStats:
        """Enter BREAK_MODE with a countdown timer."""
        async with self._eval_lock:
            now = self._now()
            await self._advance_timers_only_locked(now)
            self._context_epoch += 1
            if self._session.current_state in (
                FocusState.INTERVENING,
                FocusState.DRIFT_SUSPECTED,
            ):
                self._dismiss_overlay()
            self._stop_speech()
            self._reset_drift()
            self._privacy_mode = False
            self._current_focus_block_seconds = 0.0
            self._session.active_break_remaining_sec = float(max(1, minutes) * 60)
            await self._transition_state(
                FocusState.BREAK_MODE, f"Pausa de {max(1, minutes)} minutos iniciada", now
            )
            await self._persist_session_safe()
            return self._session.model_copy(deep=True)

    async def end_break(self) -> SessionStats:
        """End break early and resume FOCUSED state."""
        async with self._eval_lock:
            now = self._now()
            await self._advance_timers_only_locked(now)
            self._context_epoch += 1
            self._session.active_break_remaining_sec = 0.0
            self._current_focus_block_seconds = 0.0
            self._context_entered_at = now
            self._last_evaluated_sig = None
            self._last_evaluated_at = None
            if not self._privacy_mode:
                await self._transition_state(
                    FocusState.FOCUSED, "Pausa encerrada — retomando foco", now
                )
            await self._persist_session_safe()
            return self._session.model_copy(deep=True)

    async def set_conversation_mode(self, active: bool) -> FocusState:
        """Enter or leave CONVERSATION_MODE when interacting live with the Coach."""
        async with self._eval_lock:
            now = self._now()
            await self._advance_timers_only_locked(now)
            self._context_epoch += 1
            if active:
                if self._session.current_state == FocusState.INTERVENING:
                    self._dismiss_overlay()
                self._stop_speech()
                await self._transition_state(
                    FocusState.CONVERSATION_MODE, "Conversa com o Coach em andamento", now
                )
            elif not self._privacy_mode and self._session.current_state == FocusState.CONVERSATION_MODE:
                self._context_entered_at = now
                self._last_tick_at = now
                await self._transition_state(
                    FocusState.FOCUSED, "Conversa encerrada — retomando foco", now
                )
            return self._session.current_state

    async def grant_allowance(
        self, pattern: str, minutes: int, reason: str
    ) -> TemporaryAllowance:
        """Grant a temporary exception matching process/window/URL."""
        async with self._eval_lock:
            now = self._now()
            allowance = TemporaryAllowance(
                pattern=pattern,
                reason=reason,
                created_at=now,
                expires_at=now + timedelta(minutes=max(1, minutes)),
            )
            self._active_allowances.append(allowance)
            await self._storage.save_allowance(allowance)

            if self._latest_os is not None and self._matching_allowance(self._latest_os, now):
                self._context_epoch += 1
                self._allowance_active_for_sig = True
                await self._apply_verdict(
                    ActivityVerdict.STUDYING,
                    f"Liberação temporária ativa ({allowance.reason})",
                    now,
                )
            return allowance

    async def acknowledge_back_to_focus(self) -> None:
        """Acknowledge intervention and immediately return to FOCUSED state."""
        async with self._eval_lock:
            now = self._now()
            await self._advance_timers_only_locked(now)
            self._context_epoch += 1
            self._dismiss_overlay()
            self._stop_speech()
            self._reset_drift()
            self._camera_unseen_continuous_sec = 0.0
            self._absence_continuous_sec = 0.0
            self._session.camera_unseen_seconds = 0.0
            self._context_entered_at = now
            self._last_evaluated_sig = None
            self._last_evaluated_at = None
            self._last_verdict = None
            self._last_verdict_value = None
            self._last_verdict_reason = ""
            if self._session.current_state in (
                FocusState.INTERVENING,
                FocusState.DRIFT_SUSPECTED,
                FocusState.GRACE_WINDOW,
            ):
                await self._transition_state(
                    FocusState.FOCUSED, "Usuário confirmou retorno ao foco", now
                )
            await self._persist_session_safe()

    async def add_study_note(self, note: str, subject: str | None = None) -> StudyNote:
        """Save a study note to persistent storage."""
        item = StudyNote(
            timestamp=self._now(),
            subject=subject or self._session.current_subject,
            note=note,
        )
        await self._storage.save_note(item)
        return item

    async def list_study_notes(self) -> list[StudyNote]:
        """Retrieve recent study notes."""
        return await self._storage.list_notes(limit=50)

    def capture_current_screen(self) -> ScreenSnapshot | None:
        """Capture an ephemeral in-memory screenshot if privacy allows."""
        if self._privacy_mode:
            return None
        if self._session.current_state in (
            FocusState.PRIVACY_MODE,
            FocusState.PAUSED,
            FocusState.BREAK_MODE,
        ):
            return None
        if self._latest_os is not None and self._is_sensitive_os(self._latest_os):
            return None
        try:
            return self._screen_capture.capture(self._latest_os)
        except Exception:
            logger.exception("ScreenCapturePort.capture failed gracefully")
            return None

    def get_status(self) -> SystemStatusSnapshot:
        """Return a snapshot of current engine, session, budget, and sensor status."""
        now = self._now()
        active_allowances = [a for a in self._active_allowances if a.is_active(now)]
        if self._latest_vision is not None:
            camera_ok = bool(self._latest_vision.camera_available)
        elif self._vision_sensor is not None:
            camera_ok = bool(self._vision_sensor.is_available())
        else:
            camera_ok = False
        return SystemStatusSnapshot(
            timestamp=now,
            running=self._running,
            privacy_mode=self._privacy_mode,
            api_key_configured=bool(self._config.gemini_api_key.strip())
            or self._semantic_judge.is_configured(),
            camera_available=camera_ok,
            mic_available=self._mic_available,
            coach_persona=self._config.coach_persona,
            session=self._session.model_copy(deep=True),
            budget=self._budget.get_status(),
            latest_os=self._latest_os,
            latest_vision=self._latest_vision,
            active_allowances=active_allowances,
        )

    # ------------------------------------------------------------------
    # Event Handlers & Deterministic Tick
    # ------------------------------------------------------------------

    @staticmethod
    def _context_key(os_event: OSContextEvent) -> str:
        """Build a context signature that also reflects major visible secondary windows."""
        base = os_event.context_signature()
        if not os_event.visible_windows:
            return base
        sec_tokens: list[str] = []
        for win in os_event.visible_windows:
            if win.is_foreground or win.is_sensitive:
                continue
            if win.screen_area_pct > 0.0 and win.screen_area_pct < 10.0:
                continue
            sec_tokens.append(
                f"{win.monitor_label}|{win.process_name.lower()}|{win.window_title.strip()[:60]}"
            )
        if not sec_tokens:
            return base
        sec_tokens.sort()
        return f"{base}||vis={';'.join(sec_tokens[:4])}"

    async def handle_os_event(self, event: OSContextEvent) -> None:
        """Process an active window/URL update without ever blocking on cloud I/O or TTS."""
        async with self._eval_lock:
            now = self._now()
            await self._advance_timers_only_locked(now)

            sig = self._context_key(event)
            new_is_fg_yt = self._is_foreground_youtube(event)

            if sig != self._current_sig:
                self._current_sig = sig
                self._context_epoch += 1
                self._context_entered_at = now
                self._last_evaluated_sig = None
                self._allowance_active_for_sig = False
                self._active_allowance_expires_at = None
                self._awaiting_debounce = True
                self._last_verdict = None
                self._last_verdict_value = None
                self._last_verdict_reason = ""

                if new_is_fg_yt and not self._is_youtube_shorts(event):
                    # Always reset timer on video/page change so screenshot waits for render settle
                    self._youtube_entered_at = now
                else:
                    self._youtube_entered_at = None
            elif new_is_fg_yt and not self._is_youtube_shorts(event):
                if self._youtube_entered_at is None:
                    self._youtube_entered_at = now
            else:
                self._youtube_entered_at = None

            self._latest_os = event

            if self._is_monitoring_active():
                await self._evaluate_os_context_fast_locked(now, allow_judge=False)

    async def handle_vision_event(self, event: VisionSignalEvent) -> None:
        """Process a local webcam signal event immediately so the camera loop never stalls."""
        async with self._eval_lock:
            now = self._now()
            prev_vision = self._latest_vision
            await self._advance_timers_only_locked(now)
            self._latest_vision = event

            if self._is_monitoring_active() and not event.camera_available:
                self._reset_vision_episodes()
            elif (
                self._is_monitoring_active()
                and prev_vision is not None
                and (not prev_vision.face_present or prev_vision.camera_obstructed)
                and event.face_present
                and not event.camera_obstructed
                and self._last_verdict_value == ActivityVerdict.AWAY
            ):
                self._context_epoch += 1
                self._last_verdict_value = None
                self._last_verdict = None
                self._last_verdict_reason = ""
                self._last_evaluated_at = None

    async def _advance_timers_only_locked(self, now: datetime) -> None:
        """Fast pure in-memory time accounting (zero cloud/TTS I/O)."""
        now_utc = _to_utc(now)
        prev_tick = self._last_tick_at
        dt = 0.0 if prev_tick is None else max(0.0, (now_utc - prev_tick).total_seconds())
        self._last_tick_at = now_utc
        self._pending_pre_expiry_focus_dt = 0.0

        self._active_allowances = [a for a in self._active_allowances if a.is_active(now_utc)]

        state = self._session.current_state
        if state in (FocusState.PAUSED, FocusState.PRIVACY_MODE, FocusState.CONVERSATION_MODE) or self._privacy_mode:
            return

        if state == FocusState.BREAK_MODE:
            if dt > 0:
                self._session.break_seconds += dt
                self._session.active_break_remaining_sec = max(
                    0.0, self._session.active_break_remaining_sec - dt
                )
                if self._session.active_break_remaining_sec <= 0.0:
                    self._current_focus_block_seconds = 0.0
                    self._context_entered_at = now_utc
                    self._last_evaluated_sig = None
                    self._last_evaluated_at = None
                    await self._transition_state(
                        FocusState.FOCUSED, "Fim da pausa programada", now_utc
                    )
                    await self._event_bus.publish(
                        ProactiveSuggestionEvent(
                            timestamp=now_utc,
                            kind="info",
                            message="Fim da pausa! Hora de retomar o foco.",
                            speak_aloud=True,
                        )
                    )
            return

        if self._latest_os is not None and self._is_sensitive_os(self._latest_os):
            self._reset_vision_episodes()
            return

        if dt <= 0.0:
            return

        physical_distraction, physical_reason, physical_drift_dt = self._update_vision_timers(
            dt, now_utc
        )
        os_distracted = self._last_verdict_value in (
            ActivityVerdict.DISTRACTED,
            ActivityVerdict.AWAY,
        )
        is_drifting = physical_distraction or (os_distracted and not self._awaiting_debounce)

        if is_drifting:
            reason = (
                physical_reason
                if physical_distraction
                else (self._last_verdict_reason or "Distração detectada")
            )
            await self._handle_active_drift_locked(
                dt,
                physical_distraction,
                physical_drift_dt,
                os_distracted,
                reason,
                now_utc,
                allow_escalate=False,
            )
        else:
            if self._session.current_state == FocusState.GRACE_WINDOW:
                self._session.grace_seconds += dt
            elif self._session.current_state == FocusState.FOCUSED:
                self._session.focus_seconds += dt
                self._current_focus_block_seconds += dt
                self._record_subject_delta(focus_dt=dt)

        self._update_focus_ratio()

    async def tick(self, now: datetime | None = None) -> None:
        """Advance deterministic timers, countdowns, escalations, and adaptive checks.

        Slow I/O (Gemini judge, coach nudge generation, and TTS dispatch) executes OUTSIDE
        `self._eval_lock` so webcam capture and OS telemetry never freeze, and stale results
        from superseded contexts are automatically discarded.
        """
        now_utc = _to_utc(now or self._now())

        # Phase 1: Fast in-memory timer advancement & deterministic OS rules under lock
        async with self._eval_lock:
            phase1 = await self._tick_phase1_locked(now_utc)
        if phase1 is None:
            return

        judge_req, dt, physical_distraction, physical_reason, physical_drift_dt = phase1

        # Phase 2: Optional SemanticJudge evaluation OUTSIDE _eval_lock
        if judge_req is not None:
            await self._run_judge_request_unlocked(judge_req)

        # Phase 3: Drift/focus accounting & escalation decision under lock
        async with self._eval_lock:
            esc_req = await self._tick_phase3_locked(
                now_utc=now_utc,
                dt=dt,
                physical_distraction=physical_distraction,
                physical_reason=physical_reason,
                physical_drift_dt=physical_drift_dt,
                ran_judge=judge_req is not None,
            )

        # Phase 4: Optional Coach nudge generation & intervention dispatch OUTSIDE _eval_lock
        if esc_req is not None:
            await self._run_escalation_unlocked(esc_req)

    async def _tick_phase1_locked(
        self, now_utc: datetime
    ) -> tuple[dict[str, Any] | None, float, bool, str, float] | None:
        prev_tick = self._last_tick_at
        dt = 0.0 if prev_tick is None else max(0.0, (now_utc - prev_tick).total_seconds())
        self._last_tick_at = now_utc
        self._pending_pre_expiry_focus_dt = 0.0

        self._active_allowances = [a for a in self._active_allowances if a.is_active(now_utc)]

        state = self._session.current_state
        if state == FocusState.PAUSED:
            return None

        if state == FocusState.PRIVACY_MODE or self._privacy_mode:
            return None

        if state == FocusState.BREAK_MODE:
            if dt > 0:
                self._session.break_seconds += dt
                self._session.active_break_remaining_sec = max(
                    0.0, self._session.active_break_remaining_sec - dt
                )
                if self._session.active_break_remaining_sec <= 0.0:
                    self._current_focus_block_seconds = 0.0
                    self._context_entered_at = now_utc
                    self._last_evaluated_sig = None
                    self._last_evaluated_at = None
                    await self._transition_state(
                        FocusState.FOCUSED, "Fim da pausa programada", now_utc
                    )
                    await self._event_bus.publish(
                        ProactiveSuggestionEvent(
                            timestamp=now_utc,
                            kind="info",
                            message="Fim da pausa! Hora de retomar o foco.",
                            speak_aloud=True,
                        )
                    )
            return None

        if state == FocusState.CONVERSATION_MODE:
            return None

        # Sensitive window: never capture, never evaluate, pause drift without penalty.
        if self._latest_os is not None and self._is_sensitive_os(self._latest_os):
            self._reset_vision_episodes()
            return None

        # Advance physical vision timers first so excess drift splits accurately across dt.
        physical_distraction, physical_reason, physical_drift_dt = self._update_vision_timers(
            dt, now_utc
        )

        # Check fatigue suggestion independently of distraction escalation.
        await self._check_fatigue_suggestion(now_utc)

        # Evaluate fast OS rules and determine if SemanticJudge should run outside the lock.
        judge_req: dict[str, Any] | None = None
        if self._latest_os is not None:
            judge_req = await self._evaluate_os_context_fast_locked(
                now_utc, prev_tick=prev_tick, dt=dt, allow_judge=True
            )

        return judge_req, dt, physical_distraction, physical_reason, physical_drift_dt

    async def _tick_phase3_locked(
        self,
        *,
        now_utc: datetime,
        dt: float,
        physical_distraction: bool,
        physical_reason: str,
        physical_drift_dt: float,
        ran_judge: bool,
    ) -> dict[str, Any] | None:
        if not self._is_monitoring_active():
            return None
        if self._latest_os is not None and self._is_sensitive_os(self._latest_os):
            return None

        # If a cloud judge call ran, re-verify physical distraction hasn't cleared in the meantime.
        if ran_judge and physical_distraction and not self._is_physical_distraction_active():
            physical_distraction = False
            physical_reason = ""
            physical_drift_dt = 0.0

        os_distracted = self._last_verdict_value in (
            ActivityVerdict.DISTRACTED,
            ActivityVerdict.AWAY,
        )
        is_drifting = physical_distraction or (os_distracted and not self._awaiting_debounce)

        esc_req: dict[str, Any] | None = None
        if is_drifting:
            reason = (
                physical_reason
                if physical_distraction
                else (self._last_verdict_reason or "Distração detectada")
            )
            esc_req = await self._handle_active_drift_locked(
                dt,
                physical_distraction,
                physical_drift_dt,
                os_distracted,
                reason,
                now_utc,
                allow_escalate=True,
            )
        else:
            # No active distraction: account time according to current state.
            if dt > 0:
                if self._session.current_state == FocusState.GRACE_WINDOW:
                    self._session.grace_seconds += dt
                elif self._session.current_state == FocusState.FOCUSED:
                    self._session.focus_seconds += dt
                    self._current_focus_block_seconds += dt
                    self._record_subject_delta(focus_dt=dt)
            # If user had been drifting solely from physical distraction and put phone down / returned:
            if (
                not os_distracted
                and not self._awaiting_debounce
                and self._session.current_state in (FocusState.DRIFT_SUSPECTED, FocusState.INTERVENING)
            ):
                await self._apply_verdict(
                    ActivityVerdict.STUDYING,
                    "Retorno ao foco confirmado",
                    now_utc,
                )

        self._update_focus_ratio()
        await self._check_pomodoro_suggestion(now_utc)
        if self._session.current_state == FocusState.FOCUSED and not is_drifting:
            await self._check_passive_freeze_suggestion(now_utc)

        if (
            dt > 0
            and (
                self._last_session_saved_at is None
                or (now_utc - self._last_session_saved_at).total_seconds() >= _SESSION_PERSIST_INTERVAL_SEC
            )
        ):
            self._last_session_saved_at = now_utc
            await self._persist_session_safe()

        return esc_req

    # ------------------------------------------------------------------
    # Vision & Proactive Checks
    # ------------------------------------------------------------------

    def _reset_vision_episodes(self) -> None:
        self._phone_continuous_sec = 0.0
        self._phone_alert_counted_in_episode = False
        self._absence_continuous_sec = 0.0
        self._camera_unseen_continuous_sec = 0.0
        self._session.camera_unseen_seconds = 0.0
        self._drowsiness_continuous_sec = 0.0

    def _is_phone_signal(self, v: VisionSignalEvent) -> bool:
        if self._config.behavior.require_ai_phone_confirmation:
            return False
        if self._config.behavior.allow_looking_down_for_notes:
            return bool(v.phone_detected)
        return bool(
            v.phone_detected
            or v.looking_down_score > 0.60
            or (not v.looking_at_screen and v.looking_down_score > 0.42)
        )

    def _update_vision_timers(self, dt: float, now: datetime) -> tuple[bool, str, float]:
        """Update phone, camera visibility, and drowsiness timers.

        Returns (physical_distraction_active, reason, drift_dt_earned).
        """
        v = self._latest_vision
        if v is None or not v.camera_available:
            self._reset_vision_episodes()
            return False, "", 0.0

        # Phone / looking down (Leaky Bucket hysteresis)
        phone_signal = self._is_phone_signal(v)
        phone_threshold = float(self._config.behavior.phone_distraction_sec)
        phone_active = False
        phone_drift_dt = 0.0
        if phone_signal:
            prev_phone = self._phone_continuous_sec
            self._phone_continuous_sec += dt
            if self._phone_continuous_sec >= phone_threshold:
                phone_active = True
                phone_drift_dt = (
                    dt
                    if prev_phone >= phone_threshold
                    else max(0.0, self._phone_continuous_sec - phone_threshold)
                )
        else:
            self._phone_continuous_sec = max(0.0, self._phone_continuous_sec - dt * 0.5)
            if self._phone_continuous_sec <= 0.0:
                self._phone_alert_counted_in_episode = False

        # Camera visibility tracker (independent of mouse/keyboard activity)
        unseen_threshold = float(self._config.behavior.camera_not_seeing_user_sec)
        unseen_active = False
        unseen_drift_dt = 0.0
        user_actively_using_pc = (
            self._latest_os is not None and self._latest_os.idle_seconds < 45.0
        )
        if not v.face_present or v.camera_obstructed:
            prev_unseen = self._camera_unseen_continuous_sec
            self._camera_unseen_continuous_sec += dt
            if self._camera_unseen_continuous_sec >= unseen_threshold:
                unseen_active = True
                unseen_drift_dt = (
                    dt
                    if prev_unseen >= unseen_threshold
                    else max(0.0, self._camera_unseen_continuous_sec - unseen_threshold)
                )
        else:
            self._camera_unseen_continuous_sec = max(
                0.0, self._camera_unseen_continuous_sec - dt * 0.5
            )

        self._absence_continuous_sec = self._camera_unseen_continuous_sec
        self._session.camera_unseen_seconds = round(self._camera_unseen_continuous_sec, 1)

        # Drowsiness / yawn (suppress PERCLOS false positives when looking down taking notes)
        looking_down_for_notes = (
            self._config.behavior.allow_looking_down_for_notes
            and v.looking_down_score >= 0.30
        )
        drowsy = (
            (
                not looking_down_for_notes
                and v.drowsiness_perclos >= self._config.behavior.drowsiness_perclos_threshold
            )
            or v.yawn_score > 0.7
        )
        if drowsy:
            self._drowsiness_continuous_sec += dt
        else:
            self._drowsiness_continuous_sec = 0.0

        if phone_active:
            return True, "Uso de celular detectado pela câmera", phone_drift_dt
        if unseen_active:
            if v.camera_obstructed:
                return True, "Câmera tampada ou escura demais — não consigo te ver", unseen_drift_dt
            if user_actively_using_pc:
                return (
                    True,
                    "PC em uso, mas não estou conseguindo te ver na câmera (fora de enquadramento)",
                    unseen_drift_dt,
                )
            return True, "Usuário fora do campo de visão da câmera / ausente da cadeira", unseen_drift_dt
        return False, "", 0.0

    def _is_physical_distraction_active(self) -> bool:
        v = self._latest_vision
        if v is None or not v.camera_available:
            return False
        phone_active = (
            self._is_phone_signal(v)
            and self._phone_continuous_sec >= self._config.behavior.phone_distraction_sec
        )
        unseen_active = (
            (not v.face_present or v.camera_obstructed)
            and self._camera_unseen_continuous_sec
            >= self._config.behavior.camera_not_seeing_user_sec
        )
        return phone_active or unseen_active

    async def _check_fatigue_suggestion(self, now: datetime) -> None:
        if self._drowsiness_continuous_sec < self._config.behavior.drowsiness_window_sec:
            return
        if (
            self._last_fatigue_alert_at is not None
            and (now - self._last_fatigue_alert_at).total_seconds() < _FATIGUE_COOLDOWN_SEC
        ):
            return

        self._session.drowsiness_alerts_count += 1
        self._last_fatigue_alert_at = now
        await self._event_bus.publish(
            ProactiveSuggestionEvent(
                timestamp=now,
                kind="fatigue",
                message=(
                    f"{self._config.user_name}, detectei sinais claros de fadiga "
                    "(olhos pesados ou bocejo). Levanta, bebe água ou faz uma pausa "
                    "de 5 minutos para não render no piloto automático."
                ),
                speak_aloud=True,
            )
        )

    async def _check_pomodoro_suggestion(self, now: datetime) -> None:
        threshold_sec = float(self._config.behavior.pomodoro_focus_suggestion_min * 60)
        if threshold_sec <= 0 or self._current_focus_block_seconds < threshold_sec:
            return
        cooldown = max(300.0, threshold_sec)
        if (
            self._last_pomodoro_at is not None
            and (now - self._last_pomodoro_at).total_seconds() < cooldown
        ):
            return

        self._last_pomodoro_at = now
        await self._event_bus.publish(
            ProactiveSuggestionEvent(
                timestamp=now,
                kind="pomodoro",
                message=(
                    f"{self._config.user_name}, você completou "
                    f"{self._config.behavior.pomodoro_focus_suggestion_min} minutos "
                    "seguidos de foco. Considere fazer uma pausa curta de 5 minutos."
                ),
                speak_aloud=False,
            )
        )

    async def _check_passive_freeze_suggestion(self, now: datetime) -> None:
        threshold_sec = float(self._config.behavior.passive_freeze_alert_sec)
        os_evt = self._latest_os
        if threshold_sec <= 0 or os_evt is None:
            return
        if os_evt.idle_seconds < threshold_sec or os_evt.background_audio_apps:
            return

        up = os_evt.usage_pattern
        is_default_pattern = (
            up.continuous_same_screen_sec == 0.0
            and up.window_switches_60s == 0
            and up.active_input_ratio_60s == 0.0
        )
        if up.continuous_same_screen_sec < threshold_sec and not is_default_pattern:
            return

        cooldown = max(180.0, threshold_sec)
        if (
            self._last_passive_freeze_alert_at is not None
            and (now - self._last_passive_freeze_alert_at).total_seconds() < cooldown
        ):
            return

        self._last_passive_freeze_alert_at = now
        mins = max(1, int(round(threshold_sec / 60.0)))
        win_label = os_evt.window_title[:45] or os_evt.process_name
        await self._event_bus.publish(
            ProactiveSuggestionEvent(
                timestamp=now,
                kind="stuck",
                message=(
                    f"{self._config.user_name}, tela parada há {mins} minutos em "
                    f"'{win_label}' sem áudio e sem interação. Tá estudando "
                    f"{self._session.current_subject} de verdade ou congelou no piloto automático?"
                ),
                speak_aloud=True,
            )
        )

    # ------------------------------------------------------------------
    # OS Context Evaluation & Semantic Flow
    # ------------------------------------------------------------------

    def _is_monitoring_active(self) -> bool:
        if self._privacy_mode:
            return False
        return self._session.current_state in (
            FocusState.FOCUSED,
            FocusState.GRACE_WINDOW,
            FocusState.DRIFT_SUSPECTED,
            FocusState.INTERVENING,
        )

    def _is_sensitive_os(self, os_event: OSContextEvent) -> bool:
        if os_event.is_sensitive_window:
            return True
        sensitive, _ = self._privacy_shield.is_sensitive(
            os_event.process_name, os_event.window_title, os_event.browser_url
        )
        return sensitive

    def _matching_allowance(
        self, os_event: OSContextEvent, now: datetime
    ) -> TemporaryAllowance | None:
        proc = os_event.process_name.lower()
        title = os_event.window_title.lower()
        url = (os_event.browser_url or "").lower()
        for allowance in self._active_allowances:
            if not allowance.is_active(now):
                continue
            pat = allowance.pattern.lower().strip()
            if pat and (pat in proc or pat in title or pat in url):
                return allowance
        return None

    @staticmethod
    def _is_foreground_youtube(os_event: OSContextEvent) -> bool:
        title = os_event.window_title.lower()
        url = (os_event.browser_url or "").lower()
        return "youtube" in title or "youtube.com" in url or "youtu.be" in url

    @staticmethod
    def _is_youtube_shorts(os_event: OSContextEvent) -> bool:
        url = (os_event.browser_url or "").lower()
        title = os_event.window_title.lower()
        return "/shorts" in url or "youtube shorts" in title or "#shorts" in title

    @staticmethod
    def _is_background_youtube(os_event: OSContextEvent) -> bool:
        return any("youtube" in app.lower() for app in os_event.background_audio_apps)

    def _is_high_suspicion(self, os_event: OSContextEvent) -> bool:
        if self._is_foreground_youtube(os_event) and self._is_youtube_shorts(os_event):
            return True
        haystack = " ".join(
            [
                os_event.process_name.lower(),
                os_event.window_title.lower(),
                (os_event.browser_url or "").lower(),
            ]
        )
        return any(
            pat.lower() in haystack
            for pat in self._config.hints.high_suspicion_patterns
            if pat.strip()
        )

    async def _evaluate_os_context_fast_locked(
        self,
        now: datetime,
        prev_tick: datetime | None = None,
        dt: float = 0.0,
        *,
        allow_judge: bool = True,
    ) -> dict[str, Any] | None:
        """Apply fast deterministic OS rules under lock; return a judge request if cloud eval is due."""
        os_event = self._latest_os
        if os_event is None or not self._is_monitoring_active():
            return None

        if self._is_sensitive_os(os_event):
            return None

        sig = self._context_key(os_event)

        # 1. Active temporary allowance overrides suspicion & avoids judge calls.
        allowance = self._matching_allowance(os_event, now)
        if allowance is not None:
            self._allowance_active_for_sig = True
            self._active_allowance_expires_at = _to_utc(allowance.expires_at)
            self._awaiting_debounce = False
            await self._apply_verdict(
                ActivityVerdict.STUDYING,
                f"Liberação temporária ativa ({allowance.reason})",
                now,
            )
            return None

        # If an allowance just expired on the exact same context, split pre-expiry focus time.
        if self._allowance_active_for_sig:
            expired_at = self._active_allowance_expires_at
            if expired_at is not None and prev_tick is not None and expired_at > prev_tick and dt > 0:
                self._pending_pre_expiry_focus_dt = min(
                    dt, (expired_at - prev_tick).total_seconds()
                )
            self._allowance_active_for_sig = False
            self._active_allowance_expires_at = None
            self._context_entered_at = now
            self._last_evaluated_sig = None

        # 2. High suspicion (YouTube Shorts, Instagram, TikTok, etc.): immediate drift.
        if self._is_high_suspicion(os_event):
            self._awaiting_debounce = False
            reason = (
                "YouTube Shorts detectado — distração de alto risco"
                if self._is_youtube_shorts(os_event)
                else f"Conteúdo de alta suspeita: {os_event.window_title[:60]}"
            )
            await self._apply_verdict(ActivityVerdict.DISTRACTED, reason, now)
            return None

        # 3. YouTube ONLY in background audio (not foreground, not high suspicion): MUSIC_ALLOWED.
        is_fg_yt = self._is_foreground_youtube(os_event)
        if not is_fg_yt and self._is_background_youtube(os_event):
            self._awaiting_debounce = False
            self._last_evaluated_sig = sig
            self._last_evaluated_at = now
            await self._apply_verdict(
                ActivityVerdict.MUSIC_ALLOWED,
                "Áudio do YouTube em segundo plano durante estudo",
                now,
            )
            return None

        # 4. Cached positive verdict hit for this multi-monitor context key (never caches DISTRACTED/AWAY).
        cached_verdict = self._session_verdict_cache.get(sig)
        if (
            cached_verdict is not None
            and cached_verdict.verdict in (ActivityVerdict.STUDYING, ActivityVerdict.MUSIC_ALLOWED)
            and self._last_evaluated_sig != sig
        ):
            self._awaiting_debounce = False
            self._last_evaluated_sig = sig
            self._last_evaluated_at = now
            applied = cached_verdict.model_copy(update={"cached": True, "timestamp": now})
            await self._apply_verdict(
                applied.verdict,
                applied.reasoning or f"Veredito em cache ({applied.verdict})",
                now,
                source_verdict=applied,
            )
            return None

        # 5. Foreground non-shorts YouTube: grace window before evaluating video content.
        if is_fg_yt:
            yt_start = self._youtube_entered_at or self._context_entered_at or now
            yt_elapsed = (now - yt_start).total_seconds()
            if (
                self._last_evaluated_sig != sig
                and yt_elapsed < self._config.behavior.youtube_music_grace_sec
            ):
                if self._session.current_state in (
                    FocusState.DRIFT_SUSPECTED,
                    FocusState.INTERVENING,
                ):
                    self._awaiting_debounce = True
                    return None
                self._awaiting_debounce = False
                if (
                    not self._is_physical_distraction_active()
                    and self._session.current_state != FocusState.GRACE_WINDOW
                ):
                    await self._transition_state(
                        FocusState.GRACE_WINDOW,
                        "Janela de tolerância no YouTube para escolha de música",
                        now,
                    )
                return None

            # Grace expired (>= youtube_music_grace_sec) — ensure page render settled before screenshot.
            ctx_elapsed = (now - (self._context_entered_at or now)).total_seconds()
            if (
                self._last_evaluated_sig != sig
                and ctx_elapsed < self._config.ai.window_switch_debounce_sec
            ):
                self._awaiting_debounce = True
                return None
            if allow_judge and not self._judge_in_flight and self._should_evaluate_now(sig, now, os_event):
                self._awaiting_debounce = False
                return self._prepare_judge_request_locked(os_event, sig, now)
            return None

        # 6. Multi-purpose & general apps: if returning from YouTube GRACE_WINDOW with 0 drift,
        # restore FOCUSED smoothly while waiting the debounce.
        if (
            self._session.current_state == FocusState.GRACE_WINDOW
            and self._current_drift_seconds == 0.0
            and not self._is_physical_distraction_active()
        ):
            await self._transition_state(
                FocusState.FOCUSED, "Retorno para janela de estudo", now
            )

        ctx_start = self._context_entered_at or now
        ctx_elapsed = (now - ctx_start).total_seconds()
        if self._last_evaluated_sig != sig:
            if ctx_elapsed < self._config.ai.window_switch_debounce_sec:
                # Pause drift accumulation while awaiting debounce on non-suspicious switch.
                self._awaiting_debounce = True
                return None
            if allow_judge and not self._judge_in_flight:
                self._awaiting_debounce = False
                return self._prepare_judge_request_locked(os_event, sig, now)
            return None

        # Already evaluated on this context: perform adaptive periodic re-check if interval elapsed.
        if allow_judge and not self._judge_in_flight and self._should_evaluate_now(sig, now, os_event):
            self._awaiting_debounce = False
            return self._prepare_judge_request_locked(os_event, sig, now)

        return None

    def _prepare_judge_request_locked(
        self, os_event: OSContextEvent, sig: str, now: datetime
    ) -> dict[str, Any]:
        self._judge_in_flight = True
        self._last_evaluated_sig = sig
        self._last_evaluated_at = now
        return {
            "os_event": os_event,
            "vision_signal": self._latest_vision,
            "sig": sig,
            "now": now,
            "epoch": self._context_epoch,
            "study_goal": self._session.study_goal,
            "current_subject": self._session.current_subject,
            "session_id": self._session.session_id,
            "attach_webcam": (
                self._vision_sensor is not None
                and not self._privacy_mode
                and not self._is_sensitive_os(os_event)
                and self._should_attach_webcam_frame(os_event)
            ),
        }

    def _should_attach_webcam_frame(self, os_event: OSContextEvent) -> bool:
        if self._vision_sensor is None:
            return False
        v = self._latest_vision
        if v is not None and not v.camera_available:
            return False
        if self._config.behavior.always_attach_webcam_to_judge:
            return True
        if v is not None and (
            v.phone_detected or v.looking_down_score >= 0.40 or not v.looking_at_screen
        ):
            return True
        idle_thresh = float(self._config.behavior.idle_webcam_check_sec)
        return (
            os_event.idle_seconds >= idle_thresh
            or os_event.usage_pattern.continuous_same_screen_sec >= idle_thresh
        )

    def _should_evaluate_now(
        self, sig: str, now: datetime, os_event: OSContextEvent | None = None
    ) -> bool:
        if self._last_evaluated_sig != sig or self._last_evaluated_at is None:
            return True
        elapsed = (now - self._last_evaluated_at).total_seconds()
        ambiguous_interval = float(self._config.ai.ambiguous_recheck_interval_sec)
        if (
            self._session.current_state in (FocusState.DRIFT_SUSPECTED, FocusState.INTERVENING)
            or self._last_verdict_value in (ActivityVerdict.UNCERTAIN, ActivityVerdict.DISTRACTED)
        ):
            return elapsed >= ambiguous_interval

        evt = os_event or self._latest_os
        v = self._latest_vision
        if (
            evt is not None
            and evt.idle_seconds >= self._config.behavior.idle_webcam_check_sec
            and v is not None
            and v.camera_available
            and v.face_present
            and elapsed >= ambiguous_interval
        ):
            return True

        return elapsed >= float(self._config.ai.focused_recheck_interval_sec)

    async def _run_judge_request_unlocked(self, req: dict[str, Any]) -> None:
        """Execute SemanticJudge evaluation OUTSIDE _eval_lock, discarding stale results."""
        os_event: OSContextEvent = req["os_event"]
        vision_signal: VisionSignalEvent | None = req["vision_signal"]
        sig: str = req["sig"]
        now: datetime = req["now"]
        eval_epoch: int = req["epoch"]
        study_goal: str = req["study_goal"]
        current_subject: str = req["current_subject"]
        session_id: str = req["session_id"]

        try:
            if not self._semantic_judge.is_configured():
                logger.debug("SemanticJudge not configured; degrading gracefully without cloud call")
                async with self._eval_lock:
                    if eval_epoch == self._context_epoch and sig == self._current_sig and self._is_monitoring_active():
                        await self._apply_offline_fallback(
                            os_event, now, "API Gemini não configurada (modo offline)"
                        )
                return

            if not self._budget.can_spend():
                logger.warning("BudgetGuard throttled; skipping cloud SemanticJudge evaluation")
                async with self._eval_lock:
                    if eval_epoch == self._context_epoch and sig == self._current_sig and self._is_monitoring_active():
                        await self._apply_offline_fallback(
                            os_event, now, "Limite de orçamento atingido (modo local)"
                        )
                return

            screenshot = self.capture_current_screen()
            webcam_jpeg: bytes | None = None
            if req["attach_webcam"] and self._vision_sensor is not None and not self._privacy_mode:
                try:
                    webcam_jpeg = self._vision_sensor.latest_frame_jpeg(
                        max_width=480, quality=65, annotate=False
                    )
                except Exception:
                    logger.exception("VisionSensorPort.latest_frame_jpeg failed gracefully")
                    webcam_jpeg = None

            calls_before = self._budget.get_status().daily_calls
            try:
                verdict = await self._semantic_judge.evaluate(
                    os_context=os_event,
                    vision_signal=vision_signal,
                    study_goal=study_goal,
                    current_subject=current_subject,
                    screenshot=screenshot,
                    webcam_frame_jpeg=webcam_jpeg,
                )
            except Exception:
                logger.exception("SemanticJudge.evaluate failed gracefully")
                async with self._eval_lock:
                    if eval_epoch == self._context_epoch and sig == self._current_sig and self._is_monitoring_active():
                        await self._apply_verdict(
                            ActivityVerdict.UNCERTAIN,
                            "Falha temporária ao consultar juiz semântico",
                            now,
                        )
                return

            calls_after = self._budget.get_status().daily_calls
            if (
                not verdict.cached
                and calls_after == calls_before
                and (verdict.tokens_in > 0 or verdict.tokens_out > 0 or verdict.estimated_cost_usd > 0)
            ):
                cost = self._budget.record_usage(
                    verdict.model_used or self._config.ai.primary_model,
                    verdict.tokens_in,
                    verdict.tokens_out,
                )
                if verdict.estimated_cost_usd == 0.0:
                    verdict = verdict.model_copy(update={"estimated_cost_usd": cost})
                try:
                    await self._storage.save_usage(
                        verdict.model_used or self._config.ai.primary_model,
                        verdict.tokens_in,
                        verdict.tokens_out,
                        verdict.estimated_cost_usd,
                    )
                except Exception:
                    logger.exception("Failed to persist API usage")

            async with self._eval_lock:
                # Discard verdict if user already switched context / paused / acknowledged while Gemini ran.
                if (
                    eval_epoch != self._context_epoch
                    or sig != self._current_sig
                    or not self._is_monitoring_active()
                ):
                    logger.debug(
                        "Discarding stale SemanticVerdict (%s) for superseded context %r",
                        verdict.verdict,
                        os_event.window_title[:50],
                    )
                    return

                is_fg_yt = self._is_foreground_youtube(os_event)
                yt_elapsed = (
                    (now - (self._youtube_entered_at or self._context_entered_at or now)).total_seconds()
                    if is_fg_yt
                    else 0.0
                )
                if (
                    is_fg_yt
                    and verdict.verdict == ActivityVerdict.MUSIC_ALLOWED
                    and yt_elapsed >= float(self._config.behavior.youtube_music_grace_sec)
                ):
                    # User stayed parked on foreground YouTube music past grace window instead of studying.
                    verdict = verdict.model_copy(
                        update={
                            "verdict": ActivityVerdict.DISTRACTED,
                            "reasoning": (
                                f"Parado na aba de música do YouTube em primeiro plano além dos "
                                f"{self._config.behavior.youtube_music_grace_sec}s de escolha."
                            ),
                            "suggested_message": (
                                f"{self._config.user_name}, você já escolheu a música faz tempo. "
                                f"Deixa tocando em segundo plano e volta para {self._session.current_subject} agora!"
                            ),
                        }
                    )

                self._session.recent_verdicts.append(verdict)
                if len(self._session.recent_verdicts) > _MAX_RECENT_VERDICTS:
                    self._session.recent_verdicts = self._session.recent_verdicts[-_MAX_RECENT_VERDICTS:]

                # Only cache positive verdicts (STUDYING / MUSIC_ALLOWED); never cache DISTRACTED/AWAY
                # so closed distractions from the past can never haunt future revisits to the window.
                if (
                    verdict.verdict in (ActivityVerdict.STUDYING, ActivityVerdict.MUSIC_ALLOWED)
                    and verdict.confidence >= 0.6
                    and not verdict.phone_in_use
                    and not (is_fg_yt and verdict.verdict == ActivityVerdict.MUSIC_ALLOWED)
                ):
                    self._session_verdict_cache[sig] = verdict

                await self._apply_verdict(
                    verdict.verdict,
                    verdict.reasoning or f"Juiz semântico: {verdict.verdict}",
                    now,
                    source_verdict=verdict,
                )

            try:
                await self._storage.save_verdict(session_id, verdict)
            except Exception:
                logger.exception("Failed to persist semantic verdict")
        finally:
            async with self._eval_lock:
                self._judge_in_flight = False

    async def _apply_offline_fallback(
        self, os_event: OSContextEvent, now: datetime, fallback_reason: str
    ) -> None:
        haystack = " ".join(
            [
                os_event.process_name.lower(),
                os_event.window_title.lower(),
                (os_event.browser_url or "").lower(),
            ]
        )
        if any(
            kw.lower() in haystack
            for kw in self._config.hints.likely_study_keywords
            if kw.strip()
        ):
            await self._apply_verdict(
                ActivityVerdict.STUDYING,
                "Palavra-chave de estudo detectada (modo offline)",
                now,
            )
            return

        if self._session.current_state in (
            FocusState.DRIFT_SUSPECTED,
            FocusState.INTERVENING,
        ):
            await self._apply_verdict(ActivityVerdict.STUDYING, fallback_reason, now)
        else:
            self._last_verdict_value = ActivityVerdict.UNCERTAIN
            self._last_verdict_reason = fallback_reason

    async def _apply_verdict(
        self,
        verdict_value: ActivityVerdict,
        reason: str,
        now: datetime,
        source_verdict: SemanticVerdict | None = None,
    ) -> None:
        if source_verdict is not None and source_verdict.phone_in_use:
            verdict_value = ActivityVerdict.DISTRACTED
            if not reason:
                reason = "Uso de celular detectado pela câmera"
            self._current_drift_seconds = max(
                self._current_drift_seconds,
                float(self._config.behavior.distraction_level1_sec),
            )
            if not self._phone_alert_counted_in_episode:
                self._session.phone_alerts_count += 1
                self._phone_alert_counted_in_episode = True

        if (
            verdict_value == ActivityVerdict.AWAY
            or (source_verdict is not None and source_verdict.physical_state == "away")
        ):
            self._current_drift_seconds = max(
                self._current_drift_seconds,
                float(self._config.behavior.distraction_level1_sec),
            )

        self._last_verdict_value = verdict_value
        self._last_verdict_reason = reason
        self._last_verdict = source_verdict

        if self._is_physical_distraction_active():
            if self._session.current_state in (FocusState.FOCUSED, FocusState.GRACE_WINDOW):
                await self._transition_state(FocusState.DRIFT_SUSPECTED, reason, now)
            return

        if verdict_value in (ActivityVerdict.DISTRACTED, ActivityVerdict.AWAY):
            if self._session.current_state in (FocusState.FOCUSED, FocusState.GRACE_WINDOW):
                await self._transition_state(FocusState.DRIFT_SUSPECTED, reason, now)
        elif verdict_value in (ActivityVerdict.STUDYING, ActivityVerdict.MUSIC_ALLOWED):
            prev_state = self._session.current_state
            self._reset_drift()
            if prev_state in (FocusState.INTERVENING, FocusState.DRIFT_SUSPECTED):
                self._dismiss_overlay()
                if prev_state == FocusState.INTERVENING:
                    self._stop_speech()
                await self._transition_state(FocusState.FOCUSED, reason, now)
            elif prev_state == FocusState.GRACE_WINDOW:
                await self._transition_state(FocusState.FOCUSED, reason, now)
        else:
            # UNCERTAIN: move FOCUSED -> GRACE_WINDOW gently without escalating drift.
            if self._session.current_state == FocusState.FOCUSED:
                await self._transition_state(FocusState.GRACE_WINDOW, reason, now)

    # ------------------------------------------------------------------
    # Drift Accumulation & Escalation Ladder
    # ------------------------------------------------------------------

    async def _handle_active_drift_locked(
        self,
        dt: float,
        physical_distraction: bool,
        physical_drift_dt: float,
        os_distracted: bool,
        reason: str,
        now: datetime,
        *,
        allow_escalate: bool = True,
    ) -> dict[str, Any] | None:
        # Calculate exact earned drift for this tick.
        if os_distracted and not self._awaiting_debounce:
            pre_drift_focus = min(dt, self._pending_pre_expiry_focus_dt)
            earned_drift = max(0.0, dt - pre_drift_focus)
            self._pending_pre_expiry_focus_dt = 0.0
        else:
            earned_drift = physical_drift_dt
            pre_drift_focus = max(0.0, dt - physical_drift_dt)

        if pre_drift_focus > 0:
            self._session.focus_seconds += pre_drift_focus
            self._record_subject_delta(focus_dt=pre_drift_focus)

        if earned_drift > 0:
            self._session.distraction_seconds += earned_drift
            self._current_drift_seconds += earned_drift
            self._record_subject_delta(distraction_dt=earned_drift)

        self._current_focus_block_seconds = 0.0

        if physical_distraction and not self._phone_alert_counted_in_episode:
            # Count phone alert once per phone episode when escalating.
            v = self._latest_vision
            if v is not None and v.camera_available and self._is_phone_signal(v):
                self._session.phone_alerts_count += 1
                self._phone_alert_counted_in_episode = True

        is_phone_reason = (
            (physical_distraction and "celular" in reason.lower())
            or (
                os_distracted
                and self._last_verdict is not None
                and self._last_verdict.phone_in_use
            )
        )
        reason_low = reason.lower()
        is_unseen_reason = (
            (
                physical_distraction
                and (
                    "câmera" in reason_low
                    or "tampada" in reason_low
                    or "ausente" in reason_low
                    or "enquadramento" in reason_low
                )
            )
            or (os_distracted and self._last_verdict_value == ActivityVerdict.AWAY)
        )
        if is_phone_reason or is_unseen_reason:
            self._current_drift_seconds = max(
                self._current_drift_seconds,
                float(self._config.behavior.distraction_level1_sec),
            )

        if self._session.current_state in (FocusState.FOCUSED, FocusState.GRACE_WINDOW):
            await self._transition_state(FocusState.DRIFT_SUSPECTED, reason, now)

        if not allow_escalate:
            return None
        return await self._prepare_escalation_locked(reason, now)

    async def _prepare_escalation_locked(
        self, reason: str, now: datetime
    ) -> dict[str, Any] | None:
        behavior = self._config.behavior
        if self._current_drift_seconds >= behavior.distraction_level3_sec:
            level = InterventionLevel.FIRM_OVERLAY
            speak_aloud = True
            title = "FocusGuard — Foco Interrompido"
        elif self._current_drift_seconds >= behavior.distraction_level2_sec:
            level = InterventionLevel.VOICE
            speak_aloud = True
            title = "FocusGuard — Chamada de Atenção"
        elif self._current_drift_seconds >= behavior.distraction_level1_sec:
            level = InterventionLevel.SOFT
            speak_aloud = bool(behavior.speak_on_all_interventions)
            title = "FocusGuard — Alerta de Desvio"
        else:
            return None

        cooldown_ok = (
            self._last_intervention_at is None
            or (now - self._last_intervention_at).total_seconds()
            >= behavior.intervention_cooldown_sec
        )
        was_already_intervening = self._session.current_state == FocusState.INTERVENING
        level_increased = level > self._last_dispatched_level
        escalation_upgrade = was_already_intervening and level_increased

        if not was_already_intervening:
            self._session.escalation_level = level
            await self._transition_state(FocusState.INTERVENING, reason, now)
        elif level_increased:
            self._session.escalation_level = level
            await self._event_bus.publish(
                FocusStateChangedEvent(
                    timestamp=now,
                    previous_state=FocusState.INTERVENING,
                    new_state=FocusState.INTERVENING,
                    reason=reason,
                    distraction_seconds=round(self._current_drift_seconds, 2),
                    escalation_level=level,
                    focus_score=self._session.focus_ratio_pct,
                )
            )

        if not (cooldown_ok or escalation_upgrade) or self._intervention_in_flight:
            return None

        self._intervention_in_flight = True
        suggested = (
            self._last_verdict.suggested_message.strip()
            if self._last_verdict is not None and self._last_verdict.suggested_message
            else ""
        )
        return {
            "level": level,
            "title": title,
            "speak_aloud": speak_aloud,
            "reason": reason,
            "now": now,
            "epoch": self._context_epoch,
            "sig": self._current_sig,
            "os_context": self._latest_os,
            "vision_signal": self._latest_vision,
            "study_goal": self._session.study_goal,
            "current_subject": self._session.current_subject,
            "session_id": self._session.session_id,
            "suggested_message": suggested,
        }

    async def _run_escalation_unlocked(self, req: dict[str, Any]) -> None:
        """Build nudge and dispatch audio/UI OUTSIDE _eval_lock, aborting if context changed."""
        level: InterventionLevel = req["level"]
        title: str = req["title"]
        speak_aloud: bool = req["speak_aloud"]
        reason: str = req["reason"]
        now: datetime = req["now"]
        esc_epoch: int = req["epoch"]
        esc_sig: str | None = req["sig"]
        session_id: str = req["session_id"]

        try:
            message = await self._build_intervention_message_from_snapshot(req)

            cmd: InterventionCommand | None = None
            async with self._eval_lock:
                # Abort if the user already changed window / returned to focus / paused while generating nudge.
                if (
                    esc_epoch != self._context_epoch
                    or esc_sig != self._current_sig
                    or not self._is_monitoring_active()
                    or self._session.current_state
                    not in (FocusState.INTERVENING, FocusState.DRIFT_SUSPECTED)
                ):
                    logger.debug(
                        "Aborting stale intervention (%s) because context already changed.",
                        reason[:60],
                    )
                    return

                cmd = InterventionCommand(
                    timestamp=now,
                    level=level,
                    title=title,
                    message=message,
                    speak_aloud=speak_aloud,
                    reason=reason,
                )
                self._last_dispatched_level = level
                self._session.escalation_level = level
                self._last_intervention_at = now
                self._session.interventions_count += 1

            await self._event_bus.publish(cmd)
            if self._dispatcher is not None:
                try:
                    await self._dispatcher.dispatch(cmd)
                except Exception:
                    logger.exception("InterventionDispatcher.dispatch failed gracefully")

            try:
                await self._storage.save_intervention(session_id, cmd)
            except Exception:
                logger.exception("Failed to persist intervention command")
        finally:
            async with self._eval_lock:
                self._intervention_in_flight = False

    async def _build_intervention_message_from_snapshot(self, req: dict[str, Any]) -> str:
        level: InterventionLevel = req["level"]
        reason: str = req["reason"]
        study_goal: str = req["study_goal"]
        current_subject: str = req["current_subject"]
        suggested: str = req["suggested_message"]

        if self._coach_agent is not None and self._coach_agent.is_configured():
            try:
                nudge = await self._coach_agent.generate_nudge(
                    level=level,
                    reason=reason,
                    os_context=req["os_context"],
                    vision_signal=req["vision_signal"],
                    study_goal=study_goal,
                    current_subject=current_subject,
                )
                if nudge and nudge.strip():
                    return nudge.strip()
            except Exception:
                logger.exception("CoachAgent.generate_nudge failed; falling back to default message")

        if suggested:
            return suggested

        return (
            f"{self._config.user_name}, corta essa distração ({reason}). "
            f"Volta agora para {current_subject}."
        )

    # ------------------------------------------------------------------
    # State Helpers
    # ------------------------------------------------------------------

    def _reset_drift(self) -> None:
        self._current_drift_seconds = 0.0
        self._last_dispatched_level = InterventionLevel.NONE
        self._session.escalation_level = InterventionLevel.NONE

    def _dismiss_overlay(self) -> None:
        if self._dispatcher is not None:
            try:
                self._dispatcher.dismiss_overlay()
            except Exception:
                logger.exception("InterventionDispatcher.dismiss_overlay failed gracefully")

    def _stop_speech(self) -> None:
        stop_fn = getattr(self._dispatcher, "stop_speech", None)
        if callable(stop_fn):
            try:
                stop_fn()
            except Exception:
                logger.debug("InterventionDispatcher.stop_speech failed gracefully.", exc_info=True)

    async def _transition_state(
        self, new_state: FocusState, reason: str, now: datetime
    ) -> None:
        prev_state = self._session.current_state
        if prev_state == new_state:
            return
        self._session.current_state = new_state
        self._update_focus_ratio()
        await self._event_bus.publish(
            FocusStateChangedEvent(
                timestamp=now,
                previous_state=prev_state,
                new_state=new_state,
                reason=reason,
                distraction_seconds=round(self._current_drift_seconds, 2),
                escalation_level=self._session.escalation_level,
                focus_score=self._session.focus_ratio_pct,
            )
        )

    def _update_focus_ratio(self) -> None:
        total = self._session.focus_seconds + self._session.distraction_seconds
        if total <= 0.0:
            self._session.focus_ratio_pct = 100.0
            return
        denom = max(1.0, total)
        self._session.focus_ratio_pct = round((self._session.focus_seconds / denom) * 100.0, 2)

    def _record_subject_delta(self, *, focus_dt: float = 0.0, distraction_dt: float = 0.0) -> None:
        subject = (self._session.current_subject or "").strip() or "Geral"
        if focus_dt > 0.0:
            self._subject_focus_pending[subject] = self._subject_focus_pending.get(subject, 0.0) + focus_dt
        if distraction_dt > 0.0:
            self._subject_distraction_pending[subject] = (
                self._subject_distraction_pending.get(subject, 0.0) + distraction_dt
            )

    async def _flush_subject_time_safe(self) -> None:
        add_fn = getattr(self._storage, "add_subject_time", None)
        if not callable(add_fn):
            return
        subjects = set(self._subject_focus_pending) | set(self._subject_distraction_pending)
        if not subjects:
            return
        for subj in list(subjects):
            f_dt = self._subject_focus_pending.get(subj, 0.0)
            d_dt = self._subject_distraction_pending.get(subj, 0.0)
            if f_dt <= 0.0 and d_dt <= 0.0:
                self._subject_focus_pending.pop(subj, None)
                self._subject_distraction_pending.pop(subj, None)
                continue
            try:
                await add_fn(self._session.session_id, subj, f_dt, d_dt)
                self._subject_focus_pending.pop(subj, None)
                self._subject_distraction_pending.pop(subj, None)
            except Exception:
                logger.exception("Failed to flush per-subject study time for %r", subj)

    async def get_subject_study_summary(self) -> list[dict[str, Any]]:
        """Return aggregated study time per subject across history + live session."""
        async with self._eval_lock:
            await self._flush_subject_time_safe()
            current_subject = (self._session.current_subject or "").strip() or "Geral"
            pending_focus = dict(self._subject_focus_pending)
            pending_dist = dict(self._subject_distraction_pending)
            now_iso = self._now().isoformat()

        rows: list[dict[str, Any]] = []
        summary_fn = getattr(self._storage, "get_subject_study_summary", None)
        if callable(summary_fn):
            try:
                raw = await summary_fn()
                if isinstance(raw, list):
                    rows = [dict(item) for item in raw]
            except Exception:
                logger.exception("Failed to query subject study summary from storage")

        # Merge any unflushed in-memory deltas (e.g., if storage lacks subject_study_logs)
        by_key: dict[str, dict[str, Any]] = {}
        for item in rows:
            subj = str(item.get("subject") or "").strip() or "Geral"
            by_key[subj.lower()] = dict(item, subject=subj)

        for subj in set(pending_focus) | set(pending_dist):
            key = subj.lower()
            f_dt = float(pending_focus.get(subj, 0.0))
            d_dt = float(pending_dist.get(subj, 0.0))
            if key not in by_key:
                by_key[key] = {
                    "subject": subj,
                    "total_focus_seconds": 0.0,
                    "today_focus_seconds": 0.0,
                    "total_distraction_seconds": 0.0,
                    "focus_ratio_pct": 100.0,
                    "sessions_count": 1,
                    "last_studied_at": now_iso,
                }
            by_key[key]["total_focus_seconds"] = round(
                float(by_key[key]["total_focus_seconds"]) + f_dt, 1
            )
            by_key[key]["today_focus_seconds"] = round(
                float(by_key[key]["today_focus_seconds"]) + f_dt, 1
            )
            by_key[key]["total_distraction_seconds"] = round(
                float(by_key[key]["total_distraction_seconds"]) + d_dt, 1
            )
            denom = by_key[key]["total_focus_seconds"] + by_key[key]["total_distraction_seconds"]
            by_key[key]["focus_ratio_pct"] = (
                round((by_key[key]["total_focus_seconds"] / denom) * 100.0, 1) if denom > 0 else 100.0
            )

        cur_key = current_subject.lower()
        if cur_key not in by_key:
            by_key[cur_key] = {
                "subject": current_subject,
                "total_focus_seconds": 0.0,
                "today_focus_seconds": 0.0,
                "total_distraction_seconds": 0.0,
                "focus_ratio_pct": 100.0,
                "sessions_count": 1,
                "last_studied_at": now_iso,
            }

        items = list(by_key.values())
        grand_total = sum(max(0.0, float(it["total_focus_seconds"])) for it in items)
        for it in items:
            it["is_current"] = str(it["subject"]).strip().lower() == cur_key
            it["share_pct"] = (
                round((float(it["total_focus_seconds"]) / grand_total) * 100.0, 1)
                if grand_total > 0
                else 0.0
            )

        items.sort(
            key=lambda it: (
                float(it["total_focus_seconds"]),
                float(it["today_focus_seconds"]),
                1 if it["is_current"] else 0,
                str(it["subject"]).lower(),
            ),
            reverse=True,
        )
        return items

    async def _persist_session_safe(self) -> None:
        await self._flush_subject_time_safe()
        try:
            await self._storage.save_session(self._session)
        except Exception:
            logger.exception("Failed to save SessionStats to storage")
