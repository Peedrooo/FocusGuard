"""Deterministic in-memory fakes for all FocusGuard Ports (Wave 0 contract)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from focus_guard.core.budget import BudgetGuard
from focus_guard.core.config import AppConfig
from focus_guard.core.events import (
    ActivityVerdict,
    ChatMessage,
    CoachPersona,
    FocusState,
    InterventionCommand,
    InterventionLevel,
    OSContextEvent,
    ScreenSnapshot,
    SemanticVerdict,
    SessionStats,
    StudyNote,
    SystemStatusSnapshot,
    TemporaryAllowance,
    VisionSignalEvent,
)


class FakeOSSensor:
    def __init__(self, initial: OSContextEvent | None = None) -> None:
        self.current = initial or OSContextEvent(
            process_name="chrome.exe",
            window_title="Tec Concursos - Direito Constitucional",
            browser_url="https://www.tecconcursos.com.br/questoes",
        )
        self.running = False

    async def start(self) -> None:
        self.running = True

    async def stop(self) -> None:
        self.running = False

    def sample_context(self) -> OSContextEvent:
        return self.current


class FakeScreenCapture:
    def __init__(self, jpeg_bytes: bytes = b"\xff\xd8\xff\xd9", phash: str = "0123456789abcdef") -> None:
        self.jpeg_bytes = jpeg_bytes
        self.phash = phash
        self.calls: int = 0

    def capture(self, os_context: OSContextEvent | None = None) -> ScreenSnapshot | None:
        if os_context and os_context.is_sensitive_window:
            return None
        self.calls += 1
        return ScreenSnapshot(
            image_bytes_jpeg=self.jpeg_bytes,
            perceptual_hash=self.phash,
            width=768,
            height=432,
        )


class FakeVisionSensor:
    def __init__(
        self,
        signal: VisionSignalEvent | None = None,
        available: bool = True,
        frame_jpeg: bytes = b"\xff\xd8\xff\xe0FAKEWEBCAM\xff\xd9",
    ) -> None:
        self.signal = signal or VisionSignalEvent(camera_available=available, face_present=True)
        self._available = available
        self._frame_jpeg = frame_jpeg
        self.running = False
        self.calibrations: list[str] = []

    async def start(self) -> None:
        self.running = True

    async def stop(self) -> None:
        self.running = False

    def latest_signal(self) -> VisionSignalEvent | None:
        return self.signal

    def is_available(self) -> bool:
        return self._available

    def latest_frame_jpeg(
        self, max_width: int = 480, quality: int = 65, annotate: bool = False
    ) -> bytes | None:
        return self._frame_jpeg if self._available else None

    async def calibrate_pose(
        self, mode: str = "primary_screen", duration_sec: float = 2.0
    ) -> dict[str, Any]:
        self.calibrations.append(mode)
        return {"calibrated": True, "mode": mode, "screen_yaw_min_deg": -55.0, "screen_yaw_max_deg": 25.0}


class FakeSemanticJudge:
    def __init__(
        self,
        default_verdict: ActivityVerdict = ActivityVerdict.STUDYING,
        configured: bool = True,
    ) -> None:
        self.default_verdict = default_verdict
        self._configured = configured
        self.calls: list[dict[str, Any]] = []
        self.scripted_verdicts: list[SemanticVerdict] = []

    def is_configured(self) -> bool:
        return self._configured

    def reconfigure_api_key(self, api_key: str) -> None:
        self._configured = bool(api_key.strip())

    async def evaluate(
        self,
        os_context: OSContextEvent,
        vision_signal: VisionSignalEvent | None,
        study_goal: str,
        current_subject: str,
        screenshot: ScreenSnapshot | None = None,
        webcam_frame_jpeg: bytes | None = None,
    ) -> SemanticVerdict:
        self.calls.append(
            {
                "os_context": os_context,
                "vision_signal": vision_signal,
                "study_goal": study_goal,
                "current_subject": current_subject,
                "used_vision": screenshot is not None,
                "used_webcam": webcam_frame_jpeg is not None,
            }
        )
        if self.scripted_verdicts:
            return self.scripted_verdicts.pop(0)
        return SemanticVerdict(
            verdict=self.default_verdict,
            confidence=0.9,
            category="fake",
            app_context=os_context.window_title,
            reasoning="Fake judge verdict",
            suggested_message="Bora focar, Pedro." if self.default_verdict == ActivityVerdict.DISTRACTED else "",
            used_vision=screenshot is not None,
            used_webcam=webcam_frame_jpeg is not None,
            model_used="fake-gemini",
            tokens_in=500,
            tokens_out=80,
            estimated_cost_usd=0.0003,
        )


class FakeTTS:
    def __init__(self) -> None:
        self.spoken: list[str] = []

    async def synthesize(self, text: str) -> tuple[bytes, str]:
        return (f"FAKEAUDIO:{text}".encode(), "audio/mpeg")

    async def speak(self, text: str) -> bool:
        self.spoken.append(text)
        return True

    def stop(self) -> None:
        pass


class FakeMicRecorder:
    def __init__(self, available: bool = True, audio_bytes: bytes = b"RIFFfakewav") -> None:
        self._available = available
        self.audio_bytes = audio_bytes

    def is_available(self) -> bool:
        return self._available

    async def record_until_silence(
        self,
        max_seconds: float | None = None,
        silence_timeout: float | None = None,
    ) -> bytes | None:
        return self.audio_bytes if self._available else None


class FakeCoachAgent:
    def __init__(self, configured: bool = True) -> None:
        self._configured = configured
        self._history: list[ChatMessage] = []

    def is_configured(self) -> bool:
        return self._configured

    def reconfigure_api_key(self, api_key: str) -> None:
        self._configured = bool(api_key.strip())

    async def reply_text(self, user_text: str, speak: bool = True) -> ChatMessage:
        self._history.append(ChatMessage(role="user", text=user_text))
        reply = ChatMessage(
            role="coach",
            text=f"Entendido, Pedro. Sem enrolação: foco em '{user_text[:40]}'.",
        )
        self._history.append(reply)
        return reply

    async def reply_audio(
        self, audio_bytes: bytes, mime_type: str = "audio/wav", speak: bool = True
    ) -> tuple[str, ChatMessage]:
        transcript = "Fala coach, preciso de ajuda."
        reply = await self.reply_text(transcript, speak=speak)
        return transcript, reply

    async def generate_nudge(
        self,
        level: InterventionLevel,
        reason: str,
        os_context: OSContextEvent | None,
        vision_signal: VisionSignalEvent | None,
        study_goal: str,
        current_subject: str,
    ) -> str:
        return f"Pedro, para de enrolar ({reason}). Volta pra {current_subject} agora."

    def get_history(self) -> list[ChatMessage]:
        return list(self._history)

    def clear_history(self) -> None:
        self._history.clear()


class FakeStorage:
    def __init__(self) -> None:
        self.sessions: list[SessionStats] = []
        self.verdicts: list[tuple[str, SemanticVerdict]] = []
        self.interventions: list[tuple[str, InterventionCommand]] = []
        self.usages: list[tuple[str, int, int, float]] = []
        self.chats: list[ChatMessage] = []
        self.notes: list[StudyNote] = []
        self.allowances: list[TemporaryAllowance] = []

    async def init(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def save_session(self, stats: SessionStats) -> None:
        self.sessions.append(stats.model_copy(deep=True))

    async def save_verdict(self, session_id: str, verdict: SemanticVerdict) -> None:
        self.verdicts.append((session_id, verdict))

    async def save_intervention(self, session_id: str, cmd: InterventionCommand) -> None:
        self.interventions.append((session_id, cmd))

    async def save_usage(
        self, model: str, tokens_in: int, tokens_out: int, cost_usd: float
    ) -> None:
        self.usages.append((model, tokens_in, tokens_out, cost_usd))

    async def get_usage_totals(self) -> tuple[float, float, int, int, int]:
        total_cost = sum(u[3] for u in self.usages)
        total_in = sum(u[1] for u in self.usages)
        total_out = sum(u[2] for u in self.usages)
        return total_cost, total_cost, total_in, total_out, len(self.usages)

    async def save_chat_message(self, session_id: str, msg: ChatMessage) -> None:
        self.chats.append(msg)

    async def list_chat_messages(self, limit: int = 50) -> list[ChatMessage]:
        return self.chats[-limit:]

    async def save_note(self, note: StudyNote) -> None:
        self.notes.append(note)

    async def list_notes(self, limit: int = 50) -> list[StudyNote]:
        return self.notes[-limit:]

    async def save_allowance(self, allowance: TemporaryAllowance) -> None:
        self.allowances.append(allowance)

    async def list_active_allowances(self) -> list[TemporaryAllowance]:
        now = datetime.now(UTC)
        return [a for a in self.allowances if a.is_active(now)]


class FakeInterventionDispatcher:
    def __init__(self) -> None:
        self.dispatched: list[InterventionCommand] = []
        self.overlay_dismissed: int = 0

    async def dispatch(self, command: InterventionCommand) -> None:
        self.dispatched.append(command)

    def dismiss_overlay(self) -> None:
        self.overlay_dismissed += 1


class FakeFocusEngine:
    """Minimal FocusEnginePort fake for UI and Coach Agent unit tests."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig()
        self.budget = BudgetGuard(self.config.ai)
        self.running = False
        self.privacy_mode = False
        self.session = SessionStats(
            study_goal=self.config.study_goal,
            current_subject=self.config.current_subject,
            current_state=FocusState.PAUSED,
        )
        self.latest_os: OSContextEvent | None = None
        self.latest_vision: VisionSignalEvent | None = None
        self.allowances: list[TemporaryAllowance] = []
        self.notes: list[StudyNote] = []
        self.screen_capture = FakeScreenCapture()

    async def start(self) -> None:
        self.running = True

    async def stop(self) -> None:
        self.running = False

    async def start_session(
        self, study_goal: str | None = None, subject: str | None = None
    ) -> SessionStats:
        if study_goal:
            self.session.study_goal = study_goal
        if subject:
            self.session.current_subject = subject
        self.running = True
        self.session.current_state = FocusState.FOCUSED
        return self.session

    async def pause_session(self) -> SessionStats:
        self.session.current_state = FocusState.PAUSED
        return self.session

    async def toggle_privacy(self, enabled: bool | None = None) -> bool:
        self.privacy_mode = (not self.privacy_mode) if enabled is None else enabled
        self.session.current_state = (
            FocusState.PRIVACY_MODE if self.privacy_mode else FocusState.FOCUSED
        )
        return self.privacy_mode

    async def set_study_goal(self, goal: str, subject: str) -> SessionStats:
        self.session.study_goal = goal
        self.session.current_subject = subject
        return self.session

    async def start_break(self, minutes: int = 5) -> SessionStats:
        self.session.current_state = FocusState.BREAK_MODE
        self.session.active_break_remaining_sec = minutes * 60
        return self.session

    async def end_break(self) -> SessionStats:
        self.session.current_state = FocusState.FOCUSED
        self.session.active_break_remaining_sec = 0
        return self.session

    async def grant_allowance(
        self, pattern: str, minutes: int, reason: str
    ) -> TemporaryAllowance:
        now = datetime.now(UTC)
        allowance = TemporaryAllowance(
            pattern=pattern,
            reason=reason,
            created_at=now,
            expires_at=now + timedelta(minutes=minutes),
        )
        self.allowances.append(allowance)
        return allowance

    async def acknowledge_back_to_focus(self) -> None:
        self.session.current_state = FocusState.FOCUSED
        self.session.escalation_level = InterventionLevel.NONE

    async def add_study_note(self, note: str, subject: str | None = None) -> StudyNote:
        item = StudyNote(subject=subject or self.session.current_subject, note=note)
        self.notes.append(item)
        return item

    async def list_study_notes(self) -> list[StudyNote]:
        return list(self.notes)

    async def handle_os_event(self, event: OSContextEvent) -> None:
        self.latest_os = event

    async def handle_vision_event(self, event: VisionSignalEvent) -> None:
        self.latest_vision = event

    def capture_current_screen(self) -> ScreenSnapshot | None:
        return self.screen_capture.capture(self.latest_os)

    def get_status(self) -> SystemStatusSnapshot:
        return SystemStatusSnapshot(
            running=self.running,
            privacy_mode=self.privacy_mode,
            api_key_configured=bool(self.config.gemini_api_key),
            camera_available=True,
            mic_available=True,
            coach_persona=CoachPersona(self.config.coach_persona),
            session=self.session,
            budget=self.budget.get_status(),
            latest_os=self.latest_os,
            latest_vision=self.latest_vision,
            active_allowances=[a for a in self.allowances if a.is_active()],
        )
