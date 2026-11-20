"""Dependency Injection container wiring all FocusGuard subsystems."""

from __future__ import annotations

import asyncio
import logging
import webbrowser
from collections.abc import Callable

from focus_guard.ai import (
    EdgeAndOfflineTTS,
    FirmCoachAgent,
    GeminiClient,
    GeminiSemanticJudge,
    SoundDeviceMicRecorder,
)
from focus_guard.core.budget import BudgetGuard
from focus_guard.core.config import AppConfig
from focus_guard.core.event_bus import EventBus
from focus_guard.core.events import (
    ChatMessage,
    FocusState,
    FocusStateChangedEvent,
    ProactiveSuggestionEvent,
)
from focus_guard.core.privacy import PrivacyShield
from focus_guard.engine import FocusEngine, SQLiteStorage
from focus_guard.sensors.os_monitor import MSSScreenCapture, WindowsOSSensor
from focus_guard.sensors.vision import MediaPipeVisionSensor
from focus_guard.ui import (
    SystemTrayController,
    WindowsInterventionDispatcher,
    create_api_app,
)

logger = logging.getLogger(__name__)


class AppContainer:
    """Wires and manages the full FocusGuard runtime lifecycle."""

    def __init__(
        self,
        config: AppConfig,
        *,
        enable_tray: bool = True,
        enable_ui: bool = True,
        enable_camera: bool | None = None,
        auto_start_session: bool = True,
        on_request_exit: Callable[[], None] | None = None,
    ) -> None:
        self.config = config
        self._auto_start_session = auto_start_session
        self._on_request_exit = on_request_exit
        self._ptt_lock = asyncio.Lock()
        self._speech_tasks: set[asyncio.Task[None]] = set()
        self._started = False

        # Core infrastructure
        self.event_bus = EventBus()
        self.privacy_shield = PrivacyShield(config.privacy)
        self.budget = BudgetGuard(config.ai)
        self.storage = SQLiteStorage(config.db_path)

        # Sensors
        vision_cfg = config
        if enable_camera is not None and enable_camera != config.vision.enabled:
            vision_cfg = config.model_copy(deep=True)
            vision_cfg.vision.enabled = enable_camera

        self.os_sensor = WindowsOSSensor(
            config=config,
            event_bus=self.event_bus,
            privacy_shield=self.privacy_shield,
        )
        self.screen_capture = MSSScreenCapture(
            config=config,
            privacy_shield=self.privacy_shield,
        )
        self.vision_sensor = MediaPipeVisionSensor(
            config=vision_cfg,
            event_bus=self.event_bus,
        )

        # AI & Voice
        self.gemini_client = GeminiClient(
            config=config,
            budget=self.budget,
            storage=self.storage,
        )
        self.semantic_judge = GeminiSemanticJudge(
            gemini_client=self.gemini_client,
            config=config,
        )
        self.tts = EdgeAndOfflineTTS(config=config)
        self.mic = SoundDeviceMicRecorder(config=config, event_bus=self.event_bus)

        # Dispatcher & Focus Engine
        self.dispatcher = WindowsInterventionDispatcher(
            tts=self.tts,
            on_ack_focus=self._handle_ack_focus,
            on_break_5m=self._handle_break_5m,
            on_open_coach=self.open_dashboard,
            enable_ui=enable_ui,
            show_windows_toast=config.behavior.show_windows_toast,
            show_overlay=config.behavior.show_level3_overlay,
        )
        self.engine = FocusEngine(
            config=config,
            event_bus=self.event_bus,
            budget=self.budget,
            storage=self.storage,
            screen_capture=self.screen_capture,
            semantic_judge=self.semantic_judge,
            coach_agent=None,
            dispatcher=self.dispatcher,
            vision_sensor=self.vision_sensor,
        )
        self.coach = FirmCoachAgent(
            gemini_client=self.gemini_client,
            engine=self.engine,
            tts=self.tts,
            config=config,
            event_bus=self.event_bus,
            storage=self.storage,
        )
        self.engine.set_coach_agent(self.coach)
        self.engine.set_mic_available(self.mic.is_available())

        # System Tray & Global Hotkeys
        self.tray = SystemTrayController(
            on_toggle_session=self._handle_toggle_session,
            on_toggle_privacy=self._handle_toggle_privacy,
            on_talk_coach=self.trigger_ptt_conversation,
            on_open_dashboard=self.open_dashboard,
            on_quit=self._handle_quit,
            hotkey_privacy=config.voice.hotkey_privacy,
            hotkey_talk=config.voice.hotkey_talk,
            enable_tray=enable_tray,
            enable_hotkeys=enable_tray,
        )

        # FastAPI & WebSocket Dashboard
        self.api_app = create_api_app(
            engine=self.engine,
            coach=self.coach,
            judge=self.semantic_judge,
            tts=self.tts,
            mic=self.mic,
            event_bus=self.event_bus,
            config=self.config,
            dispatcher=self.dispatcher,
            vision_sensor=self.vision_sensor,
            on_ptt_trigger=self.trigger_ptt_conversation,
        )

        self._subscribe_container_handlers()

    def _subscribe_container_handlers(self) -> None:
        self.event_bus.subscribe(FocusStateChangedEvent, self._on_focus_state_changed)
        self.event_bus.subscribe(ProactiveSuggestionEvent, self._on_proactive_suggestion)

    async def _on_focus_state_changed(self, event: FocusStateChangedEvent) -> None:
        self.tray.update_state(event.new_state, tooltip=f"FocusGuard — {event.new_state.value}")

    async def _on_proactive_suggestion(self, event: ProactiveSuggestionEvent) -> None:
        if not (event.speak_aloud and event.message.strip()):
            return
        message = event.message.strip()

        async def _speak_bg() -> None:
            try:
                await self.tts.speak(message)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Failed to speak proactive suggestion.", exc_info=True)

        task = asyncio.create_task(_speak_bg(), name="focus_guard_proactive_tts")
        self._speech_tasks.add(task)
        task.add_done_callback(self._speech_tasks.discard)

    def set_exit_callback(self, callback: Callable[[], None]) -> None:
        self._on_request_exit = callback

    @property
    def dashboard_url(self) -> str:
        return f"http://{self.config.server.host}:{self.config.server.port}"

    async def start(self) -> None:
        """Start storage, focus engine, sensors, tray, and optionally begin study session."""
        if self._started:
            return
        self._started = True

        await self.engine.start()
        await self.os_sensor.start()
        await self.vision_sensor.start()

        try:
            loop = asyncio.get_running_loop()
            self.tray.start(loop=loop)
        except RuntimeError:
            self.tray.start()

        if self._auto_start_session:
            stats = await self.engine.start_session()
            self.tray.update_state(stats.current_state)
        else:
            self.tray.update_state(self.engine.get_status().session.current_state)

        logger.info("FocusGuard runtime started cleanly at %s", self.dashboard_url)

    async def stop(self) -> None:
        """Gracefully shut down all subsystems without losing persisted data."""
        if not self._started:
            return
        self._started = False

        self.tray.stop()
        self.dispatcher.dismiss_overlay()
        self.tts.stop()
        for task in list(self._speech_tasks):
            if not task.done():
                task.cancel()
        self._speech_tasks.clear()

        for stopper in (self.vision_sensor.stop, self.os_sensor.stop, self.engine.stop, self.storage.close):
            try:
                await stopper()
            except Exception:
                logger.exception("Error while stopping subsystem %r", stopper)

        logger.info("FocusGuard runtime stopped cleanly.")

    # ------------------------------------------------------------------
    # Tray / Overlay / Hotkey Actions
    # ------------------------------------------------------------------

    async def _handle_ack_focus(self) -> None:
        await self.engine.acknowledge_back_to_focus()

    async def _handle_break_5m(self) -> None:
        await self.engine.start_break(minutes=5)

    async def _handle_toggle_session(self) -> None:
        state = self.engine.get_status().session.current_state
        if state == FocusState.PAUSED:
            await self.engine.start_session()
        else:
            await self.engine.pause_session()

    async def _handle_toggle_privacy(self) -> None:
        await self.engine.toggle_privacy()

    def _handle_quit(self) -> None:
        if self._on_request_exit is not None:
            try:
                self._on_request_exit()
            except Exception:
                logger.exception("Exit callback raised an error.")

    def open_dashboard(self) -> None:
        try:
            webbrowser.open(self.dashboard_url)
        except Exception:
            logger.warning("Could not open browser automatically at %s", self.dashboard_url)

    async def trigger_ptt_conversation(self) -> None:
        """Record from native Windows mic until silence and reply locally + in dashboard."""
        if self._ptt_lock.locked():
            logger.debug("PTT conversation already in progress; ignoring duplicate trigger.")
            return

        async with self._ptt_lock:
            if not self.mic.is_available():
                msg = ChatMessage(
                    role="system",
                    text="Microfone nativo indisponível. Use o microfone do navegador ou digite no chat.",
                )
                await self.event_bus.publish(msg)
                return

            prev_state = self.engine.get_status().session.current_state
            entered_conversation = False
            if prev_state not in (FocusState.PAUSED, FocusState.PRIVACY_MODE, FocusState.BREAK_MODE):
                try:
                    await self.engine.set_conversation_mode(True)
                    entered_conversation = True
                except Exception:
                    logger.debug("Could not enter CONVERSATION_MODE cleanly.", exc_info=True)

            try:
                self.tts.stop()
                await self.event_bus.publish(
                    ChatMessage(role="system", text="Ouvindo seu microfone... fale agora.")
                )
                audio_bytes = await self.mic.record_until_silence()
                if not audio_bytes:
                    await self.event_bus.publish(
                        ChatMessage(role="system", text="Nenhuma fala detectada no microfone.")
                    )
                    return

                await self.coach.reply_audio(
                    audio_bytes,
                    mime_type="audio/wav",
                    speak=True,
                    local_playback=True,
                )
            except Exception as exc:
                logger.exception("Native PTT conversation failed: %s", exc)
                await self.event_bus.publish(
                    ChatMessage(role="system", text="Erro ao processar conversa por voz.")
                )
            finally:
                if entered_conversation:
                    try:
                        await self.engine.set_conversation_mode(False)
                    except Exception:
                        logger.debug("Could not leave CONVERSATION_MODE cleanly.", exc_info=True)
