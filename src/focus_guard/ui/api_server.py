"""FastAPI REST & WebSocket server for FocusGuard local dashboard."""

from __future__ import annotations

import asyncio
import base64
import inspect
import logging
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from focus_guard.core.config import AppConfig, save_api_key_to_env, save_user_config
from focus_guard.core.event_bus import EventBus
from focus_guard.core.events import (
    CoachPersona,
    FocusState,
    InterventionCommand,
    InterventionLevel,
)
from focus_guard.core.ports import (
    CoachAgentPort,
    FocusEnginePort,
    InterventionDispatcherPort,
    MicRecorderPort,
    SemanticJudgePort,
    TTSPort,
    VisionSensorPort,
)

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent / "web"


class SessionStartRequest(BaseModel):
    study_goal: str | None = None
    current_subject: str | None = None


class BreakRequest(BaseModel):
    minutes: int = Field(default=5, ge=1, le=180)


class PrivacyToggleRequest(BaseModel):
    enabled: bool | None = None


class GoalRequest(BaseModel):
    study_goal: str = Field(min_length=1, max_length=200)
    current_subject: str = Field(min_length=1, max_length=120)


class AllowanceRequest(BaseModel):
    pattern: str = Field(min_length=1, max_length=160)
    minutes: int = Field(default=5, ge=1, le=240)
    reason: str = Field(default="Liberado pelo usuário", max_length=240)


class NoteRequest(BaseModel):
    note: str = Field(min_length=1, max_length=2000)
    subject: str | None = Field(default=None, max_length=120)


class ChatTextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    speak: bool = True


class PTTRequest(BaseModel):
    speak: bool = True
    max_seconds: float | None = Field(default=None, gt=0.5, le=60.0)
    silence_timeout: float | None = Field(default=None, gt=0.2, le=10.0)


class ApiKeyRequest(BaseModel):
    api_key: str = Field(min_length=1, max_length=512)


class SettingsRequest(BaseModel):
    coach_persona: CoachPersona | None = None
    max_daily_usd: float | None = Field(default=None, gt=0.0, le=100.0)
    max_monthly_usd: float | None = Field(default=None, gt=0.0, le=500.0)
    youtube_music_grace_sec: int | None = Field(default=None, ge=0, le=600)
    speak_on_all_interventions: bool | None = None
    phone_distraction_sec: int | None = Field(default=None, ge=1, le=300)
    camera_not_seeing_user_sec: int | None = Field(default=None, ge=5, le=180)
    passive_freeze_alert_sec: int | None = Field(default=None, ge=0, le=1800)
    capture_mode: Literal["all_monitors", "primary_monitor", "active_monitor"] | None = None


class VisionCalibrateRequest(BaseModel):
    mode: Literal["primary_screen", "secondary_screen", "phone_pose", "reset"] = "primary_screen"
    duration_sec: float = Field(default=2.0, ge=0.2, le=15.0)


class AudioDevicesRequest(BaseModel):
    output_device_name: str | None = None
    input_device_name: str | None = None


class AudioTestRequest(BaseModel):
    text: str = Field(
        default="Teste de áudio do FocusGuard. Se você está ouvindo isso no fone, está tudo certo.",
        min_length=1,
        max_length=500,
    )


class TestNudgeRequest(BaseModel):
    level: int = Field(default=2, ge=1, le=3)


class WebSocketManager:
    """Manages active dashboard WebSocket connections and EventBus forwarding."""

    def __init__(self, event_bus: EventBus, engine: FocusEnginePort) -> None:
        self._event_bus = event_bus
        self._engine = engine
        self._connections: set[WebSocket] = set()
        self._subscribed = False

    def attach(self) -> None:
        if not self._subscribed:
            self._event_bus.subscribe_all(self._on_bus_event)
            self._subscribed = True

    def detach(self) -> None:
        if self._subscribed:
            self._event_bus.unsubscribe_all(self._on_bus_event)
            self._subscribed = False

    async def connect(self, websocket: WebSocket) -> None:
        self.attach()
        await websocket.accept()
        self._connections.add(websocket)
        try:
            status = self._engine.get_status()
            await websocket.send_json(
                {
                    "type": type(status).__name__,
                    "data": status.model_dump(mode="json"),
                }
            )
        except Exception:
            self.disconnect(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        self._connections.discard(websocket)

    async def broadcast(self, payload: dict[str, Any]) -> None:
        if not self._connections:
            return
        dead: list[WebSocket] = []
        for ws in list(self._connections):
            try:
                await asyncio.wait_for(ws.send_json(payload), timeout=1.0)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._connections.discard(ws)

    async def _on_bus_event(self, event: Any) -> None:
        if isinstance(event, BaseModel):
            data: Any = event.model_dump(mode="json")
        elif isinstance(event, dict):
            data = event
        else:
            data = {"value": str(event)}
        await self.broadcast({"type": type(event).__name__, "data": data})


def _status_payload(engine: FocusEnginePort) -> dict[str, Any]:
    return engine.get_status().model_dump(mode="json")


def _settings_payload(config: AppConfig) -> dict[str, Any]:
    return {
        "coach_persona": str(config.coach_persona),
        "max_daily_usd": config.ai.max_daily_usd,
        "max_monthly_usd": config.ai.max_monthly_usd,
        "youtube_music_grace_sec": config.behavior.youtube_music_grace_sec,
        "speak_on_all_interventions": config.behavior.speak_on_all_interventions,
        "phone_distraction_sec": config.behavior.phone_distraction_sec,
        "camera_not_seeing_user_sec": config.behavior.camera_not_seeing_user_sec,
        "passive_freeze_alert_sec": config.behavior.passive_freeze_alert_sec,
        "capture_mode": config.ai.capture_mode,
    }


def _safe_device_list(obj: Any, method_name: str) -> list[Any]:
    fn = getattr(obj, method_name, None)
    if not callable(fn):
        return []
    try:
        res = fn()
        return list(res) if res is not None else []
    except Exception:
        logger.warning("Failed calling %s on %s.", method_name, type(obj).__name__, exc_info=True)
        return []


def create_api_app(
    engine: FocusEnginePort,
    coach: CoachAgentPort,
    judge: SemanticJudgePort,
    tts: TTSPort,
    mic: MicRecorderPort,
    event_bus: EventBus,
    config: AppConfig,
    dispatcher: InterventionDispatcherPort | None = None,
    on_ptt_trigger: Callable[[], Awaitable[None]] | None = None,
    vision_sensor: VisionSensorPort | None = None,
) -> FastAPI:
    """Create and configure the FocusGuard FastAPI application."""
    ws_manager = WebSocketManager(event_bus=event_bus, engine=engine)
    ws_manager.attach()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        ws_manager.attach()
        try:
            yield
        finally:
            ws_manager.detach()

    app = FastAPI(title="FocusGuard API", version="0.1.0", lifespan=lifespan)
    app.state.ws_manager = ws_manager

    def _safe_stop_tts() -> None:
        try:
            tts.stop()
        except Exception:
            logger.debug("Ignoring TTS stop error.", exc_info=True)

    async def _safe_subject_stats() -> list[dict[str, Any]]:
        fn = getattr(engine, "get_subject_study_summary", None)
        if callable(fn):
            try:
                res = fn()
                if inspect.isawaitable(res):
                    res = await res
                if isinstance(res, list):
                    return res
            except Exception:
                logger.warning("Failed to fetch subject study summary.", exc_info=True)
        sess = engine.get_status().session
        focus_sec = round(float(sess.focus_seconds), 1)
        dist_sec = round(float(sess.distraction_seconds), 1)
        denom = focus_sec + dist_sec
        return [
            {
                "subject": (sess.current_subject or "").strip() or "Geral",
                "total_focus_seconds": focus_sec,
                "today_focus_seconds": focus_sec,
                "total_distraction_seconds": dist_sec,
                "focus_ratio_pct": round((focus_sec / denom) * 100.0, 1) if denom > 0 else 100.0,
                "sessions_count": 1,
                "share_pct": 100.0 if focus_sec > 0 else 0.0,
                "last_studied_at": sess.started_at.isoformat(),
                "is_current": True,
            }
        ]

    if WEB_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        index_path = WEB_DIR / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=404, detail="Dashboard index.html não encontrado.")
        return FileResponse(index_path)

    @app.get("/api/status")
    async def get_status() -> dict[str, Any]:
        notes = await engine.list_study_notes()
        subject_stats = await _safe_subject_stats()
        return {
            "status": _status_payload(engine),
            "settings": _settings_payload(config),
            "chat_history": [m.model_dump(mode="json") for m in coach.get_history()[-30:]],
            "notes": [n.model_dump(mode="json") for n in notes],
            "subject_stats": subject_stats,
        }

    @app.get("/api/stats/subjects")
    async def get_subject_stats() -> dict[str, Any]:
        items = await _safe_subject_stats()
        total_focus = round(sum(float(it.get("total_focus_seconds", 0.0)) for it in items), 1)
        today_focus = round(sum(float(it.get("today_focus_seconds", 0.0)) for it in items), 1)
        return {
            "subjects": items,
            "total_focus_seconds": total_focus,
            "today_focus_seconds": today_focus,
            "current_subject": engine.get_status().session.current_subject,
        }

    @app.post("/api/session/start")
    async def start_session(payload: SessionStartRequest | None = None) -> dict[str, Any]:
        goal = payload.study_goal.strip() if payload and payload.study_goal else None
        subject = payload.current_subject.strip() if payload and payload.current_subject else None
        stats = await engine.start_session(study_goal=goal, subject=subject)
        return {
            "ok": True,
            "session": stats.model_dump(mode="json"),
            "status": _status_payload(engine),
        }

    @app.post("/api/session/pause")
    async def pause_session() -> dict[str, Any]:
        _safe_stop_tts()
        stats = await engine.pause_session()
        return {
            "ok": True,
            "session": stats.model_dump(mode="json"),
            "status": _status_payload(engine),
        }

    @app.post("/api/session/break")
    async def start_break(payload: BreakRequest | None = None) -> dict[str, Any]:
        _safe_stop_tts()
        minutes = payload.minutes if payload is not None else 5
        stats = await engine.start_break(minutes=minutes)
        if dispatcher is not None:
            dispatcher.dismiss_overlay()
        return {
            "ok": True,
            "session": stats.model_dump(mode="json"),
            "status": _status_payload(engine),
        }

    @app.post("/api/session/end-break")
    async def end_break() -> dict[str, Any]:
        stats = await engine.end_break()
        return {
            "ok": True,
            "session": stats.model_dump(mode="json"),
            "status": _status_payload(engine),
        }

    @app.post("/api/session/ack-focus")
    async def ack_focus() -> dict[str, Any]:
        _safe_stop_tts()
        await engine.acknowledge_back_to_focus()
        if dispatcher is not None:
            dispatcher.dismiss_overlay()
        return {"ok": True, "status": _status_payload(engine)}

    @app.post("/api/privacy/toggle")
    async def toggle_privacy(payload: PrivacyToggleRequest | None = None) -> dict[str, Any]:
        enabled = payload.enabled if payload is not None else None
        privacy_mode = await engine.toggle_privacy(enabled=enabled)
        if privacy_mode:
            _safe_stop_tts()
            if dispatcher is not None:
                dispatcher.dismiss_overlay()
        return {
            "ok": True,
            "privacy_mode": privacy_mode,
            "status": _status_payload(engine),
        }

    @app.post("/api/goal")
    async def update_goal(payload: GoalRequest) -> dict[str, Any]:
        goal = payload.study_goal.strip()
        subject = payload.current_subject.strip()
        if not goal or not subject:
            raise HTTPException(status_code=400, detail="Concurso e matéria são obrigatórios.")
        stats = await engine.set_study_goal(goal=goal, subject=subject)
        config.study_goal = goal
        config.current_subject = subject
        subject_stats = await _safe_subject_stats()
        return {
            "ok": True,
            "session": stats.model_dump(mode="json"),
            "status": _status_payload(engine),
            "subject_stats": subject_stats,
        }

    @app.post("/api/allowance")
    async def grant_allowance(payload: AllowanceRequest) -> dict[str, Any]:
        pattern = payload.pattern.strip()
        if not pattern:
            raise HTTPException(status_code=400, detail="Padrão de app/site é obrigatório.")
        reason = payload.reason.strip() or "Liberado pelo usuário"
        allowance = await engine.grant_allowance(
            pattern=pattern,
            minutes=payload.minutes,
            reason=reason,
        )
        return {
            "ok": True,
            "allowance": allowance.model_dump(mode="json"),
            "status": _status_payload(engine),
        }

    @app.get("/api/notes")
    async def list_notes() -> dict[str, Any]:
        notes = await engine.list_study_notes()
        return {"notes": [n.model_dump(mode="json") for n in notes]}

    @app.post("/api/notes")
    async def create_note(payload: NoteRequest) -> dict[str, Any]:
        text = payload.note.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Nota não pode estar vazia.")
        subject = payload.subject.strip() if payload.subject else None
        note = await engine.add_study_note(note=text, subject=subject)
        return {"ok": True, "note": note.model_dump(mode="json")}

    @app.post("/api/chat/text")
    async def chat_text(payload: ChatTextRequest) -> dict[str, Any]:
        text = payload.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Mensagem vazia.")
        _safe_stop_tts()
        msg = await coach.reply_text(text, speak=payload.speak)
        return {"reply": msg.model_dump(mode="json")}

    @app.post("/api/chat/audio")
    async def chat_audio(request: Request) -> dict[str, Any]:
        content_type = (request.headers.get("content-type") or "").lower()
        audio_bytes = b""
        mime_type = "audio/wav"
        speak = True

        if "multipart/form-data" in content_type:
            form = await request.form()
            upload = form.get("file")
            if upload is None or not hasattr(upload, "read"):
                raise HTTPException(
                    status_code=400,
                    detail="Arquivo de áudio ('file') obrigatório no multipart.",
                )
            audio_bytes = await upload.read()
            upload_mime = getattr(upload, "content_type", None)
            form_mime = form.get("mime_type")
            mime_type = str(form_mime or upload_mime or "audio/webm")
            speak_field = form.get("speak")
            if speak_field is not None:
                speak = str(speak_field).lower() not in {"false", "0", "no"}
        else:
            try:
                body = await request.json()
            except Exception as exc:
                raise HTTPException(status_code=400, detail="JSON de áudio inválido.") from exc
            if not isinstance(body, dict) or not body.get("audio_base64"):
                raise HTTPException(status_code=400, detail="Campo 'audio_base64' obrigatório.")
            raw_b64 = "".join(str(body["audio_base64"]).split())
            if raw_b64.startswith("data:") and "," in raw_b64:
                header, raw_b64 = raw_b64.split(",", 1)
                if ":" in header and ";" in header:
                    mime_type = header.split(":", 1)[1].split(";", 1)[0] or mime_type
            try:
                audio_bytes = base64.b64decode(raw_b64, validate=True)
            except Exception as exc:
                raise HTTPException(status_code=400, detail="Base64 de áudio inválido.") from exc
            if body.get("mime_type"):
                mime_type = str(body["mime_type"])
            if "speak" in body:
                speak = bool(body["speak"])

        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Áudio enviado está vazio.")

        mime_type = mime_type.split(";", 1)[0].strip().lower() or "audio/webm"
        _safe_stop_tts()
        transcript, msg = await coach.reply_audio(audio_bytes, mime_type=mime_type, speak=speak)
        return {"transcript": transcript, "reply": msg.model_dump(mode="json")}

    @app.post("/api/chat/ptt")
    async def chat_ptt(payload: PTTRequest | None = None) -> dict[str, Any]:
        speak = payload.speak if payload is not None else True
        if not mic.is_available():
            raise HTTPException(status_code=503, detail="Microfone nativo indisponível.")
        _safe_stop_tts()

        entered_conversation = False
        set_conv = getattr(engine, "set_conversation_mode", None)
        if callable(set_conv):
            try:
                cur_state = engine.get_status().session.current_state
                if cur_state not in (
                    FocusState.PAUSED,
                    FocusState.PRIVACY_MODE,
                    FocusState.BREAK_MODE,
                ):
                    res = set_conv(True)
                    if inspect.isawaitable(res):
                        await res
                    entered_conversation = True
            except Exception:
                logger.debug("Could not enter CONVERSATION_MODE during /api/chat/ptt.", exc_info=True)

        try:
            audio_bytes = await mic.record_until_silence(
                max_seconds=payload.max_seconds if payload else None,
                silence_timeout=payload.silence_timeout if payload else None,
            )
            if not audio_bytes:
                raise HTTPException(status_code=422, detail="Nenhuma fala detectada no microfone.")
            transcript, msg = await coach.reply_audio(audio_bytes, mime_type="audio/wav", speak=speak)
            return {
                "transcript": transcript,
                "reply": msg.model_dump(mode="json"),
                "audio_base64": msg.audio_base64,
            }
        finally:
            if entered_conversation and callable(set_conv):
                try:
                    res = set_conv(False)
                    if inspect.isawaitable(res):
                        await res
                except Exception:
                    logger.debug("Could not leave CONVERSATION_MODE during /api/chat/ptt.", exc_info=True)

    @app.post("/api/chat/ptt/stop")
    async def stop_ptt() -> dict[str, Any]:
        stop_fn = getattr(mic, "stop_recording", None)
        if callable(stop_fn):
            try:
                res = stop_fn()
                if inspect.isawaitable(res):
                    await res
            except Exception:
                logger.debug("Ignoring mic.stop_recording error.", exc_info=True)
        return {"ok": True}

    @app.get("/api/vision/preview.jpg")
    async def vision_preview(annotate: bool = True) -> Response:
        if engine.get_status().privacy_mode or vision_sensor is None:
            return Response(status_code=204, headers={"Cache-Control": "no-store"})
        try:
            data = vision_sensor.latest_frame_jpeg(
                max_width=640, quality=72, annotate=annotate
            )
        except Exception:
            logger.warning("Failed fetching webcam preview frame.", exc_info=True)
            data = None
        if not data:
            return Response(status_code=204, headers={"Cache-Control": "no-store"})
        return Response(
            content=data,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    @app.post("/api/vision/calibrate")
    async def calibrate_vision(payload: VisionCalibrateRequest | None = None) -> dict[str, Any]:
        if vision_sensor is None:
            raise HTTPException(status_code=503, detail="Sensor de webcam indisponível.")
        req = payload or VisionCalibrateRequest()
        try:
            res = await vision_sensor.calibrate_pose(
                mode=req.mode, duration_sec=req.duration_sec
            )
        except Exception as exc:
            raise HTTPException(
                status_code=503, detail=f"Falha ao calibrar webcam: {exc}"
            ) from exc
        return {
            "ok": True,
            "calibration": res,
            "status": _status_payload(engine),
        }

    @app.get("/api/audio/devices")
    async def get_audio_devices() -> dict[str, Any]:
        return {
            "output_devices": _safe_device_list(tts, "list_output_devices"),
            "input_devices": _safe_device_list(mic, "list_input_devices"),
            "output_device_name": config.voice.output_device_name,
            "input_device_name": config.voice.input_device_name,
        }

    @app.post("/api/audio/devices")
    async def set_audio_devices(payload: AudioDevicesRequest) -> dict[str, Any]:
        if "output_device_name" in payload.model_fields_set:
            out_name = (payload.output_device_name or "").strip() or None
            config.voice.output_device_name = out_name
            setter = getattr(tts, "set_output_device", None)
            if callable(setter):
                try:
                    res = setter(out_name)
                    if inspect.isawaitable(res):
                        await res
                except Exception:
                    logger.warning("Failed to set TTS output device.", exc_info=True)

        if "input_device_name" in payload.model_fields_set:
            in_name = (payload.input_device_name or "").strip() or None
            config.voice.input_device_name = in_name
            setter = getattr(mic, "set_input_device", None)
            if callable(setter):
                try:
                    res = setter(in_name)
                    if inspect.isawaitable(res):
                        await res
                except Exception:
                    logger.warning("Failed to set Mic input device.", exc_info=True)

        save_user_config(config)
        return {
            "ok": True,
            "output_device_name": config.voice.output_device_name,
            "input_device_name": config.voice.input_device_name,
            "status": _status_payload(engine),
        }

    @app.post("/api/audio/test")
    async def test_audio(payload: AudioTestRequest | None = None) -> dict[str, Any]:
        text = (
            payload.text.strip()
            if payload is not None and payload.text.strip()
            else AudioTestRequest().text
        )
        try:
            played = bool(await tts.speak(text))
        except Exception:
            logger.warning("Audio test speak failed.", exc_info=True)
            played = False
        return {"ok": played}

    @app.post("/api/ptt/trigger")
    async def trigger_ptt() -> dict[str, Any]:
        if on_ptt_trigger is None:
            raise HTTPException(status_code=501, detail="Trigger PTT externo não configurado.")
        await on_ptt_trigger()
        return {"ok": True}

    @app.post("/api/tts/stop")
    async def stop_tts() -> dict[str, Any]:
        try:
            tts.stop()
        except Exception:
            logger.debug("Ignoring TTS stop error.", exc_info=True)
        return {"ok": True}

    @app.post("/api/config/api-key")
    async def set_api_key(payload: ApiKeyRequest) -> dict[str, Any]:
        clean_key = payload.api_key.strip()
        if not clean_key:
            raise HTTPException(status_code=400, detail="Chave GEMINI_API_KEY inválida.")
        save_api_key_to_env(clean_key)
        judge.reconfigure_api_key(clean_key)
        coach.reconfigure_api_key(clean_key)
        config.gemini_api_key = clean_key
        return {
            "ok": True,
            "api_key_configured": judge.is_configured() and coach.is_configured(),
            "status": _status_payload(engine),
        }

    @app.get("/api/config/settings")
    async def get_settings() -> dict[str, Any]:
        return {"settings": _settings_payload(config)}

    @app.post("/api/config/settings")
    async def update_settings(payload: SettingsRequest | None = None) -> dict[str, Any]:
        if payload is not None:
            if payload.coach_persona is not None:
                config.coach_persona = payload.coach_persona
            if payload.max_daily_usd is not None:
                config.ai.max_daily_usd = payload.max_daily_usd
            if payload.max_monthly_usd is not None:
                config.ai.max_monthly_usd = payload.max_monthly_usd
            if payload.youtube_music_grace_sec is not None:
                config.behavior.youtube_music_grace_sec = payload.youtube_music_grace_sec
            if payload.speak_on_all_interventions is not None:
                config.behavior.speak_on_all_interventions = payload.speak_on_all_interventions
            if payload.phone_distraction_sec is not None:
                config.behavior.phone_distraction_sec = payload.phone_distraction_sec
            if payload.camera_not_seeing_user_sec is not None:
                config.behavior.camera_not_seeing_user_sec = payload.camera_not_seeing_user_sec
            if payload.passive_freeze_alert_sec is not None:
                config.behavior.passive_freeze_alert_sec = payload.passive_freeze_alert_sec
            if payload.capture_mode is not None:
                config.ai.capture_mode = payload.capture_mode

            budget = getattr(engine, "budget", None)
            if budget is not None and hasattr(budget, "update_limits"):
                budget.update_limits(
                    max_daily_usd=payload.max_daily_usd,
                    max_monthly_usd=payload.max_monthly_usd,
                )

        save_user_config(config)
        return {
            "ok": True,
            "settings": _settings_payload(config),
            "status": _status_payload(engine),
        }

    @app.post("/api/test-nudge")
    async def test_nudge(payload: TestNudgeRequest | None = None) -> dict[str, Any]:
        level_val = payload.level if payload is not None else 2
        level = InterventionLevel(level_val)
        status = engine.get_status()

        message = await coach.generate_nudge(
            level=level,
            reason="Teste manual de intervenção",
            os_context=status.latest_os,
            vision_signal=status.latest_vision,
            study_goal=status.session.study_goal,
            current_subject=status.session.current_subject,
        )
        title_map = {
            InterventionLevel.SOFT: "FocusGuard — Lembrete de Foco",
            InterventionLevel.VOICE: "FocusGuard — Puxão de Orelha",
            InterventionLevel.FIRM_OVERLAY: "FocusGuard — VOLTE PRO FOCO!",
        }
        cmd = InterventionCommand(
            level=level,
            title=title_map.get(level, "FocusGuard — Puxão de Orelha"),
            message=message,
            speak_aloud=(level >= InterventionLevel.VOICE)
            or bool(config.behavior.speak_on_all_interventions),
            reason="Teste manual de intervenção",
        )
        await event_bus.publish(cmd)
        if dispatcher is not None:
            await dispatcher.dispatch(cmd)
        return {"ok": True, "command": cmd.model_dump(mode="json")}

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await ws_manager.connect(websocket)
        try:
            while True:
                msg = await websocket.receive_text()
                if msg == "ping":
                    await websocket.send_json({"type": "pong", "data": {}})
        except WebSocketDisconnect:
            ws_manager.disconnect(websocket)
        except Exception:
            ws_manager.disconnect(websocket)

    return app
