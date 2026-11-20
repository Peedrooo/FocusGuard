"""Firm Coach Agent with multimodal voice/text chat, screen inspection, and FocusEngine tools."""

from __future__ import annotations

import base64
import logging
import re
from collections import deque
from typing import Any

from google.genai import types

from focus_guard.ai.gemini_client import (
    BudgetExceededError,
    GeminiClient,
    extract_response_text,
    parse_json_object,
)
from focus_guard.core.config import AppConfig
from focus_guard.core.event_bus import EventBus
from focus_guard.core.events import (
    ChatMessage,
    CoachPersona,
    InterventionLevel,
    OSContextEvent,
    ScreenSnapshot,
    VisionSignalEvent,
)
from focus_guard.core.ports import FocusEnginePort, StoragePort, TTSPort

logger = logging.getLogger(__name__)

_MAX_HISTORY_MESSAGES = 100
_MAX_PROMPT_HISTORY = 10
_MAX_TOOL_ROUNDS = 4

_SCREEN_INTENT_MARKERS = (
    "minha tela",
    "na tela",
    "olha a tela",
    "olhe a tela",
    "veja a tela",
    "lê a tela",
    "le a tela",
    "essa questão",
    "essa questao",
    "nessa questão",
    "nessa questao",
    "explica essa",
    "explica isso",
    "o que está na tela",
    "o que esta na tela",
    "o que tá na tela",
    "o que ta na tela",
)

_TOOL_DECLARATIONS = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="get_session_summary",
            description=(
                "Retorna o resumo atual da sessão de estudo (estado, meta, matéria, "
                "tempo focado, distrações, janela ativa, liberações e orçamento)."
            ),
            parameters=types.Schema(type=types.Type.OBJECT, properties={}),
        ),
        types.FunctionDeclaration(
            name="inspect_current_screen",
            description=(
                "Captura a tela atual do usuário para ler e explicar uma questão, "
                "PDF, código ou anotação visível neste momento."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "question": types.Schema(
                        type=types.Type.STRING,
                        description="Pergunta ou objetivo do usuário ao analisar a tela.",
                    ),
                },
                required=["question"],
            ),
        ),
        types.FunctionDeclaration(
            name="grant_temporary_allowance",
            description=(
                "Concede liberação temporária para um site ou aplicativo específico "
                "(ex: youtube, spotify, whatsapp) por um período limitado em minutos."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "pattern": types.Schema(
                        type=types.Type.STRING,
                        description="Padrão de app/site/título a liberar (ex: 'youtube', 'spotify').",
                    ),
                    "minutes": types.Schema(
                        type=types.Type.INTEGER,
                        description="Duração da liberação em minutos (1 a 120).",
                    ),
                    "reason": types.Schema(
                        type=types.Type.STRING,
                        description="Motivo objetivo da liberação temporária.",
                    ),
                },
                required=["pattern", "minutes", "reason"],
            ),
        ),
        types.FunctionDeclaration(
            name="start_break",
            description="Inicia uma pausa cronometrada na sessão de estudo.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "minutes": types.Schema(
                        type=types.Type.INTEGER,
                        description="Duração da pausa em minutos (padrão 5).",
                    ),
                },
                required=["minutes"],
            ),
        ),
        types.FunctionDeclaration(
            name="end_break",
            description="Encerra a pausa atual e retorna imediatamente ao modo focado.",
            parameters=types.Schema(type=types.Type.OBJECT, properties={}),
        ),
        types.FunctionDeclaration(
            name="set_study_goal",
            description="Atualiza o objetivo principal de estudo e a matéria atual.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "goal": types.Schema(
                        type=types.Type.STRING,
                        description="Objetivo da sessão (ex: 'Simulado TRT').",
                    ),
                    "subject": types.Schema(
                        type=types.Type.STRING,
                        description="Matéria atual (ex: 'Direito Constitucional').",
                    ),
                },
                required=["goal", "subject"],
            ),
        ),
        types.FunctionDeclaration(
            name="record_study_note",
            description="Salva uma anotação rápida de estudo vinculada à matéria.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "note": types.Schema(
                        type=types.Type.STRING,
                        description="Conteúdo da anotação de estudo.",
                    ),
                    "subject": types.Schema(
                        type=types.Type.STRING,
                        description="Matéria opcional da anotação.",
                    ),
                },
                required=["note"],
            ),
        ),
    ]
)


def _should_auto_capture_screen(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _SCREEN_INTENT_MARKERS)


class FirmCoachAgent:
    """CoachAgentPort implementation with 'Coach Firme e Direto' persona and FocusEngine tools."""

    def __init__(
        self,
        gemini_client: GeminiClient,
        engine: FocusEnginePort,
        tts: TTSPort,
        config: AppConfig,
        event_bus: EventBus | None = None,
        storage: StoragePort | None = None,
    ) -> None:
        self._gemini = gemini_client
        self._engine = engine
        self._tts = tts
        self._config = config
        self._event_bus = event_bus
        self._storage = storage
        self._history: list[ChatMessage] = []
        self._recent_nudges: deque[str] = deque(maxlen=6)
        self._nudge_variant_idx: int = 0

    def is_configured(self) -> bool:
        return self._gemini.is_configured()

    def reconfigure_api_key(self, api_key: str) -> None:
        self._gemini.reconfigure_api_key(api_key)

    def get_history(self) -> list[ChatMessage]:
        return list(self._history)

    def clear_history(self) -> None:
        self._history.clear()

    # ------------------------------------------------------------------
    # Direct FocusEngine Tools
    # ------------------------------------------------------------------

    async def get_session_summary(self) -> dict[str, Any]:
        status = self._engine.get_status()
        session = status.session
        latest_os = status.latest_os
        is_sensitive = bool(latest_os and latest_os.is_sensitive_window)

        visible_windows: list[dict[str, Any]] = []
        usage_pattern: dict[str, Any] = {
            "interaction_mode": "idle",
            "window_switches_60s": 0,
            "active_input_ratio_60s": 0.0,
        }
        if latest_os is not None:
            if not is_sensitive:
                for win in latest_os.visible_windows:
                    if win.is_sensitive:
                        continue
                    visible_windows.append(
                        {
                            "monitor_label": win.monitor_label,
                            "process_name": win.process_name,
                            "window_title": win.window_title[:80],
                            "screen_area_pct": round(win.screen_area_pct, 1),
                            "is_foreground": win.is_foreground,
                        }
                    )
                    if len(visible_windows) >= 5:
                        break
            up = latest_os.usage_pattern
            usage_pattern = {
                "interaction_mode": up.interaction_mode,
                "window_switches_60s": up.window_switches_60s,
                "active_input_ratio_60s": round(up.active_input_ratio_60s, 2),
            }

        return {
            "running": status.running,
            "privacy_mode": status.privacy_mode,
            "current_state": str(session.current_state),
            "study_goal": session.study_goal,
            "current_subject": session.current_subject,
            "focus_minutes": round(session.focus_seconds / 60.0, 1),
            "distraction_minutes": round(session.distraction_seconds / 60.0, 1),
            "break_minutes": round(session.break_seconds / 60.0, 1),
            "focus_ratio_pct": round(session.focus_ratio_pct, 1),
            "interventions_count": session.interventions_count,
            "phone_alerts_count": session.phone_alerts_count,
            "active_break_remaining_sec": round(session.active_break_remaining_sec, 1),
            "active_monitor_label": latest_os.active_monitor_label if latest_os else "M1",
            "latest_process": latest_os.process_name if latest_os else "",
            "latest_window": (
                "[janela protegida]" if is_sensitive else (latest_os.window_title if latest_os else "")
            ),
            "latest_url": "" if is_sensitive else (latest_os.browser_url or "" if latest_os else ""),
            "visible_windows": visible_windows,
            "usage_pattern": usage_pattern,
            "active_allowances": [
                {"pattern": a.pattern, "reason": a.reason} for a in status.active_allowances
            ],
            "budget_daily_usd": status.budget.daily_usd,
            "budget_daily_brl": status.budget.daily_brl,
        }

    def _capture_screen_context(self) -> tuple[ScreenSnapshot | None, dict[str, Any]]:
        status = self._engine.get_status()
        latest_os = status.latest_os
        if status.privacy_mode or (latest_os is not None and latest_os.is_sensitive_window):
            return None, {
                "captured": False,
                "privacy_protected": True,
                "message": "Tela protegida pelo Modo Privacidade ou janela sensível.",
            }

        snapshot = self._engine.capture_current_screen()
        if snapshot is None or not snapshot.image_bytes_jpeg:
            return None, {
                "captured": False,
                "privacy_protected": False,
                "message": "Não foi possível capturar a tela neste momento.",
            }

        return snapshot, {
            "captured": True,
            "privacy_protected": False,
            "width": snapshot.width,
            "height": snapshot.height,
            "window_title": latest_os.window_title if latest_os else "",
            "process_name": latest_os.process_name if latest_os else "",
        }

    async def inspect_current_screen(self, question: str = "") -> dict[str, Any]:
        snapshot, info = self._capture_screen_context()
        if snapshot is None:
            return info

        prompt = (
            f"O usuário {self._config.user_name or 'Pedro'} pediu ajuda sobre o que está na tela agora.\n"
            f"Pergunta: {question or 'Explique o conteúdo visível na tela.'}\n"
            "Responda de forma direta, técnica e objetiva no estilo Coach Firme e Direto."
        )
        if self._gemini.is_configured() and self._gemini.budget.can_spend():
            try:
                contents = [
                    prompt,
                    types.Part.from_bytes(
                        data=snapshot.image_bytes_jpeg,
                        mime_type="image/jpeg",
                    ),
                ]
                response, _, _, _, cost_usd = await self._gemini.generate_content(
                    contents=contents,
                    system_instruction=self._build_system_instruction(),
                    temperature=0.3,
                )
                info["analysis"] = extract_response_text(response).strip()
                info["cost_usd"] = cost_usd
            except Exception as exc:
                logger.warning("Multimodal screen inspection failed: %s", exc)
                info["analysis"] = "Tela capturada, mas a análise remota falhou no momento."
        else:
            info["analysis"] = "Tela capturada, mas o Gemini está indisponível ou com orçamento esgotado."
        return info

    async def grant_temporary_allowance(
        self,
        pattern: str,
        minutes: int,
        reason: str,
    ) -> dict[str, Any]:
        clean_pattern = pattern.strip()
        if not clean_pattern:
            return {"granted": False, "error": "Padrão de liberação vazio."}
        clamped_minutes = max(1, min(int(minutes), 120))
        allowance = await self._engine.grant_allowance(
            pattern=clean_pattern,
            minutes=clamped_minutes,
            reason=reason.strip() or "Solicitado ao coach",
        )
        return {
            "granted": True,
            "id": allowance.id,
            "pattern": allowance.pattern,
            "minutes": clamped_minutes,
            "reason": allowance.reason,
            "expires_at": allowance.expires_at.isoformat(),
        }

    async def start_break(self, minutes: int = 5) -> dict[str, Any]:
        clamped = max(1, min(int(minutes), 60))
        stats = await self._engine.start_break(minutes=clamped)
        return {
            "started": True,
            "minutes": clamped,
            "current_state": str(stats.current_state),
            "active_break_remaining_sec": stats.active_break_remaining_sec,
        }

    async def end_break(self) -> dict[str, Any]:
        stats = await self._engine.end_break()
        return {
            "ended": True,
            "current_state": str(stats.current_state),
        }

    async def set_study_goal(self, goal: str, subject: str) -> dict[str, Any]:
        clean_goal = goal.strip() or self._config.study_goal
        clean_subject = subject.strip() or self._config.current_subject
        stats = await self._engine.set_study_goal(goal=clean_goal, subject=clean_subject)
        return {
            "updated": True,
            "study_goal": stats.study_goal,
            "current_subject": stats.current_subject,
        }

    async def record_study_note(self, note: str, subject: str = "") -> dict[str, Any]:
        clean_note = note.strip()
        if not clean_note:
            return {"saved": False, "error": "Nota vazia."}
        saved = await self._engine.add_study_note(
            note=clean_note,
            subject=subject.strip() or None,
        )
        return {
            "saved": True,
            "id": saved.id,
            "subject": saved.subject,
            "note": saved.note,
        }

    # ------------------------------------------------------------------
    # Chat (Text & Multimodal Audio)
    # ------------------------------------------------------------------

    async def reply_text(
        self,
        user_text: str,
        speak: bool = True,
        local_playback: bool = False,
    ) -> ChatMessage:
        clean_text = (user_text or "").strip()
        user_msg = ChatMessage(role="user", text=clean_text)
        await self._record_message(user_msg)

        executed_tools: list[str] = []
        total_cost = 0.0

        if not self._gemini.is_configured() or not self._gemini.budget.can_spend():
            reply_body = await self._offline_or_throttled_reply(clean_text, executed_tools)
        else:
            try:
                reply_body, total_cost = await self._run_text_turn(
                    user_text=clean_text,
                    executed_tools=executed_tools,
                )
            except BudgetExceededError:
                reply_body = await self._offline_or_throttled_reply(clean_text, executed_tools)
            except Exception:
                logger.exception("Coach text reply failed; falling back to local response.")
                reply_body = await self._offline_or_throttled_reply(
                    clean_text,
                    executed_tools,
                    api_error=True,
                )

        audio_b64, audio_mime = await self._maybe_synthesize_and_play(
            reply_body,
            speak=speak,
            local_playback=local_playback,
        )

        coach_msg = ChatMessage(
            role="coach",
            text=reply_body,
            audio_base64=audio_b64,
            audio_mime=audio_mime,
            tool_calls=executed_tools,
            cost_usd=round(total_cost, 8),
        )
        await self._record_message(coach_msg)
        return coach_msg

    async def reply_audio(
        self,
        audio_bytes: bytes,
        mime_type: str = "audio/wav",
        speak: bool = True,
        local_playback: bool = False,
    ) -> tuple[str, ChatMessage]:
        if not audio_bytes:
            empty_reply = "Não recebi áudio nenhum. Fala de novo ou digita direto aqui."
            user_msg = ChatMessage(role="user", text="(áudio vazio)")
            await self._record_message(user_msg)
            audio_b64, audio_mime = await self._maybe_synthesize_and_play(
                empty_reply,
                speak=speak,
                local_playback=local_playback,
            )
            coach_msg = ChatMessage(
                role="coach",
                text=empty_reply,
                audio_base64=audio_b64,
                audio_mime=audio_mime,
            )
            await self._record_message(coach_msg)
            return "(áudio vazio)", coach_msg

        executed_tools: list[str] = []
        total_cost = 0.0
        transcript = "(áudio enviado)"
        reply_body = ""

        if not self._gemini.is_configured() or not self._gemini.budget.can_spend():
            reason = (
                "Chave da API Gemini não configurada."
                if not self._gemini.is_configured()
                else "Limite de orçamento diário/mensal do Gemini atingido."
            )
            transcript = "(áudio não transcrito offline)"
            summary = await self.get_session_summary()
            reply_body = (
                f"{reason} Não consegui processar o áudio na nuvem agora, mas sua sessão "
                f"continua em {summary['current_subject']} (foco: {summary['focus_ratio_pct']}%). "
                "Se quiser ação imediata, manda em texto curto (ex: 'pausa 5', 'resumo')."
            )
        else:
            try:
                transcript, reply_body, total_cost = await self._run_audio_turn(
                    audio_bytes=audio_bytes,
                    mime_type=mime_type,
                    executed_tools=executed_tools,
                )
            except BudgetExceededError:
                transcript = "(áudio bloqueado por limite de orçamento)"
                reply_body = "Orçamento da API atingido por hoje. Usa comandos curtos em texto."
            except Exception:
                logger.exception("Coach audio reply failed.")
                transcript = "(falha ao processar áudio)"
                reply_body = "Deu erro ao processar seu áudio na API. Repete em texto que eu resolvo."

        user_msg = ChatMessage(role="user", text=transcript)
        await self._record_message(user_msg)

        audio_b64, audio_mime = await self._maybe_synthesize_and_play(
            reply_body,
            speak=speak,
            local_playback=local_playback,
        )
        coach_msg = ChatMessage(
            role="coach",
            text=reply_body,
            audio_base64=audio_b64,
            audio_mime=audio_mime,
            tool_calls=executed_tools,
            cost_usd=round(total_cost, 8),
        )
        await self._record_message(coach_msg)
        return transcript, coach_msg

    async def generate_nudge(
        self,
        level: InterventionLevel,
        reason: str,
        os_context: OSContextEvent | None,
        vision_signal: VisionSignalEvent | None,
        study_goal: str,
        current_subject: str,
    ) -> str:
        user_name = self._config.user_name or "Pedro"
        subject = current_subject or self._config.current_subject or "seus estudos"
        persona = CoachPersona(self._config.coach_persona)
        variant_idx = self._nudge_variant_idx
        self._nudge_variant_idx += 1

        if not self._gemini.is_configured() or not self._gemini.budget.can_spend():
            nudge = self._deterministic_nudge(
                user_name=user_name,
                persona=persona,
                level=level,
                reason=reason,
                os_context=os_context,
                vision_signal=vision_signal,
                subject=subject,
                variant_idx=variant_idx,
            )
            self._recent_nudges.append(nudge)
            return nudge

        is_sensitive = bool(os_context and os_context.is_sensitive_window)
        os_desc = "desconhecida"
        if os_context is not None:
            if is_sensitive:
                os_desc = f"{os_context.process_name} (tela protegida)"
            else:
                os_desc = f"{os_context.process_name} | {os_context.window_title}"
                if os_context.browser_url:
                    os_desc += f" | {os_context.browser_url[:100]}"

        vision_desc = "sem dados de câmera"
        if vision_signal is not None and vision_signal.camera_available:
            flags: list[str] = []
            if vision_signal.camera_obstructed:
                flags.append("câmera tampada ou escura demais")
            if vision_signal.phone_detected:
                flags.append("celular detectado")
            if not vision_signal.face_present:
                flags.append("usuário invisível na câmera")
            if vision_signal.drowsiness_perclos >= self._config.behavior.drowsiness_perclos_threshold:
                flags.append(f"sonolência PERCLOS={vision_signal.drowsiness_perclos:.2f}")
            vision_desc = ", ".join(flags) if flags else "usuário enquadrado em frente à tela"

        stats_line = ""
        extra_screens_line = ""
        usage_line = ""
        try:
            status = self._engine.get_status()
            sess = status.session
            stats_line = (
                f"Placar da sessão: intervenções={sess.interventions_count}, "
                f"foco={sess.focus_ratio_pct:.0f}%, distração={round(sess.distraction_seconds / 60.0, 1)}min, "
                f"alertas_celular={sess.phone_alerts_count}.\n"
            )
            if not status.privacy_mode and not is_sensitive and status.latest_os is not None:
                up = status.latest_os.usage_pattern
                usage_line = (
                    f"Ritmo 60s: modo={up.interaction_mode}, trocas_janela={up.window_switches_60s}, "
                    f"ocioso={status.latest_os.idle_seconds:.0f}s.\n"
                )
                extra_wins = [
                    f"[{w.monitor_label}] {w.process_name}: '{w.window_title[:45]}'"
                    for w in status.latest_os.visible_windows[:4]
                    if not w.is_sensitive
                ]
                if extra_wins:
                    extra_screens_line = f"Telas extras visíveis: {'; '.join(extra_wins)}\n"
        except Exception:
            pass

        if level >= InterventionLevel.FIRM_OVERLAY:
            strictness_rule = (
                "Escalonamento NÍVEL 3 (ULTIMATO): choque direto e intolerância máxima com o tempo perdido; "
                "cobre fechamento imediato ou pausa formal."
            )
        elif level >= InterventionLevel.VOICE:
            strictness_rule = (
                "Escalonamento NÍVEL 2 (COBRANÇA PESADA): cobre a reincidência usando o placar de foco e "
                "mostre o custo real dessa distração na prova."
            )
        else:
            strictness_rule = (
                "Escalonamento NÍVEL 1 (CIRÚRGICO E ÁCIDO): use ironia fina conectando o que está na tela "
                f"ou câmera com um conceito real de {subject} para desmontar a desculpa na hora."
            )

        recent_block = ""
        if self._recent_nudges:
            recent_lines = "\n".join(f"- {item}" for item in self._recent_nudges)
            recent_block = (
                "\nÚltimas cobranças já faladas (PROIBIDO repetir aberturas, piadas ou estruturas parecidas):\n"
                f"{recent_lines}\n"
            )

        prompt = (
            f"Gere UMA cobrança de áudio afiada, rígida e criativa (máximo 2 frases curtas, diretas e impactantes) para {user_name}.\n"
            f"Nível: {level.name} ({int(level)}) | {strictness_rule}\n"
            f"Motivo factual: {reason}\n"
            f"Meta: {study_goal} | Matéria atual: {subject}\n"
            f"{stats_line}"
            f"{usage_line}"
            f"Tela atual: {os_desc}\n"
            f"{extra_screens_line}"
            f"Câmera: {vision_desc}\n"
            f"{recent_block}"
            "Regras: zero elogio ou motivacional genérico; cite o fato exato da tela/câmera e mande agir agora."
        )
        try:
            response, _, _, _, _ = await self._gemini.generate_content(
                contents=prompt,
                system_instruction=self._build_system_instruction(),
                temperature=0.68,
            )
            raw_text = extract_response_text(response).strip()
            parsed_nudge = parse_json_object(raw_text) if raw_text.startswith("{") else None
            if parsed_nudge:
                extracted = str(
                    parsed_nudge.get("nudge") or parsed_nudge.get("reply") or ""
                ).strip()
                if extracted:
                    raw_text = extracted
            if raw_text:
                self._recent_nudges.append(raw_text)
                return raw_text
        except Exception:
            logger.warning("Gemini nudge generation failed; falling back to deterministic template.")

        nudge = self._deterministic_nudge(
            user_name=user_name,
            persona=persona,
            level=level,
            reason=reason,
            os_context=os_context,
            vision_signal=vision_signal,
            subject=subject,
            variant_idx=variant_idx,
        )
        self._recent_nudges.append(nudge)
        return nudge

    # ------------------------------------------------------------------
    # Internal Turn Runners & Tool Dispatch
    # ------------------------------------------------------------------

    def _build_system_instruction(self) -> str:
        user_name = self._config.user_name or "Pedro"
        persona = CoachPersona(self._config.coach_persona)
        if persona == CoachPersona.SARGENTO:
            tone = (
                "Personalidade 'Sargento': linha-dura implacável, frases secas de impacto, "
                "zero desculpas e cobrança cortante sobre fatos da tela/câmera sem grosseria gratuita."
            )
        elif persona == CoachPersona.EQUILIBRADO:
            tone = (
                "Personalidade 'Equilibrado': direto, racional e firme nos limites, "
                "didático quando o assunto for conteúdo do concurso."
            )
        else:
            tone = (
                "Personalidade 'Coach Firme e Direto': ácido, criativo e cirúrgico — desmonta desculpas "
                "com ironia inteligente e fatos reais da tela/câmera, sem blá-blá-blá motivacional genérico. "
                "Se o usuário pedir ajuda na matéria, responda com precisão técnica e objetividade."
            )

        return (
            f"Você é o Coach de Foco do FocusGuard falando com {user_name}, estudante de Concurso Público.\n"
            f"{tone}\n"
            "Use as ferramentas disponíveis quando o usuário pedir para:\n"
            "- ver status/resumo da sessão (`get_session_summary`);\n"
            "- olhar/ler/explicar algo na tela atual (`inspect_current_screen`);\n"
            "- liberar temporariamente um site/app (`grant_temporary_allowance`);\n"
            "- iniciar ou encerrar pausa (`start_break`, `end_break`);\n"
            "- mudar objetivo/matéria (`set_study_goal`);\n"
            "- salvar anotação (`record_study_note`).\n"
            "Nunca invente métricas ou conteúdos de tela que você não consultou."
        )

    async def _build_context_header(self) -> str:
        summary = await self.get_session_summary()
        vis_wins = summary.get("visible_windows") or []
        wins_desc = (
            "; ".join(
                f"[{w['monitor_label']}] {w['process_name']}: '{w['window_title'][:45]}'"
                for w in vis_wins
            )
            or "nenhuma extra"
        )
        up = summary.get("usage_pattern") or {}
        return (
            f"[Contexto atual: estado={summary['current_state']}, meta='{summary['study_goal']}', "
            f"matéria='{summary['current_subject']}', foco={summary['focus_minutes']}min "
            f"({summary['focus_ratio_pct']}%), monitor={summary['active_monitor_label']}, "
            f"janela='{summary['latest_window']}', "
            f"visible_windows=[{wins_desc}], "
            f"usage_pattern=(interaction_mode={up.get('interaction_mode', 'idle')}, "
            f"window_switches_60s={up.get('window_switches_60s', 0)}, "
            f"active_input_ratio_60s={up.get('active_input_ratio_60s', 0.0):.2f})]"
        )

    def _recent_history_contents(self) -> list[types.Content]:
        # Exclude the just-appended final user message (added as the active turn)
        prior = self._history[:-1][-_MAX_PROMPT_HISTORY:]
        contents: list[types.Content] = []
        for msg in prior:
            if not msg.text:
                continue
            role = "model" if msg.role == "coach" else "user"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=msg.text)],
                )
            )
        return contents

    async def _run_text_turn(
        self,
        user_text: str,
        executed_tools: list[str],
    ) -> tuple[str, float]:
        header = await self._build_context_header()
        user_parts: list[types.Part] = [
            types.Part.from_text(text=f"{header}\nMensagem do usuário: {user_text}")
        ]

        if _should_auto_capture_screen(user_text):
            snapshot, cap_info = self._capture_screen_context()
            executed_tools.append("inspect_current_screen")
            if snapshot is not None:
                user_parts.append(
                    types.Part.from_bytes(
                        data=snapshot.image_bytes_jpeg,
                        mime_type="image/jpeg",
                    )
                )
            else:
                user_parts.append(
                    types.Part.from_text(
                        text=f"[Tentativa de captura da tela: {cap_info.get('message', 'indisponível')}]"
                    )
                )

        contents: list[Any] = [
            *self._recent_history_contents(),
            types.Content(role="user", parts=user_parts),
        ]

        response, total_cost = await self._run_tool_loop(
            contents=contents,
            executed_tools=executed_tools,
        )
        raw_text = extract_response_text(response).strip()
        parsed = parse_json_object(raw_text)
        if parsed:
            await self._execute_json_tool_calls(parsed, executed_tools)
            if isinstance(parsed.get("reply"), str) and parsed["reply"].strip():
                return parsed["reply"].strip(), total_cost

        return raw_text or "Certo. Volta para o foco agora.", total_cost

    async def _run_audio_turn(
        self,
        audio_bytes: bytes,
        mime_type: str,
        executed_tools: list[str],
    ) -> tuple[str, str, float]:
        header = await self._build_context_header()
        audio_prompt = (
            f"{header}\n"
            "O usuário enviou o áudio anexo em português do Brasil.\n"
            "1) Transcreva fielmente o que foi falado.\n"
            "2) Se o usuário pedir alguma ação (pausa, liberação temporária, meta, anotação, "
            "resumo ou olhar/explicar a tela), execute a ferramenta correspondente primeiro.\n"
            "3) Na resposta final (após eventuais chamadas de ferramenta), responda EXCLUSIVAMENTE "
            'em JSON válido: {"transcript": "transcrição fiel", "reply": "resposta no tom Coach Firme e Direto"}.'
        )
        clean_mime = (mime_type or "audio/wav").split(";", 1)[0].strip().lower() or "audio/webm"
        user_parts: list[types.Part] = [
            types.Part.from_text(text=audio_prompt),
            types.Part.from_bytes(data=audio_bytes, mime_type=clean_mime),
        ]
        contents: list[Any] = [
            *self._recent_history_contents(),
            types.Content(role="user", parts=user_parts),
        ]

        response, total_cost = await self._run_tool_loop(
            contents=contents,
            executed_tools=executed_tools,
        )
        raw_text = extract_response_text(response).strip()
        parsed = parse_json_object(raw_text)
        if parsed:
            extra_snapshot = await self._execute_json_tool_calls(parsed, executed_tools)
            transcript = str(parsed.get("transcript") or "(áudio enviado)").strip()
            reply = str(parsed.get("reply") or "").strip()

            # If a JSON-only tool call asked to inspect_current_screen and returned a fresh screenshot,
            # perform one follow-up multimodal pass so Gemini actually answers based on the screen!
            if extra_snapshot is not None:
                followup_parts = [
                    types.Part.from_text(
                        text=(
                            f"Transcrição do pedido do usuário: {transcript}\n"
                            "Aqui está o screenshot atual da tela. Responda diretamente à dúvida do usuário."
                        )
                    ),
                    types.Part.from_bytes(
                        data=extra_snapshot.image_bytes_jpeg,
                        mime_type="image/jpeg",
                    ),
                ]
                follow_resp, _, _, _, follow_cost = await self._gemini.generate_content(
                    contents=followup_parts,
                    system_instruction=self._build_system_instruction(),
                    temperature=0.3,
                )
                total_cost += follow_cost
                follow_text = extract_response_text(follow_resp).strip()
                if follow_text:
                    reply = follow_text

            if not reply:
                reply = raw_text or "Entendido. De volta aos estudos."
            return transcript, reply, total_cost

        return "(áudio enviado)", (raw_text or "Entendido. De volta aos estudos."), total_cost

    async def _run_tool_loop(
        self,
        contents: list[Any],
        executed_tools: list[str],
    ) -> tuple[Any, float]:
        total_cost = 0.0
        system_instruction = self._build_system_instruction()

        response: Any = None
        for _ in range(_MAX_TOOL_ROUNDS):
            response, _, _, _, cost_usd = await self._gemini.generate_content(
                contents=contents,
                system_instruction=system_instruction,
                tools=[_TOOL_DECLARATIONS],
                temperature=0.35,
            )
            total_cost += cost_usd

            function_calls = getattr(response, "function_calls", None) or []
            if not function_calls:
                return response, total_cost

            model_content = self._extract_model_content(response, function_calls)
            contents.append(model_content)

            tool_parts: list[types.Part] = []
            attached_snapshots: list[ScreenSnapshot] = []
            for fc in function_calls:
                name = str(getattr(fc, "name", "") or "").strip()
                raw_args = getattr(fc, "args", None)
                args = dict(raw_args) if isinstance(raw_args, dict) else {}
                result, snapshot = await self._dispatch_tool(
                    name=name,
                    args=args,
                    executed_tools=executed_tools,
                    allow_nested_gemini=False,
                )
                tool_parts.append(
                    types.Part.from_function_response(name=name, response=result)
                )
                if snapshot is not None:
                    attached_snapshots.append(snapshot)

            for snap in attached_snapshots:
                tool_parts.append(
                    types.Part.from_bytes(
                        data=snap.image_bytes_jpeg,
                        mime_type="image/jpeg",
                    )
                )

            contents.append(types.Content(role="user", parts=tool_parts))

        return response, total_cost

    @staticmethod
    def _extract_model_content(response: Any, function_calls: list[Any]) -> types.Content:
        candidates = getattr(response, "candidates", None) or []
        if candidates and getattr(candidates[0], "content", None) is not None:
            return candidates[0].content
        parts = [
            types.Part.from_function_call(
                name=str(getattr(fc, "name", "")),
                args=dict(getattr(fc, "args", None) or {}),
            )
            for fc in function_calls
        ]
        return types.Content(role="model", parts=parts)

    async def _execute_json_tool_calls(
        self,
        parsed: dict[str, Any],
        executed_tools: list[str],
    ) -> ScreenSnapshot | None:
        raw_calls = parsed.get("tool_calls")
        if not isinstance(raw_calls, list):
            return None

        captured_snapshot: ScreenSnapshot | None = None
        for entry in raw_calls:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name") or entry.get("tool") or "").strip()
            if not name:
                continue
            raw_args = entry.get("args") or entry.get("arguments") or entry.get("params") or {}
            args = dict(raw_args) if isinstance(raw_args, dict) else {}
            _, snap = await self._dispatch_tool(
                name=name,
                args=args,
                executed_tools=executed_tools,
                allow_nested_gemini=False,
            )
            if snap is not None:
                captured_snapshot = snap
        return captured_snapshot

    async def _dispatch_tool(
        self,
        name: str,
        args: dict[str, Any],
        executed_tools: list[str],
        allow_nested_gemini: bool = True,
    ) -> tuple[dict[str, Any], ScreenSnapshot | None]:
        executed_tools.append(name)
        try:
            if name == "get_session_summary":
                return await self.get_session_summary(), None
            if name == "inspect_current_screen":
                question = str(args.get("question", "") or "")
                if allow_nested_gemini:
                    return await self.inspect_current_screen(question=question), None
                snapshot, info = self._capture_screen_context()
                if snapshot is not None:
                    info["question"] = question
                    info["instruction"] = "Screenshot anexado logo abaixo para leitura direta."
                return info, snapshot
            if name == "grant_temporary_allowance":
                return await self.grant_temporary_allowance(
                    pattern=str(args.get("pattern", "")),
                    minutes=int(args.get("minutes", 10)),
                    reason=str(args.get("reason", "Solicitado pelo usuário")),
                ), None
            if name == "start_break":
                return await self.start_break(minutes=int(args.get("minutes", 5))), None
            if name == "end_break":
                return await self.end_break(), None
            if name == "set_study_goal":
                return await self.set_study_goal(
                    goal=str(args.get("goal", "")),
                    subject=str(args.get("subject", "")),
                ), None
            if name == "record_study_note":
                return await self.record_study_note(
                    note=str(args.get("note", "")),
                    subject=str(args.get("subject", "")),
                ), None
            return {"error": f"Ferramenta desconhecida: {name}"}, None
        except Exception as exc:
            logger.exception("Coach tool %s failed.", name)
            return {"error": str(exc)}, None

    # ------------------------------------------------------------------
    # Offline / Budget-Throttled Local Fallback
    # ------------------------------------------------------------------

    async def _offline_or_throttled_reply(
        self,
        user_text: str,
        executed_tools: list[str],
        api_error: bool = False,
    ) -> str:
        lowered = user_text.lower().strip()
        user_name = self._config.user_name or "Pedro"

        if match := re.search(r"\b(?:pausa|break)\s*(?:de\s*)?(\d{1,2})\b", lowered):
            mins = int(match.group(1))
            await self._dispatch_tool("start_break", {"minutes": mins}, executed_tools)
            return f"Pausa de {mins} minuto(s) iniciada, {user_name}. Aproveita para descansar a vista e volta no horário."

        if any(phrase in lowered for phrase in ("encerrar pausa", "fim da pausa", "voltar da pausa")):
            await self._dispatch_tool("end_break", {}, executed_tools)
            return f"Pausa encerrada, {user_name}. De volta ao foco agora."

        if match := re.search(
            r"\blibera(?:r)?\s+([a-z0-9_.-]+)(?:\s+por)?\s+(\d{1,3})\b",
            lowered,
        ):
            pattern, mins_str = match.group(1), match.group(2)
            await self._dispatch_tool(
                "grant_temporary_allowance",
                {"pattern": pattern, "minutes": int(mins_str), "reason": "Comando local"},
                executed_tools,
            )
            return f"Liberação temporária para '{pattern}' por {mins_str} min ativada. Sem sair da linha."

        if match := re.match(r"^(?:anota|nota)\s*:\s*(.+)$", user_text.strip(), re.IGNORECASE):
            note_text = match.group(1).strip()
            await self._dispatch_tool("record_study_note", {"note": note_text}, executed_tools)
            return "Anotação salva no caderno da sessão."

        if any(word in lowered for word in ("resumo", "status", "como estou")):
            summary = await self.get_session_summary()
            executed_tools.append("get_session_summary")
            return (
                f"Resumo, {user_name}: estado {summary['current_state']}, matéria {summary['current_subject']}, "
                f"{summary['focus_minutes']} min focado ({summary['focus_ratio_pct']}%), "
                f"{summary['distraction_minutes']} min distraído."
            )

        summary = await self.get_session_summary()
        if api_error:
            prefix = "A API Gemini falhou agora."
        elif not self._gemini.is_configured():
            prefix = "Chave da API Gemini não configurada."
        else:
            prefix = "Limite financeiro diário/mensal da API atingido."

        return (
            f"{prefix} Você está em {summary['current_subject']} (foco: {summary['focus_ratio_pct']}%). "
            "Se precisar agir rápido no modo local, use: 'pausa 5', 'encerrar pausa', "
            "'liberar youtube 10', 'anota: ...' ou 'resumo'."
        )

    @staticmethod
    def _deterministic_nudge(
        user_name: str,
        persona: CoachPersona,
        level: InterventionLevel,
        reason: str,
        os_context: OSContextEvent | None,
        vision_signal: VisionSignalEvent | None,
        subject: str,
        variant_idx: int = 0,
    ) -> str:
        context_fact = ""
        if vision_signal is not None and vision_signal.camera_available and vision_signal.phone_detected:
            context_fact = "com o celular na mão"
        elif os_context is not None and not os_context.is_sensitive_window and os_context.window_title:
            context_fact = f"em '{os_context.window_title[:55]}'"

        fact_clause = f" ({context_fact})" if context_fact else ""
        reason_low = reason.lower()
        win_low = (
            ""
            if os_context is None or os_context.is_sensitive_window
            else f"{os_context.process_name} {os_context.window_title}".lower()
        )

        if (vision_signal is not None and vision_signal.camera_obstructed) or "tampada" in reason_low or "escura" in reason_low:
            pool = [
                f"{user_name}, {reason}. Se escondeu atrás de câmera tampada por quê? Destampa isso e volta para {subject} agora!",
                f"{user_name}, lente fechada não estuda {subject} sozinha ({reason}). Abre essa câmera ou clareia a sala já!",
                f"{user_name}, truque velho de tampar a webcam não funciona comigo ({reason}). Destampa e encara {subject}!",
            ]
        elif "fora de enquadramento" in reason_low or "pc em uso" in reason_low:
            pool = [
                f"{user_name}, ativou o modo invisível? ({reason}). Volta pro campo da câmera e foca em {subject}!",
                f"{user_name}, o mouse não anda sozinho e eu não consigo te ver ({reason}). Ajusta esse ângulo e retoma {subject}!",
                f"{user_name}, sair da câmera enquanto mexe no PC não cola ({reason}). Aparece no quadro e volta pra {subject}!",
            ]
        elif (
            "ausente" in reason_low
            or "campo de visão" in reason_low
            or (vision_signal is not None and vision_signal.camera_available and not vision_signal.face_present)
        ):
            pool = [
                f"{user_name}, cadeira vazia não passa na prova de {subject} ({reason}). Se parou, clica em pausa; se voltou, senta e estuda!",
                f"{user_name}, sumiu do radar faz tempo ({reason}). Volta pra frente de {subject} ou assume a pausa agora!",
                f"{user_name}, não adianta deixar {subject} aberto se não tem ninguém na câmera ({reason}). Volta pra cadeira!",
            ]
        elif (
            (vision_signal is not None and vision_signal.camera_available and vision_signal.phone_detected)
            or "celular" in reason_low
        ):
            pool = [
                f"{user_name}, essa tela pequena na mão não cai na prova de {subject} ({reason}){fact_clause}. Larga o celular agora!",
                f"{user_name}, você não vai aprovar em {subject} rolando feed ({reason}){fact_clause}. Celular virado pra baixo e volta pra tela!",
                f"{user_name}, cada olhada no celular zera teu ritmo em {subject} ({reason}){fact_clause}. Solta esse aparelho já!",
            ]
        elif "youtube" in reason_low or "youtube" in win_low:
            pool = [
                f"{user_name}, desde quando isso no YouTube cai na prova de {subject}? ({reason}){fact_clause}. Fecha essa aba agora!",
                f"{user_name}, o algoritmo do YouTube agradece sua presença, mas teu edital de {subject} não ({reason}){fact_clause}. Corta o vídeo já!",
                f"{user_name}, YouTube fora de tema é armadilha clássica contra {subject} ({reason}){fact_clause}. Volta pro estudo agora!",
            ]
        elif "segundo monitor" in reason_low or "2º monitor" in reason_low:
            pool = [
                f"{user_name}, achou que eu só olhava um monitor? ({reason}){fact_clause}. Limpa a segunda tela e volta pra {subject}!",
                f"{user_name}, deixar estudo numa tela e distração na outra não engana ninguém ({reason}){fact_clause}. Fecha o desvio e foca em {subject}!",
                f"{user_name}, seu segundo monitor está sabotando {subject} agora ({reason}){fact_clause}. Corta isso imediatamente!",
            ]
        else:
            pool = [
                f"{user_name}, esse desvio em {fact_clause or 'tela atual'} ({reason}) não coloca questão certa de {subject} no seu bolso. Corta já!",
                f"{user_name}, você prometeu estudar {subject}, não passear{fact_clause} ({reason}). Fecha isso e retoma o foco agora!",
                f"{user_name}, chega de anestesiar o tempo{fact_clause}: {reason}. De volta para {subject} imediatamente!",
            ]

        chosen = pool[abs(variant_idx) % len(pool)]
        if level >= InterventionLevel.FIRM_OVERLAY:
            if persona == CoachPersona.SARGENTO:
                return f"ULTIMATO: {chosen} Sem mais conversa!"
            return f"Chega de enrolação! {chosen}"
        if persona == CoachPersona.EQUILIBRADO:
            return f"{user_name}, retome os limites: {chosen}"
        return chosen

    # ------------------------------------------------------------------
    # History, Storage, EventBus & Voice Output Helpers
    # ------------------------------------------------------------------

    async def _record_message(self, msg: ChatMessage) -> None:
        self._history.append(msg)
        if len(self._history) > _MAX_HISTORY_MESSAGES:
            del self._history[: len(self._history) - _MAX_HISTORY_MESSAGES]

        if self._storage is not None:
            try:
                session_id = self._engine.get_status().session.session_id
                await self._storage.save_chat_message(session_id, msg)
            except Exception:
                logger.exception("Failed to save chat message to storage.")

        if self._event_bus is not None:
            await self._event_bus.publish(msg)

    async def _maybe_synthesize_and_play(
        self,
        text: str,
        *,
        speak: bool,
        local_playback: bool,
    ) -> tuple[str | None, str | None]:
        if not speak or not text.strip():
            return None, None

        if local_playback:
            # Spoken directly on local audio output; omit audio_base64 so browser tabs never double-play.
            try:
                await self._tts.speak(text)
            except Exception:
                logger.exception("Local TTS playback failed in coach reply.")
            return None, None

        audio_b64: str | None = None
        audio_mime: str | None = None
        try:
            audio_bytes, mime = await self._tts.synthesize(text)
            if audio_bytes:
                audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
                audio_mime = mime
        except Exception:
            logger.exception("TTS synthesis failed in coach reply.")

        return audio_b64, audio_mime
