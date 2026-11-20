"""Multimodal Semantic Judge implementing Zero-Prison study activity classification."""

from __future__ import annotations

import logging
from typing import Any

from google.genai import types

from focus_guard.ai.gemini_client import (
    BudgetExceededError,
    GeminiClient,
    extract_response_text,
    parse_json_object,
)
from focus_guard.core.config import AppConfig
from focus_guard.core.events import (
    ActivityVerdict,
    CoachPersona,
    OSContextEvent,
    ScreenSnapshot,
    SemanticVerdict,
    VisionSignalEvent,
)

logger = logging.getLogger(__name__)

_MUSIC_KEYWORDS = (
    "lofi",
    "lo-fi",
    "music",
    "música",
    "musica",
    "instrumental",
    "focus music",
    "study beats",
    "binaural",
    "playlist",
    "spotify",
)


def _persona_tone_rule(persona: CoachPersona) -> str:
    if persona == CoachPersona.SARGENTO:
        return (
            "Tom 'Sargento': linha-dura, cortante, zero desculpas e cobrança implacável citando "
            "exatamente o que está na tela ou na câmera. Sem sermão genérico."
        )
    if persona == CoachPersona.EQUILIBRADO:
        return (
            "Tom 'Equilibrado': firme nos limites e objetivo nos fatos da tela/câmera, "
            "sem sermão longo ou motivacional genérico."
        )
    return (
        "Tom 'Coach Firme e Direto': puxão de orelha incisivo, ácido, criativo e factual citando "
        "exatamente o que você viu na tela ou na câmera e cobrando retorno imediato à matéria, "
        "zero blá-blá-blá motivacional genérico."
    )


def _format_vision_summary(vision_signal: VisionSignalEvent | None) -> str:
    if vision_signal is None:
        return "Câmera: não fornecida neste ciclo."
    if not vision_signal.camera_available:
        return f"Câmera: indisponível ({vision_signal.error or 'sem hardware'})."
    return (
        "Câmera local: "
        f"camera_position={vision_signal.camera_position}, "
        f"calibrated={vision_signal.calibrated}, "
        f"looking_at_screen={vision_signal.looking_at_screen}, "
        f"rosto_presente={vision_signal.face_present}, "
        f"camera_obstructed={vision_signal.camera_obstructed}, "
        f"olhar_na_tela={vision_signal.gaze_on_screen_score:.2f}, "
        f"yaw={vision_signal.head_yaw_deg:.1f}° (delta={vision_signal.calibrated_yaw_delta_deg:+.1f}°), "
        f"pitch={vision_signal.head_pitch_deg:.1f}° (delta={vision_signal.calibrated_pitch_delta_deg:+.1f}°), "
        f"olhando_baixo={vision_signal.looking_down_score:.2f}, "
        f"sonolencia_perclos={vision_signal.drowsiness_perclos:.2f}, "
        f"bocejo={vision_signal.yawn_score:.2f}, "
        f"celular_detectado={vision_signal.phone_detected} "
        f"(conf={vision_signal.phone_confidence:.2f})."
    )


def _format_visible_windows(os_context: OSContextEvent) -> str:
    if not os_context.visible_windows:
        return "nenhuma janela adicional listada"
    items: list[str] = []
    for w in os_context.visible_windows[:12]:
        active_flag = ", ATIVA" if w.is_foreground else ""
        if w.is_sensitive:
            items.append(
                f"[{w.monitor_label}] {w.process_name}: [janela sensível protegida] "
                f"({w.screen_area_pct:.0f}%{active_flag})"
            )
        else:
            items.append(
                f"[{w.monitor_label}] {w.process_name}: '{w.window_title[:75]}' "
                f"({w.screen_area_pct:.0f}%{active_flag})"
            )
    return "; ".join(items)


def _format_notable_processes(os_context: OSContextEvent) -> str:
    if not os_context.notable_running_processes:
        return "nenhum processo notável listado"
    items: list[str] = []
    for p in os_context.notable_running_processes[:10]:
        audio_tag = ", ÁUDIO" if p.has_active_audio else ""
        items.append(
            f"{p.process_name}[cat={p.category}, cpu={p.cpu_percent:.1f}%, "
            f"ram={p.memory_mb:.0f}MB{audio_tag}, janelas={p.window_count}]"
        )
    return "; ".join(items)


class GeminiSemanticJudge:
    """SemanticJudgePort implementation backed by Gemini multimodal + local heuristic fallback."""

    def __init__(self, gemini_client: GeminiClient, config: AppConfig) -> None:
        self._gemini = gemini_client
        self._config = config

    def is_configured(self) -> bool:
        return self._gemini.is_configured()

    def reconfigure_api_key(self, api_key: str) -> None:
        self._gemini.reconfigure_api_key(api_key)

    async def evaluate(
        self,
        os_context: OSContextEvent,
        vision_signal: VisionSignalEvent | None,
        study_goal: str,
        current_subject: str,
        screenshot: ScreenSnapshot | None = None,
        webcam_frame_jpeg: bytes | None = None,
    ) -> SemanticVerdict:
        if os_context.is_sensitive_window:
            return SemanticVerdict(
                verdict=ActivityVerdict.UNCERTAIN,
                confidence=1.0,
                category="privacy_shield",
                reasoning="Janela sensível protegida.",
                used_vision=False,
                used_webcam=False,
            )

        if not self._gemini.is_configured() or not self._gemini.budget.can_spend():
            return self._heuristic_fallback(os_context, vision_signal, current_subject)

        has_image = bool(screenshot is not None and screenshot.image_bytes_jpeg)
        has_webcam = bool(webcam_frame_jpeg)
        system_instruction = self._build_system_instruction()
        prompt_text = self._build_user_prompt(
            os_context=os_context,
            vision_signal=vision_signal,
            study_goal=study_goal,
            current_subject=current_subject,
            has_image=has_image,
            has_webcam=has_webcam,
        )

        contents: Any
        if has_image or has_webcam:
            parts: list[Any] = [prompt_text]
            if has_image and screenshot is not None:
                parts.append(
                    types.Part.from_bytes(
                        data=screenshot.image_bytes_jpeg,
                        mime_type="image/jpeg",
                    )
                )
            if has_webcam and webcam_frame_jpeg is not None:
                parts.append(
                    types.Part.from_bytes(
                        data=bytes(webcam_frame_jpeg),
                        mime_type="image/jpeg",
                    )
                )
            contents = parts
        else:
            contents = prompt_text

        try:
            response, model_used, tokens_in, tokens_out, cost_usd = (
                await self._gemini.generate_content(
                    contents=contents,
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    temperature=0.2,
                )
            )
        except BudgetExceededError:
            logger.info("Budget throttled during semantic evaluation; using heuristic fallback.")
            return self._heuristic_fallback(os_context, vision_signal, current_subject)
        except Exception:
            logger.exception("Gemini semantic evaluation failed; using heuristic fallback.")
            return self._heuristic_fallback(os_context, vision_signal, current_subject)

        raw_text = extract_response_text(response)
        parsed = parse_json_object(raw_text)
        if not parsed:
            logger.warning("Gemini semantic judge returned unparseable JSON; using heuristic fallback.")
            fallback = self._heuristic_fallback(os_context, vision_signal, current_subject)
            fallback.tokens_in = tokens_in
            fallback.tokens_out = tokens_out
            fallback.estimated_cost_usd = cost_usd
            fallback.model_used = model_used
            return fallback

        return self._verdict_from_json(
            parsed=parsed,
            os_context=os_context,
            used_vision=has_image,
            used_webcam=has_webcam,
            model_used=model_used,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost_usd,
            user_name=self._config.user_name or "Pedro",
            current_subject=current_subject or self._config.current_subject or "seus estudos",
        )

    def _build_system_instruction(self) -> str:
        persona = CoachPersona(self._config.coach_persona)
        user_name = self._config.user_name or "Pedro"
        vis_cfg = self._config.vision
        return (
            f"Você é o Juiz Semântico rigoroso do FocusGuard monitorando a sessão de estudos de {user_name}.\n"
            "PROTOCOLO DE VERIFICAÇÃO CONJUNTIVA (4 PASSOS OBRIGATÓRIOS — VALE PARA TODO APP, SITE E MONITOR):\n"
            "PASSO 1 — Identifique 'foreground_topic': qual é o assunto/atividade EXATA da JANELA ATIVA EM PRIMEIRO PLANO "
            "(processo ativo, título da janela ativa, URL ativa e os pixels dessa janela)?\n"
            "PASSO 2 — Avalie 'matches_current_subject' (true/false): a JANELA ATIVA EM PRIMEIRO PLANO trata diretamente "
            "da 'Matéria atual' ou do 'Objetivo do estudo' (ou é o próprio painel do FocusGuard enquanto há material da matéria "
            "no outro monitor)?\n"
            "- REGRA ANTI-ÁLIBI (CRÍTICA): Uma aba, PDF ou janela de estudo aberta no segundo monitor ou em outra aba "
            "NUNCA justifica uma janela ativa fora do tema! Se a janela ativa for um vídeo aleatório no YouTube "
            "(ex: drones, tecnologia geral, vlogs, podcasts de entretenimento, jogos, futebol, clipes), redes sociais "
            "(Instagram, X/Twitter, Reddit, TikTok), notícias, compras, chat pessoal ou código de outro projeto enquanto "
            "a matéria atual é ITIL/concurso/outra disciplina, 'matches_current_subject' DEVE ser false e 'verdict' DEVE ser DISTRACTED!\n"
            "- Só aceite IDE/código (Antigravity, VS Code) como STUDYING se a Matéria atual for explicitamente programação/desenvolvimento.\n"
            "PASSO 3 — Avalie 'secondary_monitor_contamination' (true/false): analise as duas telas no print, todas as "
            "janelas em 'visible_windows' e os processos ativos em 'notable_running_processes'. Mesmo se a janela clicada for "
            "de estudo, há alguma janela/processo de entretenimento, vídeo fora do tema, rede social, jogo ou chat pessoal "
            "visível no 2º monitor ou em tela dividida? Se true, marque DISTRACTED!\n"
            "PASSO 4 — Avalie 'is_foreground_music_only' (true/false): a janela ativa em primeiro plano é apenas uma música/clipe/edit "
            "musical no YouTube/Spotify (e NÃO uma videoaula da matéria)?\n"
            "- Marque MUSIC_ALLOWED SOMENTE se for áudio/música de fundo ou escolha rápida de música (nunca para vídeo aleatório!).\n"
            "- Marque STUDYING SOMENTE quando 'matches_current_subject'=true e 'secondary_monitor_contamination'=false e "
            "'is_foreground_music_only'=false (ex: videoaula da matéria atual, Tec Concursos, QConcursos, simulado, PDF/resumo "
            "no Obsidian/NotebookLM da matéria atual).\n"
            "- Analise também 'usage_pattern': se window_switches_60s / interaction_mode='rapid_switching' indicar troca frenética "
            "distraída, marque DISTRACTED.\n\n"
            "CONSCIÊNCIA DE WEBCAM LATERAL, TABLET DE ANOTAÇÕES E ANTI-FALSO-POSITIVO DE CELULAR:\n"
            f"- A webcam fica na LATERAL em ângulo de perfil (camera_position='{vis_cfg.camera_position}', "
            f"zona calibrada da tela [{vis_cfg.screen_yaw_min_deg:.1f}°..{vis_cfg.screen_yaw_max_deg:.1f}°], "
            f"calibrated={vis_cfg.calibrated}).\n"
            "- Olhar de lado / visão de perfil (até 85°) em direção ao monitor principal ou secundário é "
            "POSTURA NORMAL DE ESTUDO ('focused_on_screen'). NUNCA marque olhar lateral como distração ou ausência!\n"
            "- TABLET DE ANOTAÇÕES: o usuário usa um TABLET perto do teclado para fazer anotações. "
            "Abaixar a cabeça ou olhar para baixo escrevendo/anotando perto do teclado é ESTUDO NORMAL ('focused_on_screen'), "
            "NÃO É CELULAR!\n"
            "- ANTI-FALSO-POSITIVO CRÍTICO: NUNCA confunda o encosto preto/mesh da cadeira, o fone de ouvido preto (headset), "
            "o microfone de mesa ou o tablet de anotações com um celular!\n"
            "- Só marque phone_in_use=true se você ENXERGAR CLARAMENTE um smartphone pequeno nas mãos do usuário "
            "sendo usado para distração (rolando feed, mensagens, vídeos no celular).\n"
            "- Só marque AWAY (physical_state='away' e user_visible_in_webcam=false) se a cadeira estiver vazia, "
            "o usuário estiver fora do enquadramento da câmera ou a lente estiver tampada/escura demais. "
            "(NÃO confie cegamente em rosto_presente=False do detector local quando a foto da webcam mostrar o usuário "
            "claramente sentado de perfil ou escrevendo no tablet — nesses casos user_visible_in_webcam=true!)\n"
            "- Se user_visible_in_webcam=false na Image 2 (Webcam), force verdict='AWAY' mesmo se o mouse/teclado estiver mexendo.\n\n"
            "REGRA DE PERSONALIDADE PARA suggested_message:\n"
            f"- {_persona_tone_rule(persona)}\n"
            "- Se o veredito NÃO for DISTRACTED nem AWAY, deixe suggested_message vazio (\"\").\n"
            "- Se o veredito FOR DISTRACTED ou AWAY, suggested_message DEVE ser uma cobrança afiada, "
            "criativa e factual citando exatamente o que está na janela ativa (ou no 2º monitor / câmera).\n\n"
            "RESPONDA EXCLUSIVAMENTE EM JSON VÁLIDO COM AS CHAVES:\n"
            '{"foreground_topic": "string_curta", "matches_current_subject": true, '
            '"secondary_monitor_contamination": false, "is_foreground_music_only": false, '
            '"user_visible_in_webcam": true, '
            '"verdict": "STUDYING|MUSIC_ALLOWED|DISTRACTED|UNCERTAIN|AWAY", '
            '"confidence": 0.0, "category": "string_curta", '
            '"reasoning": "explicacao_objetiva_ptbr", "suggested_message": "string", '
            '"phone_in_use": false, '
            '"physical_state": "focused_on_screen|using_phone|looking_away|drowsy|away"}'
        )

    def _build_user_prompt(
        self,
        os_context: OSContextEvent,
        vision_signal: VisionSignalEvent | None,
        study_goal: str,
        current_subject: str,
        has_image: bool,
        has_webcam: bool = False,
    ) -> str:
        audio_apps = ", ".join(os_context.background_audio_apps) or "nenhum"
        study_hints = ", ".join(self._config.hints.likely_study_keywords[:12]) or "n/d"
        suspicion_hints = ", ".join(self._config.hints.high_suspicion_patterns[:12]) or "n/d"
        up = os_context.usage_pattern
        usage_summary = (
            f"interaction_mode={up.interaction_mode}, "
            f"active_input_ratio_60s={up.active_input_ratio_60s:.2f}, "
            f"window_switches_60s={up.window_switches_60s}, "
            f"continuous_same_screen_sec={up.continuous_same_screen_sec:.1f}s"
        )
        if has_image and has_webcam:
            media_note = "sim — Image 1 = Screen(s), Image 2 = Webcam (analise tela/monitores e foto da webcam)"
        elif has_image:
            media_note = "sim — Image 1 = Screen(s) (sem foto da webcam neste ciclo)"
        elif has_webcam:
            media_note = "somente webcam — Image 1 = Webcam (sem screenshot neste ciclo)"
        else:
            media_note = "não"

        return (
            f"Objetivo do estudo: {study_goal}\n"
            f"Matéria atual: {current_subject}\n"
            f"Contexto do SO:\n"
            f"- Monitor ativo: {os_context.active_monitor_label}\n"
            f"- Processo ativo: {os_context.process_name}\n"
            f"- Título da janela ativa: {os_context.window_title}\n"
            f"- URL do navegador: {os_context.browser_url or 'n/d'}\n"
            f"- Janelas visíveis (visible_windows): {_format_visible_windows(os_context)}\n"
            f"- Processos em execução (notable_running_processes): {_format_notable_processes(os_context)}\n"
            f"- Padrão de uso 60s (usage_pattern): {usage_summary}\n"
            f"- Apps com áudio em segundo plano: {audio_apps}\n"
            f"- Tempo ocioso (mouse/teclado): {os_context.idle_seconds:.1f}s\n"
            f"- {_format_vision_summary(vision_signal)}\n"
            f"- Imagens anexadas: {media_note}\n"
            f"Dicas auxiliares (não são bloqueio rígido): estudo=[{study_hints}]; "
            f"alta_suspeita=[{suspicion_hints}]."
        )

    @staticmethod
    def _verdict_from_json(
        parsed: dict[str, Any],
        os_context: OSContextEvent,
        used_vision: bool,
        model_used: str,
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
        *,
        used_webcam: bool = False,
        user_name: str = "Pedro",
        current_subject: str = "seus estudos",
    ) -> SemanticVerdict:
        raw_verdict = str(parsed.get("verdict", "UNCERTAIN")).strip().upper()
        try:
            verdict = ActivityVerdict(raw_verdict)
        except ValueError:
            verdict = ActivityVerdict.UNCERTAIN

        try:
            confidence = float(parsed.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))

        raw_phone = parsed.get("phone_in_use", False)
        if isinstance(raw_phone, str):
            phone_in_use = raw_phone.strip().lower() in {"true", "1", "yes", "sim"}
        else:
            phone_in_use = bool(raw_phone)

        def _opt_bool(key: str) -> bool | None:
            if key not in parsed or parsed[key] is None:
                return None
            val = parsed[key]
            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                low = val.strip().lower()
                if low in {"true", "1", "yes", "sim"}:
                    return True
                if low in {"false", "0", "no", "nao", "não"}:
                    return False
            return bool(val)

        matches_subject = _opt_bool("matches_current_subject")
        secondary_contamination = _opt_bool("secondary_monitor_contamination")
        is_fg_music_only = _opt_bool("is_foreground_music_only")
        user_visible_webcam = _opt_bool("user_visible_in_webcam")
        fg_topic = str(parsed.get("foreground_topic") or "").strip()

        valid_physical_states = {
            "focused_on_screen",
            "using_phone",
            "looking_away",
            "drowsy",
            "away",
        }
        raw_physical = str(parsed.get("physical_state") or "").strip().lower()
        if raw_physical in valid_physical_states:
            physical_state = raw_physical
        elif phone_in_use:
            physical_state = "using_phone"
        elif verdict == ActivityVerdict.AWAY:
            physical_state = "away"
        else:
            physical_state = "focused_on_screen"

        category = str(parsed.get("category") or "semantic").strip()[:64] or "semantic"
        reasoning = str(parsed.get("reasoning") or "").strip()
        suggested = str(parsed.get("suggested_message") or "").strip()

        title_low = (os_context.window_title or "").lower()
        url_low = (os_context.browser_url or "").lower()
        is_focusguard_ui = "focusguard" in title_low or "127.0.0.1:8765" in url_low

        # Deterministic post-LLM conjunctive focus invariants:
        if secondary_contamination is True and verdict in (
            ActivityVerdict.STUDYING,
            ActivityVerdict.MUSIC_ALLOWED,
            ActivityVerdict.UNCERTAIN,
        ):
            verdict = ActivityVerdict.DISTRACTED
            if not reasoning:
                reasoning = "Distração visível no segundo monitor / tela dividida."

        if (
            matches_subject is False
            and not is_focusguard_ui
            and verdict == ActivityVerdict.STUDYING
        ):
            if is_fg_music_only is True:
                verdict = ActivityVerdict.MUSIC_ALLOWED
            else:
                verdict = ActivityVerdict.DISTRACTED
                if not reasoning:
                    topic_note = f" ({fg_topic})" if fg_topic else ""
                    reasoning = (
                        f"A janela ativa '{os_context.window_title[:60]}'{topic_note} "
                        f"não está alinhada com '{current_subject}'."
                    )

        if is_fg_music_only is True and verdict == ActivityVerdict.STUDYING:
            verdict = ActivityVerdict.MUSIC_ALLOWED

        if used_webcam and user_visible_webcam is False:
            verdict = ActivityVerdict.AWAY
            physical_state = "away"
            if not reasoning:
                reasoning = "Câmera não está vendo o usuário no enquadramento."

        if phone_in_use:
            verdict = ActivityVerdict.DISTRACTED
            if physical_state == "focused_on_screen":
                physical_state = "using_phone"
            if not suggested:
                suggested = (
                    f"{user_name or 'Pedro'}, larga o celular agora! "
                    f"Você está com o celular na mão em vez de focar em {current_subject or 'seus estudos'}."
                )

        if verdict == ActivityVerdict.DISTRACTED and not suggested:
            win_ref = os_context.window_title[:55] or os_context.process_name or "essa janela"
            suggested = (
                f"{user_name or 'Pedro'}, corta isso aí ('{win_ref}'). "
                f"Volta imediatamente para {current_subject or 'seus estudos'}!"
            )

        if verdict == ActivityVerdict.AWAY and not suggested:
            suggested = (
                f"{user_name or 'Pedro'}, não estou conseguindo te ver na câmera! "
                f"Ajusta o ângulo ou volta agora para {current_subject or 'seus estudos'}."
            )

        if verdict not in (ActivityVerdict.DISTRACTED, ActivityVerdict.AWAY):
            suggested = ""

        return SemanticVerdict(
            verdict=verdict,
            confidence=confidence,
            category=category,
            app_context=os_context.window_title,
            reasoning=reasoning,
            suggested_message=suggested,
            used_vision=used_vision,
            used_webcam=used_webcam,
            phone_in_use=phone_in_use,
            physical_state=physical_state,
            model_used=model_used,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            estimated_cost_usd=cost_usd,
        )

    def _heuristic_fallback(
        self,
        os_context: OSContextEvent,
        vision_signal: VisionSignalEvent | None,
        current_subject: str,
    ) -> SemanticVerdict:
        """Resilient local Zero-Prison heuristic used when API is unavailable or throttled."""
        user_name = self._config.user_name or "Pedro"
        subject = current_subject or self._config.current_subject or "seus estudos"

        if vision_signal is not None and vision_signal.camera_available:
            if vision_signal.phone_detected and vision_signal.phone_confidence >= 0.45:
                return SemanticVerdict(
                    verdict=ActivityVerdict.DISTRACTED,
                    confidence=max(0.75, min(1.0, vision_signal.phone_confidence)),
                    category="phone_distraction",
                    app_context=os_context.window_title,
                    reasoning="Celular detectado pela câmera local durante a sessão.",
                    suggested_message=(
                        f"{user_name}, larga o celular agora. Você está na janela "
                        f"'{os_context.window_title[:50]}' e deveria focar em {subject}."
                    ),
                    used_vision=False,
                    used_webcam=False,
                    phone_in_use=True,
                    physical_state="using_phone",
                    model_used="local-heuristic",
                )
            if not vision_signal.face_present and os_context.idle_seconds >= 30:
                return SemanticVerdict(
                    verdict=ActivityVerdict.AWAY,
                    confidence=0.8,
                    category="away",
                    app_context=os_context.window_title,
                    reasoning="Rosto ausente da câmera e sem atividade recente no teclado/mouse.",
                    used_vision=False,
                    physical_state="away",
                    model_used="local-heuristic",
                )

        proc_lower = os_context.process_name.lower()
        haystack = (
            f"{proc_lower} {os_context.window_title.lower()} "
            f"{(os_context.browser_url or '').lower()}"
        )

        for pattern in self._config.hints.high_suspicion_patterns:
            pat = pattern.strip().lower()
            if pat and pat in haystack:
                return SemanticVerdict(
                    verdict=ActivityVerdict.DISTRACTED,
                    confidence=0.82,
                    category="high_suspicion_pattern",
                    app_context=os_context.window_title,
                    reasoning=f"Padrão de distração identificado localmente: {pattern}.",
                    suggested_message=(
                        f"{user_name}, você abriu '{os_context.window_title[:55]}' ({pattern}). "
                        f"Fecha isso e volta para {subject} agora."
                    ),
                    used_vision=False,
                    model_used="local-heuristic",
                )

        if any(kw in haystack for kw in _MUSIC_KEYWORDS) or "spotify.exe" in proc_lower:
            return SemanticVerdict(
                verdict=ActivityVerdict.MUSIC_ALLOWED,
                confidence=0.75,
                category="music",
                app_context=os_context.window_title,
                reasoning="Conteúdo identificado localmente como música/áudio de fundo.",
                used_vision=False,
                model_used="local-heuristic",
            )

        study_keywords = [kw.strip().lower() for kw in self._config.hints.likely_study_keywords if kw.strip()]
        subj_lower = subject.strip().lower()
        if subj_lower and subj_lower not in {"geral", "general"}:
            study_keywords.append(subj_lower)

        for kw in study_keywords:
            if kw in haystack:
                return SemanticVerdict(
                    verdict=ActivityVerdict.STUDYING,
                    confidence=0.78,
                    category="study_keyword",
                    app_context=os_context.window_title,
                    reasoning=f"Sinal local compatível com estudo ({kw}).",
                    used_vision=False,
                    model_used="local-heuristic",
                )

        if os_context.idle_seconds >= self._config.behavior.absence_alert_sec:
            return SemanticVerdict(
                verdict=ActivityVerdict.AWAY,
                confidence=0.72,
                category="idle",
                app_context=os_context.window_title,
                reasoning=f"Sem interação por {int(os_context.idle_seconds)}s.",
                used_vision=False,
                physical_state="away",
                model_used="local-heuristic",
            )

        return SemanticVerdict(
            verdict=ActivityVerdict.UNCERTAIN,
            confidence=0.45,
            category="ambiguous",
            app_context=os_context.window_title,
            reasoning="Contexto multiuso ou ambíguo na heurística local (Zero-Prison).",
            used_vision=False,
            model_used="local-heuristic",
        )
