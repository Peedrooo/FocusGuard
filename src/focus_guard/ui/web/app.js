(() => {
  "use strict";

  const state = {
    status: null,
    settings: null,
    verdicts: [],
    chatHistory: [],
    notes: [],
    subjectStats: [],
    subjectStatsPeriod: "all",
    subjectSyncCounter: 0,
    ws: null,
    wsRetryMs: 1000,
    wsReconnectTimer: null,
    wsPingTimer: null,
    mediaRecorder: null,
    mediaStream: null,
    audioChunks: [],
    isRecordingWebMic: false,
    isRecordingNativePtt: false,
    activeAudioEl: null,
    playedAudioIds: new Set(),
    webcamPollTimer: null,
    webcamPreviewLoading: false,
    webcamObjectUrl: null,
    audioCtx: null,
    vuTimer: null,
  };

  const el = (id) => document.getElementById(id);

  function escapeHtml(value) {
    if (value === null || value === undefined) return "";
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  function formatDuration(totalSeconds) {
    const secs = Math.max(0, Math.floor(Number(totalSeconds) || 0));
    const h = Math.floor(secs / 3600);
    const m = Math.floor((secs % 3600) / 60);
    const s = secs % 60;
    const pad = (n) => String(n).padStart(2, "0");
    return `${pad(h)}:${pad(m)}:${pad(s)}`;
  }

  function formatShortDuration(totalSeconds) {
    const secs = Math.max(0, Math.floor(Number(totalSeconds) || 0));
    const m = Math.floor(secs / 60);
    const s = secs % 60;
    return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  }

  function formatTime(isoString) {
    if (!isoString) return "";
    const d = new Date(isoString);
    if (Number.isNaN(d.getTime())) return "";
    return d.toLocaleTimeString("pt-BR", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  }

  function formatBrl(usdAmount, rate) {
    const val = (Number(usdAmount) || 0) * (Number(rate) || 5.8);
    return val.toLocaleString("pt-BR", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    });
  }

  function showToast(message, variant = "info") {
    const stack = el("toast-stack");
    if (!stack) return;
    const toast = document.createElement("div");
    toast.className = `toast ${variant === "error" ? "toast-error" : variant === "success" ? "toast-success" : ""}`;
    toast.textContent = message;
    stack.appendChild(toast);
    setTimeout(() => {
      toast.remove();
    }, 4500);
  }

  async function apiJson(url, options = {}) {
    const headers = Object.assign({}, options.headers || {});
    if (options.body && !(options.body instanceof FormData) && !headers["Content-Type"]) {
      headers["Content-Type"] = "application/json";
    }
    const resp = await fetch(url, Object.assign({}, options, { headers }));
    let data = {};
    try {
      data = await resp.json();
    } catch (_err) {
      data = {};
    }
    if (!resp.ok) {
      const detail = data.detail || `Erro HTTP ${resp.status}`;
      throw new Error(detail);
    }
    return data;
  }

  function stopCoachAudio() {
    if (state.activeAudioEl) {
      try {
        state.activeAudioEl.pause();
        state.activeAudioEl.currentTime = 0;
      } catch (_err) {
        // Ignore stop errors
      }
      state.activeAudioEl = null;
    }
  }

  function playCoachAudio(base64Audio, mimeType) {
    if (!base64Audio) return;
    try {
      stopCoachAudio();
      const mime = mimeType || "audio/mpeg";
      const audio = new Audio(`data:${mime};base64,${base64Audio}`);
      state.activeAudioEl = audio;
      audio.addEventListener("ended", () => {
        if (state.activeAudioEl === audio) {
          state.activeAudioEl = null;
        }
      });
      audio.play().catch(() => {
        // Browser autoplay policy may block audio if user hasn't interacted yet
      });
    } catch (_err) {
      // Non-fatal audio playback fallback
    }
  }

  function stateCssClass(focusState) {
    switch (focusState) {
      case "FOCUSED":
        return "state-focused";
      case "GRACE_WINDOW":
        return "state-grace";
      case "BREAK_MODE":
        return "state-break";
      case "CONVERSATION_MODE":
        return "state-conversation";
      case "DRIFT_SUSPECTED":
        return "state-drift";
      case "INTERVENING":
        return "state-intervening";
      case "PRIVACY_MODE":
        return "state-privacy";
      default:
        return "state-paused";
    }
  }

  function renderStatus(status) {
    if (!status) return;
    state.status = status;

    const session = status.session || {};
    const currentState = session.current_state || "PAUSED";

    // Topbar state pill
    const pill = el("state-pill");
    const pillText = el("state-pill-text");
    if (pill && pillText) {
      pill.className = `state-pill ${stateCssClass(currentState)}`;
      pillText.textContent = currentState;
    }

    // Session timer & KPIs
    const timerStr = formatDuration(session.focus_seconds || 0);
    if (el("topbar-timer")) el("topbar-timer").textContent = timerStr;
    if (el("kpi-focus-time")) el("kpi-focus-time").textContent = timerStr;
    if (el("kpi-focus-score")) {
      const pct = Math.round(Number(session.focus_ratio_pct ?? 100));
      el("kpi-focus-score").textContent = `${pct}%`;
    }
    if (el("kpi-distraction-time")) {
      el("kpi-distraction-time").textContent = formatShortDuration(session.distraction_seconds || 0);
    }
    if (el("kpi-interventions")) {
      el("kpi-interventions").textContent = String(session.interventions_count || 0);
    }

    // Session Start/Pause button & contextual buttons
    const sessionBtn = el("btn-session-toggle");
    if (sessionBtn) {
      const isPaused = currentState === "PAUSED" || !status.running;
      sessionBtn.textContent = isPaused ? "Iniciar Sessão" : "Pausar Sessão";
      sessionBtn.classList.toggle("btn-primary", isPaused);
      sessionBtn.classList.toggle("btn-outline", !isPaused);
    }

    const ackBtn = el("btn-ack-focus");
    if (ackBtn) {
      const showAck = currentState === "INTERVENING" || currentState === "DRIFT_SUSPECTED";
      ackBtn.classList.toggle("hidden", !showAck);
    }

    const endBreakBtn = el("btn-end-break");
    if (endBreakBtn) {
      const inBreak = currentState === "BREAK_MODE";
      endBreakBtn.classList.toggle("hidden", !inBreak);
      if (inBreak && session.active_break_remaining_sec > 0) {
        endBreakBtn.textContent = `Encerrar Pausa (${formatShortDuration(session.active_break_remaining_sec)})`;
      } else {
        endBreakBtn.textContent = "Encerrar Pausa";
      }
    }

    // Privacy Kill-Switch
    const privBtn = el("btn-privacy-toggle");
    if (privBtn) {
      privBtn.classList.toggle("active-toggle", Boolean(status.privacy_mode));
      privBtn.textContent = status.privacy_mode
        ? "Privacidade: ATIVA (Ctrl+Shift+P)"
        : "Privacidade (Ctrl+Shift+P)";
    }
    const privBadge = el("privacy-hud-badge");
    if (privBadge) {
      privBadge.textContent = status.privacy_mode ? "Privacidade ON (Sensores Pausados)" : "RAM Efêmera";
    }

    // API Key status pill
    const apiBtn = el("btn-api-key-pill");
    if (apiBtn) {
      const configured = Boolean(status.api_key_configured);
      apiBtn.classList.toggle("api-ok", configured);
      apiBtn.classList.toggle("api-missing", !configured);
      apiBtn.textContent = configured ? "API Key: OK" : "API Key: Pendente";
    }

    // Study goal & subject (avoid overwriting while user is typing)
    const goalInput = el("goal-input");
    const subjectInput = el("subject-input");
    if (goalInput && document.activeElement !== goalInput && session.study_goal) {
      goalInput.value = session.study_goal;
    }
    if (subjectInput && document.activeElement !== subjectInput && session.current_subject) {
      subjectInput.value = session.current_subject;
    }
    syncSubjectChips(session.current_subject || "");

    // Persona badge & select
    const persona = status.coach_persona || "firme";
    if (el("persona-badge")) el("persona-badge").textContent = `Persona: ${persona}`;
    const personaSelect = el("setting-persona");
    if (personaSelect && document.activeElement !== personaSelect) {
      personaSelect.value = persona;
    }

    // OS Context, Vision, Allowances, Budget, Verdicts
    renderOSContext(status.latest_os);
    renderVision(status.latest_vision);
    renderAllowances(status.active_allowances || []);
    renderBudget(status.budget);
    syncWebcamPreviewState();

    if (Array.isArray(session.recent_verdicts) && session.recent_verdicts.length > 0) {
      state.verdicts = session.recent_verdicts.slice(-20);
      renderVerdicts();
    }
  }

  function applySettings(settings) {
    if (!settings) return;
    state.settings = settings;
    if (el("setting-persona") && document.activeElement !== el("setting-persona") && settings.coach_persona) {
      el("setting-persona").value = settings.coach_persona;
    }
    if (el("setting-max-daily") && document.activeElement !== el("setting-max-daily") && settings.max_daily_usd !== undefined) {
      el("setting-max-daily").value = String(settings.max_daily_usd);
    }
    if (el("setting-max-monthly") && document.activeElement !== el("setting-max-monthly") && settings.max_monthly_usd !== undefined) {
      el("setting-max-monthly").value = String(settings.max_monthly_usd);
    }
    if (el("setting-yt-grace") && document.activeElement !== el("setting-yt-grace") && settings.youtube_music_grace_sec !== undefined) {
      el("setting-yt-grace").value = String(settings.youtube_music_grace_sec);
    }
    if (el("setting-phone-sec") && document.activeElement !== el("setting-phone-sec") && settings.phone_distraction_sec !== undefined) {
      el("setting-phone-sec").value = String(settings.phone_distraction_sec);
    }
    if (el("setting-camera-unseen-sec") && document.activeElement !== el("setting-camera-unseen-sec") && settings.camera_not_seeing_user_sec !== undefined) {
      el("setting-camera-unseen-sec").value = String(settings.camera_not_seeing_user_sec);
    }
    if (el("setting-capture-mode") && document.activeElement !== el("setting-capture-mode") && settings.capture_mode) {
      el("setting-capture-mode").value = settings.capture_mode;
    }
    if (el("setting-speak-all") && settings.speak_on_all_interventions !== undefined) {
      el("setting-speak-all").checked = Boolean(settings.speak_on_all_interventions);
    }
  }

  function syncSubjectChips(currentSubject) {
    const normalized = String(currentSubject || "").trim().toLowerCase();
    document.querySelectorAll("#subject-chips .chip").forEach((chip) => {
      const sub = (chip.dataset.subject || "").trim().toLowerCase();
      chip.classList.toggle("active", Boolean(normalized && sub === normalized));
    });
  }

  function interactionModeBadgeClass(mode) {
    switch (mode) {
      case "active_work":
        return "badge-green";
      case "reading_or_watching":
        return "badge-cyan";
      case "rapid_switching":
      case "idle_with_media":
        return "badge-amber";
      default:
        return "badge-muted";
    }
  }

  function renderOSContext(osEvt) {
    if (!osEvt) return;
    if (el("os-process")) el("os-process").textContent = osEvt.process_name || "—";
    if (el("os-monitor-badge")) el("os-monitor-badge").textContent = osEvt.active_monitor_label || "M1";
    if (el("os-title")) el("os-title").textContent = osEvt.window_title || "—";
    if (el("os-url")) el("os-url").textContent = osEvt.browser_url || "—";

    const sensBadge = el("os-sensitive-badge");
    if (sensBadge) {
      sensBadge.classList.toggle("hidden", !osEvt.is_sensitive_window);
    }

    const bgBadge = el("bg-audio-badge");
    if (bgBadge) {
      const apps = Array.isArray(osEvt.background_audio_apps) ? osEvt.background_audio_apps : [];
      if (apps.length === 0) {
        bgBadge.textContent = "Sem áudio em background";
        bgBadge.className = "badge badge-muted";
      } else {
        const hasYoutube = apps.some((a) => /youtube|music|spotify/i.test(String(a)));
        bgBadge.textContent = hasYoutube
          ? `Música em Background (${apps.join(", ")})`
          : `Áudio Background: ${apps.join(", ")}`;
        bgBadge.className = "badge badge-cyan";
      }
    }

    // Usage pattern metrics
    const pattern = osEvt.usage_pattern || {};
    const mode = pattern.interaction_mode || "idle";
    const modeEl = el("usage-interaction-mode");
    if (modeEl) {
      modeEl.textContent = mode;
      modeEl.className = `badge ${interactionModeBadgeClass(mode)}`;
    }
    if (el("usage-input-ratio")) {
      const pct = Math.round((Number(pattern.active_input_ratio_60s) || 0) * 100);
      el("usage-input-ratio").textContent = `${pct}%`;
    }
    if (el("usage-switches")) {
      el("usage-switches").textContent = String(pattern.window_switches_60s || 0);
    }
    if (el("usage-same-screen")) {
      const sameSec = Math.round(Number(pattern.continuous_same_screen_sec) || 0);
      el("usage-same-screen").textContent = `${sameSec}s`;
    }

    // Visible windows across M1 / M2
    const winListEl = el("visible-windows-list");
    if (winListEl) {
      const windows = Array.isArray(osEvt.visible_windows) ? osEvt.visible_windows : [];
      if (windows.length === 0) {
        winListEl.innerHTML = '<span class="empty-hint">Nenhuma janela visível capturada ainda.</span>';
      } else {
        winListEl.innerHTML = windows
          .slice(0, 10)
          .map((w) => {
            const mon = escapeHtml(w.monitor_label || "M1");
            const proc = escapeHtml(w.process_name || "");
            const title = w.is_sensitive ? "[Janela Sensível]" : escapeHtml(w.window_title || "");
            const area = Math.round(Number(w.screen_area_pct) || 0);
            const fgBadge = w.is_foreground ? '<span class="badge badge-green">Ativa</span>' : "";
            return `<div class="window-row">
              <div class="window-row-main">
                <span class="badge badge-cyan">${mon}</span>
                <strong class="mono-sm">${proc}</strong>
                <span class="window-title-text" title="${title}">${title}</span>
              </div>
              <div class="window-meta">${fgBadge} ${area}%</div>
            </div>`;
          })
          .join("");
      }
    }

    // Notable running processes
    const procListEl = el("notable-processes-list");
    if (procListEl) {
      const procs = Array.isArray(osEvt.notable_running_processes) ? osEvt.notable_running_processes : [];
      if (procs.length === 0) {
        procListEl.innerHTML = '<span class="empty-hint">Nenhum processo destacado.</span>';
      } else {
        procListEl.innerHTML = procs
          .slice(0, 10)
          .map((p) => {
            const name = escapeHtml(p.process_name || "");
            const cpu = Number(p.cpu_percent || 0).toFixed(1);
            const mem = Math.round(Number(p.memory_mb) || 0);
            const audioTag = p.has_active_audio ? '<span class="badge badge-cyan">🔊</span>' : "";
            const cat = escapeHtml(p.category || "app");
            return `<div class="process-row">
              <div class="process-row-main">
                <strong class="mono-sm">${name}</strong>
                <span class="badge badge-muted">${cat}</span>
                ${audioTag}
              </div>
              <span class="window-meta mono-sm">${cpu}% / ${mem}MB</span>
            </div>`;
          })
          .join("");
      }
    }
  }

  function renderVision(vision) {
    const presenceVal = el("vision-presence-val");
    const presenceBar = el("vision-presence-bar");
    const gazeVal = el("vision-gaze-val");
    const gazeBar = el("vision-gaze-bar");
    const fatigueVal = el("vision-fatigue-val");
    const fatigueBar = el("vision-fatigue-bar");
    const phoneVal = el("vision-phone-val");
    const phoneBar = el("vision-phone-bar");

    const yawBadge = el("vision-yaw-badge");
    const pitchBadge = el("vision-pitch-badge");
    const posBadge = el("vision-pos-badge");
    const gazeStatusBadge = el("vision-gaze-badge");
    const unseenBadge = el("vision-unseen-timer-badge");
    const previewWrap = el("webcam-preview-wrap");

    if (!vision) return;

    if (!vision.camera_available) {
      if (presenceVal) presenceVal.textContent = "Câmera indisponível";
      if (presenceBar) presenceBar.style.width = "0%";
      if (gazeStatusBadge) {
        gazeStatusBadge.textContent = "Sem Câmera";
        gazeStatusBadge.className = "badge badge-muted";
      }
      if (unseenBadge) unseenBadge.classList.add("hidden");
      if (previewWrap) {
        previewWrap.classList.remove("webcam-status-ok", "webcam-status-warn", "webcam-status-alert");
      }
      return;
    }

    const obstructed = Boolean(vision.camera_obstructed);
    const present = Boolean(vision.face_present) && !obstructed;
    if (presenceVal) {
      presenceVal.textContent = obstructed
        ? "Câmera tampada/escura"
        : (present ? "Presente" : "Não visível");
    }
    if (presenceBar) {
      presenceBar.style.width = present ? "100%" : "10%";
      presenceBar.className = `progress-fill ${present ? "fill-green" : "fill-red"}`;
    }

    const gazePct = Math.round((Number(vision.gaze_on_screen_score) || 0) * 100);
    if (gazeVal) gazeVal.textContent = `${gazePct}%`;
    if (gazeBar) gazeBar.style.width = `${Math.min(100, Math.max(0, gazePct))}%`;

    const fatiguePct = Math.round((Number(vision.drowsiness_perclos) || 0) * 100);
    if (fatigueVal) fatigueVal.textContent = `${fatiguePct}%`;
    if (fatigueBar) {
      fatigueBar.style.width = `${Math.min(100, Math.max(0, fatiguePct))}%`;
      fatigueBar.className = `progress-fill ${fatiguePct >= 45 ? "fill-red" : "fill-amber"}`;
    }

    const phoneConfPct = Math.round((Number(vision.phone_confidence) || 0) * 100);
    const phoneDetected = Boolean(vision.phone_detected);
    if (phoneVal) {
      phoneVal.textContent = phoneDetected
        ? `DETECTADO (${phoneConfPct}%)`
        : phoneConfPct > 0
          ? `Monitorando (${phoneConfPct}%)`
          : "Não detectado";
    }
    if (phoneBar) {
      phoneBar.style.width = `${phoneDetected ? Math.max(60, phoneConfPct) : phoneConfPct}%`;
      phoneBar.className = `progress-fill ${phoneDetected ? "fill-red" : "fill-cyan"}`;
    }

    // Pose & Lateral Calibration Badges
    if (yawBadge) {
      yawBadge.textContent = `Yaw: ${Number(vision.head_yaw_deg || 0).toFixed(1)}°`;
    }
    if (pitchBadge) {
      pitchBadge.textContent = `Pitch: ${Number(vision.head_pitch_deg || 0).toFixed(1)}°`;
    }
    if (posBadge) {
      const pos = vision.camera_position || "right";
      posBadge.textContent = `Posição: ${pos}${vision.calibrated ? " ✓" : ""}`;
      posBadge.className = `badge ${vision.calibrated ? "badge-cyan" : "badge-muted"}`;
    }
    if (gazeStatusBadge) {
      const lookDown = Number(vision.looking_down_score || 0);
      const lookingOk = Boolean(vision.looking_at_screen) && !phoneDetected && lookDown <= 0.60;
      if (obstructed) {
        gazeStatusBadge.textContent = "CÂMERA ESCURA / TAMPADA ⚠";
        gazeStatusBadge.className = "badge badge-danger";
      } else if (!present) {
        gazeStatusBadge.textContent = "NÃO ESTOU TE VENDO ⚠";
        gazeStatusBadge.className = "badge badge-danger";
      } else if (lookingOk) {
        gazeStatusBadge.textContent = "OLHANDO P/ TELA ✓";
        gazeStatusBadge.className = "badge badge-green";
      } else {
        gazeStatusBadge.textContent = "FORA / BAIXO ⚠";
        gazeStatusBadge.className = "badge badge-amber";
      }
    }

    const unseenSec = Number(state.status?.session?.camera_unseen_seconds || 0);
    const unseenLimit = Number(state.settings?.camera_not_seeing_user_sec || 20);
    if (unseenBadge) {
      if (unseenSec >= 1) {
        unseenBadge.textContent = `Sem te ver: ${Math.round(unseenSec)}s / ${unseenLimit}s`;
        unseenBadge.classList.remove("hidden");
      } else {
        unseenBadge.classList.add("hidden");
      }
    }

    if (previewWrap) {
      previewWrap.classList.remove("webcam-status-ok", "webcam-status-warn", "webcam-status-alert");
      const currentState = state.status?.session?.current_state || "PAUSED";
      if (
        obstructed ||
        !present ||
        phoneDetected ||
        currentState === "INTERVENING" ||
        currentState === "DRIFT_SUSPECTED"
      ) {
        previewWrap.classList.add("webcam-status-alert");
      } else if (unseenSec >= 1 || currentState === "GRACE_WINDOW" || !vision.looking_at_screen) {
        previewWrap.classList.add("webcam-status-warn");
      } else {
        previewWrap.classList.add("webcam-status-ok");
      }
    }
  }

  function renderAllowances(allowances) {
    const container = el("allowances-list");
    if (!container) return;
    if (!allowances || allowances.length === 0) {
      container.innerHTML = '<span class="empty-hint">Nenhuma liberação temporária ativa.</span>';
      return;
    }
    container.innerHTML = allowances
      .map((item) => {
        const exp = formatTime(item.expires_at);
        return `<span class="allowance-tag">
          <strong>${escapeHtml(item.pattern)}</strong>
          <span>até ${escapeHtml(exp)}</span>
          <small>(${escapeHtml(item.reason)})</small>
        </span>`;
      })
      .join("");
  }

  function verdictBadgeClass(verdict) {
    switch (verdict) {
      case "STUDYING":
        return "badge-green";
      case "MUSIC_ALLOWED":
        return "badge-cyan";
      case "DISTRACTED":
        return "badge-danger";
      case "AWAY":
        return "badge-amber";
      default:
        return "badge-muted";
    }
  }

  function renderVerdicts() {
    const feed = el("verdict-feed");
    const countEl = el("verdicts-count");
    if (countEl) countEl.textContent = `${state.verdicts.length} deduções`;
    if (!feed) return;

    if (state.verdicts.length === 0) {
      feed.innerHTML =
        '<div class="empty-hint">Inicie a sessão para acompanhar as análises contextuais do Gemini.</div>';
      return;
    }

    const items = [...state.verdicts].reverse();
    feed.innerHTML = items
      .map((v) => {
        const confPct = Math.round((Number(v.confidence) || 0) * 100);
        const visionTag = v.used_vision
          ? '<span class="badge badge-cyan">Tela</span>'
          : "";
        const webcamTag = v.used_webcam
          ? '<span class="badge badge-cyan">Webcam</span>'
          : "";
        const phoneTag = v.phone_in_use
          ? '<span class="badge badge-danger">Celular</span>'
          : "";
        const cacheTag = v.cached
          ? '<span class="badge badge-muted">Cache</span>'
          : "";
        return `<article class="verdict-item">
          <div class="verdict-top">
            <div class="verdict-badges">
              <span class="badge ${verdictBadgeClass(v.verdict)}">${escapeHtml(v.verdict)}</span>
              <span class="badge badge-muted">${confPct}% conf.</span>
              ${visionTag}
              ${webcamTag}
              ${phoneTag}
              ${cacheTag}
            </div>
            <span class="verdict-time">${escapeHtml(formatTime(v.timestamp))}</span>
          </div>
          <div class="verdict-context">${escapeHtml(v.app_context || v.category || "Contexto")}</div>
          <div class="verdict-reason">${escapeHtml(v.reasoning || "Sem detalhes.")}</div>
        </article>`;
      })
      .join("");
  }

  function renderBudget(budget) {
    if (!budget) return;
    const rate = Number(budget.brl_exchange_rate) || 5.8;
    const dailyUsd = Number(budget.daily_usd) || 0;
    const monthlyUsd = Number(budget.monthly_usd) || 0;
    const maxDaily = Math.max(0.01, Number(budget.max_daily_usd) || 0.35);
    const maxMonthly = Math.max(0.01, Number(budget.max_monthly_usd) || 8.0);

    if (el("budget-daily-text")) {
      el("budget-daily-text").textContent = `US$ ${dailyUsd.toFixed(4)} (R$ ${formatBrl(dailyUsd, rate)})`;
    }
    if (el("budget-monthly-text")) {
      el("budget-monthly-text").textContent = `US$ ${monthlyUsd.toFixed(4)} (R$ ${formatBrl(monthlyUsd, rate)})`;
    }
    if (el("budget-daily-cap")) {
      el("budget-daily-cap").textContent = `Teto diário: US$ ${maxDaily.toFixed(2)} (R$ ${formatBrl(maxDaily, rate)})`;
    }
    if (el("budget-monthly-cap")) {
      el("budget-monthly-cap").textContent = `Teto configurado: US$ ${maxMonthly.toFixed(2)} / ref. Google AI Pro US$ 10,00/mês`;
    }

    const dailyPct = Math.min(100, Math.round((dailyUsd / maxDaily) * 100));
    const monthlyPct = Math.min(100, Math.round((monthlyUsd / 10.0) * 100));

    if (el("budget-daily-bar")) el("budget-daily-bar").style.width = `${dailyPct}%`;
    if (el("budget-monthly-bar")) {
      const mBar = el("budget-monthly-bar");
      mBar.style.width = `${monthlyPct}%`;
      mBar.className = `progress-fill ${budget.is_throttled ? "fill-red" : budget.warning_reached ? "fill-amber" : "fill-green"}`;
    }

    if (el("budget-tokens-in")) el("budget-tokens-in").textContent = String(budget.daily_tokens_in || 0);
    if (el("budget-tokens-out")) el("budget-tokens-out").textContent = String(budget.daily_tokens_out || 0);
    if (el("budget-calls")) el("budget-calls").textContent = String(budget.daily_calls || 0);

    const badge = el("budget-status-badge");
    if (badge) {
      if (budget.is_throttled) {
        badge.textContent = "Limite Atingido (Throttled)";
        badge.className = "badge badge-danger";
      } else if (budget.warning_reached) {
        badge.textContent = "Acima de 80% do Teto";
        badge.className = "badge badge-amber";
      } else {
        badge.textContent = "Dentro do teto";
        badge.className = "badge badge-green";
      }
    }

    if (el("setting-max-daily") && document.activeElement !== el("setting-max-daily")) {
      el("setting-max-daily").value = String(maxDaily);
    }
    if (el("setting-max-monthly") && document.activeElement !== el("setting-max-monthly")) {
      el("setting-max-monthly").value = String(maxMonthly);
    }
  }

  function renderChatHistory() {
    const box = el("chat-history");
    if (!box) return;
    if (!state.chatHistory || state.chatHistory.length === 0) {
      box.innerHTML =
        '<div class="empty-hint">Mande uma pergunta por texto, use o microfone do navegador ou o PTT nativo do Windows (<code>Ctrl+Shift+Space</code>).</div>';
      return;
    }

    box.innerHTML = state.chatHistory
      .map((msg) => {
        const role = msg.role || "coach";
        const roleLabel = role === "user" ? "Você" : role === "coach" ? "Coach" : "Sistema";
        const tools = Array.isArray(msg.tool_calls) && msg.tool_calls.length > 0
          ? `<div class="chat-meta"><span>Ferramentas: ${escapeHtml(msg.tool_calls.join(", "))}</span></div>`
          : "";
        return `<div class="chat-bubble ${escapeHtml(role)}">
          <div class="chat-meta">
            <strong>${escapeHtml(roleLabel)}</strong>
            <span>${escapeHtml(formatTime(msg.timestamp))}</span>
          </div>
          ${tools}
          <div>${escapeHtml(msg.text)}</div>
        </div>`;
      })
      .join("");
    box.scrollTop = box.scrollHeight;
  }

  function appendChatMessage(msg, opts = {}) {
    if (!msg) return;
    const fromWs = Boolean(opts.fromWs);
    const existsById = Boolean(msg.id && state.chatHistory.some((m) => m.id === msg.id));

    // Also deduplicate optimistic/synthetic user bubbles vs backend WebSocket ChatMessage
    let existsRecentDuplicate = false;
    if (!existsById && msg.role === "user" && msg.text) {
      const normText = String(msg.text).trim();
      const nowMs = msg.timestamp ? Date.parse(msg.timestamp) || Date.now() : Date.now();
      const recentMatch = state.chatHistory.find((m) => {
        if (m.role !== "user" || String(m.text || "").trim() !== normText) return false;
        const mMs = m.timestamp ? Date.parse(m.timestamp) || 0 : 0;
        return Math.abs(nowMs - mMs) <= 15000;
      });
      if (recentMatch) {
        existsRecentDuplicate = true;
        if (msg.id && String(recentMatch.id || "").match(/^(local|mic|ptt)-/)) {
          recentMatch.id = msg.id;
        }
      }
    }

    if (!existsById && !existsRecentDuplicate) {
      state.chatHistory.push(msg);
      if (state.chatHistory.length > 60) {
        state.chatHistory = state.chatHistory.slice(-60);
      }
      renderChatHistory();
    }

    if (msg.role === "coach" && msg.audio_base64) {
      const audioKey = msg.id || `${msg.timestamp || ""}|${String(msg.text || "").slice(0, 48)}`;
      const alreadyPlayed = state.playedAudioIds.has(audioKey);
      const hiddenTabWs = fromWs && document.hidden;
      if (!alreadyPlayed && !hiddenTabWs) {
        state.playedAudioIds.add(audioKey);
        if (state.playedAudioIds.size > 100) {
          const oldest = state.playedAudioIds.values().next().value;
          if (oldest) state.playedAudioIds.delete(oldest);
        }
        playCoachAudio(msg.audio_base64, msg.audio_mime);
      }
    }
  }

  function renderNotes() {
    const list = el("notes-list");
    const count = el("notes-count");
    if (count) count.textContent = `${state.notes.length} notas`;
    if (!list) return;
    if (state.notes.length === 0) {
      list.innerHTML = '<div class="empty-hint">Nenhuma anotação registrada nesta sessão.</div>';
      return;
    }
    list.innerHTML = [...state.notes]
      .reverse()
      .map((n) => `<div class="note-item">
        <header>
          <strong>${escapeHtml(n.subject || "Geral")}</strong>
          <span>${escapeHtml(formatTime(n.timestamp))}</span>
        </header>
        <div>${escapeHtml(n.note)}</div>
      </div>`)
      .join("");
  }

  function showInterventionBanner(cmd) {
    const banner = el("intervention-banner");
    if (!banner || !cmd) return;
    if (el("intervention-banner-title")) {
      el("intervention-banner-title").textContent = cmd.title || "FocusGuard — Intervenção";
    }
    if (el("intervention-banner-msg")) {
      el("intervention-banner-msg").textContent = cmd.message || "";
    }
    banner.classList.remove("hidden");
  }

  function hideInterventionBanner() {
    const banner = el("intervention-banner");
    if (banner) banner.classList.add("hidden");
  }

  function stopWebcamPreview(message) {
    if (state.webcamPollTimer) {
      clearInterval(state.webcamPollTimer);
      state.webcamPollTimer = null;
    }
    const img = el("webcam-preview-img");
    const placeholder = el("webcam-preview-placeholder");
    if (img) {
      img.classList.add("hidden");
      img.removeAttribute("src");
    }
    if (state.webcamObjectUrl) {
      URL.revokeObjectURL(state.webcamObjectUrl);
      state.webcamObjectUrl = null;
    }
    if (placeholder) {
      placeholder.textContent = message || "Preview oculto.";
      placeholder.classList.remove("hidden");
    }
  }

  async function fetchWebcamPreview() {
    if (state.webcamPreviewLoading) return;
    const isPrivacy = Boolean(state.status?.privacy_mode) || state.status?.session?.current_state === "PRIVACY_MODE";
    const showCam = Boolean(el("webcam-show-toggle")?.checked);
    if (isPrivacy || !showCam) {
      stopWebcamPreview(isPrivacy ? "Preview desativado (Modo Privacidade ativo)." : "Preview da câmera oculto.");
      return;
    }

    state.webcamPreviewLoading = true;
    const annotate = el("webcam-annotate-toggle")?.checked !== false;
    const url = `/api/vision/preview.jpg?annotate=${annotate ? "true" : "false"}&t=${Date.now()}`;
    const controller = typeof AbortController !== "undefined" ? new AbortController() : null;
    const timeoutId = controller ? setTimeout(() => controller.abort(), 4000) : null;
    try {
      const resp = await fetch(url, {
        cache: "no-store",
        signal: controller ? controller.signal : undefined,
      });
      if (state.status?.privacy_mode || !el("webcam-show-toggle")?.checked) {
        return;
      }
      const img = el("webcam-preview-img");
      const placeholder = el("webcam-preview-placeholder");
      if (resp.status === 200) {
        const blob = await resp.blob();
        const objUrl = URL.createObjectURL(blob);
        const prevUrl = state.webcamObjectUrl;
        state.webcamObjectUrl = objUrl;
        if (img) {
          img.src = objUrl;
          img.classList.remove("hidden");
        }
        if (placeholder) placeholder.classList.add("hidden");
        if (prevUrl) URL.revokeObjectURL(prevUrl);
      } else {
        if (img) img.classList.add("hidden");
        if (placeholder) {
          placeholder.textContent = "Aguardando frame da webcam...";
          placeholder.classList.remove("hidden");
        }
      }
    } catch (_err) {
      // Transient preview fetch error or timeout handled silently
    } finally {
      if (timeoutId) clearTimeout(timeoutId);
      state.webcamPreviewLoading = false;
    }
  }

  function syncWebcamPreviewState() {
    const isPrivacy = Boolean(state.status?.privacy_mode) || state.status?.session?.current_state === "PRIVACY_MODE";
    const showCam = Boolean(el("webcam-show-toggle")?.checked);
    if (isPrivacy) {
      stopWebcamPreview("Preview bloqueado durante o Modo Privacidade.");
      return;
    }
    if (!showCam) {
      stopWebcamPreview("Preview da câmera oculto pelo usuário.");
      return;
    }
    if (!state.webcamPollTimer) {
      fetchWebcamPreview();
      state.webcamPollTimer = setInterval(fetchWebcamPreview, 400);
    }
  }

  function renderMicState(data) {
    if (!data) return;
    const badge = el("mic-state-badge");
    const vuBar = el("mic-vu-bar");
    const stopBtn = el("btn-native-ptt-stop");
    const statusText = el("voice-status-text");

    const micState = data.state || "idle";
    if (badge) {
      const dev = data.device_name ? ` (${data.device_name})` : "";
      badge.textContent = `Mic: ${micState}${dev}`;
      if (micState === "listening") {
        badge.className = "badge badge-danger";
      } else if (micState === "processing") {
        badge.className = "badge badge-amber";
      } else if (micState === "error") {
        badge.className = "badge badge-danger";
      } else {
        badge.className = "badge badge-muted";
      }
    }

    if (vuBar && !state.isRecordingWebMic) {
      const pct = Math.min(100, Math.max(0, Math.round((Number(data.rms_level) || 0) * 800)));
      vuBar.style.width = `${pct}%`;
    }

    if (stopBtn) {
      stopBtn.classList.toggle("hidden", micState !== "listening");
    }

    if (statusText && data.message) {
      statusText.textContent = data.message;
    }
  }

  async function loadAudioDevices() {
    try {
      const data = await apiJson("/api/audio/devices");
      const outSelect = el("audio-output-select");
      const inSelect = el("audio-input-select");

      const populate = (selectEl, items, selectedName) => {
        if (!selectEl) return;
        const list = Array.isArray(items) ? items : [];
        const options = ['<option value="">Padrão do Sistema</option>'];
        list.forEach((item) => {
          const name = typeof item === "string" ? item : (item.name || item.label || "");
          if (!name) return;
          const esc = escapeHtml(name);
          const sel = selectedName && name === selectedName ? " selected" : "";
          options.push(`<option value="${esc}"${sel}>${esc}</option>`);
        });
        if (selectedName && !list.some((it) => (typeof it === "string" ? it : it.name) === selectedName)) {
          const escSel = escapeHtml(selectedName);
          options.push(`<option value="${escSel}" selected>${escSel}</option>`);
        }
        selectEl.innerHTML = options.join("");
      };

      populate(outSelect, data.output_devices, data.output_device_name);
      populate(inSelect, data.input_devices, data.input_device_name);
    } catch (_err) {
      // Audio device enumeration is optional on headless/missing hardware
    }
  }

  function renderSubjectStats() {
    const listEl = el("subject-stats-list");
    const totalBadge = el("subjects-total-badge");
    if (!listEl) return;

    const period = state.subjectStatsPeriod === "today" ? "today" : "all";
    const items = Array.isArray(state.subjectStats) ? state.subjectStats : [];
    const metricKey = period === "today" ? "today_focus_seconds" : "total_focus_seconds";

    const filtered = items
      .filter((it) => period === "all" || Number(it.today_focus_seconds || 0) > 0 || it.is_current)
      .slice()
      .sort((a, b) => {
        const diff = (Number(b[metricKey]) || 0) - (Number(a[metricKey]) || 0);
        if (Math.abs(diff) > 0.01) return diff;
        if (Boolean(b.is_current) !== Boolean(a.is_current)) return b.is_current ? 1 : -1;
        return String(a.subject || "").localeCompare(String(b.subject || ""), "pt-BR");
      });

    const totalSec = filtered.reduce((acc, it) => acc + Math.max(0, Number(it[metricKey]) || 0), 0);
    if (totalBadge) {
      const label = period === "today" ? "Hoje" : "Total";
      totalBadge.textContent = `${label}: ${formatDuration(totalSec)}`;
    }

    if (filtered.length === 0) {
      listEl.innerHTML =
        '<div class="empty-hint">Nenhum tempo de estudo registrado neste período ainda.</div>';
      return;
    }

    listEl.innerHTML = filtered
      .map((it) => {
        const subj = String(it.subject || "Geral");
        const sec = Math.max(0, Number(it[metricKey]) || 0);
        const todaySec = Math.max(0, Number(it.today_focus_seconds) || 0);
        const allSec = Math.max(0, Number(it.total_focus_seconds) || 0);
        const sharePct = totalSec > 0 ? Math.round((sec / totalSec) * 100) : 0;
        const focusPct = Math.round(Number(it.focus_ratio_pct ?? 100));
        const sessCount = Math.max(1, Number(it.sessions_count || 1));
        const sessLabel = sessCount === 1 ? "1 sessão" : `${sessCount} sessões`;
        const isCurrent = Boolean(it.is_current);
        const fillCls = isCurrent ? "fill-green" : "fill-cyan";
        const currentBadge = isCurrent ? '<span class="badge badge-green">Atual</span>' : "";
        const secondaryTime =
          period === "today"
            ? `Acumulado: ${formatDuration(allSec)}`
            : `Hoje: ${formatDuration(todaySec)}`;
        const escSubj = escapeHtml(subj);

        return `<div class="subject-stat-row${isCurrent ? " is-current" : ""}" data-select-subject="${escSubj}" title="Clique para alternar para ${escSubj}">
          <div class="subject-stat-head">
            <div class="subject-stat-name">
              <span class="name-text">${escSubj}</span>
              ${currentBadge}
            </div>
            <span class="subject-stat-time mono">${formatDuration(sec)}</span>
          </div>
          <div class="progress-track">
            <div class="progress-fill ${fillCls}" style="width: ${Math.min(100, Math.max(sec > 0 ? 3 : 0, sharePct))}%"></div>
          </div>
          <div class="subject-stat-meta">
            <span>${secondaryTime} • Foco ${focusPct}% • ${sessLabel}</span>
            <span class="mono-sm">${sharePct}%</span>
          </div>
        </div>`;
      })
      .join("");
  }

  async function refreshSubjectStats() {
    try {
      const data = await apiJson("/api/stats/subjects");
      if (Array.isArray(data.subjects)) {
        state.subjectStats = data.subjects;
        renderSubjectStats();
      }
    } catch (_err) {
      // Non-fatal periodic subject stats refresh
    }
  }

  async function refreshStatus() {
    try {
      const data = await apiJson("/api/status");
      if (data.status) renderStatus(data.status);
      if (data.settings) applySettings(data.settings);
      if (Array.isArray(data.chat_history)) {
        state.chatHistory = data.chat_history;
        renderChatHistory();
      }
      if (Array.isArray(data.notes)) {
        state.notes = data.notes;
        renderNotes();
      }
      if (Array.isArray(data.subject_stats)) {
        state.subjectStats = data.subject_stats;
        renderSubjectStats();
      }
    } catch (err) {
      showToast(`Falha ao carregar status: ${err.message}`, "error");
    }
  }

  function setWsVisual(connected, text) {
    const dot = el("ws-dot");
    const label = el("ws-status-text");
    if (dot) {
      dot.classList.toggle("online", connected);
      dot.classList.toggle("offline", !connected);
    }
    if (label) label.textContent = text;
  }

  function handleWsEvent(payload) {
    if (!payload || !payload.type) return;
    const { type, data } = payload;

    switch (type) {
      case "SystemStatusSnapshot":
        renderStatus(data);
        break;
      case "OSContextEvent":
        renderOSContext(data);
        break;
      case "VisionSignalEvent":
        renderVision(data);
        break;
      case "SemanticVerdict":
        state.verdicts.push(data);
        if (state.verdicts.length > 25) state.verdicts = state.verdicts.slice(-25);
        renderVerdicts();
        break;
      case "FocusStateChangedEvent":
        if (state.status && state.status.session) {
          state.status.session.current_state = data.new_state;
          state.status.session.focus_ratio_pct = data.focus_score ?? state.status.session.focus_ratio_pct;
          renderStatus(state.status);
        }
        if (data.new_state === "FOCUSED") {
          hideInterventionBanner();
        }
        break;
      case "InterventionCommand":
        stopCoachAudio();
        showInterventionBanner(data);
        showToast(data.message || "Hora de voltar ao foco!", "error");
        break;
      case "ProactiveSuggestionEvent":
        showToast(data.message || "Sugestão do Coach", "info");
        appendChatMessage({
          id: data.id,
          timestamp: data.timestamp,
          role: "coach",
          text: data.message,
        });
        break;
      case "ChatMessage":
        appendChatMessage(data, { fromWs: true });
        break;
      case "MicStateEvent":
        renderMicState(data);
        break;
      default:
        break;
    }
  }

  function connectWebSocket() {
    if (state.wsReconnectTimer) {
      clearTimeout(state.wsReconnectTimer);
      state.wsReconnectTimer = null;
    }
    if (state.wsPingTimer) {
      clearInterval(state.wsPingTimer);
      state.wsPingTimer = null;
    }

    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${proto}//${window.location.host}/ws`;

    try {
      const ws = new WebSocket(wsUrl);
      state.ws = ws;

      ws.onopen = () => {
        state.wsRetryMs = 1000;
        setWsVisual(true, "Conectado em tempo real");
        state.wsPingTimer = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send("ping");
          }
        }, 20000);
      };

      ws.onmessage = (event) => {
        try {
          const parsed = JSON.parse(event.data);
          if (parsed.type !== "pong") {
            handleWsEvent(parsed);
          }
        } catch (_err) {
          // Ignore malformed frame
        }
      };

      ws.onclose = () => {
        setWsVisual(false, "Reconectando...");
        scheduleWsReconnect();
      };

      ws.onerror = () => {
        setWsVisual(false, "Erro na conexão");
        ws.close();
      };
    } catch (_err) {
      setWsVisual(false, "Reconectando...");
      scheduleWsReconnect();
    }
  }

  function scheduleWsReconnect() {
    if (state.wsReconnectTimer) return;
    const wait = state.wsRetryMs;
    state.wsRetryMs = Math.min(8000, Math.round(state.wsRetryMs * 1.6));
    state.wsReconnectTimer = setTimeout(() => {
      state.wsReconnectTimer = null;
      connectWebSocket();
    }, wait);
  }

  async function sendCoachText(text) {
    const clean = String(text || "").trim();
    if (!clean) return;
    stopCoachAudio();
    const speak = el("chat-speak-toggle") ? el("chat-speak-toggle").checked : true;

    appendChatMessage({
      id: `local-${Date.now()}`,
      timestamp: new Date().toISOString(),
      role: "user",
      text: clean,
    });

    try {
      const res = await apiJson("/api/chat/text", {
        method: "POST",
        body: JSON.stringify({ text: clean, speak }),
      });
      if (res.reply) {
        appendChatMessage(res.reply);
      }
    } catch (err) {
      showToast(`Erro no Coach: ${err.message}`, "error");
    }
  }

  function stopWebVuMeter() {
    if (state.vuTimer) {
      clearInterval(state.vuTimer);
      state.vuTimer = null;
    }
    if (state.audioCtx) {
      state.audioCtx.close().catch(() => {});
      state.audioCtx = null;
    }
    const vuBar = el("mic-vu-bar");
    if (vuBar) vuBar.style.width = "0%";
  }

  function startWebVuMeter(stream) {
    stopWebVuMeter();
    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!AudioCtx) return;
      const ctx = new AudioCtx();
      const source = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);
      const dataArr = new Uint8Array(analyser.fftSize);
      state.audioCtx = ctx;
      state.vuTimer = setInterval(() => {
        if (!state.isRecordingWebMic) return;
        analyser.getByteTimeDomainData(dataArr);
        let sumSq = 0;
        for (let i = 0; i < dataArr.length; i += 1) {
          const v = (dataArr[i] - 128) / 128;
          sumSq += v * v;
        }
        const rms = Math.sqrt(sumSq / dataArr.length);
        const pct = Math.min(100, Math.max(0, Math.round(rms * 450)));
        const vuBar = el("mic-vu-bar");
        if (vuBar) vuBar.style.width = `${pct}%`;
      }, 60);
    } catch (_err) {
      // AudioContext VU feedback is best-effort
    }
  }

  async function toggleWebMic() {
    const btn = el("btn-web-mic");
    const statusText = el("voice-status-text");

    if (state.isRecordingWebMic && state.mediaRecorder) {
      state.mediaRecorder.stop();
      state.isRecordingWebMic = false;
      stopWebVuMeter();
      if (btn) {
        btn.classList.remove("recording");
        btn.textContent = "🎙️ Gravar Mic (Web)";
      }
      if (statusText) statusText.textContent = "Enviando áudio...";
      return;
    }

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showToast("Navegador não suporta gravação de microfone.", "error");
      return;
    }

    try {
      stopCoachAudio();
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      state.mediaStream = stream;
      state.audioChunks = [];
      startWebVuMeter(stream);

      const recorder = new MediaRecorder(stream);
      state.mediaRecorder = recorder;

      recorder.ondataavailable = (ev) => {
        if (ev.data && ev.data.size > 0) {
          state.audioChunks.push(ev.data);
        }
      };

      recorder.onstop = async () => {
        stopWebVuMeter();
        const rawMime = recorder.mimeType || "audio/webm";
        const mimeType = rawMime.split(";")[0].trim() || "audio/webm";
        const blob = new Blob(state.audioChunks, { type: mimeType });
        stream.getTracks().forEach((t) => t.stop());
        state.mediaStream = null;
        state.mediaRecorder = null;

        if (blob.size === 0) {
          if (statusText) statusText.textContent = "Áudio vazio.";
          return;
        }

        const speak = el("chat-speak-toggle") ? el("chat-speak-toggle").checked : true;
        const formData = new FormData();
        formData.append("file", blob, "coach_voice.webm");
        formData.append("mime_type", mimeType);
        formData.append("speak", speak ? "true" : "false");

        try {
          const res = await apiJson("/api/chat/audio", {
            method: "POST",
            body: formData,
          });
          if (res.transcript) {
            appendChatMessage({
              id: `mic-${Date.now()}`,
              timestamp: new Date().toISOString(),
              role: "user",
              text: res.transcript,
            });
          }
          if (res.reply) {
            appendChatMessage(res.reply);
          }
          if (statusText) statusText.textContent = "Pronto para ouvir";
        } catch (err) {
          if (statusText) statusText.textContent = "Falha ao enviar áudio";
          showToast(`Erro no áudio: ${err.message}`, "error");
        }
      };

      recorder.start();
      state.isRecordingWebMic = true;
      if (btn) {
        btn.classList.add("recording");
        btn.textContent = "⏹️ Parar & Enviar";
      }
      if (statusText) statusText.textContent = "Gravando pelo navegador...";
    } catch (err) {
      stopWebVuMeter();
      showToast(`Permissão de microfone negada: ${err.message}`, "error");
      if (statusText) statusText.textContent = "Mic bloqueado";
    }
  }

  async function triggerNativePtt() {
    if (state.isRecordingNativePtt) return;
    stopCoachAudio();
    const btn = el("btn-native-ptt");
    const statusText = el("voice-status-text");
    const speak = el("chat-speak-toggle") ? el("chat-speak-toggle").checked : true;

    state.isRecordingNativePtt = true;
    if (btn) {
      btn.classList.add("recording");
      btn.disabled = true;
      btn.textContent = "🎧 Ouvindo Mic Windows...";
    }
    if (statusText) statusText.textContent = "Fale no microfone (para no silêncio)...";

    try {
      const res = await apiJson("/api/chat/ptt", {
        method: "POST",
        body: JSON.stringify({ speak }),
      });
      if (res.transcript) {
        appendChatMessage({
          id: `ptt-${Date.now()}`,
          timestamp: new Date().toISOString(),
          role: "user",
          text: res.transcript,
        });
      }
      if (res.reply) {
        appendChatMessage(res.reply);
      } else if (res.audio_base64) {
        playCoachAudio(res.audio_base64, "audio/mpeg");
      }
      if (statusText) statusText.textContent = "Pronto para ouvir";
    } catch (err) {
      showToast(`Mic Windows PTT: ${err.message}`, "error");
      if (statusText) statusText.textContent = "Pronto para ouvir";
    } finally {
      state.isRecordingNativePtt = false;
      if (btn) {
        btn.classList.remove("recording");
        btn.disabled = false;
        btn.textContent = "🎧 Mic Windows (PTT)";
      }
    }
  }

  function openApiModal() {
    const modal = el("api-key-modal");
    if (!modal) return;
    modal.classList.remove("hidden");
    modal.setAttribute("aria-hidden", "false");
    const inp = el("api-key-input");
    if (inp) inp.focus();
  }

  function closeApiModal() {
    const modal = el("api-key-modal");
    if (!modal) return;
    modal.classList.add("hidden");
    modal.setAttribute("aria-hidden", "true");
  }

  function bindEvents() {
    el("btn-session-toggle")?.addEventListener("click", async () => {
      const curState = state.status?.session?.current_state || "PAUSED";
      const isPaused = curState === "PAUSED" || !state.status?.running;
      try {
        const res = isPaused
          ? await apiJson("/api/session/start", {
              method: "POST",
              body: JSON.stringify({
                study_goal: el("goal-input")?.value || null,
                current_subject: el("subject-input")?.value || null,
              }),
            })
          : await apiJson("/api/session/pause", { method: "POST" });
        if (res.status) renderStatus(res.status);
      } catch (err) {
        showToast(err.message, "error");
      }
    });

    el("btn-ack-focus")?.addEventListener("click", async () => {
      try {
        const res = await apiJson("/api/session/ack-focus", { method: "POST" });
        hideInterventionBanner();
        if (res.status) renderStatus(res.status);
      } catch (err) {
        showToast(err.message, "error");
      }
    });

    el("banner-ack-btn")?.addEventListener("click", async () => {
      try {
        const res = await apiJson("/api/session/ack-focus", { method: "POST" });
        hideInterventionBanner();
        if (res.status) renderStatus(res.status);
      } catch (err) {
        showToast(err.message, "error");
      }
    });

    el("banner-break-btn")?.addEventListener("click", async () => {
      try {
        const res = await apiJson("/api/session/break", {
          method: "POST",
          body: JSON.stringify({ minutes: 5 }),
        });
        hideInterventionBanner();
        if (res.status) renderStatus(res.status);
      } catch (err) {
        showToast(err.message, "error");
      }
    });

    el("banner-dismiss-btn")?.addEventListener("click", hideInterventionBanner);

    el("btn-end-break")?.addEventListener("click", async () => {
      try {
        const res = await apiJson("/api/session/end-break", { method: "POST" });
        if (res.status) renderStatus(res.status);
      } catch (err) {
        showToast(err.message, "error");
      }
    });

    el("btn-privacy-toggle")?.addEventListener("click", async () => {
      try {
        const res = await apiJson("/api/privacy/toggle", { method: "POST" });
        if (res.status) renderStatus(res.status);
        showToast(
          res.privacy_mode ? "Modo Privacidade ativado." : "Modo Privacidade desativado.",
          "info"
        );
      } catch (err) {
        showToast(err.message, "error");
      }
    });

    el("btn-test-nudge")?.addEventListener("click", async () => {
      const level = Number(el("nudge-level-select")?.value || 2);
      try {
        const res = await apiJson("/api/test-nudge", {
          method: "POST",
          body: JSON.stringify({ level }),
        });
        if (res.command) {
          showInterventionBanner(res.command);
          showToast(`Teste nível ${level} disparado!`, "info");
        }
      } catch (err) {
        showToast(err.message, "error");
      }
    });

    el("btn-api-key-pill")?.addEventListener("click", openApiModal);
    el("btn-close-api-modal")?.addEventListener("click", closeApiModal);
    el("btn-cancel-api-modal")?.addEventListener("click", closeApiModal);

    el("api-key-form")?.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      const apiKey = el("api-key-input")?.value?.trim() || "";
      if (!apiKey) return;
      try {
        const res = await apiJson("/api/config/api-key", {
          method: "POST",
          body: JSON.stringify({ api_key: apiKey }),
        });
        if (el("api-key-input")) el("api-key-input").value = "";
        closeApiModal();
        if (res.status) renderStatus(res.status);
        showToast("GEMINI_API_KEY salva e ativada!", "success");
      } catch (err) {
        showToast(err.message, "error");
      }
    });

    el("goal-form")?.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      const study_goal = el("goal-input")?.value?.trim() || "";
      const current_subject = el("subject-input")?.value?.trim() || "";
      if (!study_goal || !current_subject) return;
      try {
        const res = await apiJson("/api/goal", {
          method: "POST",
          body: JSON.stringify({ study_goal, current_subject }),
        });
        if (res.status) renderStatus(res.status);
        if (Array.isArray(res.subject_stats)) {
          state.subjectStats = res.subject_stats;
          renderSubjectStats();
        }
        showToast(`Foco atualizado para ${current_subject}.`, "success");
      } catch (err) {
        showToast(err.message, "error");
      }
    });

    document.querySelectorAll("#subject-chips .chip").forEach((chip) => {
      chip.addEventListener("click", async () => {
        const subject = chip.dataset.subject || "";
        const goal = el("goal-input")?.value?.trim() || "Estudo para Concurso Publico";
        if (el("subject-input")) el("subject-input").value = subject;
        try {
          const res = await apiJson("/api/goal", {
            method: "POST",
            body: JSON.stringify({ study_goal: goal, current_subject: subject }),
          });
          if (res.status) renderStatus(res.status);
          if (Array.isArray(res.subject_stats)) {
            state.subjectStats = res.subject_stats;
            renderSubjectStats();
          }
          showToast(`Matéria alterada: ${subject}`, "success");
        } catch (err) {
          showToast(err.message, "error");
        }
      });
    });

    document.querySelectorAll("[data-subject-period]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const period = btn.getAttribute("data-subject-period") || "all";
        state.subjectStatsPeriod = period === "today" ? "today" : "all";
        document.querySelectorAll("[data-subject-period]").forEach((other) => {
          other.classList.toggle("active", other === btn);
        });
        renderSubjectStats();
      });
    });

    el("subject-stats-list")?.addEventListener("click", async (ev) => {
      const row = ev.target?.closest?.("[data-select-subject]");
      if (!row) return;
      const subject = row.getAttribute("data-select-subject") || "";
      if (!subject) return;
      const goal = el("goal-input")?.value?.trim() || "Estudo para Concurso Publico";
      if (el("subject-input")) el("subject-input").value = subject;
      try {
        const res = await apiJson("/api/goal", {
          method: "POST",
          body: JSON.stringify({ study_goal: goal, current_subject: subject }),
        });
        if (res.status) renderStatus(res.status);
        if (Array.isArray(res.subject_stats)) {
          state.subjectStats = res.subject_stats;
          renderSubjectStats();
        }
        showToast(`Matéria alterada: ${subject}`, "success");
      } catch (err) {
        showToast(err.message, "error");
      }
    });

    el("allowance-form")?.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      const pattern = el("allowance-pattern")?.value?.trim() || "";
      const minutes = Number(el("allowance-minutes")?.value || 5);
      const reason = el("allowance-reason")?.value?.trim() || "Liberado pelo usuário";
      if (!pattern) return;
      try {
        const res = await apiJson("/api/allowance", {
          method: "POST",
          body: JSON.stringify({ pattern, minutes, reason }),
        });
        if (el("allowance-pattern")) el("allowance-pattern").value = "";
        if (res.status) renderStatus(res.status);
        showToast(`${pattern} liberado por ${minutes} min.`, "success");
      } catch (err) {
        showToast(err.message, "error");
      }
    });

    el("chat-form")?.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      const inp = el("chat-input");
      const text = inp?.value || "";
      if (!text.trim()) return;
      if (inp) inp.value = "";
      await sendCoachText(text);
    });

    document.querySelectorAll("[data-quick-msg]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const msg = btn.getAttribute("data-quick-msg") || "";
        await sendCoachText(msg);
      });
    });

    el("btn-quick-break")?.addEventListener("click", async () => {
      try {
        const res = await apiJson("/api/session/break", {
          method: "POST",
          body: JSON.stringify({ minutes: 5 }),
        });
        if (res.status) renderStatus(res.status);
        showToast("Pausa de 5 minutos iniciada.", "info");
      } catch (err) {
        showToast(err.message, "error");
      }
    });

    el("btn-web-mic")?.addEventListener("click", toggleWebMic);
    el("btn-native-ptt")?.addEventListener("click", triggerNativePtt);
    el("btn-native-ptt-stop")?.addEventListener("click", async () => {
      try {
        await apiJson("/api/chat/ptt/stop", { method: "POST" });
      } catch (err) {
        showToast(`Falha ao parar PTT: ${err.message}`, "error");
      }
    });

    el("webcam-show-toggle")?.addEventListener("change", syncWebcamPreviewState);
    el("webcam-annotate-toggle")?.addEventListener("change", fetchWebcamPreview);

    document.querySelectorAll(".cal-btn[data-cal-mode]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const mode = btn.getAttribute("data-cal-mode") || "primary_screen";
        const statusEl = el("webcam-cal-status");
        const allBtns = document.querySelectorAll(".cal-btn[data-cal-mode]");
        allBtns.forEach((b) => { b.disabled = true; });
        if (statusEl) {
          statusEl.textContent = mode === "reset" ? "Resetando calibração..." : "Olhe para o alvo (2s)...";
        }
        try {
          const res = await apiJson("/api/vision/calibrate", {
            method: "POST",
            body: JSON.stringify({ mode, duration_sec: 2.0 }),
          });
          if (res.status) renderStatus(res.status);
          if (statusEl) {
            statusEl.textContent = mode === "reset" ? "Calibração resetada." : `Calibrado (${mode}) ✓`;
          }
          showToast(`Calibração '${mode}' concluída!`, "success");
        } catch (err) {
          if (statusEl) statusEl.textContent = "";
          showToast(`Calibração: ${err.message}`, "error");
        } finally {
          allBtns.forEach((b) => { b.disabled = false; });
        }
      });
    });

    const saveAudioDevices = async () => {
      const output_device_name = el("audio-output-select")?.value || null;
      const input_device_name = el("audio-input-select")?.value || null;
      try {
        await apiJson("/api/audio/devices", {
          method: "POST",
          body: JSON.stringify({ output_device_name, input_device_name }),
        });
        showToast("Dispositivos de áudio atualizados!", "success");
      } catch (err) {
        showToast(`Áudio: ${err.message}`, "error");
      }
    };
    el("audio-output-select")?.addEventListener("change", saveAudioDevices);
    el("audio-input-select")?.addEventListener("change", saveAudioDevices);

    el("btn-test-audio")?.addEventListener("click", async () => {
      const btn = el("btn-test-audio");
      if (btn) btn.disabled = true;
      try {
        const res = await apiJson("/api/audio/test", { method: "POST", body: JSON.stringify({}) });
        if (res.ok) {
          showToast("Áudio reproduzido no dispositivo selecionado!", "success");
        } else {
          showToast("TTS desativado ou sem áudio reproduzido.", "info");
        }
      } catch (err) {
        showToast(`Erro no teste de voz: ${err.message}`, "error");
      } finally {
        if (btn) btn.disabled = false;
      }
    });

    el("setting-speak-all")?.addEventListener("change", async () => {
      const checked = Boolean(el("setting-speak-all")?.checked);
      try {
        const res = await apiJson("/api/config/settings", {
          method: "POST",
          body: JSON.stringify({ speak_on_all_interventions: checked }),
        });
        if (res.settings) applySettings(res.settings);
        showToast(
          checked ? "Voz ativada em todas as detecções." : "Voz apenas nos níveis 2 e 3.",
          "info"
        );
      } catch (err) {
        showToast(err.message, "error");
      }
    });

    el("settings-form")?.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      const payload = {
        coach_persona: el("setting-persona")?.value || "firme",
        max_daily_usd: Number(el("setting-max-daily")?.value || 0.35),
        max_monthly_usd: Number(el("setting-max-monthly")?.value || 8.0),
        youtube_music_grace_sec: Number(el("setting-yt-grace")?.value || 45),
        phone_distraction_sec: Number(el("setting-phone-sec")?.value || 15),
        camera_not_seeing_user_sec: Number(el("setting-camera-unseen-sec")?.value || 20),
        capture_mode: el("setting-capture-mode")?.value || "all_monitors",
        speak_on_all_interventions: Boolean(el("setting-speak-all")?.checked),
      };
      try {
        const res = await apiJson("/api/config/settings", {
          method: "POST",
          body: JSON.stringify(payload),
        });
        if (res.settings) applySettings(res.settings);
        if (res.status) renderStatus(res.status);
        showToast("Configurações salvas!", "success");
      } catch (err) {
        showToast(err.message, "error");
      }
    });

    el("btn-toggle-chat-panel")?.addEventListener("click", () => {
      const card = el("chat-bottom-card");
      const btn = el("btn-toggle-chat-panel");
      if (!card || !btn) return;
      const collapsed = card.classList.toggle("collapsed");
      btn.textContent = collapsed ? "Expandir Chat" : "Minimizar";
    });

    el("note-form")?.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      const inp = el("note-input");
      const note = inp?.value?.trim() || "";
      if (!note) return;
      const subject = el("subject-input")?.value?.trim() || null;
      try {
        const res = await apiJson("/api/notes", {
          method: "POST",
          body: JSON.stringify({ note, subject }),
        });
        if (inp) inp.value = "";
        if (res.note) {
          state.notes.push(res.note);
          renderNotes();
        }
        showToast("Nota registrada!", "success");
      } catch (err) {
        showToast(err.message, "error");
      }
    });
  }

  function startLocalTicker() {
    setInterval(() => {
      const s = state.status?.session;
      if (!s) return;
      if (s.current_state === "FOCUSED") {
        s.focus_seconds = (Number(s.focus_seconds) || 0) + 1;
        const formatted = formatDuration(s.focus_seconds);
        if (el("topbar-timer")) el("topbar-timer").textContent = formatted;
        if (el("kpi-focus-time")) el("kpi-focus-time").textContent = formatted;
        const curSubj = String(s.current_subject || "").trim().toLowerCase();
        if (curSubj && Array.isArray(state.subjectStats) && state.subjectStats.length > 0) {
          const item = state.subjectStats.find(
            (it) => String(it.subject || "").trim().toLowerCase() === curSubj
          );
          if (item) {
            item.total_focus_seconds = (Number(item.total_focus_seconds) || 0) + 1;
            item.today_focus_seconds = (Number(item.today_focus_seconds) || 0) + 1;
            renderSubjectStats();
          }
        }
      } else if (s.current_state === "BREAK_MODE" && s.active_break_remaining_sec > 0) {
        s.active_break_remaining_sec = Math.max(0, Number(s.active_break_remaining_sec) - 1);
        const endBreakBtn = el("btn-end-break");
        if (endBreakBtn) {
          endBreakBtn.textContent = `Encerrar Pausa (${formatShortDuration(s.active_break_remaining_sec)})`;
        }
      }
      state.subjectSyncCounter = (state.subjectSyncCounter || 0) + 1;
      if (state.subjectSyncCounter >= 30) {
        state.subjectSyncCounter = 0;
        refreshSubjectStats();
      }
    }, 1000);
  }

  document.addEventListener("DOMContentLoaded", () => {
    bindEvents();
    refreshStatus();
    loadAudioDevices();
    connectWebSocket();
    startLocalTicker();
  });
})();
