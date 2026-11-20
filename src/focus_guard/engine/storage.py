"""Async SQLite persistence adapter implementing StoragePort."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

from focus_guard.core.events import (
    ActivityVerdict,
    ChatMessage,
    FocusState,
    InterventionCommand,
    InterventionLevel,
    SemanticVerdict,
    SessionStats,
    StudyNote,
    TemporaryAllowance,
)

logger = logging.getLogger(__name__)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    study_goal TEXT NOT NULL,
    current_subject TEXT NOT NULL,
    current_state TEXT NOT NULL,
    focus_seconds REAL NOT NULL,
    distraction_seconds REAL NOT NULL,
    grace_seconds REAL NOT NULL,
    break_seconds REAL NOT NULL,
    focus_ratio_pct REAL NOT NULL,
    interventions_count INTEGER NOT NULL,
    phone_alerts_count INTEGER NOT NULL,
    drowsiness_alerts_count INTEGER NOT NULL,
    escalation_level INTEGER NOT NULL,
    active_break_remaining_sec REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS verdicts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    verdict TEXT NOT NULL,
    confidence REAL NOT NULL,
    category TEXT NOT NULL,
    app_context TEXT NOT NULL,
    reasoning TEXT NOT NULL,
    suggested_message TEXT NOT NULL,
    used_vision INTEGER NOT NULL,
    model_used TEXT NOT NULL,
    tokens_in INTEGER NOT NULL,
    tokens_out INTEGER NOT NULL,
    estimated_cost_usd REAL NOT NULL,
    cached INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_verdicts_session_ts ON verdicts(session_id, timestamp);

CREATE TABLE IF NOT EXISTS interventions (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    level INTEGER NOT NULL,
    title TEXT NOT NULL,
    message TEXT NOT NULL,
    speak_aloud INTEGER NOT NULL,
    reason TEXT NOT NULL,
    actions_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_interventions_session_ts ON interventions(session_id, timestamp);

CREATE TABLE IF NOT EXISTS api_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    utc_date TEXT NOT NULL,
    utc_month TEXT NOT NULL,
    model TEXT NOT NULL,
    tokens_in INTEGER NOT NULL,
    tokens_out INTEGER NOT NULL,
    cost_usd REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_api_usage_date ON api_usage(utc_date);
CREATE INDEX IF NOT EXISTS idx_api_usage_month ON api_usage(utc_month);

CREATE TABLE IF NOT EXISTS chat_messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    role TEXT NOT NULL,
    text TEXT NOT NULL,
    audio_base64 TEXT,
    audio_mime TEXT,
    tool_calls_json TEXT NOT NULL,
    cost_usd REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chat_messages_ts ON chat_messages(timestamp);

CREATE TABLE IF NOT EXISTS study_notes (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    subject TEXT NOT NULL,
    note TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_study_notes_ts ON study_notes(timestamp);

CREATE TABLE IF NOT EXISTS allowances (
    id TEXT PRIMARY KEY,
    pattern TEXT NOT NULL,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_allowances_expires ON allowances(expires_at);

CREATE TABLE IF NOT EXISTS subject_study_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    subject TEXT NOT NULL,
    utc_date TEXT NOT NULL,
    focus_seconds REAL NOT NULL DEFAULT 0.0,
    distraction_seconds REAL NOT NULL DEFAULT 0.0,
    updated_at TEXT NOT NULL,
    UNIQUE(session_id, subject, utc_date)
);
CREATE INDEX IF NOT EXISTS idx_subject_logs_subject ON subject_study_logs(subject);
CREATE INDEX IF NOT EXISTS idx_subject_logs_date ON subject_study_logs(utc_date);
"""


def _default_clock() -> datetime:
    return datetime.now(UTC)


def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _parse_utc(value: str) -> datetime:
    return _to_utc(datetime.fromisoformat(value))


class SQLiteStorage:
    """SQLite persistence layer using aiosqlite in WAL mode."""

    def __init__(
        self,
        db_path: Path,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._db_path = Path(db_path)
        self._clock: Callable[[], datetime] = clock or _default_clock
        self._conn: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    def _now_utc(self) -> datetime:
        return _to_utc(self._clock())

    async def init(self) -> None:
        """Open SQLite connection, enable WAL mode, and ensure schema tables exist."""
        async with self._lock:
            await self._init_locked()

    async def _init_locked(self) -> None:
        if self._conn is not None:
            return
        if str(self._db_path) != ":memory:":
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = await aiosqlite.connect(self._db_path)
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA journal_mode=WAL;")
        await conn.execute("PRAGMA synchronous=NORMAL;")
        await conn.execute("PRAGMA busy_timeout=5000;")
        await conn.executescript(_SCHEMA_SQL)
        await conn.commit()
        self._conn = conn

    async def _ensure_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            await self._init_locked()
        assert self._conn is not None
        return self._conn

    async def close(self) -> None:
        """Close the underlying SQLite connection safely."""
        async with self._lock:
            if self._conn is not None:
                await self._conn.close()
                self._conn = None

    async def save_session(self, stats: SessionStats) -> None:
        """Upsert session telemetry and state snapshot."""
        async with self._lock:
            conn = await self._ensure_conn()
            now_iso = self._now_utc().isoformat()
            await conn.execute(
                """
                INSERT INTO sessions (
                    session_id, started_at, updated_at, study_goal, current_subject,
                    current_state, focus_seconds, distraction_seconds, grace_seconds,
                    break_seconds, focus_ratio_pct, interventions_count,
                    phone_alerts_count, drowsiness_alerts_count, escalation_level,
                    active_break_remaining_sec
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    study_goal = excluded.study_goal,
                    current_subject = excluded.current_subject,
                    current_state = excluded.current_state,
                    focus_seconds = excluded.focus_seconds,
                    distraction_seconds = excluded.distraction_seconds,
                    grace_seconds = excluded.grace_seconds,
                    break_seconds = excluded.break_seconds,
                    focus_ratio_pct = excluded.focus_ratio_pct,
                    interventions_count = excluded.interventions_count,
                    phone_alerts_count = excluded.phone_alerts_count,
                    drowsiness_alerts_count = excluded.drowsiness_alerts_count,
                    escalation_level = excluded.escalation_level,
                    active_break_remaining_sec = excluded.active_break_remaining_sec
                """,
                (
                    stats.session_id,
                    _to_utc(stats.started_at).isoformat(),
                    now_iso,
                    stats.study_goal,
                    stats.current_subject,
                    str(FocusState(stats.current_state)),
                    float(stats.focus_seconds),
                    float(stats.distraction_seconds),
                    float(stats.grace_seconds),
                    float(stats.break_seconds),
                    float(stats.focus_ratio_pct),
                    int(stats.interventions_count),
                    int(stats.phone_alerts_count),
                    int(stats.drowsiness_alerts_count),
                    int(InterventionLevel(stats.escalation_level)),
                    float(stats.active_break_remaining_sec),
                ),
            )
            await conn.commit()

    async def save_verdict(self, session_id: str, verdict: SemanticVerdict) -> None:
        """Persist a semantic evaluation verdict."""
        async with self._lock:
            conn = await self._ensure_conn()
            await conn.execute(
                """
                INSERT INTO verdicts (
                    session_id, timestamp, verdict, confidence, category, app_context,
                    reasoning, suggested_message, used_vision, model_used,
                    tokens_in, tokens_out, estimated_cost_usd, cached
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    _to_utc(verdict.timestamp).isoformat(),
                    str(ActivityVerdict(verdict.verdict)),
                    float(verdict.confidence),
                    verdict.category,
                    verdict.app_context,
                    verdict.reasoning,
                    verdict.suggested_message,
                    int(verdict.used_vision),
                    verdict.model_used,
                    int(verdict.tokens_in),
                    int(verdict.tokens_out),
                    float(verdict.estimated_cost_usd),
                    int(verdict.cached),
                ),
            )
            await conn.commit()

    async def save_intervention(self, session_id: str, cmd: InterventionCommand) -> None:
        """Persist an escalation intervention command."""
        async with self._lock:
            conn = await self._ensure_conn()
            await conn.execute(
                """
                INSERT INTO interventions (
                    id, session_id, timestamp, level, title, message,
                    speak_aloud, reason, actions_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    level = excluded.level,
                    title = excluded.title,
                    message = excluded.message,
                    speak_aloud = excluded.speak_aloud,
                    reason = excluded.reason,
                    actions_json = excluded.actions_json
                """,
                (
                    cmd.id,
                    session_id,
                    _to_utc(cmd.timestamp).isoformat(),
                    int(InterventionLevel(cmd.level)),
                    cmd.title,
                    cmd.message,
                    int(cmd.speak_aloud),
                    cmd.reason,
                    json.dumps(cmd.actions, ensure_ascii=False),
                ),
            )
            await conn.commit()

    async def save_usage(
        self, model: str, tokens_in: int, tokens_out: int, cost_usd: float
    ) -> None:
        """Record API usage stamped with current UTC date and month."""
        async with self._lock:
            conn = await self._ensure_conn()
            now = self._now_utc()
            await conn.execute(
                """
                INSERT INTO api_usage (
                    timestamp, utc_date, utc_month, model, tokens_in, tokens_out, cost_usd
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now.isoformat(),
                    now.strftime("%Y-%m-%d"),
                    now.strftime("%Y-%m"),
                    model,
                    max(0, int(tokens_in)),
                    max(0, int(tokens_out)),
                    max(0.0, float(cost_usd)),
                ),
            )
            await conn.commit()

    async def get_usage_totals(self) -> tuple[float, float, int, int, int]:
        """Return (daily_usd, monthly_usd, daily_tokens_in, daily_tokens_out, daily_calls)."""
        async with self._lock:
            conn = await self._ensure_conn()
            now = self._now_utc()
            today = now.strftime("%Y-%m-%d")
            month = now.strftime("%Y-%m")

            async with conn.execute(
                """
                SELECT
                    COALESCE(SUM(cost_usd), 0.0) AS daily_usd,
                    COALESCE(SUM(tokens_in), 0) AS daily_in,
                    COALESCE(SUM(tokens_out), 0) AS daily_out,
                    COUNT(*) AS daily_calls
                FROM api_usage
                WHERE utc_date = ?
                """,
                (today,),
            ) as cursor:
                daily_row = await cursor.fetchone()

            async with conn.execute(
                """
                SELECT COALESCE(SUM(cost_usd), 0.0) AS monthly_usd
                FROM api_usage
                WHERE utc_month = ?
                """,
                (month,),
            ) as cursor:
                monthly_row = await cursor.fetchone()

            daily_usd = float(daily_row["daily_usd"]) if daily_row else 0.0
            daily_in = int(daily_row["daily_in"]) if daily_row else 0
            daily_out = int(daily_row["daily_out"]) if daily_row else 0
            daily_calls = int(daily_row["daily_calls"]) if daily_row else 0
            monthly_usd = float(monthly_row["monthly_usd"]) if monthly_row else 0.0
            return (daily_usd, monthly_usd, daily_in, daily_out, daily_calls)

    async def save_chat_message(self, session_id: str, msg: ChatMessage) -> None:
        """Persist a coach/user chat message."""
        async with self._lock:
            conn = await self._ensure_conn()
            await conn.execute(
                """
                INSERT INTO chat_messages (
                    id, session_id, timestamp, role, text,
                    audio_base64, audio_mime, tool_calls_json, cost_usd
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    text = excluded.text,
                    audio_base64 = excluded.audio_base64,
                    audio_mime = excluded.audio_mime,
                    tool_calls_json = excluded.tool_calls_json,
                    cost_usd = excluded.cost_usd
                """,
                (
                    msg.id,
                    session_id,
                    _to_utc(msg.timestamp).isoformat(),
                    msg.role,
                    msg.text,
                    msg.audio_base64,
                    msg.audio_mime,
                    json.dumps(msg.tool_calls, ensure_ascii=False),
                    float(msg.cost_usd),
                ),
            )
            await conn.commit()

    async def list_chat_messages(self, limit: int = 50) -> list[ChatMessage]:
        """List recent chat messages in chronological order."""
        if limit <= 0:
            return []
        async with self._lock:
            conn = await self._ensure_conn()
            async with conn.execute(
                """
                SELECT id, timestamp, role, text, audio_base64, audio_mime,
                       tool_calls_json, cost_usd
                FROM chat_messages
                ORDER BY timestamp DESC, rowid DESC
                LIMIT ?
                """,
                (int(limit),),
            ) as cursor:
                rows = list(await cursor.fetchall())

        messages: list[ChatMessage] = []
        for row in reversed(rows):
            tool_calls = json.loads(row["tool_calls_json"]) if row["tool_calls_json"] else []
            messages.append(
                ChatMessage(
                    id=row["id"],
                    timestamp=_parse_utc(row["timestamp"]),
                    role=row["role"],
                    text=row["text"],
                    audio_base64=row["audio_base64"],
                    audio_mime=row["audio_mime"],
                    tool_calls=tool_calls,
                    cost_usd=float(row["cost_usd"]),
                )
            )
        return messages

    async def save_note(self, note: StudyNote) -> None:
        """Persist a study note."""
        async with self._lock:
            conn = await self._ensure_conn()
            await conn.execute(
                """
                INSERT INTO study_notes (id, timestamp, subject, note)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    subject = excluded.subject,
                    note = excluded.note
                """,
                (
                    note.id,
                    _to_utc(note.timestamp).isoformat(),
                    note.subject,
                    note.note,
                ),
            )
            await conn.commit()

    async def list_notes(self, limit: int = 50) -> list[StudyNote]:
        """List recent study notes in chronological order."""
        if limit <= 0:
            return []
        async with self._lock:
            conn = await self._ensure_conn()
            async with conn.execute(
                """
                SELECT id, timestamp, subject, note
                FROM study_notes
                ORDER BY timestamp DESC, rowid DESC
                LIMIT ?
                """,
                (int(limit),),
            ) as cursor:
                rows = list(await cursor.fetchall())

        return [
            StudyNote(
                id=row["id"],
                timestamp=_parse_utc(row["timestamp"]),
                subject=row["subject"],
                note=row["note"],
            )
            for row in reversed(rows)
        ]

    async def save_allowance(self, allowance: TemporaryAllowance) -> None:
        """Persist a temporary allowance grant."""
        async with self._lock:
            conn = await self._ensure_conn()
            await conn.execute(
                """
                INSERT INTO allowances (id, pattern, reason, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    pattern = excluded.pattern,
                    reason = excluded.reason,
                    expires_at = excluded.expires_at
                """,
                (
                    allowance.id,
                    allowance.pattern,
                    allowance.reason,
                    _to_utc(allowance.created_at).isoformat(),
                    _to_utc(allowance.expires_at).isoformat(),
                ),
            )
            await conn.commit()

    async def list_active_allowances(self) -> list[TemporaryAllowance]:
        """Return currently active temporary allowances."""
        async with self._lock:
            conn = await self._ensure_conn()
            now = self._now_utc()
            async with conn.execute(
                """
                SELECT id, pattern, reason, created_at, expires_at
                FROM allowances
                ORDER BY created_at ASC, rowid ASC
                """
            ) as cursor:
                rows = list(await cursor.fetchall())

        active: list[TemporaryAllowance] = []
        for row in rows:
            item = TemporaryAllowance(
                id=row["id"],
                pattern=row["pattern"],
                reason=row["reason"],
                created_at=_parse_utc(row["created_at"]),
                expires_at=_parse_utc(row["expires_at"]),
            )
            if item.is_active(now):
                active.append(item)
        return active

    async def add_subject_time(
        self,
        session_id: str,
        subject: str,
        focus_seconds: float,
        distraction_seconds: float = 0.0,
    ) -> None:
        """Increment persisted focus/distraction time for a subject in the current session."""
        clean_subject = (subject or "").strip() or "Geral"
        focus_dt = max(0.0, float(focus_seconds))
        dist_dt = max(0.0, float(distraction_seconds))
        if focus_dt <= 0.0 and dist_dt <= 0.0:
            return

        async with self._lock:
            conn = await self._ensure_conn()
            now = self._now_utc()
            utc_date = now.strftime("%Y-%m-%d")
            now_iso = now.isoformat()
            await conn.execute(
                """
                INSERT INTO subject_study_logs (
                    session_id, subject, utc_date, focus_seconds, distraction_seconds, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, subject, utc_date) DO UPDATE SET
                    focus_seconds = focus_seconds + excluded.focus_seconds,
                    distraction_seconds = distraction_seconds + excluded.distraction_seconds,
                    updated_at = excluded.updated_at
                """,
                (session_id, clean_subject, utc_date, focus_dt, dist_dt, now_iso),
            )
            await conn.commit()

    async def get_subject_study_summary(self) -> list[dict[str, Any]]:
        """Aggregate total and today's study time per subject across logs and legacy sessions."""
        async with self._lock:
            conn = await self._ensure_conn()
            today = self._now_utc().strftime("%Y-%m-%d")

            async with conn.execute(
                """
                SELECT session_id, subject, utc_date, focus_seconds, distraction_seconds, updated_at
                FROM subject_study_logs
                """
            ) as cursor:
                log_rows = list(await cursor.fetchall())

            async with conn.execute(
                """
                SELECT
                    session_id,
                    current_subject AS subject,
                    substr(started_at, 1, 10) AS utc_date,
                    focus_seconds,
                    distraction_seconds,
                    updated_at
                FROM sessions
                WHERE session_id NOT IN (SELECT DISTINCT session_id FROM subject_study_logs)
                  AND (focus_seconds > 0 OR distraction_seconds > 0)
                """
            ) as cursor:
                legacy_rows = list(await cursor.fetchall())

        buckets: dict[str, dict[str, Any]] = {}
        for row in (*log_rows, *legacy_rows):
            raw_subject = str(row["subject"] or "").strip() or "Geral"
            key = raw_subject.lower()
            focus_sec = max(0.0, float(row["focus_seconds"] or 0.0))
            dist_sec = max(0.0, float(row["distraction_seconds"] or 0.0))
            row_date = str(row["utc_date"] or "")
            updated_at = str(row["updated_at"] or "")
            sess_id = str(row["session_id"] or "")

            entry = buckets.get(key)
            if entry is None:
                entry = {
                    "subject": raw_subject,
                    "total_focus_seconds": 0.0,
                    "today_focus_seconds": 0.0,
                    "total_distraction_seconds": 0.0,
                    "sessions": set(),
                    "last_studied_at": updated_at,
                }
                buckets[key] = entry

            if updated_at >= entry["last_studied_at"]:
                entry["last_studied_at"] = updated_at
                entry["subject"] = raw_subject

            entry["total_focus_seconds"] += focus_sec
            if row_date == today:
                entry["today_focus_seconds"] += focus_sec
            entry["total_distraction_seconds"] += dist_sec
            if sess_id:
                entry["sessions"].add(sess_id)

        results: list[dict[str, Any]] = []
        for entry in buckets.values():
            total_focus = round(float(entry["total_focus_seconds"]), 1)
            today_focus = round(float(entry["today_focus_seconds"]), 1)
            total_dist = round(float(entry["total_distraction_seconds"]), 1)
            denom = total_focus + total_dist
            ratio = round((total_focus / denom) * 100.0, 1) if denom > 0 else 100.0
            results.append(
                {
                    "subject": entry["subject"],
                    "total_focus_seconds": total_focus,
                    "today_focus_seconds": today_focus,
                    "total_distraction_seconds": total_dist,
                    "focus_ratio_pct": ratio,
                    "sessions_count": len(entry["sessions"]),
                    "last_studied_at": entry["last_studied_at"],
                }
            )

        results.sort(
            key=lambda item: (item["total_focus_seconds"], item["today_focus_seconds"], item["subject"].lower()),
            reverse=True,
        )
        return results
