"""Recorder — SQLite persistence for sessions, turns and reports.

Serves as the audit trail for the whole interview: every message is stored
with a timestamp, and every interviewer message carries the difficulty tier
that was active when it was asked plus the judged quality of the previous
answer. Reports are stored on the session row so a finished interview can be
re-reviewed at any time.
"""
from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .models import STATUS_FINISHED, STATUS_INTERVIEWING, Message, jdump


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id            TEXT PRIMARY KEY,
    title         TEXT,
    role_title    TEXT,
    company       TEXT,
    seniority     TEXT,
    status        TEXT NOT NULL,
    difficulty    INTEGER NOT NULL,
    profile_json  TEXT NOT NULL,
    jd_json       TEXT NOT NULL,
    facts_json    TEXT NOT NULL DEFAULT '[]',
    meta_json     TEXT NOT NULL DEFAULT '{}',
    resume        TEXT,
    report_md     TEXT,
    created_at    TEXT NOT NULL,
    ended_at      TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(id),
    speaker     TEXT NOT NULL,
    content     TEXT NOT NULL,
    difficulty  INTEGER,
    quality     INTEGER,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);

CREATE TABLE IF NOT EXISTS interview_styles (
    company      TEXT NOT NULL,
    role_title   TEXT NOT NULL,
    style_json   TEXT NOT NULL,
    sources_json TEXT NOT NULL DEFAULT '[]',
    hits         INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    PRIMARY KEY (company, role_title)
);
"""


class Recorder:
    """Thread-safe SQLite store. Uses a single shared connection guarded by a lock."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._migrate()
            self._conn.commit()

    def _migrate(self) -> None:
        """Add columns introduced after the original schema (keeps old DBs working)."""
        cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(sessions)").fetchall()}
        if "title" not in cols:
            self._conn.execute("ALTER TABLE sessions ADD COLUMN title TEXT")
        if "resume" not in cols:
            self._conn.execute("ALTER TABLE sessions ADD COLUMN resume TEXT")

    # ------------------------------------------------------------- sessions
    def create_session(
        self,
        session_id: str,
        profile: dict[str, Any],
        jd: dict[str, Any],
        difficulty: int,
        facts: Optional[list[str]] = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO sessions
                    (id, role_title, company, seniority, status, difficulty,
                     profile_json, jd_json, facts_json, meta_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    profile.get("role_title") or "",
                    profile.get("company"),
                    jd.get("seniority") or "mid",
                    STATUS_INTERVIEWING,
                    difficulty,
                    jdump(profile),
                    jdump(jd),
                    jdump(facts or []),
                    jdump({}),
                    _now(),
                ),
            )
            self._conn.commit()

    def get_session(self, session_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
        return self._decode_session(row) if row else None

    def list_sessions(self, limit: int = 20, q: str = "") -> list[dict[str, Any]]:
        """List sessions, newest first. ``q`` filters by title/role/company."""
        sql = "SELECT * FROM sessions"
        params: list[Any] = []
        if (q or "").strip():
            like = f"%{q.strip()}%"
            sql += " WHERE title LIKE ? OR role_title LIKE ? OR company LIKE ?"
            params += [like, like, like]
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        out = [self._decode_session(r) for r in rows]
        for s in out:  # 每条会话带消息数，供侧栏展示
            s["message_count"] = self.count_messages(s["id"])
        return out

    def rename_session(self, session_id: str, title: str) -> bool:
        """Set a custom display title. Returns False if the session is missing."""
        title = (title or "").strip()
        with self._lock:
            cur = self._conn.execute(
                "UPDATE sessions SET title = ? WHERE id = ?", (title or None, session_id)
            )
            self._conn.commit()
        return cur.rowcount > 0

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all of its messages. Returns False if missing."""
        with self._lock:
            self._conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            cur = self._conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            self._conn.commit()
        return cur.rowcount > 0

    def update_session_turn(
        self,
        session_id: str,
        difficulty: int,
        facts: list[str],
        meta: dict[str, Any],
    ) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET difficulty = ?, facts_json = ?, meta_json = ? WHERE id = ?",
                (difficulty, jdump(facts), jdump(meta), session_id),
            )
            self._conn.commit()

    def save_resume(self, session_id: str, resume: str) -> bool:
        """Store the candidate resume for the session. Returns False if missing."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE sessions SET resume = ? WHERE id = ?",
                ((resume or "").strip() or None, session_id),
            )
            self._conn.commit()
        return cur.rowcount > 0

    def finish_session(self, session_id: str, report_md: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET status = ?, report_md = ?, ended_at = ? WHERE id = ?",
                (STATUS_FINISHED, report_md, _now(), session_id),
            )
            self._conn.commit()

    # ------------------------------------------------------------- messages
    def save_message(self, msg: Message) -> int:
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO messages
                    (session_id, speaker, content, difficulty, quality, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    msg.session_id,
                    msg.speaker,
                    msg.content,
                    msg.difficulty,
                    msg.quality,
                    msg.created_at or _now(),
                ),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def list_messages(self, session_id: str) -> list[Message]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
        out = []
        for r in rows:
            out.append(
                Message(
                    id=int(r["id"]),
                    session_id=r["session_id"],
                    speaker=r["speaker"],
                    content=r["content"],
                    difficulty=r["difficulty"],
                    quality=r["quality"],
                    created_at=r["created_at"],
                )
            )
        return out

    # ----------------------------------------------------- interview styles
    def get_interview_style(self, company: str, role_title: str) -> Optional[dict[str, Any]]:
        """Read a cached (company, role) style summary; bumps its hit counter."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM interview_styles WHERE company = ? AND role_title = ?",
                (company, role_title),
            ).fetchone()
            if row is not None:
                self._conn.execute(
                    "UPDATE interview_styles SET hits = hits + 1, updated_at = ? "
                    "WHERE company = ? AND role_title = ?",
                    (_now(), company, role_title),
                )
                self._conn.commit()
        if row is None:
            return None
        import json

        return {
            "style": json.loads(row["style_json"]),
            "sources": json.loads(row["sources_json"]),
            "hits": int(row["hits"]) + 1,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def save_interview_style(
        self, company: str, role_title: str, style: dict[str, Any], sources: list[str]
    ) -> None:
        """Cache a (company, role) style summary (upsert)."""
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO interview_styles (company, role_title, style_json, sources_json, hits, created_at, updated_at)
                VALUES (?, ?, ?, ?, 0, ?, ?)
                ON CONFLICT(company, role_title) DO UPDATE SET
                    style_json = excluded.style_json,
                    sources_json = excluded.sources_json,
                    updated_at = excluded.updated_at
                """,
                (company, role_title, jdump(style), jdump(list(sources)), _now(), _now()),
            )
            self._conn.commit()

    # ---------------------------------------------------------------- misc
    def count_messages(self, session_id: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS c FROM messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return int(row["c"])

    @staticmethod
    def _decode_session(row: sqlite3.Row) -> dict[str, Any]:
        import json

        return {
            "id": row["id"],
            "title": row["title"],
            "role_title": row["role_title"],
            "company": row["company"],
            "seniority": row["seniority"],
            "status": row["status"],
            "difficulty": row["difficulty"],
            "profile": json.loads(row["profile_json"]),
            "jd": json.loads(row["jd_json"]),
            "facts": json.loads(row["facts_json"]),
            "meta": json.loads(row["meta_json"]),
            "resume": row["resume"],
            "report_md": row["report_md"],
            "created_at": row["created_at"],
            "ended_at": row["ended_at"],
        }
