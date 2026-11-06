"""SQLite persistence: credit/usage log + chat history.

Usage log:  usage_log(ts, month, model, kind, prompt_tokens, completion_tokens, cost, latency_ms)
Chat:       chat_sessions + chat_messages
"""
from __future__ import annotations

import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .config import ROOT, load_config

SCHEMA = """
CREATE TABLE IF NOT EXISTS usage_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    month TEXT NOT NULL,
    model TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'chat',
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    cost REAL NOT NULL DEFAULT 0.0,
    latency_ms INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_usage_month ON usage_log(month);
CREATE INDEX IF NOT EXISTS idx_usage_ts ON usage_log(ts);

CREATE TABLE IF NOT EXISTS chat_sessions (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    model TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    model TEXT,
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    cost REAL DEFAULT 0.0,
    latency_ms INTEGER DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON chat_messages(session_id);
"""


def db_path() -> Path:
    env = os.environ.get("JEET_DB_PATH")
    if env:
        path = Path(env)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    database_dir = ROOT / "database"
    database_dir.mkdir(exist_ok=True)
    return database_dir / "usage.db"


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path()), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def current_month() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m")


# ── Usage / credits ──────────────────────────────────────────────

def log_usage(model: str, kind: str, prompt_tokens: int, completion_tokens: int,
              cost: float, latency_ms: int) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO usage_log (ts, month, model, kind, prompt_tokens, completion_tokens, cost, latency_ms)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (now_iso(), current_month(), model, kind, int(prompt_tokens),
             int(completion_tokens), float(cost), int(latency_ms)),
        )


def usage_summary(month: str | None = None) -> dict:
    month = month or current_month()
    cfg = load_config()
    budget = float(cfg.get("credits", {}).get("monthly_budget", 35.0))
    threshold = float(cfg.get("credits", {}).get("warning_threshold", 5.0))

    with get_conn() as conn:
        row = conn.execute(
            "SELECT COALESCE(SUM(cost),0) AS used, COALESCE(SUM(prompt_tokens+completion_tokens),0) AS tokens,"
            "       COUNT(*) AS calls, COALESCE(SUM(prompt_tokens),0) AS ptok, COALESCE(SUM(completion_tokens),0) AS ctok,"
            "       COALESCE(AVG(latency_ms),0) AS avg_latency"
            " FROM usage_log WHERE month = ?", (month,),
        ).fetchone()
        by_model = conn.execute(
            "SELECT model, COUNT(*) AS calls, SUM(prompt_tokens+completion_tokens) AS tokens,"
            "       SUM(cost) AS cost, AVG(latency_ms) AS avg_latency"
            " FROM usage_log WHERE month = ? GROUP BY model ORDER BY cost DESC", (month,),
        ).fetchall()
        daily = conn.execute(
            "SELECT substr(ts,1,10) AS day, SUM(cost) AS cost, COUNT(*) AS calls"
            " FROM usage_log WHERE month = ? GROUP BY day ORDER BY day", (month,),
        ).fetchall()
        recent = conn.execute(
            "SELECT ts, model, kind, prompt_tokens, completion_tokens, cost, latency_ms"
            " FROM usage_log ORDER BY id DESC LIMIT 15"
        ).fetchall()

    used = float(row["used"])
    remaining = max(0.0, budget - used)
    return {
        "month": month,
        "budget": budget,
        "warning_threshold": threshold,
        "used": round(used, 4),
        "remaining": round(remaining, 4),
        "pct_used": round((used / budget * 100) if budget else 0.0, 2),
        "low_credits": remaining <= threshold,
        "tokens": int(row["tokens"]),
        "prompt_tokens": int(row["ptok"]),
        "completion_tokens": int(row["ctok"]),
        "calls": int(row["calls"]),
        "avg_latency_ms": round(float(row["avg_latency"]), 1),
        "by_model": [dict(r) for r in by_model],
        "daily": [dict(r) for r in daily],
        "recent": [dict(r) for r in recent],
    }


def model_stats() -> dict[str, dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT model, COUNT(*) AS calls, SUM(prompt_tokens+completion_tokens) AS tokens,"
            "       SUM(cost) AS cost, AVG(latency_ms) AS avg_latency"
            " FROM usage_log GROUP BY model"
        ).fetchall()
    return {r["model"]: {k: (round(v, 4) if isinstance(v, float) else v)
                         for k, v in dict(r).items() if k != "model"} for r in rows}


# ── Chat sessions ────────────────────────────────────────────────

def create_session(title: str, model: str) -> str:
    session_id = uuid.uuid4().hex[:16]
    ts = now_iso()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO chat_sessions (id, title, model, created_at, updated_at) VALUES (?,?,?,?,?)",
            (session_id, title[:80] or "New chat", model, ts, ts),
        )
    return session_id


def list_sessions(limit: int = 60) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT s.id, s.title, s.model, s.created_at, s.updated_at,"
            "       (SELECT COUNT(*) FROM chat_messages m WHERE m.session_id = s.id) AS message_count"
            " FROM chat_sessions s ORDER BY s.updated_at DESC LIMIT ?", (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_session(session_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM chat_sessions WHERE id = ?", (session_id,)).fetchone()
    return dict(row) if row else None


def delete_session(session_id: str) -> bool:
    with get_conn() as conn:
        cur = conn.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
    return cur.rowcount > 0


def add_message(session_id: str, role: str, content: str, model: str | None = None,
                prompt_tokens: int = 0, completion_tokens: int = 0,
                cost: float = 0.0, latency_ms: int = 0) -> int:
    ts = now_iso()
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO chat_messages (session_id, role, content, model, prompt_tokens,"
            " completion_tokens, cost, latency_ms, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (session_id, role, content, model, int(prompt_tokens), int(completion_tokens),
             float(cost), int(latency_ms), ts),
        )
        conn.execute("UPDATE chat_sessions SET updated_at = ?, model = COALESCE(?, model) WHERE id = ?",
                     (ts, model, session_id))
        return int(cur.lastrowid)


def list_messages(session_id: str, limit: int = 200) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, role, content, model, prompt_tokens, completion_tokens, cost, latency_ms, created_at"
            " FROM chat_messages WHERE session_id = ? ORDER BY id ASC LIMIT ?", (session_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]
