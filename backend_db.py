from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parent / "self_rag.sqlite3"


def connect() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    with connect() as connection:
        connection.executescript(
            """
            PRAGMA foreign_keys = ON;
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                owner TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                saved INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                citations TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                source_type TEXT NOT NULL,
                detail TEXT NOT NULL DEFAULT ''
            );
            """
        )


def session_payload(row: sqlite3.Row, connection: sqlite3.Connection) -> dict[str, Any]:
    session_id = row["id"]
    messages = [dict(message) for message in connection.execute("SELECT role, content AS text, citations, created_at AS time FROM messages WHERE session_id = ? ORDER BY id", (session_id,))]
    for message in messages:
        message["citations"] = json.loads(message["citations"])
    sources = [dict(source) for source in connection.execute("SELECT name, source_type AS type, detail FROM sources WHERE session_id = ? ORDER BY id", (session_id,))]
    return {"id": session_id, "title": row["title"], "description": row["description"], "saved": bool(row["saved"]), "time": row["updated_at"], "messages": messages, "sources": sources}


def upsert_session(session_id: str, owner: str, title: str, description: str, saved: bool, messages: list[dict[str, Any]], sources: list[dict[str, Any]]) -> dict[str, Any]:
    init_db()
    with connect() as connection:
        connection.execute("INSERT INTO sessions (id, owner, title, description, saved) VALUES (?, ?, ?, ?, ?) ON CONFLICT(id) DO UPDATE SET title = excluded.title, description = excluded.description, saved = MAX(sessions.saved, excluded.saved), updated_at = CURRENT_TIMESTAMP", (session_id, owner, title, description, int(saved)))
        connection.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        connection.execute("DELETE FROM sources WHERE session_id = ?", (session_id,))
        connection.executemany("INSERT INTO messages (session_id, role, content, citations) VALUES (?, ?, ?, ?)", [(session_id, message["role"], message["text"], json.dumps(message.get("citations", []))) for message in messages])
        connection.executemany("INSERT INTO sources (session_id, name, source_type, detail) VALUES (?, ?, ?, ?)", [(session_id, source["name"], source.get("type", "file"), source.get("detail", "")) for source in sources])
        row = connection.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return session_payload(row, connection)


def list_sessions(owner: str, saved: bool | None = None) -> list[dict[str, Any]]:
    init_db()
    with connect() as connection:
        query = "SELECT * FROM sessions WHERE owner = ?"
        values: list[Any] = [owner]
        if saved is not None:
            query += " AND saved = ?"
            values.append(int(saved))
        query += " ORDER BY updated_at DESC"
        return [session_payload(row, connection) for row in connection.execute(query, values)]


def get_session(session_id: str, owner: str) -> dict[str, Any] | None:
    init_db()
    with connect() as connection:
        row = connection.execute("SELECT * FROM sessions WHERE id = ? AND owner = ?", (session_id, owner)).fetchone()
        return session_payload(row, connection) if row else None


def delete_session(session_id: str, owner: str) -> bool:
    init_db()
    with connect() as connection:
        cursor = connection.execute("DELETE FROM sessions WHERE id = ? AND owner = ?", (session_id, owner))
        return cursor.rowcount == 1


def delete_session_group(session_id: str, owner: str) -> bool:
    init_db()
    with connect() as connection:
        row = connection.execute("SELECT title FROM sessions WHERE id = ? AND owner = ?", (session_id, owner)).fetchone()
        if not row:
            return False
        cursor = connection.execute("DELETE FROM sessions WHERE owner = ? AND title = ?", (owner, row["title"]))
        return cursor.rowcount > 0
