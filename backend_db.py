from __future__ import annotations

from typing import Any

from storage import HostedSessionStore, StorageUnavailable


# In-memory state is only a test/development fallback. Production requires Supabase.
_memory_sessions: dict[str, dict[str, Any]] = {}


def _store() -> HostedSessionStore | None:
    try:
        return HostedSessionStore()
    except StorageUnavailable:
        return None


def upsert_session(session_id: str, owner: str, title: str, description: str, saved: bool, messages: list[dict[str, Any]], sources: list[dict[str, Any]]) -> dict[str, Any]:
    store = _store()
    if store:
        return store.upsert(session_id, owner, title, description, saved, messages, sources)
    payload = {"id": session_id, "owner": owner, "title": title, "description": description, "saved": saved, "messages": messages, "sources": sources}
    _memory_sessions[session_id] = payload
    return payload


def list_sessions(owner: str, saved: bool | None = None) -> list[dict[str, Any]]:
    store = _store()
    if store:
        return store.list(owner, saved)
    return [item for item in _memory_sessions.values() if item["owner"] == owner and (saved is None or item["saved"] == saved)]


def get_session(session_id: str, owner: str) -> dict[str, Any] | None:
    store = _store()
    if store:
        return store.get(session_id, owner)
    item = _memory_sessions.get(session_id)
    return item if item and item["owner"] == owner else None


def delete_session_group(session_id: str, owner: str) -> bool:
    store = _store()
    if store:
        return store.delete_group(session_id, owner)
    item = get_session(session_id, owner)
    if not item:
        return False
    title = item["title"]
    deleted = [key for key, value in _memory_sessions.items() if value["owner"] == owner and value["title"] == title]
    for key in deleted:
        del _memory_sessions[key]
    return bool(deleted)


def init_db() -> None:
    """Compatibility no-op: production persistence is hosted, never local SQLite."""
    return None
