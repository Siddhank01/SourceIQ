from __future__ import annotations

import json
import os
import re
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from langchain_core.documents import Document


class StorageUnavailable(RuntimeError):
    """Raised when hosted persistence is not configured or cannot be reached."""


class SupabaseREST:
    def __init__(self) -> None:
        self.url = os.getenv("SUPABASE_URL", "").rstrip("/")
        self.key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
        if not self.url or not self.key:
            raise StorageUnavailable("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required for hosted persistence.")

    def request(self, method: str, table: str, payload: Any = None, query: str = "") -> Any:
        endpoint = f"{self.url}/rest/v1/{table}{query}"
        headers = {"apikey": self.key, "Authorization": f"Bearer {self.key}", "Content-Type": "application/json", "Prefer": "return=representation,resolution=merge-duplicates"}
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        try:
            with urlopen(Request(endpoint, data=body, headers=headers, method=method), timeout=20) as response:
                raw = response.read()
                return json.loads(raw) if raw else None
        except (HTTPError, URLError, TimeoutError) as exc:
            raise StorageUnavailable(f"Hosted storage request failed: {exc}") from exc


class HostedSessionStore:
    def __init__(self) -> None:
        self.client = SupabaseREST()

    def upsert(self, session_id: str, owner: str, title: str, description: str, saved: bool, messages: list[dict[str, Any]], sources: list[dict[str, Any]]) -> dict[str, Any]:
        rows = self.client.request("POST", "sessions", {
            "id": session_id,
            "owner": owner,
            "title": title,
            "description": description,
            "saved": saved,
            "messages": messages,
            "sources": sources,
        }, "?on_conflict=id")
        return (rows or [{}])[0]

    def list(self, owner: str, saved: bool | None = None) -> list[dict[str, Any]]:
        query = f"?owner=eq.{_quote(owner)}&order=updated_at.desc"
        if saved is not None:
            query += f"&saved=eq.{str(saved).lower()}"
        return self.client.request("GET", "sessions", query=query) or []

    def get(self, session_id: str, owner: str) -> dict[str, Any] | None:
        rows = self.client.request("GET", "sessions", query=f"?id=eq.{_quote(session_id)}&owner=eq.{_quote(owner)}") or []
        return rows[0] if rows else None

    def delete_group(self, session_id: str, owner: str) -> bool:
        current = self.get(session_id, owner)
        if not current:
            return False
        self.client.request("DELETE", "sessions", query=f"?owner=eq.{_quote(owner)}&title=eq.{_quote(current['title'])}")
        return True


class HostedDocumentStore:
    """Supabase-backed chunk store with deterministic lexical retrieval."""

    def __init__(self, session_id: str) -> None:
        self.client = SupabaseREST()
        self.session_id = session_id

    def replace(self, documents: Iterable[Document]) -> None:
        rows = [{"session_id": self.session_id, "chunk_id": str(doc.metadata["source_id"]), "content": doc.page_content, "metadata": doc.metadata} for doc in documents]
        self.client.request("DELETE", "document_chunks", query=f"?session_id=eq.{_quote(self.session_id)}")
        if rows:
            self.client.request("POST", "document_chunks", rows)

    def similarity_search_with_relevance_scores(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        rows = self.client.request("GET", "document_chunks", query=f"?session_id=eq.{_quote(self.session_id)}&select=chunk_id,content,metadata") or []
        query_tokens = _tokens(query)
        scored: list[tuple[Document, float]] = []
        for row in rows:
            content = row.get("content", "")
            content_tokens = _tokens(content)
            score = len(query_tokens & content_tokens) / len(query_tokens) if query_tokens else 0.0
            if score > 0:
                metadata = dict(row.get("metadata") or {})
                metadata["source_id"] = row.get("chunk_id")
                scored.append((Document(page_content=content, metadata=metadata), score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        return [document for document, _ in self.similarity_search_with_relevance_scores(query, k)]


def _tokens(value: str) -> set[str]:
    return {token.casefold() for token in re.findall(r"[a-z0-9]+", value or "", re.IGNORECASE)}


def _quote(value: str) -> str:
    return value.replace("%", "%25").replace(" ", "%20").replace("&", "%26")
