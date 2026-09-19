import json
import re
import sqlite3
from pathlib import Path
from typing import Any, List, Optional, Tuple

from langchain_core.documents import Document
from utils.config import get_settings


TOKEN_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _tokens(value: str) -> set[str]:
    return {token.casefold() for token in TOKEN_PATTERN.findall(value or "")}


def _split_documents(documents: List[Document], chunk_size: int = 500, overlap: int = 80) -> List[Document]:
    chunks: List[Document] = []
    for document in documents:
        content = document.page_content or ""
        if not content:
            continue
        start = 0
        index = 0
        while start < len(content):
            end = min(len(content), start + chunk_size)
            metadata = dict(document.metadata or {})
            parent_id = metadata.get("source_id") or metadata.get("source") or "source"
            metadata["parent_source_id"] = str(parent_id)
            metadata["chunk_id"] = f"{parent_id}::chunk-{index}"
            metadata["source_id"] = metadata["chunk_id"]
            chunks.append(Document(page_content=content[start:end], metadata=metadata))
            if end >= len(content):
                break
            start = max(end - overlap, start + 1)
            index += 1
    return chunks


class SQLiteDocumentStore:
    """Small persistent lexical store used when Chroma is unavailable."""

    def __init__(self, persist_directory: str):
        self.path = Path(persist_directory)
        self.path.mkdir(parents=True, exist_ok=True)
        self.database = self.path / "documents.sqlite3"
        with sqlite3.connect(self.database) as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS documents (id TEXT PRIMARY KEY, content TEXT NOT NULL, metadata TEXT NOT NULL)")

    def replace(self, documents: List[Document]) -> None:
        with sqlite3.connect(self.database) as connection:
            connection.execute("DELETE FROM documents")
            connection.executemany(
                "INSERT OR REPLACE INTO documents (id, content, metadata) VALUES (?, ?, ?)",
                [(str(doc.metadata["source_id"]), doc.page_content, json.dumps(doc.metadata)) for doc in documents],
            )

    def similarity_search_with_relevance_scores(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        query_tokens = _tokens(query)
        with sqlite3.connect(self.database) as connection:
            rows = connection.execute("SELECT id, content, metadata FROM documents").fetchall()
        scored = []
        for identifier, content, raw_metadata in rows:
            content_tokens = _tokens(content)
            score = len(query_tokens & content_tokens) / len(query_tokens) if query_tokens else 0.0
            if score > 0:
                scored.append((Document(page_content=content, metadata=json.loads(raw_metadata)), score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        return [document for document, _ in self.similarity_search_with_relevance_scores(query, k)]


class VectorStoreManager:
    """Use Chroma when installed, otherwise use a deployment-safe SQLite lexical store."""

    def __init__(self, persist_directory: Optional[str] = None):
        settings = get_settings()
        self.persist_directory = persist_directory or settings.get("CHROMA_PERSIST_DIR", "chroma_db")
        self._chroma = None
        self._embeddings = None
        try:
            from langchain_chroma import Chroma
            from rag.embeddings import get_embeddings

            self._chroma = Chroma
            self._embeddings = get_embeddings(settings.get("EMBEDDINGS_MODEL", "all-MiniLM-L6-v2"))
        except (ImportError, ModuleNotFoundError):
            self._chroma = None

    def create_vectorstore(self, documents: List[Document]):
        chunks = _split_documents(documents)
        if self._chroma is not None:
            return self._chroma.from_documents(documents=chunks, embedding=self._embeddings, persist_directory=self.persist_directory)
        store = SQLiteDocumentStore(self.persist_directory)
        store.replace(chunks)
        return store

    def load_vectorstore(self):
        if self._chroma is not None:
            return self._chroma(persist_directory=self.persist_directory, embedding_function=self._embeddings)
        return SQLiteDocumentStore(self.persist_directory)

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        return self.load_vectorstore().similarity_search(query, k=k)

    def similarity_search_with_scores(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        return self.load_vectorstore().similarity_search_with_relevance_scores(query, k=k)
