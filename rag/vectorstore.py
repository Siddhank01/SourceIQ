import re
from typing import Any, List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.embeddings import get_embeddings
from utils.config import get_settings


class VectorStoreManager:
    """Create and query a Chroma vector store using HuggingFace embeddings."""

    def __init__(self, persist_directory: Optional[str] = None):
        self.settings = get_settings()
        self.persist_directory = persist_directory or self.settings.get("CHROMA_PERSIST_DIR", "chroma_db")
        self.embeddings = get_embeddings(self.settings.get("EMBEDDINGS_MODEL", "all-MiniLM-L6-v2"))

    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
        chunks = splitter.split_documents(documents)
        for index, chunk in enumerate(chunks):
            metadata = dict(chunk.metadata or {})
            parent_id = metadata.get("source_id") or metadata.get("source") or "source"
            metadata["parent_source_id"] = str(parent_id)
            metadata["chunk_id"] = f"{parent_id}::chunk-{index}"
            metadata["source_id"] = metadata["chunk_id"]
            chunk.metadata = metadata
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
        )
        return vectorstore

    def load_vectorstore(self) -> Chroma:
        return Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
        )

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        vectorstore = self.load_vectorstore()
        exact_matches: List[Document] = []
        seen: set[str] = set()
        tokens = re.findall(r"\b\d{5,}\b", query)
        for token in tokens:
            result: dict[str, Any] = vectorstore.get(
                where_document={"$contains": token},
                include=["documents", "metadatas"],
            )
            for content, metadata in zip(result.get("documents", []), result.get("metadatas", [])):
                if content not in seen:
                    metadata = dict(metadata or {})
                    exact_matches.append(Document(page_content=content, metadata=metadata))
                    seen.add(content)
        semantic_matches = vectorstore.similarity_search_with_relevance_scores(query, k=k)
        for document, score in semantic_matches:
            if document.page_content not in seen:
                metadata = dict(document.metadata or {})
                metadata["score"] = float(score)
                metadata.setdefault("source_id", metadata.get("source_id") or metadata.get("source") or "source")
                exact_matches.append(Document(page_content=document.page_content, metadata=metadata))
                seen.add(document.page_content)
        return exact_matches[:k]

    def similarity_search_with_scores(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        vectorstore = self.load_vectorstore()
        matches = vectorstore.similarity_search_with_relevance_scores(query, k=k)
        scored_docs: List[Tuple[Document, float]] = []
        seen: set[str] = set()
        for document, score in matches:
            if document.page_content in seen:
                continue
            metadata = dict(document.metadata or {})
            metadata["score"] = float(score)
            metadata.setdefault("source_id", metadata.get("source_id") or metadata.get("source") or "source")
            scored_docs.append((Document(page_content=document.page_content, metadata=metadata), float(score)))
            seen.add(document.page_content)
        return scored_docs
