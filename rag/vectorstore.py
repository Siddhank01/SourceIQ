import re
from typing import Any, List, Optional
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.config import get_settings
from rag.embeddings import get_embeddings


class VectorStoreManager:
    """Create and query a Chroma vector store using HuggingFace embeddings."""

    def __init__(self, persist_directory: Optional[str] = None):
        self.settings = get_settings()
        self.persist_directory = persist_directory or self.settings.get("CHROMA_PERSIST_DIR", "chroma_db")
        self.embeddings = get_embeddings(self.settings.get("EMBEDDINGS_MODEL", "all-MiniLM-L6-v2"))

    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
        chunks = splitter.split_documents(documents)
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
                    exact_matches.append(Document(page_content=content, metadata=metadata or {}))
                    seen.add(content)
        semantic_matches = vectorstore.similarity_search(query, k=k)
        for document in semantic_matches:
            if document.page_content not in seen:
                exact_matches.append(document)
                seen.add(document.page_content)
        return exact_matches[:k]
