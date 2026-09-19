from typing import List

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore


class Retriever:
    """Document retrieval abstraction for the Self-RAG workflow."""

    def __init__(self, vectorstore: VectorStore):
        self.vectorstore = vectorstore

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        if hasattr(self.vectorstore, "similarity_search_with_relevance_scores"):
            results = self.vectorstore.similarity_search_with_relevance_scores(query, k=k)
            docs: List[Document] = []
            for document, score in results:
                metadata = dict(document.metadata or {})
                metadata.setdefault("score", float(score))
                metadata.setdefault("source_id", metadata.get("source_id") or metadata.get("source") or "source")
                docs.append(Document(page_content=document.page_content, metadata=metadata))
            return docs
        return self.vectorstore.similarity_search(query=query, k=k)
