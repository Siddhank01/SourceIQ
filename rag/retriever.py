from typing import Any, Dict, List
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.tools import tool


class Retriever:
    """Document retrieval abstraction for the Self-RAG workflow."""

    def __init__(self, vectorstore: VectorStore):
        self.vectorstore = vectorstore

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        return self.vectorstore.similarity_search(query=query, k=k)
