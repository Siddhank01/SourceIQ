from typing import Any, Dict, List
from langchain.agents import create_retriever_tool
from langchain_core.documents import Document

from rag.retriever import Retriever


def create_retriever_tool_from_retriever(retriever: Retriever, name: str = "document_retriever"):
    """Create a LangChain Retriever Tool consistent with the requested architecture."""
    return create_retriever_tool(
        retriever,
        name=name,
        description="Search the hosted document knowledge base for relevant evidence.",
    )


def create_simple_tool(retriever: Retriever):
    @tool
    def document_retriever(query: str) -> str:
        docs = retriever.retrieve(query, k=4)
        return "\n\n".join(doc.page_content for doc in docs)

    return document_retriever
