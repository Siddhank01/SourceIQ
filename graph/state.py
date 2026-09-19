from typing import Annotated, Any, Dict, List, Optional, TypedDict


class AgenticSelfRAGState(TypedDict):
    question: str
    conversation: List[str]
    documents: List[Dict[str, Any]]
    query: str
    retrieval_status: str
    relevance_score: float
    relevance_grades: List[Dict[str, Any]]
    rewrite_count: int
    retry_count: int
    max_retries: int
    answer: str
    grounding_score: float
    hallucination_status: str
    answer_relevance_score: float
    answer_relevance_status: str
    sources: List[Dict[str, Any]]
    final_answer: str
    confidence: float
    verification_status: str
    error: Optional[str]
    status: str


class GraphState:
    """State object placeholder for graph orchestration."""
    pass
