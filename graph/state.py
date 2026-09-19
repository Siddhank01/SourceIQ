from typing import Any, Dict, List, Optional, TypedDict


class AgenticSelfRAGState(TypedDict):
    question: str
    conversation: List[Dict[str, Any]]
    documents: List[Any]
    retrieved_documents: List[Dict[str, Any]]
    query: str
    query_history: List[str]
    retrieval_status: str
    relevance_score: float
    relevance_grades: List[Dict[str, Any]]
    passage_relevance_results: List[Dict[str, Any]]
    relevant_passage_ids: List[str]
    retrieval_attempts: int
    generation_attempts: int
    max_retrieval_attempts: int
    max_generation_attempts: int
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
    needs_retrieval: bool
    retrieval_decision: Dict[str, Any]
    grounding_verification: Dict[str, Any]
    answer_relevance: Dict[str, Any]
    answer_support_verification: Dict[str, Any]
    reflection_trace: List[Dict[str, Any]]
    reflection_failure: Optional[str]
    prompt_injection_detected: bool
    abstain_reason: Optional[str]
    rewrite_reason: Optional[str]


class GraphState:
    """State object placeholder for graph orchestration."""
    pass
