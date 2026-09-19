from typing import List, Optional
from pydantic import BaseModel, Field


class DocumentRelevance(BaseModel):
    """Structured relevance grade for a retrieved document set."""

    relevant: bool = Field(description="Whether the retrieved documents support the user query.")
    score: float = Field(description="Relevance score from 0 to 1.")
    reasoning: str = Field(description="High-level explanation for relevance grading.")
    needs_query_rewrite: bool = Field(description="Whether the query should be rewritten.")


class GroundingVerification(BaseModel):
    """Structured hallucination/grounding grade."""

    grounded: bool = Field(description="Whether the answer is grounded by retrieved source documents.")
    score: float = Field(description="Groundedness score from 0 to 1.")
    unsupported_claims: List[str] = Field(default_factory=list)
    reasoning: str = Field(description="High-level explanation for groundedness.")


class AnswerRelevance(BaseModel):
    """Structured answer relevance grade."""

    relevant: bool = Field(description="Whether the answer directly addresses the user question.")
    score: float = Field(description="Answer relevance score from 0 to 1.")
    reasoning: str = Field(description="High-level explanation for answer utility.")
    needs_retrieval_retry: bool = Field(description="Whether retrieval and generation should retry.")
