from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class RetrievalDecision(BaseModel):
    """Dynamic retrieval reflection. This is a structured Self-RAG-inspired protocol."""

    action: Literal["Retrieve", "NoRetrieve", "Abstain"] = Field(description="Whether retrieval is needed.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the decision.")
    reasoning: str = Field(description="Why the decision was made.")
    reflection_signal: str = Field(default="[Retrieve]", description="Explicit reflection signal such as [Retrieve] or [NoRetrieve].")
    needs_retrieval: bool = Field(default=True, description="Canonical boolean for routing.")


class PassageRelevance(BaseModel):
    """Passage-level relevance result."""

    passage_id: str = Field(description="Stable identifier for the chunk.")
    relevant: bool = Field(description="Whether the passage supports the query.")
    score: float = Field(..., ge=0.0, le=1.0, description="Passage relevance score.")
    reasoning: str = Field(description="Why passage was judged relevant or irrelevant.")
    supporting_evidence: List[str] = Field(default_factory=list, description="Short evidence snippets.")
    reflection_signal: str = Field(default="[Relevant]", description="Explicit signal for relevance.")


class PassageRelevanceSet(BaseModel):
    passages: List[PassageRelevance] = Field(default_factory=list)


class QueryRewrite(BaseModel):
    """Structured query rewrite output."""

    rewritten_query: str = Field(description="The improved query to use for the next retrieval attempt.")
    reason: str = Field(description="Why the query was rewritten.")
    attempt_number: int = Field(default=1, ge=1)
    previous_queries: List[str] = Field(default_factory=list)


class DocumentRelevance(BaseModel):
    """Structured relevance grade for a retrieved document set."""

    relevant: bool = Field(description="Whether the retrieved documents support the user query.")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score from 0 to 1.")
    reasoning: str = Field(description="High-level explanation for relevance grading.")
    needs_query_rewrite: bool = Field(description="Whether the query should be rewritten.")


class GroundingVerification(BaseModel):
    """Structured hallucination/grounding grade."""

    grounded: bool = Field(description="Whether the answer is grounded by retrieved source documents.")
    score: float = Field(..., ge=0.0, le=1.0, description="Groundedness score from 0 to 1.")
    unsupported_claims: List[str] = Field(default_factory=list)
    reasoning: str = Field(description="High-level explanation for groundedness.")


class AnswerRelevance(BaseModel):
    """Structured answer relevance grade."""

    relevant: bool = Field(description="Whether the answer directly addresses the user question.")
    score: float = Field(..., ge=0.0, le=1.0, description="Answer relevance score from 0 to 1.")
    reasoning: str = Field(description="High-level explanation for answer utility.")
    needs_retrieval_retry: bool = Field(description="Whether retrieval and generation should retry.")


class ClaimSupport(BaseModel):
    claim: str = Field(description="Material claim extracted from the answer.")
    supported: bool = Field(description="Whether this claim is supported by the evidence.")
    source_ids: List[str] = Field(default_factory=list, description="Valid source IDs supporting the claim.")
    reason: str = Field(description="Reasoning for claim support or rejection.")

    @field_validator("source_ids")
    @classmethod
    def validate_source_ids(cls, source_ids: List[str]) -> List[str]:
        if source_ids is None:
            return []
        cleaned = [value for value in source_ids if str(value).strip()]
        return cleaned

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


class AnswerSupportVerification(BaseModel):
    """Claim-by-claim support check for grounded generation."""

    supported: bool = Field(description="Whether the generated answer is supported by retrieved evidence.")
    claims: List[ClaimSupport] = Field(default_factory=list, description="Each claim with support data and source IDs.")
    unsupported_claims: List[str] = Field(default_factory=list, description="Unsupported claims extracted from the answer.")
    score: float = Field(..., ge=0.0, le=1.0, description="Verification score from 0 to 1.")

    @field_validator("claims")
    @classmethod
    def validate_claim_support(cls, claims: List[ClaimSupport]) -> List[ClaimSupport]:
        for claim in claims:
            if claim.supported and not claim.source_ids:
                raise ValueError(f"Supported claim '{claim.claim}' must include at least one source ID.")
        return claims


class AnswerUsefulness(BaseModel):
    useful: bool = Field(description="Whether the answer is directly useful and valid.")
    score: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(description="Why the answer is useful or not useful.")
    needs_retry: bool = Field(description="Whether the workflow should retry.")
