import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from graph.workflow import SelfRAGWorkflow
from models.schemas import (
    AnswerRelevance,
    AnswerSupportVerification,
    DocumentRelevance,
    GroundingVerification,
    PassageRelevance,
    RetrievalDecision,
)
from models.graders import GraderRunner
from models.graders import ReflectionUnavailable
from utils.helpers import bounded_retry_count
from evaluation.evaluate import EvaluationMetrics


def test_graph_construction():
    wf = SelfRAGWorkflow(max_retries=3)
    assert wf.workflow is not None


def test_graded_schema_models():
    doc_grade = DocumentRelevance(relevant=True, score=0.8, reasoning="Relevant", needs_query_rewrite=False)
    ground_grade = GroundingVerification(grounded=True, score=0.8, unsupported_claims=[], reasoning="Grounded")
    ans_grade = AnswerRelevance(relevant=True, score=0.8, reasoning="Relevant", needs_retrieval_retry=False)
    assert doc_grade.score >= 0
    assert ground_grade.score >= 0
    assert ans_grade.score >= 0


def test_bounded_retry_logic():
    assert bounded_retry_count(0, 3) is True
    assert bounded_retry_count(3, 3) is False


def test_evaluation_sample():
    metrics = EvaluationMetrics()
    result = metrics.run_sample("What is retrieval?", "Retrieval finds source evidence.", ["doc"], ["source"], 1)
    assert "retrieval" in result


def test_workflow_run_returns_state():
    wf = SelfRAGWorkflow(max_retries=3)
    result = wf.run("Explain the knowledge base.")
    assert "status" in result


def test_reflection_protocol_models_are_defined():
    decision = RetrievalDecision(action="Retrieve", confidence=0.9, reasoning="Need evidence from documents.")
    passage = PassageRelevance(passage_id="chunk-1", relevant=True, score=0.9, reasoning="Matches the question.", supporting_evidence=["the policy mentions this"])
    verification = AnswerSupportVerification(
        supported=True,
        claims=[{"claim": "The policy applies to all teams.", "supported": True, "source_ids": ["chunk-1"], "reason": "It is explicitly described in the retrieved evidence."}],
        unsupported_claims=[],
        score=0.9,
    )
    assert decision.action in {"Retrieve", "NoRetrieve"}
    assert passage.passage_id == "chunk-1"
    assert verification.supported is True
    assert verification.claims[0]["source_ids"] == ["chunk-1"]


def test_grader_runner_fails_explicitly_without_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    runner = GraderRunner(model_name="openai/gpt-oss-20b")
    runner.llm = None
    with pytest.raises(ReflectionUnavailable):
        runner.decide_retrieval("Summarize the research policy in the uploaded files.")


def test_duplicate_rewrite_abstains():
    state = {"query": "same query", "query_history": ["same query"], "documents": [], "status": "Rewriting", "reflection_trace": []}
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("graph.nodes.SelfRAGNodes._runner", lambda state: type("Runner", (), {"rewrite_query": lambda self, question, evidence: type("Rewrite", (), {"rewritten_query": "same query", "reason": "duplicate"})()})())
    from graph.nodes import SelfRAGNodes
    result = SelfRAGNodes.rewrite_query(state)
    assert result["status"] == "Abstain"
    assert "duplicate" in result["abstain_reason"]
    monkeypatch.undo()


def test_invalid_claim_citation_is_rejected():
    from graph.nodes import SelfRAGNodes
    state = {
        "question": "What?",
        "answer": "Unsupported [missing-chunk]",
        "documents": [type("Doc", (), {"metadata": {"source_id": "chunk-a"}})()],
        "reflection_trace": [],
        "status": "Verifying grounding",
    }
    verification = type("Verification", (), {
        "claims": [type("Claim", (), {"source_ids": ["missing-chunk"], "supported": True, "claim": "Unsupported"})()],
        "unsupported_claims": [],
        "supported": True,
        "score": 1.0,
        "model_dump": lambda self: {"supported": True, "score": 1.0},
    })()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("graph.nodes.SelfRAGNodes._runner", lambda state: type("Runner", (), {"verify_support": lambda self, question, answer, docs: verification})())
    result = SelfRAGNodes.verify_grounding(state)
    assert result["verification_status"] == "needs_regeneration"
    assert result["grounding_score"] == 0.0
    monkeypatch.undo()
