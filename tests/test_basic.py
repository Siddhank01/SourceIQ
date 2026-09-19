import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from graph.workflow import SelfRAGWorkflow
from models.schemas import DocumentRelevance, GroundingVerification, AnswerRelevance
from models.graders import GraderRunner
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
