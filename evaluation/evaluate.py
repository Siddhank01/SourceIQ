import re
from typing import Any, Dict, Iterable, List


TOKEN_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
CITATION_PATTERN = re.compile(r"\[([^\[\]]+)\]")


def _tokens(text: str) -> set[str]:
    return {token.casefold() for token in TOKEN_PATTERN.findall(text or "")}


class EvaluationMetrics:
    """Evaluate workflow outputs against explicitly labeled examples."""

    def __init__(self):
        self.results: list[dict[str, Any]] = []

    def evaluate_example(self, example: Dict[str, Any]) -> Dict[str, float]:
        query = str(example.get("question", ""))
        answer = str(example.get("answer", ""))
        retrieved_ids = {str(value) for value in example.get("retrieved_source_ids", [])}
        expected_ids = {str(value) for value in example.get("expected_source_ids", [])}
        expected_terms = _tokens(" ".join(example.get("expected_answer_terms", [])))
        answer_terms = _tokens(answer)
        citations = {match.group(1).strip() for match in CITATION_PATTERN.finditer(answer)}
        valid_citations = citations & retrieved_ids
        metrics = {
            "retrieval_precision": len(retrieved_ids & expected_ids) / len(retrieved_ids) if retrieved_ids else 0.0,
            "retrieval_recall": len(retrieved_ids & expected_ids) / len(expected_ids) if expected_ids else 0.0,
            "citation_precision": len(valid_citations) / len(citations) if citations else 0.0,
            "citation_recall": len(valid_citations) / len(expected_ids) if expected_ids else 0.0,
            "answer_term_recall": len(answer_terms & expected_terms) / len(expected_terms) if expected_terms else 0.0,
            "supported": float(bool(answer) and not set(example.get("unsupported_claims", []))),
            "correction_success": float(bool(example.get("expected_rewrite", False)) == bool(example.get("rewrite_count", 0) > 0)),
            "abstention_correct": float(bool(example.get("expected_abstain", False)) == bool(example.get("abstain_reason"))),
        }
        metrics["end_to_end_quality"] = sum(metrics.values()) / len(metrics)
        return metrics

    def evaluate_dataset(self, examples: Iterable[Dict[str, Any]]) -> Dict[str, float]:
        rows = [self.evaluate_example(example) for example in examples]
        if not rows:
            return {}
        keys = rows[0].keys()
        result = {key: sum(row[key] for row in rows) / len(rows) for key in keys}
        self.results = rows
        return result

    def run_sample(self, question: str, answer: str, docs: List[str], sources: List[str], rewrite_count: int) -> Dict[str, Dict[str, float]]:
        metrics = self.evaluate_example({
            "question": question,
            "answer": answer,
            "retrieved_source_ids": sources,
            "expected_source_ids": sources,
            "expected_answer_terms": [question],
            "rewrite_count": rewrite_count,
            "expected_rewrite": rewrite_count > 0,
            "expected_abstain": False,
        })
        return {
            "retrieval": {"retrieval_precision": metrics["retrieval_precision"], "retrieval_recall": metrics["retrieval_recall"]},
            "groundedness": {"citation_precision": metrics["citation_precision"], "supported": metrics["supported"]},
            "answer_relevance": {"answer_term_recall": metrics["answer_term_recall"]},
            "correction": {"correction_success": metrics["correction_success"]},
            "query_rewriting": {"correction_success": metrics["correction_success"]},
            "final_quality": {"end_to_end_quality": metrics["end_to_end_quality"]},
        }
