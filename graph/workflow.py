from typing import Any, Dict

from langgraph.graph import END, StateGraph

from graph.nodes import SelfRAGNodes
from graph.state import AgenticSelfRAGState


class SelfRAGWorkflow:
    """A real workflow that routes based on dynamic reflection signals."""

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self.workflow = self.build_workflow()

    def build_workflow(self):
        builder = StateGraph(AgenticSelfRAGState)
        builder.add_node("understand_question", SelfRAGNodes.understand_question)
        builder.add_node("decide_agentic_action", SelfRAGNodes.decide_agentic_action)
        builder.add_node("retrieve_documents", SelfRAGNodes.retrieve_documents)
        builder.add_node("grade_documents", SelfRAGNodes.grade_documents)
        builder.add_node("rewrite_query", SelfRAGNodes.rewrite_query)
        builder.add_node("generate_answer", SelfRAGNodes.generate_answer)
        builder.add_node("verify_grounding", SelfRAGNodes.verify_grounding)
        builder.add_node("answer_relevance_grade", SelfRAGNodes.answer_relevance_grade)
        builder.add_node("final_answer", SelfRAGNodes.final_answer)

        builder.set_entry_point("understand_question")
        builder.add_edge("understand_question", "decide_agentic_action")

        def route_after_decision(state: Dict[str, Any]):
            if state.get("status") == "Abstain":
                return "final_answer"
            return "retrieve_documents" if state.get("needs_retrieval", True) else "final_answer"

        builder.add_conditional_edges(
            "decide_agentic_action",
            route_after_decision,
            {
                "retrieve_documents": "retrieve_documents",
                "final_answer": "final_answer",
            },
        )

        builder.add_edge("retrieve_documents", "grade_documents")

        def route_after_grade(state: Dict[str, Any]):
            if state.get("status") == "Abstain" or not state.get("documents"):
                return "final_answer"
            if not state.get("relevant_passage_ids") or state.get("relevance_grades", [{}])[0].get("needs_query_rewrite", False):
                return "rewrite_query"
            return "generate_answer"

        builder.add_conditional_edges(
            "grade_documents",
            route_after_grade,
            {
                "rewrite_query": "rewrite_query",
                "generate_answer": "generate_answer",
                "final_answer": "final_answer",
            },
        )

        def route_after_rewrite(state: Dict[str, Any]):
            if state.get("status") == "Abstain":
                return "final_answer"
            if int(state.get("retrieval_attempts", 0)) >= int(state.get("max_retrieval_attempts", self.max_retries)):
                state["abstain_reason"] = "Maximum retrieval attempts reached without sufficient evidence."
                return "final_answer"
            return "retrieve_documents"

        builder.add_conditional_edges(
            "rewrite_query",
            route_after_rewrite,
            {
                "retrieve_documents": "retrieve_documents",
                "final_answer": "final_answer",
            },
        )

        builder.add_edge("generate_answer", "verify_grounding")

        def route_after_verification(state: Dict[str, Any]):
            if state.get("status") == "Abstain":
                return "final_answer"
            if state.get("verification_status") != "verified":
                if int(state.get("generation_attempts", 0)) < int(state.get("max_generation_attempts", self.max_retries)):
                    return "generate_answer"
                if int(state.get("retrieval_attempts", 0)) < int(state.get("max_retrieval_attempts", self.max_retries)):
                    return "rewrite_query"
                state["abstain_reason"] = "Claim verification failed after generation and retrieval limits."
                return "final_answer"
            return "answer_relevance_grade"

        builder.add_conditional_edges(
            "verify_grounding",
            route_after_verification,
            {
                "rewrite_query": "rewrite_query",
                "answer_relevance_grade": "answer_relevance_grade",
                "final_answer": "final_answer",
            },
        )

        def route_after_answer_relevance(state: Dict[str, Any]):
            if state.get("status") == "Abstain":
                return "final_answer"
            answer_relevance = state.get("answer_relevance", {})
            if not answer_relevance.get("relevant", False) or answer_relevance.get("needs_retrieval_retry", False):
                if int(state.get("generation_attempts", 0)) < int(state.get("max_generation_attempts", self.max_retries)):
                    return "generate_answer"
                if int(state.get("retrieval_attempts", 0)) < int(state.get("max_retrieval_attempts", self.max_retries)):
                    return "rewrite_query"
                state["abstain_reason"] = "Answer relevance failed after generation and retrieval limits."
                return "final_answer"
            return "final_answer"

        builder.add_conditional_edges(
            "answer_relevance_grade",
            route_after_answer_relevance,
            {
                "rewrite_query": "rewrite_query",
                "final_answer": "final_answer",
            },
        )

        builder.add_edge("final_answer", END)
        return builder.compile()

    def run(self, question: str, **options: Any) -> Dict[str, Any]:
        initial_state = {
            "question": question,
            "conversation": [],
            "documents": [],
            "retrieved_documents": [],
            "query": question,
            "query_history": [question],
            "retrieval_status": "",
            "relevance_score": 0.0,
            "relevance_grades": [],
            "passage_relevance_results": [],
            "relevant_passage_ids": [],
            "retrieval_attempts": 0,
            "generation_attempts": 0,
            "max_retrieval_attempts": int(options.get("max_retrieval_attempts", self.max_retries)),
            "max_generation_attempts": int(options.get("max_generation_attempts", self.max_retries)),
            "rewrite_count": 0,
            "retry_count": 0,
            "max_retries": self.max_retries,
            "answer": "",
            "grounding_score": 0.0,
            "hallucination_status": "unverified",
            "answer_relevance_score": 0.0,
            "answer_relevance_status": "ungraded",
            "sources": [],
            "final_answer": "",
            "confidence": 0.0,
            "verification_status": "pending",
            "error": None,
            "status": "Starting",
            "need_retrieval": True,
            "needs_retrieval": True,
            "retrieval_decision": {"action": "Retrieve", "confidence": 0.0, "reasoning": "Pending", "needs_retrieval": True},
            "grounding_verification": {},
            "answer_support_verification": {},
            "answer_relevance": {},
            "reflection_trace": [],
            "reflection_failure": None,
            "prompt_injection_detected": False,
            "abstain_reason": None,
            "rewrite_reason": None,
            "unsupported_claims": [],
        }
        initial_state.update(options)
        return self.workflow.invoke(initial_state)
