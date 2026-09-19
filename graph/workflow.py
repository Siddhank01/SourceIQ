from typing import Any, Dict, List, Optional, Tuple
from langgraph.graph import StateGraph, END

from graph.state import AgenticSelfRAGState
from graph.nodes import SelfRAGNodes


class SelfRAGWorkflow:
    """Build a LangGraph state machine for the requested Self-RAG loop."""

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
        builder.add_edge("decide_agentic_action", "retrieve_documents")
        builder.add_edge("retrieve_documents", "grade_documents")

        def route_from_grading(state: Dict[str, Any]):
            return "rewrite_query" if state.get("relevance_score", 0.0) < 0.7 else "generate_answer"

        builder.add_conditional_edges(
            "grade_documents",
            route_from_grading,
            {
                "rewrite_query": "rewrite_query",
                "generate_answer": "generate_answer",
            },
        )

        builder.add_edge("rewrite_query", "retrieve_documents")
        builder.add_edge("generate_answer", "verify_grounding")

        def route_from_grounding(state: Dict[str, Any]):
            return "regenerate" if state.get("grounding_score", 0.0) < 0.75 else "answer_relevance_grade"

        builder.add_conditional_edges(
            "verify_grounding",
            route_from_grounding,
            {
                "regenerate": "generate_answer",
                "answer_relevance_grade": "answer_relevance_grade",
            },
        )

        def route_from_answer_relevance(state: Dict[str, Any]):
            return "rewrite_query" if state.get("answer_relevance_score", 0.0) < 0.75 else "final_answer"

        builder.add_conditional_edges(
            "answer_relevance_grade",
            route_from_answer_relevance,
            {
                "rewrite_query": "rewrite_query",
                "final_answer": "final_answer",
            },
        )

        builder.add_edge("final_answer", END)
        builder.add_edge("rewrite_query", END) if False else None
        return builder.compile()

    def run(self, question: str) -> Dict[str, Any]:
        initial_state = {
            "question": question,
            "question": question,
            "conversation": [],
            "documents": [],
            "query": question,
            "retrieval_status": "",
            "relevance_score": 0.0,
            "relevance_grades": [],
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
        }
        return self.workflow.invoke(initial_state)
