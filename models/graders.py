from typing import Any, Dict
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from models.schemas import DocumentRelevance, GroundingVerification, AnswerRelevance
from utils.config import get_settings


class GraderRunner:
    """Run Pydantic structured graders using ChatGroq only."""

    def __init__(self, model_name: str = "openai/gpt-oss-20b"):
        self.settings = get_settings()
        self.model_name = model_name or self.settings.get("GROQ_MODEL", "openai/gpt-oss-20b")
        self.llm = ChatGroq(
            model=self.model_name,
            groq_api_key=self.settings.get("GROQ_API_KEY"),
            temperature=0,
        )

    def grade_document_relevance(self, query: str, docs: list) -> DocumentRelevance:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You grade document relevance for a retrieval workflow. Return a DocumentRelevance object."),
            ("user", "Question: {query}\nDocuments: {docs}"),
        ])
        structured_llm = self.llm.with_structured_output(DocumentRelevance)
        chain = prompt | structured_llm
        response = chain.invoke({"query": query, "docs": [d.page_content for d in docs]})
        return response

    def grade_hallucination_grounding(self, query: str, answer: str, docs: list) -> GroundingVerification:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You verify whether an answer is grounded in retrieved documents. Return a GroundingVerification object."),
            ("user", "Question: {query}\nAnswer: {answer}\nDocuments: {docs}"),
        ])
        structured_llm = self.llm.with_structured_output(GroundingVerification)
        chain = prompt | structured_llm
        response = chain.invoke({"query": query, "answer": answer, "docs": [d.page_content for d in docs]})
        return response

    def grade_answer_relevance(self, query: str, answer: str) -> AnswerRelevance:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You grade whether an answer is relevant to the question. Return an AnswerRelevance object."),
            ("user", "Question: {query}\nAnswer: {answer}"),
        ])
        structured_llm = self.llm.with_structured_output(AnswerRelevance)
        chain = prompt | structured_llm
        response = chain.invoke({"query": query, "answer": answer})
        return response
