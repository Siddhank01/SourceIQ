# Adaptive Self-RAG: Intelligent Document Research Assistant

(1)Problem Statement

Traditional retrieval-augmented generation (RAG) systems often retrieve a document bundle and answer immediately, without evaluating relevance, checking grounding, or correcting poor retrieval. Adaptive Self-RAG upgrades that pattern into an agentic document research workflow with retrieval, grading, query rewriting, verification, and bounded self-correction.

(2)Why Normal RAG Is Insufficient

A single-pass retrieval chain may answer with a plausible but unsupported statement, fail on follow-up questions, or miss context when the initial query is weak. Adaptive Self-RAG adds explicit retrieval decisions, relevance checks, iterative query rewriting, and hallucination mitigation before returning a grounded answer.

(3)Architecture Diagram

flowchart TD
    A[User Question] --> B[Understand Context]
    B --> C[Agentic Decision]
    C --> D[Retriever Tool]
    D --> E[Document Relevance Grader]
    E -->|Insufficient| F[Query Rewriter]
    F --> D
    E -->|Sufficient| G[Answer Generation]
    G --> H[Grounding / Hallucination Grader]
    H -->|Ungrounded| I[Regenerate]
    I --> G
    H -->|Grounded| J[Answer Relevance Grader]
    J -->|Poor| F
    J -->|Good| K[Final Grounded Answer]
    K --> L[Sources + Confidence + Verification]
```

(4)Complete Workflow

1. The user asks a question.
2. The system incorporates conversation memory and query understanding.
3. A LangGraph state graph decides whether to call the retriever or perform a rewrite.
4. A LangChain Retriever Tool performs retrieval from Chroma using HuggingFace embeddings.
5. A Pydantic document relevance grader scores the retrieved documents.
6. If relevance is low, the query is rewritten and retrieval is repeated.
7. A grounded answer is generated from the top supporting chunks.
8. A grounding score and hallucination check decide whether regeneration is needed.
9. An answer relevance grader checks usefulness and source alignment.
10. If answer relevance is weak, the query is rewritten and retrieval is attempted again.
11. The final answer is returned with sources, confidence, reliability, and workflow status.

(5)Agentic Behavior

The key idea is to treat retrieval as a tool action. Retrieval is represented by a LangChain Retriever Tool created through the retriever API in the codebase, then routed through LangGraph state transitions. Agentic behavior is expressed through stateful routing, bounded retry loops, and structured evaluation steps.

(6)Self-Correction Mechanism:-

Self-correction is implemented through:

- rel relevance grading;
- query rewriting when retrieval is insufficient;
- grounding and hallucination grading;
- answer relevance grading;
- retry count tracking and bounded loops.

(7)Hallucination Mitigation

The system tries to prevent unsupported answers by scoring each answer for groundedness against cited sources. If grounding is poor, the answer can be regenerated and re-grounded with retrieved context.

(8)Evaluation Approach

The repository includes an evaluation module that scores retrieval relevance, groundedness, answer relevance, correction success, successful query rewriting, and end-to-end answer quality using sample questions and expected source evidence.

(9)Example Use Cases

- company and internal policy research
- technical documentation Q&A
- research paper summarization
- policy or legal comparative analysis
- enterprise knowledge-base assistant

(10)Technologies

- Python
- Streamlit
- LangChain
- LangGraph
- LangChain Groq
- ChromaDB
- HuggingFace embeddings
- Pydantic
- Recursive text splitting
- document loading via LangChain and loaders

(11)Setup Instructions

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy the example environment file and add your Groq key:

```bash
copy .env.example .env
```

4. Run the app:

```bash
streamlit run app.py
```

## Project Structure

```text
app.py
graph/
    state.py
    workflow.py
    nodes.py
rag/
    loaders.py
    embeddings.py
    vectorstore.py
    retriever.py
    tools.py
models/
    graders.py
    schemas.py
utils/
    config.py
    helpers.py
evaluation/
    evaluate.py
tests/
requirements.txt
.env.example
.gitignore
README.md
```

## Limitations

- This implementation is intentionally lightweight and interview-friendly.
- Web URL retrieval is represented as an optional controlled retrieval pattern rather than a full web search engine.
- Groq API availability and model compatible availability may vary by region.

## Future Improvements

- Ranking and reranker integration
- Query planner and multi-hop reasoning
- Parallel retrieval over uploaded files and URLs
- Explanation layer for source traceability

## Interview Questions and Talking Points

- How does the workflow use LangGraph state transitions for routing?
- Where does the retriever tool live and how is retrieval explicit?
- How do Pydantic structured outputs improve grader reliability?
- Why is bounded retry important in an agentic RAG loop?
- How is grounding checked before the final answer is shown?
