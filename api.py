from __future__ import annotations

import json
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from backend_db import delete_session_group, get_session, init_db, list_sessions, upsert_session
from rag.loaders import DocumentLoader
from rag.vectorstore import VectorStoreManager
from utils.config import get_settings

app = FastAPI(title="Self-RAG API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://127.0.0.1:5173", "http://localhost:5173", "http://127.0.0.1:5175"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
init_db()

SUPPORTED_MODELS = ("openai/gpt-oss-20b", "openai/gpt-oss-120b")
CHROMA_ROOT = Path(__file__).resolve().parent / "chroma_sessions"


def error(status: int, detail: str) -> None:
    raise HTTPException(status_code=status, detail=detail)


def owner_value(owner: str | None) -> str:
    return owner.strip() if owner and owner.strip() else "guest"


def validate_model(model: str | None) -> str:
    configured = get_settings().get("GROQ_MODEL", SUPPORTED_MODELS[0])
    selected = model or configured
    if selected not in SUPPORTED_MODELS:
        error(400, f"Unsupported Groq model. Supported models: {', '.join(SUPPORTED_MODELS)}")
    return selected


def require_key() -> str:
    key = get_settings().get("GROQ_API_KEY", "")
    if not key:
        error(503, "GROQ_API_KEY is missing. Add it to .env before asking a question.")
    return key


def load_documents(files: list[UploadFile], urls: list[str], workdir: Path) -> tuple[list[Document], list[dict[str, str]]]:
    loader = DocumentLoader()
    documents: list[Document] = []
    sources: list[dict[str, str]] = []
    for upload in files:
        if not upload.filename:
            continue
        target = workdir / Path(upload.filename).name
        try:
            with target.open("wb") as destination:
                shutil.copyfileobj(upload.file, destination)
            loaded = loader.load_file(str(target))
        except Exception as exc:
            error(422, f"Could not parse {upload.filename}: {exc}")
        documents.extend(loaded)
        sources.append({"name": upload.filename, "type": "pdf" if upload.filename.lower().endswith(".pdf") else "file", "detail": f"{len(loaded)} pages/chunks indexed"})
    for url in urls:
        try:
            loaded = loader.load_url(url)
        except Exception as exc:
            error(422, f"Could not load URL {url}: {exc}")
        documents.extend(loaded)
        sources.append({"name": url, "type": "web", "detail": "Web source indexed"})
    return documents, sources


def answer_with_context(question: str, model: str, documents: list[Document], history: list[dict[str, Any]], session_id: str) -> tuple[str, list[str]]:
    require_key()
    try:
        persist_directory = CHROMA_ROOT / session_id
        persist_directory.parent.mkdir(parents=True, exist_ok=True)
        vectorstore = VectorStoreManager(persist_directory=str(persist_directory))
        if documents:
            vectorstore.create_vectorstore(documents)
        context_docs = vectorstore.similarity_search(question, k=4)
        if not context_docs:
            error(422, "No indexed source was found for this session. Upload a document or add a URL before asking a question.")
        context = "\n\n".join(document.page_content for document in context_docs)
        history_text = "\n".join(f"{item.get('role')}: {item.get('text')}" for item in history[-6:])
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a grounded document research assistant. Answer only from the provided context. If context is insufficient, say so. Cite source names in square brackets when possible."),
            ("user", "Conversation:\n{history}\n\nContext:\n{context}\n\nQuestion: {question}"),
        ])
        llm = ChatGroq(model=model, groq_api_key=require_key(), temperature=0)
        response = (prompt | llm).invoke({"history": history_text, "context": context, "question": question})
        answer = response.content if isinstance(response.content, str) else str(response.content)
        citations = sorted({Path(str(document.metadata.get("source", "source"))).name for document in context_docs})
        return answer, citations
    except HTTPException:
        raise
    except Exception as exc:
        message = str(exc).lower()
        if "collection" in message or "does not exist" in message or "no such" in message:
            error(422, "This session has no indexed source. Upload a document or add a URL before asking a question.")
        if "401" in message or "api key" in message or "authentication" in message:
            error(502, "Groq rejected the configured API key. Check GROQ_API_KEY in .env.")
        if "timeout" in message:
            error(504, "The Groq request timed out. Please retry.")
        error(502, f"LLM generation failed: {exc}")


@app.get("/api/health")
def health() -> dict[str, Any]:
    settings = get_settings()
    return {"ok": True, "provider": "groq", "model": settings.get("GROQ_MODEL", SUPPORTED_MODELS[0]), "api_key_configured": bool(settings.get("GROQ_API_KEY"))}


@app.get("/api/models")
def models() -> dict[str, list[str]]:
    return {"models": list(SUPPORTED_MODELS)}


@app.get("/api/sessions")
def sessions(owner: str = "guest", saved: bool | None = None) -> list[dict[str, Any]]:
    return list_sessions(owner_value(owner), saved)


@app.get("/api/sessions/{session_id}")
def session(session_id: str, owner: str = "guest") -> dict[str, Any]:
    result = get_session(session_id, owner_value(owner))
    if not result:
        error(404, "Session not found")
    return result


@app.delete("/api/sessions/{session_id}")
def remove_session(session_id: str, owner: str = "guest") -> dict[str, bool]:
    if not delete_session_group(session_id, owner_value(owner)):
        error(404, "Session not found")
    return {"deleted": True}


@app.post("/api/sessions")
def save_session(payload: dict[str, Any]) -> dict[str, Any]:
    required = ("id", "title", "messages")
    if any(not payload.get(field) for field in required):
        error(400, "Session id, title, and messages are required")
    return upsert_session(payload["id"], owner_value(payload.get("owner")), payload["title"], payload.get("description", ""), bool(payload.get("saved", True)), payload["messages"], payload.get("sources", []))


@app.post("/api/answer")
def answer(question: str = Form(...), session_id: str | None = Form(None), owner: str = Form("guest"), model: str | None = Form(None), history: str = Form("[]"), urls: str = Form("[]"), sources: str = Form("[]"), files: list[UploadFile] = File(default=[])) -> dict[str, Any]:
    if not question.strip():
        error(400, "Question cannot be empty")
    selected_model = validate_model(model)
    try:
        conversation = json.loads(history)
        url_list = json.loads(urls)
        existing_sources = json.loads(sources)
        if not isinstance(conversation, list) or not isinstance(url_list, list) or not isinstance(existing_sources, list):
            raise ValueError
    except ValueError:
        error(400, "history and urls must be JSON arrays")
    request_session_id = session_id or str(uuid.uuid4())
    with tempfile.TemporaryDirectory(prefix="self-rag-upload-") as directory:
        documents, uploaded_sources = load_documents(files, [str(item) for item in url_list], Path(directory))
        generated_answer, citations = answer_with_context(question, selected_model, documents, conversation, request_session_id)
    answer_message = {"role": "assistant", "text": generated_answer, "citations": citations, "time": "Now"}
    user_message = {"role": "user", "text": question, "time": "Now"}
    all_messages = conversation + [user_message, answer_message]
    all_sources = existing_sources + uploaded_sources
    upsert_session(request_session_id, owner_value(owner), question[:44], f"{sum(message.get('role') == 'user' for message in all_messages)} questions • {len(all_sources)} sources", False, all_messages, all_sources)
    return {"answer": generated_answer, "citations": citations, "session_id": request_session_id, "model": selected_model, "sources": uploaded_sources}
