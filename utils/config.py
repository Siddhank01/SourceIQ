import os
from pathlib import Path
from dotenv import load_dotenv


def get_settings():
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    return {
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY", ""),
        "GROQ_MODEL": os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        "MAX_RETRIES": int(os.getenv("MAX_RETRIES", "3")),
        "EMBEDDINGS_MODEL": os.getenv("EMBEDDINGS_MODEL", "all-MiniLM-L6-v2"),
        "CHROMA_PERSIST_DIR": os.getenv("CHROMA_PERSIST_DIR", "chroma_db"),
    }


def validate_env():
    settings = get_settings()
    if not settings.get("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY is missing. Add it to .env.")
    return settings
