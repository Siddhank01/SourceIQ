import os
from pathlib import Path
from dotenv import load_dotenv


def get_settings():
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    return {
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY", ""),
        "GROQ_MODEL": os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        "SUPABASE_URL": os.getenv("SUPABASE_URL", ""),
        "SUPABASE_SERVICE_ROLE_KEY": os.getenv("SUPABASE_SERVICE_ROLE_KEY", ""),
        "CORS_ORIGINS": os.getenv("CORS_ORIGINS", ""),
        "MAX_RETRIES": int(os.getenv("MAX_RETRIES", "3")),
        "EMBEDDINGS_MODEL": os.getenv("EMBEDDINGS_MODEL", "all-MiniLM-L6-v2"),
    }


def validate_env():
    settings = get_settings()
    if not settings.get("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY is missing. Add it to .env.")
    return settings
