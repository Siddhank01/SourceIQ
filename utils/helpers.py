from typing import Any, Dict, List
from urllib.parse import urlparse


def safe_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def normalize_sources(docs: List[Any]) -> List[Dict[str, Any]]:
    sources = []
    for doc in docs:
        meta = doc.metadata if hasattr(doc, "metadata") else {}
        source = meta.get("source") or meta.get("url") or meta.get("filename") or "unknown"
        item = {
            "source": source,
            "content": doc.page_content[:400],
            "metadata": meta,
        }
        sources.append(item)
    return sources


def bounded_retry_count(retry_count: int, max_retries: int) -> bool:
    return retry_count < max_retries
