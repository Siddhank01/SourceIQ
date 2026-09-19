from typing import List

from langchain_core.documents import Document


def _split_documents(documents: List[Document], chunk_size: int = 500, overlap: int = 80) -> List[Document]:
    """Split documents into stable, externally persisted citation chunks."""
    chunks: List[Document] = []
    for document in documents:
        content = document.page_content or ""
        if not content:
            continue
        start = 0
        index = 0
        while start < len(content):
            end = min(len(content), start + chunk_size)
            metadata = dict(document.metadata or {})
            parent_id = metadata.get("source_id") or metadata.get("source") or "source"
            metadata["parent_source_id"] = str(parent_id)
            metadata["chunk_id"] = f"{parent_id}::chunk-{index}"
            metadata["source_id"] = metadata["chunk_id"]
            chunks.append(Document(page_content=content[start:end], metadata=metadata))
            if end >= len(content):
                break
            start = max(end - overlap, start + 1)
            index += 1
    return chunks
