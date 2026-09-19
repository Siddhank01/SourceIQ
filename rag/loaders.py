import uuid
from pathlib import Path
from typing import List

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document


class DocumentLoader:
    """Load common local and web sources without heavyweight loader packages."""

    @staticmethod
    def _with_source_metadata(document: Document, source_name: str, source_type: str, index: int = 0) -> Document:
        metadata = dict(document.metadata or {})
        source_id = str(metadata.get("source_id") or uuid.uuid4())
        metadata.setdefault("source", source_name)
        metadata.setdefault("source_id", source_id)
        metadata.setdefault("source_type", source_type)
        metadata["chunk_id"] = f"{source_id}::page-{index}"
        return Document(page_content=document.page_content, metadata=metadata)

    def load_file(self, file_path: str) -> List[Document]:
        source_name = Path(file_path).name
        if file_path.lower().endswith(".pdf"):
            from pypdf import PdfReader

            reader = PdfReader(file_path)
            return [
                self._with_source_metadata(Document(page_content=page.extract_text() or ""), source_name, "pdf", index)
                for index, page in enumerate(reader.pages)
            ]
        if file_path.lower().endswith(".xlsx"):
            return self._load_workbook(file_path)
        content = Path(file_path).read_text(encoding="utf-8", errors="replace")
        return [self._with_source_metadata(Document(page_content=content), source_name, "text")]

    def _load_workbook(self, file_path: str) -> List[Document]:
        from openpyxl import load_workbook

        workbook = load_workbook(file_path, read_only=True, data_only=True)
        source = Path(file_path).name
        documents: List[Document] = []
        try:
            for worksheet in workbook.worksheets:
                rows = ["\t".join("" if value is None else str(value) for value in row) for row in worksheet.iter_rows(values_only=True)]
                content = "\n".join(row for row in rows if row.strip())
                if content:
                    documents.append(self._with_source_metadata(Document(page_content=content, metadata={"sheet": worksheet.title}), source, "xlsx"))
        finally:
            workbook.close()
        return documents

    def load_url(self, url: str) -> List[Document]:
        response = requests.get(url, timeout=20, headers={"User-Agent": "SourceIQ/1.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for element in soup(["script", "style", "noscript"]):
            element.decompose()
        text = soup.get_text(" ", strip=True)
        return [self._with_source_metadata(Document(page_content=text), url, "web")]

    def load_text(self, text: str) -> List[Document]:
        return [self._with_source_metadata(Document(page_content=text), "text_input", "text")]
