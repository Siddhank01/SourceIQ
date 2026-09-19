from pathlib import Path
from typing import List

from openpyxl import load_workbook
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader


class DocumentLoader:
    """Load local files and web URLs into LangChain documents."""

    def load_file(self, file_path: str) -> List[Document]:
        if file_path.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            return loader.load()
        if file_path.lower().endswith(".xlsx"):
            return self._load_workbook(file_path)
        loader = TextLoader(file_path)
        return loader.load()

    def _load_workbook(self, file_path: str) -> List[Document]:
        workbook = load_workbook(file_path, read_only=True, data_only=True)
        source = Path(file_path).name
        documents: List[Document] = []
        try:
            for worksheet in workbook.worksheets:
                rows = [
                    "\t".join("" if value is None else str(value) for value in row)
                    for row in worksheet.iter_rows(values_only=True)
                ]
                content = "\n".join(row for row in rows if row.strip())
                if content:
                    documents.append(Document(
                        page_content=content,
                        metadata={"source": source, "sheet": worksheet.title},
                    ))
        finally:
            workbook.close()
        return documents

    def load_url(self, url: str) -> List[Document]:
        loader = WebBaseLoader(url)
        return loader.load()

    def load_text(self, text: str) -> List[Document]:
        return [Document(page_content=text, metadata={"source": "text_input"})]
