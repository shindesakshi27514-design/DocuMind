from io import BytesIO
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_pdf_pages(file_bytes: bytes, filename: str) -> list[dict]:
    """Extract text from a PDF, keeping its source filename and page number."""
    reader = PdfReader(BytesIO(file_bytes))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append({
                "text": text,
                "source": filename,
                "page": page_number
            })
    return pages


def split_pages(pages: list[dict]) -> list[dict]:
    """Split each page into overlapping chunks while retaining its metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = []
    for page in pages:
        for part in splitter.split_text(page["text"]):
            if part.strip():
                chunks.append({
                    "text": part.strip(),
                    "source": page["source"],
                    "page": page["page"]
                })
    return chunks
