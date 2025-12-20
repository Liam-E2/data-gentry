from typing import BinaryIO

from pymupdf4llm import to_markdown # type: ignore
from pymupdf import Document # type: ignore



def preprocess_pdf(pdf_bytes: BinaryIO) -> str:
    """
    Given the contents of a PDF file, convert it to Markdown with pymupdf4llm.
    """
    doc = Document(stream=pdf_bytes.read())
    md: str = to_markdown(doc=doc)
    
    return md
