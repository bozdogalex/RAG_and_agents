# raglite/pdf_utils.py
from pathlib import Path
import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path: str | Path) -> str:
    doc = fitz.open(str(pdf_path))
    return "".join(page.get_text("text") for page in doc)
