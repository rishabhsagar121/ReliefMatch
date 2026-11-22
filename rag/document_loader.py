"""PDF loader for the ReliefMatch RAG pipeline.

Public function (name and return format unchanged):
    load_pdf_documents() -> list of {"text": str, "source": str, "page": int}
"""

import logging
import re
from pathlib import Path

try:
    import pymupdf as fitz  # newer PyMuPDF versions
except ImportError:
    try:
        import fitz  # older PyMuPDF versions
    except ImportError as error:  # pragma: no cover
        raise ImportError(
            "PyMuPDF is required to read PDF files. "
            "Install it with: pip install pymupdf"
        ) from error


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = PROJECT_ROOT / "documents"


def _clean_text(text):
    """Tidy up text extracted from a PDF page."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)        # collapse repeated spaces/tabs
    text = re.sub(r"\n{3,}", "\n\n", text)     # collapse long blank gaps
    return text.strip()


def load_pdf_documents(documents_dir=None):
    """Read every PDF in the documents folder, one entry per readable page.

    documents_dir is optional; by default the project's "documents" folder
    is used, so existing calls like load_pdf_documents() keep working.
    """
    folder = Path(documents_dir) if documents_dir else DOCUMENTS_DIR

    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Documents folder not found: {folder}")

    pdf_files = sorted(
        (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"),
        key=lambda p: p.name.casefold(),
    )

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {folder}")

    documents = []

    for pdf_path in pdf_files:
        try:
            with fitz.open(pdf_path) as pdf:
                for page_number, page in enumerate(pdf, start=1):
                    text = _clean_text(page.get_text("text"))

                    if text:
                        documents.append(
                            {
                                "text": text,
                                "source": pdf_path.name,
                                "page": page_number,
                            }
                        )
        except Exception as error:
            # One damaged or password-protected PDF should not stop the others
            logger.warning("Skipping %s: %s", pdf_path.name, error)

    if not documents:
        raise ValueError("No readable text was found in the PDF files.")

    return documents