"""Vector store for the ReliefMatch RAG pipeline.

Steps: PDF pages -> chunks -> embeddings (Sentence Transformers) -> FAISS index.
The index is saved to the "vector_index" folder, so the PDFs are only processed
again when they change.
"""

import hashlib
import json
import logging
import re
from pathlib import Path

import faiss
import numpy as np

from .document_loader import DOCUMENTS_DIR, PROJECT_ROOT, load_pdf_documents

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 900        # characters per chunk
CHUNK_OVERLAP = 150     # characters shared between neighbouring chunks
MIN_CHUNK_LENGTH = 40   # ignore tiny fragments (page numbers, stray headings)
INDEX_DIR = PROJECT_ROOT / "vector_index"


def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks, preferring to cut at sentence ends."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            window = text[start:end]
            cut = max(window.rfind(". "), window.rfind("? "), window.rfind("! "))
            if cut > chunk_size * 0.6:
                end = start + cut + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break
        start = max(end - overlap, start + 1)

    return chunks


def chunk_documents(documents):
    """Turn page-level documents into chunks that keep their source and page."""
    chunks = []
    for doc in documents:
        for piece in split_text(doc["text"]):
            if len(piece) >= MIN_CHUNK_LENGTH:
                chunks.append(
                    {"text": piece, "source": doc["source"], "page": doc["page"]}
                )
    return chunks


def _documents_fingerprint(model_name):
    """Hash of the PDFs and settings, used to know when the index is outdated."""
    digest = hashlib.sha256()
    digest.update(f"{model_name}|{CHUNK_SIZE}|{CHUNK_OVERLAP}".encode("utf-8"))

    if DOCUMENTS_DIR.exists():
        for pdf in sorted(DOCUMENTS_DIR.glob("*.pdf"), key=lambda p: p.name.casefold()):
            info = pdf.stat()
            digest.update(f"{pdf.name}|{info.st_size}|{info.st_mtime_ns}".encode("utf-8"))

    return digest.hexdigest()


class VectorStore:
    def __init__(self, model_name=EMBEDDING_MODEL, embedder=None, index_dir=INDEX_DIR):
        self.model_name = model_name
        self._embedder = embedder          # can be injected (useful for testing)
        self.index_dir = Path(index_dir)
        self.index = None
        self.chunks = []

    # ----- embeddings -----
    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self.model_name)
        return self._embedder

    def _embed(self, texts):
        vectors = self._get_embedder().encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,   # so inner product == cosine similarity
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype="float32")

    # ----- build / save / load -----
    def build(self, documents=None):
        if documents is None:
            documents = load_pdf_documents()

        self.chunks = chunk_documents(documents)
        if not self.chunks:
            raise ValueError("No text chunks could be created from the documents.")

        vectors = self._embed([c["text"] for c in self.chunks])
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        return self

    def save(self):
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_dir / "index.faiss"))
        (self.index_dir / "chunks.json").write_text(
            json.dumps(self.chunks, ensure_ascii=False), encoding="utf-8"
        )
        (self.index_dir / "meta.json").write_text(
            json.dumps({"fingerprint": _documents_fingerprint(self.model_name)}),
            encoding="utf-8",
        )

    def _load_saved(self):
        meta_path = self.index_dir / "meta.json"
        index_path = self.index_dir / "index.faiss"
        chunks_path = self.index_dir / "chunks.json"

        if not (meta_path.exists() and index_path.exists() and chunks_path.exists()):
            return False

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("fingerprint") != _documents_fingerprint(self.model_name):
                return False
            self.index = faiss.read_index(str(index_path))
            self.chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
            return self.index.ntotal == len(self.chunks)
        except Exception as error:
            logger.warning("Could not load saved index (%s); rebuilding.", error)
            return False

    def load_or_build(self):
        """Use the saved index when the PDFs are unchanged, otherwise rebuild it."""
        if self._load_saved():
            return self
        self.build()
        try:
            self.save()
        except OSError as error:
            logger.warning("Could not save the index: %s", error)
        return self

    # ----- search -----
    def search(self, query, top_k=5, min_score=0.0):
        """Return the best matching chunks, each with a similarity score."""
        if self.index is None or not self.chunks:
            raise RuntimeError("The vector store is empty. Call load_or_build() first.")
        if not str(query).strip():
            return []

        query_vector = self._embed([str(query).strip()])
        k = min(top_k, len(self.chunks))
        scores, positions = self.index.search(query_vector, k)

        results = []
        for score, position in zip(scores[0], positions[0]):
            if position < 0 or score < min_score:
                continue
            chunk = self.chunks[int(position)]
            results.append({**chunk, "score": float(score)})
        return results