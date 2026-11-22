"""RAG engine for ReliefMatch: retrieve PDF passages, then answer with Groq.

If no API key is configured, or the Groq call fails, the engine still returns
the retrieved passages so the Knowledge Assistant never shows a blank page.
"""

import logging
import os

from .vector_store import PROJECT_ROOT, VectorStore

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:  # python-dotenv is optional; real environment variables still work
    pass

logger = logging.getLogger(__name__)

# Groq retires models from time to time, so the model can be changed in .env
# with GROQ_MODEL=... without editing any code.
DEFAULT_MODEL = "openai/gpt-oss-120b"
TOP_K = 5
MIN_SCORE = 0.25   # below this, the PDFs probably don't cover the question

SYSTEM_PROMPT = (
    "You are the ReliefMatch Disaster Knowledge Assistant. "
    "Answer ONLY from the numbered context passages provided. "
    "If the passages do not contain the answer, say that the documents do not "
    "cover it; never invent facts. "
    "Be clear and concise, and use short bullet points for lists of precautions or steps. "
    "Mention the supporting source like (flood_management.pdf, p. 12). "
    "This is general guidance from official documents. In an emergency, "
    "people should follow instructions from local authorities and call 112."
)

NO_ANSWER_TEXT = (
    "I couldn't find relevant information about that in the loaded disaster "
    "management documents. Try rephrasing your question or ask about a topic "
    "covered by the PDFs."
)


def _build_context(passages):
    return "\n\n".join(
        f"[{i}] ({p['source']}, p. {p['page']})\n{p['text']}"
        for i, p in enumerate(passages, start=1)
    )


def _unique_sources(passages):
    seen, sources = set(), []
    for p in passages:
        key = (p["source"], p["page"])
        if key not in seen:
            seen.add(key)
            sources.append({"source": p["source"], "page": p["page"], "score": p["score"]})
    return sources


class RAGEngine:
    def __init__(self, store=None, client=None, model=None):
        self.store = store or VectorStore()
        self.model = model or os.getenv("GROQ_MODEL", DEFAULT_MODEL)
        self._client = client

    def prepare(self):
        """Load the saved index, or build it from the PDFs (slow only the first time)."""
        self.store.load_or_build()
        return self

    def _get_client(self):
        if self._client is None:
            api_key = os.getenv("GROQ_API_KEY", "").strip()
            if not api_key:
                return None
            from groq import Groq

            self._client = Groq(api_key=api_key)
        return self._client

    def ask(self, question, top_k=TOP_K):
        """Return {"answer", "sources", "passages", "used_llm", "notice"}."""
        question = str(question).strip()
        passages = self.store.search(question, top_k=top_k, min_score=MIN_SCORE)

        if not passages:
            return {"answer": NO_ANSWER_TEXT, "sources": [], "passages": [],
                    "used_llm": False, "notice": None}

        sources = _unique_sources(passages)
        client = self._get_client()

        if client is None:
            return {
                "answer": self._extract_only(passages),
                "sources": sources, "passages": passages, "used_llm": False,
                "notice": "GROQ_API_KEY not found in .env, so the most relevant "
                          "passages are shown instead of a generated answer.",
            }

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",
                     "content": f"Context passages:\n\n{_build_context(passages)}\n\n"
                                f"Question: {question}"},
                ],
                temperature=0.2,
                max_completion_tokens=2000,   # reasoning models also spend tokens thinking
            )
            answer = (response.choices[0].message.content or "").strip()
            if not answer:
                raise ValueError("The model returned an empty answer.")
            return {"answer": answer, "sources": sources, "passages": passages,
                    "used_llm": True, "notice": None}
        except Exception as error:
            logger.warning("Groq request failed: %s", error)
            return {
                "answer": self._extract_only(passages),
                "sources": sources, "passages": passages, "used_llm": False,
                "notice": f"The AI answer service is unavailable right now ({type(error).__name__}). "
                          "Showing the most relevant passages instead.",
            }

    @staticmethod
    def _extract_only(passages):
        lines = []
        for p in passages[:3]:
            text = p["text"] if len(p["text"]) <= 500 else p["text"][:500].rstrip() + "..."
            lines.append(f"**{p['source']}, p. {p['page']}**\n\n{text}")
        return "\n\n---\n\n".join(lines)