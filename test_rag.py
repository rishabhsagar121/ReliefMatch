"""Quick check of the RAG pipeline from the project root: python test_rag.py

Retrieval needs no API key. If GROQ_API_KEY is set in .env, you also get
a generated answer.
"""
from rag.rag_engine import RAGEngine

engine = RAGEngine().prepare()
print("Knowledge base ready.\n")

question = "What precautions should people take during a flood?"
result = engine.ask(question)

print("Question:", question)
print("Used AI answer:", result["used_llm"])
if result["notice"]:
    print("Notice:", result["notice"])
print("\nAnswer:\n", result["answer"])
print("\nSources:")
for s in result["sources"]:
    print(f"  - {s['source']}, page {s['page']} (score {s['score']:.2f})")