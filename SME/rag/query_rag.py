"""
query_rag.py
A complete standalone script to test your RAG pipeline end-to-end.

Pipeline:
1. Retrieve top-K chunks from Elasticsearch using your hybrid retriever.
2. Construct a high-quality prompt using retrieved context.
3. Send prompt to LMStudio (local LLM server).
4. Print retrieved chunks + generated answer.
"""

import os
import google.generativeai as genai
from rag.retriever import RAGRetriever
from dotenv import load_dotenv

load_dotenv()

# === Setup Google Gemini ===
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

if GEMINI_API_KEY is None:
    raise RuntimeError("❌ GEMINI_API_KEY not found in .env")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(GOOGLE_MODEL)

# === Load Retriever ===
retriever = RAGRetriever(
    es_url=os.getenv("ES_URL"),
    index_name=os.getenv("INDEX_NAME"),
    reranker_model_name=os.getenv("RERANKER_MODEL")
)

# === Answer Generation ===
def generate_answer(question: str, context: str) -> str:
    prompt = f"""
You are a Food Safety Expert AI.

Use ONLY the context provided to answer the question.
If the context does not contain the answer, say "The context does not contain specific information."

Context:
---------
{context}

Question:
---------
{question}

Answer in 4-6 concise lines, factual and precise.
"""

    response = model.generate_content(prompt)
    return response.text.strip()

# === RAG Pipeline ===
def run_rag_pipeline(question: str, top_k: int = 5):
    print("\n🔍 Retrieving relevant context...\n")
    docs = retriever.hybrid_search(
        question, 
        top_k=top_k, 
        use_bm25=True, 
        use_reranker=True
    )

    if not docs:
        print("❌ No relevant context found!")
        return

    # Build context string
    context = "\n\n---\n\n".join([d["text"] for d in docs])

    print("🧠 Generating answer using Google Gemini...\n")
    answer = generate_answer(question, context)

    # Display results
    print("=== Retrieved Context Chunks ===")
    for i, d in enumerate(docs, 1):
        print(f"\n[{i}] Metadata: {d['metadata']}")
        print(d["text"][:250], "...\n")

    print("\n=== Final Answer ===")
    print(answer)
    print("\n" + "="*60 + "\n")

# === Main ===
if __name__ == "__main__":
    print("📘 Food Safety RAG System (Google Gemini + Elasticsearch)")
    print("Type your question, or type 'exit' to quit.\n")

    while True:
        question = input("Your question: ")
        if question.lower() in ["exit", "quit"]:
            break
        
        run_rag_pipeline(question)