# agent/agent.py
import os
import time
from typing import Dict, Any, List, Optional
from pathlib import Path

# import generator function (LMStudio or Google) and retriever
# You already have a query/generation helper; we'll provide a minimal generator wrapper here.
import requests
from dotenv import load_dotenv
load_dotenv()

# Import your RAG retriever class — adjust path if needed.
# If you put retriever in rag/retriever.py:
try:
    from rag.retriever import RAGRetriever
except Exception:
    # fallback to agent/rag_retriever if you copied it there
    from rag_retriever import RAGRetriever  # type: ignore

from .tools import generate_pdf_report, generate_docx_report, send_email, OUT_DIR

LMSTUDIO_URL = os.getenv("LMSTUDIO_URL")  # e.g. http://localhost:11434/v1/generate (user might set)
LM_MODEL_NAME = os.getenv("LM_MODEL_NAME", "Mistral-7B-Instruct-v0.2")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

class SMEAgent:
    def __init__(self):
        # initialize retriever once
        print("Initializing RAG retriever for agent...")
        self.retriever = RAGRetriever()
        # tiny in-memory conversation history
        self.memory: List[Dict[str, str]] = []

    # --- LLM call abstraction ---
    def _call_llm(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
        """
        Try LMStudio (local) first if LMSTUDIO_URL provided, else try Google Gemini if key is set.
        Raises RuntimeError if no model endpoint is available.
        """
        # LMStudio REST (example endpoint expected by user earlier)
        if LMSTUDIO_URL:
            # user may provide LMSTUDIO_URL like http://localhost:11434/v1/generate
            payload = {
                "model": LM_MODEL_NAME,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            try:
                r = requests.post(LMSTUDIO_URL, json=payload, timeout=30)
                r.raise_for_status()
                data = r.json()
                # support common response shapes:
                if "results" in data and len(data["results"])>0 and "text" in data["results"][0]:
                    return data["results"][0]["text"]
                if "text" in data:
                    return data["text"]
                return str(data)
            except Exception as e:
                # fallback to Gemini if available
                print(f"LMStudio error: {e} — trying Gemini (if configured).")

        # Google Gemini via google.generativeai if available
        if GEMINI_API_KEY:
            try:
                import google.generativeai as genai
                genai.configure(api_key=GEMINI_API_KEY)
                model = genai.GenerativeModel("gemini-2.5-flash")
                # use a text/generation model; this API may differ by package version
                resp = model.generate_content(prompt)
                # many versions: check fields
                if hasattr(resp, "text"):
                    return resp.text
                if isinstance(resp, dict) and "candidates" in resp:
                    return resp["candidates"][0]["content"]
                return str(resp)
            except Exception as e:
                raise RuntimeError(f"Failed to call Gemini: {e}")

        raise RuntimeError("No LLM configured. Set LMSTUDIO_URL or GEMINI_API_KEY in .env")

    # --- simple QA using RAG + LLM ---
    def answer_question(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        # add to memory
        self.memory.append({"role": "user", "text": query, "ts": time.time()})

        # retrieve relevant chunks
        docs = self.retriever.hybrid_search(query, top_k=top_k)
        contexts = [d["text"] for d in docs]

        # build prompt: put context then question, ask for concise answer and cite sources (metadata)
        ctx_text = "\n\n---\n\n".join(contexts)
        prompt = f"""You are a helpful Subject-Matter Expert on Food Safety & Nutrition.
You were given the following context documents (excerpts). Use only the information below to answer the question.
If the answer is not in the context, say you don't have sufficient info.

CONTEXT:
{ctx_text}

QUESTION:
{query}

Answer concisely and quote the source filename when possible."""
        # call model
        answer = self._call_llm(prompt)
        # persist model answer to memory
        self.memory.append({"role": "assistant", "text": answer, "ts": time.time()})
        return {"answer": answer, "context": docs}

    # --- multi-step workflow planner (simple rule-based) ---
    def run_workflow(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
            request is a dict like:
            {
                "task": "create_handout",
                "title": "Cross contamination prevention",
                "audience": "kitchen staff",
                "length": "1 page",        # optional
                "email_to": "manager@example.com"  # optional
            }
            This function:
                - Builds a content prompt for the LLM using RAG
                - Generates content
                - Creates PDF and/or DOCX
                - Optionally emails the result
        """
        task = request.get("task")
        title = request.get("title", "Report")
        email_to = request.get("email_to")
        create_pdf = request.get("create_pdf", True)
        create_docx = request.get("create_docx", True)
        use_rag = request.get("use_rag", True)
        top_k = int(request.get("top_k", 5))

        # Build content prompt
        q = request.get("question") or f"Create a {request.get('length','one-page')} handout titled '{title}' for {request.get('audience','general users')} covering practical steps, examples, and short checklist."
        # Optionally augment with RAG context
        context_snippets = []
        if use_rag:
            ctx_docs = self.retriever.hybrid_search(q, top_k=top_k)
            context_snippets = [d["text"] for d in ctx_docs]

        ctx_text = ("\n\n---\n\n".join(context_snippets)) if context_snippets else ""
        prompt = f"""You are an expert in food safety and nutrition.
You will produce a clear, practical handout for {request.get('audience','kitchen staff')} titled: {title}.
Constraints:
- Keep it concise and actionable.
- Include a 5-item checklist at the end.
- If context is provided, ground statements in that context.

CONTEXT:
{ctx_text}

TASK INSTRUCTIONS:
{q}

Produce the handout as plain text (with headings)."""

        generated_text = self._call_llm(prompt)

        # build files
        created_files = []
        if create_pdf:
            pdf_path = generate_pdf_report(generated_text, filename=f"{title}.pdf")
            created_files.append(pdf_path)
        if create_docx:
            docx_path = generate_docx_report(generated_text, filename=f"{title}.docx")
            created_files.append(docx_path)

        email_result = None
        if email_to:
            subject = f"{title} — generated by SME Agent"
            body = f"Attached: {', '.join(Path(p).name for p in created_files)}\n\nGenerated content preview:\n\n{generated_text[:800]}"
            ok, msg = send_email(email_to, subject, body, attachments=created_files)
            email_result = {"success": ok, "message": msg}

        # record in memory
        self.memory.append({"role": "system", "text": f"workflow:{task}", "result": {"files": created_files, "email": email_result}, "ts": time.time()})

        return {
            "text": generated_text,
            "files": created_files,
            "email_result": email_result
        }