# agent/agent.py
import os
import time
from agent.tools import generate_pdf_report, generate_docx_report, send_email, OUT_DIR
from agent.feedback_store import load_feedback
from typing import Dict, Any, List, Optional
from pathlib import Path
import json

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
    from retriever import RAGRetriever  # type: ignore

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEFAULT_EMAIL = os.getenv("DEFAULT_EMAIL")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")

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
        Try Google Gemini if key is set.
        Raises RuntimeError if no model endpoint is available.
        """
        # Google Gemini via google.generativeai if available
        if GEMINI_API_KEY:
            try:
                import google.generativeai as genai
                genai.configure(api_key=GEMINI_API_KEY)
                model = genai.GenerativeModel(model_name=GEMINI_MODEL)
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

        raise RuntimeError("No LLM configured. Set GEMINI_API_KEY in .env")
    
    def _build_chat_context(self, last_n=5):
        msgs = self.memory[-last_n:]
        formatted = ""
        for m in msgs:
            role = m["role"]
            text = m["text"]
            formatted += f"{role.upper()}: {text}\n"
        return formatted

    def _fetch_feedback_context(self, query: str):
        feedback_data = load_feedback()
        context = []

        for fb in feedback_data:
            # Simple fuzzy match
            if fb["query"].lower() in query.lower():
                if fb["feedback"] == "good":
                    context.append("User previously liked this answer: " + fb["answer"])
                elif fb["feedback"] == "bad" and fb.get("corrected_answer"):
                    context.append("Important correction from user: " + fb["corrected_answer"])
        return "\n\n".join(context)

    # --- simple QA using RAG + LLM ---
    def answer_question(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        query = query.strip()
        # add to memory
        self.memory.append({"role": "user", "text": query, "ts": time.time()})

        # retrieve relevant chunks
        docs = self.retriever.hybrid_search(query, top_k=top_k)
        contexts = [d["text"] for d in docs]

        # build prompt: put context then question, ask for concise answer and cite sources (metadata)
        chat_history = self._build_chat_context(last_n=6)
        ctx_text = "\n\n---\n\n".join(contexts)
        feedback_context = self._fetch_feedback_context(query)
        prompt = f"""
You are a helpful Food Safety SME.

Here is recent chat history (use it for context if useful):
{chat_history}

Here is user feedback memory that MUST influence your answer (if relevant):
{feedback_context}

Here is RAG context:
{ctx_text}

User question:
{query}

Give a clear, improved answer. Cite sources when possible.
"""

        # call model
        answer = self._call_llm(prompt)
        # persist model answer to memory
        self.memory.append({"role": "assistant", "text": answer, "ts": time.time()})
        return {"answer": answer, "context": docs}
    
    def improve_answer(self, bad_answer: str) -> str:
        prompt = f"""
A previous answer was marked as incorrect:

BAD ANSWER:
{bad_answer}

Rewrite it correctly, clearly, and concisely.
    """
        return self._call_llm(prompt)

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

    # --- BASIC PLANNER: LLM generates a JSON plan ---
    def plan_task(self, task_description: str) -> dict:
        """
        Ask the LLM to create a simple JSON execution plan.
        Returns a dict with key "steps": list of step dicts.
        """
        task_description = task_description.strip()
        chat_history = self._build_chat_context(last_n=6)
        prompt = f"""
You are a planning assistant. Convert the following task into a JSON plan.
Only return valid JSON. No explanations.

Here is previous conversation for context (if relevant):
{chat_history}

TASK:
{task_description}

Your JSON MUST follow this structure:

{{
    "steps": [
        {{
        "id": "step1",
        "action": "rag_search" | "llm_generate" | "create_pdf" | "create_docx" | "send_email",
        "input": "...",
        "depends_on": []
        }}
    ]
}}
Rules:
- If the user's task description contains an email address, use that email in the send_email step.
- If the user asks to send email and does NOT specify an email, use the DEFAULT email "kushagra7503@gmail.com".
- For send_email, ALWAYS use structured JSON:

{{
    "to": "<email>",
    "subject": "...",
    "body": "...",
    "attachments": ["step3"]
}}
"""
        plan_text = self._call_llm(prompt)

        # try to load JSON directly, otherwise try to extract JSON substring
        try:
            plan = json.loads(plan_text)
        except Exception:
            import re
            m = re.search(r"\{[\s\S]*\}", plan_text)
            if not m:
                raise RuntimeError("LLM did not return JSON. Response:\n" + plan_text)
            try:
                plan = json.loads(m.group(0))
            except Exception as e:
                raise RuntimeError(f"Could not parse JSON from LLM output: {e}\nRaw output:\n{plan_text}")

        # basic validation
        if not isinstance(plan, dict) or "steps" not in plan or not isinstance(plan["steps"], list):
            raise RuntimeError("Plan JSON invalid: expected a dict with a 'steps' list. Plan:\n" + json.dumps(plan, indent=2))
        return plan

    # --- EXECUTE PLAN ---
    def execute_plan(self, plan: dict) -> dict:
        """
        Execute each step in order. Returns dict of results keyed by step id.
        """
        if not isinstance(plan, dict) or "steps" not in plan:
            raise RuntimeError("Invalid plan. Must be a dict with key 'steps' containing a list.")

        results = {}

        for step in plan["steps"]:
            sid = step.get("id")
            if not sid:
                raise RuntimeError(f"Step without id: {step}")
            action = step.get("action")
            # normalize input: coerce to string if not None
            raw_input = step.get("input", "")
            if isinstance(raw_input, (dict, list)):
                # convert complex input into compact JSON string to feed into LLM/rag
                input_text = json.dumps(raw_input, ensure_ascii=False)
            else:
                input_text = str(raw_input or "")

            # resolve dependencies: append outputs of deps as text
            for dep in step.get("depends_on", []):
                if dep in results:
                    dep_res = results[dep]
                    # if dependency result is a dict, convert to readable string
                    if isinstance(dep_res, dict):
                        dep_str = json.dumps(dep_res, ensure_ascii=False)
                    else:
                        dep_str = str(dep_res)
                    # append with separator
                    input_text = input_text + "\n\n[dependency:" + dep + "]\n" + dep_str

            # ---- ACTION HANDLERS ----
            if action == "rag_search":
                # input_text is query
                docs = self.retriever.hybrid_search(input_text.strip(), top_k=5)
                results[sid] = {"type": "rag", "value": "\n".join([d["text"] for d in docs]), "docs": docs}

            elif action == "llm_generate":
                gen = self._call_llm(input_text)
                results[sid] = {"type": "llm", "value": gen}

            elif action == "create_pdf":
                pdf_path = generate_pdf_report(input_text, filename=f"{sid}.pdf")
                results[sid] = {"type": "file", "value": pdf_path}

            elif action == "create_docx":
                docx_path = generate_docx_report(input_text, filename=f"{sid}.docx")
                results[sid] = {"type": "file", "value": docx_path}

            elif action == "send_email":
                # input_text may contain raw body text
                # but email config is inside step["input"] if it's a dict
                email_to = None
                subject = "Automated SME Report"
                body = input_text
                attachments = []

                # If input is dict → extract email fields
                if isinstance(raw_input, dict):
                    email_to = raw_input.get("to")
                    subject = raw_input.get("subject", subject)
                    body = raw_input.get("body", body)

                    # If user asked for email but did NOT specify address → fill default
                    if "send" in input_text.lower() and not email_to:
                        email_to = DEFAULT_EMAIL

                    # Resolve attachments from step results or direct files
                    att = raw_input.get("attachment") or raw_input.get("attachments")
                    if att:
                        if isinstance(att, str):
                            att = [att]

                        for a in att:

                            # 1. If refers to a step ID
                            if a in results:
                                step_result = results[a]

                                # Try common patterns
                                file_path = (
                                    step_result.get("value") 
                                    or step_result.get("path")
                                    or step_result.get("file")
                                )
                                if file_path and os.path.exists(file_path):
                                    attachments.append(file_path)
                                    continue

                                # If step returned {"type": "file"}
                                if step_result.get("type") == "file":
                                    # Most reliable fallback
                                    possible = step_result.get("value")
                                    if possible and os.path.exists(possible):
                                        attachments.append(possible)
                                        continue

                                print(f"[WARN] Step '{a}' found but it does not contain a file path.")
                                continue

                            # 2. If it's a clean file name inside OUT_DIR
                            out_test = OUT_DIR / a
                            if out_test.exists():
                                attachments.append(str(out_test))
                                continue

                            # 3. If it's a direct filesystem path
                            if os.path.exists(a):
                                attachments.append(a)
                                continue

                            # 4. If it's something like 'step3.file' or similar junk
                            if "." in a:
                                # Try treat as filename inside outputs
                                candidate = OUT_DIR / a.split(".")[-1]  # fallback guess
                                if candidate.exists():
                                    attachments.append(str(candidate))
                                    continue

                            print(f"[WARN] Could not resolve attachment: {a}")

                # fallback: try top-level fields
                if not email_to:
                    email_to = step.get("email_to") or step.get("to")

                ok, msg = send_email(email_to, subject, body, attachments=attachments)
                results[sid] = {"type": "email", "value": {"success": ok, "message": msg}}

            else:
                # Unknown action -> store as string response
                results[sid] = {"type": "unknown", "value": f"Unknown action: {action}", "raw": step}

        return results