# agent/main_api.py
import os, json, time, logging, requests
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")

logger = logging.getLogger("sme_agent")
logger.setLevel(logging.INFO)

# attempt to import google generative library (optional)
try:
    import google.generativeai as genai
    genai_available = True
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(model_name=GEMINI_MODEL)
except Exception as e:
    genai_available = False
    logger.info(f"google.generativeai not available: {e}")

def call_lm(prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
    """
        Robust LLM caller:
            - Google Gemini (genai), trying several API call signatures
        Returns string with model output or raises RuntimeError with details.
    """
    # 2) Try Google Gemini (google.generativeai), if available and API key set
    if genai_available and GEMINI_API_KEY:
        # try multiple call signatures depending on installed genai version
        try:
            if hasattr(genai, "generate"):
                # older/newer versions may expose generate()
                out = model.generate_content(prompt)
                # inspect structure for returned text
                if isinstance(out, dict):
                    # some versions: out['candidates'][0]['output']
                    if "candidates" in out and len(out["candidates"])>0:
                        return out["candidates"][0].get("content") or out["candidates"][0].get("output") or str(out)
                    return str(out)
                return str(out)
            elif hasattr(genai, "chat") and hasattr(genai.chat, "completions"):
                # genai.chat.completions.create(...)
                messages = [{"role": "user", "content": prompt}]
                resp = genai.chat.completions.create(model=GEMINI_MODEL, messages=messages, temperature=temperature, max_output_tokens=max_tokens)
                # resp might contain .candidates or .output. Try common patterns:
                if hasattr(resp, "candidates") and len(resp.candidates) > 0:
                    # candidate may have .content or .message
                    cand = resp.candidates[0]
                    # handle different shapes:
                    if hasattr(cand, "content"):
                        # content may be dict with 'text' key or plain text
                        c = cand.content
                        if isinstance(c, dict) and "text" in c:
                            return c["text"]
                        return str(c)
                    if hasattr(cand, "message") and isinstance(cand.message, dict):
                        return cand.message.get("content", "")
                # fallback to string
                return str(resp)
            elif hasattr(genai, "generate_text"):
                out = genai.generate_text(model=GEMINI_MODEL, text=prompt, max_output_tokens=max_tokens)
                # check typical fields
                if isinstance(out, dict) and "candidates" in out and len(out["candidates"])>0:
                    return out["candidates"][0].get("content", "")
                return str(out)
            else:
                raise RuntimeError("Installed google.generativeai does not expose supported call signatures.")
        except Exception as e:
            logger.exception("Failed to call Gemini via google.generativeai: %s", e)
            raise RuntimeError(f"Failed to call Gemini: {e}")

    # Nothing available
    raise RuntimeError("No LLM available: Google Gemini not configured/available.")



import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from .agent import SMEAgent

app = FastAPI(title="Food Safety SME Agent")

agent = SMEAgent()

# ----------------------
# Request models
# ----------------------
class ChatRequest(BaseModel):
    query: str
    top_k: int = 5

class WorkflowRequest(BaseModel):
    task: str
    title: str
    audience: str = "general"
    length: str = "one-page"
    email_to: str | None = None
    create_pdf: bool = True
    create_docx: bool = True
    use_rag: bool = True
    top_k: int = 5

# ----------------------
# Endpoints
# ----------------------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat") # RAG QA using Agent
def chat(req: ChatRequest):
    try:
        resp = agent.answer_question(req.query, top_k=req.top_k)
        return resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/workflow") # Multi-step workflow -- manual pipelines
def workflow(req: WorkflowRequest):
    payload = req.dict()
    try:
        resp = agent.run_workflow(payload)
        return resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# file download (served from reports/outputs)
OUT_DIR = Path(__file__).resolve().parents[1] / "reports" / "outputs"

@app.get("/download/{filename}") # File download endpoint
def download(filename: str):
    fpath = OUT_DIR / filename
    if not fpath.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(fpath, media_type="application/octet-stream", filename=filename)

class PlanRequest(BaseModel):
    task: str

@app.post("/plan_and_run") # LLM-planned task execution
def plan_and_run(req: PlanRequest):
    try:
        plan = agent.plan_task(req.task)
        result = agent.execute_plan(plan)
        return {"plan": plan, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from agent.feedback_store import add_feedback

class FeedbackRequest(BaseModel):
    query: str
    answer: str
    feedback: str  # "good" or "bad"
    corrected_answer: str | None = None

@app.post("/feedback")
def feedback(req: FeedbackRequest):
    add_feedback(req.dict())
    return {"status": "saved"}

class AgentRequest(BaseModel):
    query: str

@app.post("/agent") # Agent router: rule-based or LLM-planned
def agent_router(req: AgentRequest):
    text = req.query.lower()

    # If user explicitly asks for a handout/report → rule-based workflow
    if "create handout" in text or "make report" in text:
        payload = {
            "task": "create_handout",
            "title": "Auto Handout",
            "audience": "general",
            "email_to": None,
            "create_pdf": True,
            "create_docx": True
        }
        return agent.run_workflow(payload)

    # otherwise → let LLM plan
    plan = agent.plan_task(req.query)
    result = agent.execute_plan(plan)
    return {"plan": plan, "result": result}



# Run with:
# python3 -m uvicorn agent.main_api:app --reload

# Example curl commands:

# Test 1 — Basic Question
# curl -X POST http://127.0.0.1:8000/chat \
#   -H "Content-Type: application/json" \
#   -d '{"query": "What is cross contamination?", "top_k": 5}'

# Test 2 — Follow-up Question (tests memory)
# curl -X POST http://127.0.0.1:8000/chat \
#   -H "Content-Type: application/json" \
#   -d '{"query": "What is cross contamination?"}'

# Test 3 — Non-RAG Query
# curl -X POST http://127.0.0.1:8000/chat \
#   -H "Content-Type: application/json" \
#   -d '{"query": "Who won the 2022 FIFA World Cup?"}'

# Test 4 — RAG Failover Behavior
# Temporarily stop Elasticsearch: docker stop es01
# Then run:
# curl -X POST http://127.0.0.1:8000/chat \
#   -H "Content-Type: application/json" \
#   -d '{"query": "What is food hygiene?"}'

# Expected:
# • System shouldn’t crash.
# • Should tell user: “RAG unavailable, answering without context.” (Depending on how strict you want to be.)

# Test 5: Check multi-step workflows
# curl -X POST http://127.0.0.1:8000/workflow \
#   -H "Content-Type: application/json" \
#   -d '{
#         "task": "create_handout",
#         "title": "Cross Contamination Prevention",
#         "audience": "food handlers",
#         "email_to": "kushagra7503@gmail.com",
#         "create_pdf": true,
#         "create_docx": true
#       }'

# Test 6 — Plan and Run
# curl -X POST http://127.0.0.1:8000/plan_and_run \
#   -H "Content-Type: application/json" \
#   -d '{"task": "Create a one page handout on cross contamination, convert to PDF and email to kushagra7503@gmail.com"}'



# Test 7 — Test /agent router

# Rule-based workflow path
# curl -X POST http://127.0.0.1:8000/agent \
#   -H "Content-Type: application/json" \
#   -d '{"query": "Create handout on cross contamination"}'

# • It should NOT generate a plan
# • It should run:
#       run_workflow → PDF + DOCX generation

# LLM planning path
# curl -X POST http://127.0.0.1:8000/agent \
#   -H "Content-Type: application/json" \
#   -d '{"query": "Generate a quiz on foodborne pathogens and mail me the results"}'


# Test Chat + Memory
# curl -X POST http://127.0.0.1:8000/chat \
#   -H "Content-Type: application/json" \
#   -d '{"query":"What is salmonella?"}'

# curl -X POST http://127.0.0.1:8000/chat \
#   -H "Content-Type: application/json" \
#   -d '{"query":"How does this spread?"}'



# Test plan JSON validity
# curl -X POST http://127.0.0.1:8000/plan_and_run \
#   -H "Content-Type: application/json" \
#   -d '{"task":"Make a 1-page summary about allergens and mail it"}'

# Test attachment resolution
# curl -X POST http://127.0.0.1:8000/agent \
#   -H "Content-Type: application/json" \
#   -d '{"query":"Create a quiz on bacteria and mail me the PDF"}'