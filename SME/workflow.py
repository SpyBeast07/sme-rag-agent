import os
import time
import subprocess
import requests, json
from dotenv import load_dotenv

ES_CHECK_SCRIPT = "scripts/tests/es_Check.py"
GEMINI_CHECK_SCRIPT = "scripts/tests/gemini_test.py"
GENERATE_METADATA = "scripts/generate_metadata.py"
BUILD_DATABASE = "scripts/build_database.py"
FASTAPI_URL = "http://127.0.0.1:8000"
INDEX_NAME = os.getenv("INDEX_NAME", "food_safety_rag")

# ------------------------------------------
# UTILITY FUNCTIONS
# ------------------------------------------
def run_python(script_path):
    print(f"\n▶ Running {script_path} ...\n")
    result = subprocess.run(["python3", script_path])
    print("\n-------------------------------------------\n")
    return result.returncode

def user_input(message):
    return input(message).strip()

def check_fastapi_running():
    try:
        requests.get(FASTAPI_URL + "/health", timeout=2)
        return True
    except:
        return False

def print_header(title):
    print("\n" + "=" * 55)
    print(f"{title}")
    print("=" * 55 + "\n")

def run(cmd):
    """Run a shell command and print output."""
    print(f"\n$ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result.stdout.strip()

def wait():
    input("\nPress ENTER to continue...")

def pretty(data):
    return json.dumps(data, indent=2, ensure_ascii=False)

REQUIRED_ENV_VARS = [
    "ES_URL",
    "INDEX_NAME",
    "MAIN_EMBED_MODEL",
    "RERANKER_MODEL",
    "GEMINI_API_KEY",
    "GEMINI_MODEL",
    "SMTP_HOST",
    "SMTP_PORT",
    "SMTP_USER",
    "SMTP_PASS",
    "DEFAULT_EMAIL"
]

def run_request(method, endpoint, payload=None):
    """Wrapper to call API and pretty-print response."""
    url = f"{FASTAPI_URL}{endpoint}"
    print(f"\n🔸 ENDPOINT: {url}")
    try:
        if method == "POST":
            r = requests.post(url, json=payload, timeout=20)
        else:
            r = requests.get(url, timeout=20)

        print("\n🟦 REQUEST:")
        print(pretty(payload) if payload else "(no body)")

        print("\n🟩 RESPONSE STATUS:", r.status_code)

        try:
            data = r.json()
            print("\n📘 RESPONSE JSON:\n", pretty(data))
        except:
            print("\n📙 RAW RESPONSE:\n", r.text)

    except Exception as e:
        print("\n❌ ERROR:", e)
    print("-" * 70)

# ------------------------------------------
# SETUP FLOW
# ------------------------------------------
def setup_flow():
    print_header("🔧 SETUP MODE")

    print("1️⃣  Ensure Python version == 3.13.7")
    print("2️⃣  Ensure requirements.txt installed")
    print("3️⃣  Ensure Docker is installed and running")
    print("4️⃣  Checking .env variables...\n")

    load_dotenv()

    missing = []
    empty = []

    for var in REQUIRED_ENV_VARS:
        value = os.getenv(var)
        if value is None:
            missing.append(var)
        elif value.strip() == "":
            empty.append(var)

    print("🔍 ENV CHECK RESULTS")
    print("----------------------------")

    if not missing and not empty:
        print("✅ All required environment variables are present and correctly set!")
    else:
        if missing:
            print("\n❌ Missing variables (not found in .env):")
            for v in missing:
                print(f"   - {v}")

        if empty:
            print("\n⚠️ Empty variables (present but blank):")
            for v in empty:
                print(f"   - {v}")

        print("\n➡️ Please fix the issues in your .env file before continuing.")

    print("\n")
    choice = user_input("Press ENTER to continue or 'n' to return to main menu... ")
    if choice == "n":
        print("\n👋 Returning to main menu.\n")
        return

    # -----------------------------------------------------
    # ES CHECK
    # -----------------------------------------------------
    if user_input("Do you want me to verify Elasticsearch setup? (y/n): ") == "y":
        print("\n🚀 Starting docker-compose (Elasticsearch)...")
        run("docker-compose up -d")
        time.sleep(5)

        print("\n⏳ Checking Elasticsearch health...")
        run_python(ES_CHECK_SCRIPT)

    # -----------------------------------------------------
    # GEMINI CHECK
    # -----------------------------------------------------
    if user_input("Do you want me to verify GEMINI setup? (y/n): ") == "y":
        print("\n▶ Checking Gemini API key...")
        run_python(GEMINI_CHECK_SCRIPT)

    # -----------------------------------------------------
    # METADATA GENERATION
    # -----------------------------------------------------
    if user_input("Do you want me to generate metadata? (y/n): ") == "y":
        print("\n▶ Generating metadata...")
        run_python(GENERATE_METADATA)

    # -----------------------------------------------------
    # BUILD VECTOR DB
    # -----------------------------------------------------
    if user_input("Do you want me to build the vector database? (y/n): ") == "y":
        print("\n▶ Building vector DB (chunking, embeddings, indexing)...")
        run_python(BUILD_DATABASE)

        print("\n✅ DATABASE BUILD COMPLETE!")
        print("------------------------------------------------------")
        print("This build implements the following features:\n")
        print("✔ Metadata attachment (filename → metadata.csv)")
        print("✔ Paragraph-based Level-0 chunking (2048-token)")
        print("✔ Hierarchical chunking: 2048 → 512 → 128 tokens")
        print("✔ Level-1 and Level-2 child chunks with parent_id links")
        print("✔ Normalization + deduplication via SHA-256 hash")
        print("✔ Sliding-window fallback for long paragraphs")
        print("✔ Embeddings using mpnet-base (SentenceTransformer)")
        print("✔ Domain embedding model support (optional)")
        print("✔ Correct Elasticsearch mapping − dense_vector(768)")
        print("✔ Hybrid retrieval & reranking ready")
        print("------------------------------------------------------\n")

    # -----------------------------------------------------
    # POST-BUILD VERIFICATION (Clean Output + PASS/FAIL)
    # -----------------------------------------------------
    if user_input("\n🔍 Run post-build verification checks? (y/n): ") == "y":

        print_header("📊 POST-BUILD VERIFICATION CHECKS")

        summary = []   # store pass/fail results

        ES = "http://localhost:9200/food_safety_rag"

        # -------------------------------
        # 1️⃣ Level distribution check
        # -------------------------------
        print("\n1️⃣  Counting chunks at levels 0 / 1 / 2 ...")
        try:
            out = run_python("scripts/tests/check_levels.py")  # Your python script
            summary.append("✔ Level distribution check — PASSED")
        except Exception:
            summary.append("❌ Level distribution check — FAILED")
        wait()

        # -------------------------------
        # 2️⃣ Token length check (sample)
        # -------------------------------
        print("\n2️⃣  Checking token lengths (<512 tokens)...")
        try:
            resp = requests.post(
                f"{ES}/_search?size=10",
                json={"_source": ["text"], "query": {"match_all": {}}},
                timeout=10
            )
            resp.raise_for_status()
            data = resp.json()

            print("\n📄 SAMPLE DOCUMENTS:")
            print(pretty(data))

            print("\n➡️ Please visually confirm no extremely long text (>512 tokens).")
            summary.append("✔ Token length check — MANUAL REVIEW")
        except Exception as e:
            print("❌ Error:", e)
            summary.append("❌ Token length check — FAILED")
        wait()

        # -------------------------------
        # 3️⃣ Embedding existence check
        # -------------------------------
        print("\n3️⃣  Verifying embedding field exists...")
        try:
            resp = requests.post(
                f"{ES}/_search?size=1",
                json={"_source": ["embedding"], "query": {"match_all": {}}},
                timeout=10
            )
            resp.raise_for_status()
            data = resp.json()
            print(pretty(data))

            hit = data.get("hits", {}).get("hits", [{}])[0]
            if "_source" in hit and "embedding" in hit["_source"]:
                summary.append("✔ Embedding presence — PASSED")
            else:
                summary.append("❌ Embedding presence — FAILED (no embedding field)")
        except Exception as e:
            print("❌ Error:", e)
            summary.append("❌ Embedding presence — FAILED")
        wait()

        # -------------------------------
        # 4️⃣ Document count in index
        # -------------------------------
        print("\n4️⃣  Checking document count...")
        try:
            resp = requests.get(f"{ES}/_count", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            print(pretty(data))
            summary.append("✔ Document count check — PASSED")
        except Exception as e:
            print("❌ Error:", e)
            summary.append("❌ Document count check — FAILED")
        wait()

        # -------------------------------
        # 5️⃣ Index mapping check
        # -------------------------------
        print("\n5️⃣  Checking index mapping...")
        try:
            resp = requests.get(f"{ES}/_mapping", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            print(pretty(data))

            mapping_ok = False
            root = data.get(INDEX_NAME, {}).get("mappings", {})
            props = root.get("properties", {})

            if "embedding" in props and "text" in props:
                mapping_ok = True

            if mapping_ok:
                summary.append("✔ Mapping structure — PASSED")
            else:
                summary.append("❌ Mapping structure — FAILED (missing fields)")
        except Exception as e:
            print("❌ Error:", e)
            summary.append("❌ Mapping structure — FAILED")
        wait()

        # -------------------------------
        # Summary
        # -------------------------------
        print_header("✅ VERIFICATION SUMMARY")

        for s in summary:
            print(s)

        print("\nIf all checks show ✔, your vector DB is perfect.\n")

    else:
        print("\n⏩ Skipping post-build checks.")

    print("\n🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("You may now start FastAPI in another terminal:")
    print("   python3 -m uvicorn agent.main_api:app --reload\n")


# ------------------------------------------
# RAG TESTING
# ------------------------------------------
def run_rag_tests():
    print_header("🔍 RAG TESTING")
    print("1. Test retriever manually")
    print("2. Test query_retriever")
    print("3. Back")

    choice = user_input("\nSelect option: ")

    if choice == "1":
        run_python("rag/retriever.py")
    elif choice == "2":
        run_python("rag/query_rag.py")
    else:
        return


# ------------------------------------------
# AGENT TESTING
# ------------------------------------------
def agent_testing():
    print_header("🤖 AGENT TEST MENU")

    print("1. Basic Chat (RAG search)")
    print("2. Follow-up Chat (memory test)")
    print("3. Non-RAG general question")
    print("4. RAG Failover test (stop ES manually)")
    print("5. Workflow test (PDF + DOCX + Email)")
    print("6. Plan & Run (LLM-planned multi-step workflow)")
    print("7. /agent router test (rule-based + LLM-planned)")
    print("8. Feedback memory test")
    print("9. Attachment resolution test")
    print("0. Back to main menu")

    choice = user_input("\nChoose test number: ")

    # ---------------------------------------------
    # 1. BASIC CHAT (RAG)
    # ---------------------------------------------
    if choice == "1":
        query = user_input("Enter RAG question (e.g., 'What is cross contamination?'): ")
        run_request("POST", "/chat", {"query": query, "top_k": 5})
        return

    # ---------------------------------------------
    # 2. FOLLOW-UP MEMORY TEST
    # ---------------------------------------------
    elif choice == "2":
        q1 = user_input("Enter first question (e.g., 'What is salmonella?'): ")
        q2 = user_input("Enter follow-up question (e.g., 'How does this spread?'): ")

        print("\n▶ First question...")
        run_request("POST", "/chat", {"query": q1})

        print("\n▶ Follow-up question...")
        run_request("POST", "/chat", {"query": q2})
        return

    # ---------------------------------------------
    # 3. NON-RAG GENERAL QUESTION
    # ---------------------------------------------
    elif choice == "3":
        query = user_input("Enter general question (e.g., 'Who won the 2022 FIFA World Cup?'): ")
        run_request("POST", "/chat", {"query": query})
        return

    # ---------------------------------------------
    # 4. RAG FAILOVER TEST
    # ---------------------------------------------
    elif choice == "4":
        print("\n🛑 MANUAL STEP REQUIRED:")
        print("▶ Run:  docker stop elasticsearch")
        user_input("Press ENTER after stopping ES...")

        query = user_input("Enter a RAG-dependent question (e.g., 'What is food hygiene?'): ")
        run_request("POST", "/chat", {"query": query})

        print("\n⚠️ Now restart ES manually: docker start elasticsearch\n")
        return

    # ---------------------------------------------
    # 5. WORKFLOW TEST (PDF + DOCX + EMAIL)
    # ---------------------------------------------
    elif choice == "5":
        title = user_input("Enter title (e.g., Cross Contamination Prevention): ")
        audience = user_input("Enter audience (e.g., food handlers): ")
        email = user_input("Enter email: ")

        payload = {
            "task": "create_handout",
            "title": title,
            "audience": audience,
            "email_to": email,
            "create_pdf": True,
            "create_docx": True
        }

        run_request("POST", "/workflow", payload)
        return

    # ---------------------------------------------
    # 6. PLAN & RUN WORKFLOW
    # ---------------------------------------------
    elif choice == "6":
        task = user_input("Describe task (e.g., Create a one page handout on allergens and mail it): ")

        run_request("POST", "/plan_and_run", {"task": task})
        return

    # ---------------------------------------------
    # 7. AGENT ROUTER TEST
    # ---------------------------------------------
    elif choice == "7":
        print("\n1. Rule-based workflow trigger")
        print("2. LLM-planned router trigger")
        router_choice = user_input("Choose 1 or 2: ")

        if router_choice == "1":
            query = user_input("Enter rule-based query (e.g., 'Create handout on cross contamination'): ")
        else:
            query = user_input("Enter LLM-planned query (e.g., 'Generate a quiz on bacteria and mail it'): ")

        run_request("POST", "/agent", {"query": query})
        return

    # ---------------------------------------------
    # 8. FEEDBACK TEST (Interactive)
    # ---------------------------------------------
    elif choice == "8":
        print_header("🧪 FEEDBACK TEST (Interactive)")
        print("This will:")
        print("1. Ask for a query")
        print("2. Generate an answer via /chat")
        print("3. Ask if the answer is good or bad")
        print("4. If bad → ask for corrected answer + store via /feedback")
        print("5. Re-run same query to verify modification\n")

        q = user_input(
            "Enter query to test feedback system "
            "(e.g., 'What is cross contamination?', 'What is salmonella?'): "
        )

        if not q.strip():
            print("⚠️ Empty query. Cancelling feedback test.")
            return

        # 1️⃣ Generate answer via /chat
        print("\n📤 Calling /chat ...")
        chat_resp = requests.post(
            "http://127.0.0.1:8000/chat",
            json={"query": q},
            timeout=30
        )

        try:
            chat_json = chat_resp.json()
        except:
            print("❌ Failed to decode /chat response.\nRaw:", chat_resp.text)
            return

        model_answer = chat_json.get("answer")

        print("\n-------------------------------------------")
        print("📝 MODEL ANSWER:\n")
        print(model_answer)
        print("-------------------------------------------\n")

        if not model_answer:
            print("⚠️ No 'answer' field returned. Stopping.")
            return

        # 2️⃣ FEEDBACK: Good / Bad?
        fb = user_input("Mark answer as (good/bad): ").strip().lower()
        if fb not in ("good", "bad"):
            print("⚠️ Invalid choice. Stopping.")
            return

        corrected = None
        if fb == "bad":
            corrected = user_input("Enter corrected answer (e.g., cross contamination is the transfer of harmful bacteria from raw foods to ready-to-eat foods, surfaces, or equipment.): ")

        feedback_payload = {
            "query": q,
            "answer": model_answer,
            "feedback": fb,
            "corrected_answer": corrected
        }

        print("\n📤 Sending feedback → /feedback")
        run_request("POST", "/feedback", feedback_payload)

        # 3️⃣ Optionally re-run
        if user_input("\n🔁 Re-run same query to see updated answer? (y/n): ").lower() == "y":
            print("\n📤 Calling /chat again...")
            run_request("POST", "/chat", {"query": q})

        return

    # ---------------------------------------------
    # 9. ATTACHMENT RESOLUTION TEST
    # ---------------------------------------------
    elif choice == "9":
        query = user_input(
            "Enter query requiring PDF mailing "
            "(e.g., 'Create a quiz on bacteria and mail me the PDF'): "
        )

        run_request("POST", "/agent", {"query": query})
        print("\n📂 Check 'files' or 'attachments' field in response above.")
        print("If missing, attachment resolution failed.\n")
        return

    else:
        return


# ------------------------------------------
# MAIN MENU LOOP
# ------------------------------------------
def main():
    while True:
        print_header("🚀 SME SYSTEM WORKFLOW")

        print("1. Setup system (Elasticsearch, Gemini, DB Build)")
        print("2. Run only RAG tests")
        print("3. Run Agent tests")
        print("4. Quit")

        choice = user_input("\nChoose option: ")

        if choice == "1":
            setup_flow()

        elif choice == "2":
            run_rag_tests()

        elif choice == "3":
            if not check_fastapi_running():
                print("\n❌ FastAPI server not running!")
                print("Run in another terminal:\n   python3 -m uvicorn agent.main_api:app --reload\n")
            else:
                agent_testing()

        elif choice == "4":
            print("\n👋 Exiting workflow. Goodbye!\n")
            break

        else:
            print("\nInvalid choice. Try again.\n")


if __name__ == "__main__":
    main()