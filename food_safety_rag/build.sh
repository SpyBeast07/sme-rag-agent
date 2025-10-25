# #!/bin/bash

# # Exit immediately if a command exits with a non-zero status.
# set -e

# echo "--- RAG KNOWLEDGE BASE BUILD SCRIPT (RUN ONCE) ---"

# # --- Step 1: Activate Virtual Environment ---
# echo "[Step 1/3] Activating Python virtual environment..."
# source venv/bin/activate
# echo "Virtual environment activated."

# # --- Step 2: Start Backend Services ---
# echo -e "\n[Step 2/3] Starting backend services (Elasticsearch)..."
# docker-compose up -d
# echo "Services started. Waiting 15 seconds for Elasticsearch to initialize..."
# sleep 15

# # --- Step 3: Build the Knowledge Base ---
# echo -e "\n[Step 3/3] Building the knowledge base. This will take a long time..."
# python3 scripts/build_database.py
# echo "Knowledge base has been built and indexed successfully."

# # --- Verification ---
# echo -e "\n--- Verifying Index Status ---"
# python3 scripts/es_Check.py

# echo -e "\n--- Build Complete! You can now use run.sh to query the system. ---"


#!/bin/bash
set -e

echo "--- RAG KNOWLEDGE BASE BUILD SCRIPT (RUN ONCE) ---"

# --- Force correct Python version (Python 3.12 from your Anaconda env) ---
PYTHON_BIN="/Users/pratyushamitra/anaconda3/envs/har/bin/python3.12"

# --- Step 1 / 3 : Activate Virtual Environment ---
echo "[Step 1/3] Activating Python virtual environment..."
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated."
    echo "Using Python: $($PYTHON_BIN --version)"
else
    echo "❌ Virtual environment not found!"
    echo "Please create one with:  $PYTHON_BIN -m venv venv && source venv/bin/activate"
    exit 1
fi

# --- Step 2 / 3 : Start Backend Services (Elasticsearch) ---
echo -e "\n[Step 2/3] Starting backend services (Elasticsearch)..."
if command -v docker >/dev/null 2>&1; then
    echo "🐳 Docker detected — starting containers..."
    docker compose up -d
else
    echo "❌ Docker not found. Please install Docker Desktop for macOS."
    exit 1
fi

echo "⏳ Waiting 15 seconds for Elasticsearch to initialize..."
sleep 15

# --- Step 3 / 3 : Build the Knowledge Base ---
echo -e "\n[Step 3/3] Building the knowledge base (this may take a while)..."
$PYTHON_BIN scripts/build_database.py
echo "✅ Knowledge base has been built and indexed successfully."

# --- Verification ---
echo -e "\n--- Verifying Index Status ---"
$PYTHON_BIN scripts/es_Check.py

echo -e "\n🎯 Build Complete! You can now use run.sh to query the system."
