# #!/bin/bash

# # Exit immediately if a command exits with a non-zero status.
# set -e

# echo "--- RAG QUERY SCRIPT (RUN ANYTIME) ---"

# # --- Step 1: Activate Virtual Environment ---
# echo "[Step 1/3] Activating Python virtual environment..."
# source venv/bin/activate
# echo "Virtual environment activated."

# # --- Step 2: Start Backend Services ---
# echo -e "\n[Step 2/3] Starting backend services (Elasticsearch)..."
# # This is safe to run even if they're already running
# docker-compose up -d
# echo "Services started. Waiting 10 seconds for Elasticsearch to be ready..."
# sleep 10

# # --- Step 3: Run a Query ---
# # This uses the first argument passed to the script as the query.
# # If no argument is given, it uses a default question.
# DEFAULT_QUERY="What is the safe internal temperature for cooking chicken?"
# QUERY="${1:-$DEFAULT_QUERY}"

# echo -e "\n[Step 3/3] Running retriever with query: \"$QUERY\""
# python3 scripts/retriever.py --query="$QUERY"

# echo -e "\n--- Query complete! ---"


#!/bin/bash
set -e

echo "--- RAG QUERY SCRIPT (RUN ANYTIME) ---"

# --- Force correct Python version (Python 3.12 from your Anaconda env) ---
PYTHON_BIN="/Users/pratyushamitra/anaconda3/envs/har/bin/python3.12"

# --- Step 1 / 3 : Activate Virtual Environment ---
echo "[Step 1/3] Activating Python virtual environment..."
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated."
    echo "Using Python: $($PYTHON_BIN --version)"
else
    echo "⚠️  No 'venv' folder found. Using Python binary: $PYTHON_BIN"
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

echo "⏳ Waiting 10 seconds for Elasticsearch to be ready..."
sleep 10

# --- Step 3 / 3 : Run a Query ---
DEFAULT_QUERY="What is the safe internal temperature for cooking chicken?"
QUERY="${1:-$DEFAULT_QUERY}"

echo -e "\n[Step 3/3] Running retriever with query: \"$QUERY\""
$PYTHON_BIN scripts/retriever.py --query="$QUERY"

echo -e "\n🎯 Query complete!"
