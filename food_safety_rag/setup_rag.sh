#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- RAG Pipeline Setup & Rerun Script ---"

# --- Step 1: Activate Virtual Environment ---
echo "[Step 1/5] Activating Python virtual environment..."
if [ ! -d "venv" ]; then
    echo "Error: 'venv' directory not found. Please create it first using 'python3 -m venv venv'"
    exit 1
fi
source venv/bin/activate
echo "Virtual environment activated."
echo "INFO: Make sure you have installed dependencies with 'pip install -r requirements.txt'"


# --- Step 2: Start Backend Services with Docker ---
echo -e "\n[Step 2/5] Starting backend services (Elasticsearch)..."
# The '-d' flag runs the containers in the background
docker-compose up -d
echo "Services started via docker-compose."
echo "Waiting 15 seconds for Elasticsearch to initialize..."
sleep 15


# --- Step 3: Build the RAG Knowledge Base ---
echo -e "\n[Step 3/5] Building the RAG knowledge base..."
echo "========================================================================"
echo "IMPORTANT: This next step can take a very long time (45+ minutes) if"
echo "           your data isn't already indexed in Elasticsearch."
echo "           If you know your data is already indexed, you can safely"
echo "           stop this script (Ctrl+C) and run the next steps manually,"
echo "           OR comment out the 'python3 scripts/build_database.py' line."
echo "========================================================================"
# The script will pause here for 5 seconds to give you time to read the message
sleep 5
python3 scripts/build_database.py
echo "Database build process complete."


# --- Step 4: Verify the Index ---
echo -e "\n[Step 4/5] Verifying the index status..."
python3 scripts/es_Check.py


# --- Step 5: Run a Test Query ---
echo -e "\n[Step 5/5] Running a test query with the retriever..."
python3 scripts/retriever.py --query "What is the safe temperature for cooking chicken?"


echo -e "\n--- Script finished successfully! Your RAG pipeline is ready. ---"