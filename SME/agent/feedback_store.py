import json
from pathlib import Path

FEEDBACK_FILE = Path(__file__).resolve().parents[1] / "feedback.json"

def load_feedback():
    if FEEDBACK_FILE.exists():
        return json.loads(FEEDBACK_FILE.read_text())
    return []

def save_feedback(data):
    FEEDBACK_FILE.write_text(json.dumps(data, indent=2))

def add_feedback(item):
    data = load_feedback()
    data.append(item)
    save_feedback(data)

# Usage:

# Ask a question normally
# curl -X POST http://127.0.0.1:8000/chat \
#   -H "Content-Type: application/json" \
#   -d '{"query": "What is cross contamination?"}'

# Submit BAD feedback
# curl -X POST http://127.0.0.1:8000/feedback \
#   -H "Content-Type: application/json" \
#   -d '{
#         "query": "What is cross contamination?",
#         "answer": "THE_OLD_ANSWER_HERE",
#         "feedback": "bad",
#         "corrected_answer": "Cross-contamination means transfer of harmful microbes from raw to ready-to-eat foods."
#       }'

# If not worked

# curl -X POST http://127.0.0.1:8000/feedback \
#   -H "Content-Type: application/json" \
#   -d @- << 'EOF'
# {
#   "query": "What is cross contamination?",
#   "answer": "THE_OLD_ANSWER_HERE",
#   "feedback": "bad",
#   "corrected_answer": "Cross-contamination means the transfer of harmful microbes from raw foods to ready-to-eat foods, usually via hands, utensils, or cutting boards."
# }
# EOF

# Ask the same question again
# curl -X POST http://127.0.0.1:8000/chat \
#   -H "Content-Type: application/json" \
#   -d '{"query": "What is cross contamination?"}'

# Submit GOOD feedback
# curl -X POST http://127.0.0.1:8000/feedback \
#   -H "Content-Type: application/json" \
#   -d '{"query":"What is cross contamination?","answer":"THE_OLD_ANSWER_HERE","feedback":"good"}'
