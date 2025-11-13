import os
import pandas as pd
from datetime import datetime

# Path to your data directory
DATA_DIR = "../data"

metadata_records = []

# Simple helper function to guess metadata from path
def infer_metadata(filepath):
    filename = os.path.basename(filepath)
    folder = os.path.basename(os.path.dirname(filepath))

    # Infer source
    if "FSSAI" in folder or "Comp" in filename:
        source = "FSSAI"
    elif "WHO" in folder or "978924" in filename:
        source = "WHO"
    elif "nutrition" in filename.lower() or "homescience" in filename.lower():
        source = "NCERT"
    else:
        source = "Other"

    # Infer category
    if "Regulations" in filename or "Notification" in filename:
        category = "Regulation"
    elif "Manual" in filename:
        category = "Manual"
    elif "nutrition" in filename.lower() or "homescience" in filename.lower():
        category = "Textbook"
    else:
        category = "Misc"

    # Infer topic
    topic = "general"
    if "Additives" in filename:
        topic = "Food Additives"
    elif "Labelling" in filename or "Packaging" in filename:
        topic = "Labelling & Packaging"
    elif "HACCP" in filename:
        topic = "Food Safety Management"
    elif "Nutrition" in filename:
        topic = "Nutrition"
    elif "Alcoholic" in filename:
        topic = "Beverages"

    # Extract year if available
    year = None
    for token in filename.split("_"):
        if token.isdigit() and len(token) == 4:
            year = int(token)
            break

    return {
        "filename": filename,
        "source": source,
        "category": category,
        "topic": topic,
        "year": year,
        "path": filepath
    }


# Walk through all files in data/
for root, _, files in os.walk(DATA_DIR):
    for file in files:
        if file.endswith((".pdf", ".txt", ".docx")):
            filepath = os.path.join(root, file)
            metadata_records.append(infer_metadata(filepath))

# Save to CSV
df = pd.DataFrame(metadata_records)
df["timestamp"] = datetime.now().isoformat(timespec="seconds")
df.to_csv("reports/outputs/metadata.csv", index=False)

print(f"✅ Metadata generated for {len(df)} files!")
print(df.head())