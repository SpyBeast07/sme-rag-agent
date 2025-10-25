from typing import List
import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

load_dotenv()
ES_URL = os.getenv("ES_URL")
INDEX_NAME = os.getenv("INDEX_NAME")
MAIN_EMBED_MODEL = os.getenv("MAIN_EMBED_MODEL")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", None)  # optional

# Initialize Elasticsearch client
es = Elasticsearch(ES_URL)

# Initialize embedding model
embed_model = SentenceTransformer(MAIN_EMBED_MODEL)

# Initialize reranker (optional)
reranker_enabled = bool(RERANKER_MODEL)
if reranker_enabled:
    reranker_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL)
    reranker_model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL)
    reranker_model.eval()

# Step 1: Get Query Embedding
def embed_query(query: str):
    return embed_model.encode([query], convert_to_tensor=True)[0].tolist()

# Step 2: Search Elasticsearch Using Cosine Similarity
def search_es(query_embedding, top_k=5):
    body = {
        "size": top_k,
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                    "params": {"query_vector": query_embedding}
                }
            }
        }
    }
    response = es.search(index=INDEX_NAME, body=body)
    hits = response['hits']['hits']
    return hits

# Step 3: Optional Reranking
def rerank(query: str, hits: List[dict]):
    if not reranker_enabled:
        return hits
    
    texts = [hit["_source"]["text"] for hit in hits]
    inputs = reranker_tokenizer([query]*len(texts), texts, padding=True, truncation=True, return_tensors="pt")
    
    with torch.no_grad():
        scores = reranker_model(**inputs).logits.squeeze(-1)
    
    # Sort hits based on scores
    sorted_indices = torch.argsort(scores, descending=True)
    reranked_hits = [hits[i] for i in sorted_indices]
    return reranked_hits

# Step 4: Combine Everything in get_relevant_texts
def get_relevant_texts(query: str, top_k: int = 5) -> List[str]:
    """
    Retrieves top-K relevant text chunks from Elasticsearch for a given query.
    Optionally applies reranking.
    """
    query_embedding = embed_query(query)
    hits = search_es(query_embedding, top_k=top_k)
    hits = rerank(query, hits)  # optional
    
    # Return list of text chunks
    return [hit["_source"]["text"] for hit in hits]

# Step 5: Quick Test
if __name__ == "__main__":
    query = "What is the safe internal temperature for cooking chicken?"
    chunks = get_relevant_texts(query, top_k=5)
    
    print("=== Retrieved Chunks ===")
    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. {chunk[:200]}...")  # print first 200 chars