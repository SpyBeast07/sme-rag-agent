# retriever.py
from typing import List, Dict, Any, Optional
import os
import math
import numpy as np
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm

load_dotenv()

ES_URL = os.getenv("ES_URL")
INDEX_NAME = os.getenv("INDEX_NAME")
MAIN_EMBED_MODEL = os.getenv("MAIN_EMBED_MODEL")
RERANKER_MODEL = os.getenv("RERANKER_MODEL")

# Helpful utility
def safe_min_max_norm(scores: List[float]) -> List[float]:
    if not scores:
        return []
    smin = float(min(scores))
    smax = float(max(scores))
    if math.isclose(smin, smax):
        # all same -> return 1.0 for each
        return [1.0 for _ in scores]
    return [(s - smin) / (smax - smin) for s in scores]


class RAGRetriever:
    def __init__(
        self,
        es_url: str = ES_URL,
        index_name: str = INDEX_NAME,
        embed_model_name: str = MAIN_EMBED_MODEL,
        reranker_model_name: Optional[str] = RERANKER_MODEL,
        device: Optional[str] = None,
    ):
        # ES client (hosts must be a list)
        self.es = Elasticsearch(hosts=[es_url])
        self.index = index_name

        # Embedding model
        print(f"Loading embed model: {embed_model_name} ...")
        self.embedder = SentenceTransformer(embed_model_name)
        # Model max tokens (approx) - use tokenizer length if available
        try:
            from transformers import AutoTokenizer
            tk = AutoTokenizer.from_pretrained(embed_model_name)
            self.model_max_tokens = getattr(tk, "model_max_length", 512)
        except Exception:
            self.model_max_tokens = 512

        # Reranker (optional)
        self.reranker_enabled = bool(reranker_model_name)
        if self.reranker_enabled:
            print(f"Loading reranker model: {reranker_model_name} ...")
            self.rr_tokenizer = AutoTokenizer.from_pretrained(reranker_model_name)
            self.rr_model = AutoModelForSequenceClassification.from_pretrained(reranker_model_name)
            # choose device
            if device:
                self.rr_device = device
            else:
                self.rr_device = "cuda" if torch.cuda.is_available() else "cpu"
            self.rr_model.to(self.rr_device)
            self.rr_model.eval()
        print("RAGRetriever initialized.")

    def embed(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        """
        Returns numpy array (N x D) of embeddings (float32).
        """
        emb = self.embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=False)
        return np.asarray(emb).astype(np.float32)

    def vector_search(self, query_embedding: List[float], top_k: int = 100) -> List[dict]:
        """
        Runs ES vector search (script_score using cosineSimilarity on 'embedding' field).
        Returns list of hits (ES hit objects)
        """
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
        res = self.es.search(index=self.index, body=body)
        return res.get("hits", {}).get("hits", [])

    def bm25_search(self, query: str, top_k: int = 100) -> List[dict]:
        """
        BM25 search using multi_match on 'text' field (you can customize fields).
        """
        body = {
            "size": top_k,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["text"],
                    "type": "best_fields"
                }
            }
        }
        res = self.es.search(index=self.index, body=body)
        return res.get("hits", {}).get("hits", [])

    def _apply_reranker(self, query: str, candidates: List[dict], top_k: int) -> List[dict]:
        """
        Rerank candidates using pairwise reranker (query, doc_text).
        candidates: list of ES hit objects
        returns top_k hits sorted by reranker score
        """
        if not self.reranker_enabled or not candidates:
            return candidates[:top_k]

        texts = [c["_source"]["text"] for c in candidates]
        pairs_q = [query] * len(texts)
        # batching for reranker
        batch = 16
        scores = []
        for i in range(0, len(texts), batch):
            batch_q = pairs_q[i:i+batch]
            batch_docs = texts[i:i+batch]
            inputs = self.rr_tokenizer(batch_q, batch_docs, padding=True, truncation=True, return_tensors="pt").to(self.rr_device)
            with torch.no_grad():
                out = self.rr_model(**inputs)
                # logits shape: (batch, num_labels) or (batch, 1)
                logits = out.logits
                # if multi-dim, reduce to single score (take first logit or mean)
                if logits.ndim == 2 and logits.shape[1] >= 1:
                    batch_scores = logits[:, 0].detach().cpu().tolist()
                else:
                    batch_scores = logits.squeeze(-1).detach().cpu().tolist()
            scores.extend(batch_scores)

        # attach reranker scores
        for c, s in zip(candidates, scores):
            c["_rerank_score"] = float(s)

        # sort by reranker score desc
        ranked = sorted(candidates, key=lambda x: x.get("_rerank_score", 0.0), reverse=True)
        return ranked[:top_k]

    def hybrid_search(self,
                        query: str,
                        top_k: int = 5,
                        top_k_vector: int = 200,
                        top_k_bm25: int = 200,
                        alpha: float = 0.6,
                        use_bm25: bool = True,
                        use_reranker: bool = True) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval:
        1) Run vector search to get top_k_vector candidates (gives vector_score)
        2) Run BM25 to get top_k_bm25 candidates (gives bm25_score)
        3) Merge candidates, normalize both scores, compute combined = alpha*vec + (1-alpha)*bm25
        4) Return top_k docs (optionally reranked)
        """

        # 1) get query embedding
        q_emb = self.embed([query])[0].tolist()

        # 2) run vector search
        vec_hits = self.vector_search(q_emb, top_k=top_k_vector)

        # 3) run bm25 search
        bm_hits = []
        if use_bm25:
            bm_hits = self.bm25_search(query, top_k=top_k_bm25)

        # collect unique candidates keyed by _id
        candidates = {}
        for h in vec_hits:
            docid = h["_id"]
            # script_score returned value is in h["_score"] (cosine+1)
            candidates[docid] = {
                "hit": h,
                "vec_score": float(h.get("_score", 0.0) - 1.0)  # subtract 1 to get raw cosine in [-1,1]
            }

        for h in bm_hits:
            docid = h["_id"]
            if docid not in candidates:
                candidates[docid] = {"hit": h, "vec_score": 0.0}
            candidates[docid]["bm25_score"] = float(h.get("_score", 0.0))

        # convert to list and ensure fields exist
        cand_list = []
        for k, v in candidates.items():
            hit = v["hit"]
            vec_s = v.get("vec_score", 0.0)
            bm_s = v.get("bm25_score", 0.0)
            cand_list.append({
                "_id": k,
                "hit": hit,
                "vec_score": vec_s,
                "bm25_score": bm_s
            })

        if not cand_list:
            return []

        # normalize scores (vec may be in [-1,1], bm25 >=0)
        vec_scores = [c["vec_score"] for c in cand_list]
        bm_scores = [c["bm25_score"] for c in cand_list]

        vec_norm = safe_min_max_norm(vec_scores)
        bm_norm = safe_min_max_norm(bm_scores)

        for i, c in enumerate(cand_list):
            c["vec_norm"] = vec_norm[i]
            c["bm_norm"] = bm_norm[i]
            # combined score
            c["combined_score"] = alpha * c["vec_norm"] + (1.0 - alpha) * c["bm_norm"]

        # sort by combined score desc
        cand_list = sorted(cand_list, key=lambda x: x["combined_score"], reverse=True)

        # prepare results (take top_k and optionally rerank)
        top_candidates = [c["hit"] for c in cand_list[: max(top_k * 3, top_k) ]]  # keep some for reranker

        if use_reranker and self.reranker_enabled:
            reranked = self._apply_reranker(query, top_candidates, top_k=top_k)
            # return structured dicts
            results = []
            for r in reranked:
                src = r["_source"]
                results.append({
                    "id": r["_id"],
                    "text": src.get("text", ""),
                    "metadata": src.get("metadata", {}),
                    "bm25_score": float(r.get("_score", 0.0)),
                    "rerank_score": float(r.get("_rerank_score", 0.0)) if r.get("_rerank_score") else None
                })
            return results[:top_k]
        else:
            results = []
            for r in top_candidates[:top_k]:
                src = r["_source"]
                results.append({
                    "id": r["_id"],
                    "text": src.get("text", ""),
                    "metadata": src.get("metadata", {}),
                    "bm25_score": float(r.get("_score", 0.0))
                })
            return results

    def get_relevant_texts(self, query: str, top_k: int = 5, **kwargs) -> List[str]:
        """
        Convenience wrapper that returns list of strings (texts) for the top_k results.
        Pass kwargs to hybrid_search (alpha, use_bm25, use_reranker, etc.)
        """
        docs = self.hybrid_search(query, top_k=top_k, **kwargs)
        return [d["text"] for d in docs]


# -----------------------
# Quick test when run directly
# -----------------------
if __name__ == "__main__":
    r = RAGRetriever()
    q = "What is the safe internal temperature for cooking chicken?"
    results = r.hybrid_search(q, top_k=5, top_k_vector=200, top_k_bm25=400, alpha=0.6, use_bm25=True, use_reranker=True)
    print("=== Retrieved Results ===")
    for i, doc in enumerate(results, 1):
        print(f"{i}. id={doc['id']} bm25={doc.get('bm25_score')} meta_source={doc['metadata'].get('source')} top200chars:\n{doc['text'][:200]}\n")


# What is implemented:
# - hybrid retrieval (BM25 + vector search against your Elasticsearch index)
# - optional BGE reranker (model name controlled by RERANKER_MODEL env var)
# - careful batching, device handling, and score normalization
# - a single class RAGRetriever with a simple API: get_relevant_texts(query, top_k=5, use_bm25=True, use_reranker=True) returning rich result dicts
# - It uses ES script_score cosineSimilarity query for vector retrieval
# - BM25 uses multi_match on text



# Comparision of approaches:

# Vector + BM25 gives you:
# - fast
# - approximate relevance
# - based on keywords + latent similarity

# Reranker gives you:
# - slower
# - deeper semantic filtering
# - more accurate top results