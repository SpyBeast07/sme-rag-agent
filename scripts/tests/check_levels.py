from elasticsearch import Elasticsearch
import os
from dotenv import load_dotenv

load_dotenv()

es = Elasticsearch(os.getenv("ES_URL"))
INDEX = os.getenv("INDEX_NAME")

for level in [0, 1, 2]:
    q = {
        "query": { "term": { "level": level } }
    }
    res = es.count(index=INDEX, body=q)
    print(f"Level {level}: {res['count']} chunks")