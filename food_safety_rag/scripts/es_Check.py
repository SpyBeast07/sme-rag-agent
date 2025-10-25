from elasticsearch import Elasticsearch
import os
from dotenv import load_dotenv

load_dotenv()
ES_URL = os.getenv("ES_URL")
INDEX_NAME = os.getenv("INDEX_NAME")

es = Elasticsearch(ES_URL)
count = es.count(index=INDEX_NAME)
print("Total documents in index:", count['count'])