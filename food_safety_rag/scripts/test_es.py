from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")

if es.ping():
    print("✅ Elasticsearch is connected and running!")
else:
    print("❌ Could not connect to Elasticsearch.")

# Run
# python scripts/test_es.py