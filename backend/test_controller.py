from src.controllers.search_controller import execute_search

bundle = {
    "raw": "test",
    "cleaned": "test",
    "lang": "en",
    "translated_en": "test",
    "rewrites": ["test"]
}
try:
    print("Testing agentic...")
    res = execute_search(bundle, top_k=2, strategy="agentic")
    print(res.latency_ms)
except Exception as e:
    print(e)
