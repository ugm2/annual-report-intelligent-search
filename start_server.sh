sudo INIT_TAGGER=False JINA_MP_START_METHOD=spawn python -m uvicorn neural_search.api.server:app --reload --host 0.0.0.0 --port 5002