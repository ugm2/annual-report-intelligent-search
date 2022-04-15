import io
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from typing import List
from neural_search.core.search import Search
from neural_search.core.utils import DataHandler
from neural_search.core.tagger import NERTagger

app = FastAPI()

tagger = NERTagger()
data_handler = DataHandler(
    ner_tagger=tagger
)
search = Search(
    data_handler=data_handler
)

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    context_length: int = 5

class SearchResponseDict(BaseModel):
    text: str
    score: float
    tags: dict

class SearchResponse(BaseModel):
    docs: List[SearchResponseDict]
    query: str
    top_k: int
    context_length: int

@app.post('/index')
def index_docs(zipfile: UploadFile = None,
               reload: bool = False,
               reload_persisted: bool = False,
               tag: bool = True) -> None:
    print("Loading bytes")
    file_bytes = None
    if zipfile is not None:
        file_bytes = io.BytesIO(zipfile.file.read())
    print("Loading zip")
    data = data_handler.data_to_list(file_bytes)
    print("Indexing")
    search.index(data, reload, reload_persisted, tag)

@app.post('/search')
def search_docs(search_request: SearchRequest) -> SearchResponse:
    search_results = search.query(
        search_request.query,
        top_k=search_request.top_k,
        context_length=search_request.context_length)
    return SearchResponse(
        docs=search_results,
        query=search_request.query,
        top_k=search_request.top_k,
        context_length=search_request.context_length)

@app.on_event("shutdown")
def shutdown_event():
    search.close_flow()