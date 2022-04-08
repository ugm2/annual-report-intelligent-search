import io
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from typing import List
from neural_search.core.search import Search
from neural_search.core.utils import DataHandler

app = FastAPI()

search = Search()
data_handler = DataHandler()

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class SearchResponseTags(BaseModel):
    parent_text: str

class SearchResponseDict(BaseModel):
    text: str
    score: float
    tags: SearchResponseTags

class SearchResponse(BaseModel):
    docs: List[SearchResponseDict]
    query: str
    top_k: int

@app.post('/index')
def index_docs(zipfile: UploadFile = None, reload: bool = False) -> None:
    print("Loading bytes")
    file_bytes = None
    if zipfile is not None:
        file_bytes = io.BytesIO(zipfile.file.read())
    print("Loading zip")
    data = data_handler.data_to_list(file_bytes)
    print("Indexing")
    search.index(data, reload)

@app.post('/search')
def search_docs(search_request: SearchRequest) -> SearchResponse:
    search_results = search.query(
        search_request.query,
        top_k=search_request.top_k)
    return SearchResponse(
        docs=search_results,
        query=search_request.query,
        top_k=search_request.top_k)

@app.on_event("shutdown")
def shutdown_event():
    search.close_flow()