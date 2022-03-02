import io
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from neural_search.core.search import Search
from neural_search.core.utils import DataHandler

app = FastAPI()

search = Search()
data_handler = DataHandler()

class IndexPayload(BaseModel):
    files: UploadFile = Field(..., title='Optional zip file')
    num_docs: Optional[int]

class SearchResponseDict(BaseModel):
    text: str
    score: float

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
def search_docs(query: str) -> SearchResponse:
    search_results = search.query(query)
    return SearchResponse(
        docs=search_results,
        query=query,
        top_k=5)
