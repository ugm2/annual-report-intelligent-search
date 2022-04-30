import io
import os
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from typing import List
from neural_search.core.search import Search
from neural_search.core.utils import DataHandler
from neural_search.core.tagger import NERTagger

INIT_TAGGER = eval(os.environ.get('INIT_TAGGER', True))

app = FastAPI()

tagger = NERTagger() if INIT_TAGGER else None
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
    filter_by_tags: List[dict] = []
    filter_by_tags_method: str = 'OR'

class SearchResponseDict(BaseModel):
    doc_id: str
    text: str
    score: float
    tags: dict

class SearchResponse(SearchRequest):
    docs: List[SearchResponseDict]

class TagsRequest(BaseModel):
    doc_ids: List[str]

@app.post('/index')
def index_docs(zipfile: UploadFile = None,
               reload: bool = False,
               reload_persisted: bool = False,
               tag: bool = True) -> None:
    global tagger
    global data_handler
    global search
    if tagger is None and tag:
        print('Initializing NER tagger...')
        tagger = NERTagger()
        data_handler.ner_tagger = tagger
        search.data_handler = data_handler

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
        context_length=search_request.context_length,
        filter_by_tags=search_request.filter_by_tags,
        filter_by_tags_method=search_request.filter_by_tags_method)
    return SearchResponse(
        docs=search_results,
        query=search_request.query,
        top_k=search_request.top_k,
        context_length=search_request.context_length,
        filter_by_tags=search_request.filter_by_tags,
        filter_by_tags_method=search_request.filter_by_tags_method
        )

@app.post('/tags')
def get_tags(tags_request: TagsRequest) -> List[str]:
    return search.get_tags(tags_request.doc_ids)

@app.on_event("shutdown")
def shutdown_event():
    search.close_flow()