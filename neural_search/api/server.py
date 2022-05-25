import io
import os
from fastapi import FastAPI, UploadFile
from fastapi.param_functions import Depends
from pydantic import BaseModel
from typing import Dict, List
from neural_search.core.search import Search
from neural_search.core.utils import DataHandler
from neural_search.core.tagger import QuestionAnswerTagger

INIT_TAGGER = eval(os.environ.get('INIT_TAGGER', True))

app = FastAPI()

question_tags = [
    {'tag': 'year', 'question': 'What year was the document written?', 'confidence': 0.95},
    {'tag': 'who', 'question': 'Who is involved?', 'confidence': 0.5},
    {'tag': 'challenges', 'question': 'What are the main challenges?', 'confidence': 0.5},
    {'tag': 'opportunities', 'question': 'What are the main opportunities?', 'confidence': 0.5},
    {'tag': 'initiatives', 'question': 'What are the main initiatives?', 'confidence': 0.5},
]
tagger = QuestionAnswerTagger(question_tags=question_tags) if INIT_TAGGER else None
data_handler = DataHandler(
    ner_tagger=tagger
)
search = Search(
    data_handler=data_handler
)

class IndexRequest(BaseModel):
    zipfile: UploadFile
    reload: bool = False
    reload_persisted: bool = False
    tag: bool = False
    # question_tags: Dict[str, str] = question_tags
    tagging_confidence: float = 0.5

    class Config:
         orm_mode=True

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    context_length: int = 5
    filter_by_tags: List[dict] = []
    filter_by_tags_method: str = 'OR'
    
    class Config:
        schema_extra = {
            "example": {
                "query": "Companies",
                "top_k": 5,
                "context_length": 1,
                "filter_by_tags": [
                    {'tag': 'who', 'tag_value': 'BioPharmX', 'threshold': 0.5}
                ],
                "filter_by_tags_method": 'OR'
            }
        }

class SearchResponseDict(BaseModel):
    doc_id: str
    text: str
    score: float
    tags: dict

class SearchResponse(SearchRequest):
    docs: List[SearchResponseDict]

class TagsRequest(BaseModel):
    doc_ids: List[str] = []

@app.post('/index')
def index_docs(request: IndexRequest = Depends()) -> None:
    global tagger
    global data_handler
    global search
    # question_tags = request.question_tags
    if request.tag and (tagger is None or
                # tagger.questions != question_tags or
                tagger.tagging_confidence != request.tagging_confidence):
        print('Initializing NER tagger...')
        tagger = QuestionAnswerTagger(questions=question_tags,
                                      tagging_confidence=request.tagging_confidence)
        data_handler.ner_tagger = tagger
        search.data_handler = data_handler

    print("Loading bytes")
    file_bytes = None
    if request.zipfile is not None:
        file_bytes = io.BytesIO(request.zipfile.file.read())
    print("Loading zip")
    data = data_handler.data_to_list(file_bytes)
    print("Indexing")
    search.index(data, request.reload, request.reload_persisted, request.tag)

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