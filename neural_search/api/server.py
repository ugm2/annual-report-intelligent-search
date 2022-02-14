import io
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel, Field
from typing import List, Optional
from neural_search.core.index import Index
from neural_search.core.query import Query
from neural_search.core.utils import DataHandler

app = FastAPI()

index = Index()
query = Query()
data_handler = DataHandler()

class IndexPayload(BaseModel):
    files: UploadFile = Field(..., title='Optional zip file')
    num_docs: Optional[int]

class SearchResponse(BaseModel):
    docs: List[str]

@app.post('/index')
async def index_docs(zipfile: UploadFile = None) -> None:
    file_bytes = io.BytesIO(zipfile.file.read())
    data = data_handler.data_to_list(file_bytes)
    