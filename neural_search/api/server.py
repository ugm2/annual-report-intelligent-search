from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional
from neural_search.core.index import Index
from neural_search.core.query import Query

app = FastAPI()

index = Index()
query = Query()

class IndexPayload(BaseModel):
    num_docs: Optional[int]
    index_field: Optional[str]