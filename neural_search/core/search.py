from jina import Flow
from jina import AsyncFlow
from docarray import Document, DocumentArray
import shutil
import os
from typing import List
from neural_search.core.utils import DataHandler
from tqdm import tqdm

INDEX_FLOW_PATH = os.environ.get('INDEX_FLOW_PATH', 'flows/index.yml')
QUERY_FLOW_PATH = os.environ.get('QUERY_FLOW_PATH', 'flows/query.yml')

class Search:

    def __init__(self):
        self.data_handler = DataHandler()

    def to_document_array(self, list_docs: List[List[str]]) -> List[Document]:
        """
        Convert list of list of strings to list of documents.

        Args:
            list_docs: list of list of strings

        Returns:
            list of documents
        """
        jina_docs = []
        for docs in tqdm(list_docs, desc='Converting to documents'):
            root_document = Document()
            for doc in docs:
                document = Document(text=doc)
                root_document.chunks.append(document)
            jina_docs.append(root_document)
        return DocumentArray(jina_docs)

    def index(self, docs: List[str], reload: bool = False) -> None:
        """
        Index documents.
        """
        # Check if hash of docs name exists
        exists, path = self.data_handler.hash_docs_name_exists(docs)
        if exists and not reload:
            print('Data already exists. Loading persisted data...')
            # Load
            docs = self.data_handler.load_persisted_docs(path)
        else:
            # Preprocess
            docs = self.data_handler.preprocess_docs(docs)
            # Persist
            self.data_handler.persist_preprocessed_docs(docs, path)

        # Convert to documents
        docs = self.to_document_array(docs)
        # Delete workspace before indexing to avoid duplicates
        if os.path.isdir('workspace'):
            shutil.rmtree('workspace')
        
        flow = Flow.load_config(INDEX_FLOW_PATH)
        
        # Index
        with flow:
            flow.index(docs, show_progress=True)

    def query(self, query: str, top_k : int = 5) -> List[str]:
        """
        Query documents.
        """
        flow = Flow.load_config(QUERY_FLOW_PATH)
        with flow:
            query = Document(text=query)
            response = flow.search(inputs=query, return_results=True)

        # Get top k matches
        top_k_matches = []
        for match in response[:top_k]:
            top_k_matches.append(match.text)

        return top_k_matches