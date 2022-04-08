from jina import Flow, Client
from docarray import Document, DocumentArray
import shutil
import os
from typing import List, Dict
from neural_search.core.utils import DataHandler
from tqdm import tqdm

FLOW_PATH = os.environ.get('FLOW_PATH', 'flows/index_query.yml')

class Search:

    def __init__(self):
        self.data_handler = DataHandler()
        self._init_flow()

    def _init_flow(self):
        """
        Initialize flow.
        """
        self.flow = Flow.load_config(FLOW_PATH)
        self.flow.expose_endpoint('/clear')
        self.flow.expose_endpoint('/length')
        self.flow.start()
        self.client = Client(port=self.flow.port)

    def close_flow(self):
        """Close the flow."""
        self.flow.close()

    def _clear_index(self):
        """
        Clear the index calling the endpoint /clear
        """
        self.client.post('/clear', target_executor='CustomIndexer')

    def _get_length(self) -> int:
        """
        Get length of index.
        """
        response = self.client.post('/length', target_executor='CustomIndexer', return_responses=True)
        json_response = eval(response[0].json())
        results = json_response['parameters']['__results__']
        return int(results[list(results.keys())[0]]['length'])

    def to_document_array(self, list_docs: List[List[str]]) -> DocumentArray:
        """
        Convert list of list of strings to list of documents.

        Args:
            list_docs: list of list of strings

        Returns:
            list of documents
        """
        jina_docs = []
        current_num_docs = self._get_length()
        print('{} previously indexed documents'.format(current_num_docs))
        for i, docs in enumerate(tqdm(list_docs, desc='Converting to documents')):
            root_document = Document()
            root_document.text = 'Document {}'.format(int(i + current_num_docs))
            for doc in docs:
                document = Document(text=doc)
                document.tags = {'parent_text': root_document.text}
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

        # Clear documents
        if reload:
            self._clear_index()

        # Convert to documents
        docs = self.to_document_array(docs)

        self.flow.index(docs, parameters={'traversal_paths': '@c'}, show_progress=True)

        # Print number of documents indexed in total
        print('{} documents indexed in total'.format(self._get_length()))

    def query(self, query: str, top_k : int = 5) -> List[dict]:
        """
        Query documents.
        """
        query = Document(text=query)
        response = self.flow.search(
            inputs=query,
            return_results=True,
            parameters={'limit': top_k}
        )
        # Get top k matches
        top_k_matches = []
        for r in response:
            for match in r.matches:
                score = list(match.scores.values())[0].value
                top_k_matches.append({
                    'text': match.text,
                    'score': round(1.0 - score, 2),
                    'tags': {'parent_text': match.tags['parent_text']}
                })
        return top_k_matches