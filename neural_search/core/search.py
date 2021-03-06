from jina import Flow, Client
from docarray import Document, DocumentArray
import os
from typing import List
from neural_search.core.utils import DataHandler
from tqdm import tqdm

FLOW_PATH = os.environ.get('FLOW_PATH', 'flows/index_query.yml')

class Search:

    def __init__(self, data_handler: DataHandler):
        self.data_handler = data_handler if data_handler is not None else DataHandler()
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
        results = response[0].parameters['__results__']
        return int(results[list(results.keys())[0]]['length'])

    def to_document_array(self, list_docs: List[dict]) -> DocumentArray:
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
            inner_docs = DocumentArray()
            for doc, tags in zip(docs['sentences'], docs['tags']):
                document = Document(
                    text=doc,
                    tags=tags
                )
                inner_docs.append(document)
            root_document = Document(
                text='Document {}'.format(int(i + current_num_docs)),
                chunks=inner_docs)
            jina_docs.append(root_document)
        return DocumentArray(jina_docs)

    def index(self,
              docs: List[tuple],
              reload: bool = False,
              reload_persisted: bool = False,
              tag: bool = True) -> None:
        """
        Index documents.
        """
        # Check if hash of docs name exists
        exists, path, docs = self.data_handler.hash_docs_name_exists(docs)
        if exists and not reload_persisted:
            print('Data already exists. Loading persisted data...')
            # Load
            docs = self.data_handler.load_persisted_docs(path)
        else:
            # Preprocess
            docs = self.data_handler.preprocess_docs(docs, tag)
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

    def query(self,
              query: str,
              top_k : int = 5,
              context_length : int = 5,
              filter_by_tags : List[dict] = [],
              filter_by_tags_method : str = 'OR') -> List[dict]:
        """
        Query documents.
        """
        query = Document(text=query)
        response = self.flow.search(
            inputs=query,
            return_results=True,
            parameters={
                'limit': top_k,
                'context_length': context_length,
                'filter_by_tags': filter_by_tags,
                'filter_by_tags_method': filter_by_tags_method
            },
        )
        # Get top k matches
        top_k_matches = []
        for r in response:
            for match in r.matches:
                score = list(match.scores.values())[0].value
                top_k_matches.append({
                    'doc_id': match.id,
                    'text': match.text,
                    'score': round(1.0 - score, 2),
                    'tags': match.tags
                })
        return top_k_matches

    def get_tags(self, doc_ids: List[str]) -> List[str]:
        """
        Get tags.
        """
        response = self.client.post(
            '/tags',
            parameters={'doc_ids': doc_ids, 'traversal_right': '@c'},
            target_executor='CustomIndexer',
            return_responses=True)
        results = response[0].parameters['__results__']
        tags = results[list(results.keys())[0]]['tags']
        return tags