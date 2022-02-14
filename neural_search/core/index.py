from jina import Flow, Document
import shutil
import os
from typing import List
from neural_search.core.utils import DataHandler

INDEX_FLOW_PATH = os.environ.get('INDEX_FLOW_PATH', 'flows/index.yml')

class Index:

    def __init__(self):
        self.data_handler = DataHandler()

    def index_docs(self, docs: List[str]) -> None:
        """
        Index documents.
        """
        # Check if hash of docs name exists
        exists, path = self.data_handler.hash_docs_name_exists(docs)
        if exists:
            # Load
            self.data_handler.load_persisted_docs(path)
        else:
            # Preprocess
            docs = self.data_handler.preprocess_docs(docs)
            # Persist
            self.data_handler.persist_preprocessed_docs(docs, path)
        # # Delete workspace before indexing to avoid duplicates
        # if os.path.isdir('workspace'):
        #     shutil.rmtree('workspace')
        
        # flow = Flow.load_config(index_flow_path)
        
        # with flow, open(preprocess_path) as fp:
        #     flow.logger.info(f'Indexing {preprocess_path}')
        #     flow.index(from_csv(fp, field_resolver={index_field: 'text'}), show_progress=True)

        # # Remove preprocess_path file
        # os.remove(preprocess_path)

        return flow