import pandas as pd
from jina import Flow, Document
from jina.types.document.generators import from_csv
import shutil
import os

class Index:

    def preprocess_docs(self, docs, index_field):
        """
        Preprocess documents.
        """
        docs.dropna(inplace = True)
        docs.drop_duplicates(subset= [index_field], inplace = True)
        return docs


    def index_docs(self, path, num_docs, index_field, index_flow_path):
        """
        Index documents from a path.
        """
        docs = pd.read_csv(path, nrows=num_docs)
        docs = preprocess_docs(docs, index_field)
        preprocess_path = ''.join(path.split('.csv')[:-1]) + '_preprocess.csv'
        docs.to_csv(preprocess_path, index=False)
        # Delete workspace before indexing to avoid duplicates
        if os.path.isdir('workspace'):
            shutil.rmtree('workspace')
        
        flow = Flow.load_config(index_flow_path)
        
        with flow, open(preprocess_path) as fp:
            flow.logger.info(f'Indexing {preprocess_path}')
            flow.index(from_csv(fp, field_resolver={index_field: 'text'}), show_progress=True)

        # Remove preprocess_path file
        os.remove(preprocess_path)

        return flow