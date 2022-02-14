import os
from typing import List, Tuple
import zipfile
import io
import numpy as np
from tok import Tokenizer
from hashlib import sha512
import csv

# Local data path
DATA_PATH = os.environ.get('DATA_PATH', 'data/')

class DataHandler:

    def __init__(self):
        self.tokenizer = Tokenizer(
            currencies=("$", "€", "£", "¥"),
        )
        self.persist_path = os.path.join(DATA_PATH, 'persist')
        os.makedirs(self.persist_path, exist_ok=True)

    def preprocess_docs(self, docs: List[List[str]]) -> List[List[str]]:
        """
        Preprocess documents.

        Args:
            docs: list of strings

        Returns:
            list of list of strings
        """
        # Tokenize into sentences
        docs = [self.tokenizer.sent_tokenize(doc) for doc in docs]
        return docs

    def hash_docs_name_exists(self, docs: List[str]) -> Tuple[bool, str]:
        """
        Check if hash of docs name exists.

        Args:
            docs: list of strings

        Returns:
            Tuple[bool, str]:
        """
        # Compute hash of docs
        name = ''.join([doc[0] for doc in docs])
        _hash = sha512(name.encode('utf-8')).hexdigest()
        # Check if _hash exists
        path = os.path.join(self.persist_path, _hash)
        return os.path.exists(path), path

    def persist_preprocessed_docs(self, docs: List[List[str]], path: str) -> None:
        """
        Persist preprocessed documents.

        Args:
            docs: list of list of strings

        Returns:
            None
        """
        # Save to file
        with open(path + '.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerows(docs)

    def load_persisted_docs(self, path: str) -> List[List[str]]:
        """
        Load persisted docs.

        Args:
            path: path to persisted docs

        Returns:
            list of list of strings
        """
        # Load data from file
        with open(path + '.csv', 'r') as f:
            reader = csv.reader(f)
            data = list(reader)
        return data

    def _handle_data(self, data: io.BytesIO = None) -> List[str]:
        """
        Handle zip file or local data files.

        Args:
            data: data file

        Returns:
            List of strings
        """
        docs = []
        if data is not None:
            try:
                f = zipfile.ZipFile(data)
                for file in f.namelist():
                    docs.append(f.read(file).decode('utf-8'))
                f.close()
            except Exception as e:
                print('File is not a zip file. Error: ', e)
        else:
            # Load data from folder
            # List of files in folder
            files = os.listdir(DATA_PATH)
            for file in files:
                # Read file
                with open(os.path.join(DATA_PATH, file), 'r') as f:
                    data = f.read()
                # Add to docs
                docs.append(data)
        return docs

    def data_to_list(self, data: io.BytesIO = None) -> List[str]:
        """
        Convert data to list of Strings.

        Args:
            data: data file

        Returns:
            List of strings
        """
        return self._handle_data(data)