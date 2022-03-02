import os
from typing import List, Tuple
import zipfile
import io
from spacy.lang.en import English
from hashlib import sha512
import sys
import csv
csv.field_size_limit(sys.maxsize)
from tqdm import tqdm

# Local data path
DATA_PATH = os.environ.get('DATA_PATH', 'data/')

class DataHandler:

    def __init__(self):
        self.nlp = English()
        self.nlp.add_pipe("sentencizer")
        self.nlp.max_length = 10000000
        self.persist_path = os.path.join(DATA_PATH, 'persist')

    def _clean_text(self, text):
        """
        Clean text.

        Args:
            text: text

        Returns:
            text
        """
        # Remove new lines
        text = text.replace('\n', ' ')
        # Remove multiple spaces
        text = ' '.join(text.split())
        return text

    def preprocess_docs(self, docs: List[str]) -> List[List[str]]:
        """
        Preprocess documents.

        Args:
            docs: list of strings

        Returns:
            list of list of strings
        """
        # Tokenize into sentences
        docs_sentences = []
        for doc in tqdm(self.nlp.pipe(docs, batch_size=1000), desc='Preprocessing'):
            # Get sentences
            sentences = [sent.text for sent in doc.sents]
            # Clean sentences
            sentences = list(map(self._clean_text, sentences))
            # Add to docs
            docs_sentences.append(sentences)
        return docs_sentences

    def hash_docs_name_exists(self, docs: List[str]) -> Tuple[bool, str]:
        """
        Check if hash of docs name exists.

        Args:
            docs: list of strings

        Returns:
            Tuple[bool, str]:
        """
        # Create folder if it doesn't exist
        os.makedirs(self.persist_path, exist_ok=True)
        # Compute hash of docs
        name = ''.join([doc[0] for doc in docs])
        _hash = sha512(name.encode('utf-8')).hexdigest()
        # Check if _hash exists
        path = os.path.join(self.persist_path, _hash + '.csv')
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
        with open(path, 'w') as f:
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
        with open(path, 'r') as f:
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
            data_path = os.path.join(DATA_PATH, 'annual_accounts_txt')
            files = os.listdir(data_path)
            for file in files:
                # Read file
                with open(os.path.join(data_path, file), 'r') as f:
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