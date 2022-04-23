import os
from typing import List, Tuple
import zipfile
import io
from spacy.lang.en import English
from hashlib import sha512
import json
from tqdm import tqdm
from neural_search.core.tagger import NERTagger

# Local data path
DATA_PATH = os.environ.get('DATA_PATH', 'data/')

class DataHandler:

    def __init__(self, ner_tagger: NERTagger):
        self.nlp = English()
        self.nlp.add_pipe("sentencizer")
        self.nlp.max_length = 10000000
        self.persist_path = os.path.join(DATA_PATH, 'persist')
        self.ner_tagger = ner_tagger if ner_tagger is not None else NERTagger()

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
        return text.strip()

    def preprocess_docs(self, docs: List[str], tag: bool = False) -> List[List[str]]:
        """
        Preprocess documents.

        Args:
            docs: list of strings

        Returns:
            list of list of strings
        """
        # Tokenize into sentences
        docs_sentences = []
        total_len = len(docs)/1000 if len(docs) > 1000 else len(docs)
        for doc in tqdm(self.nlp.pipe(docs, batch_size=1000), desc='Preprocessing', total=total_len):
            # Get sentences
            sentences = [sent.text for sent in doc.sents]
            # Clean sentences
            sentences = list(map(self._clean_text, sentences))
            # Remove empty strings
            sentences = list(filter(lambda x: x != '', sentences))
            # Tag sentences
            tags = []
            if tag:
                for s in sentences:
                    predicted_tags = self.ner_tagger.predict(s)
                    # Filter out keys with None
                    predicted_tags = dict(filter(lambda x: x[0] is not None, predicted_tags.items()))
                    tags.append(predicted_tags)
            else:
                tags = [{}] * len(sentences)
            # Add to docs
            docs_sentences.append({
                'sentences': sentences,
                'tags': tags
            })
        return docs_sentences

    def hash_docs_name_exists(self, docs: List[tuple]) -> Tuple[bool, str]:
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
        name = ''.join([doc[1] for doc in docs])
        _hash = sha512(name.encode('utf-8')).hexdigest()
        # Check if _hash exists
        path = os.path.join(self.persist_path, _hash + '.json')
        # Return docs without filenames
        docs = [doc[0] for doc in docs]
        return os.path.exists(path), path, docs

    def persist_preprocessed_docs(self, docs: List[dict], path: str) -> None:
        """
        Persist preprocessed documents.

        Args:
            docs: list of list of strings

        Returns:
            None
        """
        # Save to file
        with open(path, 'w') as f:
            json.dump(docs, f)

    def load_persisted_docs(self, path: str) -> List[dict]:
        """
        Load persisted docs.

        Args:
            path: path to persisted docs

        Returns:
            list of list of strings
        """
        # Load data from file
        with open(path, 'r') as f:
            data = json.load(f)
        return data

    def _handle_data(self, data: io.BytesIO = None) -> List[tuple]:
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
                    docs.append((f.read(file).decode('utf-8'), file))
                f.close()
            except Exception as e:
                print('File is not a zip file. Error: ', e)
        else:
            # Load data from folder
            # List of files in folder
            data_path = os.path.join(DATA_PATH, 'annual_accounts_old')
            files = os.listdir(data_path)
            for file in files:
                # Read file
                with open(os.path.join(data_path, file), 'r') as f:
                    data = f.read()
                # Add to docs
                docs.append((data, file))
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