import os
from typing import List
import zipfile
import io
import numpy as np

# Local data path
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')

class DataHandler:

    def _handle_data(self, data: io.BytesIO = None) -> List[str]:
        """
        Handle zip file, txt files or local data files.
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
        """
        return self._handle_data(data)