import inspect
import os
from typing import Dict, Optional

from jina import DocumentArray, Executor, requests
from jina.logging.logger import JinaLogger


class CustomIndexer(Executor):
    """
    A simple indexer that stores all the Document data together in a DocumentArray,
    and can dump to and load from disk.

    To be used as a unified indexer, combining both indexing and searching
    """

    FILE_NAME = 'index.db'

    def __init__(
        self,
        match_args: Optional[Dict] = None,
        table_name: str = 'simple_indexer_table',
        traversal_right: str = '@r',
        traversal_left: str = '@r',
        **kwargs,
    ):
        """
        Initializer function for the simple indexer

        To specify storage path, use `workspace` attribute in executor `metas`
        :param match_args: the arguments to `DocumentArray`'s match function
        :param table_name: name of the table to work with for the sqlite backend
        :param traversal_right: the default traversal path for the indexer's
        DocumentArray
        :param traversal_left: the default traversal path for the query
        DocumentArray
        """
        super().__init__(**kwargs)

        self._match_args = match_args or {}
        self._index = DocumentArray(
            storage='sqlite',
            config={
                'connection': os.path.join(self.workspace, CustomIndexer.FILE_NAME),
                'table_name': table_name,
            },
        )  # with customize config
        self.logger = JinaLogger(self.metas.name)
        self.default_traversal_right = traversal_right
        self.default_traversal_left = traversal_left

    @property
    def table_name(self) -> str:
        return self._index._table_name

    @requests(on='/index')
    def index(
        self,
        docs: 'DocumentArray',
        **kwargs,
    ):
        """All Documents to the DocumentArray
        :param docs: the docs to add
        """
        if docs:
            self._index.extend(docs)

    @requests(on='/search')
    def search(
        self,
        docs: 'DocumentArray',
        parameters: Optional[Dict] = None,
        **kwargs,
    ):
        """Perform a vector similarity search and retrieve the full Document match

        :param docs: the Documents to search with
        :param parameters: the runtime arguments to `DocumentArray`'s match
        function. They overwrite the original match_args arguments.
        """
        match_args = (
            {**self._match_args, **parameters}
            if parameters is not None
            else self._match_args
        )

        traversal_right = parameters.get(
            'traversal_right', self.default_traversal_right
        )
        traversal_left = parameters.get('traversal_left', self.default_traversal_left)
        match_args = CustomIndexer._filter_match_params(docs, match_args)
        docs[traversal_left].match(self._index[traversal_right], **match_args)
        context_length = int(parameters.get('context_length', 5))
        for d in docs[traversal_left]:
            for m in d.matches:
                parent_doc = self._index[m.parent_id]
                current_doc_id = m.id
                context = ""
                for i, c in enumerate(parent_doc.chunks):
                    if c.id == current_doc_id:
                        surrounding_chunks = parent_doc.chunks[max(0, i - context_length) : i + context_length]
                        context = " ".join([c.text for c in surrounding_chunks])
                        break
                m.tags = {
                    'parent_text': parent_doc.text,
                    'context': context
                    }

    @staticmethod
    def _filter_match_params(docs, match_args):
        # get only those arguments that exist in .match
        args = set(inspect.getfullargspec(docs.match).args)
        args.discard('self')
        match_args = {k: v for k, v in match_args.items() if k in args}
        return match_args

    @requests(on='/delete')
    def delete(self, parameters: Dict, **kwargs):
        """Delete entries from the index by id

        :param parameters: parameters to the request
        """
        deleted_ids = parameters.get('ids', [])
        if len(deleted_ids) == 0:
            return
        del self._index[deleted_ids]

    @requests(on='/update')
    def update(self, docs: DocumentArray, **kwargs):
        """Update doc with the same id, if not present, append into storage

        :param docs: the documents to update
        """

        for doc in docs:
            try:
                self._index[doc.id] = doc
            except IndexError:
                self.logger.warning(
                    f'cannot update doc {doc.id} as it does not exist in storage'
                )

    @requests(on='/fill_embedding')
    def fill_embedding(self, docs: DocumentArray, **kwargs):
        """retrieve embedding of Documents by id

        :param docs: DocumentArray to search with
        """
        for doc in docs:
            doc.embedding = self._index[doc.id].embedding

    @requests(on='/clear')
    def clear(self, **kwargs):
        """clear the database"""
        self._index.clear()

    @requests(on='/length')
    def length(self, **kwargs) -> dict:
        """return the length of the index"""
        return {'length': len(self._index)}