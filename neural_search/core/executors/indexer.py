import inspect
import os
from typing import Dict, Optional

from docarray import DocumentArray
from jina import Executor, requests
from jina.logging.logger import JinaLogger
from collections import Counter


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
        self._index_splitted_cache = {}

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
            self._index_splitted_cache = {}

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
        filter_by_tags = {}
        if 'filter_by_tags' in parameters:
            filter_by_tags = parameters.pop('filter_by_tags')
        if 'filter_by_tags_method' in parameters:
            filter_by_tags_method = parameters.pop('filter_by_tags_method')
            if filter_by_tags_method not in ['OR', 'AND']:
                filter_by_tags_method = 'OR'
                print('filter_by_tags_method should be either "OR" or "AND". Defaulting to "OR"')
        else:
            filter_by_tags_method = 'OR'
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

        _index_filtered = self._filter_by_tags(filter_by_tags, filter_by_tags_method, traversal_right)

        docs[traversal_left].match(_index_filtered, **match_args)
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
                m.tags.update({
                    'parent_text': parent_doc.text,
                    'context': context
                    })

    def _filter_by_tags(self, filter_by_tags, filter_by_tags_method, traversal_right):
        """Filter the index by tags"""

        if len(filter_by_tags) > 0:
            filtered_id_docs = []
            for filter_dict in filter_by_tags:
                key = list(filter_dict.keys())[0]
                value = filter_dict[key]
                if key not in self._index_splitted_cache:
                    self._index_splitted_cache[key] = self._index[traversal_right].split_by_tag(tag=key)
                if value in self._index_splitted_cache[key]:
                    filtered_id_docs += [[doc.id for doc in self._index_splitted_cache[key][value]]]

            _index_filtered = DocumentArray()
            if filter_by_tags_method == 'OR':
                unique_doc_ids = list(set([id for docarray in filtered_id_docs for id in docarray]))
                _index_filtered = self._index[traversal_right][unique_doc_ids]
            elif filter_by_tags_method == 'AND' and len(filtered_id_docs) > 0:
                intersection_doc_ids = list(set.intersection(*map(set, filtered_id_docs)))
                _index_filtered = self._index[traversal_right][intersection_doc_ids]
        else:
            _index_filtered = self._index[traversal_right]

        return _index_filtered

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

    @requests(on='/tags')
    def tags(self, parameters: Dict, **kwargs):
        """retrieve tags of Documents by id if provided, otherwise all tags

        :param parameters: parameters to the request
        """
        print(parameters)
        ids = parameters.get('doc_ids', [])
        traversal_right = parameters.get('traversal_right', self.default_traversal_right)
        index_traversal = self._index[traversal_right]

        def count_tags_func(tags):
            count_tags = {}
            for tag_dict in tags:
                for key, value in tag_dict.items():
                    if key in count_tags:
                        count_tags[key] += [value]
                    else:
                        count_tags[key] = [value]
            count_tags = {key: dict(Counter(value)) for key, value in count_tags.items()}
            return count_tags

        if len(ids) == 0:
            tags =  [d.tags for d in index_traversal if d.tags != {}]
        else:
            tags = [index_traversal[id].tags for id in ids]

        count_tags = count_tags_func(tags)
        return {'tags': count_tags}


    @requests(on='/clear')
    def clear(self, **kwargs):
        """clear the database"""
        self._index.clear()

    @requests(on='/length')
    def length(self, **kwargs) -> dict:
        """return the length of the index"""
        return {'length': len(self._index)}