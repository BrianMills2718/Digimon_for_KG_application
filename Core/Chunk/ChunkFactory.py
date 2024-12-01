from typing import Any
from Core.Common.Utils import mdhash_id
from collections import defaultdict

from Core.Schema.ChunkSchema import TextChunk


class ChunkingFactory:
    chunk_methods: dict = defaultdict(Any)

    def register_chunking_method(
            self,
            method_name: str,
            method_func=None  # can be any classes or functions
    ):
        if self.has_chunk_method(method_name):
            return

        self.chunk_methods[method_name] = method_func

    def has_chunk_method(self, key: str) -> Any:
        return key in self.chunk_methods

    def get_method(self, key) -> Any:
        return self.chunk_methods.get(key)


# Registry instance
CHUNKING_REGISTY = ChunkingFactory()


def register_chunking_method(method_name):
    """ Register a new chunking method
    
    This is a decorator that can be used to register a new chunking method.
    The method will be stored in the self.methods dictionary.
    
    Parameters
    ----------
    method_name: str
        The name of the chunking method.
    """

    def decorator(func):
        """ Register a new chunking method """
        CHUNKING_REGISTY.register_chunking_method(method_name, func)

    return decorator


def create_chunk_method(method_name):
    chunking_method = CHUNKING_REGISTY.get_method(method_name)
    return chunking_method


async def get_chunks(new_docs, chunk_method_name, token_model, is_chunked: bool = False, **chunk_func_params):
    kv_chunks = {}

    new_docs_list = list(new_docs.items())
    docs = [new_doc[1]["content"] for new_doc in new_docs_list]
    doc_keys = [new_doc[0] for new_doc in new_docs_list]

    tokens = token_model.encode_batch(docs, num_threads=16)

    if is_chunked:
        for idx, doc in enumerate(docs):
            kv_chunks.update(
                {mdhash_id(doc.strip(), prefix="chunk-"): TextChunk(**{
                    "tokens": tokens[idx],
                    "content": doc.strip(),
                    "chunk_order_index": idx,
                    "doc_id": doc_keys[idx],
                })}
            )
        return kv_chunks

    chunk_func = create_chunk_method(chunk_method_name)

    chunks = await chunk_func(
        tokens, doc_keys=doc_keys, tiktoken_model=token_model, **chunk_func_params
    )

    for chunk in chunks:
        kv_chunks.update(
            {mdhash_id(chunk["content"], prefix="chunk-"): TextChunk(**chunk)}
        )

    return kv_chunks
