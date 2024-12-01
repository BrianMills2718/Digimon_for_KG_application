from typing import Union, Any

import tiktoken

from Core.Chunk.ChunkFactory import get_chunks
from Core.Common.Utils import mdhash_id
from Core.Graph.PassageGraph import PassageGraph


async def chunk_documents(docs: Union[str, list[Any]], is_chunked: bool = False, ENCODER = tiktoken.encoding_for_model("gpt-3.5-turbo")) -> dict[str, dict[str, str]]:
    """Chunk the given documents into smaller chunks.

    Args:
    docs (Union[str, list[str]]): The documents to chunk, either as a single string or a list of strings.

    Returns:
    dict[str, dict[str, str]]: A dictionary where the keys are the MD5 hashes of the chunks, and the values are dictionaries containing the chunk content.
    """
    if isinstance(docs, str):
        docs = [docs]

    if isinstance(docs[0], dict):
        new_docs = {doc['id']: {"content": doc['content'].strip()} for doc in docs}
    else:
        new_docs = {mdhash_id(doc.strip(), prefix="doc-"): {"content": doc.strip()} for doc in docs}
    chunks = await get_chunks(new_docs, "chunking_by_token_size", ENCODER, is_chunked=is_chunked)
    return chunks
import asyncio
if __name__ == "__main__":
    
    with open("./book.txt") as f:
        doc = f.read()
    graph = PassageGraph()

    chunks = asyncio.run(chunk_documents(doc))
    inserting_chunks = {key: value for key, value in chunks.items() if key in chunks}
    ordered_chunks = list(inserting_chunks.items())
    asyncio.run(graph.build_graph(ordered_chunks))

    
    # asyncio.run(graph.query("who are you"))
   