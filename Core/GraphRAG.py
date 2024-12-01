from typing import Union, Any

from Core.Chunk.ChunkFactory import get_chunks
from Core.Common.Utils import mdhash_id


# await self._build_ppr_context()
#
# # Augment the graph by ann searching
# if self.config.enable_graph_augmentation:
#     data_for_aug = {mdhash_id(node, prefix="ent-"): node for node in self.er_graph.graph.nodes()}
#     await self._augment_graph(queries=data_for_aug)


class GraphRAG:

    async def chunk_documents(self, docs: Union[str, list[Any]], is_chunked: bool = False) -> dict[str, dict[str, str]]:
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
        chunks = await get_chunks(new_docs, "chunking_by_seperators", self.ENCODER, is_chunked=is_chunked)
        return chunks

    ####################################################################################################
    # Chunking Stage
    ####################################################################################################



    ####################################################################################################
    # Building Graph Stage
    ####################################################################################################


    ####################################################################################################
    # Index building Stage
    ####################################################################################################


    ####################################################################################################
    # Graph Augmentation Stage (Optional)
    ####################################################################################################




    ####################################################################################################
    # Building query relevant context Stage
    ####################################################################################################




    ####################################################################################################
    # Generation Stage
    ####################################################################################################



    ####################################################################################################
    # Evaluation Stage
    ####################################################################################################
