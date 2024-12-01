from typing import Union, Any
from Core.Common.Logger import logger
import tiktoken
from Core.Chunk.ChunkFactory import get_chunks
from Core.Common.Utils import mdhash_id
from pydantic import BaseModel, Field, ConfigDict, model_validator
from Core.Common.ContextMixin import ContextMixin
from Core.Graph.BaseGraph import BaseGraph
from Core.Graph.GraphFactory import get_graph


class GraphRAG(BaseModel, ContextMixin):
    """A class representing a Graph-based Retrieval-Augmented Generation system."""

    working_dir: str = Field(default=None, exclude=True)
    graph: BaseGraph = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def _update_context(cls, data):
        cls.config = data.context.config
        cls.ENCODER = tiktoken.encoding_for_model(cls.config.token_model)
        return data

    @model_validator(mode="after")
    def _register_graph(cls, data):
        cls.graph = get_graph(data.config, data.llm, data.ENCODER)
        return data

    @model_validator(mode="after")
    def _init_storage(cls, data):
        cls.graph_storage_path  =
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
        chunks = await get_chunks(new_docs, self.config, self.ENCODER, is_chunked=is_chunked)
        return chunks

    async def insert(self, docs):

        """
        The main function that orchestrates the first step in the Graph RAG pipeline.
        This function is responsible for executing the various stages of the Graph RAG process,
        including chunking, graph construction, index building, and graph augmentation (optional).

        Configuration of the Graph RAG method is based on the parameters provided in the config file.
        For detailed information on the configuration and usage, please refer to the README.md.

        Args:
            docs (list): A list of documents to be processed and inserted into the Graph RAG pipeline.
        """

        ####################################################################################################
        # 1. Chunking Stage
        ####################################################################################################
        chunks = await self.chunk_documents(docs)

        ####################################################################################################
        # 2. Building Graph Stage
        ####################################################################################################
        logger.info(f"Starting build graph for the given documents")
        await self.graph.build_graph(chunks)
        await self.graph.persist_graph()

        ####################################################################################################
        # 3. Index building Stage
        ####################################################################################################

        ####################################################################################################
        # 4. Graph Augmentation Stage (Optional)
        ####################################################################################################

    async def query(self, query):
        """
          Executes the query by extracting the relevant context, and then generating a response.
          Args:
              query: The query to be processed.
          Returns:
          """
        ####################################################################################################
        # 1. Building query relevant context (subgraph) Stage
        ####################################################################################################

        ####################################################################################################
        # 2. Generation Stage
        ####################################################################################################
