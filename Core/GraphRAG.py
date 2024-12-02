from typing import Union, Any, Optional
from Core.Common.Logger import logger
import tiktoken
from Core.Chunk.ChunkFactory import get_chunks
from Core.Common.Utils import mdhash_id
from pydantic import BaseModel, Field, model_validator, field_validator, ConfigDict
from Core.Common.ContextMixin import ContextMixin
from Core.Graph.BaseGraph import BaseGraph
from Core.Graph.GraphFactory import get_graph
from Core.Index.VectorIndex import VectorIndex
from Core.Index.IndexConfigFactory import get_index_config
from Core.Storage.JsonKVStorage import JsonKVStorage
from Core.Storage.NameSpace import Workspace


class GraphRAG(ContextMixin, BaseModel):
    """A class representing a Graph-based Retrieval-Augmented Generation system."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    working_dir: str = Field(default="", exclude=True)
    graph: BaseGraph = Field(default=None, exclude=True)
    chunks_storage: Optional[JsonKVStorage] = Field(default=None, exclude=True)
    entities_vdb: Optional[VectorIndex] = Field(default=None, exclude=True)
    relations_vdb: Optional[VectorIndex] = Field(default=None, exclude=True)

    # The following two matrices are utilized for mapping entities to their corresponding chunks through the specified link-path:
    # Entity Matrix: Represents the entities in the dataset.
    # Chunk Matrix: Represents the chunks associated with the entities.
    # These matrices facilitate the entity -> relationship -> chunk linkage, which is integral to the HippoRAG and FastGraphRAG models.

    @field_validator("working_dir", mode="before")
    @classmethod
    def check_working_dir(cls, value: str):
        if value == "":
            logger.error("Working directory cannot be empty")
        return value

    @model_validator(mode="after")
    def _update_context(cls, data):
        cls.config = data.context.config
        cls.ENCODER = tiktoken.encoding_for_model(cls.config.token_model)
        cls.workspace = Workspace(data.working_dir, cls.config.exp_name)
        return data

    @model_validator(mode="after")
    def _register_graph(cls, data):
        cls.graph = get_graph(data.config, llm=data.llm, encoder=data.ENCODER)
        return data

    @model_validator(mode="after")
    def _init_storage_namespace(cls, data):
        cls.graph.namespace = cls.workspace.make_for("graph_storage")
        if data.config.use_entities_vdb:
            cls.entities_vdb_namespace = cls.workspace.make_for("entities_vdb")
        if data.config.use_relations_vdb:
            cls.relations_vdb_namespace = cls.workspace.make_for("relations_vdb")
        cls.chunks = data.workspace.make_for("chunks")
        return data

    @model_validator(mode="after")
    def _register_vdbs(cls, data):
        if data.config.use_entities_vdb:
            cls.entities_vdb = VectorIndex(
                get_index_config(data.config, persist_path=data.entities_vdb_namespace.get_save_path()))
        if data.config.use_relations_vdb:
            cls.relations_vdb = VectorIndex(
                get_index_config(data.config, persist_path=data.relations_vdb_namespace.get_save_path()))
        return data

    @model_validator(mode="after")
    def _register_e2r_r2c_matrix(cls, data):
        if data.config.use_entity_link_chunk:
            pass
        return data

    async def _chunk_documents(self, docs: Union[str, list[Any]], is_chunked: bool = False) -> dict[str, dict[str, str]]:
        """Chunk the given documents into smaller chunks.

        Args:
        docs (Union[str, list[str]]): The documents to chunk, either as a single string or a list of strings.

        Returns:
        dict[str, dict[str, str]]: A dictionary where the keys are the MD5 hashes of the chunks, and the values are dictionaries containing the chunk content.
        """
        if isinstance(docs, str):
            docs = [docs]

        if isinstance(docs[0], dict):
            new_docs = {doc.get("id"): {"content": doc['content'].strip()} for doc in docs}
        else:
            new_docs = {mdhash_id(doc.strip(), prefix="doc-"): {"content": doc.strip()} for doc in docs}
        #TODO: config the chunk parameters, **WE ONLY CONFIG CHUNK-METHOD NOW**
        chunks = await get_chunks(new_docs, self.config.chunk_method, self.ENCODER, is_chunked=is_chunked)
        return chunks

    async def insert(self, docs: Union[str, list[[Any]]]):

        """
        The main function that orchestrates the first step in the Graph RAG pipeline.
        This function is responsible for executing the various stages of the Graph RAG process,
        including chunking, graph construction, index building, and graph augmentation (optional).

        Configuration of the Graph RAG method is based on the parameters provided in the config file.
        For detailed information on the configuration and usage, please refer to the README.md.

        Args:
            docs (Union[str, list[[Any]]): A list of documents to be processed and inserted into the Graph RAG pipeline.
        """

        ####################################################################################################
        # 1. Chunking Stage
        ####################################################################################################
        chunks = await self._chunk_documents(docs)

        ####################################################################################################
        # 2. Building Graph Stage
        ####################################################################################################
        logger.info(f"Starting build graph for the given documents")
        await self.graph.build_graph(chunks)
        await self.graph.persist_graph()
        logger.info(f"Finished building graph for the given documents")

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
