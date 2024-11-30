from abc import ABC, abstractmethod
from Core.Common.Utils import mdhash_id
from typing import Any, Optional, Union, Type
from Core.Common.ContextMixin import ContextMixin

from pydantic import BaseModel, ConfigDict, Field, model_validator
from Core.Graph.ChunkFactory import get_chunks
import tiktoken
from Core.Common.Memory import Memory


class BaseGraph(ABC, ContextMixin, BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    chunks: Optional[str] = None
    # context: str  # all the context, including all necessary info
    llm_name_or_type: Optional[str] = None
    # working memory for constructing the graph
    working_memory: Memory = Memory()

    @model_validator(mode="after")
    def _update_context(
            cls: Type["BaseGraph"], data: "BaseGraph"
    ) -> "BaseGraph":

        cls.config = data.context.config
        cls.ENCODER = tiktoken.encoding_for_model(cls.config.token_model)
        return data

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
        self.chunks = chunks
        return chunks

    async def build_graph(self, chunks):
        """
        Builds or loads a graph based on the input chunks.

        Args:
            chunks: The input data chunks used to build the graph.

        Returns:
            The graph if it already exists, otherwise builds and returns the graph.
        """
        # If the graph already exists, load it
        if self._exist_graph():
            return self._load_graph()

        # Build the graph based on the input chunks
        self._build_graph(chunks)

    def _exist_graph(self):
        """
        Checks if the graph already exists.

        Returns:
            bool: True if the graph exists, False otherwise.
        """
        pass

    @abstractmethod
    def _extract_node_relationship(self):
        """
        Abstract method to extract relationships between nodes in the graph.

        This method should be implemented by subclasses to define how node relationships are extracted.
        """
        pass

    @abstractmethod
    def _build_graph(self, chunks):
        """
        Abstract method to build the graph based on the input chunks.

        Args:
            chunks: The input data chunks used to build the graph.

        This method should be implemented by subclasses to define how the graph is built from the input chunks.
        """
        pass
