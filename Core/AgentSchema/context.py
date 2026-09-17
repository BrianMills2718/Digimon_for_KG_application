# Core/AgentSchema/context.py

import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from Core.Chunk.ChunkFactory import ChunkFactory
from Core.Common.Logger import logger
from Core.Graph.BaseGraph import BaseGraph
from Core.Index.BaseIndex import BaseIndex
from Core.Provider.BaseLLM import BaseLLM
from Option.Config2 import Config as FullConfig
from llama_index.core.embeddings import BaseEmbedding as LlamaIndexBaseEmbedding


_GRAPH_SUFFIXES = (
    "_TreeGraphBalanced",
    "_PassageGraph",
    "_ERGraph",
    "_RKGraph",
    "_TreeGraph",
)


def _dataset_from_graph_id(graph_id: str) -> Optional[str]:
    for suffix in _GRAPH_SUFFIXES:
        if graph_id.endswith(suffix):
            return graph_id[: -len(suffix)]
    return None


class GraphRAGContext(BaseModel):
    """Runtime resources used by DIGIMON tools and operator plans."""

    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Unique identifier for this context instance",
    )
    target_dataset_name: str = Field(
        description="The name of the target dataset for the current plan."
    )
    main_config: FullConfig = Field(description="The main configuration object.")
    llm_provider: Optional[BaseLLM] = Field(default=None)
    embedding_provider: Optional[LlamaIndexBaseEmbedding] = Field(default=None)
    chunk_storage_manager: Optional[ChunkFactory] = Field(default=None)

    graphs: Dict[str, BaseGraph] = Field(default_factory=dict)
    vdbs: Dict[str, BaseIndex] = Field(default_factory=dict)
    resolved_configs: Dict[str, Any] = Field(default_factory=dict)
    active_dataset_name: Optional[str] = Field(
        default=None,
        exclude=True,
        description="Dataset most recently selected through a concrete graph lookup.",
    )

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True

    def add_graph_instance(self, graph_id: str, graph_instance: BaseGraph):
        self.graphs[graph_id] = graph_instance
        logger.info(
            f"GraphRAGContext: Added graph '{graph_id}' (type: {type(graph_instance)}). "
            f"Available graphs: {list(self.graphs.keys())}"
        )

    def get_graph_instance(self, graph_id: str) -> Optional[BaseGraph]:
        instance = self.graphs.get(graph_id)
        if instance is not None:
            dataset_name = _dataset_from_graph_id(graph_id)
            if dataset_name:
                self.active_dataset_name = dataset_name
            logger.debug(f"GraphRAGContext: Retrieved graph '{graph_id}'.")
        else:
            logger.warning(
                f"GraphRAGContext: Graph ID '{graph_id}' not found. "
                f"Available: {list(self.graphs.keys())}"
            )
        return instance

    def add_vdb_instance(self, vdb_id: str, vdb_instance: BaseIndex):
        self.vdbs[vdb_id] = vdb_instance
        logger.info(
            f"GraphRAGContext: Added VDB '{vdb_id}' (type: {type(vdb_instance)}). "
            f"Available VDBs: {list(self.vdbs.keys())}"
        )

    def get_vdb_instance(self, vdb_id: str) -> Optional[BaseIndex]:
        instance = self.vdbs.get(vdb_id)
        if instance is None:
            logger.warning(
                f"GraphRAGContext: VDB ID '{vdb_id}' not found. "
                f"Available: {list(self.vdbs.keys())}"
            )
        return instance

    def list_graphs(self) -> List[str]:
        """List graph IDs with more specific dataset names considered last.

        A few transitional MCP helpers still locate a dataset graph using
        substring matching (``if dataset_name in graph_id``). Ordering by the
        parsed dataset-name length makes the exact dataset graph the first
        matching resource: ``Test_ERGraph`` precedes ``Test2_ERGraph`` when the
        requested dataset is ``Test``. Insertion order is preserved among graph
        types belonging to the same dataset.
        """
        keys = list(self.graphs.keys())
        positions = {key: index for index, key in enumerate(keys)}
        return sorted(
            keys,
            key=lambda key: (
                len(_dataset_from_graph_id(key) or key),
                positions[key],
            ),
        )

    def list_vdbs(self) -> List[str]:
        """List all VDB IDs, prioritizing the currently active dataset.

        The MCP server may hold resources for multiple datasets in one process.
        Existing callers often choose the first entity/relationship VDB from this
        list. Prioritizing the dataset selected by the most recent graph lookup
        preserves the complete resource list while preventing cross-dataset
        index selection in those callers.
        """
        keys = list(self.vdbs.keys())
        dataset = self.active_dataset_name
        if not dataset:
            return keys

        prefix = f"{dataset}_"
        positions = {key: index for index, key in enumerate(keys)}
        return sorted(
            keys,
            key=lambda key: (not key.startswith(prefix), positions[key]),
        )
