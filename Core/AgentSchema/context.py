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


_GRAPH_SUFFIX_TO_TYPE = (
    ("_TreeGraphBalanced", "tree_graph_balanced"),
    ("_PassageGraph", "passage_graph"),
    ("_ERGraph", "er_graph"),
    ("_RKGraph", "rkg_graph"),
    ("_TreeGraph", "tree_graph"),
)
_GRAPH_SUFFIXES = tuple(suffix for suffix, _ in _GRAPH_SUFFIX_TO_TYPE)


def _dataset_from_graph_id(graph_id: str) -> Optional[str]:
    for suffix in _GRAPH_SUFFIXES:
        if graph_id.endswith(suffix):
            return graph_id[: -len(suffix)]
    return None


def _graph_type_from_graph_id(graph_id: str) -> Optional[str]:
    for suffix, graph_type in _GRAPH_SUFFIX_TO_TYPE:
        if graph_id.endswith(suffix):
            return graph_type
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
    active_graph_id: Optional[str] = Field(
        default=None,
        exclude=True,
        description="Graph most recently built/retrieved for the active dataset.",
    )

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True

    def invalidate_dataset_vdbs(self, dataset_name: str) -> List[str]:
        """Evict in-memory VDBs belonging to ``dataset_name``.

        This is intentionally small and name-based. A rebuilt graph must not keep
        serving an in-memory vector index derived from the previous graph. Disk
        invalidation for the canonical VDB paths happens in the graph build tool.
        """
        prefix = f"{dataset_name}_"
        removed = [vdb_id for vdb_id in list(self.vdbs) if vdb_id.startswith(prefix)]
        for vdb_id in removed:
            self.vdbs.pop(vdb_id, None)
        if removed:
            logger.info(
                f"GraphRAGContext: invalidated dataset VDBs for '{dataset_name}': {removed}"
            )
        return removed

    def _activate_graph(self, graph_id: str) -> None:
        dataset_name = _dataset_from_graph_id(graph_id)
        if dataset_name:
            self.active_dataset_name = dataset_name
            self.active_graph_id = graph_id

    def add_graph_instance(self, graph_id: str, graph_instance: BaseGraph):
        """Register a graph and restore its canonical dataset/type namespace.

        Transitional MCP registration code may assign a generic ER namespace
        immediately before calling this method. The graph ID is the authoritative
        resource identity, so repair the namespace here for ER/RK/tree/passage
        graphs rather than allowing a non-ER graph to read/write ER artifacts.

        Replacing a graph object for an already-registered graph ID also evicts
        that dataset's in-memory VDBs. They may have been derived from the previous
        graph and must be rebuilt or reloaded after graph replacement.
        """
        dataset_name = _dataset_from_graph_id(graph_id)
        graph_type = _graph_type_from_graph_id(graph_id)
        storage = getattr(graph_instance, "_graph", None)
        if (
            dataset_name
            and graph_type
            and storage is not None
            and hasattr(storage, "namespace")
            and self.chunk_storage_manager is not None
        ):
            storage.namespace = self.chunk_storage_manager.get_namespace(
                dataset_name,
                graph_type=graph_type,
            )

        previous_graph = self.graphs.get(graph_id)
        if (
            dataset_name
            and previous_graph is not None
            and previous_graph is not graph_instance
        ):
            self.invalidate_dataset_vdbs(dataset_name)

        self.graphs[graph_id] = graph_instance
        self._activate_graph(graph_id)
        logger.info(
            f"GraphRAGContext: Added graph '{graph_id}' (type: {type(graph_instance)}). "
            f"Available graphs: {list(self.graphs.keys())}"
        )

    def get_graph_instance(self, graph_id: str) -> Optional[BaseGraph]:
        instance = self.graphs.get(graph_id)
        if instance is not None:
            self._activate_graph(graph_id)
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
        """List graph IDs with exact-dataset and active-graph priority.

        Transitional MCP helpers still locate dataset graphs by iterating this
        list and taking the first substring match. Shorter parsed dataset names
        therefore come first so ``Test_ERGraph`` precedes ``Test2_ERGraph`` for a
        request targeting ``Test``. Within one dataset, the graph most recently
        built/retrieved is first, allowing an explicit RK/tree/etc. selection to
        survive generic context construction instead of being hidden by insertion
        order.
        """
        keys = list(self.graphs.keys())
        positions = {key: index for index, key in enumerate(keys)}
        return sorted(
            keys,
            key=lambda key: (
                len(_dataset_from_graph_id(key) or key),
                key != self.active_graph_id,
                positions[key],
            ),
        )

    def list_vdbs(self) -> List[str]:
        """List all VDB IDs with canonical active-dataset indexes first.

        Transitional MCP context construction chooses the first entity/relation
        VDB it sees. For the active dataset, the maintained canonical IDs
        ``<dataset>_entities`` and ``<dataset>_relations`` must therefore precede
        old/custom indexes such as ``<dataset>_entities_old``. All registered
        resources remain visible after those canonical entries.
        """
        keys = list(self.vdbs.keys())
        dataset = self.active_dataset_name
        if not dataset:
            return keys

        prefix = f"{dataset}_"
        canonical_entities = f"{dataset}_entities"
        canonical_relations = f"{dataset}_relations"
        positions = {key: index for index, key in enumerate(keys)}

        def priority(key: str):
            if key == canonical_entities:
                return (0, positions[key])
            if key == canonical_relations:
                return (1, positions[key])
            if key.startswith(prefix):
                return (2, positions[key])
            return (3, positions[key])

        return sorted(keys, key=priority)
