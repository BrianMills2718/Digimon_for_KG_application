# Core/AgentSchema/context.py

import uuid
from pydantic import BaseModel, Field
from typing import Any, Optional, Dict, List
from Core.Graph.BaseGraph import BaseGraph
from Core.Index.BaseIndex import BaseIndex
from Option.Config2 import Config as FullConfig # Use FullConfig alias
from Core.Provider.BaseLLM import BaseLLM
from llama_index.core.embeddings import BaseEmbedding as LlamaIndexBaseEmbedding
from Core.Chunk.ChunkFactory import ChunkFactory # If passed as chunk_storage_manager
from Core.Common.Logger import logger

class GraphRAGContext(BaseModel):
    """
    Provides the necessary context and access to GraphRAG system components
    for executing individual tools within an ExecutionPlan.
    This object will be instantiated and passed by the Agent Orchestrator.
    """

    request_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8], description="Unique identifier for this context instance")
    target_dataset_name: str = Field(description="The name of the target dataset for the current plan.")
    main_config: FullConfig = Field(description="The main configuration object.")
    llm_provider: Optional[BaseLLM] = Field(default=None, description="LLM provider instance.")
    embedding_provider: Optional[LlamaIndexBaseEmbedding] = Field(default=None, description="Embedding provider instance.")
    chunk_storage_manager: Optional[ChunkFactory] = Field(default=None, description="ChunkFactory instance for chunk access.") # Changed type

    graphs: Dict[str, BaseGraph] = Field(default_factory=dict, description="Stores graph_id: graph_instance pairs.")
    vdbs: Dict[str, BaseIndex] = Field(default_factory=dict, description="Stores vdb_id: vdb_instance pairs.")

    resolved_configs: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True 

    def add_graph_instance(self, graph_id: str, graph_instance: BaseGraph):
        logger.debug(f"GraphRAGContext: Before adding graph '{graph_id}'. Current graphs: {list(self.graphs.keys())}")
        self.graphs[graph_id] = graph_instance
        logger.info(f"GraphRAGContext: Added graph '{graph_id}' (type: {type(graph_instance)}). Updated graphs: {list(self.graphs.keys())}")

    def get_graph_instance(self, graph_id: str) -> Optional[BaseGraph]:
        logger.debug(f"GraphRAGContext: Attempting to get graph '{graph_id}'. Current available graphs: {list(self.graphs.keys())}")
        instance = self.graphs.get(graph_id)
        if instance:
            logger.debug(f"GraphRAGContext: Successfully retrieved graph '{graph_id}'.")
        else:
            logger.warning(f"GraphRAGContext: Graph ID '{graph_id}' not found. Available: {list(self.graphs.keys())}")
        return instance

    def add_vdb_instance(self, vdb_id: str, vdb_instance: BaseIndex):
        logger.debug(f"GraphRAGContext: Before adding VDB '{vdb_id}'. Current VDBs: {list(self.vdbs.keys())}")
        self.vdbs[vdb_id] = vdb_instance
        logger.info(f"GraphRAGContext: Added VDB '{vdb_id}' (type: {type(vdb_instance)}). Updated VDBs: {list(self.vdbs.keys())}")

    def get_vdb_instance(self, vdb_id: str) -> Optional[BaseIndex]:
        logger.debug(f"GraphRAGContext: Attempting to get VDB '{vdb_id}'. Current available VDBs: {list(self.vdbs.keys())}")
        instance = self.vdbs.get(vdb_id)
        if instance:
            logger.debug(f"GraphRAGContext: Successfully retrieved VDB '{vdb_id}'.")
        else:
            logger.warning(f"GraphRAGContext: VDB ID '{vdb_id}' not found. Available: {list(self.vdbs.keys())}")
        return instance

    def list_graphs(self) -> List[str]:
        keys = list(self.graphs.keys())
        logger.debug(f"GraphRAGContext: Listing all graph keys. Found: {keys}")
        return keys

    def list_vdbs(self) -> List[str]:
        keys = list(self.vdbs.keys())
        logger.debug(f"GraphRAGContext: Listing all VDB keys. Found: {keys}")
        return keys
