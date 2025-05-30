# Core/AgentSchema/context.py

from pydantic import BaseModel, Field
from typing import Any, Optional, Dict

# Import specific configuration models if they will be directly part of the context
# from Option.Config2 import Config as FullConfig # Example: If the whole config is passed
# from Core.Config.LLMConfig import LLMConfig #
# from Core.Config.EmbConfig import EmbeddingConfig #

# Placeholder types for complex components that will be properly typed later
# For example, when we decide on LLM/Embedding provider abstractions (e.g., via LiteLLM)
# or for storage/index manager classes.
LLMProviderType = Any 
EmbeddingProviderType = Any
VectorDBManagerType = Any # Could provide methods like .get_vdb(vdb_reference_id)
GraphStorageManagerType = Any # Could provide methods like .get_graph(graph_reference_id)
ChunkStorageManagerType = Any # Could provide methods like .get_chunk_store(document_collection_id)
CommunityStorageManagerType = Any # Could provide methods like .get_community_hierarchy(ref_id)


class GraphRAGContext(BaseModel):
    """
    Provides the necessary context and access to GraphRAG system components
    for executing individual tools within an ExecutionPlan.
    This object will be instantiated and passed by the Agent Orchestrator.
    """

    target_dataset_name: str = Field(description="The name of the target dataset for the current plan, used for resolving artifact paths.")

    # Access to overall system configurations
    # Option 1: Pass the fully parsed main config object (e.g., from Option/Config2.py)
    # main_config: FullConfig 
    # Option 2: Pass specific config sections or derived configurations
    # llm_config: LLMConfig
    # embedding_config: EmbeddingConfig
    # For now, let's use a more generic config dictionary that the orchestrator can populate.
    # The orchestrator can load the main_config and method-specific YAMLs, apply patches from
    # the ExecutionPlan, and then provide the relevant resolved configs here.
    resolved_configs: Dict[str, Any] = Field(default_factory=dict, description="Dictionary holding resolved configurations (e.g., LLM, embedding, specific tool configs) for the current step/plan after patches.")

    # Access to service providers and managers
    llm_provider: Optional[LLMProviderType] = Field(default=None, description="Provider for making LLM calls.")
    embedding_provider: Optional[EmbeddingProviderType] = Field(default=None, description="Provider for generating embeddings.")

    vector_db_manager: Optional[VectorDBManagerType] = Field(default=None, description="Manager to access various vector database instances.")
    graph_storage_manager: Optional[GraphStorageManagerType] = Field(default=None, description="Manager to access various graph storage instances.")
    chunk_storage_manager: Optional[ChunkStorageManagerType] = Field(default=None, description="Manager to access chunk stores.")
    community_storage_manager: Optional[CommunityStorageManagerType] = Field(default=None, description="Manager to access community data artifacts.")

    # Potentially a workspace for the current plan execution to store/retrieve
    # intermediate results if not handled solely by ToolCall.inputs/outputs passing.
    # plan_workspace: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True # Allow types like Any and future complex types for managers/providers
