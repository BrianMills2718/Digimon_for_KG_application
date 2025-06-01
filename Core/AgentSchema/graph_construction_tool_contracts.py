"""
Pydantic contracts for agent tools that construct various types of knowledge graphs.
Each tool has an input and output schema, with config overrides for graph-specific parameters.
"""
from pydantic import BaseModel, Field
from typing import Optional

# =========================
# Base Output Schema
# =========================
class BaseGraphBuildOutputs(BaseModel):
    """Common output fields for all graph build tools."""
    graph_id: str = Field(description="Unique identifier for the built graph artifact. This ID will be used by retrieval tools and other graph operations.")
    status: str = Field(description="Status of the build operation, e.g., 'success', 'failure'.")
    message: str = Field(description="A descriptive message about the outcome of the build operation, including any errors.")
    node_count: Optional[int] = Field(default=None, description="Number of nodes in the built graph.")
    edge_count: Optional[int] = Field(default=None, description="Number of edges in the built graph (if applicable).")
    layer_count: Optional[int] = Field(default=None, description="Number of layers in the built graph (for tree graphs).")
    artifact_path: Optional[str] = Field(default=None, description="Path to the primary persisted graph artifact.")

# =========================
# ERGraph
# =========================
class ERGraphConfigOverrides(BaseModel):
    extract_two_step: Optional[bool] = Field(default=None, description="Override default for two-step entity/relation extraction.")
    enable_entity_description: Optional[bool] = Field(default=None, description="Override for enabling entity descriptions.")
    enable_entity_type: Optional[bool] = Field(default=None, description="Override for enabling entity types.")
    enable_edge_description: Optional[bool] = Field(default=None, description="Override for enabling edge descriptions.")
    enable_edge_name: Optional[bool] = Field(default=None, description="Override for enabling edge names.")
    custom_ontology_path_override: Optional[str] = Field(default=None, description="Path to a custom ontology JSON file to use for this build.")

class BuildERGraphInputs(BaseModel):
    target_dataset_name: str = Field(description="Name of the dataset for input chunks and namespacing artifacts.")
    force_rebuild: bool = Field(default=False, description="If True, forces a rebuild even if artifacts exist.")
    config_overrides: Optional[ERGraphConfigOverrides] = Field(default=None, description="Specific configuration overrides for ERGraph building.")

class BuildERGraphOutputs(BaseGraphBuildOutputs):
    pass

# =========================
# RKGraph
# =========================
class RKGraphConfigOverrides(BaseModel):
    enable_edge_keywords: Optional[bool] = Field(default=None, description="Selects between ENTITY_EXTRACTION and ENTITY_EXTRACTION_KEYWORD prompts.")
    max_gleaning: Optional[int] = Field(default=None, description="Maximum number of gleaning iterations or items.")
    custom_ontology_path_override: Optional[str] = Field(default=None, description="Path to a custom ontology JSON file to use for this build.")
    enable_entity_description: Optional[bool] = Field(default=None, description="Override for enabling entity descriptions (if applicable).")

class BuildRKGraphInputs(BaseModel):
    target_dataset_name: str = Field(description="Name of the dataset for input chunks and namespacing artifacts.")
    force_rebuild: bool = Field(default=False, description="If True, forces a rebuild even if artifacts exist.")
    config_overrides: Optional[RKGraphConfigOverrides] = Field(default=None, description="Specific configuration overrides for RKGraph building.")

class BuildRKGraphOutputs(BaseGraphBuildOutputs):
    pass

# =========================
# TreeGraph
# =========================
class TreeGraphConfigOverrides(BaseModel):
    build_tree_from_leaves: Optional[bool] = Field(default=None, description="If True, build tree from leaves upward.")
    num_layers: Optional[int] = Field(default=None, description="Number of layers in the tree.")
    reduction_dimension: Optional[int] = Field(default=None, description="UMAP reduction dimension.")
    threshold: Optional[float] = Field(default=None, description="GMM clustering threshold.")
    summarization_length: Optional[int] = Field(default=None, description="Max tokens for LLM summary.")
    max_length_in_cluster: Optional[int] = Field(default=None, description="Max items per cluster for recursive clustering.")
    cluster_metric: Optional[str] = Field(default=None, description="Clustering metric, e.g., 'cosine'.")
    random_seed: Optional[int] = Field(default=None, description="Random seed for reproducibility.")

class BuildTreeGraphInputs(BaseModel):
    target_dataset_name: str = Field(description="Name of the dataset for input chunks and namespacing artifacts.")
    force_rebuild: bool = Field(default=False, description="If True, forces a rebuild even if artifacts exist.")
    config_overrides: Optional[TreeGraphConfigOverrides] = Field(default=None, description="Specific configuration overrides for TreeGraph building.")

class BuildTreeGraphOutputs(BaseGraphBuildOutputs):
    pass

# =========================
# TreeGraphBalanced
# =========================
class TreeGraphBalancedConfigOverrides(BaseModel):
    build_tree_from_leaves: Optional[bool] = Field(default=None, description="If True, build tree from leaves upward.")
    num_layers: Optional[int] = Field(default=None, description="Number of layers in the tree.")
    summarization_length: Optional[int] = Field(default=None, description="Max tokens for LLM summary.")
    size_of_clusters: Optional[int] = Field(default=None, description="Target items per cluster for balanced K-Means.")
    max_size_percentage: Optional[float] = Field(default=None, description="Allowed deviation for cluster balancing.")
    max_iter: Optional[int] = Field(default=None, description="K-Means max iterations.")
    tol: Optional[float] = Field(default=None, description="K-Means tolerance.")
    random_seed: Optional[int] = Field(default=None, description="Random seed for reproducibility.")

class BuildTreeGraphBalancedInputs(BaseModel):
    target_dataset_name: str = Field(description="Name of the dataset for input chunks and namespacing artifacts.")
    force_rebuild: bool = Field(default=False, description="If True, forces a rebuild even if artifacts exist.")
    config_overrides: Optional[TreeGraphBalancedConfigOverrides] = Field(default=None, description="Specific configuration overrides for TreeGraphBalanced building.")

class BuildTreeGraphBalancedOutputs(BaseGraphBuildOutputs):
    pass

# =========================
# PassageGraph
# =========================
class PassageGraphConfigOverrides(BaseModel):
    prior_prob: Optional[float] = Field(default=None, description="Threshold for WAT entity annotations.")
    custom_ontology_path_override: Optional[str] = Field(default=None, description="Path to a custom ontology JSON file to use for this build.")

class BuildPassageGraphInputs(BaseModel):
    target_dataset_name: str = Field(description="Name of the dataset for input chunks and namespacing artifacts.")
    force_rebuild: bool = Field(default=False, description="If True, forces a rebuild even if artifacts exist.")
    config_overrides: Optional[PassageGraphConfigOverrides] = Field(default=None, description="Specific configuration overrides for PassageGraph building.")

class BuildPassageGraphOutputs(BaseGraphBuildOutputs):
    pass
