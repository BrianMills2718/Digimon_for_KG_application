from typing import Optional, Dict, Any

from Core.Utils.YamlModel import YamlModel
from pydantic import Field


class GraphConfig(YamlModel):
    type: str = Field(
        default="er_graph",
        description="Type of graph to build (e.g., 'er_graph', 'rkg_graph', 'tree_graph').",
    )
    graph_type: str = "er_graph"

    # Building graph
    extract_two_step: bool = False
    max_gleaning: int = 1
    force: bool = False

    # ER/RK extraction metadata. The single-step prompt already extracts these
    # fields, so preserve them by default instead of paying for extraction and
    # discarding the result before retrieval/indexing.
    enable_entity_description: bool = True
    enable_entity_type: bool = True
    enable_edge_description: bool = True
    # The single-step prompt does not emit a typed relation name; keep this off
    # unless a caller supplies a graph mode/ontology that does.
    enable_edge_name: bool = False
    prior_prob: float = 0.8
    # ER stays keyword-free by default. RK construction enables this explicitly.
    enable_edge_keywords: bool = False

    # Graph clustering
    use_community: bool = False
    graph_cluster_algorithm: str = "leiden"
    max_graph_cluster_size: int = 10
    graph_cluster_seed: int = 0xDEADBEEF
    summary_max_tokens: int = 500
    llm_model_max_token_size: int = 32768

    # Tree graph config
    build_tree_from_leaves: bool = False
    reduction_dimension: int = 5
    summarization_length: int = 100
    num_layers: int = 10
    top_k: int = 5
    threshold_cluster_num: int = 5000
    start_layer: int = 5
    graph_cluster_params: Optional[dict] = None
    selection_mode: str = "top_k"
    max_length_in_cluster: int = 3500
    threshold: float = 0.1
    cluster_metric: str = "cosine"
    verbose: bool = False
    random_seed: int = 224
    enforce_sub_communities: bool = False
    max_size_percentage: float = 0.2
    tol: float = 1e-4
    max_iter: int = 300
    size_of_clusters: int = 10

    # Custom ontology
    auto_generate_ontology: bool = False
    custom_ontology_path: Optional[str] = "Config/custom_ontology.json"
    loaded_custom_ontology: Optional[Dict[str, Any]] = None

    # Graph augmentation
    similarity_threshold: float = 0.8
    similarity_top_k: int = 10
    similarity_max: float = 1.0
