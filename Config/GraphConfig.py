from typing import Optional

from Core.Utils.YamlModel import YamlModel

class GraphConfig(YamlModel):

    graph_type: str = "er_graph"
    # Building graph
    extract_two_step: bool = True
    max_gleaning: int = 1
    force: bool = False
    
    # For ER graph & KG graph and & RKG graph
    enable_entity_description: bool = False
    enable_entity_type: bool = False
    enable_edge_description: bool = False 
    enable_edge_name: bool = False
    prior_prob: float = 0.8
    enable_edge_keywords: bool = False
    # Graph clustering
    use_community: bool = False # Defaut to False
    graph_cluster_algorithm: str = "leiden"
    max_graph_cluster_size: int = 10
    graph_cluster_seed: int = 0xDEADBEEF
    summary_max_tokens: int = 500
    llm_model_max_token_size:int =  32768

    # For Tree graph config 
    reduction_dimension: int = 5
    summarization_length: int = 100
    num_layers: int = 10
    top_k: int = 5
    start_layer: int = 5
    graph_cluster_params: Optional[dict] = None
    selection_mode: str = "top_k"
    max_length_in_cluster: int = 3500
    threshold: float = 0.1
    cluster_metric: str = "cosine"
    verbose: bool = False
    random_seed: int = 224
    enforce_sub_communities: bool = False