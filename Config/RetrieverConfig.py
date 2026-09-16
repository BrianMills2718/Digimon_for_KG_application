from Core.Utils.YamlModel import YamlModel


class RetrieverConfig(YamlModel):
    """Retrieval configuration shared by maintained operator pipelines."""

    query_type: str = "ppr"
    enable_local: bool = False

    # PPR / graph diffusion
    use_entity_similarity_for_ppr: bool = True
    top_k_entity_for_ppr: int = 8
    node_specificity: bool = True
    # Probability of following graph links. Standard PageRank uses 0.85,
    # corresponding to a 0.15 teleport/reset probability.
    damping: float = 0.85

    # Ranking / neighborhood
    top_k: int = 5
    k_nei: int = 3

    # Context budgets
    max_token_for_local_context: int = 4800
    max_token_for_global_context: int = 4000
    local_max_token_for_text_unit: int = 4000
    local_max_token_for_community_report: int = 3200

    # Optional resources
    use_relations_vdb: bool = False
    use_subgraphs_vdb: bool = False

    # Community retrieval
    global_max_consider_community: int = 512
    global_min_community_rating: float = 0.0
    local_community_single_one: bool = False
