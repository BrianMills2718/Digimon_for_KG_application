"""Graph capability flags for operator compatibility checking.

Each graph type exposes a set of capabilities. Operators declare which
capabilities they require, enabling validation before execution.
"""

from __future__ import annotations

from enum import Enum


class GraphCapability(str, Enum):
    HAS_ENTITY_TYPES = "has_entity_types"
    HAS_DESCRIPTIONS = "has_descriptions"
    HAS_EDGE_KEYWORDS = "has_edge_keywords"
    HAS_EDGE_DESCRIPTIONS = "has_edge_descriptions"
    HAS_COMMUNITIES = "has_communities"
    HAS_TREE_LAYERS = "has_tree_layers"
    HAS_PASSAGES = "has_passages"
    HAS_EMBEDDINGS = "has_embeddings"
    SUPPORTS_PPR = "supports_ppr"
    SUPPORTS_SUBGRAPH = "supports_subgraph"
