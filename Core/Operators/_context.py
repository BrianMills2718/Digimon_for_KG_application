"""OperatorContext — shared state passed to all operators.

This replaces the various self.* attributes that retrievers used to access
via class inheritance. Every operator gets the same context object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class OperatorContext:
    """Shared state available to all operators."""
    graph: Any  # NetworkXStorage / BaseGraphStorage instance
    entities_vdb: Optional[Any] = None  # BaseIndex for entity search
    relations_vdb: Optional[Any] = None  # BaseIndex for relationship search
    doc_chunks: Optional[Any] = None  # ChunkKVStorage for text retrieval
    community: Optional[Any] = None  # Community detection results
    llm: Optional[Any] = None  # BaseLLM instance
    config: Optional[Any] = None  # RetrieverConfig / QueryConfig
    sparse_matrices: Optional[Dict[str, Any]] = field(default_factory=dict)
    # sparse_matrices keys: "entity_to_rel", "rel_to_chunk" (scipy CSR)
