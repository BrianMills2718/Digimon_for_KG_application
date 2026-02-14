"""Typed slot system for operator I/O compatibility.

SlotKind defines the categories of data that flow between operators.
Record dataclasses define the uniform structure for each kind.
SlotValue wraps any record list with metadata about its producer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class SlotKind(str, Enum):
    QUERY_TEXT = "query_text"
    ENTITY_SET = "entity_set"
    RELATIONSHIP_SET = "relationship_set"
    CHUNK_SET = "chunk_set"
    SUBGRAPH = "subgraph"
    COMMUNITY_SET = "community_set"
    SCORE_VECTOR = "score_vector"  # np.ndarray aligned to graph indices


@dataclass
class EntityRecord:
    entity_name: str
    source_id: str = ""
    entity_type: str = ""
    description: str = ""
    rank: int = 0
    score: Optional[float] = None
    clusters: Optional[List[Dict]] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RelationshipRecord:
    src_id: str
    tgt_id: str
    relation_name: str = ""
    description: str = ""
    weight: float = 0.0
    keywords: str = ""
    source_id: str = ""
    score: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkRecord:
    chunk_id: str
    text: str
    score: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SubgraphRecord:
    nodes: Set[str]
    edges: List[Tuple[str, str]]
    paths: Optional[List[List[str]]] = None
    nx_graph: Optional[Any] = None


@dataclass
class CommunityRecord:
    community_id: str
    level: int
    title: str = ""
    report: str = ""
    occurrence: float = 0.0
    rating: float = 0.0
    nodes: Set[str] = field(default_factory=set)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SlotValue:
    kind: SlotKind
    data: Any  # typed payload matching kind
    producer: str = ""  # operator_id that produced this
    metadata: Dict[str, Any] = field(default_factory=dict)
