"""Operator metadata descriptors for agent-readable operator capabilities.

OperatorDescriptor tells an agent what an operator does, what it needs,
and what it produces — enabling composition validation and planning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional

from Core.Schema.SlotTypes import SlotKind


@dataclass
class SlotSpec:
    name: str
    kind: SlotKind
    required: bool = True
    description: str = ""
    field_requirements: Dict[str, str] = field(default_factory=dict)


class CostTier(str, Enum):
    FREE = "free"        # graph traversal, no I/O
    CHEAP = "cheap"      # VDB lookup, matrix math
    MODERATE = "moderate"  # single LLM call
    EXPENSIVE = "expensive"  # multiple LLM calls or iteration


@dataclass
class OperatorDescriptor:
    operator_id: str
    display_name: str
    category: str  # entity|relationship|chunk|subgraph|community|meta
    input_slots: List[SlotSpec]
    output_slots: List[SlotSpec]
    cost_tier: CostTier = CostTier.CHEAP
    requires_llm: bool = False
    requires_entity_vdb: bool = False
    requires_relationship_vdb: bool = False
    requires_community: bool = False
    requires_sparse_matrices: bool = False
    when_to_use: str = ""
    limitations: str = ""
    implementation: Optional[Callable] = field(default=None, repr=False)
