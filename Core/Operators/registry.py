"""Operator Registry — machine-readable catalog of all operators.

The registry provides:
- Operator lookup by ID and category
- I/O compatibility checking between operators
- Chain discovery: find valid operator sequences from input to goal
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

from Core.Schema.OperatorDescriptor import CostTier, OperatorDescriptor, SlotSpec
from Core.Schema.SlotTypes import SlotKind


class OperatorRegistry:
    def __init__(self):
        self._operators: Dict[str, OperatorDescriptor] = {}

    def register(self, descriptor: OperatorDescriptor):
        self._operators[descriptor.operator_id] = descriptor

    def get(self, operator_id: str) -> Optional[OperatorDescriptor]:
        return self._operators.get(operator_id)

    def list_all(self) -> List[OperatorDescriptor]:
        return list(self._operators.values())

    def get_by_category(self, category: str) -> List[OperatorDescriptor]:
        return [d for d in self._operators.values() if d.category == category]

    # --- Composition helpers ---

    def _output_kinds(self, op_id: str) -> Set[SlotKind]:
        op = self._operators.get(op_id)
        if not op:
            return set()
        return {s.kind for s in op.output_slots}

    def _input_kinds(self, op_id: str) -> Set[SlotKind]:
        op = self._operators.get(op_id)
        if not op:
            return set()
        return {s.kind for s in op.input_slots if s.required}

    def get_compatible_successors(self, operator_id: str) -> List[OperatorDescriptor]:
        """Find operators that can consume any output of the given operator."""
        output_kinds = self._output_kinds(operator_id)
        if not output_kinds:
            return []
        result = []
        for op in self._operators.values():
            if op.operator_id == operator_id:
                continue
            required_inputs = {s.kind for s in op.input_slots if s.required}
            if required_inputs & output_kinds:
                result.append(op)
        return result

    def get_compatible_predecessors(self, operator_id: str) -> List[OperatorDescriptor]:
        """Find operators whose outputs can satisfy any required input of the given operator."""
        required = self._input_kinds(operator_id)
        if not required:
            return []
        result = []
        for op in self._operators.values():
            if op.operator_id == operator_id:
                continue
            outputs = {s.kind for s in op.output_slots}
            if outputs & required:
                result.append(op)
        return result

    def validate_connection(
        self, src_op: str, src_slot: str, tgt_op: str, tgt_slot: str
    ) -> bool:
        """Check if src_op's output slot can connect to tgt_op's input slot."""
        src = self._operators.get(src_op)
        tgt = self._operators.get(tgt_op)
        if not src or not tgt:
            return False

        src_spec = next((s for s in src.output_slots if s.name == src_slot), None)
        tgt_spec = next((s for s in tgt.input_slots if s.name == tgt_slot), None)
        if not src_spec or not tgt_spec:
            return False

        return src_spec.kind == tgt_spec.kind

    def find_chains_to_goal(
        self,
        available_inputs: Set[SlotKind],
        goal: SlotKind,
        max_depth: int = 5,
    ) -> List[List[str]]:
        """Find all valid operator chains from available inputs to goal output.

        Uses BFS to find chains up to max_depth operators long.
        """
        # BFS: state = (current available kinds, chain so far)
        from collections import deque

        queue = deque([(available_inputs, [])])
        found_chains = []
        visited = set()

        while queue:
            avail, chain = queue.popleft()
            if len(chain) >= max_depth:
                continue

            state_key = (frozenset(avail), tuple(chain))
            if state_key in visited:
                continue
            visited.add(state_key)

            for op in self._operators.values():
                # Check if all required inputs are available
                required = {s.kind for s in op.input_slots if s.required}
                if not required.issubset(avail):
                    continue

                # Extend available kinds with this op's outputs
                new_avail = avail | {s.kind for s in op.output_slots}
                new_chain = chain + [op.operator_id]

                if goal in {s.kind for s in op.output_slots}:
                    found_chains.append(new_chain)

                queue.append((new_avail, new_chain))

        return found_chains


# --- Global registry instance ---
REGISTRY = OperatorRegistry()


def _register_all():
    """Register all 21 operators with their descriptors."""
    from Core.Operators.entity.vdb import entity_vdb
    from Core.Operators.entity.ppr import entity_ppr
    from Core.Operators.entity.onehop import entity_onehop
    from Core.Operators.entity.link import entity_link
    from Core.Operators.entity.tfidf import entity_tfidf
    from Core.Operators.entity.agent import entity_agent
    from Core.Operators.entity.rel_node import entity_rel_node
    from Core.Operators.relationship.onehop import relationship_onehop
    from Core.Operators.relationship.vdb import relationship_vdb
    from Core.Operators.relationship.score_aggregator import relationship_score_agg
    from Core.Operators.relationship.agent import relationship_agent
    from Core.Operators.chunk.from_relation import chunk_from_relation
    from Core.Operators.chunk.occurrence import chunk_occurrence
    from Core.Operators.chunk.aggregator import chunk_aggregator
    from Core.Operators.subgraph.khop_paths import subgraph_khop_paths
    from Core.Operators.subgraph.steiner_tree import subgraph_steiner_tree
    from Core.Operators.subgraph.agent_path import subgraph_agent_path
    from Core.Operators.community.from_entity import community_from_entity
    from Core.Operators.community.from_level import community_from_level
    from Core.Operators.meta.extract_entities import meta_extract_entities
    from Core.Operators.meta.reason_step import meta_reason_step
    from Core.Operators.meta.rerank import meta_rerank
    from Core.Operators.meta.generate_answer import meta_generate_answer
    from Core.Operators.meta.pcst_optimize import meta_pcst_optimize

    descriptors = [
        # === Entity operators ===
        OperatorDescriptor(
            operator_id="entity.vdb",
            display_name="Entity VDB Search",
            category="entity",
            input_slots=[SlotSpec("query", SlotKind.QUERY_TEXT)],
            output_slots=[SlotSpec("entities", SlotKind.ENTITY_SET)],
            cost_tier=CostTier.CHEAP,
            requires_entity_vdb=True,
            when_to_use="Find entities semantically similar to a query. Good starting point for most pipelines.",
            implementation=entity_vdb,
        ),
        OperatorDescriptor(
            operator_id="entity.ppr",
            display_name="Entity PPR",
            category="entity",
            input_slots=[
                SlotSpec("query", SlotKind.QUERY_TEXT),
                SlotSpec("entities", SlotKind.ENTITY_SET, description="Seed entities for PPR"),
            ],
            output_slots=[
                SlotSpec("entities", SlotKind.ENTITY_SET),
                SlotSpec("score_vector", SlotKind.SCORE_VECTOR),
            ],
            cost_tier=CostTier.CHEAP,
            requires_sparse_matrices=True,
            when_to_use="Rank entities by graph topology from seed entities. Use after entity.vdb or meta.extract_entities.",
            implementation=entity_ppr,
        ),
        OperatorDescriptor(
            operator_id="entity.onehop",
            display_name="Entity One-Hop Neighbors",
            category="entity",
            input_slots=[SlotSpec("entities", SlotKind.ENTITY_SET)],
            output_slots=[SlotSpec("entities", SlotKind.ENTITY_SET)],
            cost_tier=CostTier.FREE,
            when_to_use="Expand an entity set by including immediate neighbors. Useful for local exploration.",
            implementation=entity_onehop,
        ),
        OperatorDescriptor(
            operator_id="entity.link",
            display_name="Entity Linking",
            category="entity",
            input_slots=[SlotSpec("entities", SlotKind.ENTITY_SET)],
            output_slots=[SlotSpec("entities", SlotKind.ENTITY_SET)],
            cost_tier=CostTier.CHEAP,
            requires_entity_vdb=True,
            when_to_use="Link entity mentions to canonical graph entities. Use before PPR to ground extracted entities.",
            implementation=entity_link,
        ),
        OperatorDescriptor(
            operator_id="entity.tfidf",
            display_name="Entity TF-IDF",
            category="entity",
            input_slots=[
                SlotSpec("query", SlotKind.QUERY_TEXT),
                SlotSpec("entities", SlotKind.ENTITY_SET, required=False, description="Optional candidate set"),
            ],
            output_slots=[SlotSpec("entities", SlotKind.ENTITY_SET)],
            cost_tier=CostTier.CHEAP,
            when_to_use="Rank entities by text similarity using TF-IDF. Fast alternative to VDB when descriptions are available.",
            implementation=entity_tfidf,
        ),
        OperatorDescriptor(
            operator_id="entity.agent",
            display_name="Entity Agent Scorer",
            category="entity",
            input_slots=[
                SlotSpec("query", SlotKind.QUERY_TEXT),
                SlotSpec("entity_relation_list", SlotKind.ENTITY_SET, description="Entities with relation/score/head in extra"),
            ],
            output_slots=[SlotSpec("entities", SlotKind.ENTITY_SET)],
            cost_tier=CostTier.MODERATE,
            requires_llm=True,
            when_to_use="LLM-scored entity candidates from ToG exploration. Use in iterative ToG pipeline.",
            limitations="Requires entity_relation_list with ToG-style extra fields.",
            implementation=entity_agent,
        ),
        OperatorDescriptor(
            operator_id="entity.rel_node",
            display_name="Entities from Relationships",
            category="entity",
            input_slots=[SlotSpec("relationships", SlotKind.RELATIONSHIP_SET)],
            output_slots=[SlotSpec("entities", SlotKind.ENTITY_SET)],
            cost_tier=CostTier.FREE,
            when_to_use="Extract entity endpoints from a relationship set. Use after relationship.vdb or relationship.onehop.",
            implementation=entity_rel_node,
        ),

        # === Relationship operators ===
        OperatorDescriptor(
            operator_id="relationship.onehop",
            display_name="Relationship One-Hop",
            category="relationship",
            input_slots=[SlotSpec("entities", SlotKind.ENTITY_SET)],
            output_slots=[SlotSpec("relationships", SlotKind.RELATIONSHIP_SET)],
            cost_tier=CostTier.FREE,
            when_to_use="Find all relationships connected to entities. Standard local retrieval step.",
            implementation=relationship_onehop,
        ),
        OperatorDescriptor(
            operator_id="relationship.vdb",
            display_name="Relationship VDB Search",
            category="relationship",
            input_slots=[SlotSpec("query", SlotKind.QUERY_TEXT)],
            output_slots=[SlotSpec("relationships", SlotKind.RELATIONSHIP_SET)],
            cost_tier=CostTier.CHEAP,
            requires_relationship_vdb=True,
            when_to_use="Find relationships semantically similar to query. Used in LightRAG and keyword-based methods.",
            implementation=relationship_vdb,
        ),
        OperatorDescriptor(
            operator_id="relationship.score_agg",
            display_name="Relationship Score Aggregator",
            category="relationship",
            input_slots=[
                SlotSpec("entities", SlotKind.ENTITY_SET),
                SlotSpec("score_vector", SlotKind.SCORE_VECTOR),
            ],
            output_slots=[SlotSpec("relationships", SlotKind.RELATIONSHIP_SET)],
            cost_tier=CostTier.CHEAP,
            requires_sparse_matrices=True,
            when_to_use="Propagate PPR node scores to relationships. Use after entity.ppr for FastGraphRAG pipeline.",
            implementation=relationship_score_agg,
        ),
        OperatorDescriptor(
            operator_id="relationship.agent",
            display_name="Relationship Agent",
            category="relationship",
            input_slots=[
                SlotSpec("query", SlotKind.QUERY_TEXT),
                SlotSpec("entities", SlotKind.ENTITY_SET),
            ],
            output_slots=[SlotSpec("relationships", SlotKind.RELATIONSHIP_SET)],
            cost_tier=CostTier.MODERATE,
            requires_llm=True,
            when_to_use="LLM-guided relation selection from entity edges. Used in ToG pipeline.",
            implementation=relationship_agent,
        ),

        # === Chunk operators ===
        OperatorDescriptor(
            operator_id="chunk.from_relation",
            display_name="Chunks from Relationships",
            category="chunk",
            input_slots=[SlotSpec("relationships", SlotKind.RELATIONSHIP_SET)],
            output_slots=[SlotSpec("chunks", SlotKind.CHUNK_SET)],
            cost_tier=CostTier.FREE,
            when_to_use="Extract text chunks referenced by relationship source_ids. Use after any relationship operator.",
            implementation=chunk_from_relation,
        ),
        OperatorDescriptor(
            operator_id="chunk.occurrence",
            display_name="Chunk Entity Occurrence",
            category="chunk",
            input_slots=[SlotSpec("entities", SlotKind.ENTITY_SET)],
            output_slots=[SlotSpec("chunks", SlotKind.CHUNK_SET)],
            cost_tier=CostTier.FREE,
            when_to_use="Find chunks where entities co-occur, ranked by relation density. Standard local retrieval.",
            implementation=chunk_occurrence,
        ),
        OperatorDescriptor(
            operator_id="chunk.aggregator",
            display_name="Chunk Score Aggregator",
            category="chunk",
            input_slots=[SlotSpec("score_vector", SlotKind.SCORE_VECTOR)],
            output_slots=[SlotSpec("chunks", SlotKind.CHUNK_SET)],
            cost_tier=CostTier.CHEAP,
            requires_sparse_matrices=True,
            when_to_use="Propagate PPR scores through sparse matrices to chunks. Used in FastGraphRAG/HippoRAG.",
            implementation=chunk_aggregator,
        ),

        # === Subgraph operators ===
        OperatorDescriptor(
            operator_id="subgraph.khop_paths",
            display_name="K-Hop Paths",
            category="subgraph",
            input_slots=[SlotSpec("entities", SlotKind.ENTITY_SET)],
            output_slots=[SlotSpec("subgraph", SlotKind.SUBGRAPH)],
            cost_tier=CostTier.FREE,
            when_to_use="Find k-hop neighborhoods or paths from entities. Used in ToG and GR methods.",
            implementation=subgraph_khop_paths,
        ),
        OperatorDescriptor(
            operator_id="subgraph.steiner_tree",
            display_name="Steiner Tree",
            category="subgraph",
            input_slots=[SlotSpec("entities", SlotKind.ENTITY_SET)],
            output_slots=[SlotSpec("subgraph", SlotKind.SUBGRAPH)],
            cost_tier=CostTier.CHEAP,
            when_to_use="Find minimum connecting subgraph for entities. Used in DALK method.",
            implementation=subgraph_steiner_tree,
        ),
        OperatorDescriptor(
            operator_id="subgraph.agent_path",
            display_name="Agent Path Filter",
            category="subgraph",
            input_slots=[
                SlotSpec("query", SlotKind.QUERY_TEXT),
                SlotSpec("subgraph", SlotKind.SUBGRAPH),
            ],
            output_slots=[SlotSpec("subgraph", SlotKind.SUBGRAPH)],
            cost_tier=CostTier.MODERATE,
            requires_llm=True,
            when_to_use="LLM-filter paths in a subgraph by relevance. Used in GR method.",
            implementation=subgraph_agent_path,
        ),

        # === Community operators ===
        OperatorDescriptor(
            operator_id="community.from_entity",
            display_name="Community from Entities",
            category="community",
            input_slots=[SlotSpec("entities", SlotKind.ENTITY_SET, description="Entities with clusters populated")],
            output_slots=[SlotSpec("communities", SlotKind.COMMUNITY_SET)],
            cost_tier=CostTier.FREE,
            requires_community=True,
            when_to_use="Find community reports for entities. Requires entities to have cluster memberships.",
            implementation=community_from_entity,
        ),
        OperatorDescriptor(
            operator_id="community.from_level",
            display_name="Community by Level",
            category="community",
            input_slots=[],  # config-driven
            output_slots=[SlotSpec("communities", SlotKind.COMMUNITY_SET)],
            cost_tier=CostTier.FREE,
            requires_community=True,
            when_to_use="Retrieve community reports by hierarchy level. Used in global/GGraphRAG queries.",
            implementation=community_from_level,
        ),

        # === Meta operators ===
        OperatorDescriptor(
            operator_id="meta.extract_entities",
            display_name="LLM Entity Extraction",
            category="meta",
            input_slots=[SlotSpec("query", SlotKind.QUERY_TEXT)],
            output_slots=[SlotSpec("entities", SlotKind.ENTITY_SET)],
            cost_tier=CostTier.MODERATE,
            requires_llm=True,
            when_to_use="Extract entity mentions from query using LLM. Use when NER is needed before entity.link or entity.ppr.",
            implementation=meta_extract_entities,
        ),
        OperatorDescriptor(
            operator_id="meta.reason_step",
            display_name="LLM Reasoning Step",
            category="meta",
            input_slots=[
                SlotSpec("query", SlotKind.QUERY_TEXT),
                SlotSpec("chunks", SlotKind.CHUNK_SET, required=False),
            ],
            output_slots=[SlotSpec("query", SlotKind.QUERY_TEXT)],
            cost_tier=CostTier.MODERATE,
            requires_llm=True,
            when_to_use="Refine query or decompose into sub-questions using LLM. Used in iterative/ToG methods.",
            implementation=meta_reason_step,
        ),
        OperatorDescriptor(
            operator_id="meta.rerank",
            display_name="LLM Reranking",
            category="meta",
            input_slots=[
                SlotSpec("query", SlotKind.QUERY_TEXT),
                SlotSpec("items", SlotKind.ENTITY_SET, description="Can also be CHUNK_SET"),
            ],
            output_slots=[SlotSpec("items", SlotKind.ENTITY_SET, description="Same kind as input, re-scored")],
            cost_tier=CostTier.MODERATE,
            requires_llm=True,
            when_to_use="Re-score entities or chunks by LLM relevance. Use to improve precision before answer generation.",
            implementation=meta_rerank,
        ),
        OperatorDescriptor(
            operator_id="meta.generate_answer",
            display_name="LLM Answer Generation",
            category="meta",
            input_slots=[
                SlotSpec("query", SlotKind.QUERY_TEXT),
                SlotSpec("chunks", SlotKind.CHUNK_SET, required=False),
            ],
            output_slots=[SlotSpec("answer", SlotKind.QUERY_TEXT)],
            cost_tier=CostTier.MODERATE,
            requires_llm=True,
            when_to_use="Generate final answer from query and retrieved context. Terminal operator in most pipelines.",
            implementation=meta_generate_answer,
        ),
        OperatorDescriptor(
            operator_id="meta.pcst_optimize",
            display_name="PCST Subgraph Optimization",
            category="meta",
            input_slots=[
                SlotSpec("entities", SlotKind.ENTITY_SET),
                SlotSpec("relationships", SlotKind.RELATIONSHIP_SET),
            ],
            output_slots=[SlotSpec("subgraph", SlotKind.SUBGRAPH)],
            cost_tier=CostTier.CHEAP,
            when_to_use="Optimize entity+relationship sets into a compact informative subgraph. Used in GR method.",
            implementation=meta_pcst_optimize,
        ),
    ]

    for d in descriptors:
        REGISTRY.register(d)


# Auto-register on import
_register_all()
