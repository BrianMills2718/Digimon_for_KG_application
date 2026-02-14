"""Modular operator system for composable KG retrieval pipelines.

Each operator is a standalone async function with uniform signature:
    async def op(inputs: Dict[str, SlotValue], ctx: OperatorContext, params: Dict) -> Dict[str, SlotValue]
"""

from Core.Operators._context import OperatorContext

# Entity operators
from Core.Operators.entity.vdb import entity_vdb
from Core.Operators.entity.ppr import entity_ppr
from Core.Operators.entity.onehop import entity_onehop
from Core.Operators.entity.link import entity_link
from Core.Operators.entity.tfidf import entity_tfidf
from Core.Operators.entity.agent import entity_agent
from Core.Operators.entity.rel_node import entity_rel_node

# Relationship operators
from Core.Operators.relationship.onehop import relationship_onehop
from Core.Operators.relationship.vdb import relationship_vdb
from Core.Operators.relationship.score_aggregator import relationship_score_agg
from Core.Operators.relationship.agent import relationship_agent

# Chunk operators
from Core.Operators.chunk.from_relation import chunk_from_relation
from Core.Operators.chunk.occurrence import chunk_occurrence
from Core.Operators.chunk.aggregator import chunk_aggregator

# Subgraph operators
from Core.Operators.subgraph.khop_paths import subgraph_khop_paths
from Core.Operators.subgraph.steiner_tree import subgraph_steiner_tree
from Core.Operators.subgraph.agent_path import subgraph_agent_path

# Community operators
from Core.Operators.community.from_entity import community_from_entity
from Core.Operators.community.from_level import community_from_level

# Meta operators
from Core.Operators.meta.extract_entities import meta_extract_entities
from Core.Operators.meta.reason_step import meta_reason_step
from Core.Operators.meta.rerank import meta_rerank
from Core.Operators.meta.generate_answer import meta_generate_answer
from Core.Operators.meta.pcst_optimize import meta_pcst_optimize
