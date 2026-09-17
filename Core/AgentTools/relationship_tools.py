from __future__ import annotations

import json as _json
import logging
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from Core.AgentSchema.context import GraphRAGContext
from Core.AgentSchema.tool_contracts import (
    RelationshipAgentInputs,
    RelationshipAgentOutputs,
    RelationshipData,
    RelationshipOneHopNeighborsInputs,
    RelationshipOneHopNeighborsOutputs,
    RelationshipScoreAggregatorInputs,
    RelationshipScoreAggregatorOutputs,
    RelationshipVDBBuildInputs,
    RelationshipVDBBuildOutputs,
    RelationshipVDBSearchInputs,
    RelationshipVDBSearchOutputs,
)
from Core.AgentTools.index_config_helper import create_faiss_index_config
from Core.Index.FaissIndex import FaissIndex

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared graph helpers
# ---------------------------------------------------------------------------

def _extract_networkx_graph(graph_instance) -> Optional[nx.Graph]:
    if graph_instance is None:
        return None
    if isinstance(graph_instance, nx.Graph):
        return graph_instance
    storage = getattr(graph_instance, "_graph", None)
    if isinstance(storage, nx.Graph):
        return storage
    if storage is not None and isinstance(getattr(storage, "graph", None), nx.Graph):
        return storage.graph
    if isinstance(getattr(graph_instance, "graph", None), nx.Graph):
        return graph_instance.graph
    return None


def _effective_relationship_embedding_fields(requested_fields: List[str]) -> List[str]:
    """Map the legacy relationship VDB default to DIGIMON's real edge schema.

    ``RelationshipVDBBuildInputs`` historically defaulted to
    ``["type", "description"]`` while maintained graph edges use
    ``relation_name``, ``keywords`` and ``description``. Treat that unchanged
    legacy default as the intended semantic edge-text profile. Explicit custom
    field lists are otherwise respected.
    """
    fields = list(requested_fields or [])
    if fields == ["type", "description"]:
        return ["relation_name", "keywords", "description"]
    return fields


def _relationship_embedding_text(
    source: str,
    target: str,
    edge_data: Dict[str, Any],
    fields: List[str],
) -> str:
    parts = []
    for field in fields:
        value = edge_data.get(field)
        if value is None and field == "relation_name":
            value = edge_data.get("type")
        if value is None and field == "type":
            value = edge_data.get("relation_name")
        if value not in (None, ""):
            parts.append(f"{field}: {value}")

    if not parts:
        relation_name = edge_data.get("relation_name") or edge_data.get("type") or "related_to"
        parts.append(f"{source} {relation_name} {target}")
    return " | ".join(parts)


async def _registered_vdb_is_usable(vdb) -> bool:
    """Return True only when a registered VDB has or can load a live index."""
    if getattr(vdb, "_index", None) is not None:
        return True
    load = getattr(vdb, "load", None)
    if load is None:
        return False
    try:
        return bool(await load())
    except Exception as exc:
        logger.warning(f"Registered relationship VDB reload failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# One-hop relationship traversal
# ---------------------------------------------------------------------------

async def relationship_one_hop_neighbors_tool(
    params: RelationshipOneHopNeighborsInputs,
    graphrag_context: GraphRAGContext,
) -> RelationshipOneHopNeighborsOutputs:
    logger.info(
        f"Executing Relationship.OneHopNeighbors for {params.entity_ids} "
        f"on graph '{params.graph_reference_id}'"
    )
    output: List[RelationshipData] = []

    graph_instance = graphrag_context.get_graph_instance(params.graph_reference_id)
    graph = _extract_networkx_graph(graph_instance)
    if graph is None:
        logger.error(
            f"Relationship.OneHopNeighbors: graph '{params.graph_reference_id}' unavailable"
        )
        return RelationshipOneHopNeighborsOutputs(one_hop_relationships=[])

    directed = graph.is_directed()
    seen = set()

    def add_edge(source, target, attributes):
        relation_name = str(
            attributes.get("relation_name", attributes.get("type", "unknown_relationship"))
        )
        if (
            params.relationship_types_to_include
            and relation_name not in params.relationship_types_to_include
        ):
            return
        key = (str(source), str(target), relation_name)
        if key in seen:
            return
        seen.add(key)
        output.append(
            RelationshipData(
                source_id=str(attributes.get("source_id", "graph_traversal_tool")),
                src_id=str(source),
                tgt_id=str(target),
                relation_name=relation_name,
                description=(
                    str(attributes.get("description"))
                    if attributes.get("description") is not None
                    else None
                ),
                weight=float(attributes.get("weight", 1.0) or 1.0),
                attributes={
                    key: value
                    for key, value in attributes.items()
                    if key not in {"relation_name", "description", "weight", "source_id"}
                }
                or None,
            )
        )

    for entity_id in params.entity_ids:
        if entity_id not in graph:
            logger.warning(f"Entity '{entity_id}' not found in graph")
            continue

        if not directed or params.direction in ("outgoing", "both"):
            neighbors = graph.successors(entity_id) if directed else graph.neighbors(entity_id)
            for neighbor in neighbors:
                edge_data = graph.get_edge_data(entity_id, neighbor) or {}
                items = (
                    edge_data.values()
                    if isinstance(graph, (nx.MultiGraph, nx.MultiDiGraph))
                    else [edge_data]
                )
                for attributes in items:
                    add_edge(entity_id, neighbor, attributes)

        if directed and params.direction in ("incoming", "both"):
            for predecessor in graph.predecessors(entity_id):
                edge_data = graph.get_edge_data(predecessor, entity_id) or {}
                items = (
                    edge_data.values()
                    if isinstance(graph, nx.MultiDiGraph)
                    else [edge_data]
                )
                for attributes in items:
                    add_edge(predecessor, entity_id, attributes)

    logger.info(f"Relationship.OneHopNeighbors: returning {len(output)} relationships")
    return RelationshipOneHopNeighborsOutputs(one_hop_relationships=output)


# ---------------------------------------------------------------------------
# Relationship VDB build/search
# ---------------------------------------------------------------------------

async def relationship_vdb_build_tool(
    params: RelationshipVDBBuildInputs,
    graphrag_context: GraphRAGContext,
) -> RelationshipVDBBuildOutputs:
    """Build/load a relationship FAISS index and register it in context."""
    graph_instance = graphrag_context.get_graph_instance(params.graph_reference_id)
    graph = _extract_networkx_graph(graph_instance)
    if graph is None:
        message = f"Graph '{params.graph_reference_id}' not found in context"
        logger.error(message)
        return RelationshipVDBBuildOutputs(
            vdb_reference_id="",
            num_relationships_indexed=0,
            status=f"Error: {message}",
        )

    vdb_id = params.vdb_collection_name
    existing = graphrag_context.get_vdb_instance(vdb_id)
    if existing is not None and not params.force_rebuild:
        if await _registered_vdb_is_usable(existing):
            return RelationshipVDBBuildOutputs(
                vdb_reference_id=vdb_id,
                num_relationships_indexed=graph.number_of_edges(),
                status="VDB already exists",
            )
        logger.warning(
            f"Registered relationship VDB '{vdb_id}' is unusable; rebuilding it"
        )

    embedding_provider = graphrag_context.embedding_provider
    if embedding_provider is None:
        return RelationshipVDBBuildOutputs(
            vdb_reference_id="",
            num_relationships_indexed=0,
            status="Error: No embedding provider available in context",
        )

    embedding_fields = _effective_relationship_embedding_fields(
        list(params.embedding_fields or [])
    )
    logger.info(
        f"Building relationship VDB '{vdb_id}' using embedding fields {embedding_fields}"
    )

    relationships_data = []
    metadata_keys = {"id", "source", "target", "src_id", "tgt_id"}
    if params.include_metadata:
        for _source, _target, data in graph.edges(data=True):
            metadata_keys.update(data.keys())
    metadata_keys.difference_update(embedding_fields)

    for source, target, edge_data in graph.edges(data=True):
        relationship_id = str(edge_data.get("id", f"{source}->{target}"))
        document = {
            "id": relationship_id,
            "content": _relationship_embedding_text(
                str(source),
                str(target),
                edge_data,
                embedding_fields,
            ),
            "source": str(source),
            "target": str(target),
            "src_id": str(source),
            "tgt_id": str(target),
        }
        if params.include_metadata:
            for key, value in edge_data.items():
                if key != "id" and key not in embedding_fields:
                    document[key] = value
        relationships_data.append(document)

    if not relationships_data:
        return RelationshipVDBBuildOutputs(
            vdb_reference_id="",
            num_relationships_indexed=0,
            status="No relationships found in graph",
        )

    config = create_faiss_index_config(
        persist_path=f"storage/vdb/{vdb_id}",
        embed_model=embedding_provider,
        name=vdb_id,
    )
    relationship_vdb = FaissIndex(config)
    build_ok = await relationship_vdb.build_index(
        elements=relationships_data,
        meta_data=sorted(metadata_keys),
        force=params.force_rebuild,
    )
    if not build_ok:
        message = f"Relationship VDB '{vdb_id}' failed to build or load a usable index"
        logger.error(message)
        return RelationshipVDBBuildOutputs(
            vdb_reference_id="",
            num_relationships_indexed=0,
            status=f"Error: {message}",
        )

    graphrag_context.add_vdb_instance(vdb_id, relationship_vdb)
    if vdb_id not in graphrag_context.list_vdbs():
        return RelationshipVDBBuildOutputs(
            vdb_reference_id="",
            num_relationships_indexed=0,
            status=f"Error: Relationship VDB '{vdb_id}' failed context registration",
        )

    return RelationshipVDBBuildOutputs(
        vdb_reference_id=vdb_id,
        num_relationships_indexed=len(relationships_data),
        status=f"Successfully built VDB with {len(relationships_data)} relationships",
    )


async def relationship_vdb_search_tool(
    params: RelationshipVDBSearchInputs,
    graphrag_context: GraphRAGContext,
) -> RelationshipVDBSearchOutputs:
    """Search a registered relationship VDB using its BaseIndex retrieval API."""
    if not params.query_text and not params.query_embedding:
        return RelationshipVDBSearchOutputs(
            similar_relationships=[],
            metadata={"error": "Either query_text or query_embedding must be provided"},
        )

    vdb = graphrag_context.get_vdb_instance(params.vdb_reference_id)
    if vdb is None:
        return RelationshipVDBSearchOutputs(
            similar_relationships=[],
            metadata={"error": f"VDB '{params.vdb_reference_id}' not found"},
        )

    if params.query_embedding is not None and not params.query_text:
        return RelationshipVDBSearchOutputs(
            similar_relationships=[],
            metadata={"error": "Direct relationship query_embedding search is not implemented"},
        )

    try:
        results = await vdb.retrieval(query=params.query_text, top_k=params.top_k)
        similar = []
        for result in results:
            node = getattr(result, "node", None)
            metadata = getattr(node, "metadata", {}) or {}
            relationship_id = str(
                metadata.get("id", getattr(node, "node_id", "unknown"))
            )
            description = str(
                getattr(node, "text", "") or metadata.get("content", "")
            )
            score = (
                float(result.score)
                if getattr(result, "score", None) is not None
                else 0.0
            )
            if params.score_threshold is not None and score < params.score_threshold:
                continue
            similar.append((relationship_id, description, score))

        similar.sort(key=lambda item: item[2], reverse=True)
        return RelationshipVDBSearchOutputs(
            similar_relationships=similar,
            metadata={
                "vdb_id": params.vdb_reference_id,
                "num_results": len(similar),
                "query_type": "text",
            },
        )
    except Exception as exc:
        logger.error(f"Error searching relationship VDB: {exc}", exc_info=True)
        return RelationshipVDBSearchOutputs(
            similar_relationships=[],
            metadata={"error": str(exc)},
        )


# ---------------------------------------------------------------------------
# Direct LLM relationship extraction
# ---------------------------------------------------------------------------

async def relationship_agent_tool(
    params: RelationshipAgentInputs,
    graphrag_context: GraphRAGContext,
) -> RelationshipAgentOutputs:
    logger.info(
        f"Executing Relationship.Agent: query='{params.query_text[:80]}', "
        f"{len(params.context_entities)} context entities"
    )

    text = params.text_context
    if isinstance(text, list):
        text = "\n\n".join(text)

    max_relationships = params.max_relationships_to_extract or 10
    entity_names = [
        getattr(entity, "entity_name", str(entity))
        for entity in params.context_entities
    ]
    types = (
        ", ".join(params.target_relationship_types)
        if params.target_relationship_types
        else "any type"
    )
    prompt = f"""Extract relationships between entities from the following text.
Known entities: {', '.join(entity_names[:20])}
Focus on relationship types: {types}
Return a JSON array of objects with fields: "src_id", "tgt_id", "relation_name", "description".
Extract at most {max_relationships} relationships.

Query: {params.query_text}

Text:
{str(text)[:4000]}

Return ONLY a JSON array. No other text."""

    llm = graphrag_context.llm_provider
    if llm is None:
        return RelationshipAgentOutputs(extracted_relationships=[])

    try:
        response = str(await llm.aask(prompt)).strip()
        if response.startswith("```"):
            response = response.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        raw = _json.loads(response)
        if not isinstance(raw, list):
            raw = [raw]

        extracted = [
            RelationshipData(
                src_id=item.get("src_id", "unknown"),
                tgt_id=item.get("tgt_id", "unknown"),
                source_id="relationship_agent_tool",
                relation_name=item.get("relation_name", "related_to"),
                description=item.get("description", ""),
            )
            for item in raw[:max_relationships]
            if isinstance(item, dict)
        ]
        return RelationshipAgentOutputs(extracted_relationships=extracted)
    except Exception as exc:
        logger.error(f"Relationship.Agent: LLM extraction failed: {exc}", exc_info=True)
        return RelationshipAgentOutputs(extracted_relationships=[])


# ---------------------------------------------------------------------------
# Direct relationship score aggregation
# ---------------------------------------------------------------------------

async def relationship_score_aggregator_tool(
    params: RelationshipScoreAggregatorInputs,
    graphrag_context: GraphRAGContext,
) -> RelationshipScoreAggregatorOutputs:
    graph = _extract_networkx_graph(
        graphrag_context.get_graph_instance(params.graph_reference_id)
    )
    if graph is None:
        return RelationshipScoreAggregatorOutputs(scored_relationships=[])

    scored: List[Tuple[RelationshipData, float]] = []
    method = params.aggregation_method or "sum"

    for source, target, edge_data in graph.edges(data=True):
        source_score = float(params.entity_scores.get(source, 0.0))
        target_score = float(params.entity_scores.get(target, 0.0))
        if source_score == 0.0 and target_score == 0.0:
            continue

        if method == "average":
            aggregate = (source_score + target_score) / 2.0
        elif method == "max":
            aggregate = max(source_score, target_score)
        else:
            aggregate = source_score + target_score

        relationship = RelationshipData(
            src_id=str(source),
            tgt_id=str(target),
            source_id=str(edge_data.get("source_id", "score_aggregator")),
            relation_name=str(
                edge_data.get("relation_name", edge_data.get("type", "unknown"))
            ),
            description=(
                str(edge_data.get("description"))
                if edge_data.get("description") is not None
                else None
            ),
            weight=float(edge_data.get("weight", 1.0) or 1.0),
        )
        scored.append((relationship, aggregate))

    scored.sort(key=lambda item: item[1], reverse=True)
    top_k = params.top_k_relationships if params.top_k_relationships is not None else 10
    return RelationshipScoreAggregatorOutputs(scored_relationships=scored[:top_k])
