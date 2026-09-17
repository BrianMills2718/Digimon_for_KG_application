"""Chunk-oriented agent tools.

Returned evidence must resolve to an exact stored source chunk (or an exact
chunk node in the graph). These helpers never fabricate placeholder text and
never guess a missing chunk by fuzzy content matching.
"""

from __future__ import annotations

from typing import Any, Dict, Union

import networkx as nx

from Core.AgentSchema.context import GraphRAGContext
from Core.AgentSchema.tool_contracts import (
    ChunkData,
    ChunkFromRelationshipsInputs,
    ChunkGetTextForEntitiesInput,
    ChunkOccurrenceInputs,
    ChunkOccurrenceOutputs,
    ChunkRelationshipScoreAggregatorInputs,
    ChunkRelationshipScoreAggregatorOutputs,
)
from Core.Common.Constants import GRAPH_FIELD_SEP
from Core.Common.Logger import logger
from Core.Common.Utils import split_string_by_multi_markers


_GRAPH_SUFFIXES = (
    "_TreeGraphBalanced",
    "_PassageGraph",
    "_ERGraph",
    "_RKGraph",
    "_TreeGraph",
)


def _dataset_from_graph_reference(graph_reference_id: str) -> str:
    for suffix in _GRAPH_SUFFIXES:
        if graph_reference_id.endswith(suffix):
            return graph_reference_id[: -len(suffix)]
    return graph_reference_id


def _extract_networkx_graph(graph_instance) -> nx.Graph | None:
    if graph_instance is None:
        return None
    if isinstance(graph_instance, nx.Graph):
        return graph_instance
    storage = getattr(graph_instance, "_graph", None)
    if isinstance(storage, nx.Graph):
        return storage
    storage_graph = getattr(storage, "graph", None)
    if isinstance(storage_graph, nx.Graph):
        return storage_graph
    direct_graph = getattr(graph_instance, "graph", None)
    if isinstance(direct_graph, nx.Graph):
        return direct_graph
    return None


def _split_source_ids(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, (list, tuple, set)):
        result = []
        for item in value:
            result.extend(_split_source_ids(item))
        return result
    return split_string_by_multi_markers(str(value), [GRAPH_FIELD_SEP])


async def _load_dataset_chunks(
    context: GraphRAGContext,
    graph_reference_id: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Return exact chunk lookup plus deterministic legacy aliases."""
    storage = context.chunk_storage_manager
    if storage is None:
        return {}, {}

    dataset_name = _dataset_from_graph_reference(graph_reference_id)
    try:
        chunks = await storage.get_chunks_for_dataset(dataset_name)
    except Exception as exc:
        logger.warning(f"Could not load chunks for dataset '{dataset_name}': {exc}")
        return {}, {}

    exact: dict[str, Any] = {}
    aliases: dict[str, str] = {}
    for chunk_id, chunk in chunks:
        actual_id = str(chunk_id)
        exact[actual_id] = chunk
        doc_id = getattr(chunk, "doc_id", None)
        if doc_id:
            aliases.setdefault(f"chunk_{doc_id}", actual_id)
    return exact, aliases


def _chunk_data(actual_id: str, chunk: Any, metadata: dict | None = None) -> ChunkData:
    """Construct ChunkData using its real TextChunk-compatible runtime init."""
    return ChunkData(
        tokens=int(getattr(chunk, "tokens", 0) or 0),
        chunk_id=actual_id,
        content=str(getattr(chunk, "content", "") or ""),
        doc_id=str(getattr(chunk, "doc_id", "") or ""),
        index=int(getattr(chunk, "index", 0) or 0),
        title=getattr(chunk, "title", None),
        metadata=metadata or {},
    )


def _relationship_matches(graph: nx.Graph, u: Any, v: Any, data: dict, spec: Any) -> bool:
    if isinstance(spec, dict):
        if spec.get("id") is not None:
            return str(data.get("id", "")) == str(spec["id"])
        if spec.get("relationship_id") is not None:
            candidate = data.get("relationship_id", data.get("id", ""))
            return str(candidate) == str(spec["relationship_id"])
        if spec.get("source") is not None and spec.get("target") is not None:
            source, target = str(spec["source"]), str(spec["target"])
            return (str(u), str(v)) == (source, target) or (
                not graph.is_directed() and (str(v), str(u)) == (source, target)
            )
        return False

    value = str(spec)
    identifiers = {
        str(data.get("id", "")),
        str(data.get("relationship_id", "")),
        str(data.get("rel_id", "")),
        str(data.get("relation_name", "")),
        f"{u}->{v}",
    }
    if not graph.is_directed():
        identifiers.add(f"{v}->{u}")
    return value in identifiers


def _edge_source_chunk_ids(edge_data: dict) -> list[str]:
    ids = _split_source_ids(edge_data.get("source_id"))
    raw_chunks = edge_data.get("chunks")
    if isinstance(raw_chunks, (list, tuple)):
        for raw in raw_chunks:
            if isinstance(raw, str):
                ids.append(raw)
            elif isinstance(raw, dict):
                chunk_id = raw.get("chunk_id") or raw.get("id")
                if chunk_id:
                    ids.append(str(chunk_id))
    return list(dict.fromkeys(str(chunk_id) for chunk_id in ids if chunk_id))


# ---------------------------------------------------------------------------
# Chunks from relationships
# ---------------------------------------------------------------------------

def chunk_from_relationships(
    input_data: Dict[str, Any],
    context: GraphRAGContext,
) -> Dict[str, Any]:
    """Legacy synchronous resolver for already-materialized graph chunk data.

    String chunk references are intentionally not converted to placeholder text;
    use the async wrapper for storage-backed source resolution.
    """
    try:
        params = ChunkFromRelationshipsInputs(**input_data)
    except Exception as exc:
        logger.error(f"Invalid Chunk.FromRelationships input: {exc}")
        return {"relevant_chunks": []}

    graph = _extract_networkx_graph(context.get_graph_instance(params.document_collection_id))
    if graph is None:
        return {"relevant_chunks": []}

    output = []
    seen = set()
    for spec in params.target_relationships:
        per_relationship = 0
        for u, v, edge_data in graph.edges(data=True):
            if not _relationship_matches(graph, u, v, edge_data, spec):
                continue
            raw_chunks = edge_data.get("chunks")
            if not isinstance(raw_chunks, list):
                continue
            for raw in raw_chunks:
                if not isinstance(raw, dict):
                    continue
                chunk_id = str(raw.get("chunk_id") or raw.get("id") or "")
                content = raw.get("content", raw.get("text", ""))
                if not chunk_id or not content or chunk_id in seen:
                    continue
                seen.add(chunk_id)
                output.append(
                    ChunkData(
                        tokens=int(raw.get("tokens", 0) or 0),
                        chunk_id=chunk_id,
                        content=str(content),
                        doc_id=str(raw.get("doc_id", "") or ""),
                        index=int(raw.get("index", 0) or 0),
                        title=raw.get("title"),
                        metadata={"relationship": str(spec)},
                    )
                )
                per_relationship += 1
                if (
                    params.max_chunks_per_relationship
                    and per_relationship >= params.max_chunks_per_relationship
                ):
                    break
        if params.top_k_total and len(output) >= params.top_k_total:
            break

    if params.top_k_total:
        output = output[: params.top_k_total]
    return {"relevant_chunks": output}


async def chunk_from_relationships_tool(
    input_data: Dict[str, Any],
    context: GraphRAGContext,
) -> Dict[str, Any]:
    """Resolve relationship source references to original stored chunks."""
    try:
        params = ChunkFromRelationshipsInputs(**input_data)
    except Exception as exc:
        logger.error(f"Invalid Chunk.FromRelationships input: {exc}")
        return {"relevant_chunks": []}

    graph = _extract_networkx_graph(context.get_graph_instance(params.document_collection_id))
    if graph is None:
        logger.error(f"Graph '{params.document_collection_id}' is unavailable")
        return {"relevant_chunks": []}

    exact_chunks, aliases = await _load_dataset_chunks(context, params.document_collection_id)
    if not exact_chunks:
        logger.warning("Chunk.FromRelationships: source chunk storage is unavailable or empty")
        return {"relevant_chunks": []}

    output = []
    seen_actual_ids = set()
    for spec in params.target_relationships:
        relationship_chunk_ids = []
        for u, v, edge_data in graph.edges(data=True):
            if _relationship_matches(graph, u, v, edge_data, spec):
                relationship_chunk_ids.extend(_edge_source_chunk_ids(edge_data))
        relationship_chunk_ids = list(dict.fromkeys(relationship_chunk_ids))

        if params.max_chunks_per_relationship:
            relationship_chunk_ids = relationship_chunk_ids[
                : params.max_chunks_per_relationship
            ]

        for requested_id in relationship_chunk_ids:
            actual_id = requested_id if requested_id in exact_chunks else aliases.get(requested_id)
            if not actual_id or actual_id in seen_actual_ids:
                if not actual_id:
                    logger.warning(
                        f"Chunk.FromRelationships: source chunk '{requested_id}' not found; skipping"
                    )
                continue
            seen_actual_ids.add(actual_id)
            metadata = {"relationship": str(spec)}
            if actual_id != requested_id:
                metadata["requested_reference_id"] = requested_id
            output.append(_chunk_data(actual_id, exact_chunks[actual_id], metadata))

            if params.top_k_total and len(output) >= params.top_k_total:
                return {"relevant_chunks": output}

    return {"relevant_chunks": output}


# ---------------------------------------------------------------------------
# Chunk occurrence
# ---------------------------------------------------------------------------

async def chunk_occurrence_tool(
    params: ChunkOccurrenceInputs,
    graphrag_context: GraphRAGContext,
) -> ChunkOccurrenceOutputs:
    """Rank exact stored chunks by co-occurrence of entity pairs."""
    graph = _extract_networkx_graph(
        graphrag_context.get_graph_instance(params.document_collection_id)
    )
    if graph is None:
        logger.error(f"Chunk.Occurrence: graph '{params.document_collection_id}' unavailable")
        return ChunkOccurrenceOutputs(ranked_occurrence_chunks=[])

    entity_chunks: Dict[str, set[str]] = {}
    for node_id, node_data in graph.nodes(data=True):
        entity_chunks[str(node_id)] = set(_split_source_ids(node_data.get("source_id")))

    chunk_scores: Dict[str, float] = {}
    for pair in params.target_entity_pairs_in_relationship:
        e1 = str(pair.get("entity1_id", ""))
        e2 = str(pair.get("entity2_id", ""))
        for chunk_id in entity_chunks.get(e1, set()) & entity_chunks.get(e2, set()):
            chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0.0) + 1.0

    exact_chunks, aliases = await _load_dataset_chunks(
        graphrag_context, params.document_collection_id
    )
    ranked = []
    seen = set()
    for requested_id, score in sorted(
        chunk_scores.items(), key=lambda item: item[1], reverse=True
    ):
        actual_id = requested_id if requested_id in exact_chunks else aliases.get(requested_id)
        if not actual_id or actual_id in seen:
            continue
        seen.add(actual_id)
        chunk = _chunk_data(
            actual_id,
            exact_chunks[actual_id],
            {"requested_reference_id": requested_id} if actual_id != requested_id else {},
        )
        chunk.relevance_score = float(score)
        ranked.append(chunk)
        if len(ranked) >= params.top_k_chunks:
            break

    return ChunkOccurrenceOutputs(ranked_occurrence_chunks=ranked)


# ---------------------------------------------------------------------------
# Candidate chunk score aggregation (legacy direct tool)
# ---------------------------------------------------------------------------

def _score_relationship_sources_in_graph(
    graph: nx.Graph,
    relationship_scores: dict[str, float],
) -> tuple[dict[str, float], int]:
    """Map scored relationship identifiers to exact graph source chunk IDs."""
    chunk_scores: dict[str, float] = {}
    matched_keys = set()

    for u, v, edge_data in graph.edges(data=True):
        for relationship_id, raw_score in relationship_scores.items():
            if not _relationship_matches(graph, u, v, edge_data, relationship_id):
                continue
            matched_keys.add(str(relationship_id))
            score = float(raw_score)
            for chunk_id in _edge_source_chunk_ids(edge_data):
                chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0.0) + score

    return chunk_scores, len(matched_keys)


def _candidate_graphs_for_direct_aggregation(
    context: GraphRAGContext,
) -> list[tuple[str, nx.Graph]]:
    """Return registered NetworkX-backed graphs without mutating active dataset."""
    candidates = []
    graphs = getattr(context, "graphs", {}) or {}
    for graph_id in context.list_graphs():
        graph = _extract_networkx_graph(graphs.get(graph_id))
        if graph is not None:
            candidates.append((graph_id, graph))
    return candidates


async def _derive_scored_chunks_from_relationships(
    params: ChunkRelationshipScoreAggregatorInputs,
    context: GraphRAGContext,
) -> list[ChunkData]:
    """Recover exact chunk candidates when the legacy MCP wrapper passes none."""
    if not params.relationship_scores:
        return []

    active_dataset = getattr(context, "active_dataset_name", None)
    ranked_graphs = []
    for graph_id, graph in _candidate_graphs_for_direct_aggregation(context):
        chunk_scores, matched_count = _score_relationship_sources_in_graph(
            graph,
            params.relationship_scores,
        )
        if matched_count <= 0 or not chunk_scores:
            continue
        dataset = _dataset_from_graph_reference(graph_id)
        ranked_graphs.append(
            (
                matched_count,
                dataset == active_dataset,
                sum(abs(score) for score in chunk_scores.values()),
                graph_id,
                chunk_scores,
            )
        )

    if not ranked_graphs:
        logger.warning(
            "Chunk.Aggregator: no registered graph matched the supplied relationship score IDs"
        )
        return []

    ranked_graphs.sort(
        key=lambda item: (item[0], item[1], item[2]),
        reverse=True,
    )
    _matches, _active, _magnitude, graph_id, chunk_scores = ranked_graphs[0]

    exact_chunks, aliases = await _load_dataset_chunks(context, graph_id)
    output = []
    seen = set()
    for requested_id, score in sorted(
        chunk_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        actual_id = requested_id if requested_id in exact_chunks else aliases.get(requested_id)
        if not actual_id or actual_id in seen:
            continue
        seen.add(actual_id)
        chunk = _chunk_data(
            actual_id,
            exact_chunks[actual_id],
            {
                "derived_from_relationship_scores": True,
                "graph_reference_id": graph_id,
                "requested_reference_id": requested_id,
            },
        )
        chunk.relevance_score = float(score)
        output.append(chunk)
        if len(output) >= params.top_k_chunks:
            break
    return output


async def chunk_aggregator_tool(
    params: ChunkRelationshipScoreAggregatorInputs,
    graphrag_context: GraphRAGContext,
) -> ChunkRelationshipScoreAggregatorOutputs:
    """Rank grounded chunks by relationship scores.

    The legacy stdio MCP wrapper currently supplies ``chunk_candidates=[]``.
    In that case recover candidates from registered graph edges and exact
    ``source_id`` chunks rather than returning an empty result. If candidates
    are explicitly provided, retain the existing candidate-ranking behavior.
    """
    if not params.chunk_candidates:
        derived = await _derive_scored_chunks_from_relationships(
            params,
            graphrag_context,
        )
        return ChunkRelationshipScoreAggregatorOutputs(
            ranked_aggregated_chunks=derived
        )

    total_relationship_score = float(sum(params.relationship_scores.values()))
    baseline = (
        total_relationship_score / len(params.chunk_candidates)
        if total_relationship_score > 0
        else 0.0
    )

    scored = []
    for chunk in params.chunk_candidates:
        metadata = getattr(chunk, "metadata", None) or {}
        matched_score = 0.0
        for relationship_id, score in params.relationship_scores.items():
            if metadata.get("relationship_id") == relationship_id:
                matched_score += float(score)
            elif relationship_id in str(metadata):
                matched_score += float(score) * 0.5
        scored.append((chunk, matched_score if matched_score > 0 else baseline))

    scored.sort(key=lambda item: item[1], reverse=True)
    output = []
    for original, score in scored[: params.top_k_chunks]:
        output.append(
            ChunkData(
                tokens=original.tokens,
                chunk_id=original.chunk_id,
                content=original.content,
                doc_id=original.doc_id,
                index=original.index,
                title=original.title,
                metadata=getattr(original, "metadata", None) or {},
                relevance_score=float(score),
            )
        )

    return ChunkRelationshipScoreAggregatorOutputs(ranked_aggregated_chunks=output)


# ---------------------------------------------------------------------------
# Exact source text for entities
# ---------------------------------------------------------------------------

async def chunk_get_text_for_entities_tool(
    params: Union[Dict[str, Any], ChunkGetTextForEntitiesInput],
    context: GraphRAGContext,
) -> Dict[str, Any]:
    """Retrieve exact original source chunks associated with graph entities."""
    try:
        validated = (
            ChunkGetTextForEntitiesInput(**params) if isinstance(params, dict) else params
        )
    except Exception as exc:
        return {"retrieved_chunks": [], "status_message": f"Invalid input: {exc}"}

    graph = _extract_networkx_graph(
        context.get_graph_instance(validated.graph_reference_id)
    )
    if graph is None:
        return {
            "retrieved_chunks": [],
            "status_message": f"Graph '{validated.graph_reference_id}' not found",
        }

    chunks_per_entity: dict[str, list[str]] = {}
    chunk_to_entities: dict[str, list[str]] = {}

    for entity_id in validated.entity_ids:
        if entity_id not in graph:
            logger.warning(f"Entity '{entity_id}' not found in graph")
            continue

        node_data = graph.nodes[entity_id]
        chunk_ids = []
        for field in (
            "chunk_id",
            "source_chunk_id",
            "source_id",
            "chunk_ids",
            "source_chunks",
        ):
            chunk_ids.extend(_split_source_ids(node_data.get(field)))

        for neighbor in graph.neighbors(entity_id):
            neighbor_data = graph.nodes[neighbor]
            if (
                neighbor_data.get("node_type") == "chunk"
                or neighbor_data.get("type") == "chunk"
            ):
                chunk_ids.append(str(neighbor))

        chunk_ids = list(dict.fromkeys(chunk_ids))
        if validated.max_chunks_per_entity is not None:
            chunk_ids = chunk_ids[: max(0, validated.max_chunks_per_entity)]
        chunks_per_entity[str(entity_id)] = chunk_ids
        for chunk_id in chunk_ids:
            chunk_to_entities.setdefault(chunk_id, []).append(str(entity_id))

    requested_ids = (
        [str(chunk_id) for chunk_id in validated.chunk_ids]
        if validated.chunk_ids
        else list(
            dict.fromkeys(
                chunk_id
                for entity_chunk_ids in chunks_per_entity.values()
                for chunk_id in entity_chunk_ids
            )
        )
    )

    exact_chunks, aliases = await _load_dataset_chunks(
        context, validated.graph_reference_id
    )
    retrieved = []
    seen_actual_ids = set()

    for requested_id in requested_ids:
        actual_id = requested_id if requested_id in exact_chunks else aliases.get(requested_id)

        if actual_id and actual_id not in seen_actual_ids:
            seen_actual_ids.add(actual_id)
            chunk = exact_chunks[actual_id]
            metadata = {
                "doc_id": getattr(chunk, "doc_id", None),
                "title": getattr(chunk, "title", None),
                "index": getattr(chunk, "index", None),
                "tokens": getattr(chunk, "tokens", None),
            }
            if actual_id != requested_id:
                metadata["requested_reference_id"] = requested_id
            entities = chunk_to_entities.get(requested_id, [])
            retrieved.append(
                {
                    "entity_id": entities[0] if len(entities) == 1 else None,
                    "chunk_id": actual_id,
                    "text_content": str(getattr(chunk, "content", "") or ""),
                    "metadata": metadata,
                }
            )
            continue

        if requested_id in graph:
            chunk_node = graph.nodes[requested_id]
            content = chunk_node.get("content", chunk_node.get("text", ""))
            if content and requested_id not in seen_actual_ids:
                seen_actual_ids.add(requested_id)
                entities = chunk_to_entities.get(requested_id, [])
                retrieved.append(
                    {
                        "entity_id": entities[0] if len(entities) == 1 else None,
                        "chunk_id": requested_id,
                        "text_content": str(content),
                        "metadata": {
                            key: value
                            for key, value in chunk_node.items()
                            if key not in {"content", "text"}
                            and not str(key).startswith("_")
                        },
                    }
                )
        else:
            logger.warning(
                f"Chunk.GetTextForEntities: exact chunk reference '{requested_id}' not found; skipping"
            )

    status = (
        f"Retrieved {len(retrieved)} exact chunks for "
        f"{len(validated.entity_ids)} requested entities"
    )
    logger.info(status)
    return {"retrieved_chunks": retrieved, "status_message": status}
