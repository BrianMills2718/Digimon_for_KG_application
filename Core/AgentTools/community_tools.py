from __future__ import annotations

from typing import Dict

from Core.AgentSchema.context import GraphRAGContext
from Core.AgentSchema.tool_contracts import (
    CommunityData,
    CommunityDetectFromEntitiesInputs,
    CommunityDetectFromEntitiesOutputs,
    CommunityGetLayerInputs,
    CommunityGetLayerOutputs,
)
from Core.Common.Logger import logger


async def _community_schema(graph_instance):
    if hasattr(graph_instance, "community_schema"):
        return await graph_instance.community_schema()
    if hasattr(graph_instance, "_graph") and hasattr(
        graph_instance._graph, "get_community_schema"
    ):
        return await graph_instance._graph.get_community_schema()
    raise AttributeError("No community schema method available on graph")


def _community_data(cluster_key, leiden_info) -> CommunityData:
    return CommunityData(
        community_id=str(cluster_key),
        level=leiden_info.level,
        title=leiden_info.title or f"Cluster {cluster_key}",
        nodes=set(leiden_info.nodes),
        edges=set(tuple(edge) for edge in leiden_info.edges),
        chunk_ids=set(leiden_info.chunk_ids),
        occurrence=leiden_info.occurrence,
        sub_communities=list(leiden_info.sub_communities),
    )


def _numeric_level(value) -> int:
    try:
        return int(value) if value != "" else 0
    except (ValueError, TypeError):
        return 0


async def community_detect_from_entities_tool(
    params: CommunityDetectFromEntitiesInputs,
    graphrag_context: GraphRAGContext,
) -> CommunityDetectFromEntitiesOutputs:
    """Return communities whose node sets intersect the supplied entities."""
    logger.info(
        f"Executing Community.DetectFromEntities: seeds={params.seed_entity_ids}, "
        f"graph='{params.graph_reference_id}'"
    )

    graph_instance = graphrag_context.get_graph_instance(params.graph_reference_id)
    if graph_instance is None:
        logger.error(
            f"Community.DetectFromEntities: Graph '{params.graph_reference_id}' not found"
        )
        return CommunityDetectFromEntitiesOutputs(relevant_communities=[])

    try:
        community_schema = await _community_schema(graph_instance)
    except Exception as exc:
        logger.error(
            f"Community.DetectFromEntities: Error getting community schema: {exc}",
            exc_info=True,
        )
        return CommunityDetectFromEntitiesOutputs(relevant_communities=[])

    if not community_schema:
        logger.warning("Community.DetectFromEntities: Empty community schema")
        return CommunityDetectFromEntitiesOutputs(relevant_communities=[])

    seed_set = set(params.seed_entity_ids)
    matching = []
    for cluster_key, leiden_info in community_schema.items():
        overlap = set(leiden_info.nodes) & seed_set
        if overlap:
            matching.append((cluster_key, leiden_info, len(overlap)))

    matching.sort(
        key=lambda item: (item[2], item[1].occurrence),
        reverse=True,
    )
    max_communities = params.max_communities_to_return or 5
    relevant = [
        _community_data(cluster_key, info)
        for cluster_key, info, _overlap in matching[:max_communities]
    ]

    logger.info(
        f"Community.DetectFromEntities: Found {len(relevant)} communities"
    )
    return CommunityDetectFromEntitiesOutputs(relevant_communities=relevant)


async def community_get_layer_tool(
    params: CommunityGetLayerInputs,
    graphrag_context: GraphRAGContext,
) -> CommunityGetLayerOutputs:
    """Return communities at or below the requested hierarchy depth."""
    logger.info(
        f"Executing Community.GetLayer: hierarchy='{params.community_hierarchy_reference_id}', "
        f"max_depth={params.max_layer_depth}"
    )

    graph_instance = graphrag_context.get_graph_instance(
        params.community_hierarchy_reference_id
    )
    if graph_instance is None:
        logger.error(
            f"Community.GetLayer: Graph '{params.community_hierarchy_reference_id}' not found"
        )
        return CommunityGetLayerOutputs(communities_in_layers=[])

    try:
        community_schema = await _community_schema(graph_instance)
    except Exception as exc:
        logger.error(
            f"Community.GetLayer: Error getting community schema: {exc}",
            exc_info=True,
        )
        return CommunityGetLayerOutputs(communities_in_layers=[])

    if not community_schema:
        logger.warning("Community.GetLayer: Empty community schema")
        return CommunityGetLayerOutputs(communities_in_layers=[])

    communities = [
        _community_data(cluster_key, info)
        for cluster_key, info in community_schema.items()
        if _numeric_level(info.level) <= params.max_layer_depth
    ]
    communities.sort(
        key=lambda community: (
            _numeric_level(community.level),
            -float(community.occurrence or 0.0),
        )
    )

    logger.info(
        f"Community.GetLayer: Found {len(communities)} communities at depth <= "
        f"{params.max_layer_depth}"
    )
    return CommunityGetLayerOutputs(communities_in_layers=communities)
