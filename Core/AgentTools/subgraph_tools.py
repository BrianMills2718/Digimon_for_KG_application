# Core/AgentTools/subgraph_tools.py

import itertools
import json
import uuid
from typing import List, Optional

import networkx as nx

from Core.AgentSchema.context import GraphRAGContext
from Core.AgentSchema.tool_contracts import (
    PathObject,
    PathSegment,
    SubgraphAgentPathInputs,
    SubgraphAgentPathOutputs,
    SubgraphKHopPathsInputs,
    SubgraphKHopPathsOutputs,
    SubgraphSteinerTreeInputs,
    SubgraphSteinerTreeOutputs,
)
from Core.Common.Logger import logger
from Core.Schema.SlotTypes import EntityRecord, SlotKind, SlotValue


def _get_nx_graph(graph_instance) -> Optional[nx.Graph]:
    """Extract the underlying NetworkX graph from a graph instance."""
    if (
        hasattr(graph_instance, "_graph")
        and hasattr(graph_instance._graph, "graph")
        and isinstance(graph_instance._graph.graph, nx.Graph)
    ):
        return graph_instance._graph.graph
    if hasattr(graph_instance, "_graph") and isinstance(graph_instance._graph, nx.Graph):
        return graph_instance._graph
    if isinstance(graph_instance, nx.Graph):
        return graph_instance
    return None


def _path_to_path_object(nx_graph: nx.Graph, node_path: List[str]) -> PathObject:
    """Convert a node path into alternating entity/relationship segments."""
    segments = []
    for index, node_id in enumerate(node_path):
        segments.append(
            PathSegment(
                item_id=node_id,
                item_type="entity",
                label=node_id,
            )
        )
        if index < len(node_path) - 1:
            next_node = node_path[index + 1]
            edge_data = nx_graph.get_edge_data(node_id, next_node) or {}
            rel_name = edge_data.get(
                "relation_name",
                edge_data.get("type", "related_to"),
            )
            segments.append(
                PathSegment(
                    item_id=f"{node_id}->{next_node}",
                    item_type="relationship",
                    label=str(rel_name),
                )
            )

    return PathObject(
        path_id=f"path_{uuid.uuid4().hex[:8]}",
        segments=segments,
        start_node_id=node_path[0],
        end_node_id=node_path[-1] if len(node_path) > 1 else None,
        hop_count=len(node_path) - 1,
    )


async def subgraph_khop_paths_tool(
    params: SubgraphKHopPathsInputs,
    graphrag_context: GraphRAGContext,
) -> SubgraphKHopPathsOutputs:
    """Find simple graph paths up to ``k_hops`` from the requested entities."""
    logger.info(
        f"Executing Subgraph.KHopPaths: starts={params.start_entity_ids}, "
        f"ends={params.end_entity_ids}, k={params.k_hops}, graph='{params.graph_reference_id}'"
    )

    graph_instance = graphrag_context.get_graph_instance(params.graph_reference_id)
    if graph_instance is None:
        logger.error(
            f"Subgraph.KHopPaths: Graph '{params.graph_reference_id}' not found"
        )
        return SubgraphKHopPathsOutputs(discovered_paths=[])

    nx_graph = _get_nx_graph(graph_instance)
    if nx_graph is None:
        logger.error("Subgraph.KHopPaths: Could not access NetworkX graph")
        return SubgraphKHopPathsOutputs(discovered_paths=[])

    max_paths = params.max_paths_to_return or 10
    discovered_paths: List[PathObject] = []

    if params.end_entity_ids:
        for start_id in params.start_entity_ids:
            if start_id not in nx_graph:
                logger.warning(
                    f"Subgraph.KHopPaths: Start entity '{start_id}' not in graph"
                )
                continue
            for end_id in params.end_entity_ids:
                if end_id not in nx_graph:
                    logger.warning(
                        f"Subgraph.KHopPaths: End entity '{end_id}' not in graph"
                    )
                    continue
                if start_id == end_id:
                    continue
                try:
                    paths_gen = nx.all_simple_paths(
                        nx_graph,
                        start_id,
                        end_id,
                        cutoff=params.k_hops,
                    )
                    for path in itertools.islice(
                        paths_gen,
                        max_paths - len(discovered_paths),
                    ):
                        discovered_paths.append(
                            _path_to_path_object(nx_graph, path)
                        )
                        if len(discovered_paths) >= max_paths:
                            break
                except nx.NetworkXError as exc:
                    logger.warning(
                        f"Subgraph.KHopPaths: NetworkX error for {start_id}->{end_id}: {exc}"
                    )
                if len(discovered_paths) >= max_paths:
                    break
            if len(discovered_paths) >= max_paths:
                break
    else:
        for start_id in params.start_entity_ids:
            if start_id not in nx_graph:
                continue
            ego = nx.ego_graph(nx_graph, start_id, radius=params.k_hops)
            for target in ego.nodes():
                if target == start_id:
                    continue
                try:
                    paths_gen = nx.all_simple_paths(
                        nx_graph,
                        start_id,
                        target,
                        cutoff=params.k_hops,
                    )
                    for path in itertools.islice(paths_gen, 2):
                        discovered_paths.append(
                            _path_to_path_object(nx_graph, path)
                        )
                        if len(discovered_paths) >= max_paths:
                            break
                except nx.NetworkXError:
                    pass
                if len(discovered_paths) >= max_paths:
                    break
            if len(discovered_paths) >= max_paths:
                break

    logger.info(
        f"Subgraph.KHopPaths: Found {len(discovered_paths)} paths"
    )
    return SubgraphKHopPathsOutputs(discovered_paths=discovered_paths)


async def subgraph_steiner_tree_tool(
    params: SubgraphSteinerTreeInputs,
    graphrag_context: GraphRAGContext,
) -> SubgraphSteinerTreeOutputs:
    """Compute the same Steiner selection used by the typed operator surface."""
    logger.info(
        f"Executing Subgraph.SteinerTree: terminals={params.terminal_node_ids}, "
        f"graph='{params.graph_reference_id}'"
    )

    graph_instance = graphrag_context.get_graph_instance(params.graph_reference_id)
    if graph_instance is None:
        logger.error(
            f"Subgraph.SteinerTree: Graph '{params.graph_reference_id}' not found"
        )
        return SubgraphSteinerTreeOutputs(steiner_tree_edges=[])

    from Core.Operators.subgraph.steiner_tree import subgraph_steiner_tree

    result = await subgraph_steiner_tree(
        inputs={
            "entities": SlotValue(
                kind=SlotKind.ENTITY_SET,
                data=[
                    EntityRecord(entity_name=str(node_id))
                    for node_id in params.terminal_node_ids
                ],
                producer="Subgraph.SteinerTree",
            )
        },
        ctx=type("DirectSubgraphContext", (), {"graph": graph_instance})(),
        params={"weight_attribute": params.edge_weight_attribute},
    )
    subgraph = result["subgraph"].data
    nx_graph = _get_nx_graph(graph_instance)

    edges = []
    for source, target in subgraph.edges:
        edge_data = (
            nx_graph.get_edge_data(source, target) if nx_graph is not None else {}
        ) or {}
        edge = {"source": source, "target": target}
        if params.edge_weight_attribute and params.edge_weight_attribute in edge_data:
            edge["weight"] = edge_data[params.edge_weight_attribute]
        elif "weight" in edge_data:
            edge["weight"] = edge_data["weight"]
        if "relation_name" in edge_data:
            edge["relation_name"] = edge_data["relation_name"]
        edges.append(edge)

    logger.info(
        f"Subgraph.SteinerTree: returning {len(edges)} edges "
        f"for {len(result['subgraph'].metadata.get('used_terminals', []))} connected terminals"
    )
    return SubgraphSteinerTreeOutputs(steiner_tree_edges=edges)


async def subgraph_agent_path_tool(
    params: SubgraphAgentPathInputs,
    graphrag_context: GraphRAGContext,
) -> SubgraphAgentPathOutputs:
    """Use an LLM to rank/filter candidate paths by question relevance."""
    logger.info(
        f"Executing Subgraph.AgentPath: question='{params.user_question[:80]}...', "
        f"{len(params.candidate_paths)} candidate paths"
    )

    if not params.candidate_paths:
        return SubgraphAgentPathOutputs(relevant_paths=[])

    max_to_return = params.max_paths_to_return or 5
    path_descriptions = []
    for index, path_obj in enumerate(params.candidate_paths):
        segments = " -> ".join(
            segment.label or segment.item_id for segment in path_obj.segments
        )
        path_descriptions.append(f"Path {index + 1}: {segments}")

    prompt = f"""Given the following question, rank the paths below by relevance.
Return a JSON array of path numbers (1-indexed) in order of relevance, most relevant first.
Only include paths that are genuinely relevant to answering the question.
Return at most {max_to_return} path numbers.

Question: {params.user_question}

Paths:
{chr(10).join(path_descriptions)}

Return ONLY a JSON array of integers, e.g. [3, 1, 5]. No other text."""

    llm = graphrag_context.llm_provider
    if llm is None:
        logger.warning(
            "Subgraph.AgentPath: No LLM provider, returning candidate paths truncated"
        )
        return SubgraphAgentPathOutputs(
            relevant_paths=params.candidate_paths[:max_to_return]
        )

    try:
        response = str(await llm.aask(prompt)).strip()
        if response.startswith("```"):
            response = response.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        ranked_indices = json.loads(response)
        if not isinstance(ranked_indices, list):
            raise ValueError(f"Expected list, got {type(ranked_indices)}")

        relevant_paths = []
        for index in ranked_indices[:max_to_return]:
            path_index = int(index) - 1
            if 0 <= path_index < len(params.candidate_paths):
                relevant_paths.append(params.candidate_paths[path_index])

        logger.info(
            f"Subgraph.AgentPath: LLM selected {len(relevant_paths)} relevant paths"
        )
        return SubgraphAgentPathOutputs(relevant_paths=relevant_paths)
    except Exception as exc:
        logger.error(
            f"Subgraph.AgentPath: LLM ranking failed: {exc}",
            exc_info=True,
        )
        return SubgraphAgentPathOutputs(
            relevant_paths=params.candidate_paths[:max_to_return]
        )
