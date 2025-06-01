"""
Agent tool functions for building ERGraph, RKGraph, TreeGraph, TreeGraphBalanced, and PassageGraph.
Each function takes its Pydantic input model and core dependencies, applies config overrides, builds the graph, and returns a Pydantic output.
"""
import asyncio
from typing import Any, Optional
from Core.AgentSchema.graph_construction_tool_contracts import (
    BuildERGraphInputs, BuildERGraphOutputs, ERGraphConfigOverrides,
    BuildRKGraphInputs, BuildRKGraphOutputs, RKGraphConfigOverrides,
    BuildTreeGraphInputs, BuildTreeGraphOutputs, TreeGraphConfigOverrides,
    BuildTreeGraphBalancedInputs, BuildTreeGraphBalancedOutputs, TreeGraphBalancedConfigOverrides,
    BuildPassageGraphInputs, BuildPassageGraphOutputs, PassageGraphConfigOverrides,
)
from Core.Graph.GraphFactory import get_graph
from Option.Config2 import Config
from Core.Common.Logger import logger
# Helper: Apply config overrides to a config object
# This mutates the config_copy in-place

def apply_overrides(config_copy, overrides: Any):
    if not overrides:
        return
    for field, value in overrides.dict(exclude_unset=True).items():
        setattr(config_copy, field, value)

# Helper: Get artifact path from graph instance

def get_artifact_path(graph_instance):
    if hasattr(graph_instance._graph, 'file_path'):
        return str(graph_instance._graph.file_path)
    if hasattr(graph_instance._graph, 'tree_pkl_file'):
        return str(graph_instance._graph.tree_pkl_file)
    return None

# Helper: Get node, edge, and layer counts
async def get_graph_counts(graph_instance) -> dict:
    node_count = None
    edge_count = None
    layer_count = None
    # Try property, then method, then storage
    if hasattr(graph_instance, 'node_num'):
        node_count = graph_instance.node_num
        if asyncio.iscoroutinefunction(graph_instance.node_num):
            node_count = await graph_instance.node_num()
        elif callable(graph_instance.node_num):
            node_count = graph_instance.node_num()
    elif hasattr(graph_instance._graph, 'get_node_num'):
        node_count = graph_instance._graph.get_node_num()
    if hasattr(graph_instance, 'edge_num'):
        edge_count = graph_instance.edge_num
        if asyncio.iscoroutinefunction(graph_instance.edge_num):
            edge_count = await graph_instance.edge_num()
        elif callable(graph_instance.edge_num):
            edge_count = graph_instance.edge_num()
    elif hasattr(graph_instance._graph, 'get_edge_num'):
        edge_count = graph_instance._graph.get_edge_num()
    if hasattr(graph_instance, 'num_layers'):
        layer_count = graph_instance.num_layers
        if asyncio.iscoroutinefunction(graph_instance.num_layers):
            layer_count = await graph_instance.num_layers()
        elif callable(graph_instance.num_layers):
            layer_count = graph_instance.num_layers()
    elif hasattr(graph_instance._graph, 'get_layer_num'):
        layer_count = graph_instance._graph.get_layer_num()
    return dict(node_count=node_count, edge_count=edge_count, layer_count=layer_count)

# =========================
# ERGraph
# =========================
async def build_er_graph(
    tool_input: BuildERGraphInputs,
    main_config: Config,
    llm_instance: Any,
    encoder_instance: Any,
    chunk_factory: Any
) -> BuildERGraphOutputs:
    try:
        current_graph_config = main_config.graph.model_copy(deep=True)
        apply_overrides(current_graph_config, tool_input.config_overrides)
        temp_full_config = main_config.model_copy(deep=True)
        temp_full_config.graph = current_graph_config
        temp_full_config.graph.type = "er_graph"
        er_graph_instance = get_graph(
            config=temp_full_config,
            llm=llm_instance,
            encoder=encoder_instance,
        )
        if hasattr(er_graph_instance._graph, 'namespace'):
            er_graph_instance._graph.namespace = chunk_factory.get_namespace(tool_input.target_dataset_name)
        input_chunks = await chunk_factory.get_chunks_for_dataset(tool_input.target_dataset_name)
        if not input_chunks:
            return BuildERGraphOutputs(
                graph_id="", status="failure",
                message=f"No input chunks found for dataset: {tool_input.target_dataset_name}",
                artifact_path=None
            )
        
        logger.info(f"Calling er_graph_instance.build_graph for dataset: {tool_input.target_dataset_name}")
        success = await er_graph_instance.build_graph(chunks=input_chunks, force=tool_input.force_rebuild)
        
        if not success:
            logger.error(f"build_er_graph tool: er_graph_instance.build_graph failed for {tool_input.target_dataset_name}")
            return BuildERGraphOutputs(
                graph_id=f"{tool_input.target_dataset_name}_ERGraph",
                status="failure",
                message=f"ERGraph building failed internally for {tool_input.target_dataset_name}.",
                node_count=None,
                edge_count=None,
                artifact_path=None
            )
        
        logger.info(f"build_er_graph tool: er_graph_instance.build_graph succeeded for {tool_input.target_dataset_name}")
        counts = await get_graph_counts(er_graph_instance)
        artifact_p = get_artifact_path(er_graph_instance)
        return BuildERGraphOutputs(
            graph_id=f"{tool_input.target_dataset_name}_ERGraph",
            status="success",
            message=f"ERGraph built successfully for {tool_input.target_dataset_name}.",
            node_count=counts['node_count'],
            edge_count=counts['edge_count'],
            layer_count=counts['layer_count'],
            artifact_path=artifact_p
        )
    except Exception as e:
        return BuildERGraphOutputs(
            graph_id=f"{tool_input.target_dataset_name}_ERGraph",
            status="failure",
            message=str(e),
            artifact_path=None
        )

# =========================
# RKGraph
# =========================
async def build_rk_graph(
    tool_input: BuildRKGraphInputs,
    main_config: Config,
    llm_instance: Any,
    encoder_instance: Any,
    chunk_factory: Any
) -> BuildRKGraphOutputs:
    try:
        current_graph_config = main_config.graph.model_copy(deep=True)
        apply_overrides(current_graph_config, tool_input.config_overrides)
        temp_full_config = main_config.model_copy(deep=True)
        temp_full_config.graph = current_graph_config
        temp_full_config.graph.type = "rk_graph"
        rk_graph_instance = get_graph(
            config=temp_full_config,
            llm=llm_instance,
            encoder=encoder_instance,
        )
        if hasattr(rk_graph_instance._graph, 'namespace'):
            rk_graph_instance._graph.namespace = chunk_factory.get_namespace(tool_input.target_dataset_name)
        input_chunks = await chunk_factory.get_chunks_for_dataset(tool_input.target_dataset_name)
        if not input_chunks:
            return BuildRKGraphOutputs(
                graph_id="", status="failure",
                message=f"No input chunks found for dataset: {tool_input.target_dataset_name}",
                artifact_path=None
            )
        await rk_graph_instance.build_graph(chunks=input_chunks, force=tool_input.force_rebuild)
        counts = await get_graph_counts(rk_graph_instance)
        artifact_p = get_artifact_path(rk_graph_instance)
        return BuildRKGraphOutputs(
            graph_id=f"{tool_input.target_dataset_name}_RKGraph",
            status="success",
            message=f"RKGraph built successfully for {tool_input.target_dataset_name}.",
            node_count=counts['node_count'],
            edge_count=counts['edge_count'],
            layer_count=counts['layer_count'],
            artifact_path=artifact_p
        )
    except Exception as e:
        return BuildRKGraphOutputs(
            graph_id=f"{tool_input.target_dataset_name}_RKGraph",
            status="failure",
            message=str(e),
            artifact_path=None
        )

# =========================
# TreeGraph
# =========================
async def build_tree_graph(
    tool_input: BuildTreeGraphInputs,
    main_config: Config,
    llm_instance: Any,
    encoder_instance: Any,
    chunk_factory: Any
) -> BuildTreeGraphOutputs:
    try:
        current_graph_config = main_config.graph.model_copy(deep=True)
        apply_overrides(current_graph_config, tool_input.config_overrides)
        temp_full_config = main_config.model_copy(deep=True)
        temp_full_config.graph = current_graph_config
        temp_full_config.graph.type = "tree_graph"
        tree_graph_instance = get_graph(
            config=temp_full_config,
            llm=llm_instance,
            encoder=encoder_instance,
        )
        if hasattr(tree_graph_instance._graph, 'namespace'):
            tree_graph_instance._graph.namespace = chunk_factory.get_namespace(tool_input.target_dataset_name)
        input_chunks = await chunk_factory.get_chunks_for_dataset(tool_input.target_dataset_name)
        if not input_chunks:
            return BuildTreeGraphOutputs(
                graph_id="", status="failure",
                message=f"No input chunks found for dataset: {tool_input.target_dataset_name}",
                artifact_path=None
            )
        await tree_graph_instance.build_graph(chunks=input_chunks, force=tool_input.force_rebuild)
        counts = await get_graph_counts(tree_graph_instance)
        artifact_p = get_artifact_path(tree_graph_instance)
        return BuildTreeGraphOutputs(
            graph_id=f"{tool_input.target_dataset_name}_TreeGraph",
            status="success",
            message=f"TreeGraph built successfully for {tool_input.target_dataset_name}.",
            node_count=counts['node_count'],
            edge_count=counts['edge_count'],
            layer_count=counts['layer_count'],
            artifact_path=artifact_p
        )
    except Exception as e:
        return BuildTreeGraphOutputs(
            graph_id=f"{tool_input.target_dataset_name}_TreeGraph",
            status="failure",
            message=str(e),
            artifact_path=None
        )

# =========================
# TreeGraphBalanced
# =========================
async def build_tree_graph_balanced(
    tool_input: BuildTreeGraphBalancedInputs,
    main_config: Config,
    llm_instance: Any,
    encoder_instance: Any,
    chunk_factory: Any
) -> BuildTreeGraphBalancedOutputs:
    try:
        current_graph_config = main_config.graph.model_copy(deep=True)
        apply_overrides(current_graph_config, tool_input.config_overrides)
        temp_full_config = main_config.model_copy(deep=True)
        temp_full_config.graph = current_graph_config
        temp_full_config.graph.type = "tree_graph_balanced"
        tree_graph_balanced_instance = get_graph(
            config=temp_full_config,
            llm=llm_instance,
            encoder=encoder_instance,
        )
        if hasattr(tree_graph_balanced_instance._graph, 'namespace'):
            tree_graph_balanced_instance._graph.namespace = chunk_factory.get_namespace(tool_input.target_dataset_name)
        input_chunks = await chunk_factory.get_chunks_for_dataset(tool_input.target_dataset_name)
        if not input_chunks:
            return BuildTreeGraphBalancedOutputs(
                graph_id="", status="failure",
                message=f"No input chunks found for dataset: {tool_input.target_dataset_name}",
                artifact_path=None
            )
        await tree_graph_balanced_instance.build_graph(chunks=input_chunks, force=tool_input.force_rebuild)
        counts = await get_graph_counts(tree_graph_balanced_instance)
        artifact_p = get_artifact_path(tree_graph_balanced_instance)
        return BuildTreeGraphBalancedOutputs(
            graph_id=f"{tool_input.target_dataset_name}_TreeGraphBalanced",
            status="success",
            message=f"TreeGraphBalanced built successfully for {tool_input.target_dataset_name}.",
            node_count=counts['node_count'],
            edge_count=counts['edge_count'],
            layer_count=counts['layer_count'],
            artifact_path=artifact_p
        )
    except Exception as e:
        return BuildTreeGraphBalancedOutputs(
            graph_id=f"{tool_input.target_dataset_name}_TreeGraphBalanced",
            status="failure",
            message=str(e),
            artifact_path=None
        )

# =========================
# PassageGraph
# =========================
async def build_passage_graph(
    tool_input: BuildPassageGraphInputs,
    main_config: Config,
    llm_instance: Any,
    encoder_instance: Any,
    chunk_factory: Any
) -> BuildPassageGraphOutputs:
    try:
        current_graph_config = main_config.graph.model_copy(deep=True)
        apply_overrides(current_graph_config, tool_input.config_overrides)
        temp_full_config = main_config.model_copy(deep=True)
        temp_full_config.graph = current_graph_config
        temp_full_config.graph.type = "passage_graph"
        passage_graph_instance = get_graph(
            config=temp_full_config,
            llm=llm_instance,
            encoder=encoder_instance,
        )
        if hasattr(passage_graph_instance._graph, 'namespace'):
            passage_graph_instance._graph.namespace = chunk_factory.get_namespace(tool_input.target_dataset_name)
        input_chunks = await chunk_factory.get_chunks_for_dataset(tool_input.target_dataset_name)
        if not input_chunks:
            return BuildPassageGraphOutputs(
                graph_id="", status="failure",
                message=f"No input chunks found for dataset: {tool_input.target_dataset_name}",
                artifact_path=None
            )
        await passage_graph_instance.build_graph(chunks=input_chunks, force=tool_input.force_rebuild)
        counts = await get_graph_counts(passage_graph_instance)
        artifact_p = get_artifact_path(passage_graph_instance)
        return BuildPassageGraphOutputs(
            graph_id=f"{tool_input.target_dataset_name}_PassageGraph",
            status="success",
            message=f"PassageGraph built successfully for {tool_input.target_dataset_name}.",
            node_count=counts['node_count'],
            edge_count=counts['edge_count'],
            layer_count=counts['layer_count'],
            artifact_path=artifact_p
        )
    except Exception as e:
        return BuildPassageGraphOutputs(
            graph_id=f"{tool_input.target_dataset_name}_PassageGraph",
            status="failure",
            message=str(e),
            artifact_path=None
        )
