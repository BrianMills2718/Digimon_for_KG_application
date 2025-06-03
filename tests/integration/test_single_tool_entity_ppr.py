# testing/test_single_tool_entity_ppr.py
import sys
from pathlib import Path

# Add project root to sys.path to allow imports from Core, Option, etc.
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import asyncio
import os
from typing import List, Optional, Any, Dict

# Core components for context and tool
from Core.AgentSchema.context import GraphRAGContext
from Core.AgentSchema.tool_contracts import EntityPPRInputs, EntityPPROutputs
from Core.AgentTools.entity_tools import entity_ppr_tool # Assuming entity_ppr_tool is in entity_tools.py

# For loading main config and embedding provider
from Option.Config2 import Config as MainConfig # Renamed to avoid conflict if Config is used locally
from Core.Index.EmbeddingFactory import get_rag_embedding

# For loading graph
from Core.Graph.GraphFactory import GraphFactory
from Core.Storage.NetworkXStorage import NetworkXStorage # For type hinting or direct instantiation if needed
from Core.Storage.NameSpace import NameSpace, Workspace

# For loading VDB (needed for entity linking in PPR)
# from Core.Index.IndexFactory import IndexFactory
from Core.Index.FaissIndex import FaissIndex # For type hinting or direct instantiation

# For RetrieverConfig (needed by EntityRetriever within the tool)
from Config.RetrieverConfig import RetrieverConfig

from Core.Common.Logger import logger # For any local logging in the test script

async def main_ppr_test():
    logger.info("--- Starting Test for entity_ppr_tool ---")

    # --- 1. Setup Test Parameters ---
    target_dataset = "MySampleTexts"
    # This refers to the graph artifact name used in storage paths (e.g., "kg_graph", "er_graph")
    graph_artifact_name = "kg_graph" 
    entities_vdb_ref_id = "entities_vdb" # VDB name within the graph_artifact_name directory
    results_root_dir = "./results"

    # --- 2. Load Main Configuration & Embedding Provider ---
    main_config: Optional[MainConfig] = None
    llama_embed_provider: Optional[Any] = None
    try:
        main_config = MainConfig.from_yaml_file("Option/Config2.yaml")
        logger.info(f"Loaded main config. Embedding type: {getattr(main_config.embedding, 'api_type', 'N/A')}")
        llama_embed_provider = get_rag_embedding(config=main_config)
        logger.info(f"Successfully created LlamaIndex embedding provider: {type(llama_embed_provider)}")
    except Exception as e:
        logger.error(f"ERROR: Could not load main configuration or embedding provider. Error: {e}", exc_info=True)
        return

    if main_config is None or llama_embed_provider is None:
        logger.error("ERROR: Main config or embedding provider was not loaded. Exiting test.")
        return

    # --- 3. Load Graph Instance ---
    async def get_loaded_graph_instance(results_root_dir, dataset_name, graph_artifact_name_for_factory, main_cfg_for_graph, embed_provider_for_graph):
        """
        Helper to load a graph instance using NetworkXStorage (simulating minimal BaseGraph subclass).
        """
        from Core.Graph.BaseGraph import BaseGraph # local import
        # Set up Workspace and NameSpace for NetworkXStorage
        workspace_obj = Workspace(
            working_dir=results_root_dir,    # e.g., "./results"
            exp_name=dataset_name            # e.g., "MySampleTexts"
        )
        namespace_str_for_storage = graph_artifact_name_for_factory
        namespace_obj = NameSpace(
            workspace=workspace_obj,
            namespace=namespace_str_for_storage
        )
        _storage = NetworkXStorage()
        _storage.namespace = namespace_obj
        _storage.name = "graph_storage_nx_data.graphml"
        loaded = await _storage.load_graph(force=False)
        if not loaded:
            logger.error(f"Test Script: Could not load graph from: {_storage.graphml_xml_file}")
            return None
        class DummyMainGraphConfig:
            pass
        class TestGraph(BaseGraph):
            def __init__(self, cfg, llm, enc, storage):
                super().__init__(cfg, llm, enc)
                self._graph = storage
            async def _extract_entity_relationship(self, chunk_key_pair): pass
            async def _build_graph(self, chunks): pass
        graph_instance_for_test = TestGraph(cfg=DummyMainGraphConfig(), llm=main_cfg_for_graph.llm, enc=embed_provider_for_graph, storage=_storage)
        logger.info(f"Test Script: Graph instance created, node_num: {getattr(graph_instance_for_test, 'node_num', 'unknown')}")
        return graph_instance_for_test

    graph_instance = await get_loaded_graph_instance(results_root_dir, target_dataset, graph_artifact_name, main_config, llama_embed_provider)
    if not graph_instance:
        logger.error("Test Script: Failed to get a loaded graph instance. Exiting.")
        return

    # --- 4. Load Entities VDB Instance ---
    entities_vdb_path = os.path.join(results_root_dir, target_dataset, graph_artifact_name, entities_vdb_ref_id)
    class DummyVDBConfig: # For FaissIndex constructor
        def __init__(self, path, embed_model):
            self.persist_path = path
            self.embed_model = embed_model # FaissIndex needs this directly in its config
            self.retrieve_top_k = 10 
    vdb_config_for_faiss = DummyVDBConfig(entities_vdb_path, llama_embed_provider)
    entities_vdb_instance = FaissIndex(config=vdb_config_for_faiss)
    if not await entities_vdb_instance.load(): # Call the load method of BaseIndex
        logger.error(f"Test Script: Failed to load Entities VDB from {entities_vdb_path}. Exiting.")
        return 
    logger.info(f"Test Script: Successfully loaded entities VDB from {entities_vdb_path}.")

    # --- 5. Prepare RetrieverConfig for the Tool ---
    retriever_config_dict_for_tool = {
        "top_k": 10,  # General top_k for retriever intermediate steps
        "use_entity_similarity_for_ppr": True, # Changed to True
        "node_specificity": True, # This might not be used now, but harmless to keep
        "top_k_entity_for_ppr": 5 # Add this, as it's used when use_entity_similarity_for_ppr is True
    }
    logger.info(f"Test Script: Using retriever_config_dict_for_tool: {retriever_config_dict_for_tool}")

    # --- 6. Instantiate GraphRAGContext ---
    test_context = GraphRAGContext(
        target_dataset_name=target_dataset,
        resolved_configs={
            "storage_root_dir": results_root_dir,
            "main_config_dict": main_config.model_dump(),
            "retriever_config_dict": retriever_config_dict_for_tool # Provide retriever config
        },
        embedding_provider=llama_embed_provider,
        graph_instance=graph_instance,
        entities_vdb_instance=entities_vdb_instance
    )
    logger.info("Test Script: GraphRAGContext instantiated.")

    # --- 7. Prepare Tool Inputs ---
    seed_entities = ["american revolution", "thirteen colonies", "george washington"]
    logger.info(f"Test Script: Using seed_entities: {seed_entities}")

    tool_inputs = EntityPPRInputs(
        graph_reference_id=graph_artifact_name,
        seed_entity_ids=seed_entities,
        top_k_results=5,
    )

    # --- 8. Call the Tool ---
    logger.info(f"\nCalling entity_ppr_tool with inputs: {tool_inputs}")
    try:
        result_outputs: EntityPPROutputs = await entity_ppr_tool(
            params=tool_inputs,
            graphrag_context=test_context
        )
        logger.info("\n--- Tool Execution Finished ---")
        logger.info("Tool Outputs (Ranked Entities - ID, PPR Score):")
        if result_outputs and result_outputs.ranked_entities:
            for entity_id, score in result_outputs.ranked_entities:
                logger.info(f"  Entity ID: {entity_id}, Score: {score:.6f}")
        else:
            logger.info("  No ranked entities returned or an error occurred.")

    except Exception as e:
        logger.error(f"\n--- Tool Execution Error ---", exc_info=True)

    logger.info("\n--- Test Script Finished ---")

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers[0].setFormatter(formatter)
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    asyncio.run(main_ppr_test())
