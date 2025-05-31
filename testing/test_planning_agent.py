# START: /home/brian/digimon/testing/test_planning_agent.py
import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace # For ad-hoc config objects if needed

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from Core.AgentBrain.agent_brain import PlanningAgent
from Core.AgentSchema.context import GraphRAGContext
from Core.Common.Logger import logger

# Use the actual Config loader from your project
from Option.Config2 import Config, default_config as main_config_loader #

# Imports needed for GraphRAGContext setup (adapt from your test_agent_orchestrator.py)
from Core.Index.EmbeddingFactory import get_rag_embedding
from Core.Index.FaissIndex import FaissIndex #
from Core.Storage.NetworkXStorage import NetworkXStorage #
from Core.Graph.GraphFactory import GraphFactory #
from Core.Storage.NameSpace import NameSpace, Workspace #

async def setup_graphrag_context_for_agent(config: Config, dataset_name="MySampleTexts", graph_name="kg_graph") -> GraphRAGContext | None:
    logger.info("Planning Agent Test: Setting up GraphRAGContext...")
    
    try:
        embedding_provider = get_rag_embedding(config=config) #
        logger.info(f"Planning Agent Test: Embedding provider type: {type(embedding_provider)}")

        graph_persist_path = os.path.join(config.working_dir, dataset_name, graph_name)
        ws_instance = Workspace(working_dir=str(config.working_dir), exp_name=dataset_name) #
        graph_namespace = NameSpace(workspace=ws_instance, namespace=graph_name) #
        
        _storage = NetworkXStorage() #
        _storage.namespace = graph_namespace
        _storage.name = "graph_storage_nx_data.graphml" 
        
        graph_factory_instance = GraphFactory() #
        graph_instance = graph_factory_instance.get_graph(
            config, 
            data_path=str(graph_persist_path),
            storage_type="networkx",       
            storage_instance=_storage
        )
        await graph_instance.load_persisted_graph() #
        logger.info(f"Planning Agent Test: Graph instance loaded. Node count: {graph_instance.node_num}. Type: {type(graph_instance)}") #

        vdb_persist_path = os.path.join(config.working_dir, dataset_name, graph_name, "entities_vdb")
        
        # Ensure embedding_provider is correctly passed if it's a LlamaIndex embedding object
        # FaissIndex might expect the core embedding model, not the LlamaIndex wrapper.
        # Adjust based on your FaissIndex implementation or ensure get_rag_embedding returns compatible type.
        # For now, assuming direct compatibility or that FaissIndex handles it.
        mock_faiss_config_dict = {
            "persist_path": vdb_persist_path,
            "embed_model": embedding_provider, 
            "name": "entities_vdb"
        }
        # If your FaissIndexConfig is a Pydantic model, use that. Otherwise, SimpleNamespace is fine for mocking.
        # from Config.IndexConfig import FaissIndexConfig # Example
        # faiss_config_obj = FaissIndexConfig(**mock_faiss_config_dict)
        faiss_config_obj = SimpleNamespace(**mock_faiss_config_dict)

        entities_vdb_instance = FaissIndex(config=faiss_config_obj) #
        if not await entities_vdb_instance.load(): #
            logger.error("Planning Agent Test: Failed to load entities VDB!")
            return None
        logger.info("Planning Agent Test: Entities VDB loaded successfully.")

        # Simplified retriever_config_dict for context, real values depend on your needs
        retriever_config_dict_for_context = config.retriever.model_dump() if config.retriever else {} #

        context = GraphRAGContext(
            target_dataset_name=dataset_name,
            graph_instance=graph_instance,
            entities_vdb_instance=entities_vdb_instance,
            relationships_vdb_instance=None, 
            chunk_vdb_instance=None,
            embedding_provider=embedding_provider,
            resolved_configs={
                "main_config_dict": config.model_dump(),
                "retriever_config_dict": retriever_config_dict_for_context,
                "storage_root_dir": config.working_dir
            },
        )
        logger.info("Planning Agent Test: GraphRAGContext created.")
        return context
    except Exception as e:
        logger.error(f"Planning Agent Test: Error setting up GraphRAGContext: {e}", exc_info=True)
        return None

async def run_planning_agent_test():
    logger.info("--- Starting Planning Agent Test ---")

    # 1. Load main configuration
    # The default_config from Option.Config2 should attempt to load Option/Config2.yaml
    # Ensure your API KEY is set in Option/Config2.yaml under llm.api_key
    try:
        config = main_config_loader # This should be the loaded Config object
        if not (config.llm and config.llm.api_key and config.llm.api_key not in ["sk-", "YOUR_API_KEY_OR_PLACEHOLDER"]): #
            logger.error("LLM API key is not properly configured in Option/Config2.yaml. Please set llm.api_key.")
            #return # Comment out to try running anyway if key is set via ENV or other means OpenaiApi handles
        logger.info(f"Loaded main config. LLM api_type: {config.llm.api_type}, Model: {config.llm.model}") #
    except Exception as e:
        logger.error(f"Failed to load main_config: {e}", exc_info=True)
        return

    # 2. Setup GraphRAGContext
    # This assumes your 'MySampleTexts' dataset with 'kg_graph' and 'entities_vdb'
    # has been built by previous runs (e.g., main.py build mode).
    graphrag_context = await setup_graphrag_context_for_agent(config=config, dataset_name="MySampleTexts", graph_name="kg_graph")
    
    if not graphrag_context:
        logger.error("Failed to setup GraphRAGContext for Planning Agent. Aborting test.")
        return
    if not graphrag_context.graph_instance or not graphrag_context.entities_vdb_instance:
         logger.error("Graph instance or Entities VDB is missing in GraphRAGContext. Aborting.")
         return

    # 3. Instantiate PlanningAgent
    try:
        agent = PlanningAgent(config=config, graphrag_context=graphrag_context)
    except Exception as e:
        logger.error(f"Failed to instantiate PlanningAgent: {e}", exc_info=True)
        return
    
    if not agent.llm_provider:
        logger.error("PlanningAgent's LLM provider was not initialized. Check config and OpenaiApi init. Aborting.")
        return
    if not agent.orchestrator:
        logger.error("PlanningAgent's Orchestrator was not initialized (likely due to missing GraphRAGContext). Aborting.")
        return


    # 4. Define a user query
    # Start with a simple query that ideally uses one or two tools
    # user_query = "What are some entities related to 'renewable energy'?" # Expected: Entity.VDBSearch
    user_query = "Find entities about 'climate change solutions' and then get their PPR scores." # Expected: VDB -> PPR

    logger.info(f"Test Query: \"{user_query}\"")

    # 5. Call the agent to process the query
    try:
        final_result = await agent.process_query(user_query)
    except Exception as e:
        logger.error(f"An error occurred during agent.process_query: {e}", exc_info=True)
        final_result = {"error": f"Exception during process_query: {e}"}


    # 6. Print the result
    logger.info("--- Planning Agent Test Finished ---")
    logger.info(f"Final Result from Planning Agent for query '{user_query}':")
    
    if isinstance(final_result, dict):
        # Pretty print if it's a dictionary (common for structured output or errors)
        import json
        logger.info(json.dumps(final_result, indent=2))
    else:
        logger.info(final_result)

if __name__ == "__main__":
    asyncio.run(run_planning_agent_test())
# END: /home/brian/digimon/testing/test_planning_agent.py