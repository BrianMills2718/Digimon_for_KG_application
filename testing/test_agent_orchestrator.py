# testing/test_agent_orchestrator.py
import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace  # For ad-hoc config objects if needed

# Add project root to sys.path to allow imports from Core, Option, etc.
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from Core.AgentOrchestrator.orchestrator import AgentOrchestrator
from Core.AgentSchema.plan import ExecutionPlan, ExecutionStep, ToolCall, ToolInputSource
from Core.AgentSchema.context import GraphRAGContext
from Core.Common.Logger import logger
from Option.Config2 import default_config as main_config  # Main config
from Core.Index.EmbeddingFactory import get_rag_embedding
from Core.Index.FaissIndex import FaissIndex
from Core.Storage.NetworkXStorage import NetworkXStorage
from Core.Graph.GraphFactory import GraphFactory  # Import the class, not a singleton
from Core.Storage.NameSpace import NameSpace, Workspace

# --- Helper function to set up GraphRAGContext (adapted from test_single_tool_entity_ppr.py) ---
async def setup_graphrag_context_for_orchestrator(dataset_name="MySampleTexts", graph_name="kg_graph") -> GraphRAGContext:
    logger.info("Orchestrator Test: Setting up GraphRAGContext...")
    
    # 1. Embedding Provider
    embedding_provider = get_rag_embedding(config=main_config)
    logger.info(f"Orchestrator Test: Embedding provider type: {type(embedding_provider)}")

    # 2. Graph Instance
    graph_persist_path = os.path.join(main_config.working_dir, dataset_name, graph_name) # This path is for BaseGraph.data_path

    # Corrected Workspace and NameSpace instantiation:
    ws_instance = Workspace(
        working_dir=str(main_config.working_dir),  # e.g., "./results"
        exp_name=dataset_name                      # e.g., "MySampleTexts"
    )
    graph_namespace = NameSpace(
        workspace=ws_instance,
        namespace=graph_name                       # e.g., "kg_graph"
    )
    
    _storage = NetworkXStorage() 
    _storage.namespace = graph_namespace 
    _storage.name = "graph_storage_nx_data.graphml" # Actual filename NetworkXStorage should look for
    
    # This uses the GraphFactory instance and its get_graph method
    graph_factory_instance = GraphFactory() 
    graph_instance = graph_factory_instance.get_graph(
        main_config,  # Pass the full main_config as the first argument for the 'config' param in get_graph
        data_path=str(graph_persist_path),
        storage_type="networkx",        
        storage_instance=_storage
    )
    await graph_instance.load()
    logger.info(f"Orchestrator Test: Graph instance loaded. Node count: {graph_instance.node_num}. Type: {type(graph_instance)}")
    if graph_instance:
        logger.info(f"Orchestrator Test: Type of graph_instance from factory: {type(graph_instance)}")
    else:
        logger.error("Orchestrator Test: GraphFactory returned None for graph_instance!")

    # 3. Entities VDB Instance
    vdb_persist_path = os.path.join(main_config.working_dir, dataset_name, graph_name, "entities_vdb")
    
    # Mock config for FaissIndex
    mock_faiss_config_dict = {
        "persist_path": vdb_persist_path,
        "embed_model": embedding_provider,  # Use the LlamaIndex embedding provider
        "name": "entities_vdb"
    }
    faiss_config_obj = SimpleNamespace(**mock_faiss_config_dict)  # Or your actual FAISSIndexConfig Pydantic model
    
    entities_vdb_instance = FaissIndex(config=faiss_config_obj)
    if not await entities_vdb_instance.load():
        logger.error("Orchestrator Test: Failed to load entities VDB for context setup!")
        entities_vdb_instance = None  # Ensure it's None if load fails
    else:
        logger.info("Orchestrator Test: Entities VDB loaded successfully for context setup.")

    # 4. RetrieverConfig
    retriever_config_dict_for_context = {
        "retriever_type": "EntityRetriever",  # Example
        "top_k": 10,
        "use_entity_similarity_for_ppr": True,
        "node_specificity": True,
        "top_k_entity_for_ppr": 5,
        "llm_config": main_config.llm.model_dump() if main_config.llm else {},
        "embedding_config": main_config.embedding.model_dump() if main_config.embedding else {}
    }

    # 5. Create GraphRAGContext
    context = GraphRAGContext(
        target_dataset_name=dataset_name,
        graph_instance=graph_instance,
        entities_vdb_instance=entities_vdb_instance,
        relationships_vdb_instance=None,  # Placeholder
        chunk_vdb_instance=None,  # Placeholder
        embedding_provider=embedding_provider,
        resolved_configs={
            "main_config_dict": main_config.model_dump(),
            "retriever_config_dict": retriever_config_dict_for_context,
            "storage_root_dir": main_config.working_dir
        },
    )
    logger.info("Orchestrator Test: GraphRAGContext created.")
    return context

async def main_orchestrator_test():
    logger.info("--- Starting Agent Orchestrator Test ---")

    # 1. Setup GraphRAGContext
    graphrag_context = await setup_graphrag_context_for_orchestrator()
    if not graphrag_context.graph_instance or not graphrag_context.entities_vdb_instance:
        logger.error("Orchestrator Test: Failed to set up essential components in GraphRAGContext. Aborting test.")
        return

    # 2. Define a simple ExecutionPlan
    plan_description = "Simple plan to perform a VDB search for 'american revolution'."
    user_query = "causes of the american revolution"

    vdb_search_step = ExecutionStep(
        step_id="step1_vdb_search",
        description="Perform a VDB search for entities related to the user query.",
        action=ToolCall(
            tool_id="Entity.VDBSearch",
            parameters={
                "vdb_reference_id": "entities_vdb",
                "top_k_results": 3
            },
            inputs={
                "query_text": "plan_inputs.main_query"
            },
            named_outputs={
                "found_entities": "List of entities found by VDB search with their scores."
            }
        )
    )
    
    test_plan = ExecutionPlan(
        plan_id="test_plan_001",
        plan_description=plan_description,
        target_dataset_name="MySampleTexts",
        plan_inputs={
            "main_query": user_query
        },
        steps=[vdb_search_step]
    )

    logger.info(f"Orchestrator Test: ExecutionPlan defined:\n{test_plan.model_dump_json(indent=2)}")

    # 3. Instantiate and Run the Orchestrator
    orchestrator = AgentOrchestrator(graphrag_context=graphrag_context)
    
    logger.info("Orchestrator Test: Executing plan...")
    final_output = await orchestrator.execute_plan(plan=test_plan)

    logger.info(f"Orchestrator Test: Plan execution finished. Final output from orchestrator: {final_output}")
    
    logger.info("--- Agent Orchestrator Test Finished ---")

if __name__ == "__main__":
    asyncio.run(main_orchestrator_test())
