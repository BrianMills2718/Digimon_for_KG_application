# testing/test_agent_orchestrator.py
import asyncio
import os
import sys
import os # Ensure os is imported for os.getcwd()

# --- TOP LEVEL DIAGNOSTIC BLOCK ---
print("TOP LEVEL DIAG: Attempting to write /tmp/top_level_diag.txt", file=sys.stderr)
try:
    # Get current timestamp for uniqueness and to confirm execution time
    import datetime
    timestamp = datetime.datetime.now().isoformat()
    cwd = os.getcwd()
    with open("/tmp/top_level_diag.txt", "w") as f_top_diag:
        f_top_diag.write(f"Top level diagnostic executed at: {timestamp}\n")
        f_top_diag.write(f"Current working directory: {cwd}\n")
        f_top_diag.write("File write successful from top level.\n")
    print("TOP LEVEL DIAG: Successfully wrote /tmp/top_level_diag.txt", file=sys.stderr)
except Exception as e_top_diag:
    print(f"TOP LEVEL DIAG: FAILED to write /tmp/top_level_diag.txt. Error: {e_top_diag}", file=sys.stderr)
# --- END TOP LEVEL DIAGNOSTIC BLOCK ---
from pathlib import Path
from types import SimpleNamespace  # For ad-hoc config objects if needed

# Add project root to sys.path to allow imports from Core, Option, etc.
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from Core.AgentOrchestrator.orchestrator import AgentOrchestrator
from Core.AgentSchema.plan import ExecutionPlan, ExecutionStep, ToolCall, ToolInputSource, DynamicToolChainConfig
from Core.AgentSchema.context import GraphRAGContext
from Core.Common.Logger import logger
from Option.Config2 import Config, default_config as main_config  # Main config

# --- CAPTURE and TEST sys.stderr REFERENCE ---
captured_stderr_ref = sys.stderr
try:
    # Use a raw write, as print itself might be an issue or rely on other sys attributes
    captured_stderr_ref.write(f"DEBUG_STDERR_CAPTURE: sys.stderr is: {repr(sys.stderr)}\n")
    captured_stderr_ref.write(f"DEBUG_STDERR_CAPTURE: captured_stderr_ref is: {repr(captured_stderr_ref)}\n")
    captured_stderr_ref.flush()
except Exception as e_stderr_capture_write:
    # Fallback if direct write to captured_stderr_ref fails
    with open("/tmp/stderr_capture_write_error.txt", "w") as f_err_cap_write:
        f_err_cap_write.write(f"Error writing to captured_stderr_ref: {repr(e_stderr_capture_write)}\n")
        f_err_cap_write.write(f"sys.stderr was: {repr(sys.stderr)}\n")
        f_err_cap_write.write(f"captured_stderr_ref was: {repr(captured_stderr_ref)}\n")
# --- END CAPTURE and TEST sys.stderr REFERENCE ---

# --- CONFIG ATTRIBUTES DIAGNOSTIC BLOCK ---
# Try writing to captured_stderr_ref instead of relying on current sys.stderr via print
try:
    captured_stderr_ref.write("CONFIG DIAG: Attempting to access main_config and write /tmp/config_attrs_diag.txt\n")
    captured_stderr_ref.flush()
except Exception as e_pre_cfg_diag_print:
    with open("/tmp/config_diag_print_error.txt", "w") as f_cfg_print_err:
        f_cfg_print_err.write(f"Error writing initial CONFIG DIAG message: {repr(e_pre_cfg_diag_print)}\n")
try:
    # Ensure datetime is available if not already imported globally for this block
    import datetime
    timestamp_cfg = datetime.datetime.now().isoformat()
    cfg_working_dir = main_config.working_dir
    cfg_exp_name = main_config.exp_name
    with open("/tmp/config_attrs_diag.txt", "w") as f_cfg_diag:
        f_cfg_diag.write(f"Config attributes diagnostic executed at: {timestamp_cfg}\n")
        f_cfg_diag.write(f"main_config.working_dir: {cfg_working_dir}\n")
        f_cfg_diag.write(f"main_config.exp_name: {cfg_exp_name}\n")
        f_cfg_diag.write("File write successful from config_attrs_diag block.\n")
    captured_stderr_ref.write("CONFIG DIAG: Successfully accessed main_config and wrote /tmp/config_attrs_diag.txt\n")
    captured_stderr_ref.flush()
except Exception as e_cfg_diag:
    captured_stderr_ref.write(f"CONFIG DIAG: FAILED to access main_config or write /tmp/config_attrs_diag.txt. Error: {repr(e_cfg_diag)}\n")
    captured_stderr_ref.flush()
# --- END CONFIG ATTRIBUTES DIAGNOSTIC BLOCK ---
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
    # Use BaseGraph API: load_persisted_graph (not .load)
    await graph_instance.load_persisted_graph()
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

    # 2. Define a 2-step ExecutionPlan
    plan_description = "2-step plan: VDB search for 'american revolution', then PPR on results."
    user_query = "causes of the american revolution" # Used by VDB search

    # == STEP 1: Entity VDB Search ==
    step1_id = "step1_vdb_search"
    vdb_search_step = ExecutionStep(
        step_id=step1_id,
        description="Perform a VDB search for entities related to the user query.",
        action=DynamicToolChainConfig(
            tools=[
                ToolCall(
                    tool_id="Entity.VDBSearch",
                    description="Find initial relevant entities using VDB.",
                    parameters={
                        "vdb_reference_id": "entities_vdb",
                        "top_k_results": 3
                    },
                    inputs={
                        "query_text": "plan_inputs.main_query"
                    },
                    named_outputs={
                        "vdb_search_results_object": "Raw EntityVDBSearchOutputs object from VDB search."
                    }
                )
            ]
        )
    )

    # == STEP 2: Entity PPR ==
    step2_id = "step2_entity_ppr"
    entity_ppr_step = ExecutionStep(
        step_id=step2_id,
        description="Run Personalized PageRank on entities found by VDB search.",
        action=DynamicToolChainConfig(
            tools=[
                ToolCall(
                    tool_id="Entity.PPR",
                    description="Calculate PPR scores for seed entities.",
                    parameters={
                        "graph_reference_id": "kg_graph",
                        "personalization_weight_alpha": 0.25,
                        "max_iterations": 50,
                        "top_k_results": 5
                    },
                    inputs={
                        "seed_entity_ids": ToolInputSource(
                            from_step_id=step1_id,
                            named_output_key="vdb_search_results_object"
                        )
                    },
                    named_outputs={
                        "ppr_ranked_entities": "List of (entity_id, ppr_score) tuples from PPR."
                    }
                )
            ]
        )
    )

    test_plan = ExecutionPlan(
        plan_id="test_plan_002_vdb_then_ppr",
        plan_description=plan_description,
        target_dataset_name="MySampleTexts",
        plan_inputs={
            "main_query": user_query
        },
        steps=[vdb_search_step, entity_ppr_step]
    )

    logger.info(f"Orchestrator Test: ExecutionPlan defined:\n{test_plan.model_dump_json(indent=2)}")

    # 3. Instantiate and Run the Orchestrator
    orchestrator = AgentOrchestrator(graphrag_context=graphrag_context)
    
    logger.info("Orchestrator Test: Executing plan...")
    final_output = await orchestrator.execute_plan(plan=test_plan)

    logger.info(f"Orchestrator Test: Plan execution finished. Final output from orchestrator (last step's output): {final_output}")
    logger.info(f"Orchestrator Test: All step outputs: {orchestrator.step_outputs}") # Log all step outputs
    
    # --- Test Plan 3: VDB Search -> One-Hop Neighbors ---
    logger.info("\n--- Starting Test Plan 3: Entity.VDBSearch -> Relationship.OneHopNeighbors ---")
    orchestrator.step_outputs.clear() # Clear previous step outputs
    plan_3_inputs = {"main_query": "causes of the french revolution"} # Or another query
    await run_vdb_to_one_hop_neighbors_plan(orchestrator, plan_3_inputs)
    logger.info("--- Test Plan 3 Finished ---")

    logger.info("--- Agent Orchestrator Test Finished ---")

async def run_vdb_to_one_hop_neighbors_plan(orchestrator: AgentOrchestrator, plan_inputs: dict):
    """
    Defines and executes a 2-step plan: Entity.VDBSearch -> Relationship.OneHopNeighbors
    """
    plan_vdb_one_hop = ExecutionPlan(
        plan_id="test_plan_003_vdb_then_one_hop",
        plan_description="2-step plan: VDB search, then find one-hop neighbors.",
        target_dataset_name="MySampleTexts",
        plan_inputs=plan_inputs,
        steps=[
            ExecutionStep(
                step_id="step1_vdb_for_one_hop",
                description="Perform a VDB search for entities.",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="Entity.VDBSearch",
                            description="Find initial relevant entities using VDB.",
                            parameters={
                                "vdb_reference_id": "entities_vdb",
                                "top_k_results": 2
                            },
                            inputs={
                                "query_text": "plan_inputs.main_query"
                            },
                            named_outputs={
                                "vdb_search_output": "Output from VDB search containing entity IDs and scores."
                            }
                        )
                    ]
                )
            ),
            ExecutionStep(
                step_id="step2_one_hop_neighbors",
                description="Find one-hop neighbors for entities found by VDB search.",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="Relationship.OneHopNeighbors",
                            description="Get one-hop neighbors and their relationship details.",
                            parameters={},
                            inputs={
                                "entity_ids": ToolInputSource(
                                    from_step_id="step1_vdb_for_one_hop",
                                    named_output_key="vdb_search_output"
                                )
                            },
                            named_outputs={
                                "one_hop_neighbor_details": "List of NeighborDetail objects."
                            }
                        )
                    ]
                )
            )
        ]
    )
    logger.info(f"Orchestrator Test: ExecutionPlan defined:\n{plan_vdb_one_hop.model_dump_json(indent=2)}")
    logger.info("Orchestrator Test: Executing VDB -> OneHopNeighbors plan...")
    final_output = await orchestrator.execute_plan(plan_vdb_one_hop)
    logger.info(f"Orchestrator Test: VDB -> OneHopNeighbors Plan execution finished. Final output from orchestrator (last step's output): {final_output}")
    logger.info(f"Orchestrator Test: All step outputs for VDB -> OneHopNeighbors plan: {orchestrator.step_outputs}")

if __name__ == "__main__":
    asyncio.run(main_orchestrator_test())
