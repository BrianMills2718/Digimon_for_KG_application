# Python import path setup for test script
import os
import sys
import json
import uuid
import asyncio
from pathlib import Path
import logging # Using standard logging

# Configure standard logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-7s | %(name)s:%(funcName)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from Option.Config2 import Config
from Core.AgentBrain.agent_brain import PlanningAgent # Actual PlanningAgent
from Core.AgentSchema.context import GraphRAGContext
# Wrapper functions will use original tools
from Core.AgentTools.graph_construction_tools import build_er_graph as original_build_er_graph_tool
from Core.AgentTools.corpus_tools import prepare_corpus_from_directory as original_prepare_corpus_tool
from Core.AgentTools.entity_tools import entity_vdb_search_tool as original_entity_vdb_search_tool
from Core.AgentTools.relationship_tools import relationship_one_hop_neighbors_tool as original_relationship_one_hop_tool

from Core.Provider.LLMProviderRegister import create_llm_instance
from Core.Index.EmbeddingFactory import get_rag_embedding
from Core.Chunk.ChunkFactory import ChunkFactory

from Core.AgentSchema.corpus_tool_contracts import PrepareCorpusInputs, PrepareCorpusOutputs
from Core.AgentSchema.graph_construction_tool_contracts import BuildERGraphInputs, BuildERGraphOutputs, ERGraphConfigOverrides
from Core.AgentSchema.tool_contracts import EntityVDBSearchInputs, EntityVDBSearchOutputs, RelationshipOneHopNeighborsInputs, RelationshipOneHopNeighborsOutputs

# For loading graph and VDB after build_er_graph_tool execution
from Core.Graph.GraphFactory import get_graph as get_graph_factory_instance # Renamed import
from Core.Graph.ERGraph import ERGraph
from Core.Storage.NetworkXStorage import NetworkXStorage
from Core.Storage.NameSpace import Workspace, NameSpace
from Core.Index.FaissIndex import FaissIndex
# Assuming FaissIndexConfig is in Core.Index.Schema, adjust if different
from Core.Index.Schema import FAISSIndexConfig # Corrected capitalization
from Core.Storage.PickleBlobStorage import PickleBlobStorage
from Core.Graph.BaseGraph import BaseGraph # For type hinting
from Core.Index.BaseIndex import BaseIndex # For type hinting
from typing import Any, Optional # For type hinting
from llama_index.core.embeddings import BaseEmbedding as LlamaIndexBaseEmbedding # Correct type hint

# This single GraphRAGContext instance will be created in the test function and passed around.
# Wrappers will update this instance.
shared_graphrag_context_instance: Optional[GraphRAGContext] = None

# --- Wrapper Functions: To intercept tool calls and manage context ---

async def prepare_corpus_wrapper(tool_input: PrepareCorpusInputs, main_config: Config) -> PrepareCorpusOutputs:
    global shared_graphrag_context_instance
    logger.info(f"WRAPPER (Corpus): Called with input: {tool_input.model_dump_json(indent=1)}")
    logger.info(f"WRAPPER (Corpus): Using shared context ID: {id(shared_graphrag_context_instance)} if available, or None.")
    # The original tool is called with main_config as per orchestrator's logic for this tool
    return await original_prepare_corpus_tool(tool_input, main_config)

async def build_er_graph_wrapper(
    tool_input: BuildERGraphInputs,
    main_config: Config,
    llm_instance: Any, # BaseLLM type hint would be good if BaseLLM is imported
    encoder_instance: LlamaIndexBaseEmbedding, # Correct type hint
    chunk_factory: ChunkFactory
) -> BuildERGraphOutputs:
    global shared_graphrag_context_instance
    if shared_graphrag_context_instance is None:
        logger.error("CRITICAL WRAPPER ERROR: shared_graphrag_context_instance is None in build_er_graph_wrapper!")
        return BuildERGraphOutputs(graph_id="", status="failure", message="Shared context not available in wrapper.", artifact_path=None, node_count=0, edge_count=0)

    logger.info(f"WRAPPER (BuildERGraph): Called. Shared context ID: {id(shared_graphrag_context_instance)}. Input: {tool_input.model_dump_json(indent=1)}")
    
    build_tool_output = await original_build_er_graph_tool(
        tool_input, main_config, llm_instance, encoder_instance, chunk_factory
    )

    if build_tool_output.status == "success" and build_tool_output.artifact_path and build_tool_output.graph_id:
        logger.info(f"WRAPPER (BuildERGraph): Tool success. Graph ID: {build_tool_output.graph_id}, Path: {build_tool_output.artifact_path}")
        try:
            logger.info(f"WRAPPER (BuildERGraph): Loading ERGraph '{build_tool_output.graph_id}' into context.")
            graph_artifact_dir = Path(build_tool_output.artifact_path)
            dataset_name_for_context = build_tool_output.graph_id.replace("_ERGraph", "")

            temp_config_for_load = main_config.model_copy(deep=True)
            temp_config_for_load.graph.type = "er_graph"

            # Ensure ERGraph is imported for type check, and NetworkXStorage for storage_instance
            # This part relies on get_graph_factory_instance correctly returning an ERGraph instance
            # with a NetworkXStorage instance as its ._graph attribute.
            loaded_er_graph_instance: ERGraph = get_graph_factory_instance(
                config=temp_config_for_load,
                llm=llm_instance,
                encoder=encoder_instance,
                storage_instance=NetworkXStorage() # Provide a new storage instance for loading
            )
            
            if hasattr(loaded_er_graph_instance._graph, 'namespace') and isinstance(loaded_er_graph_instance._graph, NetworkXStorage):
                ws = Workspace(working_dir=str(main_config.working_dir), exp_name=dataset_name_for_context)
                ns = NameSpace(workspace=ws, namespace="er_graph")
                loaded_er_graph_instance._graph.namespace = ns
                loaded_er_graph_instance._graph.name = "nx_data.graphml"
                logger.info(f"WRAPPER (BuildERGraph): Configured NetworkXStorage for loading from: {loaded_er_graph_instance._graph.graphml_xml_file}")
            
            await loaded_er_graph_instance._load_graph()
            logger.info(f"WRAPPER (BuildERGraph): Loaded graph. Nodes: {loaded_er_graph_instance.node_num}, Edges: {loaded_er_graph_instance.edge_num}")
            shared_graphrag_context_instance.add_graph_instance(build_tool_output.graph_id, loaded_er_graph_instance)

            logger.info(f"WRAPPER (BuildERGraph): Building VDB 'entities_vdb' for graph '{build_tool_output.graph_id}'.")
            vdb_persist_path = graph_artifact_dir / "faiss_entities_vdb_test_pipeline" # Unique name for this test VDB
            vdb_persist_path.mkdir(parents=True, exist_ok=True)
            
            faiss_storage_ws = Workspace(working_dir=str(graph_artifact_dir), exp_name="faiss_entities_vdb_test_pipeline")
            faiss_storage_ns = NameSpace(workspace=faiss_storage_ws, namespace="")

            # Ensure FAISSIndexConfig is correctly imported and expects embed_model
            faiss_config = FAISSIndexConfig(
                collection_name="entities_vdb",
                path_suffix="", 
                persist_path=str(vdb_persist_path), 
                embed_model=encoder_instance 
            )
            
            entity_vdb = FaissIndex(config=faiss_config)
            entity_vdb.storage_instance = PickleBlobStorage(namespace=faiss_storage_ns, name="vector_store_entities_vdb")

            nodes_data = await loaded_er_graph_instance.nodes_data()
            documents_for_vdb = [
                {"id": str(node.get("entity_name", node.get("id", uuid.uuid4().hex))), 
                 "content": node.get("description") or node.get("content") or str(node.get("entity_name",""))
                } for node in nodes_data if node.get("description") or node.get("content") or node.get("entity_name")
            ]
            
            from llama_index.core.schema import Document
            if documents_for_vdb:
                logger.info(f"WRAPPER (BuildERGraph): Indexing {len(documents_for_vdb)} docs in VDB 'entities_vdb'.")
                # FaissIndex.build_index takes elements (list of dicts) and meta_data (list of metadata keys)
                await entity_vdb.build_index(
                    elements=documents_for_vdb, 
                    meta_data=["id", "content", "name"], 
                    force=True
                )
                shared_graphrag_context_instance.add_vdb_instance("entities_vdb", entity_vdb)
                logger.info(f"WRAPPER (BuildERGraph): VDB 'entities_vdb' added to context ID {id(shared_graphrag_context_instance)}.")
            else:
                logger.warning("WRAPPER (BuildERGraph): No suitable documents from graph to build VDB.")
        except Exception as e:
            logger.error(f"WRAPPER (BuildERGraph): Error in post-build context update: {e}", exc_info=True)
    return build_tool_output

async def entity_vdb_search_wrapper(tool_input: EntityVDBSearchInputs, context: GraphRAGContext) -> EntityVDBSearchOutputs:
    global shared_graphrag_context_instance
    logger.info(f"WRAPPER (VDB Search): Called. Input: {tool_input.model_dump_json(indent=1)}. Using context ID: {id(context)}")
    if context is not shared_graphrag_context_instance:
        logger.error(f"WRAPPER (VDB Search): Context mismatch! Expected {id(shared_graphrag_context_instance)}, got {id(context)}")
    # Ensure the actual tool function is called with the context it expects
    return await original_entity_vdb_search_tool(tool_input, context)

async def relationship_one_hop_wrapper(tool_input: RelationshipOneHopNeighborsInputs, context: GraphRAGContext) -> RelationshipOneHopNeighborsOutputs:
    global shared_graphrag_context_instance
    logger.info(f"WRAPPER (OneHop): Called. Input: {tool_input.model_dump_json(indent=1)}. Using context ID: {id(context)}")
    if context is not shared_graphrag_context_instance:
         logger.error(f"WRAPPER (OneHop): Context mismatch! Expected {id(shared_graphrag_context_instance)}, got {id(context)}")
    return await original_relationship_one_hop_tool(tool_input, context)

async def test_agent_corpus_to_graph_pipeline():
    logger.info("Starting test_agent_corpus_to_graph_pipeline...")
    
    global shared_graphrag_context_instance
    
    # Load configuration using the default method
    main_config = Config.default()
    
    # Initialize LLM and Encoder
    llm_instance = create_llm_instance(main_config.llm)
    encoder_instance = get_rag_embedding(config=main_config)
    
    # Initialize ChunkFactory with correct parameter
    chunk_factory = ChunkFactory(main_config)

    main_config.data_root = str(project_root / "Data")
    main_config.working_dir = str(project_root / "results")
    Path(main_config.working_dir).mkdir(parents=True, exist_ok=True)

    if not main_config.llm:
        logger.error("LLM config missing in main_config. Exiting.")
        return
    main_config.llm.max_token = 8192
    main_config.llm.temperature = 0.2

    # Test-specific paths
    dataset_name_for_plan = "MyPipelineTestRun"
    dataset_name_for_context = dataset_name_for_plan # Use same name for context consistency
    
    # Use the MySampleTexts directory which contains american_revolution.txt
    input_dir_path = str(Path(main_config.data_root) / "MySampleTexts")
    output_dir_path = str(Path(main_config.working_dir) / dataset_name_for_plan)
    Path(output_dir_path).mkdir(parents=True, exist_ok=True)

    shared_graphrag_context_instance = GraphRAGContext(
        request_id=str(uuid.uuid4())[:8],
        target_dataset_name=dataset_name_for_plan,
        main_config=main_config,
        llm_provider=llm_instance,
        embedding_provider=encoder_instance,
        chunk_storage_manager=chunk_factory, # This now aligns with GraphRAGContext definition
        graphs={},
        vdbs={}
    )
    logger.info(f"Main test: Initialized shared_graphrag_context_instance with ID: {id(shared_graphrag_context_instance)}")

    agent_brain = PlanningAgent(config=main_config, graphrag_context=shared_graphrag_context_instance)

    if agent_brain.orchestrator and hasattr(agent_brain.orchestrator, '_tool_registry'):
        registry = agent_brain.orchestrator._tool_registry
        registry["corpus.PrepareFromDirectory"] = (prepare_corpus_wrapper, PrepareCorpusInputs)
        registry["graph.BuildERGraph"] = (build_er_graph_wrapper, BuildERGraphInputs)
        registry["Entity.VDBSearch"] = (entity_vdb_search_wrapper, EntityVDBSearchInputs)
        registry["Relationship.OneHopNeighbors"] = (relationship_one_hop_wrapper, RelationshipOneHopNeighborsInputs)
        logger.info("Patched tool registry with wrappers.")
    else:
        logger.error("Failed to patch tool registry.")
        return

    SYSTEM_TASK = (
        f"SYSTEM_TASK: Your overall goal is to provide a summary about the causes of the American Revolution based on documents in '{input_dir_path}'. To achieve this: "
        f"1. Process the text files from the input directory '{input_dir_path}' to create a corpus. "
        f"The 'output_directory_path' for this step must be '{output_dir_path}'. "
        f"The 'target_corpus_name' for this step must be '{dataset_name_for_plan}'. "
        f"The named outputs for this step MUST be {{'corpus_status': 'status', 'corpus_path': 'corpus_json_path', 'doc_count': 'document_count'}}. "
        
        f"2. After the corpus is prepared, build an Entity-Relation Graph (ERGraph). "
        f"The 'target_dataset_name' for this BuildERGraph step must be '{dataset_name_for_plan}'. "
        f"Set 'force_rebuild' to true. For 'config_overrides', use: "
        f"'extract_two_step'=true, 'enable_entity_description'=true, and 'enable_entity_type'=true. "
        f"The named outputs for this step MUST be {{'graph_id_from_build': 'graph_id', 'status_from_build': 'status'}}. This maps the tool's output field 'graph_id' to 'graph_id_from_build' in the plan context, and 'status' to 'status_from_build'. "

        f"3. Once the graph is built for dataset '{dataset_name_for_plan}', perform a VDB search on its entities. "
        f"The 'vdb_reference_id' for this search must be 'entities_vdb'. "
        f"Use the query 'causes of the American Revolution'. Retrieve the top 5 most relevant entities. "
        f"The named outputs for this step MUST be {{'vdb_search_results_list': 'similar_entities'}}. This maps the tool's output field 'similar_entities' to 'vdb_search_results_list' in the plan context. "

        f"4. For the top 3 entities found, find their one-hop neighbors and relationships. "
        f"The 'graph_reference_id' input for this step must be {{'from_step_id': 'step_2_build_er_graph', 'named_output_key': 'graph_id_from_build'}}. "
        f"The 'entity_ids' input for this step must be {{'from_step_id': 'step_3_vdb_search', 'named_output_key': 'vdb_search_results_list'}}. "
        f"The named outputs for this step MUST be {{'final_neighbor_info': 'one_hop_relationships'}}. "

        "Finally, use ALL the information retrieved from steps 3 and 4 to generate a concise natural language summary "
        "answering the question about the causes of the American Revolution."
    )
    
    logger.info(f"Test: Submitting task to agent:\n{SYSTEM_TASK}")
    agent_response = await agent_brain.process_query(SYSTEM_TASK)
    logger.info(f"Test: Agent final response: {json.dumps(agent_response, indent=2, default=str)}")
    
    # Print detailed step outputs for debugging
    if agent_response and isinstance(agent_response.get("retrieved_context"), dict):
        all_step_outputs = agent_response["retrieved_context"] # This is now a dict of dicts
        logger.info(f"✅ VERIFY: All step outputs from orchestrator: {json.dumps(all_step_outputs, indent=2, default=str)}")

        step3_output_container = all_step_outputs.get("step_3_vdb_search", {}) # Step ID from plan
        vdb_results = step3_output_container.get("vdb_search_results_list") # Named output key from plan for step 3
        if vdb_results is not None: # Check for None explicitly if empty list is valid
            logger.info(f"✅ VERIFY: 'vdb_search_results_list' (type: {type(vdb_results)}) found: {str(vdb_results)[:200]}...")
            if isinstance(vdb_results, list) and not vdb_results:
                logger.warning("⚠️ VERIFY: 'vdb_search_results_list' is an EMPTY LIST.")
        else:
            logger.warning("⚠️ VERIFY: 'vdb_search_results_list' NOT found in step_3_vdb_search outputs.")

        step4_output_container = all_step_outputs.get("step_4_one_hop_neighbors", {}) # Step ID from plan
        neighbor_rels = step4_output_container.get("final_neighbor_info") # Named output key from plan for step 4
        if neighbor_rels is not None:
            logger.info(f"✅ VERIFY: 'final_neighbor_info' (type: {type(neighbor_rels)}) found: {str(neighbor_rels)[:200]}...")
            if isinstance(neighbor_rels, list) and not neighbor_rels:
                 logger.warning("⚠️ VERIFY: 'final_neighbor_info' is an EMPTY LIST.")
        else:
            logger.warning("⚠️ VERIFY: 'final_neighbor_info' NOT found in step_4_one_hop_neighbors outputs.")

    final_answer = agent_response.get("generated_answer", "No answer generated.")
    logger.info(f"Agent's Final Generated Answer:\n{final_answer}")

    corpus_json_path = Path(output_dir_path) / "Corpus.json"
    graph_artifact_path = Path(output_dir_path) / "er_graph"
    ergraph_file_path = graph_artifact_path / "nx_data.graphml"

    # Corpus verification: compare line count to document_count from step 1
    docs_in_corpus_from_tool = 0
    if agent_response and isinstance(agent_response.get("retrieved_context"), dict):
        step1_output_container = agent_response["retrieved_context"].get("step_1_prepare_corpus", {}) # Outputs are keyed by step_id
        step1_tool_output = step1_output_container.get("corpus_prep_result") # Then by named_output key

        if isinstance(step1_tool_output, PrepareCorpusOutputs):
            docs_in_corpus_from_tool = step1_tool_output.document_count
        elif isinstance(step1_tool_output, dict): # If it was model_dumped
            docs_in_corpus_from_tool = step1_tool_output.get("document_count", 0)

    if corpus_json_path.exists(): 
        logger.info(f"✅ VERIFY: Corpus.json found at {corpus_json_path}")
        lines_in_corpus = 0
        try:
            with open(corpus_json_path, 'r', encoding='utf-8') as f: # Add encoding
                for _ in f: lines_in_corpus +=1
            logger.info(f"Corpus.json physically contains {lines_in_corpus} lines. Tool reported processing: {docs_in_corpus_from_tool} docs.")
            if docs_in_corpus_from_tool != lines_in_corpus and docs_in_corpus_from_tool > 0: # Allow 0 if tool failed
                logger.warning(f"Potential mismatch in reported documents ({docs_in_corpus_from_tool}) vs lines in file ({lines_in_corpus}).")
        except Exception as e:
            logger.error(f"Could not count lines in Corpus.json: {e}")
    else: 
        logger.error(f"❌ VERIFY: Corpus.json NOT FOUND at {corpus_json_path}")

    if ergraph_file_path.exists():
        logger.info(f"✅ SUCCESS: ERGraph file found at {ergraph_file_path}")
    else:
        logger.error(f"❌ FAILURE: ERGraph file NOT FOUND at {ergraph_file_path}")

if __name__ == "__main__":
    current_script_path = Path(__file__).resolve()
    project_root_script = current_script_path.parent.parent
    if str(project_root_script) not in sys.path:
        sys.path.insert(0, str(project_root_script))
    asyncio.run(test_agent_corpus_to_graph_pipeline())