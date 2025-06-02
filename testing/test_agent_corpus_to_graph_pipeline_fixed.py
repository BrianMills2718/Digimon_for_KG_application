import os
import sys
import json
import uuid
import asyncio
import logging
import networkx as nx
from pathlib import Path
from typing import Optional, Any, List, Dict

# Configure standard logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-7s | %(name)s:%(funcName)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from Option.Config2 import Config
from Core.AgentBrain.agent_brain import PlanningAgent
from Core.AgentSchema.context import GraphRAGContext
from Core.AgentTools.graph_construction_tools import build_er_graph as original_build_er_graph
from Core.AgentTools.corpus_tools import prepare_corpus_from_directory as original_prepare_corpus
from Core.AgentTools.entity_tools import entity_vdb_search_tool as original_entity_vdb_search
from Core.AgentTools.relationship_tools import relationship_one_hop_neighbors_tool as original_relationship_one_hop

from Core.Provider.LLMProviderRegister import create_llm_instance
from Core.Index.EmbeddingFactory import get_rag_embedding
from Core.Chunk.ChunkFactory import ChunkFactory

from Core.AgentSchema.corpus_tool_contracts import PrepareCorpusInputs, PrepareCorpusOutputs
from Core.AgentSchema.graph_construction_tool_contracts import BuildERGraphInputs, BuildERGraphOutputs, ERGraphConfigOverrides
from Core.AgentSchema.tool_contracts import EntityVDBSearchInputs, EntityVDBSearchOutputs, RelationshipOneHopNeighborsInputs, RelationshipOneHopNeighborsOutputs

# For loading graph and VDB after build_er_graph execution
from Core.Graph.GraphFactory import get_graph as get_graph_factory_instance
from Core.Graph.ERGraph import ERGraph
from Core.Storage.NetworkXStorage import NetworkXStorage
from Core.Storage.NameSpace import Workspace, NameSpace
from Core.Index.FaissIndex import FaissIndex
from Core.Index.Schema import FaissIndexConfig
from Core.Storage.PickleBlobStorage import PickleBlobStorage

from Core.AgentOrchestrator.orchestrator import AgentOrchestrator
from Core.AgentSchema.plans import Plan
from Core.AgentTools.tool_registry import ToolRegistry


# Create a simple wrapper class for handling inconsistent parameter patterns
class FixedToolCall:
    """A wrapper class to handle inconsistent parameter/input patterns in LLM-generated plans"""
    
    @staticmethod
    def get_prepare_corpus_inputs(tool_inputs_or_params):
        """Create PrepareCorpusInputs from either inputs or parameters"""
        if hasattr(tool_inputs_or_params, 'directory_path'):
            # Already a PrepareCorpusInputs
            return tool_inputs_or_params
        
        # Create from parameters
        return PrepareCorpusInputs(
            directory_path=tool_inputs_or_params.get('directory_path', ''),
            corpus_id=tool_inputs_or_params.get('corpus_id', f'corpus_{uuid.uuid4().hex[:8]}'),
            include_subdirectories=tool_inputs_or_params.get('include_subdirectories', True),
            file_types=tool_inputs_or_params.get('file_types', ['.txt', '.md', '.pdf']),
            target_chunks_count=tool_inputs_or_params.get('target_chunks_count', 10),
            overlap_percentage=tool_inputs_or_params.get('overlap_percentage', 10)
        )
    
    @staticmethod
    def get_build_er_graph_inputs(tool_inputs_or_params):
        """Create BuildERGraphInputs from either inputs or parameters"""
        if hasattr(tool_inputs_or_params, 'corpus_path'):
            # Already a BuildERGraphInputs
            return tool_inputs_or_params
        
        # Create overrides if specified
        overrides = None
        if 'graph_config_overrides' in tool_inputs_or_params:
            config_dict = tool_inputs_or_params.get('graph_config_overrides', {})
            overrides = ERGraphConfigOverrides(**config_dict)
        
        # Create from parameters
        return BuildERGraphInputs(
            corpus_path=tool_inputs_or_params.get('corpus_path', ''),
            graph_id=tool_inputs_or_params.get('graph_id', f'graph_{uuid.uuid4().hex[:8]}'),
            graph_config_overrides=overrides
        )


# Wrapper functions for all tools that need context awareness
async def patched_prepare_corpus(params, graphrag_context):
    """Simple wrapper for prepare_corpus_from_directory that passes through to original tool"""
    logger.info(f"Running patched_prepare_corpus with context ID: {graphrag_context.request_id}")
    return await original_prepare_corpus(params, graphrag_context)


async def build_er_graph_wrapper(params, graphrag_context):
    """Wrapper for build_er_graph tool that loads the graph and creates entity VDB after graph building"""
    # Get a reference to the shared context
    shared_graphrag_context = graphrag_context
    logger.info(f"Running build_er_graph_wrapper with shared context ID: {shared_graphrag_context.request_id}")
    
    # Call the actual tool to build the graph
    build_tool_output = await original_build_er_graph(params, graphrag_context)
    
    # Check if build was successful
    if build_tool_output.status == "success" and build_tool_output.artifact_path:
        logger.info(f"Graph built, now loading '{build_tool_output.graph_id}' from {build_tool_output.artifact_path} into shared context.")
        
        try:
            # Initialize graph storage
            graph_storage = NetworkXStorage()
            
            # Configure the namespace for the storage
            ws = Workspace(working_dir=str(shared_graphrag_context.main_config.working_dir), 
                          exp_name=build_tool_output.graph_id.replace("_ERGraph", ""))
            graph_storage.namespace = NameSpace(workspace=ws, namespace=shared_graphrag_context.main_config.graph.type)
            graph_storage.name = "nx_data"
            
            # Create the ERGraph instance
            loaded_graph_instance = ERGraph(
                config=shared_graphrag_context.main_config.graph,
                llm=shared_graphrag_context.llm_provider,
                encoder=shared_graphrag_context.embedding_provider,
                chunk_factory=ChunkFactory(shared_graphrag_context.main_config),
                storage_instance=graph_storage
            )
            
            # Load the graph data
            await loaded_graph_instance.load_graph(force_rebuild=False)
            
            # Add the graph to the shared context
            shared_graphrag_context.add_graph_instance(build_tool_output.graph_id, loaded_graph_instance)
            logger.info(f"Successfully loaded and added graph '{build_tool_output.graph_id}' to shared context.")
            
            # Build and register entity VDB
            logger.info(f"Building and registering VDB 'entities_vdb' for graph '{build_tool_output.graph_id}'...")
            
            # Extract entities from the loaded graph
            entities_for_vdb = []
            for n in loaded_graph_instance._graph.graph.nodes:
                node_data = loaded_graph_instance._graph.graph.nodes[n]
                entities_for_vdb.append({
                    "id": n,
                    "text": node_data.get("description", n),
                    "name": node_data.get("name", n)
                })
            
            if entities_for_vdb:
                # Create entity VDB
                entity_vdb = FaissIndex(
                    config=FaissIndexConfig(collection_name="entities_vdb", path_suffix="entities_vdb"),
                    embedding_provider=shared_graphrag_context.embedding_provider,
                    storage_instance=PickleBlobStorage(
                        namespace=graph_storage.namespace,
                        name="entities_vdb_faiss"
                    )
                )
                
                # Build the index from entities
                await entity_vdb.build_index_from_documents(entities_for_vdb)
                
                # Add VDB to the shared context
                shared_graphrag_context.add_vdb_instance("entities_vdb", entity_vdb)
                logger.info(f"Successfully built and added VDB 'entities_vdb' to shared context with {len(entities_for_vdb)} entities.")
            else:
                logger.warning("No entities found in the loaded graph to build VDB.")
                
        except Exception as e:
            logger.error(f"Error loading/registering graph or VDB into context after build: {e}", exc_info=True)
    
    # Return the original output
    return build_tool_output


async def patched_entity_vdb_search(params, graphrag_context):
    """Wrapper for entity_vdb_search_tool that ensures the proper VDB is used from shared context"""
    logger.info(f"Running patched_entity_vdb_search with context ID: {graphrag_context.request_id}")
    logger.info(f"VDB reference ID: {params.vdb_reference_id if hasattr(params, 'vdb_reference_id') else 'Not specified'}")
    
    # Check if vdb_reference_id is specified and exists in context
    if hasattr(params, 'vdb_reference_id') and params.vdb_reference_id:
        vdb_id = params.vdb_reference_id
        vdb = graphrag_context.get_vdb_instance(vdb_id)
        if vdb:
            logger.info(f"Found VDB '{vdb_id}' in shared context")
        else:
            logger.warning(f"VDB '{vdb_id}' not found in shared context")
    
    # Call the original tool
    return await original_entity_vdb_search(params, graphrag_context)


async def patched_relationship_one_hop_neighbors(params, graphrag_context):
    """Wrapper for relationship_one_hop_neighbors_tool that ensures the proper graph is used from shared context"""
    logger.info(f"Running patched_relationship_one_hop_neighbors with context ID: {graphrag_context.request_id}")
    logger.info(f"Graph reference ID: {params.graph_reference_id if hasattr(params, 'graph_reference_id') else 'Not specified'}")
    
    # Check if graph_reference_id is specified and exists in context
    if hasattr(params, 'graph_reference_id') and params.graph_reference_id:
        graph_id = params.graph_reference_id
        graph = graphrag_context.get_graph_instance(graph_id)
        if graph:
            logger.info(f"Found graph '{graph_id}' in shared context")
        else:
            logger.warning(f"Graph '{graph_id}' not found in shared context")
    
    # Call the original tool
    return await original_relationship_one_hop(params, graphrag_context)


async def test_agent_corpus_to_graph_pipeline():
    """Test the full pipeline from corpus preparation to graph building and querying with shared context"""
    
    # Create corpus output directory
    corpus_output_dir_path = "./Outputs/corpus"
    corpus_output_dir = Path(corpus_output_dir_path)
    corpus_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Paths for corpus and agent task
    corpus_directory = "./testdata/corpus/simple"
    corpus_path = os.path.join(corpus_output_dir_path, "Corpus.json")
    
    # Create a global shared GraphRAGContext instance with a unique ID
    global_graphrag_context = GraphRAGContext()
    global_graphrag_context.request_id = f"test_req_{uuid.uuid4().hex[:8]}"
    global_graphrag_context.context_id = global_graphrag_context.request_id
    logger.info(f"Created global GraphRAGContext with ID: {global_graphrag_context.request_id}")
    
    # Load configuration
    config_path = "Option/config.json"
    config = Config()
    config.load_config(config_path)
    
    # Initialize providers
    llm_instance = create_llm_instance(config)
    encoder_instance = get_rag_embedding(config)
    
    # Set up the config and providers in the context
    global_graphrag_context.main_config = config
    global_graphrag_context.llm_provider = llm_instance
    global_graphrag_context.embedding_provider = encoder_instance
    
    # Create a planning agent
    agent = PlanningAgent(
        llm_provider=llm_instance,
        graphrag_context=global_graphrag_context
    )
    
    # Create an agent orchestrator with patched tools
    tool_registry = ToolRegistry()
    
    # Register the original tools
    tool_registry.register_tool("prepare_corpus_from_directory", original_prepare_corpus)
    tool_registry.register_tool("build_er_graph", original_build_er_graph)
    tool_registry.register_tool("entity_vdb_search_tool", original_entity_vdb_search)
    tool_registry.register_tool("relationship_one_hop_neighbors_tool", original_relationship_one_hop)
    
    # Patch the registry with our wrapper functions
    tool_registry.register_tool("prepare_corpus_from_directory", patched_prepare_corpus)
    tool_registry.register_tool("build_er_graph", build_er_graph_wrapper)
    tool_registry.register_tool("entity_vdb_search_tool", patched_entity_vdb_search)
    tool_registry.register_tool("relationship_one_hop_neighbors_tool", patched_relationship_one_hop_neighbors)
    
    # Create the agent orchestrator with shared context
    agent_orchestrator = AgentOrchestrator(
        llm_provider=llm_instance,
        tool_registry=tool_registry,
        graphrag_context=global_graphrag_context
    )
    
    # Create a task description
    SYSTEM_TASK = f"""
    You are an AI assistant that can help users explore and query knowledge graphs. Please perform the following tasks in order:

    1. Prepare a corpus from the directory: '{corpus_directory}'
       - Use the 'prepare_corpus_from_directory' tool 
       - Save the corpus to: '{corpus_output_dir_path}'
       - Use corpus_id: 'test_corpus'
       - Include all file types

    2. Build an Entity-Relation graph from the corpus:
       - Use the 'build_er_graph' tool
       - Use corpus_path: '{corpus_path}'
       - Use graph_id: 'test_graph'
       - The graph will be saved automatically

    3. Search for entities in the entity vector database:
       - Use the 'entity_vdb_search_tool'
       - Search for the query: "The main character"
       - Use top_k: 3
       - Use vdb_reference_id: "entities_vdb"

    4. Find one-hop neighbors for an entity:
       - Use the 'relationship_one_hop_neighbors_tool'
       - Use entity_ids from the previous step
       - Set include_attributes: true
       - Use graph_reference_id: "test_graph"

    """
    
    # Execute the task with the agent
    logger.info(f"Starting task execution with GLOBAL context ID: {global_graphrag_context.request_id}")
    
    try:
        # Have the agent plan and execute the task
        plan = await agent.aplan(SYSTEM_TASK)
        
        # Trace plan
        logger.info(f"Generated plan with {len(plan.steps)} steps")
        
        for idx, step in enumerate(plan.steps):
            logger.info(f"Step {idx+1}: {step.action}")
        
        # Add patch to execute plan to verify context is maintained
        original_execute_plan = agent_orchestrator._execute_plan
        
        def patched_execute_plan(plan: Plan, **kwargs):
            logger.info(f"Executing plan with context ID: {global_graphrag_context.request_id}")
            logger.info(f"Plan has {len(plan.steps)} steps")
            return original_execute_plan(plan, **kwargs)
        
        agent_orchestrator._execute_plan = patched_execute_plan
        
        # Execute the plan
        result = await agent_orchestrator.aexecute_plan(plan)
        logger.info(f"Plan execution result: {result.status}")
        
        # --- Check for artifacts and log results ---
        corpus_json_path = Path(corpus_output_dir_path) / "Corpus.json"
        ergraph_path = Path(corpus_output_dir_path) / "er_graph" / "nx_data.graphml"
        
        if corpus_json_path.exists():
            logger.info(f"✅ Corpus JSON file exists at {corpus_json_path}")
        else:
            logger.error(f"❌ Corpus JSON file does NOT exist at {corpus_json_path}")
        
        if ergraph_path.exists():
            logger.info(f"✅ ER Graph file exists at {ergraph_path}")
        else:
            logger.error(f"❌ ER Graph file does NOT exist at {ergraph_path}")
        
        # Log any search results
        for step in result.step_outputs:
            if step.action == "entity_vdb_search_tool":
                logger.info(f"Entity VDB search results: {step.result}")
            
            if step.action == "relationship_one_hop_neighbors_tool":
                logger.info(f"One-hop neighbor results: {step.result}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error in pipeline test: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    asyncio.run(test_agent_corpus_to_graph_pipeline())
