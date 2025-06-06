import asyncio
import sys
from loguru import logger
from pathlib import Path
from Option.Config2 import Config
from Core.AgentSchema.context import GraphRAGContext
from Core.Provider.LiteLLMProvider import LiteLLMProvider
from Core.Index.EmbeddingFactory import get_rag_embedding
from Core.AgentOrchestrator.orchestrator import AgentOrchestrator
from Core.Graph.GraphFactory import get_graph
from Core.AgentTools.entity_vdb_tools import entity_vdb_build_tool
from Core.AgentSchema.tool_contracts import EntityVDBBuildInputs
from Core.Storage.NetworkXStorage import NetworkXStorage

# Configure logger
logger.remove()
logger.add(sys.stderr, level="DEBUG")

async def main():
    # Load config
    config = Config.from_yaml_file("Option/Config2.yaml")
    
    # Create providers
    llm_provider = LiteLLMProvider(config.llm)
    embedding_provider = get_rag_embedding(config=config)
    
    # Create context
    context = GraphRAGContext(
        target_dataset_name="Physics_Small",
        main_config=config,
        llm_provider=llm_provider,
        embedding_provider=embedding_provider
    )
    
    # Get or create graph
    graph_id = "Physics_Small_ERGraph"
    graph_type = "er_graph"
    
    # Check if we need to create the graph instance
    existing_graph = context.get_graph_instance(graph_id)
    if existing_graph:
        logger.info(f"Graph {graph_id} already exists in context")
        graph_instance = existing_graph
    else:
        logger.info(f"Creating graph {graph_id}")
        # Try to load from storage
        graph_path = Path(f"./results/{context.target_dataset_name}_ER/er_graph/graph_chunk_entity_relation.json")
        
        if graph_path.exists():
            logger.info(f"Loading graph from {graph_path}")
            # Need to use storage loading here
            storage = NetworkXStorage(namespace="entity_graph")
            await storage.load_graph(str(graph_path.parent))
            
            # Now create graph with this storage
            graph_instance = get_graph(
                config=config,
                llm=llm_provider,
                encoder=embedding_provider,
                storage_instance=storage
            )
        else:
            logger.info(f"Graph file not found at {graph_path}")
            # Create empty graph
            graph_instance = get_graph(
                config=config,
                llm=llm_provider,
                encoder=embedding_provider
            )
        context.add_graph_instance(graph_id, graph_instance)
    
    # Get nodes data directly
    logger.info("Getting nodes data directly from graph...")
    nodes = await graph_instance.nodes_data()
    logger.info(f"Retrieved {len(nodes)} nodes")
    
    # Print first few nodes
    for i, node in enumerate(nodes[:5]):
        logger.info(f"Node {i}: {node}")
    
    # Now try VDB build
    logger.info("\nTrying VDB build tool...")
    params = EntityVDBBuildInputs(
        graph_reference_id=graph_id,
        vdb_collection_name="test_entities_vdb",
        force_rebuild=True
    )
    
    result = await entity_vdb_build_tool(params, context)
    logger.info(f"VDB Build Result: {result}")

if __name__ == "__main__":
    asyncio.run(main())
