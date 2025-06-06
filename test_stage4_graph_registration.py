#!/usr/bin/env python3
"""
Stage 4: Graph Registration & Context Management
Goal: Ensure built graphs are accessible to subsequent tools
"""
import asyncio
import json
from pathlib import Path
from Core.AgentSchema.context import GraphRAGContext
from Core.AgentTools.graph_construction_tools import BuildERGraphTool
from Core.AgentTools.entity_vdb_tools import EntityVDBBuildTool
from Option.Config2 import Config
from Core.Common.Logger import logger

async def test_graph_registration():
    """Test graph registration and context management"""
    print("\n" + "="*80)
    print("STAGE 4: Graph Registration & Context Management")
    print("="*80)
    
    # Setup
    config = Config.default()
    
    # Use existing Social_Discourse_Test dataset
    dataset = "Social_Discourse_Test"
    
    # Create GraphRAGContext
    context = GraphRAGContext(
        working_directory="./results",
        user_query="Test graph registration",
        target_dataset_name=dataset
    )
    
    print(f"\nTest dataset: {dataset}")
    print(f"Initial graphs in context: {list(context.graph_registry.keys())}")
    
    # Step 1: Build ER Graph
    print("\n1. Building ER Graph...")
    graph_tool = BuildERGraphTool()
    
    try:
        graph_result = await graph_tool.run(
            target_dataset_name=dataset,
            graphrag_context=context
        )
        
        print(f"Graph build status: {graph_result.status}")
        print(f"Graph build message: {graph_result.message}")
        
        graph_id = graph_result.graph_id if hasattr(graph_result, 'graph_id') else None
        print(f"Graph ID returned: {graph_id}")
        
        # Check if graph is in context
        print(f"\nGraphs in context after build: {list(context.graph_registry.keys())}")
        graph_in_context = graph_id in context.graph_registry if graph_id else False
        
        if graph_in_context:
            print(f"✓ Graph '{graph_id}' found in context registry")
            graph_instance = context.graph_registry[graph_id]
            nodes = await graph_instance.get_nodes()
            edges = await graph_instance.get_edges()
            print(f"  Nodes: {len(nodes)}")
            print(f"  Edges: {len(edges)}")
        else:
            print(f"❌ Graph not found in context registry")
            
    except Exception as e:
        print(f"❌ Graph build error: {e}")
        import traceback
        traceback.print_exc()
        graph_id = None
        graph_in_context = False
    
    # Step 2: Build VDB using the graph
    print("\n2. Building Entity VDB...")
    vdb_success = False
    entities_indexed = 0
    
    if graph_id and graph_in_context:
        vdb_tool = EntityVDBBuildTool()
        
        try:
            vdb_result = await vdb_tool.run(
                graph_reference_id=graph_id,
                vdb_collection_name=f"{dataset}_entities",
                graphrag_context=context
            )
            
            print(f"VDB build status: {vdb_result.status}")
            print(f"VDB build message: {vdb_result.message}")
            
            if vdb_result.status == "success":
                vdb_success = True
                entities_indexed = vdb_result.entity_count if hasattr(vdb_result, 'entity_count') else 0
                print(f"✓ VDB built successfully")
                print(f"  Entities indexed: {entities_indexed}")
                
                # Check if VDB is in context
                vdb_id = vdb_result.vdb_reference_id if hasattr(vdb_result, 'vdb_reference_id') else None
                if vdb_id and vdb_id in context.vdb_registry:
                    print(f"✓ VDB '{vdb_id}' found in context registry")
                else:
                    print(f"❌ VDB not found in context registry")
                    
        except Exception as e:
            print(f"❌ VDB build error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("❌ Cannot build VDB - no graph available in context")
    
    # Step 3: Test if another tool can access the graph
    print("\n3. Testing graph access from another tool...")
    graph_accessible = False
    
    if graph_id:
        # Simulate another tool trying to access the graph
        try:
            if graph_id in context.graph_registry:
                test_graph = context.graph_registry[graph_id]
                test_nodes = await test_graph.get_nodes()
                if len(test_nodes) > 0:
                    graph_accessible = True
                    print(f"✓ Graph accessible from context")
                    print(f"  Retrieved {len(test_nodes)} nodes")
        except Exception as e:
            print(f"❌ Error accessing graph: {e}")
    
    # Final verdict
    print("\n" + "="*80)
    print("EVIDENCE SUMMARY:")
    print(f"- graph_built: {graph_id or 'FAILED'}")
    print(f"- graphs_in_context: {list(context.graph_registry.keys())}")
    print(f"- vdb_built_from_graph: {'success' if vdb_success else 'failed'}")
    print(f"- entities_indexed: {entities_indexed}")
    print(f"- graph_accessible: {graph_accessible}")
    
    if graph_in_context and vdb_success and graph_accessible:
        print("\n✅ STAGE 4: PASSED - Graph registration and context management working!")
        return True
    else:
        print("\n❌ STAGE 4: FAILED - Graph registration issues detected")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_graph_registration())