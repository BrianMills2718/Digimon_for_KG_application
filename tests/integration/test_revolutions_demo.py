#!/usr/bin/env python3
"""
Custom demo script for testing DIGIMON GraphRAG on revolution documents.
Tests the full pipeline: corpus prep → graph building → VDB → semantic search
"""

import sys
import os
import asyncio
import json
from typing import Dict, Any

# Add project root to path
sys.path.append('/home/brian/digimon')

# Import core modules using working paths from demo
from Option.Config2 import Config
from Core.Storage.BaseGraphStorage import BaseGraphStorage
from Core.Storage.NetworkXStorage import NetworkXStorage
from Core.Storage.ChunkKVStorage import ChunkKVStorage
from Core.Index.IndexFactory import RAGIndexFactory
from Core.AgentSchema.context import GraphRAGContext
from Core.Retriever.BaseRetriever import BaseRetriever

# Import orchestrator and planning
from Core.AgentOrchestrator.orchestrator import AgentOrchestrator
from Core.AgentSchema.plan import ExecutionPlan, ExecutionStep, ToolCall, ToolInputSource, DynamicToolChainConfig
from Core.Provider.LiteLLMProvider import LiteLLMProvider
from Core.Provider.BaseEmb import BaseEmb
from Core.Chunk.ChunkFactory import ChunkFactory
from Core.Common.Logger import logger
from Core.Index.EmbeddingFactory import get_rag_embedding
import uuid

async def main():
    print("="*80)
    print("🏛️  REVOLUTION DOCUMENTS - DIGIMON GraphRAG ANALYSIS")
    print("="*80)
    print("📁 Source: /home/brian/digimon/Data/Revolutions_Small/")
    print("📄 Files: american_revolution.txt, french_revolution.txt")
    print("-"*80)
    
    # Initialize system using same pattern as working demo
    main_config = Config.default()
    
    # Create required instances
    llm_instance = LiteLLMProvider(main_config.llm)
    encoder_instance = get_rag_embedding(config=main_config)
    chunk_factory = ChunkFactory(main_config)
    
    # Update GraphRAGContext with providers
    graphrag_context = GraphRAGContext(
        target_dataset_name="revolutions_demo",
        main_config=main_config,
        llm_provider=llm_instance,
        embedding_provider=encoder_instance,
        chunk_storage_manager=chunk_factory
    )
    
    # Initialize orchestrator  
    orchestrator = AgentOrchestrator(
        main_config=main_config,
        llm_instance=llm_instance,
        encoder_instance=encoder_instance,
        chunk_factory=chunk_factory,
        graphrag_context=graphrag_context
    )
    
    # Define pipeline plan with your revolution documents
    pipeline_plan = ExecutionPlan(
        plan_id="revolutions_analysis",
        plan_description="Analyze American and French Revolution documents",
        target_dataset_name="revolutions_demo",
        steps=[
            # Step 1: Prepare corpus from your revolution documents
            ExecutionStep(
                step_id="prepare_corpus",
                description="Process revolution documents into corpus",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="corpus.PrepareFromDirectory",
                            inputs={
                                "input_directory_path": "/home/brian/digimon/Data/Revolutions_Small",
                                "output_directory_path": "./results/revolutions_demo",
                                "target_corpus_name": "revolutions_corpus"
                            },
                            named_outputs={
                                "corpus_path": "corpus_json_path",
                                "doc_count": "document_count"
                            }
                        )
                    ]
                )
            ),
            
            # Step 2: Build ER Graph
            ExecutionStep(
                step_id="build_graph",
                description="Build entity-relationship graph",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="graph.BuildERGraph",
                            inputs={
                                "target_dataset_name": "revolutions_corpus",
                                "force_rebuild": True
                            },
                            named_outputs={
                                "graph_id": "graph_id"
                            }
                        )
                    ]
                )
            ),
            
            # Step 3: Build Entity VDB
            ExecutionStep(
                step_id="build_entity_vdb",
                description="Build entity vector database",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="Entity.VDB.Build",
                            inputs={
                                "graph_reference_id": ToolInputSource(
                                    from_step_id="build_graph",
                                    named_output_key="graph_id"
                                ),
                                "vdb_collection_name": "revolution_entities",
                                "force_rebuild": True
                            },
                            named_outputs={
                                "vdb_reference_id": "vdb_reference_id"
                            }
                        )
                    ]
                )
            ),
            
            # Step 4: Test semantic search
            ExecutionStep(
                step_id="test_search",
                description="Test entity search",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="Entity.VDBSearch",
                            inputs={
                                "query_text": "King Louis XVI taxation revolution",
                                "vdb_reference_id": ToolInputSource(
                                    from_step_id="build_entity_vdb",
                                    named_output_key="vdb_reference_id"
                                ),
                                "top_k_results": 5
                            }
                        )
                    ]
                )
            )
        ]
    )
    
    print("🔄 Processing your revolution documents...")
    try:
        results = await orchestrator.execute_plan(pipeline_plan)
        
        print("\n✨ RESULTS")
        print("="*60)
        
        for step_id, outputs in results.items():
            print(f"\n📌 {step_id.upper()}:")
            if isinstance(outputs, dict):
                for key, value in outputs.items():
                    if isinstance(value, (list, dict)) and len(str(value)) > 100:
                        print(f"  ✓ {key}: {type(value).__name__} with {len(value)} items")
                    else:
                        print(f"  ✓ {key}: {value}")
        
        print("\n🎯 SUCCESS! Your revolution documents are now analyzed!")
        print("✅ Knowledge graph built from your documents")
        print("✅ Entities extracted and made searchable")  
        print("✅ Ready for semantic queries and exploration")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
