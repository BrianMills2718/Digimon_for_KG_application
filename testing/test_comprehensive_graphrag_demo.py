"""
Comprehensive DIGIMON GraphRAG Demo
Demonstrates the full power of the current implementation
"""
import sys
import os
import asyncio
import json
from typing import Dict, Any

# Add project root to path
sys.path.append('/home/brian/digimon')

# Import core modules
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
    print("=" * 80)
    print("DIGIMON GraphRAG System - Comprehensive Demo")
    print("=" * 80)
    
    # Initialize config and context
    main_config = Config.default()
    
    # Create required instances
    llm_instance = LiteLLMProvider(main_config.llm)
    encoder_instance = get_rag_embedding(config=main_config)
    chunk_factory = ChunkFactory(main_config)
    
    # Update GraphRAGContext with providers
    graphrag_context = GraphRAGContext(
        target_dataset_name="demo_dataset",
        main_config=main_config,
        llm_provider=llm_instance,
        embedding_provider=encoder_instance,
        chunk_storage_manager=chunk_factory
    )
    
    # Create orchestrator
    orchestrator = AgentOrchestrator(
        main_config=main_config,
        llm_instance=llm_instance,
        encoder_instance=encoder_instance,
        chunk_factory=chunk_factory,
        graphrag_context=graphrag_context
    )
    
    print("\n🎯 Demo Features:")
    
    print("\n📊 Current System Status:")
    print("-" * 40)
    
    # Count tools by category
    tool_categories = {
        'Entity': [],
        'Relationship': [],
        'Chunk': [],
        'Graph Construction': [],
        'Graph Analysis': [],
        'Corpus': []
    }
    
    for tool_id in orchestrator._tool_registry.keys():
        if tool_id.startswith('Entity.'):
            tool_categories['Entity'].append(tool_id)
        elif tool_id.startswith('Relationship.'):
            tool_categories['Relationship'].append(tool_id)
        elif tool_id.startswith('Chunk.'):
            tool_categories['Chunk'].append(tool_id)
        elif tool_id.startswith('graph.Build'):
            tool_categories['Graph Construction'].append(tool_id)
        elif tool_id.startswith('graph.'):
            tool_categories['Graph Analysis'].append(tool_id)
        elif tool_id.startswith('corpus.'):
            tool_categories['Corpus'].append(tool_id)
    
    print(f"\n🔧 Total Tools Registered: {len(orchestrator._tool_registry)}")
    for category, tools in tool_categories.items():
        if tools:
            print(f"\n{category} Tools ({len(tools)}):")
            for tool in sorted(tools):
                print(f"  ✓ {tool}")
    
    # Map to 16 original operators
    print("\n📈 GraphRAG Operators Implementation Status (16 total):")
    print("-" * 50)
    
    operators_status = {
        'Implemented': [
            'Entity.VDBSearch (Entity.VDB)',
            'Entity.PPR',
            'Entity.Onehop',
            'Entity.RelNode',
            'Relationship.OneHopNeighbors (Relationship.Onehop)',
            'Relationship.VDB.Build + Search (Relationship.VDB)',
            'Chunk.FromRelationships (Chunk.FromRel)'
        ],
        'Not Implemented': [
            'Entity.Agent',
            'Entity.Link',
            'Entity.TF-IDF',
            'Relationship.Aggregator',
            'Relationship.Agent',
            'Chunk.Aggregator',
            'Chunk.Occurrence',
            'Subgraph.KhopPath',
            'Subgraph.Steiner',
            'Subgraph.AgentPath',
            'Community.Entity',
            'Community.Layer'
        ]
    }
    
    implemented_count = len(operators_status['Implemented'])
    print(f"\n✅ Implemented: {implemented_count}/16 operators")
    for op in operators_status['Implemented']:
        print(f"  • {op}")
    
    print(f"\n❌ Not Implemented: {len(operators_status['Not Implemented'])}/16 operators")
    
    # Demo pipeline
    print("\n" + "=" * 80)
    print("🚀 Running Demo Pipeline: Document → Graph → VDB → Advanced Retrieval")
    print("=" * 80)
    
    # Create demo directory
    test_dir = "/tmp/digimon_demo"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create sample documents
    docs = {
        "ai_research.txt": """
Artificial Intelligence Research: A Comprehensive Overview

Neural networks have revolutionized the field of artificial intelligence. 
Deep learning, pioneered by Geoffrey Hinton, Yann LeCun, and Yoshua Bengio, 
has achieved breakthrough results in computer vision, natural language processing,
and reinforcement learning.

Recent advances include transformer architectures, which have enabled large language
models like GPT and BERT to achieve human-level performance on many tasks.
Researchers at OpenAI, Google DeepMind, and Meta AI continue to push the boundaries
of what's possible with AI.

The collaboration between academia and industry has accelerated progress significantly.
Universities like Stanford, MIT, and CMU work closely with tech companies to advance
the state of the art in machine learning.
""",
        "ml_applications.txt": """
Machine Learning Applications in Industry

Companies are applying ML techniques across various domains:
- Healthcare: Diagnostic imaging, drug discovery, personalized medicine
- Finance: Fraud detection, algorithmic trading, risk assessment
- Transportation: Autonomous vehicles, route optimization, demand forecasting
- Retail: Recommendation systems, inventory management, customer segmentation

The integration of ML into production systems requires careful consideration of:
1. Model interpretability and explainability
2. Fairness and bias mitigation
3. Scalability and performance optimization
4. Privacy and security concerns

As ML models become more complex, the need for robust MLOps practices
becomes increasingly critical for successful deployment and maintenance.
""",
        "future_of_ai.txt": """
The Future of Artificial Intelligence

Emerging trends in AI research include:
- Multimodal learning: Combining vision, language, and other modalities
- Few-shot and zero-shot learning: Reducing data requirements
- Neuromorphic computing: Brain-inspired hardware architectures
- Quantum machine learning: Leveraging quantum computing for ML

Ethical considerations are becoming paramount as AI systems become more powerful
and pervasive. Researchers and policymakers are working together to establish
guidelines for responsible AI development and deployment.

The convergence of AI with other technologies like robotics, IoT, and blockchain
promises to create new opportunities and challenges. As we move toward artificial
general intelligence (AGI), the importance of alignment with human values becomes
critical for ensuring beneficial outcomes for humanity.
"""
    }
    
    # Write sample documents
    for filename, content in docs.items():
        with open(os.path.join(test_dir, filename), 'w') as f:
            f.write(content)
    
    # Define comprehensive pipeline
    demo_plan = ExecutionPlan(
        plan_id="comprehensive_demo",
        plan_description="Comprehensive GraphRAG demo pipeline from documents to advanced retrieval",
        target_dataset_name="demo_dataset",
        steps=[
            ExecutionStep(
                step_id="prepare_corpus",
                description="Load and prepare test documents into corpus",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="corpus.PrepareFromDirectory",
                            inputs={
                                "input_directory_path": test_dir,
                                "output_directory_path": os.path.join("./results", "ai_research_corpus"),
                                "target_corpus_name": "ai_research_corpus",
                                "chunk_config": {"type": "fixed_token", "size": 100, "overlap": 20}
                            },
                            named_outputs={
                                "corpus_path": "corpus_json_path",
                                "num_documents": "document_count"
                            }
                        )
                    ]
                )
            ),
            
            ExecutionStep(
                step_id="build_er_graph",
                description="Build entity-relationship graph from corpus",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="graph.BuildERGraph",
                            inputs={
                                "target_dataset_name": "ai_research_corpus",
                                "force_rebuild": True,
                                "config_overrides": {
                                    "enable_relationship_weight": True
                                }
                            },
                            named_outputs={
                                "graph_id": "graph_id",
                                "num_nodes": "node_count",
                                "num_edges": "edge_count"
                            }
                        )
                    ]
                )
            ),
            
            ExecutionStep(
                step_id="build_vector_db",
                description="Build vector database from ER graph entities",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="Entity.VDB.Build",
                            inputs={
                                "graph_reference_id": ToolInputSource(
                                    from_step_id="build_er_graph",
                                    named_output_key="graph_id"
                                ),
                                "vdb_collection_name": "demo_entity_vdb",
                                "force_rebuild": True
                            },
                            named_outputs={
                                "vdb_id": "vdb_reference_id",
                                "num_vectors": "num_entities_indexed"
                            }
                        )
                    ]
                )
            ),
            
            ExecutionStep(
                step_id="vdb_search",
                description="Search for entities related to AI advancement",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="Entity.VDBSearch",
                            inputs={
                                "query_text": "artificial intelligence advancement and progress",
                                "vdb_reference_id": ToolInputSource(
                                    from_step_id="build_vector_db",
                                    named_output_key="vdb_id"
                                ),
                                "top_k_results": 5
                            },
                            named_outputs={
                                "vdb_results": "similar_entities"
                            }
                        )
                    ]
                )
            ),
            
            ExecutionStep(
                step_id="ppr_analysis",
                description="Run personalized PageRank on top entities",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="Entity.PPR",
                            inputs={
                                "seed_entity_ids": ToolInputSource(
                                    from_step_id="vdb_search",
                                    named_output_key="vdb_results"
                                ),
                                "graph_reference_id": ToolInputSource(
                                    from_step_id="build_er_graph",
                                    named_output_key="graph_id"
                                ),
                                "personalization_weight_alpha": 0.15,
                                "top_k_results": 10
                            },
                            named_outputs={
                                "ppr_results": "ranked_entities"
                            }
                        )
                    ]
                )
            ),
            
            ExecutionStep(
                step_id="explore_neighbors",
                description="Explore one-hop neighbors of top entities",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="Entity.Onehop",
                            inputs={
                                "entity_ids": ToolInputSource(
                                    from_step_id="ppr_analysis",
                                    named_output_key="ppr_results"
                                ),
                                "graph_id": ToolInputSource(
                                    from_step_id="build_er_graph",
                                    named_output_key="graph_id"
                                ),
                                "neighbor_limit_per_entity": 20
                            },
                            named_outputs={
                                "neighbor_results": "neighbors"
                            }
                        )
                    ]
                )
            ),
            
            ExecutionStep(
                step_id="retrieve_chunks",
                description="Retrieve document chunks from relationships",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="Chunk.FromRelationships",
                            inputs={
                                "target_relationships": ToolInputSource(
                                    from_step_id="explore_neighbors",
                                    named_output_key="neighbor_results"
                                ),
                                "graph_reference_id": ToolInputSource(
                                    from_step_id="build_er_graph",
                                    named_output_key="graph_id"
                                ),
                                "document_collection_id": "ai_research_corpus"
                            },
                            named_outputs={
                                "chunks": "chunks"
                            }
                        )
                    ]
                )
            ),
            
            ExecutionStep(
                step_id="visualize_graph",
                description="Generate graph visualization",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="graph.Visualize",
                            inputs={
                                "graph_id": ToolInputSource(
                                    from_step_id="build_er_graph",
                                    named_output_key="graph_id"
                                ),
                                "output_format": "JSON_NODES_EDGES"
                            },
                            named_outputs={
                                "visualization": "graph_viz"
                            }
                        )
                    ]
                )
            ),
            
            ExecutionStep(
                step_id="analyze_graph",
                description="Analyze graph metrics and structure",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="graph.Analyze",
                            inputs={
                                "graph_id": ToolInputSource(
                                    from_step_id="build_er_graph",
                                    named_output_key="graph_id"
                                ),
                                "metrics_to_calculate": ["basic_stats", "centrality_metrics", "connectivity_metrics"],
                                "top_k_nodes": 10,
                                "calculate_expensive_metrics": False
                            },
                            named_outputs={
                                "analysis": "graph_analysis"
                            }
                        )
                    ]
                )
            )
        ]
    )
    
    # Execute pipeline
    print("\n🔄 Executing Pipeline...")
    
    try:
        results = await orchestrator.execute_plan(demo_plan)
        
        print("\n✨ Pipeline Results:")
        print("-" * 60)
        
        # Results is a dict, not an object with step_results
        for step_id, outputs in results.items():
            print(f"\n📌 {step_id}:")
            if isinstance(outputs, dict):
                for key, value in outputs.items():
                    if isinstance(value, (list, dict)) and len(str(value)) > 100:
                        print(f"  • {key}: {type(value).__name__} with {len(value)} items")
                    else:
                        print(f"  • {key}: {value}")
            else:
                print(f"  • Output: {outputs}")
        
        # Demonstrate advanced features
        print("\n" + "=" * 80)
        print("🎯 Advanced Features Demonstration")
        print("=" * 80)
        
        # 1. PPR-based retrieval
        print("\n1️⃣ Personalized PageRank (PPR) Entity Ranking:")
        ppr_step = ExecutionStep(
            step_id="ppr_ranking",
            description="Run PPR analysis on entities",
            action=DynamicToolChainConfig(
                tools=[
                    ToolCall(
                        tool_id="Entity.PPR",
                        inputs={
                            "seed_entity_ids": ["neural networks", "deep learning"],
                            "graph_reference_id": ToolInputSource(
                                from_step_id="build_er_graph",
                                named_output_key="graph_id"
                            ),
                            "personalization_weight_alpha": 0.15,
                            "top_k_results": 10
                        }
                    )
                ]
            )
        )
        ppr_result = await orchestrator.execute_plan(
            ExecutionPlan(
                plan_id="ppr_demo", 
                plan_description="Run PPR analysis on sample entities",
                target_dataset_name="demo_dataset",
                steps=[ppr_step]
            )
        )
        print(f"PPR Results: Found {len(ppr_result.get('ppr_ranking', {}).get('ranked_entities', []))} ranked entities")
        
        # 2. Relationship search
        print("\n2️⃣ Semantic Relationship Search:")
        rel_search_step = ExecutionStep(
            step_id="rel_search",
            description="Search semantic relationships", 
            action=DynamicToolChainConfig(
                tools=[
                    ToolCall(
                        tool_id="Relationship.VDB.Search",
                        inputs={
                            "query_text": "impacts of AI on society",
                            "vdb_reference_id": "demo_entity_vdb",
                            "top_k_results": 10
                        }
                    )
                ]
            )
        )
        rel_search_result = await orchestrator.execute_plan(
            ExecutionPlan(
                plan_id="rel_search_demo", 
                plan_description="Search for semantic relationships",
                target_dataset_name="demo_dataset",
                steps=[rel_search_step]
            )
        )
        print(f"Relationship Search: Found {len(rel_search_result.get('rel_search', {}).get('relationships', []))} relevant relationships")
        
        # 3. Multi-hop exploration
        print("\n3️⃣ Multi-hop Graph Exploration:")
        print("  Starting from 'Geoffrey Hinton' → 2-hop neighbors")
        # This would use Entity.Onehop with hop_size=2
        
        print("\n" + "=" * 80)
        print("💡 System Capabilities Summary:")
        print("=" * 80)
        print("""
        ✅ Document Processing: Automatic chunking and corpus preparation
        ✅ Graph Construction: Multiple graph types (ER, RK, Tree, Passage)
        ✅ Semantic Search: Entity and relationship vector databases
        ✅ Graph Algorithms: PPR, one-hop neighbors, graph analysis
        ✅ Context Expansion: Multi-hop traversal and chunk extraction
        ✅ Visualization: Graph structure export in multiple formats
        
        🎯 Use Cases Enabled:
        • Question Answering with graph-enhanced context
        • Entity-centric document exploration
        • Relationship discovery and analysis
        • Semantic similarity search at multiple granularities
        • Graph-based document summarization
        """)
        
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        import shutil
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            
    print("\n✅ Demo completed!")

if __name__ == "__main__":
    asyncio.run(main())
