#!/usr/bin/env python3
"""Test entity extraction and VDB operations"""

import asyncio
import sys
sys.path.append('.')

from Core.AgentBrain.agent_brain import PlanningAgent
from Core.AgentSchema.context import GraphRAGContext
from Option.Config2 import Config
from Core.Provider.LLMProviderRegister import create_llm_instance
from Core.Index.EmbeddingFactory import RAGEmbeddingFactory
from Core.Chunk.ChunkFactory import ChunkFactory

async def test_entity_operations():
    config = Config.default()
    llm = create_llm_instance(config.llm)
    emb_factory = RAGEmbeddingFactory()
    encoder = emb_factory.get_rag_embedding(config=config)
    chunk_factory = ChunkFactory(config)
    
    context = GraphRAGContext(
        main_config=config,
        target_dataset_name="Synthetic_Test",
        llm_provider=llm,
        embedding_provider=encoder,
        chunk_storage_manager=chunk_factory
    )
    
    agent = PlanningAgent(config=config, graphrag_context=context)
    
    print("Testing Entity Extraction and VDB Operations")
    print("=" * 60)
    
    # Test query that should:
    # 1. Build graph if needed
    # 2. Build entity VDB
    # 3. Search for entities
    # 4. Get text for entities
    query = """
    First prepare corpus from Data/Synthetic_Test if needed.
    Then build an ER graph.
    Then build an entity vector database.
    Then search for entities related to 'artificial intelligence'.
    Finally, get the text chunks for those entities to show their context.
    """
    
    result = await agent.process_query(query, "Synthetic_Test")
    
    answer = result.get("generated_answer", "")
    context_data = result.get("retrieved_context", {})
    
    print("\nGenerated Answer:")
    print("-" * 40)
    print(answer[:500] + "..." if len(answer) > 500 else answer)
    
    print("\n\nContext Steps Executed:")
    print("-" * 40)
    for step_id, outputs in context_data.items():
        if isinstance(outputs, dict):
            print(f"\n{step_id}:")
            for key, value in outputs.items():
                if isinstance(value, list):
                    print(f"  {key}: {len(value)} items")
                    if value and len(value) > 0:
                        print(f"    Sample: {str(value[0])[:100]}...")
                elif isinstance(value, str):
                    print(f"  {key}: {value[:100]}...")
                else:
                    print(f"  {key}: {type(value)}")
    
    # Check for success indicators
    success_indicators = []
    
    # Check if corpus was prepared
    if any("corpus" in str(step).lower() for step in context_data.keys()):
        success_indicators.append("✓ Corpus prepared")
    
    # Check if graph was built
    if any("graph" in str(step).lower() and "er_graph_id" in str(outputs) for step, outputs in context_data.items()):
        success_indicators.append("✓ ER graph built")
    
    # Check if VDB was built
    if any("vdb" in str(step).lower() and "build" in str(step).lower() for step in context_data.keys()):
        success_indicators.append("✓ Entity VDB built")
    
    # Check if entities were found
    entities_found = False
    for step, outputs in context_data.items():
        if isinstance(outputs, dict) and "search_results" in outputs:
            entities_found = True
            break
    if entities_found:
        success_indicators.append("✓ Entities found via search")
    
    # Check if text was retrieved
    text_retrieved = False
    for step, outputs in context_data.items():
        if isinstance(outputs, dict) and ("retrieved_chunks" in outputs or "text_chunks" in outputs):
            text_retrieved = True
            break
    if text_retrieved:
        success_indicators.append("✓ Text chunks retrieved")
    
    print("\n\nTest Results:")
    print("-" * 40)
    for indicator in success_indicators:
        print(indicator)
    
    if len(success_indicators) < 5:
        print(f"\n⚠️  Only {len(success_indicators)}/5 operations succeeded")
    else:
        print("\n🎉 All entity operations working!")
    
    return len(success_indicators) >= 4  # At least 4/5 should work

if __name__ == "__main__":
    success = asyncio.run(test_entity_operations())
    exit(0 if success else 1)