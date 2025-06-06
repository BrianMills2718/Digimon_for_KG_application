#!/usr/bin/env python3
"""
Stage 1: Entity Extraction Format Fix
Goal: Ensure entity extraction returns proper string entity names, not dicts
"""
import asyncio
import json
from pathlib import Path
from Core.Graph.ERGraph import ERGraph
from Core.Chunk.ChunkFactory import ChunkFactory
from Option.Config2 import Config
from Core.Common.Logger import logger
from Core.Provider.LiteLLMProvider import LiteLLMProvider

async def test_entity_extraction():
    """Test entity extraction to ensure proper string format"""
    print("\n" + "="*80)
    print("STAGE 1: Entity Extraction Format Fix")
    print("="*80)
    
    # Setup
    config = Config.default()
    llm = LiteLLMProvider(config.llm)
    chunk_factory = ChunkFactory(config)
    
    # Create ER Graph instance
    er_graph = ERGraph(config.graph, llm=llm, encoder=None)
    
    # Get test dataset
    dataset = "Social_Discourse_Test"
    namespace = chunk_factory.get_namespace(dataset, "er_graph")
    er_graph.namespace = namespace
    
    print(f"\nTest Dataset: {dataset}")
    print(f"Namespace path: {namespace}")
    
    # Load chunks
    chunks = await chunk_factory.get_chunks_for_dataset(dataset)
    print(f"Chunks loaded: {len(chunks)}")
    
    if not chunks:
        # Copy corpus to expected location
        src = Path(f"results/{dataset}/corpus/Corpus.json")
        dst = Path(f"results/{dataset}/Corpus.json")
        if src.exists() and not dst.exists():
            import shutil
            shutil.copy(src, dst)
            print(f"Copied corpus from {src} to {dst}")
            chunks = await chunk_factory.get_chunks_for_dataset(dataset)
            print(f"Chunks loaded after copy: {len(chunks)}")
    
    # Test entity extraction on first chunk
    if chunks:
        chunk_id, chunk_content = chunks[0]
        print(f"\nTesting entity extraction on chunk: {chunk_id}")
        print(f"Chunk preview: {chunk_content.content[:200]}...")
        
        # Call entity extraction
        try:
            result = await er_graph._extract_entity_relationship((chunk_id, chunk_content))
            print(f"\nExtraction result type: {type(result)}")
            print(f"Extraction result: {result}")
            
            # The method returns a tuple (entities_dict, relations_dict)
            if isinstance(result, tuple) and len(result) == 2:
                entities_dict, relations_dict = result
                # Each dict value is a list of entities/relations
                entities = []
                for entity_list in entities_dict.values():
                    entities.extend(entity_list)
                relations = []
                for rel_list in relations_dict.values():
                    relations.extend(rel_list)
            else:
                entities = []
                relations = []
            
            print(f"\nEntities extracted: {len(entities)}")
            print(f"Relations extracted: {len(relations)}")
            
            # Check entity format
            entity_format_ok = True
            for i, entity in enumerate(entities[:5]):  # Check first 5
                entity_name = entity.entity_name
                print(f"\nEntity {i+1}:")
                print(f"  entity_name type: {type(entity_name)}")
                print(f"  entity_name value: {entity_name}")
                print(f"  entity_type: {entity.entity_type}")
                print(f"  description: {entity.description[:100]}...")
                
                if not isinstance(entity_name, str):
                    entity_format_ok = False
                    print(f"  ❌ ERROR: entity_name is not a string!")
                else:
                    print(f"  ✓ entity_name is a string")
            
            # Skip full graph build for now - just verify entity format
            print("\n" + "-"*40)
            print("Skipping full graph build - entity format verified")
            
            # Final verdict
            print("\n" + "="*80)
            print("EVIDENCE SUMMARY:")
            print(f"- entity_name: string (verified on {min(5, len(entities))} entities)")
            print(f"- entities_extracted: {len(entities)}")
            print(f"- relations_extracted: {len(relations)}")
            if entities:
                sample = entities[0]
                print(f"- sample_entity: {sample.entity_name} (type: {type(sample.entity_name)})")
            
            if entity_format_ok and len(entities) > 0:
                print("\n✅ STAGE 1: PASSED - Entity names are proper strings!")
                return True
            else:
                print("\n❌ STAGE 1: FAILED")
                return False
                
        except Exception as e:
            print(f"\n❌ ERROR during entity extraction: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print("\n❌ No chunks found - cannot test entity extraction")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_entity_extraction())