#!/usr/bin/env python3
"""
Comprehensive analysis and fix for tool input/output field naming mismatches in DIGIMON
"""

import json
import sys
sys.path.append('.')

from typing import Dict, List, Set, Optional, Any
from pathlib import Path
from Core.AgentSchema import tool_contracts, corpus_tool_contracts, graph_construction_tool_contracts
from Core.AgentOrchestrator.orchestrator import AgentOrchestrator
from Core.Common.Logger import logger
import inspect

def analyze_tool_contracts():
    """Analyze all tool contracts to understand input/output field names"""
    
    print("=" * 80)
    print("DIGIMON Tool Contract Analysis")
    print("=" * 80)
    
    # Collect all tool contracts
    all_contracts = {}
    
    # From tool_contracts
    for name, obj in inspect.getmembers(tool_contracts):
        if name.endswith('Inputs') or name.endswith('Outputs'):
            all_contracts[name] = obj
            
    # From corpus_tool_contracts  
    for name, obj in inspect.getmembers(corpus_tool_contracts):
        if name.endswith('Inputs') or name.endswith('Outputs'):
            all_contracts[name] = obj
            
    # From graph_construction_tool_contracts
    for name, obj in inspect.getmembers(graph_construction_tool_contracts):
        if name.endswith('Inputs') or name.endswith('Outputs'):
            all_contracts[name] = obj
    
    # Analyze each contract
    input_contracts = {}
    output_contracts = {}
    
    for name, contract in all_contracts.items():
        if name.endswith('Inputs'):
            input_contracts[name] = contract
        else:
            output_contracts[name] = contract
    
    print(f"\nFound {len(input_contracts)} input contracts and {len(output_contracts)} output contracts")
    
    # Print detailed analysis
    print("\n" + "=" * 80)
    print("INPUT CONTRACTS (what tools expect)")
    print("=" * 80)
    
    for name, contract in sorted(input_contracts.items()):
        print(f"\n{name}:")
        for field_name, field_info in contract.model_fields.items():
            required = field_info.is_required()
            default = field_info.default
            print(f"  - {field_name}: {field_info.annotation} {'(required)' if required else f'(default={default})'}")
    
    print("\n" + "=" * 80)
    print("OUTPUT CONTRACTS (what tools return)")
    print("=" * 80)
    
    for name, contract in sorted(output_contracts.items()):
        print(f"\n{name}:")
        for field_name, field_info in contract.model_fields.items():
            required = field_info.is_required()
            default = field_info.default
            print(f"  - {field_name}: {field_info.annotation} {'(required)' if required else f'(default={default})'}")
    
    # Check tool registry mapping
    print("\n" + "=" * 80)
    print("TOOL REGISTRY MAPPING")
    print("=" * 80)
    
    # Create a dummy orchestrator to access tool registry
    from Option.Config2 import Config
    from Core.Provider.LLMProviderRegister import create_llm_instance
    from Core.Index.EmbeddingFactory import RAGEmbeddingFactory
    from Core.Chunk.ChunkFactory import ChunkFactory
    from Core.AgentSchema.context import GraphRAGContext
    
    config = Config.default()
    llm = create_llm_instance(config.llm)
    emb_factory = RAGEmbeddingFactory()
    encoder = emb_factory.get_rag_embedding(config=config)
    chunk_factory = ChunkFactory(config)
    context = GraphRAGContext(main_config=config, target_dataset_name="test")
    
    orchestrator = AgentOrchestrator(
        main_config=config,
        llm_instance=llm,
        encoder_instance=encoder,
        chunk_factory=chunk_factory,
        graphrag_context=context
    )
    
    print("\nRegistered tools and their input contracts:")
    for tool_id, (func, input_class) in sorted(orchestrator._tool_registry.items()):
        print(f"\n{tool_id}:")
        print(f"  Function: {func.__name__}")
        print(f"  Input class: {input_class.__name__}")
        print(f"  Expected fields:")
        for field_name, field_info in input_class.model_fields.items():
            required = field_info.is_required()
            print(f"    - {field_name} {'(required)' if required else '(optional)'}")
    
    # Identify common patterns and issues
    print("\n" + "=" * 80)
    print("COMMON PATTERNS AND POTENTIAL ISSUES")
    print("=" * 80)
    
    # Check for graph building tools
    graph_build_tools = [t for t in orchestrator._tool_registry.keys() if t.startswith('graph.Build')]
    print(f"\nGraph building tools ({len(graph_build_tools)}):")
    for tool_id in graph_build_tools:
        _, input_class = orchestrator._tool_registry[tool_id]
        print(f"  {tool_id}: expects {list(input_class.model_fields.keys())}")
    
    # Check for tools that reference other outputs
    print("\nTools that typically reference outputs from other steps:")
    reference_fields = ['graph_reference_id', 'vdb_reference_id', 'entity_ids', 'seed_entity_ids']
    for tool_id, (_, input_class) in orchestrator._tool_registry.items():
        fields = list(input_class.model_fields.keys())
        refs = [f for f in fields if f in reference_fields]
        if refs:
            print(f"  {tool_id}: references {refs}")
    
    # Generate recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS FOR FIXING FIELD MISMATCHES")
    print("=" * 80)
    
    print("""
1. Graph Building Tools:
   - All graph.Build* tools only need 'target_dataset_name', not 'corpus_ref'
   - They internally use ChunkFactory to load corpus based on dataset name
   - The agent should NOT pass corpus_ref to these tools

2. Common Reference Patterns:
   - When referencing graph outputs: use the alias defined in named_outputs
   - Entity/VDB tools expect 'graph_reference_id' from graph build steps
   - Search results are in 'similar_entities' field, not 'search_results'

3. Agent Brain Improvements Needed:
   - Update system prompt to clarify graph tools don't need corpus_ref
   - Add validation to ensure referenced outputs exist
   - Improve error messages when field references fail

4. Orchestrator Enhancements:
   - Add debug logging for field resolution failures
   - Provide clearer error messages about missing references
   - Consider adding field name suggestions when validation fails
""")

    # Generate a mapping file for the agent
    mapping = {
        "tool_inputs": {},
        "tool_outputs": {},
        "common_aliases": {
            "prepared_corpus_name": "corpus_json_path",
            "er_graph_id": "graph_id",
            "entity_vdb_id": "vdb_reference_id",
            "search_results": "similar_entities"
        }
    }
    
    for tool_id, (_, input_class) in orchestrator._tool_registry.items():
        mapping["tool_inputs"][tool_id] = {
            "class": input_class.__name__,
            "required_fields": [f for f, info in input_class.model_fields.items() if info.is_required()],
            "optional_fields": [f for f, info in input_class.model_fields.items() if not info.is_required()]
        }
    
    # Save mapping to file
    with open("tool_field_mapping.json", "w") as f:
        json.dump(mapping, f, indent=2)
    
    print(f"\nGenerated tool_field_mapping.json with complete field information")
    
    return mapping

def test_specific_scenarios():
    """Test specific scenarios that commonly fail"""
    print("\n" + "=" * 80)
    print("TESTING SPECIFIC SCENARIOS")
    print("=" * 80)
    
    # Test 1: BuildERGraphInputs with corpus_ref
    print("\nTest 1: BuildERGraphInputs with extra 'corpus_ref' field")
    try:
        from Core.AgentSchema.graph_construction_tool_contracts import BuildERGraphInputs
        inputs = BuildERGraphInputs(
            target_dataset_name="test",
            corpus_ref="some/path",  # This should be ignored
            force_rebuild=False
        )
        print("✓ Extra fields are silently ignored (corpus_ref not in model)")
        print(f"  Created fields: {inputs.model_fields_set}")
        print(f"  Model dump: {inputs.model_dump()}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test 2: Missing required field
    print("\nTest 2: BuildERGraphInputs with missing required field")
    try:
        inputs = BuildERGraphInputs(force_rebuild=False)  # Missing target_dataset_name
        print("✗ Should have failed!")
    except Exception as e:
        print(f"✓ Correctly failed with: {type(e).__name__}")
        print(f"  Error: {str(e)[:200]}...")
    
    # Test 3: EntityVDBSearchOutputs structure
    print("\nTest 3: EntityVDBSearchOutputs structure")
    from Core.AgentSchema.tool_contracts import EntityVDBSearchOutputs, VDBSearchResultItem
    output = EntityVDBSearchOutputs(
        status="success",
        message="Found entities",
        similar_entities=[
            VDBSearchResultItem(entity_name="AI", score=0.9, metadata={}),
            VDBSearchResultItem(entity_name="ML", score=0.8, metadata={})
        ]
    )
    print(f"✓ Output fields: {list(output.model_dump().keys())}")
    print(f"  Entity extraction: {[e.entity_name for e in output.similar_entities]}")

if __name__ == "__main__":
    # Run analysis
    mapping = analyze_tool_contracts()
    
    # Run tests
    test_specific_scenarios()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
The field naming mismatch issue occurs when:
1. Agent generates plans with fields that don't exist in tool contracts
2. Agent references outputs using wrong field names
3. Tools expect different field names than what agent provides

Key fixes needed:
1. Update agent brain prompts to use correct field names
2. Add validation in orchestrator for better error messages
3. Update tool contracts to have consistent naming patterns
4. Consider adding strict validation mode to catch these errors early
""")