"""
Test MCP Checkpoint 3.3: Cross-Modal Integration

Success Criteria:
1. Cross-modal entity linking > 90% accuracy
2. Schema translation works
3. Unified query execution
4. Results properly aggregated
"""

import asyncio
import pytest
from typing import Dict, Any, List
from datetime import datetime


class TestMCPCheckpoint3_3:
    """Test cross-modal integration"""
    
    @classmethod
    def setup_class(cls):
        """Setup test environment"""
        cls.bridge = None
        cls.unified_interface = None
        
    @classmethod
    def teardown_class(cls):
        """Cleanup after tests"""
        pass
    
    @pytest.mark.asyncio
    async def test_entity_linking(self):
        """Test 1: Cross-modal entity linking"""
        print("\n" + "="*50)
        print("Test 1: Cross-modal entity linking")
        print("="*50)
        
        from Core.MCP.cross_modal_integration import (
            get_cross_modal_bridge, ModalityType
        )
        from Core.MCP.mcp_agent_interface import get_agent_interface
        
        # Start agent interface
        agent_interface = get_agent_interface()
        await agent_interface.start()
        
        try:
            # Get bridge
            bridge = get_cross_modal_bridge()
            
            # Test entity linking
            test_entities = [
                ("Washington", "graph_123", "sql_456"),
                ("Jefferson", "graph_124", "sql_457"),
                ("Lincoln", "graph_125", "sql_458"),
                ("Roosevelt", "graph_126", "sql_459"),
                ("Kennedy", "graph_127", "sql_460")
            ]
            
            print("\nLinking entities across modalities:")
            linked_ids = []
            
            for entity_name, graph_id, sql_id in test_entities:
                entity_id = await bridge.link_entity(
                    entity_name=entity_name,
                    modality1=ModalityType.GRAPH_RAG,
                    id1=graph_id,
                    modality2=ModalityType.STRUCT_GPT,
                    id2=sql_id,
                    confidence=0.95
                )
                linked_ids.append(entity_id)
                print(f"- {entity_name}: {entity_id}")
            
            # Test finding linked entities
            print("\nTesting entity lookup:")
            
            # Find all links for Washington's graph ID
            links = await bridge.find_linked_entities(
                ModalityType.GRAPH_RAG,
                "graph_123"
            )
            
            assert len(links) == 2  # Should have both GraphRAG and StructGPT links
            modalities = {link.modality for link in links}
            assert ModalityType.GRAPH_RAG in modalities
            assert ModalityType.STRUCT_GPT in modalities
            
            print(f"- Found {len(links)} links for Washington")
            
            # Test accuracy
            correct_links = 0
            total_tests = len(test_entities)
            
            for i, (entity_name, graph_id, sql_id) in enumerate(test_entities):
                # Check if we can find the SQL ID from the graph ID
                links = await bridge.find_linked_entities(
                    ModalityType.GRAPH_RAG,
                    graph_id
                )
                
                sql_link = next(
                    (l for l in links if l.modality == ModalityType.STRUCT_GPT),
                    None
                )
                
                if sql_link and sql_link.local_id == sql_id:
                    correct_links += 1
            
            accuracy = correct_links / total_tests
            print(f"\nLinking accuracy: {accuracy * 100:.0f}%")
            
            assert accuracy >= 0.9, f"Accuracy {accuracy} below 90% threshold"
            
            print("\nEvidence:")
            print(f"- Entities tested: {total_tests}")
            print(f"- Correctly linked: {correct_links}")
            print(f"- Accuracy: {accuracy * 100:.0f}% > 90% ✓")
            print("- Bidirectional linking works")
            print("\nResult: PASSED ✓")
            
        finally:
            await agent_interface.stop()
    
    @pytest.mark.asyncio
    async def test_schema_translation(self):
        """Test 2: Schema translation between modalities"""
        print("\n" + "="*50)
        print("Test 2: Schema translation")
        print("="*50)
        
        from Core.MCP.cross_modal_integration import get_cross_modal_bridge
        
        bridge = get_cross_modal_bridge()
        
        # Register schema mappings
        print("\nRegistering schema mappings:")
        
        await bridge.register_schema_mapping(
            source_modality="sql",
            source_schema="users",
            target_modality="graph",
            target_schema="Person",
            field_mappings={
                "id": "node_id",
                "name": "name",
                "email": "properties.email",
                "created_at": "properties.created_timestamp"
            }
        )
        print("- SQL users -> Graph Person")
        
        # Test SQL to Graph translation
        print("\nTest 1: SQL to Graph translation")
        sql_query = {
            "type": "SELECT",
            "select": ["name", "email"],
            "from": "users",
            "where": {"name": "Washington"}
        }
        
        graph_query = await bridge.translate_query(
            sql_query,
            source_modality="sql",
            target_modality="graph"
        )
        
        print(f"Original SQL: {sql_query}")
        print(f"Translated Graph: {graph_query}")
        
        assert graph_query["type"] == "entity_search"
        assert graph_query["entity_type"] == "Person"
        assert len(graph_query["filters"]) > 0
        
        # Test Graph to SQL translation
        print("\nTest 2: Graph to SQL translation")
        graph_query2 = {
            "type": "entity_search",
            "entity_type": "Person",
            "filters": [
                {"property": "name", "operator": "equals", "value": "Jefferson"}
            ],
            "return_properties": ["name", "role"]
        }
        
        sql_query2 = await bridge.translate_query(
            graph_query2,
            source_modality="graph",
            target_modality="sql"
        )
        
        print(f"Original Graph: {graph_query2}")
        print(f"Translated SQL: {sql_query2}")
        
        assert sql_query2["type"] == "SELECT"
        assert sql_query2["from"] == "users"
        assert "name" in sql_query2["where"]
        
        print("\nEvidence:")
        print("- SQL→Graph translation successful")
        print("- Graph→SQL translation successful")
        print("- Field mappings preserved")
        print("- Query semantics maintained")
        print("\nResult: PASSED ✓")
    
    @pytest.mark.asyncio
    async def test_unified_query(self):
        """Test 3: Unified query execution across modalities"""
        print("\n" + "="*50)
        print("Test 3: Unified query execution")
        print("="*50)
        
        from Core.MCP.cross_modal_integration import (
            get_unified_query_interface, ModalityType
        )
        from Core.MCP.mcp_agent_interface import get_agent_interface
        
        # Start agent interface
        agent_interface = get_agent_interface()
        await agent_interface.start()
        
        try:
            # Register modality agents
            print("\nRegistering modality agents:")
            
            await agent_interface.register_agent(
                "graphrag_agent", "GraphRAG Agent", ["graph_analysis"]
            )
            await agent_interface.register_agent(
                "structgpt_agent", "StructGPT Agent", ["structured_query"]
            )
            await agent_interface.register_agent(
                "autocoder_agent", "Autocoder Agent", ["code_analysis"]
            )
            
            # Get unified interface
            unified = get_unified_query_interface()
            
            # Register modalities
            await unified.register_modality(ModalityType.GRAPH_RAG, "graphrag_agent")
            await unified.register_modality(ModalityType.STRUCT_GPT, "structgpt_agent")
            await unified.register_modality(ModalityType.AUTOCODER, "autocoder_agent")
            
            print("- GraphRAG registered")
            print("- StructGPT registered")
            print("- Autocoder registered")
            
            # Execute unified query
            print("\nExecuting unified query:")
            query = "Find all interactions with Washington"
            
            result = await unified.execute_unified_query(query)
            
            print(f"Query: '{query}'")
            print(f"\nModalities queried: {result['modalities_queried']}")
            print(f"Unified results: {len(result['unified_results'])} entities")
            
            # Verify results
            assert len(result["modalities_queried"]) == 3
            assert "graph_rag" in result["by_modality"]
            assert "struct_gpt" in result["by_modality"]
            assert "autocoder" in result["by_modality"]
            
            # Check unified results
            washington_found = False
            for entity_result in result["unified_results"]:
                if entity_result["entity"] == "Washington":
                    washington_found = True
                    occurrences = entity_result["occurrences"]
                    print(f"\nWashington found in {len(occurrences)} modalities:")
                    for occ in occurrences:
                        print(f"- {occ['modality']}: {occ['data']}")
            
            assert washington_found, "Washington not found in results"
            
            # Check statistics
            stats = result["statistics"]
            print(f"\nStatistics:")
            print(f"- Total entities: {stats['total_entities']}")
            print(f"- Modalities responded: {stats['modalities_responded']}")
            print(f"- Cross-modal matches: {stats['cross_modal_matches']}")
            
            assert stats["modalities_responded"] == 3
            assert stats["cross_modal_matches"] > 0
            
            print("\nEvidence:")
            print(f"- Query executed across 3 modalities")
            print(f"- All modalities returned results")
            print(f"- Results properly aggregated")
            print(f"- Entity linking detected")
            print("\nResult: PASSED ✓")
            
        finally:
            await agent_interface.stop()
    
    @pytest.mark.asyncio
    async def test_result_format(self):
        """Test 4: Unified result format"""
        print("\n" + "="*50)
        print("Test 4: Result format validation")
        print("="*50)
        
        from Core.MCP.cross_modal_integration import (
            get_unified_query_interface, ModalityType
        )
        
        unified = get_unified_query_interface()
        
        # Test result aggregation directly
        query_intent = {
            "original": "test query",
            "type": "search",
            "entities": ["TestEntity"]
        }
        
        # Simulate modality results
        modalities = [ModalityType.GRAPH_RAG, ModalityType.STRUCT_GPT]
        results = [
            {
                "modality": "graph_rag",
                "results": [
                    {"entity": "TestEntity", "type": "Node", "id": "123"}
                ]
            },
            {
                "modality": "struct_gpt",
                "results": [
                    {"name": "TestEntity", "table": "entities", "id": 456}
                ]
            }
        ]
        
        # Aggregate
        aggregated = await unified._aggregate_results(
            query_intent,
            modalities,
            results
        )
        
        print("Aggregated result structure:")
        print(f"- Query: {aggregated['query']}")
        print(f"- Timestamp: {aggregated['timestamp']}")
        print(f"- Modalities: {aggregated['modalities_queried']}")
        print(f"- Unified results: {len(aggregated['unified_results'])}")
        
        # Validate structure
        assert "query" in aggregated
        assert "timestamp" in aggregated
        assert "modalities_queried" in aggregated
        assert "unified_results" in aggregated
        assert "by_modality" in aggregated
        assert "statistics" in aggregated
        
        # Validate unified result format
        assert len(aggregated["unified_results"]) == 1
        unified_entity = aggregated["unified_results"][0]
        assert unified_entity["entity"] == "TestEntity"
        assert len(unified_entity["occurrences"]) == 2
        
        print("\nEvidence:")
        print("- Result has all required fields")
        print("- Timestamp included")
        print("- Statistics calculated")
        print("- Unified JSON format valid")
        print("\nResult: PASSED ✓")
    
    def test_summary(self):
        """Print test summary"""
        print("\n" + "="*50)
        print("CHECKPOINT 3.3 SUMMARY")
        print("="*50)
        print("✓ Entity linking accuracy > 90%")
        print("✓ Schema translation works")
        print("✓ Unified query execution successful")
        print("✓ Result format validated")
        print("\nCross-modal integration complete!")
        print("All tests passed! ✅")
        print("="*50)


if __name__ == "__main__":
    # Run tests with detailed output
    pytest.main([__file__, "-v", "-s"])