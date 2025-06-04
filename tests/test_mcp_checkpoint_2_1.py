"""
Tests for Phase 2, Checkpoint 2.1: AOT Query Preprocessor
"""

import asyncio
import pytest
from datetime import datetime
import json

from Core.AOT.aot_processor import AOTQueryProcessor, AtomicState, TransitionProbability


class TestAOTQueryProcessor:
    """Test AOT query decomposition and recomposition"""
    
    @pytest.fixture
    async def aot_processor(self):
        """Create test AOT processor"""
        return AOTQueryProcessor()
    
    async def test_query_decomposition(self, aot_processor):
        """Test: Complex queries decompose into atomic states"""
        query = "What were the main causes of the French Revolution?"
        
        states = await aot_processor.decompose_query(query)
        
        assert len(states) > 0
        
        # Check for expected state types
        state_types = {s.state_type for s in states}
        assert "entity" in state_types  # Should find "French Revolution"
        assert "attribute" in state_types  # Should find "main"
        assert "action" in state_types  # Should find "retrieve_information"
        
        # Check entity extraction
        entities = [s for s in states if s.state_type == "entity"]
        entity_contents = [e.content for e in entities]
        assert any("French Revolution" in content for content in entity_contents)
    
    async def test_atomic_states_memoryless(self, aot_processor):
        """Test: Atomic states are memoryless (Markov property)"""
        query = "Compare the French Revolution to the American Revolution"
        
        states = await aot_processor.decompose_query(query)
        
        # Each state should be self-contained
        for state in states:
            assert hasattr(state, 'state_id')
            assert hasattr(state, 'content')
            assert hasattr(state, 'state_type')
            assert hasattr(state, 'dependencies')
            assert hasattr(state, 'metadata')
            
            # State content should be atomic (single concept)
            assert len(state.content.split()) <= 3  # Simple heuristic for atomicity
    
    async def test_context_reduction(self, aot_processor):
        """Test: Context size reduced by >50%"""
        query = "What were the causes and effects of both the French and American Revolutions?"
        
        # Create a large context
        original_context = {
            "query": query,
            "user_history": ["previous query 1", "previous query 2"],
            "session_data": {
                "user_id": "test123",
                "preferences": {"detail_level": "high", "language": "en"},
                "timestamp": datetime.utcnow().isoformat()
            },
            "corpus_metadata": {
                "size": "1GB",
                "documents": 10000,
                "last_updated": "2024-01-01"
            },
            "large_data": "x" * 1000  # Simulate large context
        }
        
        states = await aot_processor.decompose_query(query, original_context)
        
        reduction = aot_processor.calculate_context_reduction(original_context, states)
        
        assert reduction > 0.5  # >50% reduction
        assert len(states) >= 4  # Should have multiple entities and actions
    
    async def test_state_dependencies(self, aot_processor):
        """Test: Dependencies correctly identified"""
        query = "How did Napoleon influence the French Revolution?"
        
        states = await aot_processor.decompose_query(query)
        
        # Find relationship states
        rel_states = [s for s in states if s.state_type == "relationship"]
        assert len(rel_states) > 0
        
        # Relationship should depend on entity states
        entity_ids = {s.state_id for s in states if s.state_type == "entity"}
        for rel in rel_states:
            assert len(rel.dependencies) > 0
            # At least some dependencies should be entities
            assert any(dep in entity_ids for dep in rel.dependencies)
    
    async def test_transition_probabilities(self, aot_processor):
        """Test: Markov transition probabilities calculated"""
        query = "What caused the French Revolution?"
        
        states = await aot_processor.decompose_query(query)
        
        # Should have transitions between states
        assert len(aot_processor.transitions) > 0
        
        # Check transition properties
        for (from_id, to_id), transition in aot_processor.transitions.items():
            assert 0.0 <= transition.probability <= 1.0
            assert transition.transition_type in ["sequential", "parallel", "conditional"]
            
            # Verify states exist
            assert from_id in aot_processor.state_cache
            assert to_id in aot_processor.state_cache
    
    async def test_result_recomposition(self, aot_processor):
        """Test: Results correctly recomposed from atomic states"""
        query = "What were the causes of the French Revolution?"
        
        states = await aot_processor.decompose_query(query)
        
        # Simulate execution results for each state
        state_results = {}
        for state in states:
            if state.state_type == "entity":
                state_results[state.state_id] = {
                    "found": True,
                    "data": f"Information about {state.content}"
                }
            elif state.state_type == "attribute":
                state_results[state.state_id] = {
                    "filter_applied": True,
                    "attribute": state.content
                }
            elif state.state_type == "action":
                state_results[state.state_id] = {
                    "executed": True,
                    "summary": f"Retrieved information about causes"
                }
        
        # Recompose results
        final_result = await aot_processor.recompose_results(state_results)
        
        assert "entities" in final_result
        assert "relationships" in final_result
        assert "attributes" in final_result
        assert "actions" in final_result
        assert "summary" in final_result
        
        # Summary should mention key findings
        assert len(final_result["summary"]) > 0
    
    async def test_parallel_execution_identification(self, aot_processor):
        """Test: Independent states identified for parallel execution"""
        query = "List the leaders of the French Revolution and the American Revolution"
        
        states = await aot_processor.decompose_query(query)
        
        # Find entity states
        entity_states = [s for s in states if s.state_type == "entity"]
        
        # French and American Revolution entities should be independent
        french_states = [s for s in entity_states if "French" in s.content]
        american_states = [s for s in entity_states if "American" in s.content]
        
        assert len(french_states) > 0
        assert len(american_states) > 0
        
        # Check they don't depend on each other
        for f_state in french_states:
            for a_state in american_states:
                assert a_state.state_id not in f_state.dependencies
                assert f_state.state_id not in a_state.dependencies
    
    async def test_complex_query_handling(self, aot_processor):
        """Test: Handle complex multi-part queries"""
        query = ("Compare the causes and effects of the French Revolution "
                "with those of the American Revolution, focusing on "
                "economic factors and social inequality")
        
        states = await aot_processor.decompose_query(query)
        
        # Should decompose into many atomic states
        assert len(states) >= 6  # Multiple entities, attributes, relationships
        
        # Check for comparison action
        action_states = [s for s in states if s.state_type == "action"]
        assert any("comparison" in s.content for s in action_states)
        
        # Check for specific attributes
        attr_states = [s for s in states if s.state_type == "attribute"]
        attr_contents = [s.content for s in attr_states]
        # Should identify "causes", "effects" as attributes
        assert any("causes" in c for c in attr_contents)
        
        # Calculate context reduction
        large_context = {"query": query, "metadata": {"size": "large", "data": "x" * 1000}}
        reduction = aot_processor.calculate_context_reduction(large_context, states)
        assert reduction > 0.5

@pytest.mark.asyncio
async def test_aot_integration():
    """Integration test of full AOT pipeline"""
    processor = AOTQueryProcessor()
    
    # Complex query requiring decomposition
    query = ("What were the primary economic and social causes of the "
            "French Revolution, and how did they lead to the eventual "
            "rise of Napoleon Bonaparte?")
    
    # Decompose
    states = await processor.decompose_query(query)
    
    # Verify decomposition quality
    assert len(states) >= 5
    
    entity_states = [s for s in states if s.state_type == "entity"]
    assert len(entity_states) >= 2  # French Revolution, Napoleon
    
    # Simulate parallel execution of independent states
    independent_states = []
    for state in states:
        # States with no dependencies can run in parallel
        if len(state.dependencies) == 0:
            independent_states.append(state)
    
    assert len(independent_states) >= 2  # Some states should be independent
    
    # Test recomposition with mock results
    results = {}
    for state in states:
        results[state.state_id] = {
            "status": "success",
            "data": f"Result for {state.content}"
        }
    
    final = await processor.recompose_results(results)
    assert final["summary"]
    assert len(final["entities"]) > 0


if __name__ == "__main__":
    asyncio.run(test_aot_integration())