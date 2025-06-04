"""
Tests for Phase 3, Checkpoint 3.1: MCP-Based Blackboard System

Tests the blackboard system built on MCP shared context with
knowledge sources, reactive control, and collaborative problem-solving.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from Core.MCP.blackboard_system import (
    BlackboardSystem, KnowledgeEntry, KnowledgeType,
    BlackboardEventType, KnowledgeSource, ReactiveControlStrategy
)
from Core.MCP.knowledge_sources import (
    EntityKnowledgeSource, RelationshipKnowledgeSource,
    HypothesisGeneratorSource, SolutionSynthesizerSource,
    create_mcp_knowledge_tools
)
from Core.MCP.shared_context import SharedContextStore


@pytest.fixture
def shared_context():
    """Create a shared context store"""
    return SharedContextStore()


@pytest.fixture
def blackboard(shared_context):
    """Create a blackboard system"""
    return BlackboardSystem(shared_context)


@pytest.fixture
def mock_entity_tool():
    """Mock entity VDB search tool"""
    tool = Mock()
    tool.run = AsyncMock(return_value={
        "status": "success",
        "entities": [
            {"name": "Entity1", "attributes": {"type": "person"}, "score": 0.9},
            {"name": "Entity2", "attributes": {"type": "place"}, "score": 0.8}
        ]
    })
    return tool


@pytest.fixture
def mock_relationship_tool():
    """Mock relationship search tool"""
    tool = Mock()
    tool.run = AsyncMock(return_value={
        "status": "success",
        "relationships": [
            {
                "subject": "Entity1",
                "predicate": "located_in",
                "object": "Entity2",
                "attributes": {}
            }
        ]
    })
    return tool


@pytest.fixture
def mock_chunk_tool():
    """Mock chunk from relationships tool"""
    tool = Mock()
    tool.run = AsyncMock(return_value={
        "status": "success",
        "chunks": ["Entity1 is located in Entity2 according to the source."]
    })
    return tool


class TestBlackboardSystem:
    """Test the core blackboard system"""
    
    @pytest.mark.asyncio
    async def test_blackboard_initialization(self, blackboard):
        """Test blackboard initializes with proper namespace"""
        blackboard_data = blackboard._blackboard_data
        assert blackboard_data is not None
        assert "knowledge" in blackboard_data
        assert "hypotheses" in blackboard_data
        assert "solutions" in blackboard_data
        assert "control_state" in blackboard_data
    
    @pytest.mark.asyncio
    async def test_add_knowledge(self, blackboard):
        """Test adding knowledge to the blackboard"""
        entry = KnowledgeEntry(
            knowledge_type=KnowledgeType.FACT,
            content={"entity": "TestEntity", "value": 42},
            source="test_source",
            confidence=0.95
        )
        
        knowledge_id = await blackboard.add_knowledge(entry)
        
        assert knowledge_id == entry.id
        
        # Verify knowledge was stored
        blackboard_data = blackboard._blackboard_data
        assert knowledge_id in blackboard_data["knowledge"]
        stored = blackboard_data["knowledge"][knowledge_id]
        assert stored["content"] == {"entity": "TestEntity", "value": 42}
        assert stored["confidence"] == 0.95
    
    @pytest.mark.asyncio
    async def test_update_knowledge(self, blackboard):
        """Test updating existing knowledge"""
        # Add initial knowledge
        entry = KnowledgeEntry(
            knowledge_type=KnowledgeType.FACT,
            content={"value": 1},
            source="test_source"
        )
        knowledge_id = await blackboard.add_knowledge(entry)
        
        # Update it
        success = await blackboard.update_knowledge(
            knowledge_id,
            {"content": {"value": 2}, "confidence": 0.8},
            "updater"
        )
        
        assert success is True
        
        # Verify update
        blackboard_data = blackboard._blackboard_data
        stored = blackboard_data["knowledge"][knowledge_id]
        assert stored["content"]["value"] == 2
        assert stored["confidence"] == 0.8
        assert stored["updated_at"] != stored["created_at"]
    
    @pytest.mark.asyncio
    async def test_get_knowledge_with_filters(self, blackboard):
        """Test retrieving knowledge with various filters"""
        # Add different types of knowledge
        fact1 = KnowledgeEntry(
            KnowledgeType.FACT, {"data": 1}, "source1", 0.9
        )
        fact2 = KnowledgeEntry(
            KnowledgeType.FACT, {"data": 2}, "source2", 0.7
        )
        hypothesis = KnowledgeEntry(
            KnowledgeType.HYPOTHESIS, {"data": 3}, "source1", 0.8
        )
        
        await blackboard.add_knowledge(fact1)
        await blackboard.add_knowledge(fact2)
        await blackboard.add_knowledge(hypothesis)
        
        # Test type filter
        facts = await blackboard.get_knowledge(knowledge_type=KnowledgeType.FACT)
        assert len(facts) == 2
        
        # Test source filter
        source1_items = await blackboard.get_knowledge(source="source1")
        assert len(source1_items) == 2
        
        # Test confidence filter
        high_conf = await blackboard.get_knowledge(min_confidence=0.8)
        assert len(high_conf) == 2
        assert all(item["confidence"] >= 0.8 for item in high_conf)
    
    @pytest.mark.asyncio
    async def test_propose_hypothesis(self, blackboard):
        """Test hypothesis proposal"""
        # Add supporting evidence
        fact = KnowledgeEntry(
            KnowledgeType.FACT, {"entity": "A"}, "source", 0.9
        )
        fact_id = await blackboard.add_knowledge(fact)
        
        # Propose hypothesis
        hypothesis = KnowledgeEntry(
            KnowledgeType.HYPOTHESIS,
            {"claim": "A is important"},
            "hypothesis_source",
            0.8
        )
        
        hyp_id = await blackboard.propose_hypothesis(
            hypothesis,
            [fact_id]
        )
        
        # Verify hypothesis storage
        blackboard_data = blackboard._blackboard_data
        assert hyp_id in blackboard_data["knowledge"]
        assert hyp_id in blackboard_data["hypotheses"]
        
        hyp_info = blackboard_data["hypotheses"][hyp_id]
        assert hyp_info["status"] == "proposed"
        assert hyp_info["validation_attempts"] == 0
        
        stored_hyp = blackboard_data["knowledge"][hyp_id]
        assert fact_id in stored_hyp["supporting_evidence"]
    
    @pytest.mark.asyncio
    async def test_validate_hypothesis(self, blackboard):
        """Test hypothesis validation"""
        # Create and propose hypothesis
        hypothesis = KnowledgeEntry(
            KnowledgeType.HYPOTHESIS,
            {"claim": "Test claim"},
            "source",
            0.7
        )
        hyp_id = await blackboard.propose_hypothesis(hypothesis, [])
        
        # Validate it
        success = await blackboard.validate_hypothesis(
            hyp_id,
            "validator1",
            is_valid=True,
            evidence=["evidence1"],
            confidence=0.85
        )
        
        assert success is True
        
        # Check validation was recorded
        blackboard_data = blackboard._blackboard_data
        hyp_info = blackboard_data["hypotheses"][hyp_id]
        assert hyp_info["validation_attempts"] == 1
        assert len(hyp_info["validators"]) == 1
        assert hyp_info["validators"][0]["is_valid"] is True
        assert hyp_info["validators"][0]["confidence"] == 0.85
        
        # Check evidence was added
        stored_hyp = blackboard_data["knowledge"][hyp_id]
        assert "evidence1" in stored_hyp["supporting_evidence"]
        assert stored_hyp["confidence"] == 0.85  # Updated confidence
    
    @pytest.mark.asyncio
    async def test_add_solution(self, blackboard):
        """Test adding solutions"""
        # Create a solution
        solution = KnowledgeEntry(
            KnowledgeType.SOLUTION,
            {"answer": "The answer is 42", "confidence": 0.9},
            "solver",
            0.9
        )
        
        sol_id = await blackboard.add_solution(
            solution,
            ["contributing1", "contributing2"]
        )
        
        # Verify solution storage
        blackboard_data = blackboard._blackboard_data
        assert sol_id in blackboard_data["knowledge"]
        assert sol_id in blackboard_data["solutions"]
        
        sol_info = blackboard_data["solutions"][sol_id]
        assert sol_info["completeness"] == 1.0
        assert sol_info["contributing_knowledge"] == ["contributing1", "contributing2"]
    
    @pytest.mark.asyncio
    async def test_event_subscription(self, blackboard):
        """Test event subscription and notification"""
        events_received = []
        
        def event_handler(event_type, data):
            events_received.append((event_type, data))
        
        # Subscribe to events
        blackboard.subscribe_to_event(
            BlackboardEventType.KNOWLEDGE_ADDED,
            event_handler
        )
        
        # Add knowledge
        entry = KnowledgeEntry(
            KnowledgeType.FACT,
            {"test": True},
            "source"
        )
        await blackboard.add_knowledge(entry)
        
        # Check event was received
        assert len(events_received) == 1
        event_type, data = events_received[0]
        assert event_type == BlackboardEventType.KNOWLEDGE_ADDED
        assert data["knowledge_id"] == entry.id


class TestKnowledgeSources:
    """Test the knowledge source implementations"""
    
    @pytest.mark.asyncio
    async def test_entity_knowledge_source(self, blackboard, mock_entity_tool):
        """Test entity knowledge source"""
        source = EntityKnowledgeSource()
        source.entity_tool = mock_entity_tool
        blackboard.register_knowledge_source(source)
        
        # Add a goal
        goal = KnowledgeEntry(
            KnowledgeType.GOAL,
            {"query": "Find important entities"},
            "user"
        )
        await blackboard.add_knowledge(goal)
        
        # Check can contribute
        assert await source.can_contribute() is True
        
        # Get contributions
        contributions = await source.contribute()
        
        assert len(contributions) == 2
        assert contributions[0].knowledge_type == KnowledgeType.FACT
        assert contributions[0].content["entity"] == "Entity1"
        assert contributions[0].confidence == 0.9
    
    @pytest.mark.asyncio
    async def test_relationship_knowledge_source(self, blackboard, mock_relationship_tool):
        """Test relationship knowledge source"""
        source = RelationshipKnowledgeSource()
        source.relationship_tool = mock_relationship_tool
        blackboard.register_knowledge_source(source)
        
        # Add entity fact
        fact = KnowledgeEntry(
            KnowledgeType.FACT,
            {"entity": "Entity1", "attributes": {}},
            "entity_source"
        )
        await blackboard.add_knowledge(fact)
        
        # Check can contribute
        assert await source.can_contribute() is True
        
        # Get contributions
        contributions = await source.contribute()
        
        assert len(contributions) == 1
        assert contributions[0].knowledge_type == KnowledgeType.INFERENCE
        assert contributions[0].content["subject"] == "Entity1"
        assert contributions[0].content["predicate"] == "located_in"
    
    @pytest.mark.asyncio
    async def test_hypothesis_generator(self, blackboard):
        """Test hypothesis generator source"""
        source = HypothesisGeneratorSource()
        blackboard.register_knowledge_source(source)
        
        # Add multiple facts about same entity
        fact1 = KnowledgeEntry(
            KnowledgeType.FACT,
            {"entity": "EntityA", "score": 0.9},
            "source1"
        )
        fact2 = KnowledgeEntry(
            KnowledgeType.INFERENCE,
            {"subject": "EntityA", "predicate": "has", "object": "property"},
            "source2"
        )
        
        await blackboard.add_knowledge(fact1)
        await blackboard.add_knowledge(fact2)
        
        # Check can contribute
        assert await source.can_contribute() is True
        
        # Generate hypotheses
        await source.contribute()
        
        # Check hypothesis was proposed
        hypotheses = await blackboard.get_knowledge(
            knowledge_type=KnowledgeType.HYPOTHESIS
        )
        assert len(hypotheses) >= 1
        assert "EntityA" in hypotheses[0]["content"]["entity"]
    
    @pytest.mark.asyncio
    async def test_solution_synthesizer(self, blackboard, mock_chunk_tool):
        """Test solution synthesizer source"""
        source = SolutionSynthesizerSource()
        source.chunk_tool = mock_chunk_tool
        blackboard.register_knowledge_source(source)
        
        # Create and validate a hypothesis
        hypothesis = KnowledgeEntry(
            KnowledgeType.HYPOTHESIS,
            {"entity": "EntityA", "hypothesis": "EntityA is key"},
            "hyp_source"
        )
        hyp_id = await blackboard.propose_hypothesis(hypothesis, [])
        
        # Validate it
        await blackboard.validate_hypothesis(
            hyp_id, "validator1", True, [], 0.9
        )
        await blackboard.validate_hypothesis(
            hyp_id, "validator2", True, [], 0.8
        )
        
        # Check can contribute
        assert await source.can_contribute() is True
        
        # Generate solution
        await source.contribute()
        
        # Check solution was created
        solutions = await blackboard.get_knowledge(
            knowledge_type=KnowledgeType.SOLUTION
        )
        assert len(solutions) >= 1


class TestReactiveControl:
    """Test the reactive control strategy"""
    
    @pytest.mark.asyncio
    async def test_reactive_control_strategy(self, blackboard):
        """Test reactive control finds solutions"""
        # Set up control strategy
        strategy = ReactiveControlStrategy(solution_threshold=0.8)
        blackboard.set_control_strategy(strategy)
        
        # Add a high-confidence solution
        solution = KnowledgeEntry(
            KnowledgeType.SOLUTION,
            {"answer": "Final answer"},
            "solver",
            confidence=0.85
        )
        sol_id = await blackboard.add_solution(solution, [])
        
        # Update solution info to mark as complete
        blackboard_data = blackboard._blackboard_data
        blackboard_data["solutions"][sol_id]["completeness"] = 0.9
        # Data is updated in memory
        
        # Run control cycle
        result = await strategy.execute_cycle(0)
        
        assert result == sol_id
    
    @pytest.mark.asyncio
    async def test_control_termination(self, blackboard):
        """Test control strategy termination conditions"""
        strategy = ReactiveControlStrategy()
        blackboard.set_control_strategy(strategy)
        
        # Add some active sources so it doesn't terminate immediately
        blackboard_data = blackboard._blackboard_data
        blackboard_data["control_state"]["active_sources"] = ["source1"]
        
        # Should not terminate with active sources
        assert await strategy.should_terminate() is False
        
        # Set high iteration count
        blackboard_data = blackboard._blackboard_data
        blackboard_data["control_state"]["iteration"] = 1001
        # Data is updated in memory
        
        # Should terminate due to iteration limit
        assert await strategy.should_terminate() is True


class TestMCPIntegration:
    """Test MCP tool integration"""
    
    @pytest.mark.asyncio
    async def test_create_mcp_tools(self, blackboard):
        """Test creating MCP tools for blackboard"""
        tools = create_mcp_knowledge_tools(blackboard)
        
        assert len(tools) == 3
        tool_names = [t.name for t in tools]
        assert "blackboard.contribute_knowledge" in tool_names
        assert "blackboard.query" in tool_names
        assert "blackboard.run_cycle" in tool_names
    
    @pytest.mark.asyncio
    async def test_contribute_knowledge_tool(self, blackboard, mock_entity_tool):
        """Test MCP tool for knowledge contribution"""
        tools = create_mcp_knowledge_tools(blackboard)
        contribute_tool = next(
            t for t in tools if t.name == "blackboard.contribute_knowledge"
        )
        
        # Patch the entity tool
        entity_source = blackboard.knowledge_sources["entity_knowledge_source"]
        entity_source.entity_tool = mock_entity_tool
        
        # Add a goal so entity source can contribute
        goal = KnowledgeEntry(
            KnowledgeType.GOAL,
            {"query": "test query"},
            "user"
        )
        await blackboard.add_knowledge(goal)
        
        # Execute tool
        result = await contribute_tool.handler(
            source_name="entity_knowledge_source",
            context={}
        )
        
        assert result["status"] == "success"
        assert result["contributions"] == 2
    
    @pytest.mark.asyncio
    async def test_query_blackboard_tool(self, blackboard):
        """Test MCP tool for querying blackboard"""
        tools = create_mcp_knowledge_tools(blackboard)
        query_tool = next(t for t in tools if t.name == "blackboard.query")
        
        # Add some knowledge
        fact = KnowledgeEntry(
            KnowledgeType.FACT,
            {"test": True},
            "source",
            0.9
        )
        await blackboard.add_knowledge(fact)
        
        # Query for facts
        result = await query_tool.handler(
            knowledge_type="fact",
            min_confidence=0.8,
            context={}
        )
        
        assert result["status"] == "success"
        assert len(result["knowledge"]) == 1
        assert result["stats"]["total_knowledge"] == 1
    
    @pytest.mark.asyncio
    async def test_run_cycle_tool(self, blackboard):
        """Test MCP tool for running control cycle"""
        tools = create_mcp_knowledge_tools(blackboard)
        cycle_tool = next(t for t in tools if t.name == "blackboard.run_cycle")
        
        # Set up control strategy
        strategy = ReactiveControlStrategy()
        blackboard.set_control_strategy(strategy)
        
        # Run cycle (no solution expected)
        result = await cycle_tool.handler(
            max_iterations=5,
            context={}
        )
        
        assert result["status"] == "no_solution"


class TestEndToEndScenario:
    """Test end-to-end blackboard scenarios"""
    
    @pytest.mark.asyncio
    async def test_full_problem_solving_cycle(
        self,
        blackboard,
        mock_entity_tool,
        mock_relationship_tool,
        mock_chunk_tool
    ):
        """Test a complete problem-solving cycle"""
        # Create knowledge sources with mocked tools
        entity_source = EntityKnowledgeSource()
        entity_source.entity_tool = mock_entity_tool
        
        relationship_source = RelationshipKnowledgeSource()
        relationship_source.relationship_tool = mock_relationship_tool
        
        hypothesis_source = HypothesisGeneratorSource()
        
        solution_source = SolutionSynthesizerSource()
        solution_source.chunk_tool = mock_chunk_tool
        
        # Register sources
        blackboard.register_knowledge_source(entity_source)
        blackboard.register_knowledge_source(relationship_source)
        blackboard.register_knowledge_source(hypothesis_source)
        blackboard.register_knowledge_source(solution_source)
        
        # Set control strategy with low threshold for testing
        strategy = ReactiveControlStrategy(solution_threshold=0.3)
        blackboard.set_control_strategy(strategy)
        
        # Add initial goal
        goal = KnowledgeEntry(
            KnowledgeType.GOAL,
            {"query": "What is the relationship between entities?"},
            "user"
        )
        await blackboard.add_knowledge(goal)
        
        # Run a few iterations to generate hypotheses
        await blackboard.run_control_cycle(max_iterations=3)
        
        # Manually validate some hypotheses to trigger solution generation
        hypotheses = await blackboard.get_knowledge(knowledge_type=KnowledgeType.HYPOTHESIS)
        if hypotheses:
            # Validate the first hypothesis
            await blackboard.validate_hypothesis(
                hypotheses[0]["id"],
                "test_validator", 
                True,
                ["test_evidence"],
                0.8
            )
        
        # Run more iterations to generate solution
        solution_id = await blackboard.run_control_cycle(max_iterations=5)
        
        # Verify we found a solution
        assert solution_id is not None
        
        # Check solution content
        blackboard_data = blackboard._blackboard_data
        solution = blackboard_data["knowledge"][solution_id]
        assert solution["knowledge_type"] == "solution"
        assert "answer" in solution["content"]
        assert solution["confidence"] > 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])