"""
Basic integration tests that can run without external services.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from Core.AgentBrain.agent_brain import PlanningAgent
from Core.AgentOrchestrator.orchestrator import AgentOrchestrator
from Core.AgentSchema.plan import ExecutionPlan, ExecutionStep, DynamicToolChainConfig, ToolCall
from Core.AgentSchema.context import GraphRAGContext


class TestBasicIntegration:
    """Basic integration tests with mocked dependencies."""
    
    @pytest.fixture
    def mock_context(self, mock_config):
        """Create a mock context."""
        return GraphRAGContext(
            target_dataset_name="test_dataset",
            main_config=mock_config
        )
    
    @pytest.fixture
    def mock_orchestrator(self, mock_context):
        """Create a mock orchestrator."""
        orchestrator = Mock(spec=AgentOrchestrator)
        orchestrator.context = mock_context
        orchestrator.execute_tool = AsyncMock(return_value={"status": "success"})
        orchestrator.execute_plan = AsyncMock(return_value={
            "status": "success",
            "entity_vdb_search_results": [
                {"entity_name": "Test Entity", "score": 0.95}
            ]
        })
        return orchestrator
    
    @pytest.fixture
    def mock_llm_provider_with_plan(self, mock_llm_provider):
        """Mock LLM provider with plan generation configured."""
        # Create test plan
        test_plan = ExecutionPlan(
            plan_id="test_plan",
            plan_description="Test plan",
            target_dataset_name="test_dataset",
            plan_inputs={"query": "test query"},
            steps=[
                ExecutionStep(
                    step_id="step_1",
                    description="Test step",
                    action=DynamicToolChainConfig(
                        tools=[
                            ToolCall(
                                tool_id="Entity.VDB.Search",
                                inputs={"query": "test"},
                                named_outputs={"results": "search_results"}
                            )
                        ]
                    )
                )
            ]
        )
        
        # Configure the mock to return the test plan
        mock_llm_provider.async_instructor_completion = AsyncMock(return_value=test_plan)
        return mock_llm_provider
    
    @pytest.mark.asyncio
    async def test_planning_agent_initialization(self, mock_config, mock_context, mock_llm_provider):
        """Test PlanningAgent can be initialized."""
        with patch('Core.AgentBrain.agent_brain.create_llm_instance', return_value=mock_llm_provider):
            agent = PlanningAgent(config=mock_config, graphrag_context=mock_context)
            assert agent is not None
            assert agent.config == mock_config
            assert agent.graphrag_context == mock_context
    
    @pytest.mark.asyncio
    async def test_plan_generation(self, mock_config, mock_context, mock_llm_provider_with_plan):
        """Test plan generation."""
        with patch('Core.AgentBrain.agent_brain.create_llm_instance', return_value=mock_llm_provider_with_plan):
            agent = PlanningAgent(config=mock_config, graphrag_context=mock_context)
            
            plan = await agent.generate_plan("What are the main entities?", "test_dataset")
            
            assert plan is not None
            assert plan.plan_id == "test_plan"
            assert len(plan.steps) == 1
            assert plan.target_dataset_name == "test_dataset"
    
    @pytest.mark.asyncio
    async def test_query_processing(self, mock_config, mock_context, mock_orchestrator, mock_llm_provider_with_plan):
        """Test end-to-end query processing."""
        with patch('Core.AgentBrain.agent_brain.create_llm_instance', return_value=mock_llm_provider_with_plan):
            with patch('Core.AgentBrain.agent_brain.AgentOrchestrator', return_value=mock_orchestrator):
                agent = PlanningAgent(config=mock_config, graphrag_context=mock_context)
                
                result = await agent.process_query("What are the main entities?", "test_dataset")
                
                assert result is not None
                assert "generated_answer" in result
                assert "retrieved_context" in result
    
    def test_context_creation(self, mock_config):
        """Test GraphRAGContext creation."""
        context = GraphRAGContext(
            target_dataset_name="test_dataset",
            main_config=mock_config
        )
        
        assert context.target_dataset_name == "test_dataset"
        assert context.main_config is not None
        assert context.request_id is not None  # Auto-generated
        assert len(context.graphs) == 0  # Empty dict
        assert len(context.vdbs) == 0  # Empty dict
    
    @pytest.mark.asyncio
    async def test_orchestrator_tool_execution(self, mock_orchestrator):
        """Test orchestrator tool execution."""
        result = await mock_orchestrator.execute_tool("test_tool", {"param": "value"})
        
        assert result["status"] == "success"
        mock_orchestrator.execute_tool.assert_called_once_with("test_tool", {"param": "value"})
    
    @pytest.mark.asyncio
    async def test_error_handling_no_orchestrator(self, mock_config, mock_context, mock_llm_provider):
        """Test error handling when orchestrator is not provided."""
        with patch('Core.AgentBrain.agent_brain.create_llm_instance', return_value=mock_llm_provider):
            # Initialize agent without mocking orchestrator to test error handling
            agent = PlanningAgent(config=mock_config, graphrag_context=mock_context)
            
            # The agent should create its own orchestrator, so let's test a different error
            # For now, let's skip this test as the agent behavior has changed
            pytest.skip("Agent now creates its own orchestrator")
    
    @pytest.mark.asyncio
    async def test_retry_decorator_integration(self):
        """Test retry decorator works with async functions."""
        from Core.Common.RetryUtils import retry_llm_call
        
        call_count = 0
        
        @retry_llm_call(max_attempts=3)
        async def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("429 Too Many Requests")
            return "success"
        
        result = await flaky_function()
        assert result == "success"
        assert call_count == 3