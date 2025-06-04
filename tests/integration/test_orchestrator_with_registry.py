# tests/integration/test_orchestrator_with_registry.py

import pytest
import asyncio
from typing import List
from unittest.mock import MagicMock, AsyncMock, patch

from Core.AgentOrchestrator.async_streaming_orchestrator_v2 import AsyncStreamingOrchestrator, UpdateType
from Core.AgentTools.tool_registry import DynamicToolRegistry, ToolCategory, ToolCapability, ToolMetadata
from Core.AgentSchema.plan import ExecutionPlan, ExecutionStep, ToolCall, DynamicToolChainConfig
from pydantic import BaseModel


class CustomToolInput(BaseModel):
    """Custom tool input for testing"""
    data: str


class CustomToolOutput(BaseModel):
    """Custom tool output for testing"""
    result: str
    processed: bool = False


@pytest.mark.integration
class TestOrchestratorWithRegistry:
    """Test async streaming orchestrator with dynamic tool registry"""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies"""
        config = MagicMock()
        llm = MagicMock()
        encoder = MagicMock()
        chunk_factory = MagicMock()
        chunk_factory.get_namespace.return_value = "test_namespace"
        context = MagicMock()
        
        return {
            "main_config": config,
            "llm_instance": llm,
            "encoder_instance": encoder,
            "chunk_factory": chunk_factory,
            "graphrag_context": context
        }
    
    @pytest.fixture
    def orchestrator(self, mock_dependencies):
        """Create orchestrator with dynamic registry"""
        return AsyncStreamingOrchestrator(**mock_dependencies)
    
    @pytest.mark.asyncio
    async def test_tool_discovery_in_orchestrator(self, orchestrator):
        """Test that orchestrator can discover tools by capability"""
        # Discover entity discovery tools
        entity_tools = orchestrator.discover_tools_for_capability("entity_discovery")
        assert len(entity_tools) > 0
        assert "Entity.VDBSearch" in entity_tools
        assert "Entity.PPR" in entity_tools
        
        # Get tool info
        tool_info = orchestrator.get_tool_info("Entity.VDBSearch")
        assert tool_info is not None
        assert tool_info["name"] == "Entity Vector Database Search"
        assert tool_info["parallelizable"] is True
    
    @pytest.mark.asyncio
    async def test_automatic_parallel_categorization(self, orchestrator):
        """Test that registry automatically categorizes tools for parallel execution"""
        # Create plan with mixed tool types
        plan = ExecutionPlan(
            plan_id="test_parallel_categorization",
            plan_description="Test automatic categorization",
            target_dataset_name="test",
            steps=[
                ExecutionStep(
                    step_id="mixed_tools",
                    description="Mixed tool types",
                    action=DynamicToolChainConfig(
                        tools=[
                            ToolCall(tool_id="Entity.VDBSearch", inputs={"query": "test"}),
                            ToolCall(tool_id="Entity.PPR", inputs={"seed_entity_ids": ["e1"]}),
                            ToolCall(tool_id="Entity.VDB.Build", inputs={"graph_id": "test"}),
                            ToolCall(tool_id="Chunk.GetTextForEntities", inputs={"entity_ids": ["e1"]}),
                        ]
                    )
                )
            ],
            plan_inputs={}
        )
        
        # Track execution order
        execution_log = []
        
        async def mock_execute(tool_call, plan_inputs):
            execution_log.append(f"{tool_call.tool_id}_start")
            await asyncio.sleep(0.01)  # Small delay
            execution_log.append(f"{tool_call.tool_id}_end")
            return MagicMock(), None
        
        with patch.object(orchestrator, '_execute_tool_async', side_effect=mock_execute):
            updates = []
            async for update in orchestrator.execute_plan_stream(plan):
                updates.append(update)
                if update.type == UpdateType.STEP_PROGRESS:
                    print(f"Progress: {update.description}")
        
        # Verify parallel tools executed together
        parallel_tools = ["Entity.VDBSearch", "Entity.PPR", "Chunk.GetTextForEntities"]
        parallel_starts = [i for i, log in enumerate(execution_log) 
                          if any(f"{tool}_start" in log for tool in parallel_tools)]
        parallel_ends = [i for i, log in enumerate(execution_log) 
                        if any(f"{tool}_end" in log for tool in parallel_tools)]
        
        # All parallel tools should start before any finish
        assert max(parallel_starts) < min(parallel_ends)
        
        # Sequential tool (VDB.Build) should run after parallel tools
        vdb_build_start = execution_log.index("Entity.VDB.Build_start")
        assert vdb_build_start > max(parallel_ends)
    
    @pytest.mark.asyncio
    async def test_custom_tool_registration_and_execution(self, orchestrator):
        """Test registering and executing a custom tool"""
        # Define custom tool
        async def custom_analysis_tool(input_data: CustomToolInput, context):
            await asyncio.sleep(0.01)  # Simulate work
            return CustomToolOutput(result=f"Analyzed: {input_data.data}", processed=True)
        
        # Register custom tool
        metadata = ToolMetadata(
            tool_id="custom.AnalysisTool",
            name="Custom Analysis Tool",
            description="Performs custom analysis",
            category=ToolCategory.ANALYZE,
            capabilities={ToolCapability.ANALYSIS},
            input_model=CustomToolInput,
            output_model=CustomToolOutput,
            tags=["custom", "analysis"],
            performance_hint="Very fast analysis"
        )
        
        orchestrator.tool_registry.register_tool(
            tool_id="custom.AnalysisTool",
            function=custom_analysis_tool,
            metadata=metadata
        )
        
        # Create plan using custom tool
        plan = ExecutionPlan(
            plan_id="test_custom_tool",
            plan_description="Test custom tool execution",
            target_dataset_name="test",
            steps=[
                ExecutionStep(
                    step_id="analyze",
                    description="Custom analysis",
                    action=DynamicToolChainConfig(
                        tools=[
                            ToolCall(
                                tool_id="custom.AnalysisTool",
                                inputs={"data": "test data"},
                                named_outputs={"analysis_result": "result"}
                            )
                        ]
                    )
                )
            ],
            plan_inputs={}
        )
        
        # Execute plan
        updates = []
        async for update in orchestrator.execute_plan_stream(plan):
            updates.append(update)
        
        # Verify execution
        tool_complete_updates = [u for u in updates if u.type == UpdateType.TOOL_COMPLETE]
        assert len(tool_complete_updates) == 1
        assert tool_complete_updates[0].tool_id == "custom.AnalysisTool"
        
        # Check outputs
        assert "analyze" in orchestrator.step_outputs
        assert "analysis_result" in orchestrator.step_outputs["analyze"]
    
    @pytest.mark.asyncio
    async def test_tool_metadata_in_updates(self, orchestrator):
        """Test that tool metadata is included in streaming updates"""
        plan = ExecutionPlan(
            plan_id="test_metadata",
            plan_description="Test metadata in updates",
            target_dataset_name="test",
            steps=[
                ExecutionStep(
                    step_id="search",
                    description="Entity search",
                    action=DynamicToolChainConfig(
                        tools=[
                            ToolCall(tool_id="Entity.VDBSearch", inputs={"query": "test"})
                        ]
                    )
                )
            ],
            plan_inputs={}
        )
        
        # Mock tool execution
        with patch.object(orchestrator, '_execute_tool_async', return_value=(MagicMock(), None)):
            tool_start_update = None
            async for update in orchestrator.execute_plan_stream(plan):
                if update.type == UpdateType.TOOL_START:
                    tool_start_update = update
                    break
            
            # Verify metadata in update
            assert tool_start_update is not None
            assert tool_start_update.data is not None
            assert tool_start_update.data["category"] == "read_only"
            assert "entity_discovery" in tool_start_update.data["capabilities"]
    
    @pytest.mark.asyncio
    async def test_tool_with_processors(self, orchestrator):
        """Test tool with pre and post processors"""
        processed_input = None
        processed_output = None
        
        async def tool_func(input_data, context):
            return {"value": input_data.data}
        
        async def pre_processor(params):
            nonlocal processed_input
            processed_input = params.copy()
            params["data"] = params["data"].upper()
            return params
        
        async def post_processor(output):
            nonlocal processed_output
            processed_output = output.copy()
            output["post_processed"] = True
            return output
        
        # Register tool with processors
        metadata = ToolMetadata(
            tool_id="custom.ProcessorTool",
            name="Tool with Processors",
            description="Test processors",
            category=ToolCategory.TRANSFORM,
            capabilities={ToolCapability.ANALYSIS},
            input_model=CustomToolInput
        )
        
        orchestrator.tool_registry.register_tool(
            tool_id="custom.ProcessorTool",
            function=tool_func,
            metadata=metadata,
            pre_processor=pre_processor,
            post_processor=post_processor
        )
        
        # Execute
        plan = ExecutionPlan(
            plan_id="test_processors",
            plan_description="Test pre/post processors",
            target_dataset_name="test",
            steps=[
                ExecutionStep(
                    step_id="process",
                    description="Process with tool",
                    action=DynamicToolChainConfig(
                        tools=[
                            ToolCall(
                                tool_id="custom.ProcessorTool",
                                inputs={"data": "test"}
                            )
                        ]
                    )
                )
            ],
            plan_inputs={}
        )
        
        async for _ in orchestrator.execute_plan_stream(plan):
            pass
        
        # Verify processors were called
        assert processed_input is not None
        assert processed_input["data"] == "test"  # Original input
        
        assert processed_output is not None
        assert "post_processed" in processed_output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])