# tests/integration/test_async_streaming_integration.py

import pytest
import asyncio
from typing import List
import tempfile
import os

from Core.AgentOrchestrator.async_streaming_orchestrator import (
    AsyncStreamingOrchestrator, 
    StreamingUpdate,
    UpdateType
)
from Core.AgentBrain.agent_brain import PlanningAgent
from Core.AgentSchema.context import GraphRAGContext
from Core.AgentSchema.plan import ExecutionPlan
from Option.Config2 import load_config
from Core.Provider.LiteLLMProvider import LiteLLMProvider
from Core.Chunk.ChunkFactory import ChunkFactory
from Core.Index.EmbeddingFactory import EmbeddingFactory


@pytest.mark.integration
class TestAsyncStreamingIntegration:
    """Integration tests for async streaming orchestrator with real components"""
    
    @pytest.fixture
    def config(self):
        """Load test configuration"""
        return load_config("Option/Config2.yaml")
    
    @pytest.fixture
    def llm_instance(self, config):
        """Create LLM instance"""
        return LiteLLMProvider(config.llm)
    
    @pytest.fixture
    def encoder_instance(self, config):
        """Create encoder instance"""
        factory = EmbeddingFactory()
        return factory.get_embedding(config.embedding)
    
    @pytest.fixture
    def chunk_factory(self, config):
        """Create chunk factory"""
        return ChunkFactory(config.chunk)
    
    @pytest.fixture
    def test_corpus_dir(self):
        """Create a temporary directory with test text files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            test_files = {
                "doc1.txt": "The capital of France is Paris. Paris is known for the Eiffel Tower.",
                "doc2.txt": "London is the capital of England. Big Ben is located in London.",
                "doc3.txt": "Tokyo is the capital of Japan. Mount Fuji is near Tokyo."
            }
            
            for filename, content in test_files.items():
                with open(os.path.join(tmpdir, filename), 'w') as f:
                    f.write(content)
            
            yield tmpdir
    
    @pytest.fixture
    def graphrag_context(self, config):
        """Create GraphRAG context"""
        return GraphRAGContext(
            working_directory=config.working_dir,
            dataset_name="test_streaming"
        )
    
    @pytest.fixture
    def streaming_orchestrator(self, config, llm_instance, encoder_instance, chunk_factory, graphrag_context):
        """Create streaming orchestrator"""
        return AsyncStreamingOrchestrator(
            main_config=config,
            llm_instance=llm_instance,
            encoder_instance=encoder_instance,
            chunk_factory=chunk_factory,
            graphrag_context=graphrag_context
        )
    
    @pytest.mark.asyncio
    async def test_corpus_preparation_streaming(self, streaming_orchestrator, test_corpus_dir):
        """Test corpus preparation with streaming updates"""
        # Create plan for corpus preparation
        plan = ExecutionPlan(
            plan_id="corpus_prep_test",
            plan_description="Prepare corpus from directory",
            target_dataset_name="test_streaming",
            steps=[{
                "step_id": "prepare_corpus",
                "description": "Prepare corpus from test directory",
                "action": {
                    "tools": [{
                        "tool_id": "corpus.PrepareFromDirectory",
                        "inputs": {
                            "directory_path": test_corpus_dir,
                            "dataset_name": "test_streaming"
                        },
                        "named_outputs": {
                            "corpus_path": "corpus_json_path"
                        }
                    }]
                }
            }],
            plan_inputs={}
        )
        
        # Collect streaming updates
        updates: List[StreamingUpdate] = []
        async for update in streaming_orchestrator.execute_plan_stream(plan):
            updates.append(update)
            print(f"Update: {update.type.value} - {update.description}")
        
        # Verify streaming updates
        assert any(u.type == UpdateType.PLAN_START for u in updates)
        assert any(u.type == UpdateType.TOOL_START and u.tool_id == "corpus.PrepareFromDirectory" for u in updates)
        assert any(u.type == UpdateType.TOOL_COMPLETE for u in updates)
        assert any(u.type == UpdateType.PLAN_COMPLETE for u in updates)
        
        # Verify corpus was created
        assert "prepare_corpus" in streaming_orchestrator.step_outputs
        corpus_path = streaming_orchestrator.step_outputs["prepare_corpus"].get("corpus_path")
        assert corpus_path is not None
        assert os.path.exists(corpus_path)
    
    @pytest.mark.asyncio
    async def test_parallel_entity_operations_streaming(self, streaming_orchestrator, monkeypatch):
        """Test parallel execution of entity operations with streaming"""
        # Mock entity operations to avoid needing real graph/VDB
        execution_times = {}
        
        async def mock_entity_vdb_search(inputs, context):
            import time
            start = time.time()
            await asyncio.sleep(0.1)  # Simulate work
            execution_times["vdb_search"] = (start, time.time())
            return {"similar_entities": [{"node_id": "e1", "entity_name": "Entity1", "score": 0.9}]}
        
        async def mock_entity_ppr(inputs, context):
            import time
            start = time.time()
            await asyncio.sleep(0.05)  # Simulate work
            execution_times["ppr"] = (start, time.time())
            return {"ranked_entities": [{"entity_id": "e1", "score": 0.8}]}
        
        # Patch the tool functions
        import Core.AgentTools.entity_tools
        monkeypatch.setattr(Core.AgentTools.entity_tools, "entity_vdb_search_tool", mock_entity_vdb_search)
        monkeypatch.setattr(Core.AgentTools.entity_tools, "entity_ppr_tool", mock_entity_ppr)
        
        # Create plan with parallel entity operations
        plan = ExecutionPlan(
            plan_id="parallel_entity_test",
            plan_description="Test parallel entity operations",
            target_dataset_name="test_streaming",
            steps=[{
                "step_id": "parallel_search",
                "description": "Parallel entity searches",
                "action": {
                    "tools": [
                        {
                            "tool_id": "Entity.VDBSearch",
                            "inputs": {"query": "test query"},
                            "named_outputs": {"vdb_results": "similar_entities"}
                        },
                        {
                            "tool_id": "Entity.PPR",
                            "inputs": {"seed_entity_ids": ["e1"]},
                            "named_outputs": {"ppr_results": "ranked_entities"}
                        }
                    ]
                }
            }],
            plan_inputs={}
        )
        
        # Execute with streaming
        tool_updates = []
        async for update in streaming_orchestrator.execute_plan_stream(plan):
            if update.type in [UpdateType.TOOL_START, UpdateType.TOOL_COMPLETE]:
                tool_updates.append((update.type, update.tool_id))
        
        # Verify parallel execution - both tools should start before either completes
        tool_starts = [(t, id) for t, id in tool_updates if t == UpdateType.TOOL_START]
        tool_completes = [(t, id) for t, id in tool_updates if t == UpdateType.TOOL_COMPLETE]
        
        assert len(tool_starts) == 2
        assert len(tool_completes) == 2
        
        # Check execution times overlap (parallel execution)
        vdb_start, vdb_end = execution_times["vdb_search"]
        ppr_start, ppr_end = execution_times["ppr"]
        
        # PPR should start before VDB finishes (parallel)
        assert ppr_start < vdb_end
    
    @pytest.mark.asyncio
    async def test_streaming_with_planning_agent(self, config, llm_instance, encoder_instance, chunk_factory, graphrag_context):
        """Test streaming orchestrator with planning agent generated plan"""
        # Create planning agent
        planning_agent = PlanningAgent(llm_instance)
        
        # Generate a simple plan
        query = "What are the main entities in the corpus?"
        
        # For this test, we'll create a simple predefined plan
        # In real usage, this would come from planning_agent.generate_plan(query)
        plan = ExecutionPlan(
            plan_id="agent_generated_plan",
            plan_description="Find main entities in corpus",
            target_dataset_name="test_streaming",
            steps=[{
                "step_id": "analyze_graph",
                "description": "Analyze graph structure",
                "action": {
                    "tools": [{
                        "tool_id": "graph.Analyze",
                        "inputs": {
                            "graph_id": "test_graph",
                            "analysis_type": "entity_statistics"
                        }
                    }]
                }
            }],
            plan_inputs={"query": query}
        )
        
        # Create streaming orchestrator
        streaming_orchestrator = AsyncStreamingOrchestrator(
            main_config=config,
            llm_instance=llm_instance,
            encoder_instance=encoder_instance,
            chunk_factory=chunk_factory,
            graphrag_context=graphrag_context
        )
        
        # Execute with streaming and collect updates
        updates = []
        async for update in streaming_orchestrator.execute_plan_stream(plan):
            updates.append(update)
            
            # Could send updates to websocket, UI, etc. here
            if update.type == UpdateType.STEP_PROGRESS:
                print(f"Progress: {update.progress * 100:.1f}% - {update.description}")
        
        # Verify we got meaningful updates
        assert len(updates) > 0
        assert updates[0].type == UpdateType.PLAN_START
        assert updates[-1].type == UpdateType.PLAN_COMPLETE


@pytest.mark.asyncio
async def test_streaming_performance_benchmark():
    """Benchmark streaming overhead"""
    import time
    from unittest.mock import MagicMock, AsyncMock
    
    # Create minimal orchestrator
    orchestrator = AsyncStreamingOrchestrator(
        main_config=MagicMock(),
        llm_instance=MagicMock(),
        encoder_instance=MagicMock(),
        chunk_factory=MagicMock(),
        graphrag_context=MagicMock()
    )
    
    # Mock fast tool execution
    async def fast_tool(*args, **kwargs):
        await asyncio.sleep(0.001)  # 1ms
        return {"result": "success"}, None
    
    orchestrator._execute_tool_async = AsyncMock(side_effect=fast_tool)
    
    # Create plan with many tools
    num_tools = 50
    plan = ExecutionPlan(
        plan_id="perf_test",
        plan_description="Performance test",
        target_dataset_name="test",
        steps=[{
            "step_id": "step1",
            "description": "Many tools",
            "action": {
                "tools": [
                    {"tool_id": "Entity.VDBSearch", "inputs": {"query": f"q{i}"}}
                    for i in range(num_tools)
                ]
            }
        }],
        plan_inputs={}
    )
    
    # Measure streaming overhead
    start = time.time()
    update_count = 0
    
    async for update in orchestrator.execute_plan_stream(plan):
        update_count += 1
    
    elapsed = time.time() - start
    
    print(f"\nPerformance: {num_tools} tools in {elapsed:.3f}s")
    print(f"Updates generated: {update_count}")
    print(f"Overhead per tool: {(elapsed - num_tools * 0.001) / num_tools * 1000:.1f}ms")
    
    # Streaming should add minimal overhead
    assert elapsed < 2.0  # Should complete in under 2 seconds
    assert update_count > num_tools  # At least one update per tool