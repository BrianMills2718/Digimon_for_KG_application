# tests/unit/test_dynamic_tool_registry.py

import pytest
from typing import Set
from unittest.mock import MagicMock
from pydantic import BaseModel

from Core.AgentTools.tool_registry import (
    DynamicToolRegistry, 
    ToolCategory, 
    ToolCapability,
    ToolMetadata,
    ToolRegistration
)


class MockToolInput(BaseModel):
    """Mock input model for testing"""
    query: str


class TestDynamicToolRegistry:
    """Test cases for the dynamic tool registry"""
    
    @pytest.fixture
    def registry(self):
        """Create a fresh registry instance"""
        return DynamicToolRegistry()
    
    def test_initialization(self, registry):
        """Test that registry initializes with default tools"""
        # Should have default tools registered
        assert len(registry) > 0
        assert "Entity.VDBSearch" in registry
        assert "graph.BuildERGraph" in registry
        assert "corpus.PrepareFromDirectory" in registry
    
    def test_tool_categorization(self, registry):
        """Test that tools are correctly categorized"""
        # Check specific tool categories
        entity_search = registry.get_tool_metadata("Entity.VDBSearch")
        assert entity_search.category == ToolCategory.READ_ONLY
        
        entity_build = registry.get_tool_metadata("Entity.VDB.Build")
        assert entity_build.category == ToolCategory.WRITE
        
        graph_build = registry.get_tool_metadata("graph.BuildERGraph")
        assert graph_build.category == ToolCategory.BUILD
    
    def test_tool_capabilities(self, registry):
        """Test tool capability indexing"""
        # Find all entity discovery tools
        entity_tools = registry.discover_tools(
            capabilities={ToolCapability.ENTITY_DISCOVERY}
        )
        assert "Entity.VDBSearch" in entity_tools
        assert "Entity.PPR" in entity_tools
        assert "Entity.Onehop" in entity_tools
        
        # Find all graph construction tools
        graph_tools = registry.discover_tools(
            capabilities={ToolCapability.GRAPH_CONSTRUCTION}
        )
        assert all(tid.startswith("graph.Build") for tid in graph_tools)
    
    def test_parallelizable_tools(self, registry):
        """Test identification of parallelizable tools"""
        tool_ids = [
            "Entity.VDBSearch",  # READ_ONLY - parallelizable
            "Entity.PPR",        # READ_ONLY - parallelizable
            "Entity.VDB.Build",  # WRITE - not parallelizable
            "graph.BuildERGraph" # BUILD - not parallelizable
        ]
        
        parallel, sequential = registry.get_parallelizable_tools(tool_ids)
        
        assert "Entity.VDBSearch" in parallel
        assert "Entity.PPR" in parallel
        assert "Entity.VDB.Build" in sequential
        assert "graph.BuildERGraph" in sequential
    
    def test_tool_discovery_by_category(self, registry):
        """Test discovering tools by category"""
        read_only_tools = registry.get_tools_by_category(ToolCategory.READ_ONLY)
        assert len(read_only_tools) > 0
        assert all(
            registry.get_tool_metadata(tid).category == ToolCategory.READ_ONLY 
            for tid in read_only_tools
        )
        
        write_tools = registry.get_tools_by_category(ToolCategory.WRITE)
        assert len(write_tools) > 0
        assert "Entity.VDB.Build" in write_tools
    
    def test_tool_discovery_by_tags(self, registry):
        """Test discovering tools by tags"""
        entity_tools = registry.discover_tools(tags=["entity"])
        assert len(entity_tools) > 0
        assert all("Entity" in tid for tid in entity_tools)
        
        graph_tools = registry.discover_tools(tags=["graph"])
        assert len(graph_tools) > 0
    
    def test_custom_tool_registration(self, registry):
        """Test registering a custom tool"""
        # Create custom tool
        def custom_tool(input_data):
            return {"result": "success"}
        
        metadata = ToolMetadata(
            tool_id="custom.TestTool",
            name="Custom Test Tool",
            description="A custom tool for testing",
            category=ToolCategory.ANALYZE,
            capabilities={ToolCapability.ANALYSIS},
            input_model=MockToolInput,
            tags=["custom", "test"],
            performance_hint="Very fast"
        )
        
        # Register tool
        registry.register_tool(
            tool_id="custom.TestTool",
            function=custom_tool,
            metadata=metadata
        )
        
        # Verify registration
        assert "custom.TestTool" in registry
        assert registry.get_tool("custom.TestTool") is not None
        
        # Check metadata
        retrieved_metadata = registry.get_tool_metadata("custom.TestTool")
        assert retrieved_metadata.name == "Custom Test Tool"
        assert retrieved_metadata.category == ToolCategory.ANALYZE
        assert ToolCapability.ANALYSIS in retrieved_metadata.capabilities
    
    def test_tool_validation_chain(self, registry):
        """Test tool dependency validation"""
        # Create a chain where tool B depends on tool A
        tool_ids = ["Entity.VDBSearch", "Entity.PPR"]
        issues = registry.validate_tool_chain(tool_ids)
        assert len(issues) == 0  # No dependencies, should be valid
        
        # Test with missing tool
        tool_ids_with_missing = ["Entity.VDBSearch", "nonexistent.tool"]
        issues = registry.validate_tool_chain(tool_ids_with_missing)
        assert len(issues) > 0
        assert "not found" in issues[0]
    
    def test_list_all_tools(self, registry):
        """Test listing all tools with metadata"""
        all_tools = registry.list_all_tools()
        
        assert len(all_tools) > 0
        assert all(isinstance(tool, dict) for tool in all_tools)
        
        # Check structure of tool info
        first_tool = all_tools[0]
        assert "tool_id" in first_tool
        assert "name" in first_tool
        assert "description" in first_tool
        assert "category" in first_tool
        assert "capabilities" in first_tool
        assert "parallelizable" in first_tool
    
    def test_backward_compatibility(self, registry):
        """Test backward compatibility with get_tool_function"""
        # Should return tuple of (function, input_model)
        tool_info = registry.get_tool_function("Entity.VDBSearch")
        assert tool_info is not None
        assert len(tool_info) == 2
        assert callable(tool_info[0])
        assert issubclass(tool_info[1], BaseModel)
    
    def test_tool_with_processors(self, registry):
        """Test tool with pre/post processors"""
        # Create tool with processors
        def tool_func(input_data):
            return {"value": input_data.query}
        
        async def pre_processor(params):
            params["query"] = params["query"].upper()
            return params
        
        async def post_processor(output):
            output["processed"] = True
            return output
        
        metadata = ToolMetadata(
            tool_id="custom.ProcessorTool",
            name="Tool with Processors",
            description="Test tool with pre/post processing",
            category=ToolCategory.TRANSFORM,
            capabilities={ToolCapability.ANALYSIS},
            input_model=MockToolInput
        )
        
        registry.register_tool(
            tool_id="custom.ProcessorTool",
            function=tool_func,
            metadata=metadata,
            pre_processor=pre_processor,
            post_processor=post_processor
        )
        
        # Verify registration includes processors
        tool_reg = registry.get_tool("custom.ProcessorTool")
        assert tool_reg.pre_processor is not None
        assert tool_reg.post_processor is not None
    
    def test_tool_metadata_properties(self):
        """Test ToolMetadata properties"""
        metadata = ToolMetadata(
            tool_id="test.tool",
            name="Test Tool",
            description="Test",
            category=ToolCategory.READ_ONLY,
            capabilities={ToolCapability.ANALYSIS},
            input_model=MockToolInput
        )
        
        # Test parallelizable property
        assert metadata.is_parallelizable is True
        
        metadata.category = ToolCategory.WRITE
        assert metadata.is_parallelizable is False
        
        # Test requires_context property
        assert metadata.requires_context is True
        
        metadata.tool_id = "corpus.PrepareFromDirectory"
        assert metadata.requires_context is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])