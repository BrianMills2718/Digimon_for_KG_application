"""
Pytest configuration and fixtures for DIGIMON tests.
"""

import os
import sys
import asyncio
import pytest
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import Mock, AsyncMock, patch

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from Config.LLMConfig import LLMConfig
from Core.Provider.LiteLLMProvider import LiteLLMProvider
from Core.AgentSchema.context import GraphRAGContext
from Core.AgentOrchestrator.orchestrator import AgentOrchestrator


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_data_dir():
    """Path to test data directory."""
    return Path(__file__).parent / "fixtures" / "data"


@pytest.fixture
def mock_llm_config():
    """Mock LLM configuration for testing."""
    return LLMConfig(
        api_type="litellm",
        model="openai/gpt-3.5-turbo",
        api_key="test-key",
        base_url="http://localhost:11434",
        temperature=0.0,
        max_token=1000,
        calc_usage=False
    )


@pytest.fixture
def mock_llm_provider(mock_llm_config):
    """Mock LLM provider that doesn't make real API calls."""
    with patch('Core.Provider.LiteLLMProvider.LiteLLMProvider') as MockProvider:
        provider = MockProvider(mock_llm_config)
        
        # Mock common methods
        provider.acompletion = AsyncMock(return_value=Mock(
            choices=[Mock(message=Mock(content="Test response"))]
        ))
        provider.async_instructor_completion = AsyncMock()
        provider.model = "openai/gpt-3.5-turbo"
        provider.max_tokens = 1000
        
        yield provider


@pytest.fixture
def mock_context():
    """Mock GraphRAG context for testing."""
    return GraphRAGContext(
        corpus_name="test_corpus",
        dataset_name="test_dataset",
        graph_id="test_graph",
        vdb_collection_name="test_vdb"
    )


@pytest.fixture
def mock_orchestrator(mock_context):
    """Mock agent orchestrator for testing."""
    orchestrator = Mock(spec=AgentOrchestrator)
    orchestrator.context = mock_context
    orchestrator.execute_plan = AsyncMock(return_value={"status": "success"})
    orchestrator.execute_tool = AsyncMock(return_value={"result": "test"})
    return orchestrator


@pytest.fixture
def sample_corpus_json():
    """Sample corpus JSON structure for testing."""
    return {
        "corpus_name": "test_corpus",
        "documents": [
            {
                "doc_id": "doc1",
                "title": "Test Document 1",
                "content": "This is a test document about artificial intelligence.",
                "metadata": {"source": "test"}
            },
            {
                "doc_id": "doc2", 
                "title": "Test Document 2",
                "content": "This document discusses machine learning techniques.",
                "metadata": {"source": "test"}
            }
        ]
    }


@pytest.fixture
def sample_execution_plan():
    """Sample execution plan for testing."""
    return {
        "plan_id": "test_plan",
        "plan_description": "Test execution plan",
        "target_dataset_name": "test_dataset",
        "plan_inputs": {"query": "test query"},
        "steps": [
            {
                "step_id": "step_1",
                "description": "Prepare corpus",
                "action": {
                    "tools": [{
                        "tool_id": "corpus.PrepareFromDirectory",
                        "inputs": {
                            "input_directory_path": "Data/test",
                            "output_directory_path": "results/test",
                            "target_corpus_name": "test"
                        },
                        "named_outputs": {
                            "corpus_path": "corpus_json_path"
                        }
                    }]
                }
            }
        ]
    }


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Set up test environment variables."""
    monkeypatch.setenv("DIGIMON_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("DIGIMON_LOG_STRUCTURED", "false")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


@pytest.fixture
def cleanup_test_files():
    """Cleanup test files after test execution."""
    test_files = []
    
    def register_file(filepath):
        test_files.append(filepath)
    
    yield register_file
    
    # Cleanup
    for filepath in test_files:
        if Path(filepath).exists():
            if Path(filepath).is_file():
                Path(filepath).unlink()
            elif Path(filepath).is_dir():
                import shutil
                shutil.rmtree(filepath)


# Markers for expensive operations
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "llm: mark test as requiring LLM API calls (expensive)"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_data: mark test as requiring specific data files"
    )


# Skip expensive tests by default
def pytest_collection_modifyitems(config, items):
    """Modify test collection to skip expensive tests unless explicitly requested."""
    if not config.getoption("--run-expensive"):
        skip_expensive = pytest.mark.skip(reason="need --run-expensive option to run")
        for item in items:
            if "llm" in item.keywords:
                item.add_marker(skip_expensive)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-expensive",
        action="store_true",
        default=False,
        help="run expensive tests that require LLM API calls"
    )