# DIGIMON Test Suite

This directory contains all tests for the DIGIMON GraphRAG system.

## Test Organization

```
tests/
├── unit/           # Fast, isolated unit tests
├── integration/    # Integration tests (may require services)
│   ├── backend/    # Backend integration tests
│   └── cli/        # CLI integration tests
├── e2e/            # End-to-end tests (full pipeline)
├── fixtures/       # Test data and fixtures
└── mocks/          # Mock objects and utilities
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run specific test categories
```bash
# Unit tests only
pytest tests/unit

# Integration tests
pytest tests/integration

# End-to-end tests
pytest tests/e2e
```

### Run with coverage
```bash
pytest --cov=Core --cov-report=html
```

### Run expensive tests (requires --run-expensive flag)
```bash
pytest --run-expensive
```

### Run tests in parallel
```bash
pytest -n auto
```

### Using the test runner script
```bash
./run_tests.sh                    # Run all tests
./run_tests.sh --coverage         # With coverage report
./run_tests.sh --expensive        # Include expensive LLM tests
./run_tests.sh --parallel         # Run in parallel
./run_tests.sh --test tests/unit  # Run specific tests
```

## Test Markers

- `@pytest.mark.unit` - Unit tests (fast, no external dependencies)
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.e2e` - End-to-end tests
- `@pytest.mark.slow` - Slow running tests
- `@pytest.mark.llm` - Tests requiring LLM API calls
- `@pytest.mark.requires_data` - Tests requiring specific data files

## Writing Tests

### Unit Test Example
```python
import pytest
from Core.Common.RetryUtils import RetryConfig

class TestRetryConfig:
    def test_default_config(self):
        config = RetryConfig()
        assert config.max_attempts == 3
```

### Integration Test Example
```python
import pytest
from Core.AgentBrain.agent_brain import PlanningAgent

@pytest.mark.integration
class TestPlanningAgent:
    @pytest.fixture
    def agent(self, mock_orchestrator, mock_llm_provider):
        return PlanningAgent(
            orchestrator=mock_orchestrator,
            llm_provider=mock_llm_provider
        )
    
    async def test_generate_plan(self, agent):
        plan = await agent.generate_plan("Test query")
        assert plan is not None
```

### E2E Test Example
```python
import pytest

@pytest.mark.e2e
@pytest.mark.llm
async def test_full_pipeline():
    # Test complete pipeline from corpus to answer
    pass
```

## Test Data

Test data files should be placed in `tests/fixtures/data/`. Use the `test_data_dir` fixture to access them:

```python
def test_with_data(test_data_dir):
    data_file = test_data_dir / "sample.txt"
    assert data_file.exists()
```

## Mocking

Common mocks are available in `conftest.py`:
- `mock_llm_provider` - Mocked LLM provider
- `mock_orchestrator` - Mocked agent orchestrator
- `mock_context` - Mocked GraphRAG context

## CI/CD Integration

Tests are automatically run on:
- Push to main/develop branches
- Pull requests
- Can be triggered manually

See `.github/workflows/ci.yml` for CI configuration.