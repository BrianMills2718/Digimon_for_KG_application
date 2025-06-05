# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup
```bash
# Primary setup using conda
conda env create -f environment.yml -n digimon
conda activate digimon

# Alternative using experiment.yml
conda env create -f experiment.yml -n digimon
```

### Configuration
```bash
# Copy and edit main config (required before running)
cp Option/Config2.example.yaml Option/Config2.yaml
# Edit API keys and model settings in Config2.yaml

# Default models (update in Config2.yaml):
# - OpenAI: o4-mini
# - Gemini: gemini-2.0-flash  
# - Claude: claude-sonnet-4-20250514
```

### Core System Operations
```bash
# Build knowledge graph
python main.py build -opt Option/Method/LGraphRAG.yaml -dataset_name MySampleTexts

# Query the system
python main.py query -opt Option/Method/LGraphRAG.yaml -dataset_name MySampleTexts -question "Your question here"

# Run evaluation
python main.py evaluate -opt Option/Method/LGraphRAG.yaml -dataset_name MySampleTexts

# Interactive agent CLI
python digimon_cli.py -c /path/to/corpus -i --react

# Single query mode
python digimon_cli.py -c /path/to/corpus -q "Your question here"
```

### Backend Services
```bash
# Start Flask API server
python api.py

# Start Streamlit frontend
./run_streamlit.sh
# or manually:
streamlit run streamlit_agent_frontend.py --server.port 8502
```

### Testing
```bash
# Run all tests with pytest
pytest -v

# Run specific test categories
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/e2e/ -v

# Run a single test file
pytest tests/integration/test_agent_orchestrator.py -v

# Run with coverage
pytest --cov=Core --cov-report=html

# Common test files:
# - test_discourse_analysis_framework.py - Discourse analysis testing
# - test_agent_orchestrator.py - Agent system tests
# - test_graph_tools.py - Graph construction tests
# - test_corpus_tools.py - Corpus preparation tests
```

### Linting and Code Quality
```bash
# Run linters (if configured)
ruff check .
mypy Core/

# Format code
black Core/
```

## Architecture Overview

DIGIMON is a modular GraphRAG system built around an intelligent agent framework that can autonomously process data from raw text to insights.

### Core Components

**Agent Framework (`Core/AgentBrain/`, `Core/AgentOrchestrator/`, `Core/AgentSchema/`)**
- `AgentBrain`: LLM-driven planning and reasoning engine
- `AgentOrchestrator`: Tool execution and workflow management 
- `AgentSchema`: Pydantic contracts for all agent operations and tool interfaces

**Graph Construction (`Core/Graph/`)**
- Multiple graph types: `ERGraph`, `RKGraph`, `TreeGraph`, `PassageGraph`
- Agent tools available for building all graph types
- Support for custom ontologies via `Config/custom_ontology.json`

**Retrieval System (`Core/AgentTools/`, `Core/Retriever/`)**
- 16+ granular retrieval operators as agent tools
- Entity, relationship, chunk, subgraph, and community-based retrieval
- Vector database integration (Faiss, ColBERT)

**Provider Abstraction (`Core/Provider/`)**
- LiteLLM integration for multiple LLM backends
- Support for OpenAI, Anthropic, Ollama, local models
- Configurable via `Option/Config2.yaml`

### Operational Modes

**Agent Mode (Primary Development Focus)**
- Agent autonomously plans and executes multi-step workflows
- Corpus preparation → graph construction → retrieval → answer synthesis
- ReAct-style reasoning (experimental)

**Direct Pipeline Mode**
- Traditional build/query/evaluate workflow via `main.py`
- Pre-configured methods: LGraphRAG, GGraphRAG, HippoRAG, KGP, etc.

### Configuration Hierarchy

1. **Base Config**: `Option/Config2.yaml` (API keys, models)
2. **Method Configs**: `Option/Method/*.yaml` (algorithm-specific)
3. **Custom Ontology**: `Config/custom_ontology.json` (domain schemas)
4. **Runtime Overrides**: Agent tools can override configs dynamically

### Data Flow

1. **Corpus Preparation**: Raw `.txt` files → `Corpus.json` via agent tool
2. **Graph Construction**: Agent selects graph type and builds knowledge structure
3. **Index Building**: Vector databases for entities, relationships, communities
4. **Query Processing**: Agent composes retrieval strategies and synthesizes answers

### Key Design Patterns

**Tool-Based Architecture**: All operations exposed as Pydantic-validated agent tools with contracts in `Core/AgentSchema/`

**Multi-Graph Support**: System supports 5 different graph types for different use cases

**Provider Agnostic**: LLM and embedding providers abstracted through LiteLLM

**Modular Retrieval**: Granular operators can be chained by agent for complex queries

### Development Notes

**Agent Tool Implementation**: New tools require both Pydantic contracts in `AgentSchema/` and implementations in `AgentTools/`

**Graph Storage**: Uses NetworkX with custom storage backends in `Core/Storage/`

**LLM Integration**: All LLM calls go through `Core/Provider/LiteLLMProvider.py`
- Dynamic token calculation: System automatically uses maximum available tokens based on model limits
- Token counting integrated for cost tracking via `Core/Utils/TokenCounter.py`

**Testing Pattern**: Tools tested both individually and in integrated agent workflows

**Configuration Override**: Agent tools can override default configs for dynamic operation

**ColBERT Dependency Issues**: If you encounter ColBERT/transformers compatibility errors:
- Set `vdb_type: faiss` in method configs instead of `colbert`
- Or add `disable_colbert: true` to your Config2.yaml
- The system will automatically fall back to FAISS for vector indexing
- Note: Existing ColBERT indexes must be rebuilt as FAISS indexes

### Special Analysis Capabilities

**Discourse Analysis Framework**
- Enhanced planner for social media discourse analysis (`Core/AgentTools/discourse_enhanced_planner.py`)
- Supports WHO/SAYS WHAT/TO WHOM/IN WHAT SETTING/WITH WHAT EFFECT analysis paradigm
- Automated interrogative planning for generating research questions
- Mini-ontology generation for focused entity/relationship extraction

**Social Media Analysis**
- Specialized tools in `Core/AgentTools/social_media_dataset_tools.py`
- COVID-19 conspiracy theory dataset included for testing
- Execution framework in `social_media_discourse_executor.py`

### MCP (Model Context Protocol) Integration
- MCP server implementation in `Core/MCP/`
- Blackboard architecture for shared context
- Knowledge sources for dynamic information sharing
- Run MCP server: `./run_mcp_server.sh`