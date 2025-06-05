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

## Recent Fixes (2025-06-05)

### Critical Issues Resolved

1. **Agent Failure Detection** - Fixed in `Core/AgentBrain/agent_brain.py`
   - Agent now properly detects when tools return `status: "failure"`
   - Previously only checked for `error` field, missing most failures

2. **Path Resolution** - Fixed in `Core/AgentTools/corpus_tools.py`
   - Corpus tool now resolves relative paths under `Data/` directory
   - Handles both absolute and relative paths correctly

3. **Graph Type Naming** - Fixed in `Core/AgentTools/graph_construction_tools.py`
   - Changed "rk_graph" to "rkg_graph" to match factory expectations

4. **Graph Factory Parameters** - Fixed in `Core/Graph/GraphFactory.py`
   - TreeGraph and PassageGraph now receive correct parameters (config, llm, encoder)
   - Removed incorrect data_path and storage parameters

5. **LLM Call Parameters** - Fixed in `Core/Graph/ERGraph.py`
   - Removed unsupported `operation` parameter from `aask()` calls

### Working Example: Russian Troll Tweets Analysis

```bash
# Prepare sample dataset
python create_troll_sample.py  # Creates small sample from larger dataset

# Quick test with agent (now works!)
python digimon_cli.py -c Data/Russian_Troll_Sample -q "Analyze the themes in these Russian troll tweets"

# Direct graph building test
python test_basic_graph.py  # Successfully builds graph with 142 nodes, 95 edges

# Important: Agent needs namespace set properly for graphs to work
# This is now handled automatically in graph construction tools
```

### Known Working Configuration

- **Model**: o4-mini (OpenAI) - configured in Option/Config2.yaml
- **Embeddings**: text-embedding-3-small (OpenAI)
- **Vector DB**: FAISS (ColBERT disabled due to dependency issues)
- **Graph Types**: All types now working (ER, RKG, Tree, Passage)
- **Namespace Handling**: Automatic via ChunkFactory.get_namespace()

### Testing the Fixes

```bash
# Run test suite to verify all fixes
python test_final.py  # Comprehensive test of agent pipeline

# Check specific functionality
python -c "
from Option.Config2 import Config
config = Config.default()
print(f'LLM Model: {config.llm.model}')
print(f'Graph Type: {config.graph.type}')
"
```

## DIGIMON Complete Testing Protocol

### System Capabilities to Test

DIGIMON has the following capabilities that MUST all be verified:

1. **Corpus Preparation** - Convert raw text files to structured corpus
2. **Graph Construction** - Build 5 types: ERGraph, RKGraph, TreeGraph, TreeGraphBalanced, PassageGraph  
3. **Entity Extraction** - Extract and resolve entities from text
4. **Relationship Extraction** - Identify relationships between entities
5. **Vector Database Operations** - Build and search entity/relationship embeddings
6. **Graph Traversal** - One-hop neighbors, Personalized PageRank, subgraph extraction
7. **Text Retrieval** - Get chunks for entities, relationships, and subgraphs
8. **Graph Analysis** - Statistics, centrality, community detection
9. **Visualization** - Generate graph visualizations
10. **Discourse Analysis** - Social media and policy discourse patterns
11. **Multi-step Reasoning** - ReAct-style iterative planning and execution
12. **Memory & Learning** - Pattern recognition and optimization

### Testing Methodology

**For each capability, I will:**

1. **Run specific test** targeting that capability
2. **Examine output** for:
   - Success/failure status of each tool
   - Actual data produced (node counts, entities found, etc.)
   - Error messages and stack traces
   - Performance metrics
3. **Fix any issues** found in:
   - Tool implementations
   - Schema/contract mismatches
   - LLM prompt engineering
   - Data flow between tools
4. **Re-test** until that capability works perfectly
5. **Document** the fix and verify with multiple test cases

### Test Suite

```bash
# Test 1: Basic Corpus → Graph → Query Pipeline
python digimon_cli.py -c Data/Russian_Troll_Sample -q "Build an ER graph and list all entities"

# Test 2: All Graph Types
for graph_type in ER RK Tree TreeBalanced Passage; do
  python digimon_cli.py -c Data/Russian_Troll_Sample -q "Build a $graph_type graph and show statistics"
done

# Test 3: Entity Operations
python digimon_cli.py -c Data/Russian_Troll_Sample -q "Find entities about 'Trump' and their relationships"

# Test 4: Graph Analysis
python digimon_cli.py -c Data/Russian_Troll_Sample -q "Run PageRank and find most central entities"

# Test 5: Discourse Analysis
python digimon_cli.py -c Data/Russian_Troll_Sample -q "Analyze discourse patterns in these tweets"

# Test 6: Complex Multi-step
python digimon_cli.py -c Data/Russian_Troll_Sample -q "Build all graph types, find key entities, analyze their relationships, and summarize the network structure"
```

### Verification Checklist

For each test, verify:

- [ ] Corpus preparation succeeds (creates Corpus.json)
- [ ] Graph building completes (reports node/edge counts)
- [ ] Entity extraction finds actual entities (not empty)
- [ ] Relationships are extracted (with descriptions)
- [ ] VDB operations complete (reports indexed count)
- [ ] Search operations return results
- [ ] Graph traversal returns neighbors/subgraphs
- [ ] Analysis produces metrics
- [ ] Final answer contains real insights from data

### Iteration Protocol

1. **Run test** → Observe failure point
2. **Debug** → Add logging to identify exact issue
3. **Fix code** → Update tool/schema/prompt
4. **Test fix** → Verify specific issue resolved
5. **Full retest** → Ensure no regressions
6. **Repeat** until 100% success rate

**I will NOT stop until:**
- Every single tool works correctly
- All graph types build successfully  
- Entity/relationship extraction produces real data
- Complex queries complete end-to-end
- The system can answer analytical questions with actual insights from the data

**Current Status:** System is broken - cannot even complete basic ER graph building due to orchestrator output key mismatches. This MUST be fixed first.

## Recent Fixes Applied (2025-06-05 Update)

### Critical Fixes Implemented

1. **Orchestrator State Preservation (FIXED)**
   - Issue: ReAct mode was losing state between iterations
   - Fix: Modified `Core/AgentOrchestrator/orchestrator.py` to preserve `step_outputs` between plan executions
   - Result: Steps can now reference outputs from previous plans in ReAct mode

2. **Ontology Generator (FIXED)**  
   - Issue: `aask()` was called with unsupported 'operation' parameter
   - Fix: Removed `operation="ontology_generation"` from call in `Core/Graph/ontology_generator.py`
   - Result: Ontology generation no longer crashes

3. **Corpus Path Resolution (WORKING)**
   - Issue: ChunkFactory looks for corpus in multiple locations
   - Current behavior: Checks both `results/{dataset}/Corpus.json` and `Data/{dataset}/Corpus.json`
   - Result: Corpus preparation tool creates files in correct location

### Synthetic Test Data Available

Created comprehensive test datasets for all capabilities:

```bash
# Create synthetic test data
python create_synthetic_test_data.py

# Available datasets:
# - Synthetic_Test: 4 documents covering AI, climate, politics, space
# - Discourse_Test: UBI debate with clear discourse patterns  
# - Community_Test: Startup ecosystem with community structures
```

### Current Testing Status

1. ✓ Orchestrator state preservation - Verified working
2. ✓ ReAct mode execution - Completes 7 iterations successfully
3. ✓ Corpus preparation - Creates Corpus.json correctly
4. ~ Graph building - Builds but agent doesn't recognize success
5. ? Entity/relationship extraction - Needs verification
6. ? VDB operations - Not yet tested
7. ? Complex queries - Not yet tested

### Final Status (2025-06-05) - CORE FIXES COMPLETE

**5-Stage Fix Protocol Successfully Applied**

All critical orchestrator and agent issues have been resolved:

**Stage 1: ✓ Orchestrator Output Storage** 
- Fixed: Orchestrator now preserves `_status` and `_message` fields alongside named outputs
- Impact: Agent can now detect tool failures properly

**Stage 2: ✓ Agent Failure Detection**
- Fixed: Agent's `_react_observe` method checks both 'status' and '_status' fields
- Impact: Failed steps are properly identified and skipped in ReAct mode

**Stage 3: ✓ Tool-Specific Fixes**
- Fixed: EntityRetriever handles missing `entities_to_relationships` attribute gracefully
- Fixed: Identified non-existent tools (Chunk.GetTextForClusters, report.Generate) as LLM hallucinations
- Impact: PPR calculations work without crashes

**Stage 4: ✓ Field Naming & Tool Validation**
- Fixed: Added tool validation to warn about non-existent tools
- Fixed: Updated agent prompts to explicitly list invalid tool names
- Fixed: Clarified named_output alias usage in documentation
- Impact: Agent generates valid plans with correct field references

**Stage 5: ✓ Dataset Resolution**
- Verified: ChunkFactory correctly resolves dataset names and paths
- Verified: Corpus tool handles relative/absolute paths properly
- Verified: Graph namespace extraction from IDs works correctly
- Impact: No code changes needed - already working correctly

**Current System Status**

Core functionality restored:
- ✓ Corpus preparation from text files
- ✓ All graph types build successfully (ER, RK, Tree, TreeBalanced, Passage)
- ✓ Graph instances properly registered in context
- ✓ Entity VDB operations work when VDB is built first
- ✓ Tool failure detection and handling
- ✓ ReAct mode preserves state between iterations

**Known Remaining Issues**

1. **VDB Auto-Building**: Agent doesn't always realize it needs to build VDB before searching
   - Workaround: Explicitly mention "build entity VDB" in queries
   
2. **Empty Results**: Some queries return empty results due to:
   - Small test datasets with limited entities
   - Need for better entity extraction prompts
   
3. **Token Limits**: Complex multi-step plans may exceed token limits
   - Workaround: Break complex queries into smaller parts

**Recommended Usage Pattern**

```bash
# Step 1: Prepare corpus (if not already done)
python digimon_cli.py -c Data/YourDataset -q "Prepare the corpus"

# Step 2: Build graph and VDB explicitly
python digimon_cli.py -c Data/YourDataset -q "Build an ER graph and create entity VDB"

# Step 3: Run analysis queries
python digimon_cli.py -c Data/YourDataset -q "Find main themes and entities"
```

**Test Commands That Work**

```bash
# Test basic graph building
python test_basic_graph.py

# Test orchestrator fixes
python test_orchestrator_state.py

# Test agent fixes  
python test_stage4_simple.py

# Full system test
python digimon_cli.py -c Data/Russian_Troll_Sample -q "Analyze the content"
```
# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.

      
      IMPORTANT: this context may or may not be relevant to your tasks. You should not respond to this context or otherwise consider it in your response unless it is highly relevant to your task. Most of the time, it is not relevant.