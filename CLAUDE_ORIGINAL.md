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

## Critical Issues & Staged Fix Protocol (2025-06-05)

### Current Critical Issues

Based on latest testing with Social_Discourse_Test dataset:

1. **Entity Extraction Format Error**
   - ER Graph builder receives malformed entity data (dict instead of string for entity_name)
   - Error: `TypeError: unhashable type: 'dict'`
   - Root cause: LLM returning `entity_name={'text': 'SOCIAL NETWORK ACTOR PROFILES', 'type': 'TextSegment'}`

2. **Missing Tool Implementations**
   - Agent references non-existent tools: `vector_db.CreateIndex`, `vector_db.QueryIndex`, `graph.GetClusters`
   - These appear to be LLM hallucinations

3. **Configuration Attribute Errors**
   - PassageGraph expects `config.summary_max_tokens` but Config object lacks this attribute
   - Fixed with getattr default, but more config issues may exist

4. **Corpus Path Resolution**
   - ChunkFactory expects corpus at `results/{dataset}/Corpus.json`
   - Corpus tool creates at `results/{dataset}/corpus/Corpus.json`
   - Mismatch causes "No input chunks found" errors

5. **Graph Registration Failures**
   - Successfully built graphs not registered in GraphRAGContext
   - Subsequent tools can't find graphs even when built

## Staged Fix Protocol

### Stage 1: Entity Extraction Format Fix
**Goal**: Ensure entity extraction returns proper string entity names, not dicts

**Test Criteria**:
```python
# test_stage1_entity_extraction.py
# MUST verify:
# 1. Entity names are strings, not dicts
# 2. ER graph builds successfully with >0 nodes and edges
# 3. Can retrieve entity data from built graph

# EVIDENCE REQUIRED:
# - entity_name: <string value> (NOT dict)
# - node_count: <integer > 0>
# - edge_count: <integer > 0>
# - sample_entity: <actual entity name and description>

# STATUS: [ ] NOT STARTED
# EVIDENCE: 
# COMMIT: 
```

**Implementation**:
1. Check ERGraph entity extraction prompts
2. Add validation in entity extraction to ensure string names
3. Add type checking before graph construction

### Stage 2: Tool Hallucination Prevention
**Goal**: Prevent agent from using non-existent tools

**Test Criteria**:
```python
# test_stage2_tool_validation.py
# MUST verify:
# 1. Agent only uses tools from registered tool list
# 2. No attempts to call non-existent tools
# 3. Agent adapts plan when tools not available

# EVIDENCE REQUIRED:
# - registered_tools: [list of actual tool IDs]
# - plan_tools: [list of tools in generated plan]
# - validation: ALL plan_tools IN registered_tools
# - no_errors: No "Tool ID 'X' not found" errors

# STATUS: [ ] NOT STARTED
# EVIDENCE:
# COMMIT:
```

**Implementation**:
1. Update agent prompts to include exact tool list
2. Add tool validation in plan generation
3. Add fallback strategies for missing tools

### Stage 3: Corpus Path Standardization
**Goal**: Ensure corpus files are found regardless of creation method

**Test Criteria**:
```python
# test_stage3_corpus_paths.py
# MUST verify:
# 1. Corpus created by tool is found by ChunkFactory
# 2. Graphs can load chunks successfully
# 3. No "Corpus file not found" errors

# EVIDENCE REQUIRED:
# - corpus_created_at: <path where corpus tool creates file>
# - corpus_expected_at: <path where ChunkFactory looks>
# - chunks_loaded: <integer > 0>
# - graph_built: success with chunks

# STATUS: [ ] NOT STARTED
# EVIDENCE:
# COMMIT:
```

**Implementation**:
1. Standardize corpus output location
2. Update ChunkFactory to check multiple locations
3. Add corpus path resolution logic

### Stage 4: Graph Registration & Context Management
**Goal**: Ensure built graphs are accessible to subsequent tools

**Test Criteria**:
```python
# test_stage4_graph_registration.py
# MUST verify:
# 1. Built graphs appear in GraphRAGContext
# 2. Subsequent tools can access graphs
# 3. VDB build succeeds using registered graph

# EVIDENCE REQUIRED:
# - graph_built: <graph_id>
# - graphs_in_context: [list containing graph_id]
# - vdb_built_from_graph: success
# - entities_indexed: <integer > 0>

# STATUS: [ ] NOT STARTED
# EVIDENCE:
# COMMIT:
```

**Implementation**:
1. Fix graph registration in orchestrator
2. Ensure GraphRAGContext persists between steps
3. Add graph instance passing between tools

### Stage 5: End-to-End Query Success
**Goal**: Complete full query pipeline successfully

**Test Criteria**:
```python
# test_stage5_e2e_query.py
# MUST verify:
# 1. Full pipeline: corpus → graph → VDB → search → retrieve text
# 2. Final answer contains actual data from corpus
# 3. No errors in any pipeline stage

# EVIDENCE REQUIRED:
# - corpus_docs: <integer > 0>
# - graph_nodes: <integer > 0>
# - vdb_entities: <integer > 0>
# - search_results: [list of found entities]
# - retrieved_text: <actual text from corpus>
# - final_answer: <meaningful answer with corpus data>

# STATUS: [ ] NOT STARTED
# EVIDENCE:
# COMMIT:
```

**Implementation**:
1. Run full test on Social_Discourse_Test dataset
2. Verify each stage completes successfully
3. Ensure final answer contains real data

## Testing Protocol

**CRITICAL**: Each stage MUST be completed and verified before proceeding to the next:

1. Create test file for stage
2. Run test and capture output
3. Verify ALL criteria are met
4. Update test file with:
   - STATUS: [X] PASSED
   - EVIDENCE: <paste actual output proving success>
   - COMMIT: <commit hash after fixes>
5. Commit changes with message: "fix: Stage N - <description>"
6. Only then proceed to next stage

**Test Datasets Available**:
- `Data/Social_Discourse_Test`: Rich social network with 10 actors, 20 posts, clear communities
- `Data/Russian_Troll_Sample`: Real Twitter data but sparse
- `Data/MySampleTexts`: Historical documents (American/French Revolution)

## Success Metrics

The system is considered functional when:
1. Can build ER graph with proper entities (strings not dicts)
2. Uses only registered tools (no hallucinations)
3. Finds corpus files reliably
4. Maintains graph context between tools
5. Completes full query pipeline returning real data

Each stage must show concrete evidence of success before moving forward.

## Previous Fixes Applied (2025-06-05)

### Already Fixed Issues

1. **Orchestrator State Preservation** ✓
   - Fixed orchestrator to preserve `_status` and `_message` fields
   - Agent can now detect tool failures

2. **PassageGraph Config Error** ✓
   - Added default for missing `summary_max_tokens`
   - PassageGraph no longer crashes

3. **Basic Path Resolution** ✓
   - Corpus tool resolves paths under Data/ directory
   - Basic corpus preparation works

### Test Data Created

1. **Social_Discourse_Test** - New!
   - 10 Twitter accounts with clear roles
   - 4 distinct communities
   - 20+ posts with rich mention network
   - 5 discourse phases over 24 hours
   - Perfect for testing social network analysis
# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.

      
      IMPORTANT: this context may or may not be relevant to your tasks. You should not respond to this context or otherwise consider it in your response unless it is highly relevant to your task. Most of the time, it is not relevant.