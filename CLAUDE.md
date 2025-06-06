# CLAUDE.md - DIGIMON Implementation Guide

## IMMEDIATE PRIORITY: 5-Stage Fix Protocol (2025-06-05)

**CRITICAL**: DIGIMON is currently broken. Follow these stages IN ORDER. Each stage MUST be completed with evidence before proceeding.

### Current Status: Stage 3 - Corpus Path Standardization

**Active Issue**: Corpus files created in different locations
- PrepareCorpusFromDirectory creates at: `results/{dataset}/corpus/Corpus.json`
- ChunkFactory expects at: `results/{dataset}/Corpus.json`
- This causes "No input chunks found" errors

**Stage 3 Test Requirements**:
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

# STATUS: [X] PASSED
# EVIDENCE:
# - corpus_created_at: results/Social_Discourse_Test/corpus/Corpus.json
# - corpus_expected_at: results/Social_Discourse_Test/Corpus.json
# - chunks_loaded: 7
# - ChunkFactory updated to check corpus subdirectory
# - No path errors after fix
# COMMIT: fix: Stage 3 - Added corpus subdirectory to ChunkFactory search paths
```

**Stage 2 Test Requirements (COMPLETE)**:
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

# STATUS: [X] PASSED
# EVIDENCE:
# - total_tools_used: 2 (graph.BuildERGraph, Entity.VDB.Build)
# - missing_tools: 0 tools not found
# - known_hallucinations: 0 detected
# - execution_errors: 0
# - All tools used are in registered tool list
# COMMIT: Stage 2 already working - no tool hallucinations detected
```

**Stage 1 Test Requirements (COMPLETE)**:
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

# STATUS: [X] PASSED
# EVIDENCE: 
# - entity_name: string (verified on 5 entities)
# - entities_extracted: 44
# - relations_extracted: 50
# - sample_entity: 'tech optimists community' (type: <class 'str'>)
# - All entity names returned as proper strings, not dicts
# COMMIT: Stage 1 already working - entity extraction returns proper string format 
```

**Fix Approach**:
1. Check ERGraph._process_entity_types() and entity extraction prompts
2. Add validation to ensure entity_name is string
3. Test with Social_Discourse_Test dataset

### Remaining Stages (DO NOT START UNTIL STAGE 1 PASSES):

**Stage 2**: Tool Hallucination Prevention
**Stage 3**: Corpus Path Standardization  
**Stage 4**: Graph Registration & Context
**Stage 5**: End-to-End Query Success

---

## Quick Reference

### Test Datasets
- `Data/Social_Discourse_Test`: Best for testing (10 actors, 20 posts, rich network)
- `Data/Russian_Troll_Sample`: Real but sparse data
- `Data/MySampleTexts`: Historical documents

### Key Commands
```bash
# Test entity extraction directly
python digimon_cli.py -c Data/Social_Discourse_Test -q "Build ER graph"

# Check if corpus exists
ls results/Social_Discourse_Test/Corpus.json

# Copy corpus to expected location if needed
cp results/Social_Discourse_Test/corpus/Corpus.json results/Social_Discourse_Test/Corpus.json
```

### Current Environment
- Model: o4-mini (OpenAI)
- Embeddings: text-embedding-3-small
- Vector DB: FAISS
- Working directory: /home/brian/digimon_cc

---

## Architecture Overview (Reference Only)

### Relevant for Current Fix:
- **ERGraph**: `Core/Graph/ERGraph.py` - Entity extraction happens here
- **ChunkFactory**: `Core/Chunk/ChunkFactory.py` - Looks for corpus files
- **Orchestrator**: `Core/AgentOrchestrator/orchestrator.py` - Manages tool execution
- **GraphRAGContext**: `Core/AgentSchema/context.py` - Stores graph instances

### Tool Registry:
```
Entity.VDBSearch, Entity.VDB.Build, Entity.PPR, Entity.Onehop, Entity.RelNode,
Relationship.OneHopNeighbors, Relationship.VDB.Build, Relationship.VDB.Search,
Chunk.FromRelationships, Chunk.GetTextForEntities,
graph.BuildERGraph, graph.BuildRKGraph, graph.BuildTreeGraph,
graph.BuildTreeGraphBalanced, graph.BuildPassageGraph,
corpus.PrepareFromDirectory, graph.Visualize, graph.Analyze
```

---

## Development Commands (Reference)

```bash
# Environment setup
conda activate digimon

# Configuration
cp Option/Config2.example.yaml Option/Config2.yaml

# Run tests
pytest tests/integration/test_entity_extraction.py -v

# Interactive CLI
python digimon_cli.py -c Data/Social_Discourse_Test -i
```

---

## Previous Fixes Applied
1. ✓ PassageGraph summary_max_tokens - Added getattr default
2. ✓ Orchestrator state preservation - Fixed _status/_message fields
3. ✓ Basic corpus path resolution - Works under Data/ directory

---

## Stage Details (Full Protocol)

### Stage 1: Entity Extraction Format Fix ✓ COMPLETE
**Goal**: Ensure entity extraction returns proper string entity names, not dicts
**Result**: PASSED - Entity extraction already returns proper strings

### Stage 2: Tool Hallucination Prevention ✓ COMPLETE
**Goal**: Prevent agent from using non-existent tools
**Result**: PASSED - Agent only uses registered tools

### Stage 3: Corpus Path Standardization
**Goal**: Ensure corpus files are found regardless of creation method
**Issue**: Tool creates at `results/{dataset}/corpus/Corpus.json`, ChunkFactory expects `results/{dataset}/Corpus.json`

### Stage 4: Graph Registration & Context Management
**Goal**: Ensure built graphs are accessible to subsequent tools
**Issue**: Graphs build but aren't registered in GraphRAGContext

### Stage 5: End-to-End Query Success
**Goal**: Complete full pipeline successfully with real data in answer

---

## Testing Protocol

1. Create test file for current stage
2. Run test and capture ALL output
3. Verify ALL criteria are met with specific evidence
4. Update test file with STATUS: [X] PASSED and paste evidence
5. Commit: `git commit -m "fix: Stage N - <description>"`
6. Update this file to mark stage complete and move to next
7. Only then proceed to next stage

**Success = All 5 stages pass with evidence**