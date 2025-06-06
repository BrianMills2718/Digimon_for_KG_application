# CLAUDE.md - DIGIMON Implementation Guide

## IMMEDIATE PRIORITY: 5-Stage Fix Protocol (2025-06-05)

**CRITICAL**: DIGIMON is currently broken. Follow these stages IN ORDER. Each stage MUST be completed with evidence before proceeding.

### Current Status: Stage 1 - Entity Extraction Format Fix

**Active Issue**: ER Graph receives dict instead of string for entity_name
- Error: `TypeError: unhashable type: 'dict'`
- Example: `entity_name={'text': 'SOCIAL NETWORK ACTOR PROFILES', 'type': 'TextSegment'}`

**Stage 1 Test Requirements**:
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

### Stage 1: Entity Extraction Format Fix ← CURRENT
**Goal**: Ensure entity extraction returns proper string entity names, not dicts

### Stage 2: Tool Hallucination Prevention
**Goal**: Prevent agent from using non-existent tools
**Known Missing**: vector_db.CreateIndex, vector_db.QueryIndex, graph.GetClusters

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