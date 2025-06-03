# DIGIMON Information for StructGPT Integration

## 1. Overall Architecture & Project Structure

### Directory Structure

```
digimon_cc/
├── Core/                           # Core system implementation
│   ├── AgentBrain/                # LLM-driven planning and reasoning
│   │   ├── __init__.py
│   │   └── agent_brain.py         # Plan generation, reasoning engine
│   ├── AgentOrchestrator/         # Tool execution and workflow management
│   │   ├── __init__.py
│   │   └── orchestrator.py        # Executes plans, manages tool calls
│   ├── AgentSchema/               # Pydantic contracts for all operations
│   │   ├── context.py             # Execution context and state
│   │   ├── plan.py                # Plan and step definitions
│   │   ├── tool_contracts.py      # Base tool interfaces
│   │   ├── corpus_tool_contracts.py     # Corpus preparation contracts
│   │   └── graph_construction_tool_contracts.py  # Graph building contracts
│   ├── AgentTools/                # Tool implementations (16+ tools)
│   │   ├── corpus_tools.py        # PrepareCorpusFromDirectory
│   │   ├── graph_construction_tools.py  # BuildERGraph, BuildTreeGraph, etc.
│   │   ├── entity_tools.py        # EntitySearch, EntityPPR
│   │   ├── entity_vdb_tools.py   # EntityVDBSearch, EntityVDBBuild
│   │   ├── entity_onehop_tools.py # EntityOneHopNeighbor
│   │   ├── entity_relnode_tools.py # EntityRelationshipNodeRetrieval
│   │   ├── relationship_tools.py  # RelationshipVDBSearch
│   │   ├── chunk_tools.py         # ChunkSearch, ChunkFromRelationships
│   │   ├── community_tools.py     # CommunitySearch
│   │   ├── subgraph_tools.py     # SubgraphExtract
│   │   ├── graph_analysis_tools.py # GraphAnalysis
│   │   ├── graph_visualization_tools.py # GraphVisualization
│   │   └── query_expansion.py     # GenerateAnalyticChain
│   ├── Graph/                     # Knowledge graph implementations
│   │   ├── BaseGraph.py           # Abstract base class
│   │   ├── ERGraph.py             # Entity-Relationship graph
│   │   ├── RKGraph.py             # Relationship-Knowledge graph
│   │   ├── TreeGraph.py           # Hierarchical tree structure
│   │   ├── TreeGraphBalanced.py   # Balanced tree variant
│   │   ├── PassageGraph.py        # Passage-based graph
│   │   ├── GraphFactory.py        # Factory for graph creation
│   │   └── ontology_generator.py  # Ontology management
│   ├── Provider/                  # LLM/Embedding providers
│   │   ├── LiteLLMProvider.py     # Primary LLM interface
│   │   ├── OpenaiApi.py           # OpenAI/compatible APIs
│   │   └── LLMProviderRegister.py # Provider registry
│   ├── Storage/                   # Data persistence (empty - uses networkx)
│   ├── Retriever/                 # Retrieval implementations
│   │   ├── BaseRetriever.py       # Abstract retriever
│   │   ├── EntityRetriever.py     # Entity-based retrieval
│   │   ├── ChunkRetriever.py      # Chunk-based retrieval
│   │   ├── CommunityRetriever.py  # Community-based retrieval
│   │   ├── RelationshipRetriever.py # Relationship retrieval
│   │   ├── SubgraphRetriever.py   # Subgraph extraction
│   │   └── MixRetriever.py        # Combined retrieval strategies
│   ├── Index/                     # Vector database implementations
│   │   ├── FaissIndex.py          # Faiss vector search
│   │   ├── ColBertIndex.py        # ColBERT retrieval
│   │   └── VectorIndex.py         # Base vector index
│   ├── Chunk/                     # Document chunking
│   │   ├── ChunkFactory.py        # Chunking strategies
│   │   └── DocChunk.py            # Chunk representation
│   ├── Community/                 # Graph clustering
│   │   ├── LeidenCommunity.py     # Leiden algorithm
│   │   └── ClusterFactory.py      # Clustering factory
│   ├── Query/                     # Query processing engines
│   │   ├── BasicQuery.py          # Simple query processor
│   │   ├── GRQuery.py             # GraphRAG query
│   │   ├── ToGQuery.py            # Tree-of-Graph query
│   │   └── QueryFactory.py        # Query engine factory
│   ├── Schema/                    # Data schemas
│   │   ├── EntityRelation.py      # Entity/relation definitions
│   │   ├── ChunkSchema.py         # Document chunks
│   │   └── RetrieverContext.py    # Retrieval context
│   └── GraphRAG.py               # Main GraphRAG coordinator
├── Config/                        # Configuration files
│   ├── ChunkConfig.py            # Chunking parameters
│   ├── EmbConfig.py              # Embedding settings
│   ├── GraphConfig.py            # Graph construction config
│   ├── LLMConfig.py              # LLM provider config
│   ├── QueryConfig.py            # Query processing config
│   └── custom_ontology.json      # Domain-specific schemas
├── Option/                       # Runtime configurations
│   ├── Config2.yaml              # Main config (API keys, models)
│   ├── Config2.example.yaml      # Template config
│   └── Method/                   # Pre-configured methods
│       ├── LGraphRAG.yaml        # Local GraphRAG
│       ├── GGraphRAG.yaml        # Global GraphRAG
│       ├── HippoRAG.yaml         # HippoRAG variant
│       └── [other methods]
├── main.py                       # CLI entry point (build/query/evaluate)
├── api.py                        # Flask REST API server
├── digimon_cli.py               # Interactive agent CLI
└── streamlit_agent_frontend.py   # Web UI frontend

### Core Components and Responsibilities

1. **AgentBrain** (`Core/AgentBrain/agent_brain.py`)
   - Generates execution plans from natural language queries
   - Uses LLM to decompose complex tasks into tool sequences
   - Implements ReAct-style reasoning loops
   - Manages conversation history and context

2. **AgentOrchestrator** (`Core/AgentOrchestrator/orchestrator.py`)
   - Executes plans generated by AgentBrain
   - Manages tool registration and invocation
   - Handles error recovery and retries
   - Maintains execution state and results

3. **GraphRAG** (`Core/GraphRAG.py`)
   - Coordinates the entire pipeline
   - Manages build/query/evaluate workflows
   - Handles storage namespace isolation
   - Integrates all components

4. **Tool System** (`Core/AgentTools/`)
   - Each tool is a self-contained operation
   - Tools have Pydantic contracts for validation
   - Tools can be composed into complex workflows
   - 16+ tools covering corpus, graph, retrieval, and analysis

### Main Entry Points

- `main.py`: Traditional CLI for build/query/evaluate operations
- `digimon_cli.py`: Interactive agent mode with ReAct capabilities
- `api.py`: REST API server for programmatic access
- `streamlit_agent_frontend.py`: Web UI for interactive exploration

### Data Flow

```
Raw Text Files → PrepareCorpusFromDirectory → Corpus.json
                                                    ↓
                                            Graph Construction
                                            (5 types available)
                                                    ↓
                                            Vector Index Building
                                                    ↓
Query → AgentBrain (Plan Generation) → AgentOrchestrator (Tool Execution)
                                                    ↓
                                            Retrieval & Synthesis
                                                    ↓
                                                Answer
```

## 2. Agent Framework Details

### Agent Architecture

**AgentBrain Implementation:**
- Located in `Core/AgentBrain/agent_brain.py`
- Uses LLM to generate structured plans from queries
- Plans consist of steps with tool calls and parameters
- Supports both simple queries and complex analytical chains
- Implements conversation memory and context management

**AgentOrchestrator Implementation:**
- Located in `Core/AgentOrchestrator/orchestrator.py`
- Executes plans step by step
- Manages tool registry and invocation
- Handles parameter validation and type conversion
- Collects and formats results for downstream steps

**Agent-Tool-Operator Relationship:**
- Agents generate plans that specify tool sequences
- Tools are atomic operations with defined contracts
- Operators are lower-level implementations used by tools
- Tools can compose multiple operators internally

### Tool System

**Complete Tool List:**

1. **Corpus Tools**
   - `PrepareCorpusFromDirectory`: Convert raw text to structured corpus

2. **Graph Construction Tools**
   - `BuildERGraph`: Entity-Relationship graph
   - `BuildRKGraph`: Relationship-Knowledge graph
   - `BuildTreeGraph`: Hierarchical tree structure
   - `BuildTreeBalancedGraph`: Balanced tree variant
   - `BuildPassageGraph`: Passage-based graph

3. **Entity Tools**
   - `EntitySearch`: Find entities by keywords
   - `EntityPPR`: Personalized PageRank for entities
   - `EntityVDBSearch`: Vector similarity search
   - `EntityVDBBuild`: Build entity embeddings
   - `EntityOneHopNeighbor`: Get connected entities
   - `EntityRelationshipNodeRetrieval`: Get relationships for entity

4. **Retrieval Tools**
   - `ChunkSearch`: Search document chunks
   - `ChunkFromRelationships`: Get chunks from relationships
   - `RelationshipVDBSearch`: Vector search on relationships
   - `CommunitySearch`: Search graph communities
   - `SubgraphExtract`: Extract relevant subgraphs

5. **Analysis Tools**
   - `GraphAnalysis`: Analyze graph structure
   - `GraphVisualization`: Generate visual representations
   - `GenerateAnalyticChain`: Create multi-step analysis plans

**Tool Definition Pattern:**
```python
# Contract in Core/AgentSchema/tool_contracts.py
class EntitySearchInput(BaseModel):
    keywords: List[str]
    graph_type: str
    top_k: int = 10

# Implementation in Core/AgentTools/entity_tools.py
class EntitySearchTool(BaseTool):
    def execute(self, input: EntitySearchInput) -> EntitySearchOutput:
        # Implementation
```

**Tool Registration:**
- Tools are auto-discovered by the orchestrator
- Each tool has a unique name and description
- Tools declare their input/output contracts
- Registry maintains tool metadata for planning

### Execution Model

**Analytic Chain Definition:**
- Chains are sequences of tool invocations
- Each step can use outputs from previous steps
- Chains support conditional logic and loops
- Plans are generated dynamically based on query

**ReAct Implementation Status:**
- Basic ReAct loop implemented in `digimon_cli.py`
- Supports thought-action-observation cycles
- Agent can refine plans based on results
- Currently experimental, being refined

**Error Handling:**
- Tools return success/failure status
- Orchestrator implements retry logic
- Failed steps can trigger replanning
- Errors are propagated with context

## 3. Knowledge Graph Implementation

### Graph Types & Structure

**1. ERGraph (Entity-Relationship Graph)**
- Traditional knowledge graph with entities and typed relationships
- Nodes: Entities with properties
- Edges: Typed relationships with attributes
- Best for: Structured information extraction

**2. RKGraph (Relationship-Knowledge Graph)**
- Focuses on relationships as first-class citizens
- Nodes: Both entities and relationship nodes
- Edges: Connections between all node types
- Best for: Complex relationship analysis

**3. TreeGraph (Hierarchical Tree)**
- Recursive summarization structure
- Nodes: Document chunks and summaries
- Edges: Parent-child relationships
- Best for: Multi-level document understanding

**4. TreeGraphBalanced**
- Balanced variant of TreeGraph
- Ensures uniform tree depth
- Better for consistent retrieval
- Best for: Large document collections

**5. PassageGraph**
- Passage-centric representation
- Nodes: Document passages
- Edges: Semantic connections
- Best for: Dense retrieval tasks

**Graph Library:**
- Uses NetworkX as the core graph library
- Custom serialization for persistence
- In-memory operations for performance
- Supports graph algorithms (PageRank, community detection)

### Data Model

**Node Schema:**
```python
class Entity(BaseModel):
    id: str
    name: str
    type: str
    properties: Dict[str, Any]
    embeddings: Optional[List[float]]
    source_chunks: List[str]
```

**Edge Schema:**
```python
class Relationship(BaseModel):
    source: str
    target: str
    type: str
    properties: Dict[str, Any]
    weight: float = 1.0
    source_chunks: List[str]
```

**Property Validation:**
- Pydantic models for type safety
- Custom validators for domain constraints
- Ontology-based property restrictions
- Runtime type checking

### Query Interface

**Current Query Methods:**
1. Direct NetworkX queries (graph.nodes(), graph.edges())
2. Custom traversal algorithms
3. Vector similarity search via embeddings
4. SPARQL-like pattern matching (planned)

**Performance Characteristics:**
- In-memory graphs scale to ~100K nodes
- Vector search via Faiss for large datasets
- Community detection for graph partitioning
- Indexing strategies for common queries

## 4. LLM Integration

### LLM Provider Setup

**Supported Providers:**
- OpenAI (GPT-3.5, GPT-4)
- Anthropic (Claude models)
- Local models via Ollama
- Any LiteLLM-compatible provider

**Configuration Management:**
```yaml
# Option/Config2.yaml
LLMConfig:
  DefaultModel: "gpt-4"
  Temperature: 0.7
  MaxTokens: 2000
  
ProviderSettings:
  OPENAI_API_KEY: "..."
  ANTHROPIC_API_KEY: "..."
```

**Prompt Templates:**
- Located in `Core/Prompt/`
- Separate templates for each operation
- Support for few-shot examples
- Dynamic template construction

### LLM Usage Patterns

**LLM Call Locations:**
1. Plan generation (AgentBrain)
2. Entity/relationship extraction
3. Query understanding
4. Answer synthesis
5. Graph construction guidance

**Response Handling:**
- Structured output parsing (JSON)
- Retry logic for malformed responses
- Validation against expected schemas
- Fallback strategies for failures

**Cost Management:**
- Token counting and tracking
- Model selection based on task
- Caching of repeated queries
- Batch processing where possible

## 5. Data Processing Pipeline

### Input Processing

**PrepareCorpusFromDirectory Implementation:**
```python
def prepare_corpus(directory: str) -> Dict:
    # 1. Scan directory for text files
    # 2. Read and validate content
    # 3. Extract metadata
    # 4. Create document entries
    # 5. Generate corpus.json
```

**Supported Formats:**
- Plain text (.txt)
- Markdown (.md)
- JSON documents
- PDF (with conversion)

**Chunking Strategies:**
- Sentence-based chunking
- Token-count chunking
- Semantic chunking
- Overlap strategies for context

### Extraction Operators

**Implementation Example - extract_categorical_value:**
```python
def extract_categorical_value(text: str, categories: List[str]) -> str:
    # Use LLM to classify text into categories
    # Validate against allowed values
    # Return with confidence score
```

**Reliability Measures:**
- Confidence scoring
- Multiple extraction attempts
- Validation against ontology
- Human-in-the-loop options

### Data Storage

**Storage Architecture:**
- NetworkX graphs serialized to disk
- Vector indices in Faiss format
- Document chunks in JSON
- Metadata in SQLite (planned)

**File Organization:**
```
storage/
├── MySampleTexts/
│   ├── graph.pkl          # Serialized NetworkX graph
│   ├── entities.json      # Entity registry
│   ├── relationships.json # Relationship registry
│   ├── chunks.json        # Document chunks
│   ├── entity_vdb.index   # Faiss index
│   └── metadata.json      # Build metadata
```

## 6. Operator System

### Retrieval Operators

**Implemented Operators:**
1. `EntitySearch`: Keyword-based entity search
2. `EntityPPR`: Personalized PageRank
3. `EntityVDBSearch`: Vector similarity
4. `ChunkSearch`: Full-text chunk search
5. `RelationshipVDBSearch`: Relationship vectors
6. `CommunitySearch`: Community detection
7. `SubgraphExtract`: k-hop subgraphs
8. `EntityOneHopNeighbor`: Direct connections
9. `ChunkFromRelationships`: Associated chunks

**Operator-Graph Interface:**
- Operators receive graph instance
- Standard query interface
- Result formatting protocols
- Performance monitoring

### Transformation Operators

**Implemented Transformations:**
1. Graph construction (multiple types)
2. Entity merging/resolution
3. Relationship inference
4. Community detection
5. Embedding generation

**Operator Chaining:**
- Output of one operator feeds next
- Type conversion handled automatically
- State management between operators
- Pipeline optimization

### Operator Registry

**Discovery Mechanism:**
- Tools auto-register on import
- Metadata extracted from docstrings
- Parameter schemas from Pydantic
- Version tracking via git

**Documentation Standards:**
- Each operator has description
- Input/output schemas defined
- Example usage provided
- Performance characteristics noted

## 7. Configuration Management

### Configuration Files

**Hierarchy:**
1. `Option/Config2.yaml` - Global settings
2. `Option/Method/*.yaml` - Method-specific
3. `Config/custom_ontology.json` - Domain schemas
4. Environment variables - Overrides

**Configuration Flow:**
```
Environment → Config2.yaml → Method.yaml → Runtime
```

### Ontology Management

**Custom Ontology Structure:**
```json
{
  "entities": {
    "Person": {
      "properties": ["name", "age", "occupation"],
      "constraints": {...}
    }
  },
  "relationships": {
    "KNOWS": {
      "properties": ["since", "context"],
      "valid_source": ["Person"],
      "valid_target": ["Person"]
    }
  }
}
```

**Runtime Modifications:**
- Ontology loaded at startup
- Can be modified via API
- Changes affect extraction
- Validation rules enforced

## 8. Dependencies & Infrastructure

### Core Dependencies

```
# Key packages from requirements.txt
networkx==3.1          # Graph operations
faiss-cpu==1.7.4      # Vector search
litellm==1.35.17      # LLM abstraction
pydantic==2.6.1       # Data validation
sentence-transformers  # Embeddings
streamlit==1.29.0     # Web UI
flask==3.0.0          # API server
```

### Development Setup

**Testing Framework:**
- pytest for unit/integration tests
- Mock LLM responses for testing
- Fixture system for test data
- ~80% code coverage

**Environment Management:**
```bash
# Development
conda env create -f environment.yml

# Production
pip install -r requirements.txt
```

## 9. Current Limitations & Pain Points

### Known Issues

1. **Performance Bottlenecks:**
   - Large graph operations slow (>100K nodes)
   - Embedding generation not batched optimally
   - Sequential tool execution (no parallelism)

2. **Reliability Concerns:**
   - LLM response parsing occasionally fails
   - Entity resolution needs improvement
   - Some tools lack proper error handling

3. **Architectural Debt:**
   - Storage layer needs abstraction
   - Config system is complex
   - Some circular dependencies

### Development Challenges

**Hardest Implementations:**
1. Reliable entity/relationship extraction
2. Multi-graph type support
3. ReAct loop stability
4. Ontology enforcement

**Areas Needing Help:**
1. Distributed graph processing
2. Advanced reasoning capabilities
3. Better evaluation metrics
4. Production deployment patterns

## 10. Future Architecture Plans

### Planned Changes

1. **Storage Abstraction Layer**
   - Support for graph databases (Neo4j)
   - Distributed storage options
   - Better versioning support

2. **Advanced Reasoning**
   - Multi-hop reasoning chains
   - Counterfactual analysis
   - Temporal reasoning

3. **Performance Improvements**
   - Parallel tool execution
   - Incremental graph updates
   - Caching strategies

### Integration Points

**External Reasoning Engines:**
- Tool interface is the main extension point
- New tools can wrap external systems
- Standard contracts ensure compatibility
- Results integrate into existing flow

**Interface Design:**
```python
class ExternalReasoningTool(BaseTool):
    def execute(self, input: ReasoningInput) -> ReasoningOutput:
        # Call external system
        # Transform results
        # Return in standard format
```

**Coexistence Strategy:**
- Different tools for different reasoning types
- Agent selects appropriate tools
- Results aggregated by orchestrator
- Unified output format

## 11. Research-Specific Requirements

### Discourse Analysis Needs

**Current Challenges:**
1. Multi-document relationship tracking
2. Argument structure extraction
3. Claim-evidence linking
4. Stance detection and analysis

**Where Sophisticated Reasoning Needed:**
- Cross-document entity resolution
- Implicit relationship inference
- Contradiction detection
- Narrative structure analysis

### Evaluation & Validation

**Current Metrics:**
- Retrieval accuracy (P@K, R@K)
- Answer quality (LLM-based eval)
- Graph construction metrics
- End-to-end task success

**Academic Integration:**
- Support for standard benchmarks
- Reproducible experiments
- Ablation study support
- Statistical significance testing

## Questions Answered

### 1. What's working well that we should preserve?

- **Modular tool architecture** - Easy to extend and test
- **Multi-graph support** - Flexibility for different use cases
- **Pydantic contracts** - Type safety and validation
- **LiteLLM abstraction** - Provider independence
- **Agent planning** - Natural language to execution

### 2. What's causing the most friction in development?

- **Complex configuration** - Too many config files and options
- **Storage limitations** - Need better abstraction
- **Sequential execution** - No parallelism in tools
- **Entity resolution** - Accuracy and performance issues
- **Testing complexity** - Hard to test LLM-dependent code

### 3. Where do you see the biggest opportunities for StructGPT integration?

- **Advanced reasoning tools** - Plug into tool system
- **Multi-hop query planning** - Enhance AgentBrain
- **Structure-aware retrieval** - New retrieval operators
- **Reasoning validation** - Verify analytical chains
- **Hybrid approaches** - Combine with existing tools

### 4. What would make your research workflow significantly easier?

- **Better evaluation framework** - Automated metrics
- **Reasoning trace visualization** - Debug complex chains
- **Incremental processing** - Update without full rebuild
- **Multi-model collaboration** - Different models for different tasks
- **Academic paper integration** - Direct citation handling

## Integration Recommendations

Based on this analysis, StructGPT integration should focus on:

1. **New Tool Development** - Create StructGPT-powered tools that plug into existing system
2. **Enhanced Planning** - Augment AgentBrain with structure-aware planning
3. **Retrieval Operators** - Add sophisticated retrieval strategies
4. **Validation Layer** - Verify reasoning chains and results
5. **Hybrid Workflows** - Combine DIGIMON's extraction with StructGPT's reasoning

The modular architecture makes it straightforward to add new capabilities without disrupting existing functionality. The tool system provides a clean integration point for external reasoning engines.