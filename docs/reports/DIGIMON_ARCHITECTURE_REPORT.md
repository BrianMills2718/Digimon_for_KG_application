# DIGIMON Architecture Report

## Executive Summary

DIGIMON (Deep Analysis of Graph-Based RAG Systems) is a sophisticated, modular GraphRAG framework built on an intelligent agent architecture. The system autonomously processes text data through a multi-stage pipeline that includes corpus preparation, graph construction, retrieval operations, and answer synthesis. Its design emphasizes flexibility, extensibility, and multi-modal graph support.

## Directory Structure (3-4 Levels Deep)

```
/home/brian/digimon_cc/
├── Core/                               # Core system components
│   ├── AgentBrain/                    # LLM-driven planning and reasoning
│   │   └── agent_brain.py             # Main planning agent implementation
│   ├── AgentOrchestrator/             # Tool execution and workflow management
│   │   └── orchestrator.py            # Executes plans, manages tool registry
│   ├── AgentSchema/                   # Pydantic contracts for all operations
│   │   ├── context.py                 # GraphRAG context management
│   │   ├── corpus_tool_contracts.py   # Corpus preparation contracts
│   │   ├── graph_construction_tool_contracts.py
│   │   ├── plan.py                    # Execution plan schemas
│   │   └── tool_contracts.py          # Tool input/output schemas
│   ├── AgentTools/                    # Granular retrieval operators (16+ tools)
│   │   ├── chunk_tools.py             # Text chunk retrieval
│   │   ├── community_tools.py         # Community-based operations
│   │   ├── corpus_tools.py            # Corpus preparation
│   │   ├── entity_*.py                # Entity-based tools (VDB, PPR, etc.)
│   │   ├── graph_*.py                 # Graph construction/analysis
│   │   ├── relationship_tools.py      # Relationship operations
│   │   └── subgraph_tools.py          # Subgraph extraction
│   ├── Chunk/                         # Document chunking system
│   │   ├── ChunkFactory.py            # Factory for chunk methods
│   │   ├── DocChunk.py                # Main chunking implementation
│   │   └── Separator.py               # Text separation utilities
│   ├── Common/                        # Shared utilities
│   │   ├── BaseFactory.py             # Base factory pattern
│   │   ├── Context.py                 # Context management
│   │   ├── Logger.py                  # Logging system
│   │   ├── LLM.py                     # LLM interface
│   │   └── Utils.py                   # General utilities
│   ├── Community/                     # Community detection algorithms
│   │   ├── BaseCommunity.py          # Base community interface
│   │   ├── ClusterFactory.py          # Community algorithm factory
│   │   └── LeidenCommunity.py        # Leiden algorithm implementation
│   ├── Graph/                         # Graph implementations
│   │   ├── BaseGraph.py               # Abstract graph interface
│   │   ├── ERGraph.py                 # Entity-Relation Graph
│   │   ├── GraphFactory.py            # Graph type factory
│   │   ├── PassageGraph.py            # Passage-based graph
│   │   ├── RKGraph.py                 # Relation-Knowledge Graph
│   │   ├── TreeGraph.py               # Hierarchical tree graph
│   │   └── TreeGraphBalanced.py      # Balanced tree variant
│   ├── Index/                         # Vector database implementations
│   │   ├── BaseIndex.py               # Abstract index interface
│   │   ├── ColBertIndex.py            # ColBERT index
│   │   ├── EmbeddingFactory.py        # Embedding provider factory
│   │   ├── FaissIndex.py              # Faiss vector index
│   │   └── VectorIndex.py             # General vector index
│   ├── Provider/                      # LLM/Embedding providers
│   │   ├── BaseLLM.py                 # Abstract LLM interface
│   │   ├── BaseEmb.py                 # Abstract embedding interface
│   │   ├── LiteLLMProvider.py         # LiteLLM integration
│   │   └── LLMProviderRegister.py     # Provider registration
│   ├── Query/                         # Query processing strategies
│   │   ├── BaseQuery.py               # Abstract query interface
│   │   ├── BasicQuery.py              # Basic query implementation
│   │   ├── PPRQuery.py                # Personalized PageRank
│   │   ├── QueryFactory.py            # Query strategy factory
│   │   └── [Various]Query.py          # Specialized query types
│   ├── Retriever/                     # Retrieval components
│   │   ├── BaseRetriever.py           # Abstract retriever
│   │   ├── ChunkRetriever.py          # Chunk-based retrieval
│   │   ├── EntityRetriever.py         # Entity-based retrieval
│   │   ├── RetrieverFactory.py        # Retriever factory
│   │   └── SubgraphRetriever.py      # Subgraph retrieval
│   ├── Schema/                        # Data schemas
│   │   ├── ChunkSchema.py             # Chunk data structures
│   │   ├── EntityRelation.py          # Entity/relation schemas
│   │   ├── GraphSchema.py             # Graph data structures
│   │   └── RetrieverContext.py        # Retrieval context
│   ├── Storage/                       # Storage backends
│   │   ├── BaseStorage.py             # Abstract storage
│   │   ├── NetworkXStorage.py         # NetworkX graph storage
│   │   └── PickleBlobStorage.py       # Pickle-based storage
│   └── GraphRAG.py                    # Main GraphRAG orchestrator
├── Config/                            # Configuration files
│   ├── ChunkConfig.py                 # Chunking parameters
│   ├── GraphConfig.py                 # Graph construction config
│   ├── LLMConfig.py                   # LLM provider config
│   └── custom_ontology.json           # Domain-specific schemas
├── Option/                            # Method configurations
│   ├── Config2.yaml                   # Base configuration
│   ├── Config2.example.yaml           # Example config template
│   └── Method/                        # Pre-configured methods
│       ├── LGraphRAG.yaml             # Local GraphRAG
│       ├── GGraphRAG.yaml             # Global GraphRAG
│       ├── HippoRAG.yaml              # PPR-based retrieval
│       └── [Various].yaml             # Other methods
├── Data/                              # Dataset storage
│   └── [dataset_name]/                # Per-dataset directories
│       ├── Corpus.json                # Processed corpus
│       ├── Question.json              # Evaluation questions
│       └── *.txt                      # Raw text files
├── Results/                           # Output storage
│   └── [dataset_name]/                # Per-dataset results
│       └── [graph_type]/              # Graph artifacts
├── main.py                            # CLI entry point (build/query/evaluate)
├── digimon_cli.py                     # Agent-based CLI interface
├── api.py                             # Flask REST API server
└── streamlit_agent_frontend.py        # Streamlit web UI
```

## Core Components and Responsibilities

### 1. Agent Framework

The agent framework is the brain of DIGIMON, providing intelligent planning and execution capabilities:

#### **AgentBrain** (`Core/AgentBrain/agent_brain.py`)
- **PlanningAgent**: LLM-driven planning engine
  - Generates ExecutionPlans from natural language queries
  - Supports both standard and ReAct-style (iterative) planning
  - Manages tool documentation and prompt engineering
  - Handles plan-to-action translation

#### **AgentOrchestrator** (`Core/AgentOrchestrator/orchestrator.py`)
- Executes plans generated by AgentBrain
- Maintains tool registry (16+ specialized tools)
- Resolves tool inputs and dependencies
- Manages step outputs and data flow between tools
- Handles graph instance registration and lifecycle

#### **AgentSchema** (`Core/AgentSchema/`)
- Pydantic models for all agent operations
- Ensures type safety and validation
- Key schemas:
  - `ExecutionPlan`: Multi-step workflow definition
  - `ToolCall`: Individual tool invocation
  - `GraphRAGContext`: System-wide context management
  - Tool-specific input/output contracts

### 2. Graph Construction System

DIGIMON supports 5 distinct graph types, each optimized for different use cases:

#### **GraphFactory** (`Core/Graph/GraphFactory.py`)
- Factory pattern for graph instantiation
- Supported graph types:
  1. **ERGraph** (Entity-Relation): Traditional knowledge graph with entities and typed relationships
  2. **RKGraph** (Relation-Knowledge): Keyword/entity extraction based
  3. **TreeGraph**: Hierarchical clustering with dimensionality reduction
  4. **TreeGraphBalanced**: Optimized variant with balanced clusters
  5. **PassageGraph**: Links passages based on entity co-occurrence

#### **Graph Implementations**
Each graph type inherits from `BaseGraph` and implements:
- `build_graph()`: Construct from text chunks
- `load_persisted_graph()`: Load from storage
- `node_metadata()` / `edge_metadata()`: Provide indexing data
- Graph-specific algorithms (e.g., entity extraction, clustering)

### 3. Retrieval System

The retrieval system provides granular access to graph data through multiple paradigms:

#### **Retriever Components**
- **ChunkRetriever**: Direct text chunk access
- **EntityRetriever**: Entity-based search
- **RelationshipRetriever**: Edge-based retrieval
- **SubgraphRetriever**: Multi-hop graph traversal
- **CommunityRetriever**: Cluster-based retrieval

#### **Query Strategies** (`Core/Query/`)
Pre-configured retrieval pipelines:
- **BasicQuery**: Simple vector search
- **PPRQuery**: Personalized PageRank (HippoRAG)
- **KGPQuery**: Knowledge graph pathfinding
- **ToGQuery**: Agent-based graph traversal
- **GRQuery**: Global retrieval with subgraph discovery

### 4. Tool System and Operators

The tool system exposes all GraphRAG operations as composable, validated tools:

#### **Tool Categories**
1. **Corpus Tools**: Document preparation and processing
2. **Graph Construction Tools**: Build various graph types
3. **Entity Tools**: VDB search, PPR, one-hop neighbors
4. **Relationship Tools**: Edge operations, path finding
5. **Chunk Tools**: Text retrieval for entities/relationships
6. **Community Tools**: Cluster-based operations
7. **Analysis Tools**: Graph statistics and visualization

#### **Tool Contract System**
- Every tool has Pydantic input/output models
- Contracts ensure type safety and validation
- Tools are registered in orchestrator's registry
- Dynamic tool chaining supported

### 5. Storage and Persistence

DIGIMON uses a namespace-based storage system for artifact persistence:

#### **Storage Backends**
- **NetworkXStorage**: Graph persistence using GraphML
- **PickleBlobStorage**: Binary storage for complex objects
- **ChunkKVStorage**: Key-value store for text chunks
- **JsonKVStorage**: JSON-based storage

#### **Namespace System**
- Each dataset/experiment gets isolated storage
- Artifacts organized by type (graph, VDB, chunks, etc.)
- Supports incremental updates and caching

### 6. Configuration Management

Hierarchical configuration system with multiple override levels:

#### **Configuration Hierarchy**
1. **Base Config** (`Option/Config2.yaml`): API keys, models
2. **Method Configs** (`Option/Method/*.yaml`): Algorithm settings
3. **Custom Ontology** (`Config/custom_ontology.json`): Domain schemas
4. **Runtime Overrides**: Dynamic configuration via tools

#### **Key Configuration Classes**
- `Config`: Main configuration container
- `LLMConfig`: LLM provider settings
- `GraphConfig`: Graph construction parameters
- `ChunkConfig`: Text chunking settings
- `RetrieverConfig`: Retrieval parameters

## Data Flow (Input to Output)

### 1. **Corpus Preparation**
```
Raw .txt files → corpus.PrepareFromDirectory → Corpus.json
```
- Reads text files from directory
- Creates structured corpus with metadata
- Assigns document IDs

### 2. **Graph Construction**
```
Corpus.json → graph.Build[Type] → Graph artifacts + indices
```
- Chunks documents based on token size
- Extracts entities/relationships (graph-type specific)
- Builds graph structure
- Persists to storage

### 3. **Index Building**
```
Graph nodes/edges → Entity.VDB.Build → Vector indices
```
- Creates embeddings for graph elements
- Builds Faiss/ColBERT indices
- Enables semantic search

### 4. **Query Processing**
```
User query → Planning → Tool execution → Context retrieval → Answer
```
- PlanningAgent creates ExecutionPlan
- Orchestrator executes tools sequentially
- Tools retrieve relevant context
- LLM synthesizes final answer

### 5. **Answer Generation**
The final stage combines retrieved context:
- Entity information from VDB search
- Relationship data from graph traversal
- Text chunks associated with entities
- Community summaries (if applicable)

## Operational Modes

### 1. **Agent Mode** (Primary)
- Fully autonomous operation
- Dynamic plan generation
- Adaptive tool selection
- Supports ReAct-style iterative refinement

### 2. **Direct Pipeline Mode**
- Traditional build/query/evaluate workflow
- Pre-configured method execution
- Batch processing support
- Evaluation metrics generation

### 3. **Interactive CLI Mode**
- Real-time query processing
- Session state management
- Multi-turn conversations
- Progress visualization

### 4. **API Server Mode**
- RESTful endpoints for all operations
- Async request handling
- Multi-tenant support
- Graph visualization endpoints

## Key Design Patterns

### 1. **Factory Pattern**
Used extensively for component creation:
- GraphFactory for graph types
- ChunkFactory for chunking methods
- RetrieverFactory for retrieval strategies
- QueryFactory for query processors

### 2. **Tool-Based Architecture**
- All operations exposed as tools
- Pydantic contracts for type safety
- Composable and chainable
- Dynamic discovery and registration

### 3. **Context Management**
- GraphRAGContext maintains global state
- Thread-safe access to shared resources
- Lazy loading of artifacts
- Resource lifecycle management

### 4. **Provider Abstraction**
- LLM operations through BaseLLM interface
- Embedding operations through BaseEmb
- LiteLLM for multi-provider support
- Easy provider switching

### 5. **Namespace Isolation**
- Per-dataset storage isolation
- Artifact versioning support
- Cache management
- Concurrent experiment support

## Extension Points

The system is designed for extensibility:

1. **New Graph Types**: Implement BaseGraph interface
2. **Custom Tools**: Add to AgentTools with contracts
3. **Query Strategies**: Extend BaseQuery
4. **Storage Backends**: Implement BaseStorage
5. **LLM Providers**: Register with LiteLLM
6. **Retrieval Methods**: Add to RetrieverFactory

## Performance Optimizations

- Batch processing for embeddings
- Incremental graph updates
- Caching at multiple levels
- Async I/O throughout
- Lazy loading of artifacts
- Connection pooling for LLMs

## Conclusion

DIGIMON represents a sophisticated approach to GraphRAG, combining the flexibility of an agent-based architecture with the performance of specialized graph algorithms. Its modular design, comprehensive tool system, and multi-graph support make it suitable for a wide range of knowledge-intensive applications.