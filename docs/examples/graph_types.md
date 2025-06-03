# Graph Types in DIGIMON

DIGIMON supports multiple graph types for different use cases. This guide explains each type with examples.

## 1. Entity-Relationship Graph (ERGraph)

The ERGraph extracts entities and their relationships from text, creating a knowledge graph structure.

### Example Usage

```python
from Core.AgentTools.graph_construction_tools import build_er_graph

# Build ER graph from corpus
result = await build_er_graph(
    context=agent_context,
    target_dataset_name="MySampleTexts",
    custom_ontology_path="Config/custom_ontology.json"  # Optional
)

print(f"Graph ID: {result['graph_id']}")
print(f"Entities: {result['entity_count']}")
print(f"Relationships: {result['relationship_count']}")
```

### When to Use
- Extracting structured knowledge from unstructured text
- Building knowledge bases
- Entity-centric analysis
- Relationship discovery

### Example Output
```
Entities:
- French Revolution (Event)
- Louis XVI (Person)
- Paris (Location)
- 1789 (Date)

Relationships:
- Louis XVI --[ruled_during]--> French Revolution
- French Revolution --[occurred_in]--> Paris
- French Revolution --[started_in]--> 1789
```

## 2. Relationship Knowledge Graph (RKGraph)

RKGraph focuses on relationships as first-class citizens, storing more detailed relationship metadata.

### Example Usage

```python
from Core.AgentTools.graph_construction_tools import build_rk_graph

result = await build_rk_graph(
    context=agent_context,
    target_dataset_name="MySampleTexts"
)
```

### When to Use
- Relationship-heavy analysis
- Temporal relationship tracking
- Multi-hop reasoning
- Complex relationship queries

## 3. Tree Graph (Hierarchical Summarization)

Tree graphs create hierarchical summaries of documents, useful for multi-scale analysis.

### Example Usage

```python
from Core.AgentTools.graph_construction_tools import build_tree_graph

# Build balanced tree with automatic summarization
result = await build_tree_graph_balanced(
    context=agent_context,
    target_dataset_name="MySampleTexts",
    max_depth=3,
    branch_factor=4
)
```

### When to Use
- Document summarization at multiple scales
- Hierarchical information retrieval
- Top-down analysis
- RAPTOR-style retrieval

### Example Structure
```
Root Summary
├── Chapter 1 Summary
│   ├── Section 1.1 Summary
│   └── Section 1.2 Summary
└── Chapter 2 Summary
    ├── Section 2.1 Summary
    └── Section 2.2 Summary
```

## 4. Passage Graph

Passage graphs connect text passages through shared entities, maintaining original text chunks.

### Example Usage

```python
from Core.AgentTools.graph_construction_tools import build_passage_graph

result = await build_passage_graph(
    context=agent_context,
    target_dataset_name="MySampleTexts",
    chunk_size=512,
    chunk_overlap=50
)
```

### When to Use
- Preserving original text context
- Passage-level retrieval
- Entity-based passage linking
- Question answering systems

## 5. Choosing the Right Graph Type

| Use Case | Recommended Graph Type |
|----------|----------------------|
| Knowledge extraction | ERGraph |
| Relationship analysis | RKGraph |
| Document summarization | TreeGraph |
| QA systems | PassageGraph |
| Multi-hop reasoning | ERGraph or RKGraph |
| Hierarchical search | TreeGraph |

## Complete Example: Building Multiple Graph Types

```python
from Core.AgentBrain.agent_brain import PlanningAgent
from Core.AgentOrchestrator.orchestrator import AgentOrchestrator

# Initialize agent
orchestrator = AgentOrchestrator()
agent = PlanningAgent(orchestrator=orchestrator)

# Process query that builds multiple graph types
query = "Build both an ER graph and a tree graph for my documents"
result = await agent.process_query(
    user_query=query,
    actual_corpus_name="MySampleTexts"
)

print(result['generated_answer'])
```

## Graph Querying Examples

### Entity Search
```python
# Search for entities in the graph
entities = await entity_vdb_search(
    context=context,
    query="French Revolution leaders",
    top_k=5
)
```

### Relationship Traversal
```python
# Get one-hop neighbors
neighbors = await get_entity_one_hop_neighbors(
    context=context,
    entity_name="Napoleon Bonaparte"
)
```

### Community Detection
```python
# Find communities in the graph
communities = await get_community_summaries(
    context=context,
    query="political movements",
    top_k=3
)
```