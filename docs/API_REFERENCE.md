# DIGIMON API Reference

## Core Classes

### PlanningAgent

The main agent class that orchestrates GraphRAG operations.

```python
from Core.AgentBrain.agent_brain import PlanningAgent

agent = PlanningAgent(
    orchestrator: AgentOrchestrator,
    llm_config: Optional[LLMConfig] = None
)
```

#### Methods

##### `async generate_plan(user_query: str, actual_corpus_name: Optional[str] = None) -> Optional[ExecutionPlan]`

Generates an execution plan based on natural language query.

**Parameters:**
- `user_query` (str): Natural language query
- `actual_corpus_name` (Optional[str]): Target corpus name

**Returns:**
- `ExecutionPlan`: Structured plan for execution

**Example:**
```python
plan = await agent.generate_plan(
    "What are the main entities in my documents?",
    actual_corpus_name="MySampleTexts"
)
```

##### `async process_query(user_query: str, actual_corpus_name: Optional[str] = None) -> Dict[str, Any]`

End-to-end query processing with plan generation, execution, and answer synthesis.

**Parameters:**
- `user_query` (str): Natural language query
- `actual_corpus_name` (Optional[str]): Target corpus name

**Returns:**
- Dict containing:
  - `generated_answer` (str): Natural language answer
  - `retrieved_context` (Dict): Retrieved information
  - `execution_plan` (ExecutionPlan): Generated plan

### AgentOrchestrator

Executes agent plans and manages tool execution.

```python
from Core.AgentOrchestrator.orchestrator import AgentOrchestrator

orchestrator = AgentOrchestrator(context: Optional[GraphRAGContext] = None)
```

#### Methods

##### `async execute_plan(plan: ExecutionPlan) -> Dict[str, Any]`

Executes a complete execution plan.

##### `async execute_tool(tool_id: str, inputs: Dict[str, Any]) -> Dict[str, Any]`

Executes a single tool with given inputs.

## Agent Tools

### Corpus Tools

#### `corpus.PrepareFromDirectory`

Prepares corpus from directory of text files.

**Inputs:**
- `input_directory_path` (str): Path to text files
- `output_directory_path` (str): Output path
- `target_corpus_name` (str): Corpus name

**Outputs:**
- `corpus_json_path` (str): Path to generated Corpus.json

### Graph Construction Tools

#### `graph.BuildERGraph`

Builds Entity-Relationship graph.

**Inputs:**
- `target_dataset_name` (str): Dataset name
- `custom_ontology_path` (Optional[str]): Custom ontology

**Outputs:**
- `graph_id` (str): Graph identifier
- `entity_count` (int): Number of entities
- `relationship_count` (int): Number of relationships

#### `graph.BuildTreeGraph`

Builds hierarchical tree graph.

**Inputs:**
- `target_dataset_name` (str): Dataset name
- `max_depth` (int): Maximum tree depth
- `branch_factor` (int): Branching factor

**Outputs:**
- `graph_id` (str): Graph identifier
- `tree_depth` (int): Actual depth
- `node_count` (int): Number of nodes

### Retrieval Tools

#### `Entity.VDB.Search`

Searches entity vector database.

**Inputs:**
- `query` (str): Search query
- `top_k` (int): Number of results
- `vdb_collection_name` (str): Collection name

**Outputs:**
- `entities` (List[Dict]): Found entities with scores

#### `Entity.OneHopNeighbors`

Gets one-hop neighbors of entity.

**Inputs:**
- `entity_name` (str): Entity name
- `graph_reference_id` (str): Graph ID

**Outputs:**
- `neighbors` (List[Dict]): Neighboring entities
- `relationships` (List[Dict]): Connecting relationships

## Configuration

### LLMConfig

```python
from Config.LLMConfig import LLMConfig

config = LLMConfig(
    api_type="litellm",
    model="openai/gpt-4",
    api_key="your-key",
    temperature=0.0,
    max_token=2000
)
```

### GraphRAGContext

```python
from Core.AgentSchema.context import GraphRAGContext

context = GraphRAGContext(
    corpus_name="MySampleTexts",
    dataset_name="MySampleTexts",
    graph_id="er_graph_123",
    vdb_collection_name="entities_vdb"
)
```

## Error Handling

All agent tools use structured error responses:

```python
{
    "error": "Error type",
    "message": "Detailed error message",
    "details": {...}  # Additional context
}
```

## Async/Await Usage

All agent operations are asynchronous:

```python
import asyncio

async def main():
    agent = PlanningAgent(orchestrator)
    result = await agent.process_query("Your query")
    print(result['generated_answer'])

asyncio.run(main())
```

## Logging

Configure logging using environment variables:

```bash
export DIGIMON_LOG_LEVEL=INFO
export DIGIMON_LOG_FILE=digimon.log
export DIGIMON_LOG_STRUCTURED=true
```

Or programmatically:

```python
from Core.Common.LoggerConfig import setup_logger

logger = setup_logger(
    name=__name__,
    level="DEBUG",
    log_file="agent.log",
    structured=True
)
```