# DIGIMON KG-RAG: What It Does

## One-liner

Point Claude Code at a folder of text files, and it builds a knowledge graph, indexes it, and answers questions grounded in your source material.

## How It Works

You give Claude Code a goal and a folder of `.txt` files. Claude Code calls DIGIMON's tools in the right order to:

1. **Ingest** your documents into a structured corpus
2. **Build a knowledge graph** — extracting entities (people, orgs, places, concepts) and the relationships between them
3. **Index** those entities for fast semantic search
4. **Search and retrieve** relevant entities and source text based on your questions
5. **Synthesize** answers grounded in the actual documents

You don't manage any of this. You state what you want to know, Claude Code figures out the steps.

## What's in the Toolbox

### Corpus Preparation
- **corpus_prepare**: Turns a directory of `.txt` files into a structured corpus

### Graph Construction (5 types)
- **graph_build_er**: Entity-Relationship graph. Best general-purpose option — extracts named entities and how they relate.
- **graph_build_rk**: Relationship-Keyword graph. Like ER but enriches edges with keywords for better retrieval.
- **graph_build_tree**: Hierarchical summary tree (RAPTOR-style). Clusters and summarizes chunks at multiple levels.
- **graph_build_tree_balanced**: Balanced tree using K-Means. More uniform cluster sizes than basic tree.
- **graph_build_passage**: Passage graph. Nodes are text passages, linked when they share entities.

### Search and Retrieval
- **entity_vdb_build**: Build a vector index over graph entities (required before search)
- **entity_vdb_search**: Find entities relevant to a natural language query
- **entity_onehop**: Get direct neighbors of an entity in the graph
- **entity_ppr**: Personalized PageRank — find structurally important entities related to seed entities
- **relationship_onehop**: Get the relationships (edges) connected to given entities
- **chunk_get_text**: Retrieve the original source text associated with entities

### Analysis
- **graph_analyze**: Node count, edge count, centrality, clustering metrics
- **graph_visualize**: Export graph structure as JSON or GML
- **list_available_resources**: Show what graphs and indexes exist in the current session

## Typical Session

```
You: "I have 50 news articles about defense contracting in ~/data/defense/.
      Who are the key players and how are they connected?"

Claude Code:
  1. corpus_prepare(~/data/defense/, "defense_contracts")
  2. graph_build_er("defense_contracts")          → 500 entities, 380 edges
  3. entity_vdb_build("defense_contracts_ERGraph") → 500 entities indexed
  4. entity_vdb_search("key defense contractors")  → Lockheed Martin, Raytheon, ...
  5. relationship_onehop(["lockheed martin"])       → contracts with, lobbies, partners...
  6. chunk_get_text(["lockheed martin", ...])       → original article passages
  7. Synthesizes answer from all retrieved context
```

You see the final answer. The intermediate steps happen automatically.


