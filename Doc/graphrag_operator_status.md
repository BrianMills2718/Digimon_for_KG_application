# GraphRAG Operators Implementation Status

## Current Status (✅ = Implemented, ⚠️ = Partial, ❌ = Not Implemented)

### Entity Operators
| Operator | Status | DIGIMON Implementation | Description |
|----------|--------|----------------------|-------------|
| VDB | ✅ | `Entity.VDBSearch` | Select top-k nodes from vector database |
| PPR | ✅ | `Entity.PPR` | Run PPR on graph, return top-k nodes with scores |
| RelNode | ✅ | `Entity.RelNode` | Extract nodes from given relationships |
| Agent | ❌ | - | Use LLM to find useful entities |
| Onehop | ✅ | `Entity.Onehop` | Select one-hop neighbor entities |
| Link | ❌ | - | Return top-1 similar entity for each given entity |
| TF-IDF | ❌ | - | Rank entities based on TF-IDF matrix |

### Relationship Operators
| Operator | Status | DIGIMON Implementation | Description |
|----------|--------|----------------------|-------------|
| Onehop | ✅ | `Relationship.OneHopNeighbors` | Select relationships linked by one-hop neighbors |
| VDB | ❌ | - | Retrieve relationships by vector database |
| Aggregator | ❌ | - | Compute relationship scores from entity PPR matrix |
| Agent | ❌ | - | Use LLM to find useful relationships |

### Chunk Operators
| Operator | Status | DIGIMON Implementation | Description |
|----------|--------|----------------------|-------------|
| FromRel | ⚠️ | `Chunk.FromRelationships` (commented) | Return chunks containing given relationships |
| Aggregator | ❌ | - | Use relationship scores to select top-k chunks |
| Occurrence | ❌ | - | Rank chunks by entity occurrence |

### Subgraph Operators
| Operator | Status | DIGIMON Implementation | Description |
|----------|--------|----------------------|-------------|
| KhopPath | ❌ | - | Find k-hop paths between entities |
| Steiner | ❌ | - | Compute Steiner tree |
| AgentPath | ❌ | - | LLM-filtered k-hop paths |

### Community Operators
| Operator | Status | DIGIMON Implementation | Description |
|----------|--------|----------------------|-------------|
| Entity | ❌ | - | Detect communities containing entities |
| Layer | ❌ | - | Return communities below required layer |

### Graph Construction Tools (Additional)
| Tool | Status | Description |
|------|--------|-------------|
| BuildERGraph | ✅ | Build Entity-Relationship Graph (KG) |
| BuildRKGraph | ✅ | Build Rich Knowledge Graph (RKG) |
| BuildTreeGraph | ✅ | Build Tree Graph |
| BuildTreeGraphBalanced | ✅ | Build Balanced Tree Graph |
| BuildPassageGraph | ✅ | Build Passage Graph |

### Analysis & Visualization Tools (Additional)
| Tool | Status | Description |
|------|--------|-------------|
| graph.Visualize | ✅ | Export graphs in JSON/GML formats |
| graph.Analyze | ✅ | Calculate graph metrics |

## Implementation Priority Recommendation

Based on the GraphRAG methods' requirements and impact, here's the suggested implementation order:

### High Priority (Core Retrieval)
1. **Chunk.FromRel** - Complete the partially implemented tool
2. **Relationship.VDB** - Vector search for relationships
3. **Entity.Link** - Entity similarity matching
4. **Relationship.Aggregator** - PPR-based relationship scoring

### Medium Priority (Enhanced Retrieval)
5. **Chunk.Aggregator** - Score-based chunk selection
6. **Subgraph.KhopPath** - Multi-hop path finding
7. **Entity.Agent** - LLM-based entity discovery
8. **Relationship.Agent** - LLM-based relationship discovery

### Low Priority (Advanced Features)
9. **Subgraph.AgentPath** - LLM-filtered paths
10. **Subgraph.Steiner** - Steiner tree computation
11. **Community.Entity** - Community detection
12. **Community.Layer** - Hierarchical community search
13. **Entity.TF-IDF** - TF-IDF ranking
14. **Chunk.Occurrence** - Occurrence-based ranking

## Rationale for Priority

1. **High Priority** operators enable basic GraphRAG functionality used by multiple methods (HippoRAG, LightRAG, FastGraphRAG)
2. **Medium Priority** operators add sophisticated retrieval capabilities
3. **Low Priority** operators are either specialized (Communities for MS GraphRAG) or computationally expensive (Agent-based)

## Next Steps

1. Complete implementation of Chunk.FromRel (uncomment and test)
2. Implement Relationship.VDB operator
3. Continue with medium priority operators
