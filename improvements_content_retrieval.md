# Content Retrieval Improvements

## Problem: Missing Relevant Content

Even when entities are found, the system often misses related content because:
- Only direct entity text is retrieved
- Relationship context is ignored
- No semantic expansion of search terms

## Solutions:

### 1. Comprehensive Context Assembly
Don't just get entity text - build complete context:

```python
async def get_comprehensive_context(entities: List[str], graph: BaseGraph) -> Dict[str, Any]:
    context = {
        "primary_entities": {},
        "related_entities": {},
        "relationships": {},
        "text_chunks": {}
    }
    
    # 1. Get primary entity text
    for entity in entities:
        context["primary_entities"][entity] = await get_entity_text(entity)
    
    # 2. Find related entities (1-hop, 2-hop)
    related = await get_related_entities(entities, max_hops=2)
    for entity in related:
        context["related_entities"][entity] = await get_entity_text(entity)
    
    # 3. Get relationship descriptions
    relationships = await get_relationships_between(entities + related)
    context["relationships"] = relationships
    
    # 4. Get chunks containing any relevant entities
    all_entities = entities + related
    chunks = await get_chunks_containing_entities(all_entities)
    context["text_chunks"] = chunks
    
    return context
```

### 2. Semantic Text Expansion
Find text that's semantically related even without exact entity matches:

```python
async def get_semantic_context(query: str, initial_entities: List[str]) -> List[str]:
    # Use embeddings to find text similar to the query
    query_embedding = await embed_text(query)
    
    # Search all chunks for semantic similarity
    similar_chunks = await vector_search_chunks(query_embedding, top_k=10)
    
    # Filter chunks that contain related concepts
    relevant_chunks = []
    for chunk in similar_chunks:
        if any(entity.lower() in chunk.content.lower() for entity in initial_entities):
            relevant_chunks.append(chunk)
        elif semantic_similarity(query, chunk.content) > 0.7:
            relevant_chunks.append(chunk)
    
    return relevant_chunks
```

### 3. Hierarchical Information Gathering
Use different strategies based on query complexity:

**Simple queries ("What is X?"):**
- Direct entity lookup
- Basic relationship context

**Complex queries ("What caused Y?"):**
- Multi-hop relationship traversal
- Temporal relationship analysis
- Causal chain reconstruction

**Historical queries ("Who was X?"):**
- Biographical entity search
- Event participation analysis
- Social network context

### 4. Redundant Retrieval Strategies
Use multiple retrieval methods and combine results:
- VDB search on entities
- Keyword search on text
- Graph traversal
- Semantic similarity search

## Implementation Priority:
1. Comprehensive context assembly (high impact)
2. Semantic text expansion (medium impact)
3. Hierarchical strategies (medium impact)
4. Redundant retrieval (lower priority)