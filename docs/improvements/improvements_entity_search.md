# Entity Search Improvements

## Problem: Low VDB Search Relevance

Current issues:
- "Crystal Plague" query doesn't find "the crystal plague" entity
- "Emperor Zorthak III" doesn't find the entity despite it existing in text
- "Aerophantis" descriptions aren't retrieved

## Solutions:

### 1. Entity Name Normalization
- Normalize entity names during graph building
- Remove articles ("the", "a") and normalize case
- Create aliases for entities (e.g., "Crystal Plague" → ["the crystal plague", "great crystal plague"])

### 2. Enhanced Query Expansion  
- Expand user queries with synonyms before VDB search
- Use LLM to generate related terms
- Example: "Crystal Plague" → ["crystal plague", "plague", "disease", "corruption", "levitite corruption"]

### 3. Multi-Strategy Entity Search
Instead of just VDB search, use:
- Fuzzy string matching on entity names
- Keyword search within entity descriptions
- Hierarchical search (find broader concepts first)

### 4. Improved Embedding Strategy
- Fine-tune embeddings on domain-specific text
- Use multiple embedding models and ensemble results
- Store multiple representations per entity (name, description, context)

## Implementation Priority:
1. Query expansion (quick win)
2. Entity normalization (medium effort)
3. Multi-strategy search (medium effort)  
4. Embedding improvements (longer term)