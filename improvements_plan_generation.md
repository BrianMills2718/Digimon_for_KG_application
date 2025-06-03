# Plan Generation Improvements

## Problem: Inconsistent and Incomplete Plans

Current issues:
- Plans vary significantly between similar queries
- Missing crucial steps like relationship traversal
- No fallback strategies when initial search fails

## Solutions:

### 1. Template-Based Planning
Create query-type specific templates:

```python
QUERY_TEMPLATES = {
    "what_is": {
        "pattern": r"what is|tell me about|describe",
        "steps": [
            "entity_search",
            "expand_related_entities", 
            "get_entity_text",
            "get_relationship_context"
        ]
    },
    "causal": {
        "pattern": r"what caused|why did|how did.*happen",
        "steps": [
            "entity_search",
            "find_causal_relationships",
            "trace_event_chain",
            "get_comprehensive_text"
        ]
    },
    "who_is": {
        "pattern": r"who (is|was)|tell me about.*person",
        "steps": [
            "person_entity_search",
            "get_biographical_context",
            "find_related_events",
            "get_relationship_text"
        ]
    }
}
```

### 2. Plan Validation and Repair
Before execution, validate plans:
- Ensure text retrieval steps are present
- Check that entity expansion is included for broad queries
- Add missing steps automatically

```python
def validate_and_repair_plan(plan: ExecutionPlan) -> ExecutionPlan:
    issues = []
    
    # Check for text retrieval
    has_text_step = any("text" in step.step_id.lower() for step in plan.steps)
    if not has_text_step:
        issues.append("missing_text_retrieval")
    
    # Check for entity expansion  
    has_expansion = any("expand" in step.step_id.lower() or "onehop" in step.step_id.lower() 
                       for step in plan.steps)
    if not has_expansion:
        issues.append("missing_entity_expansion")
    
    return repair_plan(plan, issues)
```

### 3. Multi-Stage Planning
For complex queries, break into sub-queries:
- Stage 1: Find primary entities
- Stage 2: Expand context based on what was found  
- Stage 3: Retrieve comprehensive text
- Stage 4: Generate answer with all context

### 4. Adaptive Planning
Monitor plan success and adapt:
- Track which plan patterns work for which query types
- Learn from failed executions
- Automatically retry with enhanced plans

## Implementation:
1. Implement template matching (high impact, medium effort)
2. Add plan validation (high impact, low effort)
3. Multi-stage planning (medium impact, high effort)
4. Adaptive learning (low priority, high effort)