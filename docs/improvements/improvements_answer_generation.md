# Answer Generation Improvements

## Problem: Poor Context Utilization

The LLM often says "no information found" even when relevant content is retrieved.

## Root Causes:
1. Context is passed in wrong format
2. Too much irrelevant information drowns out relevant content  
3. Context lacks structure and priority indicators
4. LLM prompt doesn't emphasize using all available information

## Solutions:

### 1. Structured Context Presentation
Instead of dumping all context, structure it by relevance:

```python
def build_structured_context(query: str, retrieved_data: Dict) -> str:
    context_parts = []
    
    # 1. Direct entity information (highest priority)
    if retrieved_data.get("primary_entities"):
        context_parts.append("## Primary Information:")
        for entity, text in retrieved_data["primary_entities"].items():
            context_parts.append(f"**{entity}**: {text}")
    
    # 2. Related context (medium priority)  
    if retrieved_data.get("relationships"):
        context_parts.append("## Related Information:")
        for rel in retrieved_data["relationships"]:
            context_parts.append(f"- {rel['description']}")
    
    # 3. Supporting text (lower priority)
    if retrieved_data.get("text_chunks"):
        context_parts.append("## Supporting Context:")
        for chunk in retrieved_data["text_chunks"][:3]:  # Limit to top 3
            context_parts.append(f"- {chunk['content'][:200]}...")
    
    return "\n\n".join(context_parts)
```

### 2. Enhanced LLM Prompts
Use more directive prompts that force the LLM to use available information:

```python
ENHANCED_SYSTEM_PROMPT = """You are an expert analyst. Your task is to answer questions using ONLY the provided context.

CRITICAL INSTRUCTIONS:
1. You MUST use information from the context if it's relevant to the question
2. If the context contains relevant information, you CANNOT say "no information available"
3. Synthesize information from multiple sources in the context
4. Be specific and cite details from the context
5. If truly no relevant information exists, explain what information IS available instead

The context is structured by priority:
- Primary Information: Direct answers to the query
- Related Information: Connected concepts and relationships  
- Supporting Context: Additional background information

Question: {query}

Context:
{structured_context}

Instructions for your response:
- Start with the most direct answer from Primary Information
- Expand with details from Related Information
- Use Supporting Context for additional depth
- If information is incomplete, state what IS known rather than what ISN'T
"""
```

### 3. Multi-Pass Answer Generation
Generate answers in multiple passes:

**Pass 1 - Direct Answer:**
- Use only primary entity information
- Generate core answer

**Pass 2 - Context Enhancement:**
- Add relationship and supporting information
- Expand and enrich the answer

**Pass 3 - Completeness Check:**
- Verify all relevant context was used
- Add any missed important details

### 4. Answer Validation and Retry
Validate generated answers and retry if needed:

```python
def validate_answer(answer: str, context: Dict, query: str) -> bool:
    issues = []
    
    # Check if answer claims no information exists
    if "no information" in answer.lower() or "not mentioned" in answer.lower():
        # But context actually has relevant information
        if has_relevant_context(context, query):
            issues.append("false_negative")
    
    # Check if answer is too generic
    if len(answer) < 50 and len(context.get("primary_entities", {})) > 0:
        issues.append("too_generic")
    
    # Check if specific details from context are used
    specific_details = extract_specific_details(context)
    if not any(detail.lower() in answer.lower() for detail in specific_details):
        issues.append("missing_specifics")
    
    return len(issues) == 0

async def generate_answer_with_retry(query: str, context: Dict) -> str:
    for attempt in range(3):
        answer = await generate_answer(query, context, attempt=attempt)
        
        if validate_answer(answer, context, query):
            return answer
        
        # Enhance prompt for retry
        context = enhance_context_for_retry(context, attempt)
    
    return answer  # Return best attempt
```

## Implementation Priority:
1. Structured context presentation (high impact, low effort)
2. Enhanced LLM prompts (high impact, low effort)  
3. Answer validation and retry (medium impact, medium effort)
4. Multi-pass generation (lower priority, higher effort)