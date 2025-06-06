# DIGIMON Field Naming Mismatch Analysis & Fix Report

## Executive Summary

The tool input/output field naming mismatches in DIGIMON occur when the agent generates execution plans with field names that don't match the actual Pydantic contracts expected by the tools. This report provides a comprehensive analysis and specific fixes.

## Root Causes Identified

1. **Agent Brain System Prompt Issues**
   - The agent's system prompt includes examples with `corpus_ref` field for graph building tools
   - Graph building tools actually only need `target_dataset_name` (they load corpus internally)
   - The agent doesn't know the exact field names expected by each tool

2. **Pydantic Validation Behavior**
   - Extra fields are silently ignored (not forbidden by default)
   - This masks errors where agent passes wrong field names
   - Real errors occur when required fields are missing or have wrong types

3. **Output Reference Confusion**
   - Agent must use aliases defined in `named_outputs` when referencing step outputs
   - But the agent sometimes uses the original field name instead of the alias
   - Example: Using "corpus_json_path" instead of "prepared_corpus_name"

## Specific Issues Found

### 1. Graph Building Tools Don't Need corpus_ref

**Current (Wrong):**
```json
{
  "tool_id": "graph.BuildERGraph",
  "inputs": {
    "target_dataset_name": "MySampleTexts",
    "corpus_ref": {"from_step_id": "step_1", "named_output_key": "prepared_corpus_name"}
  }
}
```

**Correct:**
```json
{
  "tool_id": "graph.BuildERGraph",
  "inputs": {
    "target_dataset_name": "MySampleTexts"
  }
}
```

### 2. Common Field Mapping Errors

| Tool | Expected Input Fields | Common Mistakes |
|------|----------------------|-----------------|
| graph.BuildERGraph | target_dataset_name, force_rebuild, config_overrides | Adding corpus_ref |
| Entity.VDBSearch | vdb_reference_id, query_text, top_k_results | Using wrong reference names |
| Chunk.GetTextForEntities | graph_reference_id, entity_ids | entity_ids format issues |

### 3. Output Field References

When a step defines `named_outputs`:
```json
"named_outputs": {
  "my_custom_name": "actual_field_name"
}
```

Other steps MUST reference it as `my_custom_name`, not `actual_field_name`.

## Recommended Fixes

### 1. Update Agent Brain System Prompt

**File:** `Core/AgentBrain/agent_brain.py`

Remove the corpus_ref from graph building examples in the system prompt. The graphs internally use ChunkFactory to load the corpus based on target_dataset_name.

### 2. Add Field Validation Helper

Create a validation helper that checks if the agent's plan uses correct field names:

```python
def validate_tool_inputs(tool_id: str, inputs: dict, tool_registry: dict) -> tuple[bool, str]:
    """Validate that inputs match expected fields for a tool"""
    if tool_id not in tool_registry:
        return False, f"Unknown tool: {tool_id}"
    
    _, input_class = tool_registry[tool_id]
    required_fields = {f for f, info in input_class.model_fields.items() if info.is_required()}
    provided_fields = set(inputs.keys())
    
    missing = required_fields - provided_fields
    if missing:
        return False, f"Missing required fields: {missing}"
    
    # Check for None values in required fields
    for field in required_fields:
        if inputs.get(field) is None:
            return False, f"Required field '{field}' is None"
    
    return True, "Valid"
```

### 3. Improve Error Messages

In the orchestrator, when field resolution fails, provide helpful error messages:

```python
# In orchestrator._resolve_tool_inputs
if source_value is None:
    # Suggest the correct field name
    suggestions = self._suggest_field_names(target_input_name, available_outputs)
    logger.error(f"Could not resolve '{target_input_name}'. Available outputs: {available_outputs}. Did you mean: {suggestions}?")
```

### 4. Tool Input/Output Documentation

Generate and maintain a machine-readable mapping file (already created as `tool_field_mapping.json`) that the agent can reference.

## Implementation Priority

1. **High Priority:** Fix agent brain system prompt to remove corpus_ref from examples
2. **High Priority:** Add validation for None values in required fields
3. **Medium Priority:** Implement field name suggestions in error messages
4. **Low Priority:** Add strict validation mode option

## Testing Recommendations

1. Create unit tests for each tool with correct and incorrect field names
2. Test multi-step plans with output references
3. Verify error messages are helpful when fields don't match
4. Test with all graph types to ensure consistency

## Conclusion

The field naming mismatch issue is primarily caused by incorrect examples in the agent's system prompt and lack of clear error messages when field resolution fails. The fixes are straightforward and will significantly improve the reliability of the agent system.