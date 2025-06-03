# Handoff Document: DIGIMON ReACT Loop Debugging

**Date:** 2025-06-02
**Current Objective:** Debug and enhance the DIGIMON pipeline's ReACT loop within `Core/AgentBrain/agent_brain.py` (`PlanningAgent`). The goal is to enable robust multi-step reasoning and tool execution for answering complex queries, using `testing/test_react_with_existing_data.py` as the primary test case.

## Current Status

The ReACT loop is partially functional. It can:
- Initialize with pre-built resources (ER Graph and VDB).
- Generate a sequence of reasoning steps and actions via an LLM.
- Identify actions that require tool execution.

However, issues remain in the translation of these actions into fully executable plans (specifically with input mapping) and in handling errors gracefully during tool execution.

## Achievements in This Session

1.  **Corrected Tool Invocation Identification**:
    *   **Issue**: LLM-generated actions (e.g., "Call the search_vector_db tool...") were misclassified as "Non-tool steps" because they didn't match the expected format (`startswith("Use ")` or specific tool prefixes like `corpus.`).
    *   **Fix**: The system prompt within `PlanningAgent._react_reason` in `Core/AgentBrain/agent_brain.py` was updated. It now explicitly instructs the LLM that if its `next_step` involves calling a tool, the description **MUST** start with "Use " (e.g., "Use search_vector_db with ...").
    *   **Result**: The agent now correctly identifies these actions as tool steps and attempts to translate and execute them.

## Current Blockers & Detailed Next Steps

### Blocker 1: Input Resolution for `Relationship.OneHopNeighbors` Tool

*   **Symptom**: During the execution of `testing/test_react_with_existing_data.py`, the `Relationship.OneHopNeighbors` tool fails. The logs show:
    ```
    ERROR Core.AgentOrchestrator.orchestrator:_resolve_single_input_source:149 - Orchestrator (input field) 'entity_ids': Output key 'search_results' not in step 'step_1_vdb_search_zorathian'. Available: []
    ERROR Core.AgentOrchestrator.orchestrator:execute_plan:359 - Orchestrator: Error during execution of tool Relationship.OneHopNeighbors in step step_1_onehop_neighbors. Exception: 1 validation error for RelationshipOneHopNeighborsInputs entity_ids Input should be a valid list [type=list_type, input_value=None, input_type=NoneType]
    ```
*   **Root Cause Analysis**:
    1.  The `Entity.VDBSearch` tool (defined by `EntityVDBSearchOutputs` in `Core/AgentSchema/tool_contracts.py`) returns its output in a field named `similar_entities` (which is a `List[VDBSearchResultItem]`)
    2.  The `Relationship.OneHopNeighbors` tool (defined by `RelationshipOneHopNeighborsInputs` in `Core/AgentSchema/tool_contracts.py`) expects an input field `entity_ids` as a `List[str]`.
    3.  The LLM, when generating the `next_step` for `Relationship.OneHopNeighbors` (e.g., `Use Relationship.OneHopNeighbors with graph_name=Fictional_Test_ERGraph, from_step_id=step_1_vdb_search_zorathian`), does not specify *how* to map the output of the VDB search step (`step_1_vdb_search_zorathian`) to the `entity_ids` input.
    4.  The `AgentOrchestrator._resolve_single_input_source` method was looking for an output field named `search_results` from the previous step, which doesn't match the actual output field `similar_entities`.
*   **Proposed Next Steps for Blocker 1**:
    1.  **Modify ReACT Reasoning Prompt for Explicit Input Mapping**:
        *   **File**: `Core/AgentBrain/agent_brain.py`
        *   **Method**: `PlanningAgent._react_reason`
        *   **Action**: Update the system prompt to instruct the LLM to explicitly specify input mappings from previous steps using a `step_id.output_field_name` convention. For example, the `next_step` for `Relationship.OneHopNeighbors` should look like: `"Use Relationship.OneHopNeighbors with graph_name=Fictional_Test_ERGraph, entity_ids=step_1_vdb_search_zorathian.similar_entities"` (Note: `similar_entities` is the actual output field from `EntityVDBSearchOutputs`).
    2.  **Verify/Enhance Orchestrator's Input Resolution Logic**:
        *   **File**: `Core/AgentOrchestrator/orchestrator.py`
        *   **Method**: `_resolve_single_input_source`
        *   **Action**:
            *   Ensure this method can correctly parse and use `step_id.output_field_name` references for input values.
            *   Crucially, ensure it can transform the `List[VDBSearchResultItem]` (from `similar_entities`) into the `List[str]` (entity names) expected by `RelationshipOneHopNeighborsInputs.entity_ids`. This might involve iterating through the `VDBSearchResultItem` list and extracting the `entity_name` from each item. (Refer to MEMORY[99475c86-2aab-4848-a6bf-d92e667e19d9] which suggests prior work on `entity_name` extraction).

### Blocker 2: Error Handling in ReACT Loop for Failed Tool Execution

*   **Symptom**: When the `Relationship.OneHopNeighbors` tool execution fails (due to Blocker 1), a subsequent `AttributeError` occurs in the ReACT loop:
    ```
    ERROR Core.AgentBrain.agent_brain:process_query_react:1034 - ReAct: Error executing step: 'dict' object has no attribute 'step_results'
    ```
*   **Root Cause Analysis**:
    *   When `AgentOrchestrator.execute_plan()` encounters an error during a tool's execution (like the Pydantic validation error for `entity_ids`), it appears to return a simple Python `dict` (likely containing error details) instead of the standard `PlanResult` object.
    *   The `PlanningAgent.process_query_react` method expects a `PlanResult` object and attempts to access `results.step_results`, which does not exist on a simple `dict`, leading to the `AttributeError`.
*   **Proposed Next Steps for Blocker 2**:
    1.  **Investigate Orchestrator's Error Return Value**:
        *   **File**: `Core/AgentOrchestrator/orchestrator.py`
        *   **Method**: `execute_plan`
        *   **Action**: Determine precisely what `execute_plan` returns when a tool execution fails internally (e.g., due to input validation errors or exceptions within the tool itself).
    2.  **Implement Robust Error Handling in `process_query_react`**:
        *   **File**: `Core/AgentBrain/agent_brain.py`
        *   **Method**: `PlanningAgent.process_query_react` (around line 1034, where `results` from `execute_plan` is processed).
        *   **Action**: Modify the logic to check the type of `results` returned by `execute_plan`. If it's not a `PlanResult` (e.g., if it's a `dict` indicating an error), it should still extract the error information and create a proper "observation" for the ReACT loop, rather than crashing. This observation should indicate the step failure and the reason.

## Key Files & Context

*   **Primary Logic**: `Core/AgentBrain/agent_brain.py` (contains `PlanningAgent` with `process_query_react`, `_react_reason`, `_translate_nl_step_to_pydantic`).
*   **Tool Definitions/Contracts**: `Core/AgentSchema/tool_contracts.py` (Pydantic models for tool inputs/outputs like `EntityVDBSearchOutputs`, `RelationshipOneHopNeighborsInputs`).
*   **Orchestration**: `Core/AgentOrchestrator/orchestrator.py` (handles plan execution and input resolution via `execute_plan`, `_resolve_single_input_source`).
*   **Test Script**: `testing/test_react_with_existing_data.py`.
*   **Changelog**: `Doc/change_log.md` (should be updated with any fixes).

## LLM Interaction Points

*   The system prompt in `PlanningAgent._react_reason` is critical for guiding the LLM on how to:
    *   Reason about the next step.
    *   Format the `next_step` string, especially for tool invocations (using "Use " prefix and correct input mapping like `step_id.output_field_name`).
*   The system prompt in `PlanningAgent._translate_nl_step_to_pydantic` is used to translate the LLM's `next_step` string into a structured `ExecutionPlan`.

By addressing these two blockers, the ReACT loop should become significantly more robust and capable of executing multi-step plans involving tool dependencies.
