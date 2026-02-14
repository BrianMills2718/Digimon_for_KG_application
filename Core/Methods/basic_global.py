"""Basic Global query: Community by level -> LLM map/reduce.

Corresponds to BasicQuery._retrieve_relevant_contexts_global.
"""

from Core.AgentSchema.plan import (
    DynamicToolChainConfig,
    ExecutionPlan,
    ExecutionStep,
    ToolCall,
    ToolInputSource,
)


def basic_global_plan(query: str, **kwargs) -> ExecutionPlan:
    return ExecutionPlan(
        plan_description="Basic Global: Community reports by level -> answer generation",
        target_dataset_name=kwargs.get("dataset", ""),
        plan_inputs={"query": query},
        steps=[
            ExecutionStep(
                step_id="s1",
                description="Retrieve community reports by hierarchy level",
                action=DynamicToolChainConfig(tools=[
                    ToolCall(
                        tool_id="community.from_level",
                        inputs={},
                        named_outputs={"communities": "community_set"},
                    ),
                ]),
            ),
        ],
    )
