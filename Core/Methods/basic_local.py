"""Basic Local query: Entity VDB -> Relationship onehop -> Chunk occurrence.

Corresponds to BasicQuery._retrieve_relevant_contexts_local.
"""

from Core.AgentSchema.plan import (
    DynamicToolChainConfig,
    ExecutionPlan,
    ExecutionStep,
    ToolCall,
    ToolInputSource,
)


def basic_local_plan(query: str, **kwargs) -> ExecutionPlan:
    return ExecutionPlan(
        plan_description="Basic Local: VDB entities -> one-hop relationships -> chunk co-occurrence",
        target_dataset_name=kwargs.get("dataset", ""),
        plan_inputs={"query": query},
        steps=[
            ExecutionStep(
                step_id="s1",
                description="Find entities similar to query via VDB",
                action=DynamicToolChainConfig(tools=[
                    ToolCall(
                        tool_id="entity.vdb",
                        inputs={"query": "plan_inputs.query"},
                        named_outputs={"entities": "entity_set"},
                    ),
                ]),
            ),
            ExecutionStep(
                step_id="s2",
                description="Find relationships connected to retrieved entities",
                action=DynamicToolChainConfig(tools=[
                    ToolCall(
                        tool_id="relationship.onehop",
                        inputs={"entities": ToolInputSource(from_step_id="s1", named_output_key="entities")},
                        named_outputs={"relationships": "relationship_set"},
                    ),
                ]),
            ),
            ExecutionStep(
                step_id="s3",
                description="Find text chunks where entities co-occur",
                action=DynamicToolChainConfig(tools=[
                    ToolCall(
                        tool_id="chunk.occurrence",
                        inputs={"entities": ToolInputSource(from_step_id="s1", named_output_key="entities")},
                        named_outputs={"chunks": "chunk_set"},
                    ),
                ]),
            ),
        ],
    )
