"""Basic Local reference plan: entity VDB -> local chunk evidence -> answer."""

from Core.AgentSchema.plan import (
    DynamicToolChainConfig,
    ExecutionPlan,
    ExecutionStep,
    ToolCall,
    ToolInputSource,
)


def basic_local_plan(query: str, **kwargs) -> ExecutionPlan:
    return ExecutionPlan(
        plan_description="Basic Local: VDB entities -> local co-occurrence evidence -> answer",
        target_dataset_name=kwargs.get("dataset", ""),
        plan_inputs={"query": query},
        steps=[
            ExecutionStep(
                step_id="entities",
                description="Find entities similar to the query",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="entity.vdb",
                            inputs={"query": "plan_inputs.query"},
                            named_outputs={"entities": "entity_set"},
                        )
                    ]
                ),
            ),
            ExecutionStep(
                step_id="evidence",
                description="Retrieve source chunks associated with local entity relations",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="chunk.occurrence",
                            inputs={
                                "entities": ToolInputSource(
                                    from_step_id="entities",
                                    named_output_key="entities",
                                )
                            },
                            named_outputs={"chunks": "chunk_set"},
                        )
                    ]
                ),
            ),
            ExecutionStep(
                step_id="answer",
                description="Generate an answer from local source evidence",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="meta.generate_answer",
                            inputs={
                                "query": "plan_inputs.query",
                                "chunks": ToolInputSource(
                                    from_step_id="evidence",
                                    named_output_key="chunks",
                                ),
                            },
                            named_outputs={"answer": "text"},
                        )
                    ]
                ),
            ),
        ],
    )
