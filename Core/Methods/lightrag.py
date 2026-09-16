"""LightRAG reference plan: relationship VDB -> source chunks -> answer."""

from Core.AgentSchema.plan import (
    DynamicToolChainConfig,
    ExecutionPlan,
    ExecutionStep,
    ToolCall,
    ToolInputSource,
)


def lightrag_plan(query: str, **kwargs) -> ExecutionPlan:
    return ExecutionPlan(
        plan_description="LightRAG: relationship VDB retrieval -> source evidence -> answer",
        target_dataset_name=kwargs.get("dataset", ""),
        plan_inputs={"query": query},
        steps=[
            ExecutionStep(
                step_id="relationships",
                description="Search relationship descriptions/keywords by semantic similarity",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="relationship.vdb",
                            inputs={"query": "plan_inputs.query"},
                            named_outputs={"relationships": "relationship_set"},
                        )
                    ]
                ),
            ),
            ExecutionStep(
                step_id="evidence",
                description="Retrieve original chunks supporting the selected relationships",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="chunk.from_relation",
                            inputs={
                                "relationships": ToolInputSource(
                                    from_step_id="relationships",
                                    named_output_key="relationships",
                                )
                            },
                            named_outputs={"chunks": "chunk_set"},
                        )
                    ]
                ),
            ),
            ExecutionStep(
                step_id="answer",
                description="Generate answer from relationship-grounded evidence",
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
