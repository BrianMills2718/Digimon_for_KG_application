"""Basic Global reference plan: community reports -> answer synthesis."""

from Core.AgentSchema.plan import (
    DynamicToolChainConfig,
    ExecutionPlan,
    ExecutionStep,
    ToolCall,
    ToolInputSource,
)
from Core.Operators.community.materialize import ensure_community_materialize_registered


def basic_global_plan(query: str, **kwargs) -> ExecutionPlan:
    ensure_community_materialize_registered()

    return ExecutionPlan(
        plan_description="Basic Global: community reports -> global answer synthesis",
        target_dataset_name=kwargs.get("dataset", ""),
        plan_inputs={"query": query},
        steps=[
            ExecutionStep(
                step_id="communities",
                description="Retrieve relevant community reports by hierarchy level",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="community.from_level",
                            inputs={},
                            named_outputs={"communities": "community_set"},
                        )
                    ]
                ),
            ),
            ExecutionStep(
                step_id="community_evidence",
                description="Convert community reports into answer context",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="community.materialize",
                            inputs={
                                "communities": ToolInputSource(
                                    from_step_id="communities",
                                    named_output_key="communities",
                                )
                            },
                            named_outputs={"chunks": "chunk_set"},
                        )
                    ]
                ),
            ),
            ExecutionStep(
                step_id="answer",
                description="Synthesize a global answer from community reports",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="meta.generate_answer",
                            inputs={
                                "query": "plan_inputs.query",
                                "chunks": ToolInputSource(
                                    from_step_id="community_evidence",
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
