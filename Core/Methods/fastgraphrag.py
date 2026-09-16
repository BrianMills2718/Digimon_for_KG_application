"""FastGraphRAG reference plan: VDB seeds -> PPR -> sparse propagation -> answer."""

from Core.AgentSchema.plan import (
    DynamicToolChainConfig,
    ExecutionPlan,
    ExecutionStep,
    ToolCall,
    ToolInputSource,
)


def fastgraphrag_plan(query: str, **kwargs) -> ExecutionPlan:
    return ExecutionPlan(
        plan_description=(
            "FastGraphRAG: VDB seeds -> PPR -> entity/relationship/chunk score "
            "propagation -> answer"
        ),
        target_dataset_name=kwargs.get("dataset", ""),
        plan_inputs={"query": query},
        steps=[
            ExecutionStep(
                step_id="seed_entities",
                description="Find seed entities by vector similarity",
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
                step_id="ppr",
                description="Diffuse relevance through graph topology",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="entity.ppr",
                            inputs={
                                "query": "plan_inputs.query",
                                "entities": ToolInputSource(
                                    from_step_id="seed_entities",
                                    named_output_key="entities",
                                ),
                            },
                            named_outputs={
                                "entities": "entity_set",
                                "score_vector": "score_vector",
                            },
                        )
                    ]
                ),
            ),
            ExecutionStep(
                step_id="evidence",
                description="Propagate graph scores through relationships to source chunks",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="chunk.aggregator",
                            inputs={
                                "score_vector": ToolInputSource(
                                    from_step_id="ppr",
                                    named_output_key="score_vector",
                                )
                            },
                            named_outputs={"chunks": "chunk_set"},
                        )
                    ]
                ),
            ),
            ExecutionStep(
                step_id="answer",
                description="Generate answer from propagated source evidence",
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
