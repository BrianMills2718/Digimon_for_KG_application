"""HippoRAG reference plan: extract/link -> PPR -> source chunks -> answer."""

from Core.AgentSchema.plan import (
    DynamicToolChainConfig,
    ExecutionPlan,
    ExecutionStep,
    ToolCall,
    ToolInputSource,
)


def hipporag_plan(query: str, **kwargs) -> ExecutionPlan:
    return ExecutionPlan(
        plan_description=(
            "HippoRAG: LLM entity extraction -> entity linking -> IDF-aware PPR -> "
            "source evidence -> answer"
        ),
        target_dataset_name=kwargs.get("dataset", ""),
        plan_inputs={"query": query},
        steps=[
            ExecutionStep(
                step_id="extracted_entities",
                description="Extract query entity mentions",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="meta.extract_entities",
                            inputs={"query": "plan_inputs.query"},
                            named_outputs={"entities": "entity_set"},
                        )
                    ]
                ),
            ),
            ExecutionStep(
                step_id="linked_entities",
                description="Link extracted mentions to canonical graph entities",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="entity.link",
                            inputs={
                                "entities": ToolInputSource(
                                    from_step_id="extracted_entities",
                                    named_output_key="entities",
                                )
                            },
                            named_outputs={"entities": "entity_set"},
                        )
                    ]
                ),
            ),
            ExecutionStep(
                step_id="ppr",
                description="Diffuse linked-entity relevance through graph topology",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="entity.ppr",
                            inputs={
                                "query": "plan_inputs.query",
                                "entities": ToolInputSource(
                                    from_step_id="linked_entities",
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
                description="Propagate graph relevance into source chunks",
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
                description="Generate answer from HippoRAG evidence",
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
