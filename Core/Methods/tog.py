"""Think-on-Graph reference plan.

The plan is explicitly unrolled to the requested depth so each hop consumes the
entities selected by the previous hop. Selected relationship evidence is merged
across hops before source-chunk retrieval, keeping multi-hop answers grounded in
the full explored chain rather than only the final edge.
"""

from Core.AgentSchema.plan import (
    DynamicToolChainConfig,
    ExecutionPlan,
    ExecutionStep,
    ToolCall,
    ToolInputSource,
)
from Core.Operators.relationship.merge import ensure_relationship_merge_registered


def tog_plan(query: str, **kwargs) -> ExecutionPlan:
    ensure_relationship_merge_registered()
    depth = max(1, int(kwargs.get("depth", 3)))
    width = max(1, int(kwargs.get("width", 3)))

    steps = [
        ExecutionStep(
            step_id="seed_entities",
            description="Extract and link seed entities from the query",
            action=DynamicToolChainConfig(
                tools=[
                    ToolCall(
                        tool_id="meta.extract_entities",
                        inputs={"query": "plan_inputs.query"},
                        named_outputs={"entities": "entity_set"},
                    ),
                    ToolCall(
                        tool_id="entity.link",
                        inputs={
                            "entities": ToolInputSource(
                                from_step_id="seed_entities",
                                named_output_key="entities",
                            )
                        },
                        named_outputs={"entities": "entity_set"},
                    ),
                ]
            ),
        )
    ]

    previous_entity_step = "seed_entities"
    accumulated_relationship_step = None

    for hop in range(1, depth + 1):
        relationship_step = f"hop_{hop}_relationships"
        candidate_step = f"hop_{hop}_candidates"
        entity_step = f"hop_{hop}_entities"

        steps.append(
            ExecutionStep(
                step_id=relationship_step,
                description=f"Hop {hop}: choose relevant graph relations",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="relationship.agent",
                            parameters={"width": width},
                            inputs={
                                "query": "plan_inputs.query",
                                "entities": ToolInputSource(
                                    from_step_id=previous_entity_step,
                                    named_output_key="entities",
                                ),
                            },
                            named_outputs={"relationships": "relationship_set"},
                        )
                    ]
                ),
            )
        )

        if accumulated_relationship_step is None:
            accumulated_relationship_step = relationship_step
        else:
            history_step = f"hop_{hop}_relationship_history"
            steps.append(
                ExecutionStep(
                    step_id=history_step,
                    description=f"Hop {hop}: accumulate selected relationship evidence",
                    action=DynamicToolChainConfig(
                        tools=[
                            ToolCall(
                                tool_id="relationship.merge",
                                inputs={
                                    "left": ToolInputSource(
                                        from_step_id=accumulated_relationship_step,
                                        named_output_key="relationships",
                                    ),
                                    "right": ToolInputSource(
                                        from_step_id=relationship_step,
                                        named_output_key="relationships",
                                    ),
                                },
                                named_outputs={"relationships": "relationship_set"},
                            )
                        ]
                    ),
                )
            )
            accumulated_relationship_step = history_step

        steps.append(
            ExecutionStep(
                step_id=candidate_step,
                description=f"Hop {hop}: adapt scored relations into entity candidates",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="entity.rel_node",
                            inputs={
                                "relationships": ToolInputSource(
                                    from_step_id=relationship_step,
                                    named_output_key="relationships",
                                )
                            },
                            named_outputs={"entities": "entity_set"},
                        )
                    ]
                ),
            )
        )

        steps.append(
            ExecutionStep(
                step_id=entity_step,
                description=f"Hop {hop}: score and select next entity candidates",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="entity.agent",
                            parameters={"width": width},
                            inputs={
                                "query": "plan_inputs.query",
                                "entity_relation_list": ToolInputSource(
                                    from_step_id=candidate_step,
                                    named_output_key="entities",
                                ),
                            },
                            named_outputs={"entities": "entity_set"},
                        )
                    ]
                ),
            )
        )

        previous_entity_step = entity_step

    steps.extend(
        [
            ExecutionStep(
                step_id="evidence_chunks",
                description="Retrieve source chunks for all selected reasoning hops",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="chunk.from_relation",
                            inputs={
                                "relationships": ToolInputSource(
                                    from_step_id=accumulated_relationship_step,
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
                description="Generate an answer from the accumulated source evidence",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="meta.generate_answer",
                            inputs={
                                "query": "plan_inputs.query",
                                "chunks": ToolInputSource(
                                    from_step_id="evidence_chunks",
                                    named_output_key="chunks",
                                ),
                            },
                            named_outputs={"answer": "text"},
                        )
                    ]
                ),
            ),
        ]
    )

    return ExecutionPlan(
        plan_description=(
            f"ToG: linked seeds -> {depth} explicit relation/entity exploration hops -> "
            "accumulated source evidence -> answer"
        ),
        target_dataset_name=kwargs.get("dataset", ""),
        plan_inputs={"query": query},
        steps=steps,
    )
