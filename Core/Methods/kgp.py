"""KGP (KG-based Pathfinding) reference plan.

Each hop expands from the previously selected entities, retrieves source
evidence for those neighbors, refines the information need from that evidence,
and reranks the neighbor set with TF-IDF before continuing. Evidence is retained
across hops so the final grounded answer can cite the full pathfinding history,
not only the terminal entity set.
"""

from Core.AgentSchema.plan import (
    DynamicToolChainConfig,
    ExecutionPlan,
    ExecutionStep,
    ToolCall,
    ToolInputSource,
)
from Core.Operators.chunk.merge import ensure_chunk_merge_registered


def kgp_plan(query: str, **kwargs) -> ExecutionPlan:
    ensure_chunk_merge_registered()
    depth = max(1, int(kwargs.get("depth", 3)))
    top_k = max(1, int(kwargs.get("top_k", 5)))

    steps = [
        ExecutionStep(
            step_id="seed_entities",
            description="Find initial graph entities by TF-IDF",
            action=DynamicToolChainConfig(
                tools=[
                    ToolCall(
                        tool_id="entity.tfidf",
                        inputs={"query": "plan_inputs.query"},
                        parameters={"top_k": top_k},
                        named_outputs={"entities": "entity_set"},
                    )
                ]
            ),
        )
    ]

    previous_entity_step = "seed_entities"
    previous_query_source = "plan_inputs.query"
    accumulated_evidence_step = None

    for hop in range(1, depth + 1):
        expand_step = f"hop_{hop}_expand"
        evidence_step = f"hop_{hop}_evidence"
        reason_step = f"hop_{hop}_reason"
        rerank_step = f"hop_{hop}_rerank"

        steps.append(
            ExecutionStep(
                step_id=expand_step,
                description=f"Hop {hop}: expand selected entities to one-hop neighbors",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="entity.onehop",
                            inputs={
                                "entities": ToolInputSource(
                                    from_step_id=previous_entity_step,
                                    named_output_key="entities",
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
                step_id=evidence_step,
                description=f"Hop {hop}: retrieve source chunks for expanded entities",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="chunk.occurrence",
                            inputs={
                                "entities": ToolInputSource(
                                    from_step_id=expand_step,
                                    named_output_key="entities",
                                )
                            },
                            named_outputs={"chunks": "chunk_set"},
                        )
                    ]
                ),
            )
        )

        if accumulated_evidence_step is None:
            accumulated_evidence_step = evidence_step
        else:
            history_step = f"hop_{hop}_evidence_history"
            steps.append(
                ExecutionStep(
                    step_id=history_step,
                    description=f"Hop {hop}: retain deduplicated evidence from all explored hops",
                    action=DynamicToolChainConfig(
                        tools=[
                            ToolCall(
                                tool_id="chunk.merge",
                                inputs={
                                    "left": ToolInputSource(
                                        from_step_id=accumulated_evidence_step,
                                        named_output_key="chunks",
                                    ),
                                    "right": ToolInputSource(
                                        from_step_id=evidence_step,
                                        named_output_key="chunks",
                                    ),
                                },
                                named_outputs={"chunks": "chunk_set"},
                            )
                        ]
                    ),
                )
            )
            accumulated_evidence_step = history_step

        steps.append(
            ExecutionStep(
                step_id=reason_step,
                description=f"Hop {hop}: refine the information need from retrieved evidence",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="meta.reason_step",
                            inputs={
                                "query": previous_query_source,
                                "chunks": ToolInputSource(
                                    from_step_id=evidence_step,
                                    named_output_key="chunks",
                                ),
                            },
                            parameters={"mode": "refine"},
                            named_outputs={"query": "refined_query"},
                        )
                    ]
                ),
            )
        )

        steps.append(
            ExecutionStep(
                step_id=rerank_step,
                description=f"Hop {hop}: rerank expanded entities for the refined query",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="entity.tfidf",
                            inputs={
                                "query": ToolInputSource(
                                    from_step_id=reason_step,
                                    named_output_key="query",
                                ),
                                "entities": ToolInputSource(
                                    from_step_id=expand_step,
                                    named_output_key="entities",
                                ),
                            },
                            parameters={"top_k": top_k},
                            named_outputs={"entities": "entity_set"},
                        )
                    ]
                ),
            )
        )

        previous_entity_step = rerank_step
        previous_query_source = ToolInputSource(
            from_step_id=reason_step,
            named_output_key="query",
        )

    steps.append(
        ExecutionStep(
            step_id="final_evidence",
            description="Retrieve source chunks for the final selected entities",
            action=DynamicToolChainConfig(
                tools=[
                    ToolCall(
                        tool_id="chunk.occurrence",
                        inputs={
                            "entities": ToolInputSource(
                                from_step_id=previous_entity_step,
                                named_output_key="entities",
                            )
                        },
                        named_outputs={"chunks": "chunk_set"},
                    )
                ]
            ),
        )
    )

    steps.append(
        ExecutionStep(
            step_id="answer_evidence",
            description="Combine terminal and intermediate source evidence",
            action=DynamicToolChainConfig(
                tools=[
                    ToolCall(
                        tool_id="chunk.merge",
                        inputs={
                            "left": ToolInputSource(
                                from_step_id=accumulated_evidence_step,
                                named_output_key="chunks",
                            ),
                            "right": ToolInputSource(
                                from_step_id="final_evidence",
                                named_output_key="chunks",
                            ),
                        },
                        named_outputs={"chunks": "chunk_set"},
                    )
                ]
            ),
        )
    )

    steps.append(
        ExecutionStep(
            step_id="answer",
            description="Generate the final answer from accumulated pathfinding evidence",
            action=DynamicToolChainConfig(
                tools=[
                    ToolCall(
                        tool_id="meta.generate_answer",
                        inputs={
                            "query": "plan_inputs.query",
                            "chunks": ToolInputSource(
                                from_step_id="answer_evidence",
                                named_output_key="chunks",
                            ),
                        },
                        named_outputs={"answer": "text"},
                    )
                ]
            ),
        )
    )

    return ExecutionPlan(
        plan_description=(
            f"KGP: TF-IDF seeds -> {depth} evidence-guided neighbor/rerank hops -> "
            "accumulated source evidence -> answer"
        ),
        target_dataset_name=kwargs.get("dataset", ""),
        plan_inputs={"query": query},
        steps=steps,
    )
