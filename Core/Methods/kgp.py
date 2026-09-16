"""KGP (KG-based Pathfinding) reference plan.

Each hop expands from the previously selected entities, retrieves evidence for
those neighbors, refines the query from that evidence, and reranks the neighbor
set with TF-IDF before continuing.
"""

from Core.AgentSchema.plan import (
    DynamicToolChainConfig,
    ExecutionPlan,
    ExecutionStep,
    ToolCall,
    ToolInputSource,
)


def kgp_plan(query: str, **kwargs) -> ExecutionPlan:
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

        query_input = (
            previous_query_source
            if isinstance(previous_query_source, str)
            else previous_query_source
        )
        steps.append(
            ExecutionStep(
                step_id=reason_step,
                description=f"Hop {hop}: refine the information need from retrieved evidence",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="meta.reason_step",
                            inputs={
                                "query": query_input,
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

    steps.extend(
        [
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
            ),
            ExecutionStep(
                step_id="answer",
                description="Generate the final answer from pathfinding evidence",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="meta.generate_answer",
                            inputs={
                                "query": "plan_inputs.query",
                                "chunks": ToolInputSource(
                                    from_step_id="final_evidence",
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
            f"KGP: TF-IDF seeds -> {depth} evidence-guided neighbor/rerank hops -> answer"
        ),
        target_dataset_name=kwargs.get("dataset", ""),
        plan_inputs={"query": query},
        steps=steps,
    )
