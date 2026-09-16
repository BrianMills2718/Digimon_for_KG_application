"""Medical subgraph reference plan.

Expand around VDB seeds, rerank the neighborhood into terminal entities, connect
those terminals with a Steiner tree, then answer only from evidence supporting
that selected structural subgraph.
"""

from Core.AgentSchema.plan import (
    DynamicToolChainConfig,
    ExecutionPlan,
    ExecutionStep,
    ToolCall,
    ToolInputSource,
)
from Core.Operators.subgraph.materialize import ensure_subgraph_materialize_registered


def med_plan(query: str, **kwargs) -> ExecutionPlan:
    ensure_subgraph_materialize_registered()
    k_hop = max(1, int(kwargs.get("k_hop", 2)))
    terminal_count = max(2, int(kwargs.get("terminal_count", 8)))

    return ExecutionPlan(
        plan_description=(
            "Med: VDB seeds -> k-hop neighborhood -> query-ranked terminals -> "
            "Steiner tree -> source evidence -> answer"
        ),
        target_dataset_name=kwargs.get("dataset", ""),
        plan_inputs={"query": query},
        steps=[
            ExecutionStep(
                step_id="seed_entities",
                description="Find relevant seed entities via VDB",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="entity.vdb",
                            inputs={"query": "plan_inputs.query"},
                            parameters={"top_k": 20},
                            named_outputs={"entities": "entity_set"},
                        )
                    ]
                ),
            ),
            ExecutionStep(
                step_id="neighborhood",
                description=f"Build the {k_hop}-hop neighborhood around seed entities",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="subgraph.khop_paths",
                            inputs={
                                "entities": ToolInputSource(
                                    from_step_id="seed_entities",
                                    named_output_key="entities",
                                )
                            },
                            parameters={"k": k_hop, "mode": "neighbors"},
                            named_outputs={"subgraph": "subgraph"},
                        )
                    ]
                ),
            ),
            ExecutionStep(
                step_id="neighborhood_entities",
                description="Materialize neighborhood entities for terminal selection",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="subgraph.materialize",
                            inputs={
                                "subgraph": ToolInputSource(
                                    from_step_id="neighborhood",
                                    named_output_key="subgraph",
                                )
                            },
                            named_outputs={"entities": "entity_set"},
                        )
                    ]
                ),
            ),
            ExecutionStep(
                step_id="terminals",
                description="Rank neighborhood entities and choose Steiner terminals",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="entity.tfidf",
                            inputs={
                                "query": "plan_inputs.query",
                                "entities": ToolInputSource(
                                    from_step_id="neighborhood_entities",
                                    named_output_key="entities",
                                ),
                            },
                            parameters={"top_k": terminal_count},
                            named_outputs={"entities": "entity_set"},
                        )
                    ]
                ),
            ),
            ExecutionStep(
                step_id="steiner",
                description="Connect query-relevant terminals with a compact Steiner tree",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="subgraph.steiner_tree",
                            inputs={
                                "entities": ToolInputSource(
                                    from_step_id="terminals",
                                    named_output_key="entities",
                                )
                            },
                            named_outputs={"subgraph": "subgraph"},
                        )
                    ]
                ),
            ),
            ExecutionStep(
                step_id="evidence",
                description="Retrieve source chunks supporting the Steiner tree",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="subgraph.materialize",
                            inputs={
                                "subgraph": ToolInputSource(
                                    from_step_id="steiner",
                                    named_output_key="subgraph",
                                )
                            },
                            named_outputs={"chunks": "chunk_set"},
                        )
                    ]
                ),
            ),
            ExecutionStep(
                step_id="answer",
                description="Generate answer from structural subgraph evidence",
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
