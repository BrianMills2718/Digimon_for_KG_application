"""DALK reference plan: linked entities -> graph paths -> LLM path filter -> evidence."""

from Core.AgentSchema.plan import (
    DynamicToolChainConfig,
    ExecutionPlan,
    ExecutionStep,
    ToolCall,
    ToolInputSource,
)
from Core.Operators.subgraph.materialize import ensure_subgraph_materialize_registered


def dalk_plan(query: str, **kwargs) -> ExecutionPlan:
    ensure_subgraph_materialize_registered()
    k_hop = max(1, int(kwargs.get("k_hop", 3)))

    return ExecutionPlan(
        plan_description=(
            "DALK: entity linking -> k-hop reasoning paths -> LLM path filtering -> "
            "source evidence -> answer"
        ),
        target_dataset_name=kwargs.get("dataset", ""),
        plan_inputs={"query": query},
        steps=[
            ExecutionStep(
                step_id="seed_entities",
                description="Extract and link entities from the query",
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
            ),
            ExecutionStep(
                step_id="paths",
                description="Find graph paths connecting linked entities",
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
                            parameters={"mode": "paths", "cutoff": k_hop},
                            named_outputs={"subgraph": "subgraph"},
                        )
                    ]
                ),
            ),
            ExecutionStep(
                step_id="filtered_paths",
                description="Filter reasoning paths by relevance to the query",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="subgraph.agent_path",
                            inputs={
                                "query": "plan_inputs.query",
                                "subgraph": ToolInputSource(
                                    from_step_id="paths",
                                    named_output_key="subgraph",
                                ),
                            },
                            named_outputs={"subgraph": "subgraph"},
                        )
                    ]
                ),
            ),
            ExecutionStep(
                step_id="evidence",
                description="Retrieve source chunks supporting the selected paths",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="subgraph.materialize",
                            inputs={
                                "subgraph": ToolInputSource(
                                    from_step_id="filtered_paths",
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
                description="Generate the answer from selected-path evidence",
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
