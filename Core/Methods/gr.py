"""GR (Graph Retrieval via PCST) reference plan."""

from Core.AgentSchema.plan import (
    DynamicToolChainConfig,
    ExecutionPlan,
    ExecutionStep,
    ToolCall,
    ToolInputSource,
)
from Core.Operators.subgraph.materialize import ensure_subgraph_materialize_registered


def gr_plan(query: str, **kwargs) -> ExecutionPlan:
    ensure_subgraph_materialize_registered()

    return ExecutionPlan(
        plan_description=(
            "GR: entity + relationship VDB retrieval -> PCST-selected subgraph -> "
            "source evidence -> answer"
        ),
        target_dataset_name=kwargs.get("dataset", ""),
        plan_inputs={"query": query},
        steps=[
            ExecutionStep(
                step_id="entities",
                description="Find query-relevant entities",
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
                step_id="relationships",
                description="Find query-relevant relationships",
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
                step_id="pcst",
                description="Select a compact informative subgraph",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="meta.pcst_optimize",
                            inputs={
                                "entities": ToolInputSource(
                                    from_step_id="entities",
                                    named_output_key="entities",
                                ),
                                "relationships": ToolInputSource(
                                    from_step_id="relationships",
                                    named_output_key="relationships",
                                ),
                            },
                            named_outputs={"subgraph": "subgraph"},
                        )
                    ]
                ),
            ),
            ExecutionStep(
                step_id="evidence",
                description="Materialize original source chunks supporting the PCST subgraph",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="subgraph.materialize",
                            inputs={
                                "subgraph": ToolInputSource(
                                    from_step_id="pcst",
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
                description="Generate answer from PCST-selected evidence",
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
