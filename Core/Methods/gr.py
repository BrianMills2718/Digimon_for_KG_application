"""GR (Graph Retrieval via PCST): Entity VDB + Relationship VDB -> PCST optimization.

Find informative subgraph using Prize-Collecting Steiner Tree.
"""

from Core.AgentSchema.plan import (
    DynamicToolChainConfig,
    ExecutionPlan,
    ExecutionStep,
    ToolCall,
    ToolInputSource,
)


def gr_plan(query: str, **kwargs) -> ExecutionPlan:
    return ExecutionPlan(
        plan_description="GR: Entity VDB + Relationship VDB -> PCST subgraph optimization -> answer",
        target_dataset_name=kwargs.get("dataset", ""),
        plan_inputs={"query": query},
        steps=[
            ExecutionStep(
                step_id="s1",
                description="Find entities similar to query via VDB",
                action=DynamicToolChainConfig(tools=[
                    ToolCall(
                        tool_id="entity.vdb",
                        inputs={"query": "plan_inputs.query"},
                        named_outputs={"entities": "entity_set"},
                    ),
                ]),
            ),
            ExecutionStep(
                step_id="s2",
                description="Find relationships similar to query via VDB",
                action=DynamicToolChainConfig(tools=[
                    ToolCall(
                        tool_id="relationship.vdb",
                        inputs={"query": "plan_inputs.query"},
                        named_outputs={"relationships": "relationship_set"},
                    ),
                ]),
            ),
            ExecutionStep(
                step_id="s3",
                description="Optimize subgraph via PCST",
                action=DynamicToolChainConfig(tools=[
                    ToolCall(
                        tool_id="meta.pcst_optimize",
                        inputs={
                            "entities": ToolInputSource(from_step_id="s1", named_output_key="entities"),
                            "relationships": ToolInputSource(from_step_id="s2", named_output_key="relationships"),
                        },
                        named_outputs={"subgraph": "subgraph"},
                    ),
                ]),
            ),
            ExecutionStep(
                step_id="s4",
                description="Get chunks from relationships",
                action=DynamicToolChainConfig(tools=[
                    ToolCall(
                        tool_id="chunk.from_relation",
                        inputs={"relationships": ToolInputSource(from_step_id="s2", named_output_key="relationships")},
                        named_outputs={"chunks": "chunk_set"},
                    ),
                ]),
            ),
            ExecutionStep(
                step_id="s5",
                description="Generate answer from retrieved chunks",
                action=DynamicToolChainConfig(tools=[
                    ToolCall(
                        tool_id="meta.generate_answer",
                        inputs={
                            "query": "plan_inputs.query",
                            "chunks": ToolInputSource(from_step_id="s4", named_output_key="chunks"),
                        },
                        named_outputs={"answer": "text"},
                    ),
                ]),
            ),
        ],
    )
