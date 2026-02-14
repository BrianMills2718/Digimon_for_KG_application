"""LightRAG: Entity VDB + Relationship VDB -> Chunks from relations.

Keywords-based global retrieval with entity and relationship VDB search.
"""

from Core.AgentSchema.plan import (
    DynamicToolChainConfig,
    ExecutionPlan,
    ExecutionStep,
    ToolCall,
    ToolInputSource,
)


def lightrag_plan(query: str, **kwargs) -> ExecutionPlan:
    return ExecutionPlan(
        plan_description="LightRAG: Entity VDB + Relationship VDB -> entity extraction -> chunk retrieval",
        target_dataset_name=kwargs.get("dataset", ""),
        plan_inputs={"query": query},
        steps=[
            ExecutionStep(
                step_id="s1",
                description="Search relationships by semantic similarity",
                action=DynamicToolChainConfig(tools=[
                    ToolCall(
                        tool_id="relationship.vdb",
                        inputs={"query": "plan_inputs.query"},
                        named_outputs={"relationships": "relationship_set"},
                    ),
                ]),
            ),
            ExecutionStep(
                step_id="s2",
                description="Extract entities from found relationships",
                action=DynamicToolChainConfig(tools=[
                    ToolCall(
                        tool_id="entity.rel_node",
                        inputs={"relationships": ToolInputSource(from_step_id="s1", named_output_key="relationships")},
                        named_outputs={"entities": "entity_set"},
                    ),
                ]),
            ),
            ExecutionStep(
                step_id="s3",
                description="Get text chunks from relationships",
                action=DynamicToolChainConfig(tools=[
                    ToolCall(
                        tool_id="chunk.from_relation",
                        inputs={"relationships": ToolInputSource(from_step_id="s1", named_output_key="relationships")},
                        named_outputs={"chunks": "chunk_set"},
                    ),
                ]),
            ),
        ],
    )
