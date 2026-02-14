"""FastGraphRAG: Entity VDB -> PPR -> Relationship score aggregation -> Chunk aggregation.

Uses entity similarity for PPR reset probabilities and sparse matrix propagation.
"""

from Core.AgentSchema.plan import (
    DynamicToolChainConfig,
    ExecutionPlan,
    ExecutionStep,
    ToolCall,
    ToolInputSource,
)


def fastgraphrag_plan(query: str, **kwargs) -> ExecutionPlan:
    return ExecutionPlan(
        plan_description="FastGraphRAG: VDB seeds -> PPR -> score propagation through sparse matrices to chunks",
        target_dataset_name=kwargs.get("dataset", ""),
        plan_inputs={"query": query},
        steps=[
            ExecutionStep(
                step_id="s1",
                description="Find seed entities via VDB",
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
                description="Run PPR from seed entities",
                action=DynamicToolChainConfig(tools=[
                    ToolCall(
                        tool_id="entity.ppr",
                        inputs={
                            "query": "plan_inputs.query",
                            "entities": ToolInputSource(from_step_id="s1", named_output_key="entities"),
                        },
                        named_outputs={"entities": "entity_set", "score_vector": "score_vector"},
                    ),
                ]),
            ),
            ExecutionStep(
                step_id="s3",
                description="Propagate PPR scores to relationships",
                action=DynamicToolChainConfig(tools=[
                    ToolCall(
                        tool_id="relationship.score_agg",
                        inputs={
                            "entities": ToolInputSource(from_step_id="s2", named_output_key="entities"),
                            "score_vector": ToolInputSource(from_step_id="s2", named_output_key="score_vector"),
                        },
                        named_outputs={"relationships": "relationship_set"},
                    ),
                ]),
            ),
            ExecutionStep(
                step_id="s4",
                description="Propagate scores to text chunks",
                action=DynamicToolChainConfig(tools=[
                    ToolCall(
                        tool_id="chunk.aggregator",
                        inputs={
                            "score_vector": ToolInputSource(from_step_id="s2", named_output_key="score_vector"),
                        },
                        named_outputs={"chunks": "chunk_set"},
                    ),
                ]),
            ),
        ],
    )
