"""HippoRAG: Extract entities -> Link -> PPR -> Chunk aggregation.

Uses IDF-weighted PPR reset probabilities with iterative reasoning.
"""

from Core.AgentSchema.plan import (
    DynamicToolChainConfig,
    ExecutionPlan,
    ExecutionStep,
    ToolCall,
    ToolInputSource,
)


def hipporag_plan(query: str, **kwargs) -> ExecutionPlan:
    max_ir_steps = kwargs.get("max_ir_steps", 3)

    return ExecutionPlan(
        plan_description="HippoRAG: LLM entity extraction -> linking -> IDF-weighted PPR -> chunk retrieval",
        target_dataset_name=kwargs.get("dataset", ""),
        plan_inputs={"query": query},
        steps=[
            ExecutionStep(
                step_id="s1",
                description="Extract entities from query using LLM",
                action=DynamicToolChainConfig(tools=[
                    ToolCall(
                        tool_id="meta.extract_entities",
                        inputs={"query": "plan_inputs.query"},
                        named_outputs={"entities": "entity_set"},
                    ),
                ]),
            ),
            ExecutionStep(
                step_id="s2",
                description="Link extracted entities to graph",
                action=DynamicToolChainConfig(tools=[
                    ToolCall(
                        tool_id="entity.link",
                        inputs={"entities": ToolInputSource(from_step_id="s1", named_output_key="entities")},
                        named_outputs={"entities": "entity_set"},
                    ),
                ]),
            ),
            ExecutionStep(
                step_id="s3",
                description="Run PPR with linked entities as seeds",
                action=DynamicToolChainConfig(tools=[
                    ToolCall(
                        tool_id="entity.ppr",
                        inputs={
                            "query": "plan_inputs.query",
                            "entities": ToolInputSource(from_step_id="s2", named_output_key="entities"),
                        },
                        named_outputs={"entities": "entity_set", "score_vector": "score_vector"},
                    ),
                ]),
            ),
            ExecutionStep(
                step_id="s4",
                description="Propagate PPR scores to chunks via sparse matrices",
                action=DynamicToolChainConfig(tools=[
                    ToolCall(
                        tool_id="chunk.aggregator",
                        inputs={
                            "score_vector": ToolInputSource(from_step_id="s3", named_output_key="score_vector"),
                        },
                        named_outputs={"chunks": "chunk_set"},
                    ),
                ]),
            ),
        ],
    )
