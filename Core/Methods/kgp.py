"""KGP (KG-based Pathfinding): TF-IDF -> iterative neighbor exploration -> answer.

Iteratively explores graph neighbors using TF-IDF ranking with LLM reasoning.
"""

from Core.AgentSchema.plan import (
    DynamicToolChainConfig,
    ExecutionPlan,
    ExecutionStep,
    LoopConfig,
    ToolCall,
    ToolInputSource,
)


def kgp_plan(query: str, **kwargs) -> ExecutionPlan:
    return ExecutionPlan(
        plan_description="KGP: TF-IDF entity ranking -> iterative (reason + neighbor TF-IDF) -> answer",
        target_dataset_name=kwargs.get("dataset", ""),
        plan_inputs={"query": query},
        steps=[
            ExecutionStep(
                step_id="s1",
                description="Find initial entities by TF-IDF",
                action=DynamicToolChainConfig(tools=[
                    ToolCall(
                        tool_id="entity.tfidf",
                        inputs={"query": "plan_inputs.query"},
                        parameters={"top_k": 5},
                        named_outputs={"entities": "entity_set"},
                    ),
                ]),
            ),
            ExecutionStep(
                step_id="s2_expand",
                description="Expand to neighbors",
                action=DynamicToolChainConfig(tools=[
                    ToolCall(
                        tool_id="entity.onehop",
                        inputs={"entities": ToolInputSource(from_step_id="s1", named_output_key="entities")},
                        named_outputs={"entities": "entity_set"},
                    ),
                ]),
            ),
            ExecutionStep(
                step_id="s3_reason",
                description="LLM reason over expanded entities",
                action=DynamicToolChainConfig(tools=[
                    ToolCall(
                        tool_id="meta.reason_step",
                        inputs={
                            "query": "plan_inputs.query",
                        },
                        parameters={"mode": "refine"},
                        named_outputs={"query": "refined_query"},
                    ),
                ]),
            ),
            ExecutionStep(
                step_id="s4_rerank",
                description="Re-rank neighbors by refined query TF-IDF",
                action=DynamicToolChainConfig(tools=[
                    ToolCall(
                        tool_id="entity.tfidf",
                        inputs={
                            "query": ToolInputSource(from_step_id="s3_reason", named_output_key="query"),
                            "entities": ToolInputSource(from_step_id="s2_expand", named_output_key="entities"),
                        },
                        named_outputs={"entities": "entity_set"},
                    ),
                ]),
            ),
            ExecutionStep(
                step_id="s5_loop",
                description="Iterate neighbor exploration",
                action=LoopConfig(
                    body_step_ids=["s2_expand", "s3_reason", "s4_rerank"],
                    max_iterations=3,
                    termination_condition="len(entities.data) == 0",
                    carry_forward_outputs=["entities"],
                ),
            ),
            ExecutionStep(
                step_id="s6_chunks",
                description="Get chunks for accumulated entities",
                action=DynamicToolChainConfig(tools=[
                    ToolCall(
                        tool_id="chunk.occurrence",
                        inputs={"entities": ToolInputSource(from_step_id="s4_rerank", named_output_key="entities")},
                        named_outputs={"chunks": "chunk_set"},
                    ),
                ]),
            ),
            ExecutionStep(
                step_id="s7_answer",
                description="Generate answer",
                action=DynamicToolChainConfig(tools=[
                    ToolCall(
                        tool_id="meta.generate_answer",
                        inputs={
                            "query": "plan_inputs.query",
                            "chunks": ToolInputSource(from_step_id="s6_chunks", named_output_key="chunks"),
                        },
                        named_outputs={"answer": "text"},
                    ),
                ]),
            ),
        ],
    )
