"""ToG (Think-on-Graph): Iterative entity-relation exploration via LLM.

Iterative loop: extract entities -> link -> (relation agent -> entity agent) x depth.
Uses LoopConfig for the iterative exploration.
"""

from Core.AgentSchema.plan import (
    DynamicToolChainConfig,
    ExecutionPlan,
    ExecutionStep,
    LoopConfig,
    ToolCall,
    ToolInputSource,
)


def tog_plan(query: str, **kwargs) -> ExecutionPlan:
    depth = kwargs.get("depth", 3)

    return ExecutionPlan(
        plan_description=f"ToG: Entity linking -> iterative (relation agent -> entity agent) x {depth} -> answer",
        target_dataset_name=kwargs.get("dataset", ""),
        plan_inputs={"query": query},
        steps=[
            ExecutionStep(
                step_id="s1",
                description="Extract and link entities from query",
                action=DynamicToolChainConfig(tools=[
                    ToolCall(
                        tool_id="meta.extract_entities",
                        inputs={"query": "plan_inputs.query"},
                        named_outputs={"entities": "entity_set"},
                    ),
                    ToolCall(
                        tool_id="entity.link",
                        inputs={"entities": ToolInputSource(from_step_id="s1", named_output_key="entities")},
                        named_outputs={"entities": "entity_set"},
                    ),
                ]),
            ),
            ExecutionStep(
                step_id="s2_explore",
                description="Explore relations from current entities using LLM",
                action=DynamicToolChainConfig(tools=[
                    ToolCall(
                        tool_id="relationship.agent",
                        inputs={
                            "query": "plan_inputs.query",
                            "entities": ToolInputSource(from_step_id="s1", named_output_key="entities"),
                        },
                        named_outputs={"relationships": "relationship_set"},
                    ),
                ]),
            ),
            ExecutionStep(
                step_id="s3_score",
                description="Score entity candidates from explored relations",
                action=DynamicToolChainConfig(tools=[
                    ToolCall(
                        tool_id="entity.agent",
                        inputs={
                            "query": "plan_inputs.query",
                            "entity_relation_list": ToolInputSource(from_step_id="s1", named_output_key="entities"),
                        },
                        named_outputs={"entities": "entity_set"},
                    ),
                ]),
            ),
            ExecutionStep(
                step_id="s4_loop",
                description=f"Iterate exploration up to {depth} times",
                action=LoopConfig(
                    body_step_ids=["s2_explore", "s3_score"],
                    max_iterations=depth,
                    termination_condition="len(entities.data) == 0",
                    carry_forward_outputs=["entities", "relationships"],
                ),
            ),
            ExecutionStep(
                step_id="s5_chunks",
                description="Get text chunks from accumulated relationships",
                action=DynamicToolChainConfig(tools=[
                    ToolCall(
                        tool_id="chunk.from_relation",
                        inputs={"relationships": ToolInputSource(from_step_id="s2_explore", named_output_key="relationships")},
                        named_outputs={"chunks": "chunk_set"},
                    ),
                ]),
            ),
            ExecutionStep(
                step_id="s6_answer",
                description="Generate answer from accumulated context",
                action=DynamicToolChainConfig(tools=[
                    ToolCall(
                        tool_id="meta.generate_answer",
                        inputs={
                            "query": "plan_inputs.query",
                            "chunks": ToolInputSource(from_step_id="s5_chunks", named_output_key="chunks"),
                        },
                        named_outputs={"answer": "text"},
                    ),
                ]),
            ),
        ],
    )
