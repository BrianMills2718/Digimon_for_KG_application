"""DALK: Extract entities -> Link -> k-hop paths + neighbors -> LLM filter -> answer.

Direct Augmented Language-based Knowledge graph navigation.
"""

from Core.AgentSchema.plan import (
    DynamicToolChainConfig,
    ExecutionPlan,
    ExecutionStep,
    ToolCall,
    ToolInputSource,
)


def dalk_plan(query: str, **kwargs) -> ExecutionPlan:
    k_hop = kwargs.get("k_hop", 3)

    return ExecutionPlan(
        plan_description="DALK: Entity linking -> k-hop paths -> LLM path filter -> answer",
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
                step_id="s2",
                description="Find k-hop paths from linked entities",
                action=DynamicToolChainConfig(tools=[
                    ToolCall(
                        tool_id="subgraph.khop_paths",
                        inputs={"entities": ToolInputSource(from_step_id="s1", named_output_key="entities")},
                        parameters={"mode": "paths", "cutoff": k_hop},
                        named_outputs={"subgraph": "subgraph"},
                    ),
                ]),
            ),
            ExecutionStep(
                step_id="s3",
                description="LLM-filter paths by relevance",
                action=DynamicToolChainConfig(tools=[
                    ToolCall(
                        tool_id="subgraph.agent_path",
                        inputs={
                            "query": "plan_inputs.query",
                            "subgraph": ToolInputSource(from_step_id="s2", named_output_key="subgraph"),
                        },
                        named_outputs={"subgraph": "subgraph"},
                    ),
                ]),
            ),
            ExecutionStep(
                step_id="s4",
                description="Get relationships from linked entities",
                action=DynamicToolChainConfig(tools=[
                    ToolCall(
                        tool_id="relationship.onehop",
                        inputs={"entities": ToolInputSource(from_step_id="s1", named_output_key="entities")},
                        named_outputs={"relationships": "relationship_set"},
                    ),
                ]),
            ),
            ExecutionStep(
                step_id="s5",
                description="Get chunks from relationships",
                action=DynamicToolChainConfig(tools=[
                    ToolCall(
                        tool_id="chunk.from_relation",
                        inputs={"relationships": ToolInputSource(from_step_id="s4", named_output_key="relationships")},
                        named_outputs={"chunks": "chunk_set"},
                    ),
                ]),
            ),
            ExecutionStep(
                step_id="s6",
                description="Generate answer from retrieved chunks",
                action=DynamicToolChainConfig(tools=[
                    ToolCall(
                        tool_id="meta.generate_answer",
                        inputs={
                            "query": "plan_inputs.query",
                            "chunks": ToolInputSource(from_step_id="s5", named_output_key="chunks"),
                        },
                        named_outputs={"answer": "text"},
                    ),
                ]),
            ),
        ],
    )
