"""Med (Medical Subgraph): VDB entities -> k-hop subgraph -> Steiner tree -> answer.

Medical-domain subgraph extraction with iterative entity collection.
"""

from Core.AgentSchema.plan import (
    DynamicToolChainConfig,
    ExecutionPlan,
    ExecutionStep,
    ToolCall,
    ToolInputSource,
)


def med_plan(query: str, **kwargs) -> ExecutionPlan:
    k_hop = kwargs.get("k_hop", 2)

    return ExecutionPlan(
        plan_description="Med: VDB entity search -> k-hop subgraph -> Steiner tree -> answer",
        target_dataset_name=kwargs.get("dataset", ""),
        plan_inputs={"query": query},
        steps=[
            ExecutionStep(
                step_id="s1",
                description="Find relevant entities via VDB",
                action=DynamicToolChainConfig(tools=[
                    ToolCall(
                        tool_id="entity.vdb",
                        inputs={"query": "plan_inputs.query"},
                        parameters={"top_k": 20},
                        named_outputs={"entities": "entity_set"},
                    ),
                ]),
            ),
            ExecutionStep(
                step_id="s2",
                description="Find k-hop neighborhood subgraph",
                action=DynamicToolChainConfig(tools=[
                    ToolCall(
                        tool_id="subgraph.khop_paths",
                        inputs={"entities": ToolInputSource(from_step_id="s1", named_output_key="entities")},
                        parameters={"k": k_hop, "mode": "neighbors"},
                        named_outputs={"subgraph": "subgraph"},
                    ),
                ]),
            ),
            ExecutionStep(
                step_id="s3",
                description="Build Steiner tree connecting entities",
                action=DynamicToolChainConfig(tools=[
                    ToolCall(
                        tool_id="subgraph.steiner_tree",
                        inputs={"entities": ToolInputSource(from_step_id="s1", named_output_key="entities")},
                        named_outputs={"subgraph": "subgraph"},
                    ),
                ]),
            ),
            ExecutionStep(
                step_id="s4",
                description="Get relationships from subgraph entities",
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
                description="Generate answer",
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
