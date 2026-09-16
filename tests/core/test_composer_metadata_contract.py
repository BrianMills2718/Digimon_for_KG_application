import pytest

from Core.AgentSchema.plan import (
    DynamicToolChainConfig,
    ExecutionPlan,
    ExecutionStep,
    ToolCall,
)
from Core.Composition.OperatorComposer import OperatorComposer
from Core.Operators.registry import OperatorRegistry
from Core.Schema.OperatorDescriptor import CostTier, OperatorDescriptor, SlotSpec
from Core.Schema.SlotTypes import SlotKind, SlotValue


async def answer_with_provenance(inputs, ctx, params):
    return {
        "answer": SlotValue(
            kind=SlotKind.QUERY_TEXT,
            data="Grounded answer [chunk-a].",
            producer="test.answer",
            metadata={
                "status": "grounded_answer",
                "evidence_chunk_ids": ["chunk-a"],
            },
        )
    }


@pytest.mark.asyncio
async def test_composer_preserves_slot_metadata_alongside_existing_output_shape():
    registry = OperatorRegistry()
    registry.register(
        OperatorDescriptor(
            operator_id="test.answer",
            display_name="Test answer",
            category="test",
            input_slots=[SlotSpec("query", SlotKind.QUERY_TEXT)],
            output_slots=[SlotSpec("answer", SlotKind.QUERY_TEXT)],
            cost_tier=CostTier.FREE,
            implementation=answer_with_provenance,
        )
    )
    composer = OperatorComposer(registry)
    plan = ExecutionPlan(
        plan_description="metadata transport",
        target_dataset_name="test",
        plan_inputs={"query": "question"},
        steps=[
            ExecutionStep(
                step_id="answer",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="test.answer",
                            inputs={"query": "plan_inputs.query"},
                            named_outputs={"answer": "text"},
                        )
                    ]
                ),
            )
        ],
    )

    result = await composer.execute(plan, ctx=object())

    assert result["final_output"]["answer"] == "Grounded answer [chunk-a]."
    metadata = result["final_metadata"]["answer"]
    assert metadata["status"] == "grounded_answer"
    assert metadata["evidence_chunk_ids"] == ["chunk-a"]
    assert metadata["producer"] == "test.answer"
    assert metadata["kind"] == "query_text"
    assert result["all_step_metadata"]["answer"]["answer"] == metadata
