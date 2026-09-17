from Core.AgentSchema.plan import (
    DynamicToolChainConfig,
    ExecutionPlan,
    ExecutionStep,
    ToolCall,
)
from Core.Composition.ChainValidator import ChainValidator
from Core.Schema.OperatorDescriptor import CostTier, OperatorDescriptor, SlotSpec
from Core.Schema.SlotTypes import SlotKind


class TinyRegistry:
    def __init__(self, descriptor):
        self.descriptor = descriptor

    def get(self, operator_id):
        return self.descriptor if operator_id == self.descriptor.operator_id else None


def test_validator_rejects_query_plan_input_wired_to_entity_slot():
    descriptor = OperatorDescriptor(
        operator_id="test.needs_entities",
        display_name="Needs entities",
        category="test",
        input_slots=[SlotSpec("entities", SlotKind.ENTITY_SET)],
        output_slots=[],
        cost_tier=CostTier.FREE,
        implementation=None,
    )
    plan = ExecutionPlan(
        plan_description="wrong explicit plan input type",
        target_dataset_name="test",
        plan_inputs={"query": "plain text"},
        steps=[
            ExecutionStep(
                step_id="bad",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="test.needs_entities",
                            inputs={"entities": "plan_inputs.query"},
                            named_outputs={},
                        )
                    ]
                ),
            )
        ],
    )

    result = ChainValidator(TinyRegistry(descriptor)).validate(
        plan,
        plan_input_kinds={SlotKind.QUERY_TEXT},
    )

    assert result.valid is False
    assert len(result.errors) == 1
    error = result.errors[0]
    assert error.slot_name == "entities"
    assert error.expected_kind == SlotKind.ENTITY_SET
    assert "Type mismatch" in error.message
    assert "plan_inputs.query" in error.message
    assert "QUERY_TEXT" in error.message
