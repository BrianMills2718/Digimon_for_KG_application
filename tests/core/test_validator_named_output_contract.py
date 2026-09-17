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


def test_validator_rejects_named_output_key_operator_does_not_emit():
    descriptor = OperatorDescriptor(
        operator_id="test.query",
        display_name="Query op",
        category="test",
        input_slots=[SlotSpec("query", SlotKind.QUERY_TEXT)],
        output_slots=[SlotSpec("entities", SlotKind.ENTITY_SET)],
        cost_tier=CostTier.FREE,
        implementation=None,
    )
    plan = ExecutionPlan(
        plan_description="bad output name",
        target_dataset_name="test",
        plan_inputs={"query": "question"},
        steps=[
            ExecutionStep(
                step_id="bad_output",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="test.query",
                            inputs={"query": "plan_inputs.query"},
                            named_outputs={"not_entities": "entity_set"},
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
    assert "Unknown named output 'not_entities'" in result.errors[0].message
    assert "entities" in result.errors[0].message
