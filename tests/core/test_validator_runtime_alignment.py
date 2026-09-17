from Core.AgentSchema.plan import (
    DynamicToolChainConfig,
    ExecutionPlan,
    ExecutionStep,
    ToolCall,
)
from Core.Composition.ChainValidator import ChainValidator
from Core.Operators.registry import REGISTRY
from Core.Schema.SlotTypes import SlotKind


def test_validator_rejects_unwired_required_slot_even_when_kind_is_available():
    plan = ExecutionPlan(
        plan_description="validator/runtime alignment",
        target_dataset_name="Demo",
        plan_inputs={"query": "Who is connected?"},
        steps=[
            ExecutionStep(
                step_id="extract",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="meta.extract_entities",
                            inputs={"query": "plan_inputs.query"},
                            named_outputs={"entities": "entity_set"},
                        )
                    ]
                ),
            ),
            ExecutionStep(
                step_id="relationships",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="relationship.onehop",
                            # Deliberately omit required `entities` wiring. The
                            # old validator saw an ENTITY_SET somewhere upstream
                            # and incorrectly treated this as valid; the runtime
                            # executor cannot and does not auto-wire it.
                            inputs={},
                            named_outputs={"relationships": "relationship_set"},
                        )
                    ]
                ),
            ),
        ],
    )

    result = ChainValidator(REGISTRY).validate(
        plan,
        plan_input_kinds={SlotKind.QUERY_TEXT},
    )

    assert result.valid is False
    assert any(
        error.step_id == "relationships"
        and error.slot_name == "entities"
        and "not explicitly wired" in error.message
        for error in result.errors
    )


def test_validator_rejects_wrong_named_input_instead_of_kind_matching_it():
    plan = ExecutionPlan(
        plan_description="wrong input name",
        target_dataset_name="Demo",
        plan_inputs={"query": "Who is connected?"},
        steps=[
            ExecutionStep(
                step_id="extract",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="meta.extract_entities",
                            inputs={"query": "plan_inputs.query"},
                            named_outputs={"entities": "entity_set"},
                        )
                    ]
                ),
            ),
            ExecutionStep(
                step_id="relationships",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="relationship.onehop",
                            inputs={"wrong_name": "plan_inputs.query"},
                            named_outputs={"relationships": "relationship_set"},
                        )
                    ]
                ),
            ),
        ],
    )

    result = ChainValidator(REGISTRY).validate(
        plan,
        plan_input_kinds={SlotKind.QUERY_TEXT},
    )

    assert result.valid is False
    assert any("Unknown input 'wrong_name'" in error.message for error in result.errors)
    assert any(error.slot_name == "entities" for error in result.errors)
