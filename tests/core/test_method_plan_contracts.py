import pytest

from Core.Composition.ChainValidator import ChainValidator
from Core.Methods import METHOD_PLANS
from Core.Operators.registry import REGISTRY
from Core.Schema.SlotTypes import SlotKind


@pytest.mark.parametrize("method_name", sorted(METHOD_PLANS))
def test_reference_method_plan_is_statically_valid(method_name):
    plan = METHOD_PLANS[method_name](
        query="How are the relevant entities connected?",
        dataset="ContractTest",
    )

    result = ChainValidator(REGISTRY).validate(
        plan,
        plan_input_kinds={SlotKind.QUERY_TEXT},
    )

    assert result.valid, [
        f"{error.step_id}/{error.tool_id}/{error.slot_name}: {error.message}"
        for error in result.errors
    ]
