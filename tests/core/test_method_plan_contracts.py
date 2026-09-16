import pytest

from Core.Composition.ChainValidator import ChainValidator
from Core.Composition.OperatorComposer import OperatorComposer
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


@pytest.mark.parametrize("method_name", sorted(METHOD_PLANS))
def test_reference_method_normally_ends_with_answer_generation(method_name):
    plan = METHOD_PLANS[method_name](
        query="How are the relevant entities connected?",
        dataset="ContractTest",
    )

    last_step = plan.steps[-1]
    assert last_step.action.tools[-1].tool_id == "meta.generate_answer"


def test_context_only_removes_only_terminal_answer_step_for_all_methods():
    composer = OperatorComposer(REGISTRY)

    for method_name in sorted(METHOD_PLANS):
        full_plan = composer.build_plan(
            method_name,
            query="How are the relevant entities connected?",
            dataset="ContractTest",
        )
        context_plan = composer.build_plan(
            method_name,
            query="How are the relevant entities connected?",
            dataset="ContractTest",
            return_context_only=True,
        )

        assert full_plan.steps[-1].action.tools[-1].tool_id == "meta.generate_answer"
        assert len(context_plan.steps) == len(full_plan.steps) - 1
        assert context_plan.steps[-1].action.tools[-1].tool_id != "meta.generate_answer"
