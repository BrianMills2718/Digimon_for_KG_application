"""Fast deterministic checks for the canonical composition core.

These tests intentionally avoid graph builds, external providers, and the global
operator registry. They verify the typed plan/validator/executor plumbing that
should work in any clean development environment.
"""

import pytest

from Core.AgentSchema.plan import (
    DynamicToolChainConfig,
    ExecutionPlan,
    ExecutionStep,
    ToolCall,
    ToolInputSource,
)
from Core.Composition.ChainValidator import ChainValidator
from Core.Composition.OperatorComposer import OperatorComposer
from Core.Composition.PipelineExecutor import PipelineExecutionError, PipelineExecutor
from Core.Schema.OperatorDescriptor import CostTier, OperatorDescriptor, SlotSpec
from Core.Schema.SlotTypes import (
    EntityRecord,
    RelationshipRecord,
    SlotKind,
    SlotValue,
)


class TinyRegistry:
    def __init__(self, descriptors):
        self._descriptors = {d.operator_id: d for d in descriptors}

    def get(self, operator_id):
        return self._descriptors.get(operator_id)


async def query_to_entities(inputs, ctx, params):
    assert inputs["query"].kind == SlotKind.QUERY_TEXT
    return {
        "entities": SlotValue(
            kind=SlotKind.ENTITY_SET,
            data=[EntityRecord(entity_name="alpha", source_id="chunk-1")],
            producer="test.query_to_entities",
        )
    }


async def entities_to_relationships(inputs, ctx, params):
    assert inputs["entities"].kind == SlotKind.ENTITY_SET
    return {
        "relationships": SlotValue(
            kind=SlotKind.RELATIONSHIP_SET,
            data=[
                RelationshipRecord(
                    src_id="alpha",
                    tgt_id="beta",
                    relation_name="related_to",
                    source_id="chunk-1",
                )
            ],
            producer="test.entities_to_relationships",
        )
    }


def make_registry():
    return TinyRegistry(
        [
            OperatorDescriptor(
                operator_id="test.query_to_entities",
                display_name="Query to entities",
                category="test",
                input_slots=[SlotSpec("query", SlotKind.QUERY_TEXT)],
                output_slots=[SlotSpec("entities", SlotKind.ENTITY_SET)],
                cost_tier=CostTier.FREE,
                implementation=query_to_entities,
            ),
            OperatorDescriptor(
                operator_id="test.entities_to_relationships",
                display_name="Entities to relationships",
                category="test",
                input_slots=[SlotSpec("entities", SlotKind.ENTITY_SET)],
                output_slots=[SlotSpec("relationships", SlotKind.RELATIONSHIP_SET)],
                cost_tier=CostTier.FREE,
                implementation=entities_to_relationships,
            ),
        ]
    )


def make_valid_plan():
    return ExecutionPlan(
        plan_description="deterministic composition smoke",
        target_dataset_name="test",
        plan_inputs={"query": "How is alpha connected?"},
        steps=[
            ExecutionStep(
                step_id="entities",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="test.query_to_entities",
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
                            tool_id="test.entities_to_relationships",
                            inputs={
                                "entities": ToolInputSource(
                                    from_step_id="entities",
                                    named_output_key="entities",
                                )
                            },
                            named_outputs={"relationships": "relationship_set"},
                        )
                    ]
                ),
            ),
        ],
    )


def make_invalid_plan():
    return ExecutionPlan(
        plan_description="invalid missing input",
        target_dataset_name="test",
        plan_inputs={"query": "unused"},
        steps=[
            ExecutionStep(
                step_id="relationships",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="test.entities_to_relationships",
                            inputs={},
                            named_outputs={"relationships": "relationship_set"},
                        )
                    ]
                ),
            )
        ],
    )


def make_composer():
    composer = object.__new__(OperatorComposer)
    composer.registry = make_registry()
    composer.profiles = {}
    return composer


def test_chain_validator_accepts_explicit_typed_wiring():
    result = ChainValidator(make_registry()).validate(
        make_valid_plan(), plan_input_kinds={SlotKind.QUERY_TEXT}
    )
    assert result.valid, result.errors


def test_chain_validator_rejects_missing_required_input():
    result = ChainValidator(make_registry()).validate(
        make_invalid_plan(), plan_input_kinds={SlotKind.QUERY_TEXT}
    )
    assert not result.valid
    assert any(error.slot_name == "entities" for error in result.errors)


@pytest.mark.asyncio
async def test_pipeline_executor_runs_explicit_typed_chain():
    executor = PipelineExecutor(make_registry(), ctx=object())
    outputs = await executor.execute(make_valid_plan())

    final = outputs["relationships"]["relationships"]
    assert final.kind == SlotKind.RELATIONSHIP_SET
    assert final.data[0].source_id == "chunk-1"


@pytest.mark.asyncio
async def test_pipeline_executor_rejects_wrong_plan_input_kind():
    plan = ExecutionPlan(
        plan_description="invalid type wiring",
        target_dataset_name="test",
        plan_inputs={"query": "not an entity set"},
        steps=[
            ExecutionStep(
                step_id="relationships",
                action=DynamicToolChainConfig(
                    tools=[
                        ToolCall(
                            tool_id="test.entities_to_relationships",
                            inputs={"entities": "plan_inputs.query"},
                            named_outputs={"relationships": "relationship_set"},
                        )
                    ]
                ),
            )
        ],
    )

    executor = PipelineExecutor(make_registry(), ctx=object())
    with pytest.raises(PipelineExecutionError, match="Type mismatch"):
        await executor.execute(plan)


@pytest.mark.asyncio
async def test_operator_composer_rejects_invalid_plan_by_default():
    with pytest.raises(PipelineExecutionError, match="failed static validation"):
        await make_composer().execute(make_invalid_plan(), ctx=object())


@pytest.mark.asyncio
async def test_operator_composer_best_effort_does_not_disable_runtime_contracts():
    with pytest.raises(PipelineExecutionError, match="Missing required input"):
        await make_composer().execute(
            make_invalid_plan(),
            ctx=object(),
            allow_invalid_plan=True,
        )
