from types import SimpleNamespace

import pytest

from Core.AgentSchema.plan import (
    ConditionalBranch,
    DynamicToolChainConfig,
    ExecutionPlan,
    ExecutionStep,
    LoopConfig,
    ToolCall,
)
from Core.Composition.PipelineExecutor import PipelineExecutor
from Core.Schema.OperatorDescriptor import CostTier, OperatorDescriptor, SlotSpec
from Core.Schema.SlotTypes import ChunkRecord, SlotKind, SlotValue


class TinyRegistry:
    def __init__(self, descriptors):
        self._descriptors = {descriptor.operator_id: descriptor for descriptor in descriptors}

    def get(self, operator_id):
        return self._descriptors.get(operator_id)


@pytest.mark.asyncio
async def test_loop_body_is_not_executed_outside_loop_and_preserves_chunk_kind():
    state = {"calls": 0}

    async def emit_chunk(inputs, ctx, params):
        state["calls"] += 1
        call = state["calls"]
        return {
            "chunks": SlotValue(
                kind=SlotKind.CHUNK_SET,
                data=[
                    ChunkRecord(
                        chunk_id=f"chunk-{call}",
                        text=f"evidence {call}",
                    )
                ],
                producer="test.emit_chunk",
                metadata={"last_call": call},
            )
        }

    registry = TinyRegistry(
        [
            OperatorDescriptor(
                operator_id="test.emit_chunk",
                display_name="Emit chunk",
                category="test",
                input_slots=[],
                output_slots=[SlotSpec("chunks", SlotKind.CHUNK_SET)],
                cost_tier=CostTier.FREE,
                implementation=emit_chunk,
            )
        ]
    )

    body = ExecutionStep(
        step_id="body",
        action=DynamicToolChainConfig(
            tools=[
                ToolCall(
                    tool_id="test.emit_chunk",
                    named_outputs={"chunks": "chunk_set"},
                )
            ]
        ),
    )
    loop = ExecutionStep(
        step_id="loop",
        action=LoopConfig(
            body_step_ids=["body"],
            max_iterations=2,
            termination_condition="False",
            carry_forward_outputs=["chunks"],
        ),
    )
    plan = ExecutionPlan(
        plan_description="loop contract",
        target_dataset_name="test",
        steps=[body, loop],
    )

    outputs = await PipelineExecutor(registry, ctx=SimpleNamespace()).execute(plan)

    assert state["calls"] == 2
    carried = outputs["loop"]["chunks"]
    assert carried.kind == SlotKind.CHUNK_SET
    assert [chunk.chunk_id for chunk in carried.data] == ["chunk-1", "chunk-2"]
    assert carried.metadata["iterations_accumulated"] == 2
    assert carried.metadata["last_call"] == 2


@pytest.mark.asyncio
async def test_conditional_runs_only_selected_control_owned_step_once():
    state = {"true": 0, "false": 0}

    def descriptor(operator_id, key):
        async def implementation(inputs, ctx, params):
            state[key] += 1
            return {
                "chunks": SlotValue(
                    kind=SlotKind.CHUNK_SET,
                    data=[],
                    producer=operator_id,
                )
            }

        return OperatorDescriptor(
            operator_id=operator_id,
            display_name=operator_id,
            category="test",
            input_slots=[],
            output_slots=[SlotSpec("chunks", SlotKind.CHUNK_SET)],
            cost_tier=CostTier.FREE,
            implementation=implementation,
        )

    registry = TinyRegistry(
        [descriptor("test.true", "true"), descriptor("test.false", "false")]
    )

    true_step = ExecutionStep(
        step_id="true_step",
        action=DynamicToolChainConfig(
            tools=[ToolCall(tool_id="test.true", named_outputs={"chunks": "chunk_set"})]
        ),
    )
    false_step = ExecutionStep(
        step_id="false_step",
        action=DynamicToolChainConfig(
            tools=[ToolCall(tool_id="test.false", named_outputs={"chunks": "chunk_set"})]
        ),
    )
    branch = ExecutionStep(
        step_id="branch",
        action=ConditionalBranch(
            condition="True",
            if_true_steps=["true_step"],
            if_false_steps=["false_step"],
        ),
    )
    plan = ExecutionPlan(
        plan_description="conditional contract",
        target_dataset_name="test",
        steps=[true_step, false_step, branch],
    )

    await PipelineExecutor(registry, ctx=SimpleNamespace()).execute(plan)

    assert state == {"true": 1, "false": 0}
