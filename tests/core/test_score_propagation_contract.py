from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from Core.Operators.chunk.aggregator import (
    _normalize_nonnegative_scores,
    chunk_aggregator,
)
from Core.Operators.relationship.score_aggregator import relationship_score_agg
from Core.Schema.SlotTypes import SlotKind, SlotValue


class FakeChunkStore:
    def __init__(self):
        self._chunks = {
            "chunk-a": "alpha evidence",
            "chunk-b": "beta evidence",
            "chunk-c": "gamma evidence",
        }

    async def get_data_by_indices(self, indices):
        values = list(self._chunks.values())
        return [values[index] for index in indices]


class FakeGraph:
    async def get_edge_by_indices(self, indices):
        edges = [
            {"src_id": "a", "tgt_id": "b", "source_id": "chunk-a"},
            {"src_id": "b", "tgt_id": "c", "source_id": "chunk-b"},
        ]
        return [edges[int(index)] for index in indices]


@pytest.mark.asyncio
async def test_relationship_propagation_returns_highest_score_first():
    ctx = SimpleNamespace(
        sparse_matrices={
            "entity_to_rel": csr_matrix(
                np.array(
                    [
                        [1.0, 0.0],
                        [1.0, 1.0],
                    ]
                )
            )
        },
        graph=FakeGraph(),
        config=SimpleNamespace(top_k=2),
    )
    inputs = {
        "score_vector": SlotValue(
            kind=SlotKind.SCORE_VECTOR,
            data=np.array([0.8, 0.2]),
            producer="test",
        )
    }

    result = await relationship_score_agg(inputs, ctx, {})
    records = result["relationships"].data

    assert [record.extra["edge_index"] for record in records] == [0, 1]
    assert records[0].score > records[1].score


@pytest.mark.asyncio
async def test_chunk_propagation_preserves_real_chunk_ids_and_rank_order():
    ctx = SimpleNamespace(
        sparse_matrices={
            "entity_to_rel": csr_matrix(np.array([[1.0], [0.0]])),
            "rel_to_chunk": csr_matrix(np.array([[1.0, 0.5, 0.0]])),
        },
        doc_chunks=FakeChunkStore(),
        config=SimpleNamespace(top_k=2),
    )
    inputs = {
        "score_vector": SlotValue(
            kind=SlotKind.SCORE_VECTOR,
            data=np.array([1.0, 0.0]),
            producer="test",
        )
    }

    result = await chunk_aggregator(inputs, ctx, {})
    records = result["chunks"].data

    assert [record.chunk_id for record in records] == ["chunk-a", "chunk-b"]
    assert [record.text for record in records] == ["alpha evidence", "beta evidence"]
    assert records[0].score > records[1].score
    assert records[0].extra["matrix_index"] == 0


def test_score_normalization_is_finite_for_zero_and_flat_inputs():
    zeros = _normalize_nonnegative_scores(np.array([0.0, 0.0]))
    flat = _normalize_nonnegative_scores(np.array([3.0, 3.0]))

    assert np.all(np.isfinite(zeros))
    assert np.all(zeros == 0.0)
    assert np.all(np.isfinite(flat))
    assert np.all(flat == 1.0)
