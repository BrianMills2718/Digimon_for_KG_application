import json

import numpy as np

from Core.Composition.OperatorComposer import _to_transport_value
from Core.Schema.SlotTypes import ChunkRecord, EntityRecord, SubgraphRecord


def test_context_transport_preserves_record_structure_and_is_json_serializable():
    payload = {
        "entities": [
            EntityRecord(
                entity_name="alpha",
                source_id="chunk-a",
                score=0.9,
            )
        ],
        "chunks": [
            ChunkRecord(
                chunk_id="chunk-a",
                text="source text",
                score=0.8,
                extra={"rank": 1},
            )
        ],
        "subgraph": SubgraphRecord(
            nodes={"alpha", "beta"},
            edges=[("alpha", "beta")],
        ),
        "scores": np.array([0.7, 0.3]),
    }

    serialized = _to_transport_value(payload)

    assert serialized["entities"][0]["entity_name"] == "alpha"
    assert serialized["entities"][0]["source_id"] == "chunk-a"
    assert serialized["chunks"][0] == {
        "chunk_id": "chunk-a",
        "text": "source text",
        "score": 0.8,
        "extra": {"rank": 1},
    }
    assert serialized["subgraph"]["nodes"] == ["alpha", "beta"]
    assert serialized["subgraph"]["edges"] == [["alpha", "beta"]]
    assert serialized["scores"] == [0.7, 0.3]

    # This is the actual property the MCP facade needs.
    encoded = json.dumps(serialized)
    assert "chunk-a" in encoded
    assert "EntityRecord(" not in encoded
    assert "ChunkRecord(" not in encoded
