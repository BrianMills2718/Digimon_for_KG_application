import pytest

from Core.Common.Constants import GRAPH_FIELD_SEP
from Core.Utils.MergeER import MergeEntity, MergeRelationship


def test_source_id_and_keyword_merges_are_deterministic():
    assert MergeEntity.merge_source_ids(
        ["chunk-b", "chunk-a"],
        ["chunk-c", "chunk-a"],
    ) == GRAPH_FIELD_SEP.join(["chunk-a", "chunk-b", "chunk-c"])

    assert MergeRelationship.merge_source_ids(
        ["chunk-b"],
        ["chunk-a", "chunk-b"],
    ) == GRAPH_FIELD_SEP.join(["chunk-a", "chunk-b"])

    assert MergeRelationship.merge_keywords(
        ["zeta", "alpha"],
        ["beta", "alpha"],
    ) == GRAPH_FIELD_SEP.join(["alpha", "beta", "zeta"])


@pytest.mark.asyncio
async def test_legacy_relationship_merge_helper_no_longer_references_missing_function():
    result = await MergeRelationship.merge_info(
        {
            "source_id": ["chunk-b"],
            "weight": [1.0],
            "description": ["old"],
            "keywords": ["beta"],
            "relation_name": ["relationship"],
        },
        {
            "source_id": ["chunk-a"],
            "weight": [2.0],
            "description": ["new"],
            "keywords": ["alpha"],
            "relation_name": ["relationship"],
        },
    )

    assert result[0] == GRAPH_FIELD_SEP.join(["chunk-a", "chunk-b"])
    assert result[1] == pytest.approx(3.0)
    assert result[2] == GRAPH_FIELD_SEP.join(["new", "old"])
