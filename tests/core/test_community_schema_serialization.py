import json

from Core.Schema.CommunitySchema import LeidenInfo


def test_leiden_info_as_dict_is_json_safe_and_deterministic():
    info = LeidenInfo(
        level=1,
        title="demo",
        edges={("beta", "alpha"), ("alpha", "gamma")},
        nodes={"gamma", "alpha", "beta"},
        chunk_ids={"chunk-b", "chunk-a"},
        occurrence=0.5,
        sub_communities=["2", "1"],
    )

    data = info.as_dict
    encoded = json.dumps(data)

    assert encoded
    assert data["nodes"] == ["alpha", "beta", "gamma"]
    assert data["chunk_ids"] == ["chunk-a", "chunk-b"]
    assert all(isinstance(edge, list) for edge in data["edges"])
    assert data["sub_communities"] == ["2", "1"]
