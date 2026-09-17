from Core.AgentTools.relationship_tools import (
    _effective_relationship_embedding_fields,
    _relationship_embedding_text,
)


def test_legacy_relationship_embedding_default_maps_to_real_edge_schema():
    assert _effective_relationship_embedding_fields(["type", "description"]) == [
        "relation_name",
        "keywords",
        "description",
    ]


def test_relationship_embedding_text_contains_relation_semantics_and_keywords():
    text = _relationship_embedding_text(
        "alpha",
        "beta",
        {
            "relation_name": "founded",
            "keywords": "company origin",
            "description": "Alpha founded Beta in 2020.",
        },
        ["relation_name", "keywords", "description"],
    )

    assert "relation_name: founded" in text
    assert "keywords: company origin" in text
    assert "description: Alpha founded Beta in 2020." in text


def test_relationship_embedding_text_falls_back_to_endpoints_and_relation():
    text = _relationship_embedding_text(
        "alpha",
        "beta",
        {"relation_name": "connected_to"},
        ["description"],
    )

    assert text == "alpha connected_to beta"
