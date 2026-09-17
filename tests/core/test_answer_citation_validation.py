from Core.Operators.meta.generate_answer import _citation_metadata


def test_citation_validation_accepts_only_known_evidence_ids():
    metadata = _citation_metadata(
        "Supported claim [chunk-a] and another [chunk-b].",
        ["chunk-a", "chunk-b"],
    )

    assert metadata == {
        "citation_status": "valid",
        "cited_evidence_ids": ["chunk-a", "chunk-b"],
        "invalid_citation_ids": [],
    }


def test_citation_validation_flags_invented_evidence_id():
    metadata = _citation_metadata(
        "Claim [chunk-a], unsupported citation [chunk-made-up].",
        ["chunk-a"],
    )

    assert metadata["citation_status"] == "invalid"
    assert metadata["cited_evidence_ids"] == ["chunk-a"]
    assert metadata["invalid_citation_ids"] == ["chunk-made-up"]


def test_citation_validation_marks_uncited_answer_missing():
    metadata = _citation_metadata(
        "A fluent answer with no source marker.",
        ["chunk-a"],
    )

    assert metadata["citation_status"] == "missing"
    assert metadata["cited_evidence_ids"] == []
    assert metadata["invalid_citation_ids"] == []
