from Core.Index.FaissIndex import FaissIndex
from Core.Schema.SlotTypes import EntityRecord


def test_faiss_score_matrix_uses_entity_name_not_dataclass_repr():
    seed = EntityRecord(
        entity_name="alpha",
        source_id="chunk-a",
        description="Alpha entity",
    )

    assert FaissIndex._query_text_from_seed(seed) == "alpha"
    assert FaissIndex._query_text_from_seed({"entity_name": "beta"}) == "beta"
    assert FaissIndex._query_text_from_seed("gamma") == "gamma"
