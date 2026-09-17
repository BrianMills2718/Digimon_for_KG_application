from types import SimpleNamespace

from Core.AgentTools import derived_resource_cleanup as cleanup


def test_forced_er_cleanup_removes_canonical_vdbs_and_sparse_matrices(tmp_path, monkeypatch):
    monkeypatch.setattr(cleanup, "_project_root", lambda: tmp_path)
    config = SimpleNamespace(working_dir="results")

    entity_vdb = tmp_path / "storage" / "vdb" / "Demo_entities"
    relation_vdb = tmp_path / "storage" / "vdb" / "Demo_relations"
    entity_vdb.mkdir(parents=True)
    relation_vdb.mkdir(parents=True)
    (entity_vdb / "index.bin").write_text("old entity index")
    (relation_vdb / "index.bin").write_text("old relationship index")

    er_dir = tmp_path / "results" / "Demo" / "er_graph"
    er_dir.mkdir(parents=True)
    e2r = er_dir / "sparse_e2r.pkl"
    r2c = er_dir / "sparse_r2c.pkl"
    e2r.write_bytes(b"old")
    r2c.write_bytes(b"old")

    unrelated = tmp_path / "storage" / "vdb" / "Other_entities"
    unrelated.mkdir(parents=True)

    removed = cleanup.invalidate_after_forced_graph_rebuild(
        config,
        "Demo",
        invalidate_sparse_matrices=True,
    )

    assert not entity_vdb.exists()
    assert not relation_vdb.exists()
    assert not e2r.exists()
    assert not r2c.exists()
    assert unrelated.exists()
    assert len(removed) == 4


def test_non_er_cleanup_leaves_er_sparse_matrices_but_removes_canonical_vdbs(tmp_path, monkeypatch):
    monkeypatch.setattr(cleanup, "_project_root", lambda: tmp_path)
    config = SimpleNamespace(working_dir="results")

    entity_vdb = tmp_path / "storage" / "vdb" / "Demo_entities"
    relation_vdb = tmp_path / "storage" / "vdb" / "Demo_relations"
    entity_vdb.mkdir(parents=True)
    relation_vdb.mkdir(parents=True)

    er_dir = tmp_path / "results" / "Demo" / "er_graph"
    er_dir.mkdir(parents=True)
    e2r = er_dir / "sparse_e2r.pkl"
    r2c = er_dir / "sparse_r2c.pkl"
    e2r.write_bytes(b"keep")
    r2c.write_bytes(b"keep")

    cleanup.invalidate_after_forced_graph_rebuild(
        config,
        "Demo",
        invalidate_sparse_matrices=False,
    )

    assert not entity_vdb.exists()
    assert not relation_vdb.exists()
    assert e2r.exists()
    assert r2c.exists()
