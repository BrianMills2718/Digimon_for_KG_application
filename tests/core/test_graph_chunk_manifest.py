from pathlib import Path
from types import SimpleNamespace

from Core.AgentTools.graph_chunk_manifest import (
    manifest_matches,
    read_manifest,
    write_manifest,
)


class Namespace:
    def __init__(self, path: Path):
        self.path = path

    def get_save_path(self, suffix=None):
        return str(self.path / suffix) if suffix else str(self.path)


def graph_at(tmp_path):
    return SimpleNamespace(
        _graph=SimpleNamespace(namespace=Namespace(tmp_path / "graph"))
    )


def pairs(*ids):
    return [(chunk_id, object()) for chunk_id in ids]


def test_missing_manifest_is_unknown_and_first_write_is_stable(tmp_path):
    graph = graph_at(tmp_path)
    chunks = pairs("chunk-a", "chunk-b")

    assert manifest_matches(graph, chunks) is None
    assert write_manifest(graph, chunks) is True
    assert read_manifest(graph) == ["chunk-a", "chunk-b"]
    assert manifest_matches(graph, chunks) is True


def test_added_or_removed_chunks_change_manifest_even_when_old_ids_still_exist(tmp_path):
    graph = graph_at(tmp_path)
    old_chunks = pairs("chunk-a", "chunk-b")
    write_manifest(graph, old_chunks)

    # The old graph could still reference only a/b, so a source-id subset check
    # would miss this new document/chunk. Exact manifest comparison must not.
    assert manifest_matches(graph, pairs("chunk-a", "chunk-b", "chunk-c")) is False
    assert manifest_matches(graph, pairs("chunk-a")) is False


def test_manifest_comparison_is_order_independent(tmp_path):
    graph = graph_at(tmp_path)
    write_manifest(graph, pairs("chunk-b", "chunk-a"))

    assert manifest_matches(graph, pairs("chunk-a", "chunk-b")) is True
