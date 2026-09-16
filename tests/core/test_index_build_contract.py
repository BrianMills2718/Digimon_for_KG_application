from pathlib import Path

import pytest

from Core.Index.BaseIndex import BaseIndex


class FakeConfig:
    def __init__(self, persist_path: Path):
        self.persist_path = persist_path


class FakeIndex(BaseIndex):
    def __init__(self, persist_path: Path, fail_update: bool = False):
        super().__init__(FakeConfig(persist_path))
        self.fail_update = fail_update

    async def retrieval(self, query, top_k):
        return []

    def _get_index(self):
        return object()

    async def retrieval_batch(self, queries, top_k):
        return []

    async def _update_index(self, elements, meta_data):
        self._index = None if self.fail_update else object()

    def _get_retrieve_top_k(self):
        return 5

    def _storage_index(self):
        if self._index is not None:
            Path(self.config.persist_path).mkdir(parents=True, exist_ok=True)

    async def _load_index(self) -> bool:
        if not self.exist_index():
            return False
        self._index = object()
        return True

    async def retrieval_nodes(self, query, top_k, graph):
        return []


@pytest.mark.asyncio
async def test_build_index_reports_false_when_update_loses_index(tmp_path):
    index = FakeIndex(tmp_path / "failed", fail_update=True)

    ok = await index.build_index([{"content": "x"}], ["content"])

    assert ok is False
    assert index._index is None


@pytest.mark.asyncio
async def test_build_index_reports_true_only_after_persistence(tmp_path):
    index = FakeIndex(tmp_path / "ok")

    ok = await index.build_index([{"content": "x"}], ["content"])

    assert ok is True
    assert index.exist_index()


@pytest.mark.asyncio
async def test_build_index_does_not_use_legacy_get_index_hook(tmp_path):
    class ExplodingGetIndex(FakeIndex):
        def _get_index(self):
            raise AssertionError("build_index should let _update_index initialize the backend")

    index = ExplodingGetIndex(tmp_path / "dynamic")

    assert await index.build_index([{"content": "x"}], ["content"]) is True
