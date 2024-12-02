
from Core.Common.Utils import load_json, write_json
from Core.Common.Logger import logger
from Core.Storage.BaseKVStorage import (
    BaseKVStorage,
)
from pydantic import BaseModel, ConfigDict, Field, model_validator


class JsonKVStorage(BaseKVStorage):
    # def __post_init__(self):
    # #     working_dir = self.global_config["working_dir"]
    # #     self._file_name = os.path.join(working_dir, f"kv_store_{self.namespace}.json")
    #     self._data = {}
    # #     logger.info(f"Load KV {self.namespace} with {len(self._data)} data")

    @model_validator(mode="after")
    def _load_from_file(cls, data):
        cls._data = {}
        return data
    async def all_keys(self) -> list[str]:
        return list(self._data.keys())

    async def index_done_callback(self):
        write_json(self._data, self._file_name)

    async def get_by_id(self, id):
        return self._data.get(id, None)

    async def get_by_ids(self, ids, fields=None):
        if fields is None:
            return [self._data.get(id, None) for id in ids]
        return [
            (
                {k: v for k, v in self._data[id].items() if k in fields}
                if self._data.get(id, None)
                else None
            )
            for id in ids
        ]

    async def filter_keys(self, data: list[str]) -> set[str]:
        return set([s for s in data if s not in self._data])

    async def upsert(self, data: dict[str, dict]):
        self._data.update(data)

    async def drop(self):
        self._data = {}
