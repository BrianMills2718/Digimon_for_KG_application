from typing import Any
from Core.Storage.BaseStorage import BaseStorage # This should now be the Pydantic BaseModel version

class BaseBlobStorage(BaseStorage):
    # This class primarily defines an interface, so it might not have many fields itself.
    # The __init__ is inherited from BaseStorage (Pydantic's __init__).

    async def get(self) -> Any: # Added return type hint
        raise NotImplementedError

    async def set(self, blob: Any) -> None:
        raise NotImplementedError
