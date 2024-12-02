from pydantic import BaseModel, Field
from typing import (
    Any,
    Optional

)
from Core.Storage.NameSpace import Namespace


class BaseStorage(BaseModel):
    config: Optional[Any] = Field(default=None, exclude=True)
    namespace: Optional[Namespace] = Field(default=None, exclude=True)
