from pydantic import BaseModel, Field, ConfigDict
from typing import (
    Any,
    Optional

)
from Core.Storage.NameSpace import Namespace


class BaseStorage(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    config: Optional[Any] = Field(default=None, exclude=True)
    namespace: Optional[Namespace] = Field(default=None, exclude=True)
