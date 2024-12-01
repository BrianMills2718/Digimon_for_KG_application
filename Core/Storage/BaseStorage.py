from dataclasses import field

from pydantic import BaseModel
from typing import (
    Any,
    Optional

)
from Core.Storage.NameSpace import Namespace


class BaseStorage(BaseModel):
    config: Optional[Any] = field()
    namespace: Optional[Namespace] = field(default=None)
