from pydantic import BaseModel, ConfigDict
from typing import Any, Optional
# Assuming NameSpace is defined in Core.Storage.NameSpace and is not a Pydantic model itself,
# or if it is, it's compatible. For arbitrary_types_allowed=True, it should be fine.
from Core.Storage.NameSpace import Namespace 

class BaseStorage(BaseModel):
    # Allow arbitrary types for fields like 'namespace' if Namespace is not a Pydantic model
    model_config = ConfigDict(arbitrary_types_allowed=True) 

    config: Optional[Any] = None
    namespace: Optional[Namespace] = None

    # If there was an __init__ here, it would typically be removed when converting to Pydantic,
    # as Pydantic handles initialization. If custom __init__ logic is needed,
    # it should be adapted using Pydantic's initializers or validators.
