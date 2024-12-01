import asyncio
import json
from Core.Config2 import Config
from QueryConfig import QueryConfig
from Core.Common.ContextMixin import ContextMixin
from Core.Common.LLM import LLM
from abc import ABC, abstractmethod


class BaseQuery(ABC, ContextMixin):
    config: Config
    query_config: QueryConfig
    llm: LLM
    
    def __init__(self, config: Config, query_config: QueryConfig, llm: LLM):
        self.global_config = config
        self.query_config = query_config
        self.llm = llm

    def reset(self):
        self.memory.clear()
