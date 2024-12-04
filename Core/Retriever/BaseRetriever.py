import asyncio
from Core.Config2 import Config
from Core.Common.ContextMixin import ContextMixin
from abc import ABC, abstractmethod


class BaseRetriever(ABC):
  
        
    def __init__(self, config, ):
          self.config = config

    def reset(self):
        self.memory.clear()
