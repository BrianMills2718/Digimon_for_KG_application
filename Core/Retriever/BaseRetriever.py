import asyncio
from Core.Config2 import Config
from Core.Common.ContextMixin import ContextMixin
from abc import ABC, abstractmethod


class BaseRetriever(ABC):
  
        
    def __init__(self, config):
          self.config = config

    def reset(self):
        self.memory.clear()

    @abstractmethod
    async def find_relevant_contexts(self, query, top_k=10, **context):
         """
        Find the top-k relevant contexts for the given query.
        :param query: The query string.
        :param top_k: The number of top-k relevant contexts to return.
        :return: A list of tuples, where each tuple contains the document id and the context text.
        """