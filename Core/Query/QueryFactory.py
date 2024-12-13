"""
Query Factory.
"""
from Core.Query.BaseQuery import BaseQuery
from Core.Query.BasicQuery import BasicQuery
from Core.Query.PPRQuery import PPRQuery
from Core.Query.KGPQuery import KGPQuery


class QueryFactory:
    def __init__(self):
        self.creators = {
            "basic": self._create_base_query,
            "ppr": self._create_hippo_query,
            "kgp": self._create_kgp_query,
        }

    def get_query(self, name, config, retriever) -> BaseQuery:
        """Key is PersistType."""
        return self.creators[name](config, retriever)

    @staticmethod
    def _create_base_query(config, retriever):
        return BasicQuery(config, retriever)

    @staticmethod
    def _create_hippo_query(config, retriever):
        return PPRQuery(config, retriever)

    @staticmethod
    def _create_kgp_query(config, retriever):
        return KGPQuery(config, retriever)


get_query = QueryFactory().get_query
