"""
Query Factory.
"""
from Core.Query.BaseQuery import BaseQuery
from Core.Query.BasicQuery import BasicQuery
from Core.Query.PPRQuery import PPRQuery



class QueryFactory():
    def __init__(self):
        self.creators = {
            "basic": self._create_base_query,
            "ppr": self._create_hippo_query
        }

    def get_query(self, name, retriever) -> BaseQuery:
        """Key is PersistType."""
        return self.creators[name](retriever)

    @staticmethod
    def _create_base_query(retriever):
        return BasicQuery(retriever)


    @staticmethod
    def _create_hippo_query(retriever):
        return PPRQuery(retriever)
    
get_query = QueryFactory().get_query
