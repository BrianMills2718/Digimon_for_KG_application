
from Core.Retriever.BaseRetriever import BaseRetriever
class EntityRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
    



    async def find_relevant_contexts(self, query, top_k, entity_vdb, llm, graph, mode = "vdb"):
         """
        Find the top-k relevant contexts for the given query.
        :param query: The query string.
        :param top_k: The number of top-k relevant contexts to return.
        :return: A list of tuples, where each tuple contains the document id and the context text.
        """
         