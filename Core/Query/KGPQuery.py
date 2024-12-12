from Core.Query.BaseQuery import BaseQuery
from Core.Common.Logger import logger
from Core.Common.Constants import Retriever
from Core.Common.Utils import to_str_by_maxtokens
from Core.Prompt import QueryPrompt 
from Core.Common.Memory import Memory
from Core.Schema.Message import Message
from Core.Common.Constants import TOKEN_TO_CHAR_RATIO   
class KGPQuery(BaseQuery):
    def __init__(self, config, retirever_context):
        super().__init__(config, retirever_context)


    async def _retrieve_relevant_contexts(self, query):
        corpus, candidates_idx = await self._retirever.retrieve_relevant_content(key = "description", type=Retriever.ENTITY, mode="all")
        cur_contexts, idxs = await self._retirever.retrieve_relevant_content(seed = query, corpus = corpus, candidates_idx = candidates_idx,  type = Retriever.ENTITY, mode = "tf_df")  
        contexts = []
        next_reasons = [query + '\n' + (await self.llm.aask(QueryPrompt.KGP_QUERY_PROMPT.format(question=query, context=context))) for context in cur_contexts]

        logger.info("next_reasons: {next_reasons}".format(next_reasons=next_reasons))

        visited = []

        for idx, next_reason in zip(idxs, next_reasons):
            nei_candidates_idx = list(await self.graph.get_neighbors(idx))
            nei_candidates_idx = [_ for _ in nei_candidates_idx if _ not in visited]
            if (nei_candidates_idx == []):
                continue

            next_contexts = await self.tf_idf(next_reason, nei_candidates_idx, corpus, k = self.k_nei)
            contexts.extend([corpus[idx] + '\n' + corpus[_] for _ in next_contexts if corpus[_] != corpus[idx]])
            visited.append(idx)
            visited.extend([_ for _ in next_contexts])
        return contexts
        

    async def generation_qa(self, query, context):
        import pdb
        pdb.set_trace()
        if context is None:
            return QueryPrompt.FAIL_RESPONSE
    
    async def generation_summary(self, query, context):

        if context is None:
            return QueryPrompt.FAIL_RESPONSE
        