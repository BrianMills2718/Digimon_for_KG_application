from Core.Query.BaseQuery import BaseQuery
from Core.Common.Logger import logger
from Core.Common.Constants import Retriever
from Core.Common.Utils import list_to_quoted_csv_string, truncate_list_by_token_size, combine_contexts
from Core.Prompt import QueryPrompt 
class PPRQuery(BaseQuery):
    def __init__(self, retirever_context):
        super().__init__(retirever_context)

    
    async def _retrieve_relevant_contexts(self, query):
        
        entities = await self.extract_query_entities(query)
        contexts = await self._retirever.retrieve_relevant_content(query = query, seed_entities = entities, type = Retriever.CHUNK, mode = "ppr")
        import pdb
        pdb.set_trace()
        return contexts  
    
 
 
 
    
        
    async def query(self, query):

        context = await self._retrieve_relevant_contexts(query)
        response = await self.generation(query, context)
        return response
    

    async def generation(self, query, context):

        if context is None:
            return QueryPrompt.FAIL_RESPONSE

        if self.config.use_community and self.config.use_global_query:
            sys_prompt_temp = QueryPrompt.GLOBAL_REDUCE_RAG_RESPONSE
        elif not self.config.use_community and self.config.use_keywords:
            sys_prompt_temp = QueryPrompt.RAG_RESPONSE
        elif self.config.use_community and not self.config.use_keywords and self.config.enable_local:
            sys_prompt_temp = QueryPrompt.LOCAL_RAG_RESPONSE
        else:
            logger.error("Invalid query configuration")
            return QueryPrompt.FAIL_RESPONSE
        response = await self.llm.aask(
            query,
            system_msgs= [sys_prompt_temp.format(
                report_data=context, response_type=self.query_config.response_type
            )],
        )  
        return response