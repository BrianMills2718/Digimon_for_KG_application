from Core.Query.BaseQuery import BaseQuery
from Core.Common.Logger import logger
from Core.Common.Constants import Retriever
from Core.Common.Utils import list_to_quoted_csv_string, truncate_list_by_token_size, combine_contexts
from Core.Prompt import QueryPrompt 
class PPRQuery(BaseQuery):
    def __init__(self, config, retirever_context):
        super().__init__(config, retirever_context)

    async def reason_step(self, few_shot: list, query: str, passages: list, thoughts: list):
        """
        Given few-shot samples, query, previous retrieved passages, and previous thoughts, generate the next thought with OpenAI models. The generated thought is used for further retrieval step.
        :return: next thought
        """
        prompt_demo = ''
        for sample in few_shot:
            prompt_demo += f'{sample["document"]}\n\nQuestion: {sample["question"]}\nThought: {sample["answer"]}\n\n'

        prompt_user = ''

        #TODO: merge title for the hotpotQA dataset
        for passage in passages:
            prompt_user += f'{passage}\n\n'
        prompt_user += f'Question: {query} \n Thought:' + ' '.join(thoughts)
        import pdb
            
        pdb.set_trace()

        try:
            response_content = await self.llm.aask(msg = prompt_demo + prompt_user, system_msgs = [QueryPrompt.IRCOT_REASON_INSTRUCTION])
            import pdb
            pdb.set_trace()
        except Exception as e:
            print(e)
            return ''
        return response_content
    async def _retrieve_relevant_contexts(self, query):
        
        entities = await self.extract_query_entities(query)
        retrieved_passages, scores = await self._retirever.retrieve_relevant_content(query = query, seed_entities = entities, type = Retriever.CHUNK, mode = "ppr")
        thoughts = []
    
        passage_scores = {passage: score for passage, score in zip(retrieved_passages, scores)}
        few_shot_examples =  []
        # Iterative refinement loop
        for iteration in range(2, self.config.max_ir_steps + 1):
            logger.info("Entering the ir-cot iteration: {}".format(iteration))
            # Generate a new thought based on current passages and thoughts
            new_thought = await self.reason_step(few_shot_examples, query, retrieved_passages[ : self.config.top_k], thoughts)
            thoughts.append(new_thought)
            
            # Check if the thought contains the answer
            if 'So the answer is:' in new_thought:
                break
            

            # Retrieve new passages based on the new thought
            new_passages, new_scores  = await self._retirever.retrieve_relevant_content(query = query, seed_entities = thoughts, type = Retriever.CHUNK, mode = "ppr")
            
            # Update passage scores
            for passage, score in zip(new_passages, new_scores):
                if passage in passage_scores:
                    passage_scores[passage] = max(passage_scores[passage], score)
                else:
                    passage_scores[passage] = score
            
            # Sort passages by score in descending order
            sorted_passages = sorted(
                passage_scores.items(), key=lambda item: item[1], reverse=True
            )
            retrieved_passages, scores = zip(*sorted_passages)
        import pdb
        
        pdb.set_trace()
        return  retrieved_passages   
    
 
 
 
    
        
    async def query(self, query):

        context = await self._retrieve_relevant_contexts(query)
        
        response = await self.generation(query, context)
        return response
    

    async def generation(self, query, context):

        if context is None:
            return QueryPrompt.FAIL_RESPONSE
        if self.config.augmentation_ppr:
            
        return response