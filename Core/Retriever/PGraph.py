from Core.Prompt import KGP_QUERY_PROMPT

async def retrieve(self, question):
    corpus = dict({id: (await self.kgp_graph.get_node(id))['chunk'] for id in list(self.kgp_graph.graph.nodes)})
    candidates_idx = list(id for id in list(self.kgp_graph.graph.nodes))

    seed = question
    contexts = []

    idxs = await self.tf_idf(seed, candidates_idx, corpus, k=self.k // self.k_nei)

    cur_contexts = [corpus[_] for _ in idxs]
    next_reasons = [seed + '\n' + (await self.llm.aask(KGP_QUERY_PROMPT.format(question=question, context=context))) for
                    context in cur_contexts]

    logger.info("next_reasons: {next_reasons}".format(next_reasons=next_reasons))

    visited = []

    for idx, next_reason in zip(idxs, next_reasons):
        nei_candidates_idx = list(self.kgp_graph.graph.neighbors(idx))
        nei_candidates_idx = [_ for _ in nei_candidates_idx if _ not in visited]
        if (nei_candidates_idx == []):
            continue

        next_contexts = await self.tf_idf(next_reason, nei_candidates_idx, corpus, k=self.k_nei)
        contexts.extend([corpus[idx] + '\n' + corpus[_] for _ in next_contexts if corpus[_] != corpus[idx]])
        visited.append(idx)
        visited.extend([_ for _ in next_contexts])

    return contexts


async def tf_idf(self, seed, candidates_idx, corpus, k):
    # for _ in candidates_idx:
    #     logger.info("context: {context}".format(context=corpus[_]))
    # logger.info("question: {question}".format(question=seed))

    index = TFIDFIndex()
    index._build_index_from_list([corpus[_] for _ in candidates_idx])
    idxs = index.query(query_str=seed, top_k=k)
    return [candidates_idx[_] for _ in idxs]


async def answer_question(self, question):
    contexts = await self.retrieve(question)
    context_str = '\n'.join('{index}: {context}'.format(index=i, context=c) for i, c in enumerate(contexts, start=1))

    logger.info(context_str)
    logger.info(question)

    answer = await self.llm.aask(KGP_QUERY_PROMPT.format(question=question, context=context_str))
    return answer
