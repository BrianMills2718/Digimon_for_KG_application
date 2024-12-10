async def retrieve(self, question):
        
    self.k: int = 30
    self.k_nei: int = 3
    graph_nodes = list(await self.graph.get_nodes())
    # corpus = dict({id: (await self.graph.get_node(id))['chunk'] for id in list(self.gra.graph.nodes)})
    corpus = dict({id: (await self.graph.get_node(id))['description'] for id in graph_nodes})
    candidates_idx = list(id for id in graph_nodes)
    import pdb

    seed = question
    contexts = []
    
    idxs = await self.tf_idf(seed, candidates_idx, corpus, k = self.k // self.k_nei)

    cur_contexts = [corpus[_] for _ in idxs]
    next_reasons = [seed + '\n' + (await self.llm.aask(KGP_QUERY_PROMPT.format(question=question, context=context))) for context in cur_contexts]

    logger.info("next_reasons: {next_reasons}".format(next_reasons=next_reasons))

    visited = []

    for idx, next_reason in zip(idxs, next_reasons):
        nei_candidates_idx = list(await self.graph.get_neighbors(idx))
        import pdb
        nei_candidates_idx = [_ for _ in nei_candidates_idx if _ not in visited]
        if (nei_candidates_idx == []):
            continue

        next_contexts = await self.tf_idf(next_reason, nei_candidates_idx, corpus, k = self.k_nei)
        contexts.extend([corpus[idx] + '\n' + corpus[_] for _ in next_contexts if corpus[_] != corpus[idx]])
        visited.append(idx)
        visited.extend([_ for _ in next_contexts])
    import pdb
    
    return contexts

async def tf_idf(self, seed, candidates_idx, corpus, k):

    index = TFIDFIndex()

    index._build_index_from_list([corpus[_] for _ in candidates_idx])
    idxs = index.query(query_str = seed, top_k = k)

    return [candidates_idx[_] for _ in idxs]