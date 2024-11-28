from Core.Graph.BaseGraph import BaseGraph
from Core.Schema.ChunkSchema import TextChunk
from Core.Storage.NetworkXStorage import NetworkXStorage
from Core.Common.Utils import logger
from Core.Index.TFIDFStore import TFIDFIndex
from Core.Prompt import KGP_QUERY_PROMPT

from collections import defaultdict
from tqdm import tqdm
from multiprocessing import Pool
from itertools import chain
import concurrent.futures
import requests
import time

MY_GCUBE_TOKEN = '07e1bd33-c0f5-41b0-979b-4c9a859eec3f-843339462'

class WATAnnotation:
    # An entity annotated by WAT

    def __init__(self, d):

        # char offset (included)
        self.start = d['start']
        # char offset (not included)
        self.end = d['end']

        # annotation accuracy
        self.rho = d['rho']
        # spot-entity probability
        self.prior_prob = d['explanation']['prior_explanation']['entity_mention_probability']

        # annotated text
        self.spot = d['spot']

        # Wikpedia entity info
        self.wiki_id = d['id']
        self.wiki_title = d['title']


    def json_dict(self):
        # Simple dictionary representation
        return {'wiki_title': self.wiki_title,
                'wiki_id': self.wiki_id,
                'start': self.start,
                'end': self.end,
                'rho': self.rho,
                'prior_prob': self.prior_prob
                }

class KGPGraph(BaseGraph):
    kgp_graph: NetworkXStorage = NetworkXStorage()
    k: int = 30
    k_nei: int = 3

    def wat_entity_linking(self, text):
        # Main method, text annotation with WAT entity linking system
        wat_url = 'https://wat.d4science.org/wat/tag/tag'
        payload = [("gcube-token", MY_GCUBE_TOKEN),
                ("text", text),
                ("lang", 'en'),
                ("tokenizer", "nlp4j"),
                ('debug', 9),
                ("method",
                    "spotter:includeUserHint=true:includeNamedEntity=true:includeNounPhrase=true,prior:k=50,filter-valid,centroid:rescore=true,topk:k=5,voting:relatedness=lm,ranker:model=0046.model,confidence:model=pruner-wiki.linear")]

        response = requests.get(wat_url, params=payload)

        return [WATAnnotation(a) for a in response.json()['annotations']]

    def wiki_kw_extract_chunk(self, chunk, prior_prob = 0.8):

        wat_annotations = self.wat_entity_linking(chunk)
        json_list = [w.json_dict() for w in wat_annotations]
        kw2chunk = defaultdict(set)
        chunk2kw = defaultdict(set)
        
        for wiki in json_list:
            if wiki['wiki_title'] != '' and wiki['prior_prob'] > prior_prob:
                kw2chunk[wiki['wiki_title']].add(chunk)
                chunk2kw[chunk].add(wiki['wiki_title'])

        return kw2chunk, chunk2kw

    async def _construct_graph(self, chunks: dict[str, TextChunk], prior_prob = 0.8):
        kw2chunk = defaultdict(set) #kw1 -> [chunk1, chunk2, ...]
        chunk2kw = defaultdict(set) #chunk -> [kw1, kw2, ...]

        with concurrent.futures.ProcessPoolExecutor() as executor:
            future_to_chunk = {executor.submit(self.wiki_kw_extract_chunk, value['content'], prior_prob): value['content'] for key, value in chunks.items()}
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk, inv_chunk = future.result()
                for key, value in chunk.items():
                    kw2chunk[key].update(value)
                for key, value in inv_chunk.items():
                    chunk2kw[key].update(value)

        for key in kw2chunk:
            kw2chunk[key] = list(kw2chunk[key])

        for key in chunk2kw:
            chunk2kw[key] = list(chunk2kw[key])

        chunk2id = {}
        for key, value in chunks.items():
            await self.kgp_graph.upsert_node(node_id = key, node_data = dict(chunk=value['content']))
            chunk2id[value['content']] = key
        
        for kw, chunk_list in kw2chunk.items():
            for i in range(len(chunk_list)):
                for j in range(i+1, len(chunk_list)):
                    # logger.info("{src_id} ---{kw}---> {tgt_id}".format(src_id = chunk2id[chunk_list[i]], tgt_id = chunk2id[chunk_list[j]], kw = kw))
                    await self.kgp_graph.upsert_edge(source_node_id=chunk2id[chunk_list[i]], target_node_id=chunk2id[chunk_list[j]], edge_data=dict(kw = kw))

    async def retrieve(self, question):
        corpus = dict({id: (await self.kgp_graph.get_node(id))['chunk'] for id in list(self.kgp_graph.graph.nodes)})
        candidates_idx = list(id for id in list(self.kgp_graph.graph.nodes))

        seed = question
        contexts = []
        
        idxs = await self.tf_idf(seed, candidates_idx, corpus, k = self.k // self.k_nei)

        cur_contexts = [corpus[_] for _ in idxs]
        next_reasons = [seed + '\n' + (await self.llm.aask(KGP_QUERY_PROMPT.format(question=question, context=context))) for context in cur_contexts]

        logger.info("next_reasons: {next_reasons}".format(next_reasons=next_reasons))

        visited = []

        for idx, next_reason in zip(idxs, next_reasons):
            nei_candidates_idx = list(self.kgp_graph.graph.neighbors(idx))
            nei_candidates_idx = [_ for _ in nei_candidates_idx if _ not in visited]
            if (nei_candidates_idx == []):
                continue

            next_contexts = await self.tf_idf(next_reason, nei_candidates_idx, corpus, k = self.k_nei)
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
        idxs = index.query(query_str = seed, top_k = k)
        return [candidates_idx[_] for _ in idxs]
    
    async def answer_question(self, question):
        contexts = await self.retrieve(question)
        context_str = '\n'.join('{index}: {context}'.format(index = i, context = c) for i, c in enumerate(contexts, start=1))
        
        logger.info(context_str)
        logger.info(question)

        answer = await self.llm.aask(KGP_QUERY_PROMPT.format(question=question, context=context_str))
        return answer

    def _exist_graph(self):
        pass

    def _extract_node(self):
        pass

    def _extract_relationship(self):
        pass
