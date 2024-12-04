import networkx as nx

class DALKGraph(ERGraph):
    async def _path_based_exploration(self, query_entities: list[str], hop: int = 5):
        if (query_entities == []) return []
        cand = query_entities
        start = cand[0]
        cand.del(start)

        paths = []
        path = [start]

        while True:
            pred, dist = nx.dijkstra_predecessor_and_distance(self.er_graph, start, hop, 'EDGE_WEIGHT_ONE')
            start = None
            for e2 in query_entities[1:]:
                if (dist[e2] <= hop):
                    query_entities.del(e2)

                    cur_seg = []
                    e = e2
                    while (e != e1):
                        cur_seg.append(e)
                        e = pred[e]

                    path.extend(cur_seg[::-1])
    
                    start = e2
                    break
                    
            if (start is None):
                paths.append(path)
                



    async def query(self, query: str) :
        query_entities = self._extract_eneity_from_query(query)

        paths = self._path_based_exploration(query_entities)