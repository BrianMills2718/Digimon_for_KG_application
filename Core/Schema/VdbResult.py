from abc import ABC, abstractmethod
import asyncio
class EntityResult(ABC):
    @abstractmethod
    def get_node_data(self):
        pass


class ColbertNodeResult(EntityResult):
    def __init__(self, node_idxs, ranks, scores):
        self.node_idxs = node_idxs
        self.ranks = ranks
        self.scores = scores

    async def get_node_data(self, graph, score = False):
        nodes =  await asyncio.gather(*[graph.get_node_by_index(node_idx) for node_idx in self.node_idxs])
        if score:

            return nodes, [r for r in self.scores]
        else:
            return nodes
    async def get_tree_node_data(self, graph, score = False):
  
        nodes = await asyncio.gather( *[ graph.get_node(node_idx) for node_idx in self.node_idxs])
        if score:

            return nodes, [r for r in self.scores]
        else:
            return nodes

class VectorIndexNodeResult(EntityResult):
    def __init__(self, results):
        self.results = results

    async def get_node_data(self, graph, score = False):

        nodes = await asyncio.gather( *[ graph.get_node(r.metadata["entity_name"]) for r in self.results])
        if score:

            return nodes, [r.score for r in self.results]
        else:
            return nodes
    
    async def get_tree_node_data(self, graph, score = False): # graph is TreeGraphStorage (or any graph providing .entity_metakey)
        processed_nodes = []
        for r_idx, r in enumerate(self.results):
            node_id_from_meta = r.metadata.get(graph.entity_metakey) # graph.entity_metakey should be "index"
            layer_from_meta = r.metadata.get("layer", -1) # Get layer directly from VDB metadata
            if node_id_from_meta is None:
                logger.warning(f"Node ID (via metakey '{graph.entity_metakey}') not found in VDB result metadata: {r.metadata}")
                continue
            node_obj = await graph.get_node(node_id_from_meta)
            if node_obj and hasattr(node_obj, 'text'):
                node_text = node_obj.text
                current_node_data = {
                    "id": node_id_from_meta, # This is the original node ID
                    "text": node_text,
                    "layer": layer_from_meta 
                }
                if score:
                    current_node_data["vdb_score"] = r.score
                processed_nodes.append(current_node_data)
            else:
                logger.warning(f"Could not retrieve text for node id: {node_id_from_meta} or node object is incomplete. Metadata: {r.metadata}")
        if not processed_nodes:
            logger.warning("No nodes were successfully processed in get_tree_node_data.")
        if score:
            return processed_nodes, [res.score for res in self.results if res.metadata.get(graph.entity_metakey) is not None]
        else:
            return processed_nodes

class RelationResult(ABC):
    @abstractmethod
    def get_edge_data(self):
        pass

class  VectorIndexEdgeResult(RelationResult):
    def __init__(self, results):
        self.results = results

    async def get_edge_data(self, graph, score = False):

        nodes = await asyncio.gather( *[ graph.get_edge(r.metadata["src_id"], r.metadata["tgt_id"]) for r in self.results])
        if score:

            return nodes, [r.score for r in self.results]
        else:
            return nodes


class SubgraphResult(ABC):
    @abstractmethod
    def get_subgraph_data(self):
        pass


class  VectorIndexSubgraphResult(SubgraphResult):
    def __init__(self, results):
        self.results = results

    async def get_subgraph_data(self,score = False):
        subgraphs_data = list(map(lambda x: {"source_id" : x.metadata["source_id"], "subgraph_content": x.text}, self.results))
        if score:
            return subgraphs_data, [r.score for r in self.results]
        else:
            return subgraphs_data

class ColbertEdgeResult(RelationResult):
    def __init__(self, edge_idxs, ranks, scores):
        self.edge_idxs = edge_idxs
        self.ranks = ranks
        self.scores = scores

    async def get_edge_data(self, graph):
        return await asyncio.gather(
            *[(graph.get_edge_by_index(edge_idx), self.scores[idx]) for idx, edge_idx in enumerate(self.edge_idxs)]
        )