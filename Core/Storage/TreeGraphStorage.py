from Core.Storage.BaseGraphStorage import BaseGraphStorage
from Core.Schema.TreeSchema import TreeNode, TreeSchema
from Core.Common.Logger import logger

from typing import Dict, Any

import os
import pickle


class TreeGraphStorage(BaseGraphStorage):
    name: str = "tree_data.pkl"
    _tree: TreeSchema = TreeSchema()

    async def _persist(self, force):
        if (os.path.exists(self.tree_pkl_file) and not force):
            return
        logger.info(f"Writing graph into {self.tree_pkl_file}")
        TreeGraphStorage.write_tree_graph(self.tree, self.tree_pkl_file)

    def write_tree_graph(tree, tree_pkl_file):
        with open(tree_pkl_file, "wb") as file:
            pickle.dump(tree, file)

    async def load_tree_graph(self, force) -> bool:
        # Attempting to load the graph from the specified pkl file
        logger.info(f"Attempting to load the tree from: {self.tree_pkl_file}")
        if os.path.exists(self.tree_pkl_file):
            try:
                with open(self.tree_pkl_file, "rb") as file:
                    self._tree = pickle.load(file)
                logger.info(
                    f"Successfully loaded tree from: {self.tree_pkl_file} with {len(self._tree.all_nodes)} nodes and {self._tree.num_layers} layers")
                return True
            except Exception as e:
                logger.error(
                    f"Failed to load tree from: {self.tree_pkl_file} with {e}! Need to re-build the tree.")
                return False
        else:
            # Pkl file doesn't exist; need to construct the tree from scratch
            logger.info("Pkl file does not exist! Need to build the tree from scratch.")
            return False

    @property
    def tree(self):
        return self._tree

    @property
    def tree_pkl_file(self):
        assert self.namespace is not None
        return self.namespace.get_save_path(self.name)

    @property
    def root_nodes(self):
        return self.tree.root_nodes

    @property
    def leaf_nodes(self):
        return self.tree.leaf_nodes

    @property
    def num_layers(self):
        return self.tree.num_layers

    @property
    def num_nodes(self):
        return self.tree.num_nodes

    def add_layer(self):
        if (self.num_layers == 0):
            self._tree.layer_to_nodes = []
            self._tree.all_nodes = []
        self._tree.layer_to_nodes.append([])

    def get_layer(self, layer: int):
        return self._tree.layer_to_nodes[layer]

    def upsert_node(self, node_id: int, node_data: Dict[str, Any]):
        node = TreeNode(index=node_id, text=node_data['text'], children=node_data['children'],
                        embedding=node_data['embedding'])
        layer = node_data['layer']
        self._tree.layer_to_nodes[layer].append(node)
        self._tree.all_nodes.append(node)
        return

    async def load_graph(self, force: bool = False) -> bool:
        return await self.load_tree_graph(force)

    async def persist(self, force):
        return await self._persist(force)

    async def get_nodes(self):
        return [{"content": node.text, "index": node.index} for node in self.tree.all_nodes]

    async def get_node_metadata(self):
        return ["index"]