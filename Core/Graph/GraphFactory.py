"""
Graph Factory.
"""
import os
from Core.Graph.BaseGraph import BaseGraph
from Core.Graph.ERGraph import ERGraph
from Core.Graph.PassageGraph import PassageGraph
from Core.Graph.TreeGraph import TreeGraph
from Core.Graph.RKGraph import RKGraph

from Core.Common.BaseFactory import ConfigBasedFactory


class GraphFactory(ConfigBasedFactory):
    def __init__(self):
        creators = {
            "er_graph": self._create_er_graph,
            "rkg_graph": self._create_rkg_graph,
            "tree_graph": self._create_tree_graph,
            "passage_graph": self._crease_passage_graph
        }
        super().__init__(creators)

    def get_graph(self, config, **kwargs) -> BaseGraph:
        """Key is PersistType."""
        return super().get_instance(config.graph_type, **kwargs)

    @staticmethod
    def _create_er_graph(config, **kwargs):
        return ERGraph(
            config, **kwargs
        )

    @staticmethod
    def _create_rkg_graph(config, **kwargs):
        return RKGraph(config, **kwargs)

    @staticmethod
    def _create_tree_graph(config, **kwargs):
        return TreeGraph(config, **kwargs)

    @staticmethod
    def _crease_passage_graph(config, **kwargs):
        return PassageGraph(config, **kwargs)


get_graph = GraphFactory().get_graph
