"""Graph factory with lazy graph implementation imports.

The canonical ER path should not require optional dependencies used only by
Tree/RK/Passage graph implementations. Implementations are imported only when
the corresponding graph type is requested.
"""
from typing import Any, Callable, Dict

from Config.GraphConfig import GraphConfig
from Core.Common.Logger import logger
from Core.Graph.BaseGraph import BaseGraph


class GraphFactory:
    def __init__(self):
        self.creators: Dict[str, Callable[..., BaseGraph]] = {
            "er_graph": self._create_er_graph,
            "rkg_graph": self._create_rkg_graph,
            "tree_graph": self._create_tree_graph,
            "tree_graph_balanced": self._create_tree_graph_balanced,
            "passage_graph": self._create_passage_graph,
        }

    def get_graph(self, config: Any, **kwargs: Any) -> BaseGraph:
        """Create the graph implementation selected by ``config.graph.type``."""
        graph_type = config.graph.type
        creator = self.creators.get(graph_type)
        if creator is None:
            logger.error(
                "GraphFactory: unknown graph type '{}'. Available types: {}",
                graph_type,
                list(self.creators.keys()),
            )
            raise ValueError(f"Unknown graph type specified in config: {graph_type}")

        logger.info("GraphFactory: creating graph type '{}'", graph_type)
        return creator(config, **kwargs)

    @staticmethod
    def _validate_graph_config(full_config: Any, graph_type: str) -> None:
        if not isinstance(full_config.graph, GraphConfig):
            logger.warning(
                "GraphFactory: {} requested with unexpected graph config type {}",
                graph_type,
                type(full_config.graph),
            )

    @staticmethod
    def _create_er_graph(full_config: Any, **kwargs: Any) -> BaseGraph:
        from Core.Graph.ERGraph import ERGraph

        GraphFactory._validate_graph_config(full_config, "er_graph")
        return ERGraph(
            config=full_config.graph,
            llm=kwargs.get("llm"),
            encoder=kwargs.get("encoder"),
            storage_instance=kwargs.get("storage_instance"),
        )

    @staticmethod
    def _create_rkg_graph(full_config: Any, **kwargs: Any) -> BaseGraph:
        from Core.Graph.RKGraph import RKGraph

        GraphFactory._validate_graph_config(full_config, "rkg_graph")
        return RKGraph(
            config=full_config,
            llm=kwargs.get("llm"),
            encoder=kwargs.get("encoder"),
        )

    @staticmethod
    def _create_tree_graph(full_config: Any, **kwargs: Any) -> BaseGraph:
        from Core.Graph.TreeGraph import TreeGraph

        GraphFactory._validate_graph_config(full_config, "tree_graph")
        return TreeGraph(
            config=full_config,
            llm=kwargs.get("llm"),
            encoder=kwargs.get("encoder"),
        )

    @staticmethod
    def _create_tree_graph_balanced(full_config: Any, **kwargs: Any) -> BaseGraph:
        from Core.Graph.TreeGraphBalanced import TreeGraphBalanced

        GraphFactory._validate_graph_config(full_config, "tree_graph_balanced")
        return TreeGraphBalanced(
            config=full_config,
            llm=kwargs.get("llm"),
            encoder=kwargs.get("encoder"),
        )

    @staticmethod
    def _create_passage_graph(full_config: Any, **kwargs: Any) -> BaseGraph:
        from Core.Graph.PassageGraph import PassageGraph

        GraphFactory._validate_graph_config(full_config, "passage_graph")
        return PassageGraph(
            config=full_config,
            llm=kwargs.get("llm"),
            encoder=kwargs.get("encoder"),
        )


get_graph = GraphFactory().get_graph
