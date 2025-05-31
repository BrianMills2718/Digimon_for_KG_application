"""
Graph Factory.
"""
from typing import Any, Dict # Ensure Any and Dict are imported
from Core.Graph.BaseGraph import BaseGraph # Assuming GraphType is also in BaseGraph or imported elsewhere
from Core.Graph.ERGraph import ERGraph
from Core.Graph.PassageGraph import PassageGraph
from Core.Graph.TreeGraph import TreeGraph
from Core.Graph.TreeGraphBalanced import TreeGraphBalanced
from Core.Graph.RKGraph import RKGraph
from Config.GraphConfig import GraphConfig # For type hinting of the config passed to graph constructors
from Core.Common.Logger import logger # For logging potential issues

# If GraphType enum is used by main_config.graph.type, ensure it's accessible.
# It's often defined in BaseGraph.py or GraphSchema.py. For now, assuming string keys.
# from Core.Schema.GraphSchema import GraphType # Example if GraphType is an Enum

class GraphFactory():
    def __init__(self):
        # Using string keys as derived from config.graph.type
        self.creators: Dict[str, callable] = {
            "er_graph": self._create_er_graph,
            "rkg_graph": self._create_rkg_graph,
            "tree_graph": self._create_tree_graph,
            "tree_graph_balanced": self._create_tree_graph_balanced,
            "passage_graph": self._create_passage_graph # Corrected typo here
        }

    def get_graph(self, config: Any, **kwargs) -> BaseGraph: # config is the full Config object
        """
        Creates a graph instance based on the type specified in config.graph.type.
        'config' is the full application Config object.
        'kwargs' should contain data_path, storage_type, storage_instance.
        """
        graph_type_str = config.graph.type # This should work as GraphConfig now has 'type'
        
        creator_method = self.creators.get(graph_type_str)
        
        if not creator_method:
            logger.error(f"GraphFactory: Unknown graph type specified in config: '{graph_type_str}'. Available types: {list(self.creators.keys())}")
            raise ValueError(f"Unknown graph type specified in config: {graph_type_str}")
        
        logger.info(f"GraphFactory: Creating graph of type '{graph_type_str}' with kwargs: {kwargs}")
        # The 'config' passed to creator methods is the full main_config.
        # **kwargs will contain data_path, storage_type, storage_instance.
        return creator_method(config, **kwargs)

    @staticmethod
    def _create_er_graph(full_config: Any, **constructor_kwargs: Any) -> ERGraph:
        logger.debug(f"GraphFactory._create_er_graph: full_config type: {type(full_config)}, constructor_kwargs: {constructor_kwargs}")
        if not isinstance(full_config.graph, GraphConfig):
             logger.error(f"_create_er_graph: full_config.graph is not a GraphConfig object. Type: {type(full_config.graph)}")
             # Potentially raise error or return None
        # ERGraph expects (config, llm, encoder) as its constructor signature
        # Unlike other graph types, do not pass data_path or storage_type, but pass storage_instance if provided
        return ERGraph(
            config=full_config.graph,  # Only the GraphConfig part
            llm=constructor_kwargs.get("llm"),
            encoder=constructor_kwargs.get("encoder"),
            storage_instance=constructor_kwargs.get("storage_instance")
        )

    @staticmethod
    def _create_rkg_graph(full_config: Any, **constructor_kwargs: Any) -> RKGraph:
        logger.debug(f"GraphFactory._create_rkg_graph: full_config type: {type(full_config)}, constructor_kwargs: {constructor_kwargs}")
        if not isinstance(full_config.graph, GraphConfig):
             logger.error(f"_create_rkg_graph: full_config.graph is not a GraphConfig object. Type: {type(full_config.graph)}")
        return RKGraph(
            data_path=constructor_kwargs.get("data_path"),
            config=full_config.graph, # Pass the GraphConfig part
            storage_type=constructor_kwargs.get("storage_type", "networkx"),
            storage_instance=constructor_kwargs.get("storage_instance")
        )

    @staticmethod
    def _create_tree_graph(full_config: Any, **constructor_kwargs: Any) -> TreeGraph:
        logger.debug(f"GraphFactory._create_tree_graph: full_config type: {type(full_config)}, constructor_kwargs: {constructor_kwargs}")
        if not isinstance(full_config.graph, GraphConfig):
             logger.error(f"_create_tree_graph: full_config.graph is not a GraphConfig object. Type: {type(full_config.graph)}")
        # Assuming TreeGraph.__init__ also expects data_path, config (GraphConfig), storage_type, storage_instance
        return TreeGraph(
            data_path=constructor_kwargs.get("data_path"),
            config=full_config.graph, # Pass the GraphConfig part
            storage_type=constructor_kwargs.get("storage_type", "networkx"),
            storage_instance=constructor_kwargs.get("storage_instance")
        )

    @staticmethod
    def _create_tree_graph_balanced(full_config: Any, **constructor_kwargs: Any) -> TreeGraphBalanced:
        logger.debug(f"GraphFactory._create_tree_graph_balanced: full_config type: {type(full_config)}, constructor_kwargs: {constructor_kwargs}")
        if not isinstance(full_config.graph, GraphConfig):
             logger.error(f"_create_tree_graph_balanced: full_config.graph is not a GraphConfig object. Type: {type(full_config.graph)}")
        # Assuming TreeGraphBalanced.__init__ also expects data_path, config (GraphConfig), storage_type, storage_instance
        return TreeGraphBalanced(
            data_path=constructor_kwargs.get("data_path"),
            config=full_config.graph, # Pass the GraphConfig part
            storage_type=constructor_kwargs.get("storage_type", "networkx"),
            storage_instance=constructor_kwargs.get("storage_instance")
        )

    @staticmethod
    def _create_passage_graph(full_config: Any, **constructor_kwargs: Any) -> PassageGraph: # Corrected typo here
        logger.debug(f"GraphFactory._create_passage_graph: full_config type: {type(full_config)}, constructor_kwargs: {constructor_kwargs}")
        if not isinstance(full_config.graph, GraphConfig):
             logger.error(f"_create_passage_graph: full_config.graph is not a GraphConfig object. Type: {type(full_config.graph)}")
        # Assuming PassageGraph.__init__ also expects data_path, config (GraphConfig), storage_type, storage_instance
        return PassageGraph(
            data_path=constructor_kwargs.get("data_path"),
            config=full_config.graph, # Pass the GraphConfig part
            storage_type=constructor_kwargs.get("storage_type", "networkx"),
            storage_instance=constructor_kwargs.get("storage_instance")
        )

# Make the get_graph method available at the module level, bound to a GraphFactory instance
get_graph = GraphFactory().get_graph

get_graph = GraphFactory().get_graph
