[2025-05-30 05:06] Updated Core/GraphRAG.py:
- In build_and_persist_artifacts, replaced hardcoded force=True with force=self.config.graph.force for entities_vdb.build_index.
- Updated preceding logger.info to a conditional log statement based on self.config.graph.force, for clarity and accuracy.

[2025-05-30 05:34] Updated testing/test_single_tool_entity_vdb_search.py:
- Changed VDB existence check to look for default__vector_store.json instead of vector_index.faiss.
- Improved warnings to check for directory and file existence, and provide more helpful output for debugging VDB presence.

[2025-05-30 06:07] Updated Core/AgentSchema/context.py:
- Added imports for BaseGraph, BaseIndex, and BaseCommunity.
- Added new fields to GraphRAGContext: graph_instance (BaseGraph), entities_vdb_instance (BaseIndex), relations_vdb_instance (BaseIndex), and community_instance (BaseCommunity), enabling direct resource access for agent tools.

[2025-05-30 06:25] Implemented entity_ppr_tool in Core/AgentTools/entity_tools.py:
- Added full async implementation for entity_ppr_tool, using EntityRetriever and RetrieverConfig to perform Personalized PageRank (PPR) for entities in a graph, with robust config handling and output formatting.

[2025-05-30 06:36] Created testing/test_single_tool_entity_ppr.py:
- Added a standalone async test script for entity_ppr_tool. The script loads config, embedding provider, graph, and VDB, builds a GraphRAGContext, and runs the tool with example seed entities, logging results.

[2025-05-30 06:52] Updated Option/Config2.py:
- Made Config.default() robust: it now tries Option/Config2.example.yaml, then Option/Config2.yaml, and finally falls back to programmatic defaults for all required fields (llm, embedding, graph, chunk, retriever, query_config) to prevent ValidationError at import time.

[2025-05-30 06:58] Fixed circular import in Option/Config2.py:
- Moved `from Core.Common.Logger import logger` into the Config.default() method body to resolve ImportError due to circular dependency with Core.Common.Logger.

[2025-05-30 07:02] Fully broke circular import in Option/Config2.py:
- Removed all logger usage from Config.default() and replaced with print statements, so Option/Config2.py no longer imports Core.Common.Logger at all. This severs the import cycle with Core.Common.Logger.

[2025-05-30 07:06] Fixed import error in Core/AgentTools/entity_tools.py:
- Changed import to match typo in actual filename: from Core.Retriever.EntitiyRetriever import EntityRetriever (was EntityRetriever.py, actual file is EntitiyRetriever.py).

[2025-05-30 07:10] Fixed RetrieverConfig import path in Core/AgentTools/entity_tools.py:
- Changed from Core.Config.RetrieverConfig import RetrieverConfig to Config.RetrieverConfig import RetrieverConfig to match project structure.

[2025-05-30 07:14] Removed unused/incorrect IndexFactory import from testing/test_single_tool_entity_ppr.py:
- Commented out from Core.Index.IndexFactory import IndexFactory because IndexFactory does not exist and is not needed in this script.

[2025-05-30 07:17] Fixed RetrieverConfig import path in testing/test_single_tool_entity_ppr.py:
- Changed from Core.Config.RetrieverConfig import RetrieverConfig to Config.RetrieverConfig import RetrieverConfig to match project structure.

[2025-05-30 07:34] Fixed NetworkXStorage instantiation in testing/test_single_tool_entity_ppr.py:
- Removed old direct instantiation with config/namespace args.
- Updated to use async helper get_loaded_graph_instance with correct parameters, matching TestGraph construction logic and avoiding TypeError.

[2025-05-30 07:41] Updated get_loaded_graph_instance in testing/test_single_tool_entity_ppr.py:
- Now creates Workspace object and passes it to NameSpace for correct storage path resolution, aligning with project design.

[2025-05-30 07:44] Fixed async graph loading and Pydantic input error in testing/test_single_tool_entity_ppr.py:
- Awaited NetworkXStorage.load_graph(force=False) to ensure the graph is loaded and avoid RuntimeWarning.
- Added required graph_reference_id argument to EntityPPRInputs to resolve ValidationError.

[2025-05-30 07:50] Fixed graph file path logic for NetworkXStorage:
- Updated Core/Storage/NetworkXStorage.py: graphml_xml_file now joins directory from NameSpace.get_save_path() with self.name.
- Updated testing/test_single_tool_entity_ppr.py: explicitly sets _storage.name to "graph_storage_nx_data.graphml" to match actual file.

[2025-05-30 07:54] Replaced entity_ppr_tool placeholder with robust EntityRetriever-based implementation in Core/AgentTools/entity_tools.py:
- Now calls EntityRetriever._find_relevant_entities_by_ppr and returns real PPR results, handling config and context robustly.

[2025-05-30 10:06] Updated Core/Retriever/BaseRetriever.py:
- Added logger.critical("BaseRetriever: ENTERING _run_personalized_pagerank method NOW!") as the first line inside async def _run_personalized_pagerank, before any other code, for method entry diagnostics.

[2025-05-30 10:25] Enhanced diagnostic logging:
- Added logger.critical statement at the start of EntityRetriever._find_relevant_entities_by_ppr in Core/Retriever/EntitiyRetriever.py.
- Added logger.critical diagnostic statement before calling entity_retriever._find_relevant_entities_by_ppr in entity_ppr_tool in Core/AgentTools/entity_tools.py.

[2025-05-30 10:27] Bugfix: Decorator return in RetrieverFactory
- Modified register_retriever_method in Core/Retriever/RetrieverFactory.py so that the inner decorator(func) returns func, ensuring decorator chaining and registration work as expected.

[2025-05-30 10:44] Cleanup: Removed temporary diagnostic logging
- Core/Retriever/BaseRetriever.py: Removed logger.critical from _run_personalized_pagerank and reverted logger.error to logger.debug in the first diagnostic block.
- Core/Retriever/EntitiyRetriever.py: Removed logger.critical from _find_relevant_entities_by_ppr.
- Core/AgentTools/entity_tools.py: Removed the diagnostic logger.critical and associated block before calling entity_retriever._find_relevant_entities_by_ppr in entity_ppr_tool.
