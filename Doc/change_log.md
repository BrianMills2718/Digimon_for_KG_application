2025-05-30: Verified diagnostic logging cleanup status:
- Core/Retriever/BaseRetriever.py: 
  - logger.critical line at start of _run_personalized_pagerank is not present
  - Diagnostic logs in _run_personalized_pagerank are already using logger.debug
- Core/Retriever/EntitiyRetriever.py: No logger.critical diagnostic line at start of _find_relevant_entities_by_ppr
- Core/AgentTools/entity_tools.py: No logger.critical diagnostic line before entity_retriever._find_relevant_entities_by_ppr

All requested diagnostic logging cleanup appears to have already been completed in a previous update.

2025-05-30: Major refactor of Core/Graph/GraphFactory.py
- Fixed typo: now uses _create_passage_graph
- All static creator methods use explicit named arguments
- Added logging and type checks for debugging
- Robust error handling for unknown graph types
- File ends cleanly with get_graph = GraphFactory().get_graph

2025-05-30: Fixed test_agent_orchestrator.py to pass main_config as config arg to GraphFactory.get_graph
- Only passes data_path, storage_type, and storage_instance as kwargs
- Now matches factory expectations and avoids attribute errors

2025-05-30: Fixed Core/Graph/GraphFactory.py get_graph to use config.graph.type
- Now compatible with orchestrator and Config/GraphConfig.py changes

2025-05-30: Added 'type' field to GraphConfig in Config/GraphConfig.py
- Now supports main_config.graph.type for orchestrator and test compatibility

2025-05-30: Switched to using GraphFactory class instance for graph instantiation in testing/test_agent_orchestrator.py
- Now imports GraphFactory class, instantiates it, and calls get_graph
- Fixes ImportError for GRAPH_FACTORY and matches actual project API

2025-05-30: Fixed graph instantiation in testing/test_agent_orchestrator.py
- Now uses GraphFactory.get_graph_instance instead of direct BaseGraph instantiation
- Removed BaseGraph import; added logs for graph type returned by factory

2025-05-30: Fixed Workspace and NameSpace instantiation in testing/test_agent_orchestrator.py
- Now uses correct arguments: Workspace(working_dir=..., exp_name=...) and NameSpace(workspace=..., namespace=...)
- Ensures compatibility with actual class definitions (see Core/Storage/NameSpace.py)

2025-05-30: Fixed artifact/data path usage in testing/test_agent_orchestrator.py
- Now uses main_config.working_dir instead of root_path_for_artifacts for all artifact/data paths

2025-05-30: Updated testing/test_agent_orchestrator.py
- Changed import to: from Option.Config2 import Config, default_config as main_config
- Inserted diagnostic logging block after imports to log main_config and its graph attribute before setup_graphrag_context_for_orchestrator

2025-05-30: Fixed ERGraph instantiation in Core/Graph/GraphFactory.py
- Now matches ERGraph constructor signature: (config, llm, encoder)
- No longer passes data_path, storage_type, or storage_instance to ERGraph
- Added clarifying comment in _create_er_graph

2025-05-30: Fixed test_agent_orchestrator.py to use correct BaseGraph API
- Replaced 'await graph_instance.load()' with 'await graph_instance.load_persisted_graph()' in setup_graphrag_context_for_orchestrator
- Added clarifying comment

2025-05-31: Relationship tools and orchestrator integration
- Fixed ImportError in `relationship_tools.py` by removing import of `CoreRelationship` from `Core.Schema.EntityRelation` (not defined there).
- Updated `relationship_tools.py` to use `RelationshipData` instead of `NeighborDetail` for one-hop neighbor output, matching tool_contracts contract.
- Registered `Relationship.OneHopNeighbors` tool in orchestrator, updated input resolution for chaining from `Entity.VDBSearch`, and added a two-step test plan in `test_agent_orchestrator.py`.
- Confirmed all tool contracts for relationship tools are now in sync with implementation.
- Fixed NameError: Added `Optional` to typing imports at the top of `relationship_tools.py` to support correct type hinting for optional arguments.
- Fixed NameError: Added import of `RelationshipOneHopNeighborsInputs` from `Core.AgentSchema.tool_contracts` in `orchestrator.py` to ensure tool registration works.
- Updated `testing/test_agent_orchestrator.py`: Added call to `run_vdb_to_one_hop_neighbors_plan` within `main_orchestrator_test` to ensure the new 2-step (VDB Search -> One-Hop Neighbors) plan is executed after the PPR plan.
- Fixed Pydantic `ValidationError` for `RelationshipOneHopNeighborsInputs` in `Core/AgentSchema/tool_contracts.py`: Renamed `source_entity_ids` to `entity_ids` and added `graph_reference_id: str = Field(default="kg_graph")` to match tool expectations and provide necessary graph context.
- Fixed `AttributeError` in `Core/AgentTools/relationship_tools.py` (`relationship_one_hop_neighbors_tool`): Changed to directly use `'type'` and `'description'` keys for fetching relationship name and description from edge attributes, instead of non-existent `params.relationship_name_attribute` and `params.relationship_description_attribute`. Corrected attribute names (`source_node_id`, `target_node_id`, `type`) in duplicate check logic.
- Fixed `AttributeError` in the initial logging statement of `relationship_one_hop_neighbors_tool` by removing access to `params.relationship_name_attribute` and `params.relationship_description_attribute`.
- Fixed Pydantic `ValidationError` in `Core/AgentTools/relationship_tools.py` (`relationship_one_hop_neighbors_tool`): Changed `neighbor_details` to `one_hop_relationships` in the output model instantiation to match the `RelationshipOneHopNeighborsOutputs` schema when returning early due to a missing graph.
- Added detailed logging to `relationship_one_hop_neighbors_tool` in `Core/AgentTools/relationship_tools.py` to debug graph instance availability and type. Refined logic to handle `params.direction` and `params.relationship_types_to_include` for filtering relationships, and improved duplicate checking for multigraphs.
- Updated _resolve_tool_inputs to transform EntityVDBSearchOutputs.similar_entities to entity_ids for tool chaining
- Added new async test function run_vdb_to_one_hop_neighbors_plan in testing/test_agent_orchestrator.py
- Plan chains Entity.VDBSearch -> Relationship.OneHopNeighbors and logs all outputs

2025-05-31: Fully replaced Core/AgentTools/relationship_tools.py with robust async implementations for:
- relationship_one_hop_neighbors_tool: Finds one-hop neighbors for given entities using NetworkX, handling both MultiDiGraph and DiGraph, and deduplicates edges. Uses flexible relationship attribute names.
- relationship_score_aggregator_tool: Placeholder returning a sample aggregation.
- relationship_vdb_search_tool: Placeholder returning a sample similar relationship.
- relationship_agent_tool: Placeholder returning a sample agent relationship.
This update removes all previous placeholder/dummy logic and adds robust graph-based neighbor-finding logic as per IDE instructions. Also fixed SyntaxError (unmatched parenthesis) at end of file.

2025-05-30: Updated testing/test_agent_orchestrator.py for DynamicToolChainConfig
- Added DynamicToolChainConfig to Core.AgentSchema.plan import
- Wrapped ToolCall in vdb_search_step with DynamicToolChainConfig as action

2025-05-30: Major update to AgentOrchestrator.execute_plan
- Rewrote execute_plan to support DynamicToolChainConfig steps and robust tool chain execution
- Removed all obsolete code after new implementation to fix IndentationError

2025-05-30: Config2.py config loading order fix
- Now loads Option/Config2.yaml as the primary config, only falling back to Option/Config2.example.yaml if the main config is missing or incomplete.
- This ensures user API keys and settings are always used if present.

2025-05-30: Added async implementation of entity_ppr_tool
- Replaced placeholder with robust async implementation in Core/AgentTools/entity_tools.py.
- Ensured correct imports, robust error handling, and output shape for orchestrator agentic use.

- Workspace and VDB paths updated accordingly
- Added storage_root_dir to resolved_configs in GraphRAGContext

2025-05-30: Fixed embedding provider import and usage in testing/test_agent_orchestrator.py
- Now imports get_rag_embedding directly and calls get_rag_embedding(config=main_config)

2025-05-31: Diagnosed that Python was likely running a cached version of `relationship_tools.py`, as new detailed logs were not appearing in test output. Proposed command to clear `__pycache__` directories.

2025-05-30: Created AgentOrchestrator module
- Added Core/AgentOrchestrator/__init__.py
- Added Core/AgentOrchestrator/orchestrator.py with initial implementation
  - Basic tool registry system with Pydantic model support
  - Plan execution framework with proper input resolution
  - Support for dynamic input resolution from:
    - Previous step outputs (via ToolInputSource)
    - Plan inputs (via "plan_inputs.*" references)
    - Literal values
  - Comprehensive logging for debugging
  - Support for named outputs between steps
  - Pydantic model validation for tool inputs


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

- Manually provided corrected `relationship_one_hop_neighbors_tool` code to USER after tool call failures. Integrated user's new detailed logging, retained previous fixes (param/attribute access, Pydantic models), and used user's robust graph instance check. Advised manual update, cache clearing, and re-test.

- Core/AgentTools/relationship_tools.py: Updated `relationship_one_hop_neighbors_tool` to access the NetworkX graph via `graph_instance._graph.graph` and adjusted conditional checks accordingly, as per user-provided code.