# Core/AgentOrchestrator/orchestrator.py

from typing import Dict, Any, List, Optional, Tuple, Type, Union
from Core.AgentSchema.plan import ExecutionPlan, ExecutionStep, ToolCall, ToolInputSource, DynamicToolChainConfig
from Core.AgentSchema.tool_contracts import (
    EntityVDBSearchInputs,
    EntityVDBBuildInputs,
    EntityPPRInputs,
    EntityOneHopInput,
    EntityRelNodeInput,
    RelationshipOneHopNeighborsInputs,
    RelationshipVDBBuildInputs,
    RelationshipVDBSearchInputs,
    ChunkFromRelationshipsInputs,
    ChunkGetTextForEntitiesInput,
    EntityVDBSearchOutputs, 
    VDBSearchResultItem, 
    GraphVisualizerInput, 
    GraphAnalyzerInput
)
from Core.AgentSchema.corpus_tool_contracts import PrepareCorpusInputs
from Core.AgentSchema.graph_construction_tool_contracts import BuildERGraphInputs, BuildRKGraphInputs, BuildTreeGraphInputs, BuildTreeGraphBalancedInputs, BuildPassageGraphInputs
from Core.AgentSchema.context import GraphRAGContext
from Core.Common.Logger import logger
from pydantic import BaseModel
from Option.Config2 import Config
from Core.Provider.BaseLLM import BaseLLM
from Core.Provider.BaseEmb import BaseEmb as LlamaIndexBaseEmbedding
from Core.Chunk.ChunkFactory import ChunkFactory

# Import tool functions (we'll add more as they become relevant)
from Core.AgentTools.entity_tools import entity_vdb_search_tool, entity_ppr_tool
from Core.AgentTools.entity_onehop_tools import entity_onehop_neighbors_tool
from Core.AgentTools.entity_relnode_tools import entity_relnode_extract_tool
from Core.AgentTools.relationship_tools import relationship_one_hop_neighbors_tool, relationship_vdb_build_tool, relationship_vdb_search_tool
from Core.AgentTools.graph_construction_tools import (
    build_er_graph, build_rk_graph, build_tree_graph, build_tree_graph_balanced, build_passage_graph
)
from Core.AgentTools.corpus_tools import prepare_corpus_from_directory
from Core.AgentTools.graph_visualization_tools import visualize_graph
from Core.AgentTools.graph_analysis_tools import analyze_graph
from Core.AgentTools.chunk_tools import chunk_from_relationships_tool, chunk_get_text_for_entities_tool
from Core.AgentTools.entity_vdb_tools import entity_vdb_build_tool
# from Core.AgentTools.subgraph_tools import ...
# from Core.AgentTools.community_tools import ...

class AgentOrchestrator:
    def __init__(self, 
                 main_config: Config, 
                 llm_instance: BaseLLM, 
                 encoder_instance: LlamaIndexBaseEmbedding, 
                 chunk_factory: ChunkFactory, 
                 graphrag_context: Optional[GraphRAGContext] = None):
        self.main_config = main_config
        self.llm = llm_instance
        self.encoder = encoder_instance
        self.chunk_factory = chunk_factory
        self.graphrag_context = graphrag_context
        self._tool_registry = self._register_tools()
        self.step_outputs: Dict[str, Dict[str, Any]] = {} # To store outputs of each step

    def _register_tools(self) -> Dict[str, Tuple[callable, Type[BaseModel]]]:
        """
        Registry mapping tool_id strings to (callable_function, pydantic_input_model_class) tuples.
        """
        registry = {
            "Entity.VDBSearch": (entity_vdb_search_tool, EntityVDBSearchInputs),
            "Entity.VDB.Build": (entity_vdb_build_tool, EntityVDBBuildInputs),
            "Entity.PPR": (entity_ppr_tool, EntityPPRInputs),
            "Entity.Onehop": (entity_onehop_neighbors_tool, EntityOneHopInput),
            "Entity.RelNode": (entity_relnode_extract_tool, EntityRelNodeInput),
            "Relationship.OneHopNeighbors": (relationship_one_hop_neighbors_tool, RelationshipOneHopNeighborsInputs),
            "Relationship.VDB.Build": (relationship_vdb_build_tool, RelationshipVDBBuildInputs),
            "Relationship.VDB.Search": (relationship_vdb_search_tool, RelationshipVDBSearchInputs),
            "Chunk.FromRelationships": (chunk_from_relationships_tool, ChunkFromRelationshipsInputs),
            "Chunk.GetTextForEntities": (chunk_get_text_for_entities_tool, ChunkGetTextForEntitiesInput),
            # Add other tools here as they are implemented
            "graph.BuildERGraph": (build_er_graph, BuildERGraphInputs),
            "graph.BuildRKGraph": (build_rk_graph, BuildRKGraphInputs),
            "graph.BuildTreeGraph": (build_tree_graph, BuildTreeGraphInputs),
            "graph.BuildTreeGraphBalanced": (build_tree_graph_balanced, BuildTreeGraphBalancedInputs),
            "graph.BuildPassageGraph": (build_passage_graph, BuildPassageGraphInputs),
            "corpus.PrepareFromDirectory": (prepare_corpus_from_directory, PrepareCorpusInputs),
            "graph.Visualize": (visualize_graph, GraphVisualizerInput),
            "graph.Analyze": (analyze_graph, GraphAnalyzerInput),
        }
        logger.info(f"AgentOrchestrator: Registered {len(registry)} tools with Pydantic models: {list(registry.keys())}")
        return registry

    def _resolve_single_input_source(
        self,
        target_input_name: str, 
        source_identifier: Any, 
        plan_inputs: Dict[str, Any], 
        source_location_type: str 
    ) -> Any:
        source_value = None
        current_tis = None
        plan_inputs = plan_inputs or {}

        if isinstance(source_identifier, dict) and \
           "from_step_id" in source_identifier and \
           "named_output_key" in source_identifier:
            try:
                current_tis = ToolInputSource(**source_identifier)
                logger.debug(f"Orchestrator ({source_location_type}) '{target_input_name}': Parsed dict into ToolInputSource.")
            except Exception as e_tis:
                logger.error(f"Orchestrator ({source_location_type}) '{target_input_name}': Failed to parse ToolInputSource dict {source_identifier}. Error: {e_tis}. Treating as literal.")
                source_value = source_identifier
        elif isinstance(source_identifier, ToolInputSource):
            current_tis = source_identifier

        if current_tis:
            from_step_id = current_tis.from_step_id
            named_output_key = current_tis.named_output_key
            if from_step_id in self.step_outputs and \
               named_output_key in self.step_outputs[from_step_id]:
                raw_source_value = self.step_outputs[from_step_id][named_output_key]
                logger.debug(f"Orchestrator ({source_location_type}) '{target_input_name}': Retrieved from step '{from_step_id}', key '{named_output_key}'. Raw type: {type(raw_source_value)}")

                if (target_input_name == "seed_entity_ids" or target_input_name == "entity_ids") and \
                    isinstance(raw_source_value, (EntityVDBSearchOutputs, list)): 

                    entities_list_for_ids = []
                    if isinstance(raw_source_value, EntityVDBSearchOutputs):
                        entities_list_for_ids = raw_source_value.similar_entities
                    elif isinstance(raw_source_value, list): 
                        entities_list_for_ids = raw_source_value

                    extracted_ids = []
                    for item in entities_list_for_ids:
                        if isinstance(item, VDBSearchResultItem) and hasattr(item, 'entity_name'): 
                            extracted_ids.append(item.entity_name)
                        elif isinstance(item, dict):
                            # Handle plain dictionary results from VDB search
                            # For graph operations, prioritize entity_name since graphs use entity names as node IDs
                            if 'entity_name' in item:
                                extracted_ids.append(item['entity_name'])
                            elif 'node_id' in item:
                                extracted_ids.append(item['node_id'])
                        elif isinstance(item, tuple) and len(item) > 0 and isinstance(item[0], str): 
                            extracted_ids.append(item[0])
                    source_value = extracted_ids
                    logger.info(f"Orchestrator ({source_location_type}) '{target_input_name}': Transformed VDB output for. Extracted {len(source_value)} IDs.")
                else: 
                    source_value = raw_source_value
                    logger.debug(f"Orchestrator ({source_location_type}) '{target_input_name}': Using raw output from previous step.")
            else:
                logger.error(f"Orchestrator ({source_location_type}) '{target_input_name}': Output key '{named_output_key}' not in step '{from_step_id}'. Available: {list(self.step_outputs.get(from_step_id, {}).keys())}")
                source_value = None 
        elif isinstance(source_identifier, str) and source_identifier.startswith("plan_inputs."):
            input_key = source_identifier.split("plan_inputs.")[1]
            if input_key in plan_inputs:
                source_value = plan_inputs[input_key]
                logger.debug(f"Orchestrator ({source_location_type}) '{target_input_name}': Resolved from plan_inputs, key '{input_key}'.")
            else:
                logger.error(f"Orchestrator ({source_location_type}) '{target_input_name}': Key '{input_key}' not in plan_inputs. Keys: {list(plan_inputs.keys())}")
                source_value = None
        else: 
            source_value = source_identifier
            logger.debug(f"Orchestrator ({source_location_type}) '{target_input_name}': Resolved as literal value.")
        return source_value

    def _resolve_tool_inputs(
        self,
        tool_call_inputs: Optional[Dict[str, Any]],
        tool_call_parameters: Optional[Dict[str, Any]],
        plan_inputs: Optional[Dict[str, Any]] 
    ) -> Dict[str, Any]:
        final_resolved_params: Dict[str, Any] = {}

        if tool_call_parameters:
            for param_name, source_identifier in tool_call_parameters.items():
                final_resolved_params[param_name] = self._resolve_single_input_source(
                    param_name, source_identifier, plan_inputs or {}, "parameter"
                )

        if tool_call_inputs:
            for input_name, source_identifier in tool_call_inputs.items():
                final_resolved_params[input_name] = self._resolve_single_input_source(
                    input_name, source_identifier, plan_inputs or {}, "input field"
                )

        return final_resolved_params

    async def execute_plan(self, plan: ExecutionPlan) -> Dict[str, Any]: 
        logger.info(f"Orchestrator: Starting execution of plan ID: {plan.plan_id} - {plan.plan_description}")
        self.step_outputs: Dict[str, Dict[str, Any]] = {}

        for step_index, step in enumerate(plan.steps):
            logger.info(f"Orchestrator: Executing Step {step_index + 1}/{len(plan.steps)}: {step.step_id} - {step.description}")
            tool_calls_in_step: List[ToolCall] = []

            if isinstance(step.action, DynamicToolChainConfig) and step.action.tools:
                tool_calls_in_step = step.action.tools
            else:
                logger.warning(f"Orchestrator: Step {step.step_id} has unsupported/empty action. Skipping.")
                self.step_outputs[step.step_id] = {"error": "Unsupported or empty action"}
                continue

            current_step_outputs = {} 

            for tool_call_index, tool_call in enumerate(tool_calls_in_step):
                logger.info(f"Orchestrator: Tool {tool_call_index + 1}/{len(tool_calls_in_step)} in {step.step_id}: {tool_call.tool_id}")
                if not tool_call.tool_id: logger.error("Tool call missing tool_id. Skipping."); continue

                tool_info = self._tool_registry.get(tool_call.tool_id)
                if not tool_info: logger.error(f"Tool ID '{tool_call.tool_id}' not found. Skipping."); continue

                tool_function, pydantic_input_model_class = tool_info

                try:
                    final_tool_params = self._resolve_tool_inputs(
                        tool_call_inputs=tool_call.inputs,
                        tool_call_parameters=tool_call.parameters,
                        plan_inputs=plan.plan_inputs
                    )

                    logger.debug(f"Orchestrator: Instantiating {pydantic_input_model_class.__name__} with: {final_tool_params}")
                    current_tool_input_instance = pydantic_input_model_class(**final_tool_params) 
                    logger.info(f"Orchestrator: Instantiated {pydantic_input_model_class.__name__} for {tool_call.tool_id}")

                    logger.info(f"Orchestrator: Calling {tool_function.__name__} for ID: {tool_call.tool_id}")

                    tool_output: Any = None
                    if tool_call.tool_id.startswith("graph.Build"):
                        tool_output = await tool_function(
                            tool_input=current_tool_input_instance, main_config=self.main_config, 
                            llm_instance=self.llm, encoder_instance=self.encoder,
                            chunk_factory=self.chunk_factory
                        )
                    elif tool_call.tool_id == "corpus.PrepareFromDirectory":
                        tool_output = await tool_function(
                            tool_input=current_tool_input_instance, main_config=self.main_config 
                        )
                    else:
                        if not self.graphrag_context:
                            raise ValueError(f"GraphRAGContext is None, required by tool {tool_call.tool_id}")
                        tool_output = await tool_function(current_tool_input_instance, self.graphrag_context) 

                    logger.info(f"Orchestrator: Tool {tool_call.tool_id} executed. Output type: {type(tool_output)}")
                    logger.debug(f"Orchestrator: Raw output: {str(tool_output)[:500]}...")

                    # Register graph instances after building
                    logger.debug(f"Orchestrator: Checking if tool {tool_call.tool_id} is a graph build tool...")
                    if tool_call.tool_id.startswith("graph.Build") and tool_output is not None:
                        logger.debug(f"Orchestrator: Tool is graph build, checking attributes...")
                        logger.debug(f"Orchestrator: Has graph_id: {hasattr(tool_output, 'graph_id')}, Has status: {hasattr(tool_output, 'status')}")
                        if hasattr(tool_output, 'graph_id') and hasattr(tool_output, 'status'):
                            logger.debug(f"Orchestrator: Status={tool_output.status}, Graph ID={tool_output.graph_id}")
                            if tool_output.status == "success" and tool_output.graph_id:
                                # First check if the tool returned a graph instance
                                actual_built_graph_instance = getattr(tool_output, "graph_instance", None)
                                
                                if actual_built_graph_instance:
                                    logger.info(f"Orchestrator: Using graph instance returned by tool for '{tool_output.graph_id}'")
                                    
                                    # Set the namespace if needed
                                    if hasattr(actual_built_graph_instance._graph, 'namespace'):
                                        # Extract dataset name from graph_id
                                        dataset_name = tool_output.graph_id
                                        for suffix in ["_ERGraph", "_RKGraph", "_TreeGraphBalanced", "_TreeGraph", "_PassageGraph"]:
                                            if dataset_name.endswith(suffix):
                                                dataset_name = dataset_name[:-len(suffix)]
                                                break
                                        
                                        actual_built_graph_instance._graph.namespace = self.chunk_factory.get_namespace(dataset_name)
                                        logger.debug(f"Orchestrator: Set graph namespace to {dataset_name}")
                                    
                                    # Register the populated instance
                                    self.graphrag_context.add_graph_instance(tool_output.graph_id, actual_built_graph_instance)
                                    logger.info(f"Orchestrator: Successfully registered ACTUAL BUILT graph instance '{tool_output.graph_id}' in GraphRAGContext. Instance type: {type(actual_built_graph_instance)}")
                                else:
                                    # Fallback to the old behavior if no instance was returned
                                    logger.warning(f"Orchestrator: Graph build tool did not return a graph_instance, falling back to creating new instance")
                                    
                                    # First check if the graph already exists in context
                                    existing_graph = self.graphrag_context.get_graph_instance(tool_output.graph_id)
                                    if existing_graph:
                                        logger.info(f"Orchestrator: Graph '{tool_output.graph_id}' already exists in context, skipping recreation")
                                        graph_instance = existing_graph
                                    else:
                                        logger.info(f"Orchestrator: Creating new graph instance for '{tool_output.graph_id}'")
                                        temp_config = self.main_config.model_copy(deep=True)
                                        graph_type = None
                                        if "ERGraph" in tool_output.graph_id:
                                            graph_type = "er_graph"
                                        elif "RKGraph" in tool_output.graph_id:
                                            graph_type = "rk_graph"
                                        elif "TreeGraphBalanced" in tool_output.graph_id:
                                            graph_type = "tree_graph_balanced"
                                        elif "TreeGraph" in tool_output.graph_id:
                                            graph_type = "tree_graph"
                                        elif "PassageGraph" in tool_output.graph_id:
                                            graph_type = "passage_graph"
                                        
                                        if graph_type:
                                            temp_config.graph.type = graph_type
                                            graph_instance = get_graph(
                                                config=temp_config,
                                                llm=self.llm,
                                                encoder=self.encoder
                                            )
                                            # Set the namespace for the graph
                                            if hasattr(graph_instance._graph, 'namespace'):
                                                dataset_name = tool_output.graph_id.replace(f"_{graph_type.replace('_', '').title()}", "").replace("Graph", "").replace("_ERGraph", "")
                                                graph_instance._graph.namespace = self.chunk_factory.get_namespace(dataset_name)
                                                logger.debug(f"Orchestrator: Set graph namespace to {dataset_name}")
                                            
                                            # Try to load persisted graph if it exists
                                            try:
                                                loaded = await graph_instance.load_persisted_graph()
                                                if loaded:
                                                    logger.info(f"Orchestrator: Successfully loaded persisted graph for '{tool_output.graph_id}'")
                                                else:
                                                    logger.warning(f"Orchestrator: No persisted graph found for '{tool_output.graph_id}'")
                                            except Exception as e:
                                                logger.warning(f"Orchestrator: Could not load persisted graph for '{tool_output.graph_id}': {e}")
                                        
                                            # Register in context
                                            if not existing_graph:
                                                self.graphrag_context.add_graph_instance(tool_output.graph_id, graph_instance)
                                                logger.info(f"Orchestrator: Successfully registered graph '{tool_output.graph_id}' in GraphRAGContext")
                            else:
                                logger.debug(f"Orchestrator: Graph build not successful or no graph_id")
                        else:
                            logger.debug(f"Orchestrator: Tool output missing required attributes")
                    else:
                        logger.debug(f"Orchestrator: Not a graph build tool or no output")

                    if tool_call.named_outputs and tool_output is not None:
                        output_data_to_store = {}
                        actual_output_dict = {}
                        if hasattr(tool_output, "model_dump") and callable(tool_output.model_dump):
                            actual_output_dict = tool_output.model_dump()
                        elif isinstance(tool_output, dict):
                            actual_output_dict = tool_output

                        if actual_output_dict: 
                            for plan_key, source_key_in_model in tool_call.named_outputs.items():
                                if source_key_in_model in actual_output_dict:
                                    output_data_to_store[plan_key] = actual_output_dict[source_key_in_model]
                                else:
                                    logger.warning(f"Orchestrator: Named output source key '{source_key_in_model}' (planned as '{plan_key}') not in tool output fields {list(actual_output_dict.keys())} for {tool_call.tool_id}.")
                        elif not actual_output_dict and len(tool_call.named_outputs) == 1: 
                            plan_key = list(tool_call.named_outputs.keys())[0]
                            output_data_to_store[plan_key] = tool_output
                        else:
                            logger.warning(f"Orchestrator: Tool {tool_call.tool_id} output was not a Pydantic model or dict, and named_outputs was not singular. Output: {str(tool_output)[:100]}")

                        for key, value in output_data_to_store.items():
                            current_step_outputs[key] = value 
                            logger.info(f"Orchestrator: Stored output '{key}' for step {step.step_id} (tool {tool_call.tool_id}).")

                    elif tool_output is not None:
                         logger.info(f"Orchestrator: Tool {tool_call.tool_id} produced output, but no 'named_outputs' defined in plan. Output not stored by specific name.")

                except Exception as e:
                    logger.error(f"Orchestrator: Error during execution of tool {tool_call.tool_id} in step {step.step_id}. Exception: {str(e)}", exc_info=True)
                    current_step_outputs[tool_call.tool_id + "_error"] = str(e) 

            self.step_outputs[step.step_id] = current_step_outputs 

        logger.info(f"Orchestrator: Plan execution of all steps finished.")
        return self.step_outputs