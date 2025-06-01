# Core/AgentOrchestrator/orchestrator.py

from typing import Dict, Any, List, Optional, Tuple, Type, Union
from Core.AgentSchema.plan import ExecutionPlan, ExecutionStep, ToolCall, ToolInputSource, DynamicToolChainConfig
from Core.AgentSchema.tool_contracts import EntityVDBSearchOutputs, EntityVDBSearchInputs, EntityPPRInputs, RelationshipOneHopNeighborsInputs, VDBSearchResultItem
from Core.AgentSchema.context import GraphRAGContext
from Core.Common.Logger import logger
from pydantic import BaseModel

# Import tool functions (we'll add more as they become relevant)
from Core.AgentTools.entity_tools import entity_vdb_search_tool, entity_ppr_tool
from Core.AgentTools.relationship_tools import relationship_one_hop_neighbors_tool
from Core.AgentTools.graph_construction_tools import (
    build_er_graph, build_rk_graph, build_tree_graph, build_tree_graph_balanced, build_passage_graph
)
from Core.AgentTools.corpus_tools import prepare_corpus_from_directory
# from Core.AgentTools.chunk_tools import ...
# from Core.AgentTools.subgraph_tools import ...
# from Core.AgentTools.community_tools import ...

class AgentOrchestrator:
    def __init__(self, graphrag_context: GraphRAGContext):
        self.graphrag_context = graphrag_context
        self._tool_registry = self._register_tools()
        self.step_outputs: Dict[str, Dict[str, Any]] = {} # To store outputs of each step

    def _register_tools(self) -> Dict[str, Tuple[callable, Type[BaseModel]]]:
        """
        Registry mapping tool_id strings to (callable_function, pydantic_input_model_class) tuples.
        """
        registry = {
            "Entity.VDBSearch": (entity_vdb_search_tool, EntityVDBSearchInputs),
            "Entity.PPR": (entity_ppr_tool, EntityPPRInputs),
            "Relationship.OneHopNeighbors": (relationship_one_hop_neighbors_tool, RelationshipOneHopNeighborsInputs),
            # Add other tools here as they are implemented
            # "Chunk.FromRelationships": (chunk_from_relationships_tool, ChunkFromRelationshipsInputs),
            "graph.BuildERGraph": (build_er_graph, None),
            "graph.BuildRKGraph": (build_rk_graph, None),
            "graph.BuildTreeGraph": (build_tree_graph, None),
            "graph.BuildTreeGraphBalanced": (build_tree_graph_balanced, None),
            "graph.BuildPassageGraph": (build_passage_graph, None),
            "corpus.PrepareFromDirectory": (prepare_corpus_from_directory, None),
        }
        logger.info(f"AgentOrchestrator: Registered {len(registry)} tools with Pydantic models: {list(registry.keys())}")
        return registry

    async def execute_plan(self, plan: ExecutionPlan) -> Optional[Dict[str, Any]]:
        logger.info(f"Orchestrator: Starting execution of plan ID: {plan.plan_id} - {plan.plan_description}")
        self.step_outputs: Dict[str, Dict[str, Any]] = {} # Reset for each plan execution

        for step_index, step in enumerate(plan.steps):
            logger.info(f"Orchestrator: Executing Step {step_index + 1}/{len(plan.steps)}: {step.step_id} - {step.description}")

            tool_calls_in_step: List[Any] = []

            if hasattr(step.action, "tools"):
                logger.info(f"Orchestrator: Step {step.step_id} is a DynamicToolChainConfig with {len(step.action.tools)} tool(s).")
                tool_calls_in_step = step.action.tools
            elif hasattr(step.action, "method_yaml_name"):
                logger.warning(f"Orchestrator: Step {step.step_id} action is PredefinedMethodConfig. Logic not yet implemented. Skipping.")
                continue
            else:
                logger.warning(f"Orchestrator: Step {step.step_id} has an unsupported action type: {type(step.action)}. Skipping.")
                continue

            if not tool_calls_in_step:
                logger.warning(f"Orchestrator: No tools to execute for step {step.step_id}. Skipping.")
                continue

            # Initialize step output storage for this step
            self.step_outputs[step.step_id] = {}

            for tool_call_index, tool_call in enumerate(tool_calls_in_step):
                logger.info(f"Orchestrator: Executing tool {tool_call_index + 1}/{len(tool_calls_in_step)} in step {step.step_id}: {tool_call.tool_id}")

                tool_info = self._tool_registry.get(tool_call.tool_id)
                if not tool_info:
                    logger.error(f"Orchestrator: Tool ID '{tool_call.tool_id}' not found in registry. Skipping tool.")
                    continue

                tool_function, pydantic_input_model_class = tool_info

                try:
                    # 1. Resolve inputs for the current tool_call
                    resolved_inputs_dict = self._resolve_tool_inputs(
                        tool_call.inputs,
                        plan.plan_inputs
                    )
                    if hasattr(resolved_inputs_dict, "__await__"):
                        resolved_inputs_dict = await resolved_inputs_dict

                    # 2. Merge with direct parameters
                    final_tool_params = {**(tool_call.parameters or {}), **resolved_inputs_dict}

                    # 3. Instantiate the tool's specific Pydantic input model
                    logger.debug(f"Orchestrator: Attempting to instantiate {pydantic_input_model_class.__name__} with params: {final_tool_params}")
                    tool_input_instance = pydantic_input_model_class(**final_tool_params)
                    logger.info(f"Orchestrator: Successfully instantiated {pydantic_input_model_class.__name__} for tool {tool_call.tool_id}")

                    # 4. Execute the tool function
                    logger.info(f"Orchestrator: Calling tool function: {tool_function.__name__} for tool ID: {tool_call.tool_id}")
                    tool_output = await tool_function(tool_input_instance, self.graphrag_context)
                    logger.info(f"Orchestrator: Tool {tool_call.tool_id} executed. Output type: {type(tool_output)}")
                    logger.debug(f"Orchestrator: Raw output from tool {tool_call.tool_id}: {tool_output}")

                    # 5. Store named outputs for this tool_call within the current step's outputs
                    if tool_call.named_outputs and tool_output:
                        if hasattr(tool_output, "model_dump") and callable(tool_output.model_dump): # If tool returns a Pydantic model
                            output_dict = tool_output.model_dump()
                            for output_name_in_plan, _ in tool_call.named_outputs.items():
                                if output_name_in_plan in output_dict:
                                    self.step_outputs[step.step_id][output_name_in_plan] = output_dict[output_name_in_plan]
                                    logger.info(f"Orchestrator: Stored output '{output_name_in_plan}' for step {step.step_id} from tool {tool_call.tool_id}.")
                                elif len(tool_call.named_outputs) == 1:
                                    first_output_name = list(tool_call.named_outputs.keys())[0]
                                    if hasattr(tool_output, first_output_name):
                                        self.step_outputs[step.step_id][first_output_name] = getattr(tool_output, first_output_name)
                                    else:
                                        self.step_outputs[step.step_id][first_output_name] = tool_output
                                    logger.info(f"Orchestrator: Stored output '{first_output_name}' for step {step.step_id} (tool {tool_call.tool_id}) as Pydantic model or its primary attribute.")
                        elif isinstance(tool_output, dict):
                            for output_name_in_plan, _ in tool_call.named_outputs.items():
                                if output_name_in_plan in tool_output:
                                    self.step_outputs[step.step_id][output_name_in_plan] = tool_output[output_name_in_plan]
                                    logger.info(f"Orchestrator: Stored output '{output_name_in_plan}' for step {step.step_id} from tool {tool_call.tool_id}.")
                        else:
                            if len(tool_call.named_outputs) == 1:
                                output_name_in_plan = list(tool_call.named_outputs.keys())[0]
                                self.step_outputs[step.step_id][output_name_in_plan] = tool_output
                                logger.info(f"Orchestrator: Stored single output '{output_name_in_plan}' for step {step.step_id} from tool {tool_call.tool_id}.")
                            else:
                                logger.warning(f"Orchestrator: Tool {tool_call.tool_id} returned a single value but multiple named_outputs were specified. Cannot map. Output: {tool_output}")
                    elif tool_output:
                        logger.info(f"Orchestrator: Tool {tool_call.tool_id} produced output, but no 'named_outputs' were specified in the plan for this tool call. Output not stored in step_outputs by name.")
                except Exception as e:
                    logger.error(f"Orchestrator: Error executing tool {tool_call.tool_id} in step {step.step_id}: {e}", exc_info=True)

        final_step_id_for_result = plan.steps[-1].step_id if plan.steps else ""
        final_result = self.step_outputs.get(final_step_id_for_result, None)
        logger.info(f"Orchestrator: Plan execution finished. Final result from step '{final_step_id_for_result}': {final_result}")
        return final_result

    # START (REPLACE _resolve_tool_inputs method in AgentOrchestrator)
    def _resolve_tool_inputs(
        self,
        step_inputs_config: Dict[str, Union[Any, "ToolInputSource", Dict[str, Any]]],  # Allow dict for TIS
        plan_inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Resolves tool inputs from plan_inputs or previous step_outputs.
        Handles direct values, references to plan_inputs, and ToolInputSource objects/dicts.
        Also performs specific transformations (e.g., VDB output to list of entity IDs).
        """
        resolved_params = {}
        for input_name, source_identifier in step_inputs_config.items():
            source_value = None

            # Check if source_identifier is a dictionary representing a ToolInputSource
            current_tis = None
            if isinstance(source_identifier, dict) and \
            "from_step_id" in source_identifier and \
            "named_output_key" in source_identifier:
                try:
                    current_tis = ToolInputSource(**source_identifier)
                    logger.debug(f"Orchestrator: Parsed dict into ToolInputSource for input '{input_name}'.")
                except Exception as e_tis:
                    logger.error(f"Orchestrator: Failed to parse dict as ToolInputSource for input '{input_name}': {source_identifier}. Error: {e_tis}")
                    # Treat as literal if parsing fails, though this is unlikely to be correct.
                    source_value = source_identifier
            elif isinstance(source_identifier, ToolInputSource):
                current_tis = source_identifier

            if current_tis:  # Process if it's a ToolInputSource object
                from_step_id = current_tis.from_step_id
                named_output_key = current_tis.named_output_key

                if from_step_id in self.step_outputs and \
                named_output_key in self.step_outputs[from_step_id]:

                    raw_source_value = self.step_outputs[from_step_id][named_output_key]
                    logger.debug(f"Orchestrator: Retrieving input '{input_name}' from step '{from_step_id}', output_key '{named_output_key}'. Raw value type: {type(raw_source_value)}")

                    # --- Specific Data Transformations ---
                    # 1. For 'seed_entity_ids' from 'EntityVDBSearchOutputs'
                    if input_name == "seed_entity_ids" and \
                    hasattr(raw_source_value, 'similar_entities') and \
                    isinstance(raw_source_value.similar_entities, list):

                        # raw_source_value is likely EntityVDBSearchOutputs
                        # similar_entities is List[VDBSearchResultItem] or List[Tuple[str, float]]
                        extracted_ids = []
                        for item in raw_source_value.similar_entities:
                            if hasattr(item, 'entity_name'):  # VDBSearchResultItem case
                                extracted_ids.append(item.entity_name)
                            elif isinstance(item, tuple) and len(item) > 0 and isinstance(item[0], str):  # Tuple case
                                extracted_ids.append(item[0])
                        source_value = extracted_ids
                        logger.info(f"Orchestrator: Transformed 'similar_entities' from step '{from_step_id}' into List[str] of entity names for input '{input_name}'. Extracted {len(source_value)} IDs.")

                    # 2. For 'entity_ids' (e.g., for Relationship.OneHopNeighbors) from 'EntityVDBSearchOutputs'
                    elif input_name == "entity_ids" and \
                        hasattr(raw_source_value, 'similar_entities') and \
                        isinstance(raw_source_value.similar_entities, list):
                        extracted_ids = []
                        for item in raw_source_value.similar_entities:
                            if hasattr(item, 'entity_name'):
                                extracted_ids.append(item.entity_name)
                            elif isinstance(item, tuple) and len(item) > 0 and isinstance(item[0], str):
                                extracted_ids.append(item[0])
                        source_value = extracted_ids
                        logger.info(f"Orchestrator: Transformed 'similar_entities' from step '{from_step_id}' into List[str] of entity names for input '{input_name}'. Extracted {len(source_value)} IDs.")

                    # Add other transformations here if needed

                    else:
                        # Default: use the raw output value if no specific transformation applies
                        source_value = raw_source_value
                        logger.debug(f"Orchestrator: Using raw output from previous step for '{input_name}'.")
                else:
                    logger.error(f"Orchestrator: Could not find output for key '{named_output_key}' in step '{from_step_id}' for input '{input_name}'. Available outputs for step: {self.step_outputs.get(from_step_id, {}).keys()}")
                    source_value = None  # Or raise an error

            elif isinstance(source_identifier, str) and source_identifier.startswith("plan_inputs."):
                input_key = source_identifier.split("plan_inputs.")[1]
                if input_key in plan_inputs:
                    source_value = plan_inputs[input_key]
                    logger.debug(f"Orchestrator: Resolved input '{input_name}' from plan_inputs, key '{input_key}'")
                else:
                    logger.error(f"Orchestrator: Input key '{input_key}' not found in plan_inputs for '{input_name}'. Plan inputs: {plan_inputs.keys()}")
                    source_value = None  # Or raise

            else:  # Literal value (or unparsed dict that wasn't a TIS)
                source_value = source_identifier
                logger.debug(f"Orchestrator: Resolved input '{input_name}' as literal value.")

            resolved_params[input_name] = source_value
        return resolved_params
    # END (REPLACE _resolve_tool_inputs method in AgentOrchestrator)