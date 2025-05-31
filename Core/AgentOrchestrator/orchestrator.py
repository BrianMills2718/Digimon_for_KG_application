# Core/AgentOrchestrator/orchestrator.py

from typing import Dict, Any, List, Optional, Tuple, Type
from Core.AgentSchema.plan import ExecutionPlan, ExecutionStep, ToolCall, ToolInputSource
from Core.AgentSchema.tool_contracts import EntityVDBSearchOutputs, EntityVDBSearchInputs, EntityPPRInputs, RelationshipOneHopNeighborsInputs, VDBSearchResultItem
from Core.AgentSchema.context import GraphRAGContext
from Core.Common.Logger import logger
from pydantic import BaseModel

# Import tool functions (we'll add more as they become relevant)
from Core.AgentTools.entity_tools import entity_vdb_search_tool, entity_ppr_tool
from Core.AgentTools.relationship_tools import relationship_one_hop_neighbors_tool
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
                        plan.plan_inputs,
                        self.step_outputs
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

    def _resolve_tool_inputs(
        self,
        tool_inputs_spec: Optional[Dict[str, Any]],
        plan_inputs: Optional[Dict[str, Any]],
        all_step_outputs: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        resolved_inputs_dict: Dict[str, Any] = {}
        if not tool_inputs_spec:
            return resolved_inputs_dict

        for input_name, input_source_val in tool_inputs_spec.items():
            if isinstance(input_source_val, str) and input_source_val.startswith("plan_inputs."):
                plan_input_key = input_source_val.split("plan_inputs.", 1)[1]
                if plan_inputs and plan_input_key in plan_inputs:
                    resolved_inputs_dict[input_name] = plan_inputs[plan_input_key]
                    logger.debug(f"Orchestrator: Resolved input '{input_name}' from plan_inputs, key '{plan_input_key}'")
                else:
                    logger.warning(f"Orchestrator: Plan input key '{plan_input_key}' for tool input '{input_name}' not found in plan_inputs. Skipping this input.")
            
            elif isinstance(input_source_val, ToolInputSource):
                source_step_id = input_source_val.from_step_id
                named_output_key = input_source_val.named_output_key

                if source_step_id not in all_step_outputs or named_output_key not in all_step_outputs[source_step_id]:
                    logger.error(f"Orchestrator: Could not find source output for step '{source_step_id}', key '{named_output_key}' for input '{input_name}'")
                    # Decide on error handling: skip, use default, or raise. For now, skipping.
                    continue 
            
                source_value_container = all_step_outputs[source_step_id][named_output_key]

                # Check if the container is an EntityVDBSearchOutputs object
                if isinstance(source_value_container, EntityVDBSearchOutputs):
                    # Check if the target input name is one that expects a list of entity names
                    if input_name in ["seed_entity_ids", "entity_ids"]:
                        transformed_value = []
                        if hasattr(source_value_container, 'similar_entities') and isinstance(source_value_container.similar_entities, list):
                            for item in source_value_container.similar_entities:
                                if isinstance(item, VDBSearchResultItem) and hasattr(item, 'entity_name'):
                                    transformed_value.append(item.entity_name)
                                else:
                                    logger.warning(f"Orchestrator: Item in similar_entities from step '{source_step_id}' is not a VDBSearchResultItem with entity_name. Item: {item}")
                    
                        resolved_inputs_dict[input_name] = transformed_value
                        logger.info(f"Orchestrator: Transformed 'similar_entities' from step '{source_step_id}' "
                                     f"into List[str] of entity names for input '{input_name}'. Extracted {len(transformed_value)} IDs.")
                    else:
                        # If it's EntityVDBSearchOutputs but not for seed/entity_ids, pass it as is (or handle other specific transformations)
                        resolved_inputs_dict[input_name] = source_value_container
                        logger.debug(f"Orchestrator: Using EntityVDBSearchOutputs object directly for input '{input_name}' from step '{source_step_id}'.")
                # Add more specific transformations here if needed for other input_names or source_value types
                else:
                    # Default: use the source value directly if no specific transformation is defined
                    resolved_inputs_dict[input_name] = source_value_container
                    logger.debug(f"Orchestrator: Resolved input '{input_name}' from step '{source_step_id}', output_key '{named_output_key}' by direct assignment.")
            else:
                resolved_inputs_dict[input_name] = input_source_val
                logger.debug(f"Orchestrator: Resolved input '{input_name}' as literal value.")
        return resolved_inputs_dict
