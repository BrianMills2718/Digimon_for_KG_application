# Core/AgentOrchestrator/orchestrator.py

from typing import Dict, Any, List, Optional, Tuple, Type
from Core.AgentSchema.plan import ExecutionPlan, ExecutionStep, ToolCall, ToolInputSource
from Core.AgentSchema.context import GraphRAGContext
from Core.Common.Logger import logger
from pydantic import BaseModel

# Import tool functions (we'll add more as they become relevant)
from Core.AgentTools.entity_tools import entity_vdb_search_tool, entity_ppr_tool
from Core.AgentSchema.tool_contracts import (
    EntityVDBSearchInputs, EntityPPRInputs
    # We will add other input types here as we register more tools
)
# from Core.AgentTools.relationship_tools import ...
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
            # Add other tools here as they are implemented
            # "Relationship.OneHopNeighbors": (relationship_one_hop_neighbors_tool, RelationshipOneHopInputs), 
            # "Chunk.FromRelationships": (chunk_from_relationships_tool, ChunkFromRelationshipsInputs),
        }
        logger.info(f"AgentOrchestrator: Registered {len(registry)} tools with Pydantic models: {list(registry.keys())}")
        return registry

    async def execute_plan(self, plan: ExecutionPlan) -> Any:
        """
        Executes a given ExecutionPlan.
        """
        logger.info(f"Orchestrator: Starting execution of plan ID: {plan.plan_id} - {plan.plan_description}")
        self.step_outputs = {} # Clear previous outputs for a new plan

        final_result = None
        for step_index, step in enumerate(plan.steps):
            logger.info(f"Orchestrator: Executing Step {step_index + 1}/{len(plan.steps)}: {step.step_id} - {step.description}")
            
            if not isinstance(step.action, ToolCall):
                logger.warning(f"Orchestrator: Step {step.step_id} action is not a ToolCall. Type: {type(step.action)}. Skipping. (Only ToolCall supported initially)")
                # Later, we can add support for PredefinedMethodConfig, etc.
                continue

            tool_call: ToolCall = step.action
            tool_info = self._tool_registry.get(tool_call.tool_id)

            if not tool_info:
                logger.error(f"Orchestrator: Tool ID '{tool_call.tool_id}' not found in registry for step {step.step_id}. Skipping.")
                continue

            tool_function, pydantic_input_model_class = tool_info
            logger.info(f"Orchestrator: Preparing to call tool: {tool_call.tool_id} with input model {pydantic_input_model_class.__name__}")

            # Resolve inputs for the tool
            resolved_dynamic_inputs = await self._resolve_tool_inputs(
                tool_inputs_spec=tool_call.inputs,
                plan_inputs=plan.plan_inputs
            )

            # Combine direct parameters with resolved dynamic inputs
            final_tool_args = {}
            if tool_call.parameters:
                final_tool_args.update(tool_call.parameters)
            final_tool_args.update(resolved_dynamic_inputs)

            logger.debug(f"Orchestrator: Final resolved/merged arguments for tool {tool_call.tool_id}: {final_tool_args}")

            try:
                # Instantiate the Pydantic input model for the tool
                tool_params_instance = pydantic_input_model_class(**final_tool_args)
                logger.info(f"Orchestrator: Successfully created Pydantic input model '{pydantic_input_model_class.__name__}' for tool '{tool_call.tool_id}'.")
                logger.info(f"Orchestrator: Calling tool '{tool_call.tool_id}' with Pydantic params: {tool_params_instance.model_dump_json(indent=2)}")
                
                current_step_output = await tool_function(
                    params=tool_params_instance, 
                    graphrag_context=self.graphrag_context
                )
                logger.info(f"Orchestrator: Tool '{tool_call.tool_id}' executed. Output type: {type(current_step_output)}")
                if hasattr(current_step_output, 'model_dump_json'):
                    logger.debug(f"Orchestrator: Tool '{tool_call.tool_id}' output: {current_step_output.model_dump_json(indent=2)}")
                else:
                    logger.debug(f"Orchestrator: Tool '{tool_call.tool_id}' output: {current_step_output}")

                if tool_call.named_outputs:
                    if len(tool_call.named_outputs) == 1:
                        # If only one named output, store the direct result
                        output_key = list(tool_call.named_outputs.keys())[0]
                        if step.step_id not in self.step_outputs:
                            self.step_outputs[step.step_id] = {}
                        self.step_outputs[step.step_id][output_key] = current_step_output
                        logger.info(f"Orchestrator: Stored output for step {step.step_id}, key '{output_key}'")
                    else:
                        # If multiple named outputs, the tool should return a dict matching these keys
                        if isinstance(current_step_output, dict):
                            if step.step_id not in self.step_outputs:
                                self.step_outputs[step.step_id] = {}
                            for output_key in tool_call.named_outputs.keys():
                                if output_key in current_step_output:
                                    self.step_outputs[step.step_id][output_key] = current_step_output[output_key]
                                    logger.info(f"Orchestrator: Stored output for step {step.step_id}, key '{output_key}'")
                                else:
                                    logger.warning(f"Orchestrator: Output key '{output_key}' named in plan not found in result from {tool_call.tool_id}")
                        else:
                            logger.error(f"Orchestrator: Tool {tool_call.tool_id} was expected to return a dict for multiple named_outputs, but returned {type(current_step_output)}")
                
                final_result = current_step_output # Keep track of the last step's output as the plan's result for now

            except Exception as e:
                logger.error(f"Orchestrator: Error executing tool {tool_call.tool_id} in step {step.step_id}: {e}", exc_info=True)
                # Optionally, decide if the plan should halt on error
                # return {"error": str(e), "step_failed": step.step_id}
        
        logger.info(f"Orchestrator: Plan execution finished. Final result: {final_result}")
        return final_result

    async def _resolve_tool_inputs(
        self, 
        tool_inputs_spec: Optional[Dict[str, Any]], 
        plan_inputs: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Resolves the actual values for a tool's inputs based on the plan's specification.
        Handles fetching from plan_inputs and previous step_outputs.
        """
        resolved_args: Dict[str, Any] = {}
        if not tool_inputs_spec:
            return resolved_args

        # Process dynamic inputs from tool_call.inputs
        for input_name, input_spec in tool_inputs_spec.items():
            if isinstance(input_spec, ToolInputSource):
                source_step_id = input_spec.from_step_id
                source_output_key = input_spec.named_output_key
                if source_step_id in self.step_outputs and \
                   source_output_key in self.step_outputs[source_step_id]:
                    resolved_args[input_name] = self.step_outputs[source_step_id][source_output_key]
                    logger.debug(f"Orchestrator: Resolved input '{input_name}' from step '{source_step_id}', output_key '{source_output_key}'")
                else:
                    logger.error(f"Orchestrator: Could not resolve input '{input_name}'. Output not found for step_id '{source_step_id}' with key '{source_output_key}'.")
                    resolved_args[input_name] = None 
            elif isinstance(input_spec, str) and input_spec.startswith("plan_inputs."):
                key_path = input_spec.split("plan_inputs.")[1]
                if plan_inputs and key_path in plan_inputs:
                    resolved_args[input_name] = plan_inputs[key_path]
                    logger.debug(f"Orchestrator: Resolved input '{input_name}' from plan_inputs, key '{key_path}'")
                else:
                    logger.error(f"Orchestrator: Could not resolve input '{input_name}'. Key '{key_path}' not found in plan_inputs.")
                    resolved_args[input_name] = None
            else:  # Literal value
                resolved_args[input_name] = input_spec
                logger.debug(f"Orchestrator: Using literal value for input '{input_name}'")
        
        return resolved_args
