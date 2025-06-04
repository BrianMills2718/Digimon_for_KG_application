# Core/AgentOrchestrator/async_streaming_orchestrator_v2.py

from typing import Dict, Any, List, Optional, Tuple, Type, Union, AsyncGenerator
from dataclasses import dataclass
from datetime import datetime
import asyncio
from enum import Enum

from Core.AgentSchema.plan import ExecutionPlan, ExecutionStep, ToolCall, ToolInputSource, DynamicToolChainConfig
from Core.AgentSchema.tool_contracts import *
from Core.AgentSchema.context import GraphRAGContext
from Core.Common.Logger import logger
from pydantic import BaseModel
from Option.Config2 import Config
from Core.Provider.BaseLLM import BaseLLM
from Core.Provider.BaseEmb import BaseEmb as LlamaIndexBaseEmbedding
from Core.Chunk.ChunkFactory import ChunkFactory

# Import the dynamic tool registry
from Core.AgentTools.tool_registry import global_tool_registry, ToolCategory

class UpdateType(Enum):
    """Types of updates that can be streamed during execution"""
    PLAN_START = "plan_start"
    STEP_START = "step_start"
    STEP_PROGRESS = "step_progress"
    STEP_COMPLETE = "step_complete"
    STEP_ERROR = "step_error"
    TOOL_START = "tool_start"
    TOOL_PROGRESS = "tool_progress"
    TOOL_COMPLETE = "tool_complete"
    TOOL_ERROR = "tool_error"
    PLAN_COMPLETE = "plan_complete"
    PLAN_ERROR = "plan_error"

@dataclass
class StreamingUpdate:
    """Update object streamed during execution"""
    type: UpdateType
    timestamp: datetime
    step_id: Optional[str] = None
    tool_id: Optional[str] = None
    description: Optional[str] = None
    progress: Optional[float] = None
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class AsyncStreamingOrchestrator:
    """
    Enhanced async streaming orchestrator that uses dynamic tool registry
    for improved tool discovery and parallel execution optimization.
    """
    
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
        self.tool_registry = global_tool_registry
        self.step_outputs: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"AsyncStreamingOrchestrator V2: Using dynamic tool registry with {len(self.tool_registry)} tools")
    
    def _resolve_single_input_source(
        self,
        target_input_name: str, 
        source_identifier: Any, 
        plan_inputs: Dict[str, Any], 
        source_location_type: str 
    ) -> Any:
        """Resolve input sources (same as original orchestrator)"""
        source_value = None
        current_tis = None
        plan_inputs = plan_inputs or {}

        if isinstance(source_identifier, dict) and \
           "from_step_id" in source_identifier and \
           "named_output_key" in source_identifier:
            try:
                current_tis = ToolInputSource(**source_identifier)
                logger.debug(f"AsyncOrchestrator ({source_location_type}) '{target_input_name}': Parsed dict into ToolInputSource.")
            except Exception as e_tis:
                logger.error(f"AsyncOrchestrator ({source_location_type}) '{target_input_name}': Failed to parse ToolInputSource dict {source_identifier}. Error: {e_tis}")
                source_value = source_identifier
        elif isinstance(source_identifier, ToolInputSource):
            current_tis = source_identifier

        if current_tis:
            from_step_id = current_tis.from_step_id
            named_output_key = current_tis.named_output_key
            if from_step_id in self.step_outputs and \
               named_output_key in self.step_outputs[from_step_id]:
                raw_source_value = self.step_outputs[from_step_id][named_output_key]
                
                # Handle entity ID extraction for graph operations
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
                            if 'entity_name' in item:
                                extracted_ids.append(item['entity_name'])
                            elif 'node_id' in item:
                                extracted_ids.append(item['node_id'])
                        elif isinstance(item, tuple) and len(item) > 0 and isinstance(item[0], str): 
                            extracted_ids.append(item[0])
                    source_value = extracted_ids
                else: 
                    source_value = raw_source_value
            else:
                logger.error(f"AsyncOrchestrator ({source_location_type}) '{target_input_name}': Output key '{named_output_key}' not in step '{from_step_id}'")
                source_value = None 
        elif isinstance(source_identifier, str) and source_identifier.startswith("plan_inputs."):
            input_key = source_identifier.split("plan_inputs.")[1]
            if input_key in plan_inputs:
                source_value = plan_inputs[input_key]
            else:
                logger.error(f"AsyncOrchestrator ({source_location_type}) '{target_input_name}': Key '{input_key}' not in plan_inputs")
                source_value = None
        else: 
            source_value = source_identifier
            
        return source_value
    
    def _resolve_tool_inputs(
        self,
        tool_call_inputs: Optional[Dict[str, Any]],
        tool_call_parameters: Optional[Dict[str, Any]],
        plan_inputs: Optional[Dict[str, Any]] 
    ) -> Dict[str, Any]:
        """Resolve all tool inputs"""
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
    
    async def _execute_tool_async(
        self, 
        tool_call: ToolCall, 
        plan_inputs: Dict[str, Any]
    ) -> Tuple[Optional[Any], Optional[str]]:
        """Execute a single tool asynchronously"""
        if not tool_call.tool_id:
            return None, "Tool call missing tool_id"
        
        # Get tool from registry
        tool_registration = self.tool_registry.get_tool(tool_call.tool_id)
        if not tool_registration:
            return None, f"Tool ID '{tool_call.tool_id}' not found in registry"
        
        tool_function = tool_registration.function
        tool_metadata = tool_registration.metadata
        pydantic_input_model_class = tool_metadata.input_model
        
        try:
            # Resolve inputs
            final_tool_params = self._resolve_tool_inputs(
                tool_call_inputs=tool_call.inputs,
                tool_call_parameters=tool_call.parameters,
                plan_inputs=plan_inputs
            )
            
            # Apply pre-processor if available
            if tool_registration.pre_processor:
                final_tool_params = await tool_registration.pre_processor(final_tool_params)
            
            # Create input instance
            current_tool_input_instance = pydantic_input_model_class(**final_tool_params)
            
            # Validate if validator provided
            if tool_registration.validator:
                validation_result = await tool_registration.validator(current_tool_input_instance)
                if not validation_result:
                    return None, f"Tool input validation failed for {tool_call.tool_id}"
            
            # Execute tool based on type
            tool_output: Any = None
            if tool_call.tool_id.startswith("graph.Build"):
                tool_output = await tool_function(
                    tool_input=current_tool_input_instance, 
                    main_config=self.main_config, 
                    llm_instance=self.llm, 
                    encoder_instance=self.encoder,
                    chunk_factory=self.chunk_factory
                )
            elif tool_call.tool_id == "corpus.PrepareFromDirectory":
                tool_output = await tool_function(
                    tool_input=current_tool_input_instance, 
                    main_config=self.main_config 
                )
            else:
                if not self.graphrag_context:
                    raise ValueError(f"GraphRAGContext is None, required by tool {tool_call.tool_id}")
                tool_output = await tool_function(current_tool_input_instance, self.graphrag_context)
            
            # Apply post-processor if available
            if tool_registration.post_processor:
                tool_output = await tool_registration.post_processor(tool_output)
                
            return tool_output, None
            
        except Exception as e:
            logger.error(f"Error executing tool {tool_call.tool_id}: {str(e)}", exc_info=True)
            return None, str(e)
    
    async def execute_plan_stream(self, plan: ExecutionPlan) -> AsyncGenerator[StreamingUpdate, None]:
        """
        Execute a plan and stream updates in real-time.
        Uses dynamic tool registry for optimized parallel execution.
        """
        # Yield plan start
        yield StreamingUpdate(
            type=UpdateType.PLAN_START,
            timestamp=datetime.utcnow(),
            description=f"Starting execution of plan: {plan.plan_description}",
            data={"plan_id": plan.plan_id, "total_steps": len(plan.steps)}
        )
        
        self.step_outputs = {}
        
        try:
            # Execute each step
            for step_index, step in enumerate(plan.steps):
                # Yield step start
                yield StreamingUpdate(
                    type=UpdateType.STEP_START,
                    timestamp=datetime.utcnow(),
                    step_id=step.step_id,
                    description=step.description,
                    progress=step_index / len(plan.steps),
                    data={"step_index": step_index + 1, "total_steps": len(plan.steps)}
                )
                
                # Extract tool calls from step
                tool_calls_in_step: List[ToolCall] = []
                if isinstance(step.action, DynamicToolChainConfig) and step.action.tools:
                    tool_calls_in_step = step.action.tools
                else:
                    yield StreamingUpdate(
                        type=UpdateType.STEP_ERROR,
                        timestamp=datetime.utcnow(),
                        step_id=step.step_id,
                        error="Step has unsupported or empty action"
                    )
                    self.step_outputs[step.step_id] = {"error": "Unsupported or empty action"}
                    continue
                
                current_step_outputs = {}
                
                # Use registry to separate parallelizable tools
                tool_ids = [tc.tool_id for tc in tool_calls_in_step]
                parallel_tool_ids, sequential_tool_ids = self.tool_registry.get_parallelizable_tools(tool_ids)
                
                # Group tool calls by parallelizability
                parallel_tools = [tc for tc in tool_calls_in_step if tc.tool_id in parallel_tool_ids]
                sequential_tools = [tc for tc in tool_calls_in_step if tc.tool_id in sequential_tool_ids]
                
                # Execute parallel tools
                if parallel_tools:
                    yield StreamingUpdate(
                        type=UpdateType.STEP_PROGRESS,
                        timestamp=datetime.utcnow(),
                        step_id=step.step_id,
                        description=f"Executing {len(parallel_tools)} parallelizable tools"
                    )
                    
                    # Create tasks for parallel execution
                    tasks = []
                    for tool_call in parallel_tools:
                        tasks.append(self._execute_tool_with_updates(
                            tool_call, plan.plan_inputs, step.step_id
                        ))
                    
                    # Execute in parallel and collect results
                    async for update in self._merge_async_generators(tasks):
                        yield update
                        # Store outputs as they complete
                        if update.type == UpdateType.TOOL_COMPLETE and update.data:
                            self._store_tool_outputs(
                                tool_call=next(tc for tc in parallel_tools if tc.tool_id == update.tool_id),
                                tool_output=update.data.get("output"),
                                current_step_outputs=current_step_outputs
                            )
                
                # Execute sequential tools
                if sequential_tools:
                    yield StreamingUpdate(
                        type=UpdateType.STEP_PROGRESS,
                        timestamp=datetime.utcnow(),
                        step_id=step.step_id,
                        description=f"Executing {len(sequential_tools)} sequential tools"
                    )
                    
                    for tool_call in sequential_tools:
                        async for update in self._execute_tool_with_updates(
                            tool_call, plan.plan_inputs, step.step_id
                        ):
                            yield update
                            # Store outputs
                            if update.type == UpdateType.TOOL_COMPLETE and update.data:
                                self._store_tool_outputs(
                                    tool_call=tool_call,
                                    tool_output=update.data.get("output"),
                                    current_step_outputs=current_step_outputs
                                )
                                
                                # Handle graph registration
                                if tool_call.tool_id.startswith("graph.Build"):
                                    await self._handle_graph_registration(
                                        tool_call.tool_id,
                                        update.data.get("output")
                                    )
                
                # Store step outputs
                self.step_outputs[step.step_id] = current_step_outputs
                
                # Yield step complete
                yield StreamingUpdate(
                    type=UpdateType.STEP_COMPLETE,
                    timestamp=datetime.utcnow(),
                    step_id=step.step_id,
                    description=f"Completed step: {step.description}",
                    progress=(step_index + 1) / len(plan.steps),
                    data={"outputs": list(current_step_outputs.keys())}
                )
            
            # Yield plan complete
            yield StreamingUpdate(
                type=UpdateType.PLAN_COMPLETE,
                timestamp=datetime.utcnow(),
                description="Plan execution completed successfully",
                data={"total_outputs": len(self.step_outputs)}
            )
            
        except Exception as e:
            # Yield plan error
            yield StreamingUpdate(
                type=UpdateType.PLAN_ERROR,
                timestamp=datetime.utcnow(),
                error=f"Plan execution failed: {str(e)}",
                data={"exception_type": type(e).__name__}
            )
            raise
    
    async def _execute_tool_with_updates(
        self, 
        tool_call: ToolCall, 
        plan_inputs: Dict[str, Any],
        step_id: str
    ) -> AsyncGenerator[StreamingUpdate, None]:
        """Execute a single tool and yield updates"""
        # Get tool metadata for enhanced updates
        tool_metadata = self.tool_registry.get_tool_metadata(tool_call.tool_id)
        
        # Yield tool start with metadata
        yield StreamingUpdate(
            type=UpdateType.TOOL_START,
            timestamp=datetime.utcnow(),
            step_id=step_id,
            tool_id=tool_call.tool_id,
            description=f"Starting {tool_metadata.name if tool_metadata else tool_call.tool_id}",
            data={
                "category": tool_metadata.category.value if tool_metadata else None,
                "capabilities": [cap.value for cap in tool_metadata.capabilities] if tool_metadata else None
            }
        )
        
        # Execute tool
        tool_output, error = await self._execute_tool_async(tool_call, plan_inputs)
        
        if error:
            # Yield tool error
            yield StreamingUpdate(
                type=UpdateType.TOOL_ERROR,
                timestamp=datetime.utcnow(),
                step_id=step_id,
                tool_id=tool_call.tool_id,
                error=error
            )
        else:
            # Yield tool complete
            yield StreamingUpdate(
                type=UpdateType.TOOL_COMPLETE,
                timestamp=datetime.utcnow(),
                step_id=step_id,
                tool_id=tool_call.tool_id,
                description=f"Completed {tool_metadata.name if tool_metadata else tool_call.tool_id}",
                data={"output": tool_output}
            )
    
    async def _merge_async_generators(
        self, 
        generators: List[AsyncGenerator[StreamingUpdate, None]]
    ) -> AsyncGenerator[StreamingUpdate, None]:
        """Merge multiple async generators into a single stream"""
        tasks = [asyncio.create_task(self._consume_generator(gen)) for gen in generators]
        
        while tasks:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            
            for task in done:
                result = await task
                if result is not None:
                    update, gen = result
                    yield update
                    # Continue consuming from this generator
                    new_task = asyncio.create_task(self._consume_generator(gen))
                    tasks.append(new_task)
                
                tasks.remove(task)
    
    async def _consume_generator(
        self, 
        gen: AsyncGenerator[StreamingUpdate, None]
    ) -> Optional[Tuple[StreamingUpdate, AsyncGenerator[StreamingUpdate, None]]]:
        """Consume one item from a generator"""
        try:
            update = await gen.__anext__()
            return (update, gen)
        except StopAsyncIteration:
            return None
    
    def _store_tool_outputs(
        self,
        tool_call: ToolCall,
        tool_output: Any,
        current_step_outputs: Dict[str, Any]
    ) -> None:
        """Store tool outputs in the step outputs dictionary"""
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
            elif not actual_output_dict and len(tool_call.named_outputs) == 1: 
                plan_key = list(tool_call.named_outputs.keys())[0]
                output_data_to_store[plan_key] = tool_output

            for key, value in output_data_to_store.items():
                current_step_outputs[key] = value
                
        elif tool_output is not None:
            # Store the entire output when no named_outputs defined
            if isinstance(tool_output, dict):
                for key, value in tool_output.items():
                    current_step_outputs[key] = value
            else:
                current_step_outputs[tool_call.tool_id] = tool_output
    
    async def _handle_graph_registration(self, tool_id: str, tool_output: Any) -> None:
        """Handle graph instance registration after building"""
        if tool_output is None:
            return
            
        if hasattr(tool_output, 'graph_id') and hasattr(tool_output, 'status'):
            if tool_output.status == "success" and tool_output.graph_id:
                # Check if tool returned a graph instance
                actual_built_graph_instance = getattr(tool_output, "graph_instance", None)
                
                if actual_built_graph_instance and self.graphrag_context:
                    # Set namespace if needed
                    if hasattr(actual_built_graph_instance._graph, 'namespace'):
                        dataset_name = tool_output.graph_id
                        for suffix in ["_ERGraph", "_RKGraph", "_TreeGraphBalanced", "_TreeGraph", "_PassageGraph"]:
                            if dataset_name.endswith(suffix):
                                dataset_name = dataset_name[:-len(suffix)]
                                break
                        
                        actual_built_graph_instance._graph.namespace = self.chunk_factory.get_namespace(dataset_name)
                    
                    # Register the graph instance
                    self.graphrag_context.add_graph_instance(tool_output.graph_id, actual_built_graph_instance)
                    logger.info(f"AsyncOrchestrator: Registered graph instance '{tool_output.graph_id}'")
    
    # Backward compatibility method
    async def execute_plan(self, plan: ExecutionPlan) -> Dict[str, Any]:
        """Execute plan without streaming (for backward compatibility)"""
        updates = []
        async for update in self.execute_plan_stream(plan):
            updates.append(update)
            
            # Log important updates
            if update.type in [UpdateType.STEP_COMPLETE, UpdateType.TOOL_ERROR, UpdateType.PLAN_ERROR]:
                if update.error:
                    logger.error(f"AsyncOrchestrator: {update.error}")
                else:
                    logger.info(f"AsyncOrchestrator: {update.description}")
        
        return self.step_outputs
    
    def discover_tools_for_capability(self, capability: str) -> List[str]:
        """Discover tools that provide a specific capability"""
        from Core.AgentTools.tool_registry import ToolCapability
        
        try:
            cap = ToolCapability(capability)
            return self.tool_registry.discover_tools(capabilities={cap})
        except ValueError:
            logger.warning(f"Unknown capability: {capability}")
            return []
    
    def get_tool_info(self, tool_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a tool"""
        metadata = self.tool_registry.get_tool_metadata(tool_id)
        if metadata:
            return {
                "tool_id": tool_id,
                "name": metadata.name,
                "description": metadata.description,
                "category": metadata.category.value,
                "capabilities": [cap.value for cap in metadata.capabilities],
                "parallelizable": metadata.is_parallelizable,
                "tags": metadata.tags,
                "performance_hint": metadata.performance_hint
            }
        return None