# Core/AgentOrchestrator/enhanced_orchestrator.py

from typing import Dict, Any, List, Optional, Tuple, Type, Union
import asyncio
from datetime import datetime

from Core.AgentOrchestrator.orchestrator import AgentOrchestrator
from Core.AgentSchema.plan import ExecutionPlan, ExecutionStep, ToolCall
from Core.AgentSchema.context import GraphRAGContext
from Core.Common.Logger import logger
from Core.Common.PerformanceMonitor import PerformanceMonitor
from Core.Common.StructuredErrors import (
    StructuredError, ErrorCategory, ErrorSeverity, 
    LLMRateLimitError, LLMTimeoutError, EmbeddingError
)
from Core.Common.LLMEnhancements import enhanced_llm_call, AdaptiveTimeout
from Core.Provider.EnhancedLiteLLMProvider import EnhancedLiteLLMProvider
from Core.Index.EnhancedFaissIndex import EnhancedFaissIndex
from pydantic import BaseModel
from Option.Config2 import Config
from Core.Provider.BaseLLM import BaseLLM
from Core.Provider.BaseEmb import BaseEmb as LlamaIndexBaseEmbedding
from Core.Chunk.ChunkFactory import ChunkFactory


class EnhancedAgentOrchestrator(AgentOrchestrator):
    """Enhanced orchestrator with performance monitoring and structured error handling."""
    
    def __init__(self, 
                 main_config: Config, 
                 llm_instance: BaseLLM, 
                 encoder_instance: LlamaIndexBaseEmbedding, 
                 chunk_factory: ChunkFactory, 
                 graphrag_context: Optional[GraphRAGContext] = None):
        # Initialize performance monitor
        self.performance_monitor = PerformanceMonitor()
        
        # Wrap LLM and encoder with enhanced versions if not already wrapped
        if not isinstance(llm_instance, EnhancedLiteLLMProvider):
            logger.info("Wrapping LLM with EnhancedLiteLLMProvider")
            self.enhanced_llm = EnhancedLiteLLMProvider(
                api_key=getattr(llm_instance, '_api_key', main_config.llm.api_key),
                model=getattr(llm_instance, '_model_name', main_config.llm.model),
                temperature=getattr(llm_instance, '_temperature', main_config.llm.temperature),
                base_provider=llm_instance
            )
        else:
            self.enhanced_llm = llm_instance
            
        # Initialize parent with enhanced providers
        super().__init__(
            main_config=main_config,
            llm_instance=self.enhanced_llm,
            encoder_instance=encoder_instance,
            chunk_factory=chunk_factory,
            graphrag_context=graphrag_context
        )
        
        # Initialize adaptive timeout
        self.adaptive_timeout = AdaptiveTimeout(
            initial_timeout=60.0,
            max_timeout=600.0,
            min_timeout=10.0
        )
        
    async def _execute_tool_with_monitoring(
        self,
        tool_function: callable,
        tool_input: BaseModel,
        tool_id: str,
        **kwargs
    ) -> Any:
        """Execute a tool with performance monitoring and error handling."""
        start_time = datetime.now()
        
        try:
            # Monitor tool execution
            with self.performance_monitor.measure_operation(f"tool_{tool_id}"):
                # Estimate timeout based on tool type
                if "VDB.Build" in tool_id:
                    estimated_tokens = 5000  # VDB builds process many items
                elif "graph.Build" in tool_id:
                    estimated_tokens = 10000  # Graph builds are intensive
                else:
                    estimated_tokens = 1000  # Default for other tools
                    
                timeout = self.adaptive_timeout.get_timeout(estimated_tokens)
                
                # Execute with timeout
                result = await asyncio.wait_for(
                    tool_function(tool_input, **kwargs),
                    timeout=timeout
                )
                
                # Update adaptive timeout on success
                duration = (datetime.now() - start_time).total_seconds()
                self.adaptive_timeout.update(estimated_tokens, duration, success=True)
                
                # Log performance metrics
                logger.info(f"Tool {tool_id} completed in {duration:.2f}s")
                
                return result
                
        except asyncio.TimeoutError:
            duration = (datetime.now() - start_time).total_seconds()
            self.adaptive_timeout.update(estimated_tokens, duration, success=False)
            
            raise LLMTimeoutError(
                message=f"Tool {tool_id} timed out after {timeout:.0f}s",
                context={
                    "tool_id": tool_id,
                    "timeout": timeout,
                    "duration": duration
                },
                recovery_strategies=[
                    {"strategy": "retry", "params": {"timeout_multiplier": 2}},
                    {"strategy": "skip", "params": {"reason": "timeout"}}
                ]
            )
            
        except Exception as e:
            # Wrap in structured error
            if "rate limit" in str(e).lower():
                raise LLMRateLimitError(
                    message=f"Rate limit hit for tool {tool_id}",
                    context={"tool_id": tool_id, "error": str(e)},
                    recovery_strategies=[
                        {"strategy": "wait", "params": {"duration": 60}},
                        {"strategy": "retry", "params": {"backoff": "exponential"}}
                    ]
                )
            elif "embedding" in tool_id.lower():
                raise EmbeddingError(
                    message=f"Embedding error in tool {tool_id}: {str(e)}",
                    context={"tool_id": tool_id, "error": str(e)},
                    recovery_strategies=[
                        {"strategy": "retry", "params": {"max_attempts": 3}},
                        {"strategy": "fallback", "params": {"method": "tfidf"}}
                    ]
                )
            else:
                raise StructuredError(
                    message=f"Error in tool {tool_id}: {str(e)}",
                    category=ErrorCategory.TOOL_ERROR,
                    severity=ErrorSeverity.ERROR,
                    context={"tool_id": tool_id, "error": str(e)},
                    cause=e
                )
    
    async def execute_plan(self, plan: ExecutionPlan) -> Dict[str, Any]:
        """Execute plan with enhanced monitoring and error handling."""
        logger.info(f"Enhanced Orchestrator: Starting execution of plan ID: {plan.plan_id}")
        
        # Monitor overall plan execution
        with self.performance_monitor.measure_operation("plan_execution"):
            self.step_outputs: Dict[str, Dict[str, Any]] = {}
            
            for step_index, step in enumerate(plan.steps):
                logger.info(f"Enhanced Orchestrator: Executing Step {step_index + 1}/{len(plan.steps)}: {step.step_id}")
                
                # Monitor step execution
                with self.performance_monitor.measure_operation(f"step_{step.step_id}"):
                    try:
                        await self._execute_step(step, plan)
                    except StructuredError as e:
                        logger.error(f"Structured error in step {step.step_id}: {e}")
                        
                        # Try recovery strategies
                        recovered = False
                        for strategy in e.recovery_strategies or []:
                            if strategy["strategy"] == "retry":
                                logger.info(f"Retrying step {step.step_id}")
                                try:
                                    await self._execute_step(step, plan)
                                    recovered = True
                                    break
                                except:
                                    continue
                            elif strategy["strategy"] == "skip":
                                logger.warning(f"Skipping step {step.step_id}")
                                self.step_outputs[step.step_id] = {"error": str(e), "skipped": True}
                                recovered = True
                                break
                                
                        if not recovered:
                            self.step_outputs[step.step_id] = {"error": str(e)}
                            
                    except Exception as e:
                        logger.error(f"Unexpected error in step {step.step_id}: {e}")
                        self.step_outputs[step.step_id] = {"error": str(e)}
        
        # Log performance summary
        summary = self.performance_monitor.get_summary()
        logger.info(f"Plan execution complete. Performance summary: {summary}")
        
        return self.step_outputs
    
    async def _execute_step(self, step: ExecutionStep, plan: ExecutionPlan):
        """Execute a single step with enhanced error handling."""
        from Core.AgentSchema.plan import DynamicToolChainConfig
        
        tool_calls_in_step: List[ToolCall] = []
        
        if isinstance(step.action, DynamicToolChainConfig) and step.action.tools:
            tool_calls_in_step = step.action.tools
        else:
            logger.warning(f"Step {step.step_id} has unsupported/empty action")
            self.step_outputs[step.step_id] = {"error": "Unsupported or empty action"}
            return
            
        current_step_outputs = {}
        
        for tool_call_index, tool_call in enumerate(tool_calls_in_step):
            logger.info(f"Tool {tool_call_index + 1}/{len(tool_calls_in_step)} in {step.step_id}: {tool_call.tool_id}")
            
            if not tool_call.tool_id:
                logger.error("Tool call missing tool_id")
                continue
                
            tool_info = self._tool_registry.get(tool_call.tool_id)
            if not tool_info:
                logger.error(f"Tool ID '{tool_call.tool_id}' not found")
                continue
                
            tool_function, pydantic_input_model_class = tool_info
            
            try:
                # Resolve inputs
                final_tool_params = self._resolve_tool_inputs(
                    tool_call_inputs=tool_call.inputs,
                    tool_call_parameters=tool_call.parameters,
                    plan_inputs=plan.plan_inputs
                )
                
                # Create input instance
                current_tool_input_instance = pydantic_input_model_class(**final_tool_params)
                
                # Prepare kwargs based on tool type
                kwargs = {}
                if tool_call.tool_id.startswith("graph.Build"):
                    kwargs = {
                        "main_config": self.main_config,
                        "llm_instance": self.enhanced_llm,  # Use enhanced LLM
                        "encoder_instance": self.encoder,
                        "chunk_factory": self.chunk_factory
                    }
                elif tool_call.tool_id == "corpus.PrepareFromDirectory":
                    kwargs = {"main_config": self.main_config}
                else:
                    if not self.graphrag_context:
                        raise ValueError(f"GraphRAGContext is None, required by tool {tool_call.tool_id}")
                    kwargs = {"graphrag_context": self.graphrag_context}
                
                # Execute tool with monitoring
                tool_output = await self._execute_tool_with_monitoring(
                    tool_function=tool_function,
                    tool_input=current_tool_input_instance,
                    tool_id=tool_call.tool_id,
                    **kwargs
                )
                
                # Process output (same as parent class)
                self._process_tool_output(tool_call, tool_output, step, current_step_outputs)
                
            except StructuredError:
                raise  # Re-raise structured errors for handling at step level
            except Exception as e:
                logger.error(f"Error in tool {tool_call.tool_id}: {e}")
                raise StructuredError(
                    message=f"Tool execution failed: {tool_call.tool_id}",
                    category=ErrorCategory.TOOL_ERROR,
                    severity=ErrorSeverity.ERROR,
                    context={
                        "tool_id": tool_call.tool_id,
                        "step_id": step.step_id,
                        "error": str(e)
                    },
                    cause=e
                )
        
        self.step_outputs[step.step_id] = current_step_outputs
    
    def _process_tool_output(self, tool_call: ToolCall, tool_output: Any, step: ExecutionStep, current_step_outputs: Dict[str, Any]):
        """Process tool output (extracted from parent for clarity)."""
        # This is the same logic as in the parent class for handling outputs
        # Including graph registration, named outputs, etc.
        
        # Register graph instances after building
        if tool_call.tool_id.startswith("graph.Build") and tool_output is not None:
            if hasattr(tool_output, 'graph_id') and hasattr(tool_output, 'status'):
                if tool_output.status == "success" and tool_output.graph_id:
                    actual_built_graph_instance = getattr(tool_output, "graph_instance", None)
                    
                    if actual_built_graph_instance:
                        logger.info(f"Using graph instance returned by tool for '{tool_output.graph_id}'")
                        
                        # Set namespace if needed
                        if hasattr(actual_built_graph_instance._graph, 'namespace'):
                            dataset_name = tool_output.graph_id
                            for suffix in ["_ERGraph", "_RKGraph", "_TreeGraphBalanced", "_TreeGraph", "_PassageGraph"]:
                                if dataset_name.endswith(suffix):
                                    dataset_name = dataset_name[:-len(suffix)]
                                    break
                            
                            actual_built_graph_instance._graph.namespace = self.chunk_factory.get_namespace(dataset_name)
                            logger.debug(f"Set graph namespace to {dataset_name}")
                        
                        # Register the populated instance
                        self.graphrag_context.add_graph_instance(tool_output.graph_id, actual_built_graph_instance)
                        logger.info(f"Successfully registered graph instance '{tool_output.graph_id}'")
        
        # Handle named outputs
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
                        logger.warning(f"Named output key '{source_key_in_model}' not found")
            elif not actual_output_dict and len(tool_call.named_outputs) == 1:
                plan_key = list(tool_call.named_outputs.keys())[0]
                output_data_to_store[plan_key] = tool_output
            
            for key, value in output_data_to_store.items():
                current_step_outputs[key] = value
                logger.info(f"Stored output '{key}' for step {step.step_id}")
                
        elif tool_output is not None:
            # Store output even without named_outputs
            if isinstance(tool_output, dict):
                for key, value in tool_output.items():
                    current_step_outputs[key] = value
            else:
                current_step_outputs[tool_call.tool_id] = tool_output