# Core/AgentOrchestrator/memory_enhanced_orchestrator.py

from typing import Dict, Any, Optional, AsyncGenerator
import time
from datetime import datetime

from Core.AgentOrchestrator.async_streaming_orchestrator_v2 import (
    AsyncStreamingOrchestrator, 
    StreamingUpdate,
    UpdateType
)
from Core.Memory.memory_system import GraphRAGMemory
from Core.AgentSchema.plan import ExecutionPlan
from Core.Common.Logger import logger
from Option.Config2 import Config
from Core.Provider.BaseLLM import BaseLLM
from Core.Provider.BaseEmb import BaseEmb as LlamaIndexBaseEmbedding
from Core.Chunk.ChunkFactory import ChunkFactory
from Core.AgentSchema.context import GraphRAGContext


class MemoryEnhancedOrchestrator(AsyncStreamingOrchestrator):
    """
    Orchestrator enhanced with memory system for learning and strategy recommendation
    """
    
    def __init__(
        self,
        main_config: Config,
        llm_instance: BaseLLM,
        encoder_instance: LlamaIndexBaseEmbedding,
        chunk_factory: ChunkFactory,
        graphrag_context: Optional[GraphRAGContext] = None,
        memory_path: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        super().__init__(
            main_config=main_config,
            llm_instance=llm_instance,
            encoder_instance=encoder_instance,
            chunk_factory=chunk_factory,
            graphrag_context=graphrag_context
        )
        
        # Initialize memory system
        self.memory = GraphRAGMemory(storage_path=memory_path)
        self.current_user_id = user_id
        
        # Execution tracking
        self.current_execution_start: Optional[datetime] = None
        self.current_query: Optional[str] = None
        self.execution_metrics: Dict[str, Any] = {}
        
        logger.info("MemoryEnhancedOrchestrator: Initialized with memory system")
        
    def set_user_id(self, user_id: str):
        """Set current user ID for personalized memory"""
        self.current_user_id = user_id
        
    async def execute_plan_stream(
        self, 
        plan: ExecutionPlan,
        query: Optional[str] = None,
        use_memory_recommendation: bool = True
    ) -> AsyncGenerator[StreamingUpdate, None]:
        """
        Execute plan with memory-based enhancements and learning
        """
        # Start execution tracking
        self.current_execution_start = datetime.utcnow()
        self.current_query = query or plan.plan_description
        self.execution_metrics = {
            "start_time": self.current_execution_start,
            "plan_id": plan.plan_id,
            "steps_completed": 0,
            "tools_executed": [],
            "errors": []
        }
        
        # Try to get strategy recommendation from memory
        if use_memory_recommendation and query:
            recommendation = self.memory.recommend_strategy(query, self.current_user_id)
            
            if recommendation and recommendation["confidence"] > 0.8:
                # Yield memory recommendation update
                yield StreamingUpdate(
                    type=UpdateType.PLAN_START,
                    timestamp=datetime.utcnow(),
                    description=f"Using recommended strategy based on {recommendation['confidence']:.0%} success rate",
                    data={
                        "recommendation": recommendation,
                        "source": "memory"
                    }
                )
                
                # Could potentially modify the plan based on recommendation
                # For now, just log it
                logger.info(f"Memory recommendation: {recommendation['strategy_id']} "
                          f"with expected quality {recommendation['expected_quality']:.2f}")
        
        # Add session context
        if query:
            self.memory.session_memory.add_conversation_turn(
                query=query,
                response=None,  # Will update later
                metadata={"plan_id": plan.plan_id}
            )
        
        # Execute plan with base orchestrator
        execution_start_ms = int(time.time() * 1000)
        
        try:
            async for update in super().execute_plan_stream(plan):
                # Track execution progress
                if update.type == UpdateType.STEP_COMPLETE:
                    self.execution_metrics["steps_completed"] += 1
                elif update.type == UpdateType.TOOL_COMPLETE:
                    if update.tool_id:
                        self.execution_metrics["tools_executed"].append(update.tool_id)
                elif update.type in [UpdateType.STEP_ERROR, UpdateType.TOOL_ERROR]:
                    self.execution_metrics["errors"].append({
                        "type": update.type.value,
                        "error": update.error,
                        "timestamp": update.timestamp
                    })
                    
                # Yield the update
                yield update
                
                # Check for plan completion
                if update.type == UpdateType.PLAN_COMPLETE:
                    # Calculate execution time
                    execution_time_ms = int(time.time() * 1000) - execution_start_ms
                    
                    # Calculate quality score (simplified - could be more sophisticated)
                    quality_score = self._calculate_quality_score()
                    
                    # Learn from this execution
                    await self._learn_from_execution(
                        plan=plan,
                        execution_time_ms=execution_time_ms,
                        quality_score=quality_score
                    )
                    
                    # Update session with final response
                    if self.memory.session_memory.conversation_history:
                        self.memory.session_memory.conversation_history[-1]["response"] = self.step_outputs
                        
        except Exception as e:
            # Record failure in memory
            if query:
                query_type = self.memory._classify_query(query)
                self.memory.pattern_memory.update_pattern_failure(query_type, plan.plan_id)
            raise
            
    async def _learn_from_execution(
        self,
        plan: ExecutionPlan,
        execution_time_ms: int,
        quality_score: float
    ):
        """Learn from the execution results"""
        if not self.current_query:
            return
            
        # Gather execution results
        execution_results = {
            "outputs": self.step_outputs,
            "metrics": self.execution_metrics,
            "success": len(self.execution_metrics["errors"]) == 0
        }
        
        # Learn from execution
        self.memory.learn_from_execution(
            query=self.current_query,
            user_id=self.current_user_id,
            plan=plan,
            execution_results=execution_results,
            quality_score=quality_score,
            execution_time_ms=execution_time_ms
        )
        
        logger.info(f"MemoryEnhancedOrchestrator: Learned from execution - "
                   f"Quality: {quality_score:.2f}, Time: {execution_time_ms}ms")
        
        # Persist memories periodically
        if self.memory.pattern_memory.patterns and len(self.memory.pattern_memory.patterns) % 10 == 0:
            self.memory.persist_memories()
            
    def _calculate_quality_score(self) -> float:
        """
        Calculate quality score for the execution
        TODO: This should be enhanced with actual quality evaluation
        """
        base_score = 1.0
        
        # Deduct for errors
        error_penalty = len(self.execution_metrics["errors"]) * 0.2
        base_score -= min(error_penalty, 0.5)
        
        # Deduct for missing outputs
        expected_outputs = sum(
            len(step.action.tools) if hasattr(step.action, 'tools') else 0
            for step in self.execution_metrics.get("plan", {}).get("steps", [])
        )
        actual_outputs = len(self.step_outputs)
        if expected_outputs > 0:
            output_ratio = actual_outputs / expected_outputs
            base_score *= output_ratio
            
        return max(0.0, min(1.0, base_score))
        
    def get_session_context(self, num_turns: int = 5) -> List[Dict[str, Any]]:
        """Get recent session context from memory"""
        return self.memory.get_session_context(num_turns)
        
    def get_user_preferences(self) -> Dict[str, Any]:
        """Get current user's preferences"""
        if self.current_user_id:
            return self.memory.user_memory.get_user_preferences(self.current_user_id)
        return {}
        
    def update_user_preference(
        self,
        preference_type: str,
        value: Any,
        confidence: float = 1.0
    ):
        """Update user preference"""
        if self.current_user_id:
            self.memory.update_user_preference(
                user_id=self.current_user_id,
                preference_type=preference_type,
                value=value,
                confidence=confidence
            )
            
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system-wide statistics from memory"""
        return self.memory.get_system_stats()
        
    def get_recommended_tools_for_query(self, query: str) -> Optional[List[str]]:
        """Get recommended tool sequence for a query type"""
        recommendation = self.memory.recommend_strategy(query, self.current_user_id)
        if recommendation:
            return recommendation.get("tool_sequence", [])
        return None
        
    async def cleanup_memory(self):
        """Clean up expired memory entries"""
        cleaned = await self.memory.cleanup_expired()
        logger.info(f"MemoryEnhancedOrchestrator: Cleaned {cleaned} expired entries")
        
    def persist_memory(self):
        """Persist memory to disk"""
        self.memory.persist_memories()