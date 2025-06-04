"""
MCP-Based Blackboard System for Cognitive Architecture

This module implements a blackboard system built on top of MCP's shared context,
enabling knowledge sources to collaborate on complex problem-solving tasks.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Callable, Set
from datetime import datetime
from enum import Enum
from collections import defaultdict

from .shared_context import SharedContextStore

logger = logging.getLogger(__name__)


class BlackboardEventType(Enum):
    """Types of blackboard events"""
    KNOWLEDGE_ADDED = "knowledge_added"
    KNOWLEDGE_UPDATED = "knowledge_updated"
    KNOWLEDGE_REMOVED = "knowledge_removed"
    HYPOTHESIS_PROPOSED = "hypothesis_proposed"
    HYPOTHESIS_VALIDATED = "hypothesis_validated"
    HYPOTHESIS_REJECTED = "hypothesis_rejected"
    SOLUTION_FOUND = "solution_found"
    CONTROL_CHANGE = "control_change"


class KnowledgeType(Enum):
    """Types of knowledge that can be stored on the blackboard"""
    FACT = "fact"
    HYPOTHESIS = "hypothesis"
    INFERENCE = "inference"
    CONSTRAINT = "constraint"
    GOAL = "goal"
    SOLUTION = "solution"
    PARTIAL_SOLUTION = "partial_solution"
    EVIDENCE = "evidence"


class KnowledgeEntry:
    """Represents a piece of knowledge on the blackboard"""
    
    def __init__(
        self,
        knowledge_type: KnowledgeType,
        content: Any,
        source: str,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.id = f"{knowledge_type.value}_{datetime.utcnow().timestamp()}"
        self.knowledge_type = knowledge_type
        self.content = content
        self.source = source
        self.confidence = confidence
        self.metadata = metadata or {}
        self.created_at = datetime.utcnow()
        self.updated_at = self.created_at
        self.supporting_evidence: List[str] = []
        self.contradicting_evidence: List[str] = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "knowledge_type": self.knowledge_type.value,
            "content": self.content,
            "source": self.source,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "supporting_evidence": self.supporting_evidence,
            "contradicting_evidence": self.contradicting_evidence
        }


class BlackboardSystem:
    """
    MCP-Based Blackboard System
    
    Extends SharedContextStore to provide a collaborative problem-solving environment
    where multiple knowledge sources can contribute to finding solutions.
    """
    
    def __init__(self, shared_context: Optional[SharedContextStore] = None, session_id: str = "blackboard"):
        self.shared_context = shared_context or SharedContextStore()
        self.session_id = session_id
        self.knowledge_sources: Dict[str, 'KnowledgeSource'] = {}
        self.event_subscribers: Dict[BlackboardEventType, List[Callable]] = defaultdict(list)
        self.control_strategy: Optional['ControlStrategy'] = None
        self._lock = asyncio.Lock()
        
        # Initialize blackboard namespace in shared context
        self._init_blackboard_namespace()
    
    def _init_blackboard_namespace(self):
        """Initialize the blackboard namespace in shared context"""
        blackboard_data = {
            "knowledge": {},
            "hypotheses": {},
            "solutions": {},
            "control_state": {
                "active_sources": [],
                "current_focus": None,
                "iteration": 0
            }
        }
        # Store directly in memory for simplicity - can be made async later
        self._blackboard_data = blackboard_data
    
    async def add_knowledge(
        self,
        knowledge: KnowledgeEntry,
        notify: bool = True
    ) -> str:
        """Add a piece of knowledge to the blackboard"""
        async with self._lock:
            # Get current blackboard data
            blackboard_data = self._blackboard_data
            knowledge_dict = blackboard_data.get("knowledge", {})
            
            # Add the knowledge entry
            knowledge_dict[knowledge.id] = knowledge.to_dict()
            blackboard_data["knowledge"] = knowledge_dict
            
            # Notify subscribers
            if notify:
                await self._notify_event(
                    BlackboardEventType.KNOWLEDGE_ADDED,
                    {"knowledge_id": knowledge.id, "knowledge": knowledge.to_dict()}
                )
            
            logger.info(f"Added knowledge {knowledge.id} from {knowledge.source}")
            return knowledge.id
    
    async def update_knowledge(
        self,
        knowledge_id: str,
        updates: Dict[str, Any],
        source: str,
        notify: bool = True
    ) -> bool:
        """Update an existing knowledge entry"""
        async with self._lock:
            blackboard_data = self._blackboard_data
            knowledge_dict = blackboard_data.get("knowledge", {})
            
            if knowledge_id not in knowledge_dict:
                return False
            
            # Update the knowledge entry
            entry = knowledge_dict[knowledge_id]
            entry.update(updates)
            entry["updated_at"] = datetime.utcnow().isoformat()
            
            # Notify subscribers
            if notify:
                await self._notify_event(
                    BlackboardEventType.KNOWLEDGE_UPDATED,
                    {"knowledge_id": knowledge_id, "updates": updates}
                )
            
            return True
    
    async def get_knowledge(
        self,
        knowledge_type: Optional[KnowledgeType] = None,
        source: Optional[str] = None,
        min_confidence: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Retrieve knowledge from the blackboard"""
        blackboard_data = self._blackboard_data or {}
        knowledge_dict = blackboard_data.get("knowledge", {})
        
        results = []
        for entry in knowledge_dict.values():
            # Apply filters
            if knowledge_type and entry["knowledge_type"] != knowledge_type.value:
                continue
            if source and entry["source"] != source:
                continue
            if entry["confidence"] < min_confidence:
                continue
            
            results.append(entry)
        
        return sorted(results, key=lambda x: x["confidence"], reverse=True)
    
    async def propose_hypothesis(
        self,
        hypothesis: KnowledgeEntry,
        supporting_evidence: List[str]
    ) -> str:
        """Propose a new hypothesis based on available knowledge"""
        hypothesis.supporting_evidence = supporting_evidence
        hypothesis_id = await self.add_knowledge(hypothesis, notify=False)
        
        # Store in hypotheses section
        blackboard_data = self._blackboard_data or {}
        hypotheses = blackboard_data.get("hypotheses", {})
        hypotheses[hypothesis_id] = {
            "status": "proposed",
            "validation_attempts": 0,
            "validators": []
        }
        blackboard_data["hypotheses"] = hypotheses
        
        # Notify subscribers
        await self._notify_event(
            BlackboardEventType.HYPOTHESIS_PROPOSED,
            {"hypothesis_id": hypothesis_id, "hypothesis": hypothesis.to_dict()}
        )
        
        return hypothesis_id
    
    async def validate_hypothesis(
        self,
        hypothesis_id: str,
        validator: str,
        is_valid: bool,
        evidence: List[str],
        confidence: float
    ) -> bool:
        """Validate or reject a hypothesis"""
        async with self._lock:
            blackboard_data = self._blackboard_data or {}
            hypotheses = blackboard_data.get("hypotheses", {})
            
            if hypothesis_id not in hypotheses:
                return False
            
            # Update hypothesis status
            hypothesis_info = hypotheses[hypothesis_id]
            hypothesis_info["validation_attempts"] += 1
            hypothesis_info["validators"].append({
                "validator": validator,
                "is_valid": is_valid,
                "confidence": confidence,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Update knowledge entry
            knowledge_dict = blackboard_data.get("knowledge", {})
            if hypothesis_id in knowledge_dict:
                entry = knowledge_dict[hypothesis_id]
                if is_valid:
                    entry["supporting_evidence"].extend(evidence)
                else:
                    entry["contradicting_evidence"].extend(evidence)
                
                # Update confidence based on validation
                validations = hypothesis_info["validators"]
                avg_confidence = sum(v["confidence"] for v in validations if v["is_valid"]) / len(validations)
                entry["confidence"] = avg_confidence
            
            # Update shared context
            # Data is updated in memory
            
            # Notify subscribers
            event_type = (
                BlackboardEventType.HYPOTHESIS_VALIDATED 
                if is_valid 
                else BlackboardEventType.HYPOTHESIS_REJECTED
            )
            await self._notify_event(
                event_type,
                {"hypothesis_id": hypothesis_id, "validator": validator, "is_valid": is_valid}
            )
            
            return True
    
    async def add_solution(
        self,
        solution: KnowledgeEntry,
        contributing_knowledge: List[str]
    ) -> str:
        """Add a solution to the blackboard"""
        solution.supporting_evidence = contributing_knowledge
        solution_id = await self.add_knowledge(solution, notify=False)
        
        # Store in solutions section
        blackboard_data = self._blackboard_data or {}
        solutions = blackboard_data.get("solutions", {})
        solutions[solution_id] = {
            "completeness": 1.0 if solution.knowledge_type == KnowledgeType.SOLUTION else 0.5,
            "contributing_knowledge": contributing_knowledge,
            "timestamp": datetime.utcnow().isoformat()
        }
        blackboard_data["solutions"] = solutions
        # Data is updated in memory
        
        # Notify subscribers
        await self._notify_event(
            BlackboardEventType.SOLUTION_FOUND,
            {"solution_id": solution_id, "solution": solution.to_dict()}
        )
        
        return solution_id
    
    def register_knowledge_source(self, source: 'KnowledgeSource'):
        """Register a knowledge source with the blackboard"""
        self.knowledge_sources[source.name] = source
        source.blackboard = self
        logger.info(f"Registered knowledge source: {source.name}")
    
    def subscribe_to_event(
        self,
        event_type: BlackboardEventType,
        callback: Callable
    ):
        """Subscribe to blackboard events"""
        self.event_subscribers[event_type].append(callback)
    
    async def _notify_event(
        self,
        event_type: BlackboardEventType,
        data: Dict[str, Any]
    ):
        """Notify all subscribers of an event"""
        for callback in self.event_subscribers[event_type]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event_type, data)
                else:
                    callback(event_type, data)
            except Exception as e:
                logger.error(f"Error in event callback: {e}")
    
    def set_control_strategy(self, strategy: 'ControlStrategy'):
        """Set the control strategy for the blackboard"""
        self.control_strategy = strategy
        strategy.blackboard = self
    
    async def run_control_cycle(self, max_iterations: int = 100) -> Optional[str]:
        """Run the blackboard control cycle"""
        if not self.control_strategy:
            raise ValueError("No control strategy set")
        
        for iteration in range(max_iterations):
            # Update control state
            blackboard_data = self._blackboard_data or {}
            control_state = blackboard_data.get("control_state", {})
            control_state["iteration"] = iteration
            blackboard_data["control_state"] = control_state
            # Data is updated in memory
            
            # Execute control strategy
            result = await self.control_strategy.execute_cycle(iteration)
            
            if result:  # Solution found
                return result
            
            # Check for termination conditions
            if await self.control_strategy.should_terminate():
                break
        
        return None


class KnowledgeSource:
    """Base class for knowledge sources that contribute to the blackboard"""
    
    def __init__(self, name: str):
        self.name = name
        self.blackboard: Optional[BlackboardSystem] = None
    
    async def contribute(self) -> List[KnowledgeEntry]:
        """Generate knowledge contributions"""
        raise NotImplementedError
    
    async def can_contribute(self) -> bool:
        """Check if this source can contribute given current blackboard state"""
        return True


class ControlStrategy:
    """Base class for blackboard control strategies"""
    
    def __init__(self):
        self.blackboard: Optional[BlackboardSystem] = None
    
    async def execute_cycle(self, iteration: int) -> Optional[str]:
        """Execute one control cycle, return solution ID if found"""
        raise NotImplementedError
    
    async def should_terminate(self) -> bool:
        """Check if the control cycle should terminate"""
        raise NotImplementedError


class ReactiveControlStrategy(ControlStrategy):
    """Reactive control strategy that responds to blackboard events"""
    
    def __init__(self, solution_threshold: float = 0.8):
        super().__init__()
        self.solution_threshold = solution_threshold
        self.pending_validations: Set[str] = set()
    
    async def execute_cycle(self, iteration: int) -> Optional[str]:
        """Execute one reactive control cycle"""
        # Check for complete solutions
        blackboard_data = self.blackboard._blackboard_data
        solutions = blackboard_data.get("solutions", {})
        
        for solution_id, solution_info in solutions.items():
            if solution_info["completeness"] >= self.solution_threshold:
                knowledge = blackboard_data["knowledge"].get(solution_id, {})
                if knowledge.get("confidence", 0) >= self.solution_threshold:
                    return solution_id
        
        # Activate knowledge sources that can contribute
        active_sources = []
        for source_name, source in self.blackboard.knowledge_sources.items():
            if await source.can_contribute():
                contributions = await source.contribute()
                for contribution in contributions:
                    await self.blackboard.add_knowledge(contribution)
                active_sources.append(source_name)
        
        # Update control state
        control_state = blackboard_data.get("control_state", {})
        control_state["active_sources"] = active_sources
        blackboard_data["control_state"] = control_state
        # Data is updated in memory
        
        # Process pending validations
        hypotheses = blackboard_data.get("hypotheses", {})
        for hypothesis_id, hypothesis_info in hypotheses.items():
            if hypothesis_info["status"] == "proposed" and hypothesis_id not in self.pending_validations:
                self.pending_validations.add(hypothesis_id)
                # Trigger validation by knowledge sources
                # This would be implemented by specific knowledge sources
        
        return None
    
    async def should_terminate(self) -> bool:
        """Check if we should stop the control cycle"""
        blackboard_data = self.blackboard._blackboard_data
        
        # Terminate if we have no active sources and no pending validations
        control_state = blackboard_data.get("control_state", {})
        if not control_state.get("active_sources") and not self.pending_validations:
            return True
        
        # Terminate if we've been running too long
        if control_state.get("iteration", 0) > 1000:
            return True
        
        return False