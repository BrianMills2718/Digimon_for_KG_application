"""
Atom of Thoughts (AOT) Query Preprocessor
Decomposes complex queries into atomic states using Markov process
"""

import asyncio
import logging
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json

from Core.Common.Logger import logger

@dataclass
class AtomicState:
    """Represents an atomic, memoryless state in the Markov process"""
    state_id: str
    content: str
    state_type: str  # "entity", "relationship", "attribute", "action"
    dependencies: Set[str]  # IDs of states this depends on
    metadata: Dict[str, Any]
    
    def __hash__(self):
        return hash(self.state_id)
    
    def to_dict(self) -> dict:
        return {
            "state_id": self.state_id,
            "content": self.content,
            "state_type": self.state_type,
            "dependencies": list(self.dependencies),
            "metadata": self.metadata
        }

@dataclass
class TransitionProbability:
    """Represents transition probability between atomic states"""
    from_state: str
    to_state: str
    probability: float
    transition_type: str  # "sequential", "parallel", "conditional"


class AOTQueryProcessor:
    """
    Implements Atom of Thoughts decomposition for queries
    Reduces context size by breaking queries into atomic, memoryless states
    """
    
    def __init__(self):
        self.state_cache: Dict[str, AtomicState] = {}
        self.transitions: Dict[Tuple[str, str], TransitionProbability] = {}
        self.execution_history: List[Dict[str, Any]] = []
        
    async def decompose_query(self, query: str, context: Dict[str, Any] = None) -> List[AtomicState]:
        """
        Decompose a complex query into atomic states
        
        Args:
            query: The input query to decompose
            context: Optional context that might influence decomposition
            
        Returns:
            List of atomic states representing the query
        """
        logger.info(f"AOT: Decomposing query: {query[:100]}...")
        
        # Extract atomic components from the query
        atomic_states = []
        
        # 1. Extract entities
        entities = await self._extract_entities(query)
        for entity in entities:
            state = AtomicState(
                state_id=self._generate_state_id("entity", entity),
                content=entity,
                state_type="entity",
                dependencies=set(),
                metadata={"source": "query", "timestamp": datetime.utcnow().isoformat()}
            )
            atomic_states.append(state)
            self.state_cache[state.state_id] = state
        
        # 2. Extract relationships
        relationships = await self._extract_relationships(query, entities)
        for rel in relationships:
            deps = {self._generate_state_id("entity", e) for e in rel["entities"]}
            state = AtomicState(
                state_id=self._generate_state_id("relationship", rel["type"]),
                content=rel["type"],
                state_type="relationship",
                dependencies=deps,
                metadata={"entities": rel["entities"], "timestamp": datetime.utcnow().isoformat()}
            )
            atomic_states.append(state)
            self.state_cache[state.state_id] = state
        
        # 3. Extract attributes/filters
        attributes = await self._extract_attributes(query)
        for attr in attributes:
            state = AtomicState(
                state_id=self._generate_state_id("attribute", attr),
                content=attr,
                state_type="attribute",
                dependencies=set(),  # Attributes can apply to multiple entities
                metadata={"source": "query", "timestamp": datetime.utcnow().isoformat()}
            )
            atomic_states.append(state)
            self.state_cache[state.state_id] = state
        
        # 4. Extract actions/operations
        actions = await self._extract_actions(query)
        for action in actions:
            # Actions depend on entities they operate on
            deps = {s.state_id for s in atomic_states if s.state_type in ["entity", "relationship"]}
            state = AtomicState(
                state_id=self._generate_state_id("action", action),
                content=action,
                state_type="action",
                dependencies=deps,
                metadata={"source": "query", "timestamp": datetime.utcnow().isoformat()}
            )
            atomic_states.append(state)
            self.state_cache[state.state_id] = state
        
        # Calculate transition probabilities
        await self._calculate_transitions(atomic_states)
        
        logger.info(f"AOT: Decomposed into {len(atomic_states)} atomic states")
        return atomic_states
    
    async def _extract_entities(self, query: str) -> List[str]:
        """Extract entity mentions from query"""
        # Simplified entity extraction - in production would use NER
        entities = []
        
        # Common entity patterns
        entity_keywords = ["what", "who", "which", "where", "about", "regarding"]
        query_lower = query.lower()
        
        # Look for capitalized words (potential entities)
        words = query.split()
        for i, word in enumerate(words):
            if word[0].isupper() and word.lower() not in entity_keywords:
                entities.append(word)
            # Check for multi-word entities
            if i < len(words) - 1 and word[0].isupper() and words[i+1][0].isupper():
                entities.append(f"{word} {words[i+1]}")
        
        # Look for quoted entities
        import re
        quoted = re.findall(r'"([^"]*)"', query)
        entities.extend(quoted)
        
        return list(set(entities))  # Remove duplicates
    
    async def _extract_relationships(self, query: str, entities: List[str]) -> List[Dict[str, Any]]:
        """Extract relationships between entities"""
        relationships = []
        
        # Relationship indicators
        rel_patterns = {
            "caused": ["caused", "led to", "resulted in", "influence"],
            "related": ["related to", "connected to", "associated with"],
            "part_of": ["part of", "belongs to", "member of"],
            "temporal": ["before", "after", "during", "while"],
            "comparison": ["compare", "versus", "vs", "different from"],
            "influence": ["influence", "affect", "impact", "shape"]
        }
        
        query_lower = query.lower()
        
        for rel_type, patterns in rel_patterns.items():
            for pattern in patterns:
                if pattern in query_lower:
                    # Find entities involved in this relationship
                    involved_entities = []
                    for entity in entities:
                        if entity.lower() in query_lower:
                            involved_entities.append(entity)
                    
                    if len(involved_entities) >= 2:
                        relationships.append({
                            "type": rel_type,
                            "entities": involved_entities[:2]  # Simplified: take first two
                        })
                    elif len(involved_entities) == 1 and len(entities) >= 2:
                        # If only one entity matched but we have multiple entities,
                        # create relationship between them
                        relationships.append({
                            "type": rel_type,
                            "entities": entities[:2]
                        })
        
        return relationships
    
    async def _extract_attributes(self, query: str) -> List[str]:
        """Extract attributes or filters from query"""
        attributes = []
        
        # Attribute patterns
        attr_patterns = ["main", "primary", "important", "significant", "key", "major", "causes", "effects", "characteristics"]
        
        query_lower = query.lower()
        for pattern in attr_patterns:
            if pattern in query_lower:
                attributes.append(pattern)
        
        # Time-based attributes
        time_patterns = ["recent", "latest", "historical", "past", "current", "future"]
        for pattern in time_patterns:
            if pattern in query_lower:
                attributes.append(f"temporal:{pattern}")
        
        return attributes
    
    async def _extract_actions(self, query: str) -> List[str]:
        """Extract actions or operations from query"""
        actions = []
        
        # Question type determines action
        query_lower = query.lower()
        
        if query_lower.startswith("what"):
            actions.append("retrieve_information")
        elif query_lower.startswith("how"):
            actions.append("explain_process")
        elif query_lower.startswith("why"):
            actions.append("find_causation")
        elif query_lower.startswith("when"):
            actions.append("temporal_search")
        elif query_lower.startswith("where"):
            actions.append("location_search")
        elif "compare" in query_lower:
            actions.append("comparison")
        elif "list" in query_lower or "enumerate" in query_lower:
            actions.append("enumeration")
        else:
            actions.append("general_search")
        
        return actions
    
    async def _calculate_transitions(self, states: List[AtomicState]):
        """Calculate transition probabilities between states"""
        # For each pair of states, determine transition probability
        for i, state1 in enumerate(states):
            for j, state2 in enumerate(states):
                if i != j:
                    prob = await self._compute_transition_probability(state1, state2)
                    if prob > 0:
                        transition = TransitionProbability(
                            from_state=state1.state_id,
                            to_state=state2.state_id,
                            probability=prob,
                            transition_type=self._determine_transition_type(state1, state2)
                        )
                        self.transitions[(state1.state_id, state2.state_id)] = transition
    
    async def _compute_transition_probability(self, state1: AtomicState, state2: AtomicState) -> float:
        """Compute probability of transitioning from state1 to state2"""
        # Simple heuristic-based probability
        prob = 0.0
        
        # Dependencies create high transition probability
        if state1.state_id in state2.dependencies:
            prob = 0.9
        # Same type states have moderate probability
        elif state1.state_type == state2.state_type:
            prob = 0.3
        # Entity to relationship has high probability
        elif state1.state_type == "entity" and state2.state_type == "relationship":
            prob = 0.7
        # Action states depend on data states
        elif state2.state_type == "action" and state1.state_type in ["entity", "relationship"]:
            prob = 0.8
        
        return prob
    
    def _determine_transition_type(self, state1: AtomicState, state2: AtomicState) -> str:
        """Determine the type of transition between states"""
        if state1.state_id in state2.dependencies:
            return "sequential"
        elif state1.state_type == state2.state_type:
            return "parallel"
        else:
            return "conditional"
    
    def _generate_state_id(self, state_type: str, content: str) -> str:
        """Generate unique ID for atomic state"""
        combined = f"{state_type}:{content}"
        return hashlib.md5(combined.encode()).hexdigest()[:8]
    
    async def recompose_results(self, state_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recompose results from atomic states back into coherent response
        
        Args:
            state_results: Results from executing each atomic state
            
        Returns:
            Unified result combining all atomic state outputs
        """
        logger.info("AOT: Recomposing results from atomic states")
        
        # Group results by type
        entities = {}
        relationships = []
        attributes = {}
        actions = {}
        
        for state_id, result in state_results.items():
            if state_id in self.state_cache:
                state = self.state_cache[state_id]
                
                if state.state_type == "entity":
                    entities[state.content] = result
                elif state.state_type == "relationship":
                    relationships.append({
                        "type": state.content,
                        "entities": state.metadata.get("entities", []),
                        "data": result
                    })
                elif state.state_type == "attribute":
                    attributes[state.content] = result
                elif state.state_type == "action":
                    actions[state.content] = result
        
        # Compose final result
        composed_result = {
            "entities": entities,
            "relationships": relationships,
            "attributes": attributes,
            "actions": actions,
            "summary": await self._generate_summary(entities, relationships, attributes, actions)
        }
        
        return composed_result
    
    async def _generate_summary(self, entities: Dict, relationships: List, 
                                attributes: Dict, actions: Dict) -> str:
        """Generate a coherent summary from atomic results"""
        summary_parts = []
        
        # Summarize entities
        if entities:
            entity_names = list(entities.keys())
            summary_parts.append(f"Found {len(entity_names)} entities: {', '.join(entity_names[:3])}")
        
        # Summarize relationships
        if relationships:
            rel_types = list(set(r["type"] for r in relationships))
            summary_parts.append(f"Identified {len(relationships)} relationships of types: {', '.join(rel_types)}")
        
        # Summarize key findings from actions
        for action, result in actions.items():
            if isinstance(result, dict) and "summary" in result:
                summary_parts.append(result["summary"])
        
        return " ".join(summary_parts)
    
    def calculate_context_reduction(self, original_context: Dict[str, Any], 
                                  atomic_states: List[AtomicState]) -> float:
        """
        Calculate the context size reduction achieved by AOT decomposition
        
        Returns:
            Reduction percentage (0.0 to 1.0)
        """
        # Estimate original context size
        original_size = len(json.dumps(original_context))
        
        # Calculate atomic context size (sum of individual state contexts)
        atomic_size = 0
        for state in atomic_states:
            # Each atomic state only needs minimal context
            state_context = {
                "state": state.to_dict(),
                "dependencies": [self.state_cache.get(dep_id).to_dict() if dep_id in self.state_cache else {} 
                                for dep_id in state.dependencies]
            }
            atomic_size += len(json.dumps(state_context))
        
        # Average context per state (since they can be processed independently)
        avg_atomic_size = atomic_size / len(atomic_states) if atomic_states else atomic_size
        
        reduction = 1.0 - (avg_atomic_size / original_size) if original_size > 0 else 0.0
        logger.info(f"AOT: Context reduction: {reduction:.2%} "
                   f"(Original: {original_size} bytes, Atomic avg: {avg_atomic_size:.0f} bytes)")
        
        return reduction


# Example usage for testing
async def example_aot_usage():
    processor = AOTQueryProcessor()
    
    # Example complex query
    query = "What were the main causes of the French Revolution and how did they compare to the American Revolution?"
    
    # Decompose into atomic states
    atomic_states = await processor.decompose_query(query)
    
    print(f"Decomposed into {len(atomic_states)} atomic states:")
    for state in atomic_states:
        print(f"  - {state.state_type}: {state.content}")
        if state.dependencies:
            print(f"    Dependencies: {state.dependencies}")
    
    # Show transition probabilities
    print("\nTransition probabilities:")
    for (from_id, to_id), transition in processor.transitions.items():
        from_state = processor.state_cache[from_id]
        to_state = processor.state_cache[to_id]
        print(f"  {from_state.content} -> {to_state.content}: "
              f"{transition.probability:.2f} ({transition.transition_type})")


if __name__ == "__main__":
    asyncio.run(example_aot_usage())