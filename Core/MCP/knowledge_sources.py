"""
MCP-based Knowledge Sources for Blackboard System

These knowledge sources are exposed as MCP tools and contribute
to the blackboard for collaborative problem-solving.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from .blackboard_system import (
    KnowledgeSource, KnowledgeEntry, KnowledgeType,
    BlackboardSystem, BlackboardEventType
)
from .mcp_server import MCPTool

logger = logging.getLogger(__name__)


class EntityKnowledgeSource(KnowledgeSource):
    """Knowledge source that contributes entity-based knowledge"""
    
    def __init__(self):
        super().__init__("entity_knowledge_source")
        self.entity_tool = None  # Will be set by test or actual implementation
        self.last_query_entities = set()
    
    async def can_contribute(self) -> bool:
        """Check if we have goals that need entity information"""
        goals = await self.blackboard.get_knowledge(
            knowledge_type=KnowledgeType.GOAL
        )
        
        # Can contribute if there are unresolved goals
        for goal in goals:
            if "entities_needed" in goal.get("metadata", {}):
                return True
        
        return len(goals) > 0
    
    async def contribute(self) -> List[KnowledgeEntry]:
        """Search for entities and contribute facts"""
        contributions = []
        
        # Get current goals
        goals = await self.blackboard.get_knowledge(
            knowledge_type=KnowledgeType.GOAL
        )
        
        for goal in goals:
            goal_content = goal.get("content", {})
            if isinstance(goal_content, dict) and "query" in goal_content:
                query = goal_content["query"]
                
                # Search for entities
                try:
                    if not self.entity_tool:
                        logger.warning("Entity tool not configured")
                        continue
                        
                    result = await self.entity_tool.run({
                        "query": query,
                        "top_k": 10
                    })
                    
                    if result["status"] == "success":
                        entities = result.get("entities", [])
                        
                        # Add each entity as a fact
                        for entity in entities:
                            # Avoid duplicates
                            entity_name = entity.get("name", "")
                            if entity_name not in self.last_query_entities:
                                self.last_query_entities.add(entity_name)
                                
                                entry = KnowledgeEntry(
                                    knowledge_type=KnowledgeType.FACT,
                                    content={
                                        "entity": entity_name,
                                        "attributes": entity.get("attributes", {}),
                                        "score": entity.get("score", 0.0)
                                    },
                                    source=self.name,
                                    confidence=entity.get("score", 0.5),
                                    metadata={
                                        "goal_id": goal.get("id"),
                                        "extraction_type": "entity_search"
                                    }
                                )
                                contributions.append(entry)
                
                except Exception as e:
                    logger.error(f"Error in entity search: {e}")
        
        return contributions


class RelationshipKnowledgeSource(KnowledgeSource):
    """Knowledge source that contributes relationship-based knowledge"""
    
    def __init__(self):
        super().__init__("relationship_knowledge_source")
        self.relationship_tool = None  # Will be set by test or actual implementation
        self.processed_entities = set()
    
    async def can_contribute(self) -> bool:
        """Check if we have entity facts that need relationship exploration"""
        facts = await self.blackboard.get_knowledge(
            knowledge_type=KnowledgeType.FACT
        )
        
        # Can contribute if there are entity facts we haven't processed
        for fact in facts:
            content = fact.get("content", {})
            if isinstance(content, dict) and "entity" in content:
                if content["entity"] not in self.processed_entities:
                    return True
        
        return False
    
    async def contribute(self) -> List[KnowledgeEntry]:
        """Find relationships for known entities"""
        contributions = []
        
        # Get entity facts
        facts = await self.blackboard.get_knowledge(
            knowledge_type=KnowledgeType.FACT
        )
        
        for fact in facts:
            content = fact.get("content", {})
            if isinstance(content, dict) and "entity" in content:
                entity_name = content["entity"]
                
                if entity_name in self.processed_entities:
                    continue
                
                self.processed_entities.add(entity_name)
                
                # Search for relationships
                try:
                    if not self.relationship_tool:
                        logger.warning("Relationship tool not configured")
                        continue
                        
                    result = await self.relationship_tool.run({
                        "entity": entity_name,
                        "max_relationships": 5
                    })
                    
                    if result["status"] == "success":
                        relationships = result.get("relationships", [])
                        
                        for rel in relationships:
                            # Create inference about the relationship
                            entry = KnowledgeEntry(
                                knowledge_type=KnowledgeType.INFERENCE,
                                content={
                                    "subject": rel.get("subject"),
                                    "predicate": rel.get("predicate"),
                                    "object": rel.get("object"),
                                    "attributes": rel.get("attributes", {})
                                },
                                source=self.name,
                                confidence=0.7,  # Relationships are inferences
                                metadata={
                                    "parent_fact_id": fact.get("id"),
                                    "extraction_type": "relationship_search"
                                }
                            )
                            contributions.append(entry)
                
                except Exception as e:
                    logger.error(f"Error in relationship search: {e}")
        
        return contributions


class HypothesisGeneratorSource(KnowledgeSource):
    """Knowledge source that generates hypotheses from facts and inferences"""
    
    def __init__(self):
        super().__init__("hypothesis_generator")
        self.min_evidence_count = 2
    
    async def can_contribute(self) -> bool:
        """Check if we have enough evidence to form hypotheses"""
        facts = await self.blackboard.get_knowledge(
            knowledge_type=KnowledgeType.FACT
        )
        inferences = await self.blackboard.get_knowledge(
            knowledge_type=KnowledgeType.INFERENCE
        )
        
        return len(facts) + len(inferences) >= self.min_evidence_count
    
    async def contribute(self) -> List[KnowledgeEntry]:
        """Generate hypotheses from available evidence"""
        contributions = []
        
        # Get all evidence
        facts = await self.blackboard.get_knowledge(
            knowledge_type=KnowledgeType.FACT
        )
        inferences = await self.blackboard.get_knowledge(
            knowledge_type=KnowledgeType.INFERENCE
        )
        
        # Group evidence by entities
        entity_evidence = {}
        for evidence in facts + inferences:
            content = evidence.get("content", {})
            
            # Extract entities from content
            entities = []
            if "entity" in content:
                entities.append(content["entity"])
            if "subject" in content:
                entities.append(content["subject"])
            if "object" in content:
                entities.append(content["object"])
            
            for entity in entities:
                if entity not in entity_evidence:
                    entity_evidence[entity] = []
                entity_evidence[entity].append(evidence)
        
        # Generate hypotheses for entities with multiple pieces of evidence
        for entity, evidences in entity_evidence.items():
            if len(evidences) >= self.min_evidence_count:
                # Create a hypothesis about the entity's importance
                hypothesis = KnowledgeEntry(
                    knowledge_type=KnowledgeType.HYPOTHESIS,
                    content={
                        "entity": entity,
                        "hypothesis": f"{entity} is a key entity based on multiple evidence pieces",
                        "evidence_count": len(evidences)
                    },
                    source=self.name,
                    confidence=min(0.9, 0.3 + 0.1 * len(evidences)),
                    metadata={
                        "hypothesis_type": "entity_importance"
                    }
                )
                
                # Use blackboard's propose_hypothesis method
                supporting_ids = [e.get("id", "") for e in evidences]
                await self.blackboard.propose_hypothesis(
                    hypothesis,
                    supporting_ids
                )
        
        return contributions  # Hypotheses are added via propose_hypothesis


class SolutionSynthesizerSource(KnowledgeSource):
    """Knowledge source that synthesizes solutions from validated hypotheses"""
    
    def __init__(self):
        super().__init__("solution_synthesizer")
        self.chunk_tool = None  # Will be set by test or actual implementation
    
    async def can_contribute(self) -> bool:
        """Check if we have validated hypotheses to synthesize"""
        blackboard_data = self.blackboard._blackboard_data or {}
        hypotheses = blackboard_data.get("hypotheses", {})
        
        # Check for validated hypotheses
        for hypothesis_id, info in hypotheses.items():
            validators = info.get("validators", [])
            if validators:
                valid_count = sum(1 for v in validators if v["is_valid"])
                if valid_count > len(validators) / 2:  # Majority validated
                    return True
        
        return False
    
    async def contribute(self) -> List[KnowledgeEntry]:
        """Synthesize solutions from validated hypotheses"""
        contributions = []
        
        blackboard_data = self.blackboard._blackboard_data or {}
        hypotheses_info = blackboard_data.get("hypotheses", {})
        knowledge_dict = blackboard_data.get("knowledge", {})
        
        # Find validated hypotheses
        validated_hypotheses = []
        for hypothesis_id, info in hypotheses_info.items():
            validators = info.get("validators", [])
            if validators:
                valid_count = sum(1 for v in validators if v["is_valid"])
                if valid_count > len(validators) / 2:
                    hypothesis = knowledge_dict.get(hypothesis_id)
                    if hypothesis:
                        validated_hypotheses.append(hypothesis)
        
        if validated_hypotheses:
            # Get all supporting evidence
            all_evidence_ids = set()
            for hypothesis in validated_hypotheses:
                all_evidence_ids.update(hypothesis.get("supporting_evidence", []))
            
            # Get chunk content for evidence
            chunk_content = []
            try:
                # Get relationships from evidence
                relationships = []
                for evidence_id in all_evidence_ids:
                    evidence = knowledge_dict.get(evidence_id, {})
                    content = evidence.get("content", {})
                    if "subject" in content and "predicate" in content:
                        relationships.append({
                            "subject": content["subject"],
                            "predicate": content["predicate"],
                            "object": content.get("object", "")
                        })
                
                if relationships and self.chunk_tool:
                    result = await self.chunk_tool.run({
                        "relationships": relationships[:5]  # Limit to top 5
                    })
                    
                    if result["status"] == "success":
                        chunk_content = result.get("chunks", [])
            
            except Exception as e:
                logger.error(f"Error getting chunk content: {e}")
            
            # Create solution
            solution_content = {
                "answer": self._synthesize_answer(validated_hypotheses, chunk_content),
                "key_entities": list(set(
                    h.get("content", {}).get("entity", "")
                    for h in validated_hypotheses
                    if h.get("content", {}).get("entity")
                )),
                "confidence": max(h.get("confidence", 0.5) for h in validated_hypotheses),
                "evidence_count": len(all_evidence_ids)
            }
            
            solution = KnowledgeEntry(
                knowledge_type=KnowledgeType.SOLUTION,
                content=solution_content,
                source=self.name,
                confidence=solution_content["confidence"],
                metadata={
                    "synthesis_method": "hypothesis_validation",
                    "hypothesis_count": len(validated_hypotheses)
                }
            )
            
            # Add solution via blackboard
            await self.blackboard.add_solution(
                solution,
                list(all_evidence_ids)
            )
        
        return contributions
    
    def _synthesize_answer(
        self,
        hypotheses: List[Dict[str, Any]],
        chunk_content: List[str]
    ) -> str:
        """Synthesize an answer from hypotheses and content"""
        # Extract key points from hypotheses
        key_points = []
        for hypothesis in hypotheses:
            content = hypothesis.get("content", {})
            if "hypothesis" in content:
                key_points.append(content["hypothesis"])
        
        # Combine with chunk content
        answer_parts = ["Based on the analysis:"]
        answer_parts.extend(key_points[:3])  # Top 3 points
        
        if chunk_content:
            answer_parts.append("\nSupporting evidence:")
            answer_parts.extend(chunk_content[:2])  # Top 2 chunks
        
        return " ".join(answer_parts)


def create_mcp_knowledge_tools(blackboard: BlackboardSystem) -> List[MCPTool]:
    """Create MCP tools for knowledge sources"""
    tools = []
    
    # Create and register knowledge sources
    entity_source = EntityKnowledgeSource()
    relationship_source = RelationshipKnowledgeSource()
    hypothesis_source = HypothesisGeneratorSource()
    solution_source = SolutionSynthesizerSource()
    
    blackboard.register_knowledge_source(entity_source)
    blackboard.register_knowledge_source(relationship_source)
    blackboard.register_knowledge_source(hypothesis_source)
    blackboard.register_knowledge_source(solution_source)
    
    # Create MCP tool for triggering knowledge contribution
    async def contribute_knowledge(source_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """MCP tool to trigger knowledge contribution from a specific source"""
        if source_name not in blackboard.knowledge_sources:
            return {
                "status": "error",
                "message": f"Unknown knowledge source: {source_name}"
            }
        
        source = blackboard.knowledge_sources[source_name]
        
        try:
            if await source.can_contribute():
                contributions = await source.contribute()
                return {
                    "status": "success",
                    "contributions": len(contributions),
                    "source": source_name
                }
            else:
                return {
                    "status": "skipped",
                    "message": f"Source {source_name} cannot contribute at this time"
                }
        
        except Exception as e:
            logger.error(f"Error in knowledge contribution: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    contribute_tool = MCPTool(
        name="blackboard.contribute_knowledge",
        handler=contribute_knowledge,
        schema={
            "type": "object",
            "properties": {
                "source_name": {
                    "type": "string",
                    "enum": [
                        "entity_knowledge_source",
                        "relationship_knowledge_source",
                        "hypothesis_generator",
                        "solution_synthesizer"
                    ],
                    "description": "Name of the knowledge source"
                }
            },
            "required": ["source_name"]
        }
    )
    tools.append(contribute_tool)
    
    # Create MCP tool for querying blackboard state
    async def query_blackboard(
        knowledge_type: Optional[str] = None,
        min_confidence: float = 0.0,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Query the current state of the blackboard"""
        try:
            # Convert string to enum if provided
            kt = None
            if knowledge_type:
                kt = KnowledgeType(knowledge_type)
            
            knowledge = await blackboard.get_knowledge(
                knowledge_type=kt,
                min_confidence=min_confidence
            )
            
            # Get additional stats
            blackboard_data = blackboard._blackboard_data or {}
            stats = {
                "total_knowledge": len(blackboard_data.get("knowledge", {})),
                "total_hypotheses": len(blackboard_data.get("hypotheses", {})),
                "total_solutions": len(blackboard_data.get("solutions", {})),
                "control_iteration": blackboard_data.get("control_state", {}).get("iteration", 0)
            }
            
            return {
                "status": "success",
                "knowledge": knowledge,
                "stats": stats
            }
        
        except Exception as e:
            logger.error(f"Error querying blackboard: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    query_tool = MCPTool(
        name="blackboard.query",
        handler=query_blackboard,
        schema={
            "type": "object",
            "properties": {
                "knowledge_type": {
                    "type": "string",
                    "enum": ["fact", "hypothesis", "inference", "constraint", "goal", "solution"],
                    "description": "Type of knowledge to query"
                },
                "min_confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Minimum confidence threshold"
                }
            }
        }
    )
    tools.append(query_tool)
    
    # Create MCP tool for running control cycle
    async def run_cycle(max_iterations: int = 10, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run the blackboard control cycle"""
        try:
            solution_id = await blackboard.run_control_cycle(max_iterations)
            
            if solution_id:
                # Get the solution details
                blackboard_data = blackboard._blackboard_data or {}
                solution = blackboard_data.get("knowledge", {}).get(solution_id, {})
                
                return {
                    "status": "solution_found",
                    "solution_id": solution_id,
                    "solution": solution
                }
            else:
                return {
                    "status": "no_solution",
                    "message": "Control cycle completed without finding a solution"
                }
        
        except Exception as e:
            logger.error(f"Error in control cycle: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    cycle_tool = MCPTool(
        name="blackboard.run_cycle", 
        handler=run_cycle,
        schema={
            "type": "object",
            "properties": {
                "max_iterations": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Maximum iterations for control cycle"
                }
            }
        }
    )
    tools.append(cycle_tool)
    
    return tools