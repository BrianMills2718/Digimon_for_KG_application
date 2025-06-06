"""
Cross-Modal Integration for UKRF

Enables DIGIMON to integrate with StructGPT and Autocoder systems
via MCP protocol for unified knowledge reasoning.
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import re
from datetime import datetime

from Core.Common.Logger import logger
from Core.MCP.mcp_agent_interface import get_agent_interface
from Core.MCP.coordination_protocols import get_blackboard


class ModalityType(Enum):
    """Types of reasoning modalities"""
    GRAPH_RAG = "graph_rag"       # DIGIMON's graph-based reasoning
    STRUCT_GPT = "struct_gpt"      # Structured data reasoning
    AUTOCODER = "autocoder"        # Code understanding/generation
    

@dataclass
class EntityLink:
    """Cross-modal entity link"""
    entity_id: str
    modality: ModalityType
    local_id: str  # ID within the modality
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SchemaMapping:
    """Mapping between schemas across modalities"""
    source_schema: str
    target_schema: str
    field_mappings: Dict[str, str]  # source_field -> target_field
    transform_functions: Dict[str, str] = field(default_factory=dict)


class CrossModalBridge:
    """
    Bridge for cross-modal communication and integration.
    Handles entity linking, schema translation, and query routing.
    """
    
    def __init__(self):
        self._entity_links: Dict[str, List[EntityLink]] = {}  # entity_id -> links
        self._schema_mappings: Dict[Tuple[str, str], SchemaMapping] = {}
        self._agent_interface = get_agent_interface()
        self._blackboard = get_blackboard()
        self._link_cache: Dict[str, str] = {}  # For fast lookups
        
    async def link_entity(
        self,
        entity_name: str,
        modality1: ModalityType,
        id1: str,
        modality2: ModalityType,
        id2: str,
        confidence: float = 1.0
    ) -> str:
        """
        Link an entity across two modalities.
        
        Returns:
            str: Unified entity ID
        """
        # Generate or retrieve unified entity ID
        entity_id = self._get_unified_id(entity_name)
        
        # Create links
        link1 = EntityLink(
            entity_id=entity_id,
            modality=modality1,
            local_id=id1,
            confidence=confidence
        )
        
        link2 = EntityLink(
            entity_id=entity_id,
            modality=modality2,
            local_id=id2,
            confidence=confidence
        )
        
        # Store links
        if entity_id not in self._entity_links:
            self._entity_links[entity_id] = []
            
        self._entity_links[entity_id].extend([link1, link2])
        
        # Update cache for fast lookups
        self._link_cache[f"{modality1.value}:{id1}"] = entity_id
        self._link_cache[f"{modality2.value}:{id2}"] = entity_id
        
        # Write to blackboard for other agents
        await self._blackboard.write(
            f"entity_links/{entity_id}",
            {
                "entity_id": entity_id,
                "name": entity_name,
                "links": [
                    {"modality": modality1.value, "id": id1},
                    {"modality": modality2.value, "id": id2}
                ],
                "confidence": confidence
            }
        )
        
        logger.info(f"Linked entity '{entity_name}' across {modality1.value} and {modality2.value}")
        return entity_id
    
    def _get_unified_id(self, entity_name: str) -> str:
        """Generate unified entity ID"""
        # Simple approach: normalize and hash
        normalized = entity_name.lower().strip()
        return f"entity_{hash(normalized) % 1000000}"
    
    async def find_linked_entities(
        self,
        modality: ModalityType,
        local_id: str
    ) -> List[EntityLink]:
        """Find all linked entities for a given local ID"""
        cache_key = f"{modality.value}:{local_id}"
        
        if cache_key in self._link_cache:
            entity_id = self._link_cache[cache_key]
            return self._entity_links.get(entity_id, [])
        
        return []
    
    async def register_schema_mapping(
        self,
        source_modality: str,
        source_schema: str,
        target_modality: str,
        target_schema: str,
        field_mappings: Dict[str, str],
        transform_functions: Optional[Dict[str, str]] = None
    ):
        """Register a schema mapping between modalities"""
        mapping = SchemaMapping(
            source_schema=f"{source_modality}.{source_schema}",
            target_schema=f"{target_modality}.{target_schema}",
            field_mappings=field_mappings,
            transform_functions=transform_functions or {}
        )
        
        key = (mapping.source_schema, mapping.target_schema)
        self._schema_mappings[key] = mapping
        
        logger.info(f"Registered schema mapping: {source_schema} -> {target_schema}")
    
    async def translate_query(
        self,
        query: Dict[str, Any],
        source_modality: str,
        target_modality: str
    ) -> Dict[str, Any]:
        """
        Translate a query from one modality to another.
        
        Args:
            query: Query in source modality format
            source_modality: Source modality name
            target_modality: Target modality name
            
        Returns:
            Translated query
        """
        # Example: SQL to Graph traversal
        if source_modality == "sql" and target_modality == "graph":
            return await self._sql_to_graph(query)
        elif source_modality == "graph" and target_modality == "sql":
            return await self._graph_to_sql(query)
        else:
            # Default: pass through with metadata
            return {
                "original_query": query,
                "source": source_modality,
                "target": target_modality,
                "translated": query  # No translation available
            }
    
    async def _sql_to_graph(self, sql_query: Dict[str, Any]) -> Dict[str, Any]:
        """Translate SQL query to graph traversal"""
        # Simplified example
        if sql_query.get("type") == "SELECT":
            table = sql_query.get("from", "")
            conditions = sql_query.get("where", {})
            
            # Map table to entity type
            entity_type = self._map_table_to_entity(table)
            
            # Map conditions to graph filters
            filters = []
            for field, value in conditions.items():
                filters.append({
                    "property": field,
                    "operator": "equals",
                    "value": value
                })
            
            return {
                "type": "entity_search",
                "entity_type": entity_type,
                "filters": filters,
                "return_properties": sql_query.get("select", ["*"])
            }
        
        return sql_query
    
    async def _graph_to_sql(self, graph_query: Dict[str, Any]) -> Dict[str, Any]:
        """Translate graph query to SQL"""
        # Simplified example
        if graph_query.get("type") == "entity_search":
            entity_type = graph_query.get("entity_type", "")
            
            # Map entity to table
            table = self._map_entity_to_table(entity_type)
            
            # Build WHERE clause from filters
            where_conditions = {}
            for filter_spec in graph_query.get("filters", []):
                field = filter_spec["property"]
                value = filter_spec["value"]
                where_conditions[field] = value
            
            return {
                "type": "SELECT",
                "select": graph_query.get("return_properties", ["*"]),
                "from": table,
                "where": where_conditions
            }
        
        return graph_query
    
    def _map_table_to_entity(self, table: str) -> str:
        """Map SQL table name to entity type"""
        # Simple mapping
        mappings = {
            "users": "Person",
            "posts": "Post",
            "comments": "Comment"
        }
        return mappings.get(table, table)
    
    def _map_entity_to_table(self, entity_type: str) -> str:
        """Map entity type to SQL table"""
        # Simple mapping
        mappings = {
            "Person": "users",
            "Post": "posts",
            "Comment": "comments"
        }
        return mappings.get(entity_type, entity_type.lower())


class UnifiedQueryInterface:
    """
    Unified interface for cross-modal queries.
    Routes queries to appropriate modalities and aggregates results.
    """
    
    def __init__(self):
        self._bridge = CrossModalBridge()
        self._agent_interface = get_agent_interface()
        self._modality_agents: Dict[ModalityType, str] = {}
        
    async def register_modality(
        self,
        modality: ModalityType,
        agent_id: str
    ):
        """Register an agent for a modality"""
        self._modality_agents[modality] = agent_id
        logger.info(f"Registered {modality.value} with agent {agent_id}")
    
    async def execute_unified_query(
        self,
        query: str,
        modalities: Optional[List[ModalityType]] = None
    ) -> Dict[str, Any]:
        """
        Execute a query across multiple modalities.
        
        Args:
            query: Natural language or structured query
            modalities: List of modalities to query (default: all)
            
        Returns:
            Unified results from all modalities
        """
        if modalities is None:
            modalities = list(self._modality_agents.keys())
        
        # Parse query to determine intent
        query_intent = self._parse_query_intent(query)
        
        # Execute across modalities in parallel
        tasks = []
        for modality in modalities:
            if modality in self._modality_agents:
                task = self._query_modality(modality, query_intent)
                tasks.append(task)
        
        # Gather results
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Aggregate results
        aggregated = await self._aggregate_results(
            query_intent,
            modalities,
            results
        )
        
        return aggregated
    
    def _parse_query_intent(self, query: str) -> Dict[str, Any]:
        """Parse query to determine intent and structure"""
        # Simple parsing - in production, use NLP
        intent = {
            "original": query,
            "type": "search",
            "entities": [],
            "relationships": [],
            "filters": {}
        }
        
        # Extract entity mentions (simplified)
        entity_pattern = r'\b[A-Z][a-z]+\b'
        entities = re.findall(entity_pattern, query)
        intent["entities"] = entities
        
        # Detect relationship queries
        if any(word in query.lower() for word in ["connected", "related", "between"]):
            intent["type"] = "relationship"
        
        return intent
    
    async def _query_modality(
        self,
        modality: ModalityType,
        query_intent: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Query a specific modality"""
        agent_id = self._modality_agents.get(modality)
        if not agent_id:
            return {"error": f"No agent for {modality.value}"}
        
        # Send query to modality agent
        message_id = await self._agent_interface.send_message(
            sender_id="unified_query_interface",
            recipient_id=agent_id,
            message_type="query",
            payload={
                "query": query_intent,
                "modality": modality.value
            }
        )
        
        # In production, wait for response
        # For now, simulate response
        return {
            "modality": modality.value,
            "results": self._simulate_modality_results(modality, query_intent),
            "message_id": message_id
        }
    
    def _simulate_modality_results(
        self,
        modality: ModalityType,
        query_intent: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Simulate results from a modality"""
        if modality == ModalityType.GRAPH_RAG:
            return [
                {"entity": "Washington", "type": "Person", "relationships": 12},
                {"entity": "Jefferson", "type": "Person", "relationships": 8}
            ]
        elif modality == ModalityType.STRUCT_GPT:
            return [
                {"record_id": 1, "name": "Washington", "role": "President"},
                {"record_id": 2, "name": "Jefferson", "role": "President"}
            ]
        elif modality == ModalityType.AUTOCODER:
            return [
                {"function": "get_president_info", "references": ["Washington"]},
                {"function": "analyze_presidency", "references": ["Jefferson"]}
            ]
        return []
    
    async def _aggregate_results(
        self,
        query_intent: Dict[str, Any],
        modalities: List[ModalityType],
        results: List[Any]
    ) -> Dict[str, Any]:
        """Aggregate results from multiple modalities"""
        aggregated = {
            "query": query_intent["original"],
            "timestamp": datetime.utcnow().isoformat(),
            "modalities_queried": [m.value for m in modalities],
            "unified_results": [],
            "by_modality": {}
        }
        
        # Group by modality
        for i, modality in enumerate(modalities):
            if i < len(results) and not isinstance(results[i], Exception):
                aggregated["by_modality"][modality.value] = results[i]
        
        # Perform entity linking
        entity_map = {}
        
        for modality_name, modality_result in aggregated["by_modality"].items():
            if "results" in modality_result:
                for item in modality_result["results"]:
                    # Extract entity references
                    entity_name = None
                    if "entity" in item:
                        entity_name = item["entity"]
                    elif "name" in item:
                        entity_name = item["name"]
                    
                    if entity_name:
                        if entity_name not in entity_map:
                            entity_map[entity_name] = {
                                "entity": entity_name,
                                "occurrences": []
                            }
                        
                        entity_map[entity_name]["occurrences"].append({
                            "modality": modality_name,
                            "data": item
                        })
        
        # Convert to list
        aggregated["unified_results"] = list(entity_map.values())
        
        # Calculate statistics
        aggregated["statistics"] = {
            "total_entities": len(entity_map),
            "modalities_responded": len(aggregated["by_modality"]),
            "cross_modal_matches": sum(
                1 for e in entity_map.values() 
                if len(e["occurrences"]) > 1
            )
        }
        
        return aggregated


# Global instances
_cross_modal_bridge = None
_unified_query_interface = None


def get_cross_modal_bridge() -> CrossModalBridge:
    """Get global cross-modal bridge instance"""
    global _cross_modal_bridge
    if _cross_modal_bridge is None:
        _cross_modal_bridge = CrossModalBridge()
    return _cross_modal_bridge


def get_unified_query_interface() -> UnifiedQueryInterface:
    """Get global unified query interface"""
    global _unified_query_interface
    if _unified_query_interface is None:
        _unified_query_interface = UnifiedQueryInterface()
    return _unified_query_interface