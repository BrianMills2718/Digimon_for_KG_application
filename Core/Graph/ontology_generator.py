"""
Dynamic ontology generation for domain-specific entity and relationship extraction.
"""
import json
from typing import Dict, Any, Optional
from Core.Common.Logger import logger
from Core.Prompt import GraphPrompt

ONTOLOGY_GENERATION_PROMPT = """Given the following context about a corpus or user query, generate a custom ontology for entity and relationship extraction.

Context: {context}

Create a JSON ontology with:
1. Entity types relevant to this domain (e.g., Person, Event, Location, etc.)
2. Relationship types that would connect these entities (e.g., PARTICIPATED_IN, CAUSED_BY, LED_TO, etc.)

The ontology should be in this exact JSON format:
{{
  "entities": [
    {{
      "name": "EntityTypeName",
      "description": "Description of this entity type"
    }}
  ],
  "relations": [
    {{
      "name": "RELATIONSHIP_NAME",
      "description": "Description of this relationship",
      "source_entity": "SourceEntityType",
      "target_entity": "TargetEntityType"
    }}
  ]
}}

Focus on domain-specific types. For example:
- For historical events: Person, Event, Location, Document, Organization with relationships like PARTICIPATED_IN, OCCURRED_AT, CAUSED_BY
- For scientific papers: Author, Paper, Institution, Topic with relationships like AUTHORED, CITES, AFFILIATED_WITH
- For business: Company, Person, Product, Market with relationships like WORKS_FOR, PRODUCES, COMPETES_WITH

Generate a focused ontology with 3-7 entity types and 5-10 relationship types most relevant to the context.
"""

async def generate_custom_ontology(context: str, llm_instance: Any) -> Optional[Dict[str, Any]]:
    """
    Generate a custom ontology based on the provided context using an LLM.
    
    Args:
        context: Description of the domain or user query
        llm_instance: LLM instance for generation
        
    Returns:
        Dictionary containing the custom ontology or None if generation fails
    """
    try:
        prompt = ONTOLOGY_GENERATION_PROMPT.format(context=context)
        logger.info(f"Generating custom ontology for context: {context[:100]}...")
        
        response = await llm_instance.aask(prompt, format="json")
        logger.debug(f"Raw ontology generation response: {response}")
        
        # Parse response - handle various response formats
        if isinstance(response, str):
            # Clean up the response string
            cleaned = response.strip()
            # Remove markdown code blocks if present
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            cleaned = cleaned.strip()
            
            # Try to parse JSON
            try:
                ontology = json.loads(cleaned)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON from string response: {cleaned[:200]}...")
                return None
        elif isinstance(response, dict):
            ontology = response
        else:
            logger.error(f"Unexpected response type: {type(response)}")
            return None
            
        # Validate basic structure
        if not isinstance(ontology, dict) or 'entities' not in ontology or 'relations' not in ontology:
            logger.error(f"Invalid ontology structure: {ontology}")
            return None
            
        logger.info(f"Generated ontology with {len(ontology['entities'])} entity types and {len(ontology['relations'])} relationship types")
        return ontology
        
    except Exception as e:
        logger.error(f"Failed to generate custom ontology: {e}")
        return None
