# START: /home/brian/digimon/Core/AgentBrain/agent_brain.py
import json
from typing import Dict, Any, List, Tuple, Type, get_origin, get_args, Optional # Added Optional
from datetime import datetime

from pydantic import BaseModel, Field # Ensure Field is imported if used explicitly in models, though not directly here
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from Core.AgentSchema.plan import ExecutionPlan
from Core.AgentSchema.context import GraphRAGContext
from Core.AgentOrchestrator.orchestrator import AgentOrchestrator
from Core.Common.Logger import logger
from Core.Common.LoggerConfig import get_logger, log_with_context
from Option.Config2 import Config
from Config.LLMConfig import LLMConfig, LLMType # LLMType likely used by create_llm_instance
from Core.Provider.BaseLLM import BaseLLM # Import BaseLLM for type hinting
from Core.Provider.LiteLLMProvider import LiteLLMProvider # Needed for isinstance check
from Core.Provider.LLMProviderRegister import create_llm_instance

# Use structured logging
logger = get_logger(__name__)

# Import all tool contract models
from Core.AgentSchema import tool_contracts
from Core.AgentSchema import graph_construction_tool_contracts
from Core.AgentSchema import corpus_tool_contracts
import inspect

# Note: The old import for OpenAILLM from Core.Provider.OpenaiApi is no longer used directly by PlanningAgent
# as LLM interaction is through the BaseLLM interface via self.llm_provider.


def _format_pydantic_model_for_prompt(model_cls: Type[BaseModel]) -> str:
    """
    Formats a Pydantic model's schema into a human-readable string for LLM prompts.
    """
    if not model_cls:
        return "  Schema: Not defined.\n"

    schema_parts = []
    for field_name, field_info in model_cls.model_fields.items():
        field_type = field_info.annotation
        description = field_info.description or ""

        type_str = str(field_type)
        if 'typing.' in type_str:
            type_str = type_str.replace('typing.', '')
        type_str = type_str.replace("<class '", "").replace("'>","")

        default_value_str = ""
        if field_info.default is not PydanticUndefined:
            default_value_str = f" (default: {field_info.default})"
        elif field_info.default_factory is not None:
            default_value_str = f" (default_factory exists)"

        required_str = " (required)" if field_info.is_required() else ""

        schema_parts.append(f"    - {field_name} ({type_str}){required_str}{default_value_str}: {description}")

    if not schema_parts:
        return "  Schema: Contains no fields.\n"
    return "\n".join(schema_parts) + "\n"

class PlanningAgent:
    def __init__(self, config: Config, graphrag_context: GraphRAGContext = None):
        self.config: Config = config
        self.graphrag_context: Optional[GraphRAGContext] = graphrag_context # Use Optional

        # LLM Provider setup first since it's needed for Orchestrator
        self.llm_provider: Optional[BaseLLM] = None # Type hint to Optional[BaseLLM]
        if self.config.llm:
            try:
                self.llm_provider = create_llm_instance(self.config.llm)
                logger.info(f"PlanningAgent initialized with LLM provider for api_type: {self.config.llm.api_type}, model: {self.config.llm.model}")
            except Exception as e:
                logger.error(f"Failed to create LLM instance via factory for api_type {self.config.llm.api_type}: {e}", exc_info=True)
        else:
            logger.warning("LLM configuration (self.config.llm) is missing. LLM provider not created.")

        if not self.llm_provider:
            logger.error("LLM provider is NOT initialized in PlanningAgent. Plan generation will fail.")
            
        # Get encoder instance from GraphRAGContext if available
        encoder_instance = None
        chunk_factory = None
        if graphrag_context:
            # Extract encoder and chunk_factory from graphrag_context if available
            if hasattr(graphrag_context, 'embedding_provider'):
                encoder_instance = graphrag_context.embedding_provider
            if hasattr(graphrag_context, 'chunk_storage_manager'):
                chunk_factory = graphrag_context.chunk_storage_manager
            
            # Initialize the orchestrator with all required dependencies
            self.orchestrator: Optional[AgentOrchestrator] = AgentOrchestrator(
                main_config=self.config,
                llm_instance=self.llm_provider,
                encoder_instance=encoder_instance,
                chunk_factory=chunk_factory,
                graphrag_context=graphrag_context
            )
        else:
            self.orchestrator = None
            logger.warning("PlanningAgent initialized without GraphRAGContext. Orchestrator will be None and execution will fail.")

        # LLM Provider is already initialized at the beginning of __init__

    def _get_system_task_prompt_for_planning(self, tool_documentation: str, actual_corpus_name: str) -> str:
        """
        Returns the system task prompt for planning with tool documentation embedded.
        """
        base_prompt = f"""You are an expert at creating structured execution plans for a GraphRAG system.
Based on the user's query, generate an ExecutionPlan JSON that orchestrates the appropriate tools.

## CRITICAL REQUIREMENTS:
- **TEXT RETRIEVAL IS MANDATORY**: After finding entities or relationships, you MUST ALWAYS include a text retrieval step using either:
  - Chunk.GetTextForEntities: To get the actual text content about discovered entities
  - Chunk.FromRelationships: To get text describing relationships between entities
- **NEVER** return entity names without their associated text content!

## Guidelines:
1. **Corpus and Graph**: Most queries need corpus preparation and ER graph building as initial steps
2. **Entity Search**: For informational queries, use Entity.VDBSearch to find relevant entities
3. **TEXT RETRIEVAL**: ALWAYS follow entity/relationship discovery with Chunk.GetTextForEntities or Chunk.FromRelationships
4. **Data References**: 
   - Plan inputs: Use `"plan_inputs.main_query"` string format for user query reference
   - Step outputs: Use `{{"from_step_id": "step_id", "named_output_key": "alias"}}` for step references
   - Named outputs: `{{"alias": "actual_tool_field"}}` (alias as key, actual field name as value)
   - **CRITICAL named_output_key Rule**: When referencing a step's output:
     * IF the source step's `named_outputs` defines an alias for the tool's output field (e.g., `named_outputs: {{"my_alias": "actual_tool_field"}}`), 
       THEN you MUST use `my_alias` as the `named_output_key`.
     * ELSE (if `named_outputs` is empty OR doesn't define an alias for the needed field),
       THEN use the tool's original Pydantic output field name as the `named_output_key`.
     * NEVER use the original field name when an alias is defined in the source step's `named_outputs`.
4. **Required Format**: Output must be valid JSON ExecutionPlan with `plan_id`, `plan_description`, `target_dataset_name`, `plan_inputs`, and `steps`.

## Example ExecutionPlan:
{{
  "plan_id": "informational_query_plan",
  "plan_description": "Comprehensive information retrieval using VDB search and graph exploration",
  "target_dataset_name": "{actual_corpus_name}",
  "plan_inputs": {{"main_query": "tell me about the american revolution?"}},
  "steps": [
    {{
      "step_id": "step_1_prepare_corpus",
      "description": "Prepare the corpus from the specified directory",
      "action": {{
        "tools": [{{
          "tool_id": "corpus.PrepareFromDirectory",
          "inputs": {{
            "input_directory_path": "Data/{actual_corpus_name}",
            "output_directory_path": "results/{actual_corpus_name}/corpus", 
            "target_corpus_name": "{actual_corpus_name}"
          }},
          "named_outputs": {{
            "prepared_corpus_name": "corpus_json_path"
          }}
        }}]
      }}
    }},
    {{
      "step_id": "step_2_build_er_graph",
      "description": "Build ER graph from corpus",
      "action": {{
        "tools": [{{
          "tool_id": "graph.BuildERGraph",
          "inputs": {{
            "target_dataset_name": "{actual_corpus_name}"
          }},
          "named_outputs": {{
            "er_graph_id": "graph_id"
          }}
        }}]
      }}
    }},
    {{
      "step_id": "step_3_build_vdb",
      "description": "Build VDB for entity search",
      "action": {{
        "tools": [{{
          "tool_id": "Entity.VDB.Build",
          "inputs": {{
            "graph_reference_id": {{"from_step_id": "step_2_build_er_graph", "named_output_key": "er_graph_id"}},
            "vdb_collection_name": "{actual_corpus_name}_entities"
          }},
          "named_outputs": {{
            "entity_vdb_id": "vdb_reference_id"
          }}
        }}]
      }}
    }},
    {{
      "step_id": "step_4_search_entities",
      "description": "Search for relevant entities",
      "action": {{
        "tools": [{{
          "tool_id": "Entity.VDBSearch",
          "inputs": {{
            "vdb_reference_id": {{"from_step_id": "step_3_build_vdb", "named_output_key": "entity_vdb_id"}},
            "query_text": "plan_inputs.main_query",
            "top_k_results": 5
          }},
          "named_outputs": {{
            "search_results": "similar_entities"
          }}
        }}]
      }}
    }},
    {{
      "step_id": "step_5_onehop",
      "description": "Retrieve one-hop related entities",
      "action": {{
        "tools": [{{
          "tool_id": "Relationship.OneHopNeighbors",
          "inputs": {{
            "graph_reference_id": {{"from_step_id": "step_2_build_er_graph", "named_output_key": "er_graph_id"}},
            "entity_ids": {{"from_step_id": "step_4_search_entities", "named_output_key": "search_results"}}
          }},
          "named_outputs": {{}}
        }}]
      }}
    }},
    {{
      "step_id": "step_6_get_text",
      "description": "Get text chunks for entities",
      "action": {{
        "tools": [{{
          "tool_id": "Chunk.GetTextForEntities",
          "inputs": {{
            "graph_reference_id": {{"from_step_id": "step_2_build_er_graph", "named_output_key": "er_graph_id"}},
            "entity_ids": {{"from_step_id": "step_4_search_entities", "named_output_key": "search_results"}}
          }},
          "named_outputs": {{}}
        }}]
      }}
    }}
  ]
}}

Generate the ExecutionPlan JSON for the user's query."""
        
        # Optionally include corpus instruction if provided
        if actual_corpus_name:
            base_prompt += f"\n\nTarget dataset: '{actual_corpus_name}'"
        
        full_prompt = base_prompt + "\n\n" + tool_documentation
        return full_prompt

    def _get_tool_documentation_for_prompt(self) -> str:
        """
        Generates a string containing the documentation for all available tools,
        derived from their Pydantic input/output models in tool_contracts.py.
        """
        docs_parts = ["## Available Tools Documentation:"]

        tools_to_document = [
            {
                "tool_id": "Entity.VDBSearch",
                "description": "Searches a Vector Database (VDB) for entities semantically similar to a natural language query. Returns a list of entity IDs and their similarity scores.",
                "inputs_model": tool_contracts.EntityVDBSearchInputs,
                "outputs_model": tool_contracts.EntityVDBSearchOutputs,
            },
            {
                "tool_id": "Entity.PPR",
                "description": "Runs Personalized PageRank (PPR) on a graph. It starts from a list of seed entity IDs and returns a ranked list of entities based on their PPR scores.",
                "inputs_model": tool_contracts.EntityPPRInputs,
                "outputs_model": tool_contracts.EntityPPROutputs,
            },
            {
                "tool_id": "Relationship.OneHopNeighbors",
                "description": "Retrieves all directly connected (one-hop) relationships and neighboring entities for a given list of source entity IDs from the graph. Allows specifying relationship direction and types.",
                "inputs_model": tool_contracts.RelationshipOneHopNeighborsInputs,
                "outputs_model": tool_contracts.RelationshipOneHopNeighborsOutputs,
            },
            {
                "tool_id": "corpus.PrepareFromDirectory",
                "description": "Processes all .txt files in a specified input directory, creates a Corpus.json (JSON Lines format) in an output directory, and returns information about the created corpus.",
                "inputs_model": corpus_tool_contracts.PrepareCorpusInputs,
                "outputs_model": corpus_tool_contracts.PrepareCorpusOutputs,
            },
            {
                "tool_id": "graph.BuildERGraph",
                "description": "Builds an Entity-Relation Graph (ERGraph) for a specified dataset. Processes text chunks to extract entities and relationships, then constructs and persists the graph.",
                "inputs_model": graph_construction_tool_contracts.BuildERGraphInputs,
                "outputs_model": graph_construction_tool_contracts.BuildERGraphOutputs,
            },
            {
                "tool_id": "graph.BuildRKGraph",
                "description": "Builds a Relation-Knowledge Graph (RKGraph) for a specified dataset. Extracts relations using keyword/entity extraction, then constructs and persists the graph.",
                "inputs_model": graph_construction_tool_contracts.BuildRKGraphInputs,
                "outputs_model": graph_construction_tool_contracts.BuildRKGraphOutputs,
            },
            {
                "tool_id": "graph.BuildTreeGraph",
                "description": "Builds a hierarchical TreeGraph for a specified dataset using clustering and dimensionality reduction.",
                "inputs_model": graph_construction_tool_contracts.BuildTreeGraphInputs,
                "outputs_model": graph_construction_tool_contracts.BuildTreeGraphOutputs,
            },
            {
                "tool_id": "graph.BuildTreeGraphBalanced",
                "description": "Builds a balanced hierarchical TreeGraph for a specified dataset, optimizing for balanced cluster sizes.",
                "inputs_model": graph_construction_tool_contracts.BuildTreeGraphBalancedInputs,
                "outputs_model": graph_construction_tool_contracts.BuildTreeGraphBalancedOutputs,
            },
            {
                "tool_id": "graph.BuildPassageGraph",
                "description": "Builds a PassageGraph for a specified dataset, linking passages based on entity annotation and prior probability threshold.",
                "inputs_model": graph_construction_tool_contracts.BuildPassageGraphInputs,
                "outputs_model": graph_construction_tool_contracts.BuildPassageGraphOutputs,
            },
            {
                "tool_id": "Chunk.GetTextForEntities",
                "description": "CRITICAL: Retrieves the actual text content/chunks associated with given entities. This tool MUST be used after entity discovery to get the actual information about entities.",
                "inputs_model": tool_contracts.ChunkGetTextForEntitiesInput,
                "outputs_model": tool_contracts.ChunkGetTextForEntitiesOutput,
            },
            {
                "tool_id": "Chunk.FromRelationships",
                "description": "Retrieves text chunks that contain specified relationships. Use this to get context about how entities are connected.",
                "inputs_model": tool_contracts.ChunkFromRelationshipsInputs,
                "outputs_model": tool_contracts.ChunkFromRelationshipsOutputs,
            },
        ]

        for tool_info in tools_to_document:
            docs_parts.append(f"\n### Tool: {tool_info['tool_id']}")
            docs_parts.append(f"  Description: {tool_info['description']}")

            docs_parts.append("  Input Schema:")
            docs_parts.append(_format_pydantic_model_for_prompt(tool_info['inputs_model']))

            docs_parts.append("  Output Schema:")
            docs_parts.append(_format_pydantic_model_for_prompt(tool_info['outputs_model']))

        return "\n".join(docs_parts)

    def _extract_json(self, text: str) -> Optional[dict]:
        """
        Extract JSON from the LLM response, handling various formats.
        """
        try:
            # First try to parse the entire text as JSON
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON within the text
            import re
            
            # Look for JSON enclosed in ```json ... ``` blocks
            json_pattern = r'```json\s*([\s\S]*?)\s*```'
            match = re.search(json_pattern, text)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass
            
            # Look for JSON starting with { and ending with }
            json_pattern = r'\{[\s\S]*\}'
            matches = re.findall(json_pattern, text)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
                    
            logger.error(f"Could not extract valid JSON from text: {text[:500]}...")
            return None

    async def _translate_nl_step_to_pydantic(
        self, 
        nl_step: str, 
        user_query: str,
        corpus_name: str,
        previous_context: Optional[dict] = None
    ) -> Optional[ExecutionPlan]:
        """
        Translate a single natural language step into an ExecutionPlan.
        This version considers previous context for better step planning.
        """
        # Get tool documentation
        tools_doc = self._get_tool_documentation_for_prompt()
        
        # Build context-aware prompt
        system_prompt = self._get_system_task_prompt_for_planning(
            tool_documentation=tools_doc,
            actual_corpus_name=corpus_name
        )

        # Add context information to the prompt
        context_info = ""
        if previous_context:
            context_info = "\n\nAvailable Context from Previous Steps:\n"
            for key, value in previous_context.items():
                if isinstance(value, dict) and "output" in value:
                    context_info += f"- {key}: Available with keys {list(value.get('output', {}).keys())}\n"
                elif isinstance(value, list):
                    context_info += f"- {key}: List with {len(value)} items\n"
                else:
                    context_info += f"- {key}: Available\n"
        
        user_prompt = f"""Task: Given the overall user query '{user_query}', translate the following single natural language step into a complete ExecutionPlan JSON object.
The ExecutionPlan must include all required fields: 'plan_description', 'target_dataset_name', and 'steps'.
The 'plan_description' should be a concise summary of this specific step: '{nl_step}'.
The 'target_dataset_name' must be '{corpus_name}'.
The 'steps' list must contain exactly one ExecutionStep, which is the translation of: '{nl_step}'.

{context_info}

Available Tools:
{tools_doc}

Generate a complete ExecutionPlan JSON object that fulfills these requirements for the single step: '{nl_step}'.
Remember to use named_output_key when referencing outputs from previous steps that are in the context.
The output MUST be a single JSON object conforming to the ExecutionPlan schema.
"""

        try:
            response = await self.llm_provider.acompletion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            plan_json = self._extract_json(response.choices[0].message.content)
            
            if plan_json:
                try:
                    # Ensure plan_json has the required fields before unpacking
                    if not all(k in plan_json for k in ["plan_description", "target_dataset_name", "steps"]):
                        logger.error(f"Extracted JSON for step '{nl_step}' is missing required ExecutionPlan fields. Got keys: {list(plan_json.keys()) if isinstance(plan_json, dict) else 'Not a dict'}. JSON: {plan_json}")
                        return None

                    # Validate that 'steps' is a list and not empty if present
                    if not isinstance(plan_json.get("steps"), list) or not plan_json.get("steps"):
                        logger.error(f"Extracted JSON for step '{nl_step}' has invalid 'steps' field (must be a non-empty list). Got: {plan_json.get('steps')}. JSON: {plan_json}")
                        return None
                        
                    single_step_plan = ExecutionPlan(**plan_json)
                    logger.info(f"Successfully translated NL step to ExecutionPlan: '{nl_step}' -> Plan ID: {single_step_plan.plan_id}, Description: {single_step_plan.plan_description}")
                    return single_step_plan
                except Exception as e_parse:
                    logger.error(f"Error parsing extracted JSON into ExecutionPlan for step '{nl_step}': {e_parse}. JSON: {plan_json}", exc_info=True)
                    return None
            else:
                logger.error(f"Failed to extract valid JSON for step: {nl_step}")
                return None
                
        except Exception as e:
            logger.error(f"Error translating NL step to Pydantic: {e}")
            return None

    async def _generate_plan_with_llm(self, system_prompt: str, user_prompt_content: str) -> Optional[ExecutionPlan]: # Use Optional
        """
        Calls the configured LLM using LiteLLMProvider's instructor completion
        to generate and parse an ExecutionPlan.
        """
        if not self.llm_provider:
            logger.error("LLM provider not initialized. Cannot call LLM for plan generation.")
            return None

        if not isinstance(self.llm_provider, LiteLLMProvider) or \
           not hasattr(self.llm_provider, 'async_instructor_completion'):
            logger.error(f"LLM provider is not LiteLLMProvider or does not support instructor completion. Type: {type(self.llm_provider)}. Cannot generate structured plan.")
            return None

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt_content}
        ]

        full_prompt_for_logging = f"System Prompt:\n{system_prompt}\n\nUser Prompt Content:\n{user_prompt_content}"
        logger.info(f"PlanningAgent: Sending messages to LLM for ExecutionPlan (approx {len(full_prompt_for_logging)} chars).")

        try:
            log_with_context(logger, "INFO", "Calling LLM for plan generation", 
                           prompt_length=len(full_prompt_for_logging),
                           model=str(self.llm_provider.model))
            
            execution_plan: Optional[ExecutionPlan] = await self.llm_provider.async_instructor_completion(
                messages=messages,
                response_model=ExecutionPlan,
                max_retries=2,
                max_tokens=4000,  # Add max_tokens to handle longer plans
            )

            if execution_plan:
                logger.info(f"PlanningAgent: Successfully received and parsed ExecutionPlan from LLM.")
                log_with_context(logger, "INFO", "Plan generation successful",
                               plan_id=execution_plan.plan_id,
                               steps_count=len(execution_plan.steps))
            else:
                logger.error(f"PlanningAgent: LLM call for plan generation returned None or failed parsing with instructor.")
            return execution_plan
        except json.JSONDecodeError as e:
            logger.error(f"PlanningAgent: JSON parsing error in plan generation: {e}", exc_info=True)
            log_with_context(logger, "ERROR", "JSON parsing failed in plan generation",
                           error_type="JSONDecodeError",
                           error_message=str(e))
            return None
        except TimeoutError as e:
            logger.error(f"PlanningAgent: Timeout during LLM call for plan generation: {e}", exc_info=True)
            log_with_context(logger, "ERROR", "LLM timeout in plan generation",
                           error_type="TimeoutError",
                           timeout_seconds=getattr(e, 'timeout', 'unknown'))
            return None
        except Exception as e:
            logger.error(f"PlanningAgent: Error during LLM call for plan generation with instructor: {e}", exc_info=True)
            log_with_context(logger, "ERROR", "Unexpected error in plan generation",
                           error_type=type(e).__name__,
                           error_message=str(e))
            return None

    async def _generate_natural_language_plan(self, user_query: str, actual_corpus_name: Optional[str] = None) -> List[str]:
        """
        Generate a high-level natural language plan for the query.
        Returns a list of natural language steps.
        """
        if not self.llm_provider:
            logger.error("LLM provider not initialized. Cannot generate natural language plan.")
            return []
            
        tool_docs = self._get_tool_documentation_for_prompt()
        
        system_prompt = """You are an expert planning agent. Create a high-level, step-by-step plan in natural language to answer the user's query.

## Available Tools (for reference):
{tool_docs}

## Planning Guidelines:
1. Break down the task into clear, sequential steps
2. Each step should be a complete sentence describing what needs to be done
3. Consider prerequisites (e.g., "Build a graph before searching it")
4. Use tool names when relevant but write in natural language
5. Return ONLY a JSON array of strings, where each string is one step

Example output format:
[
    "Prepare the corpus from the specified directory",
    "Build an entity-relationship graph from the prepared corpus",
    "Create a vector database index for the entities",
    "Search the vector database for entities matching the query",
    "Retrieve the associated text for the found entities",
    "Summarize the findings to answer the user's question"
]"""

        corpus_instruction = ""
        if actual_corpus_name:
            corpus_instruction = f"\nThe target dataset is: {actual_corpus_name}\n"
            
        user_prompt = f"""User Query: "{user_query}"{corpus_instruction}

Generate a natural language plan as a JSON array of steps."""

        messages = [
            {"role": "system", "content": system_prompt.format(tool_docs=tool_docs)},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = await self.llm_provider.acompletion(messages=messages, temperature=0.0)
            content = response.choices[0].message.content.strip()
            
            # Parse JSON array from response
            import json
            steps = json.loads(content)
            
            if isinstance(steps, list) and all(isinstance(s, str) for s in steps):
                logger.info(f"Generated natural language plan with {len(steps)} steps")
                return steps
            else:
                logger.error("Invalid natural language plan format")
                return []
                
        except Exception as e:
            logger.error(f"Error generating natural language plan: {e}")
            return []

    async def generate_plan(self, user_query: str, actual_corpus_name: Optional[str] = None) -> Optional[ExecutionPlan]: # Use Optional
        """
        Generates an ExecutionPlan based on the user_query using an LLM.
        """
        tool_docs = self._get_tool_documentation_for_prompt()

        system_prompt = self._get_system_task_prompt_for_planning(
            tool_documentation=tool_docs,
            actual_corpus_name=actual_corpus_name
        )

        # Build the user prompt content
        corpus_instruction = ""
        if actual_corpus_name:
            corpus_instruction = f"\n## IMPORTANT: The target dataset for this operation is '{actual_corpus_name}'. " \
                               f"Use '{actual_corpus_name}' as the 'target_dataset_name' field in the ExecutionPlan.\n"

        user_prompt_parts = [
            "## Available Tools:",
            tool_docs,
            corpus_instruction,
            "## User Query:",
            f'"{user_query}"',
            "",
            '## Example for "What are the main entities in my documents?":',
            """{
  "plan_id": "extract_main_entities",
  "plan_description": "Prepare corpus, build graph, create VDB, and find main entities from documents",
  "target_dataset_name": "MySampleTexts",
  "plan_inputs": { "main_query": "What are the main entities in my documents?" },
  "steps": [
    {
      "step_id": "prepare_corpus",
      "description": "Prepare corpus from the source directory",
      "action": {
        "tools": [{
          "tool_id": "corpus.PrepareFromDirectory",
          "inputs": {
            "input_directory_path": "Data/MySampleTexts",
            "output_directory_path": "results/MySampleTexts",
            "target_corpus_name": "MySampleTexts"
          },
          "named_outputs": {
            "prepared_corpus_name": "corpus_json_path"
          }
        }]
      }
    },
    {
      "step_id": "step_2_build_er_graph",
      "description": "Build ER graph from corpus",
      "action": {
        "tools": [{
          "tool_id": "graph.BuildERGraph",
          "inputs": {
            "target_dataset_name": "MySampleTexts"
          },
          "named_outputs": {
            "er_graph_id": "graph_id"
          }
        }]
      }
    },
    {
      "step_id": "step_3_build_entity_vdb",
      "description": "Build entity VDB from graph",
      "action": {
        "tools": [{
          "tool_id": "Entity.VDB.Build",
          "inputs": {
            "graph_reference_id": {"from_step_id": "step_2_build_er_graph", "named_output_key": "er_graph_id"},
            "vdb_collection_name": "MySampleTexts_entities"
          },
          "named_outputs": {
            "entity_vdb_id": "vdb_reference_id"
          }
        }]
      }
    },
    {
      "step_id": "step_4_search_entities",
      "description": "Search for relevant entities",
      "action": {
        "tools": [{
          "tool_id": "Entity.VDBSearch",
          "inputs": {
            "vdb_reference_id": {"from_step_id": "step_3_build_entity_vdb", "named_output_key": "entity_vdb_id"},
            "query_text": "plan_inputs.main_query",
            "top_k_results": 5
          },
          "named_outputs": {
            "search_results": "similar_entities"
          }
        }]
      }
    },
    {
      "step_id": "step_5_onehop",
      "description": "Retrieve one-hop related entities",
      "action": {
        "tools": [{
          "tool_id": "Relationship.OneHopNeighbors",
          "inputs": {
            "graph_reference_id": {"from_step_id": "step_2_build_er_graph", "named_output_key": "er_graph_id"},
            "entity_ids": {"from_step_id": "step_4_search_entities", "named_output_key": "search_results"}
          },
          "named_outputs": {}
        }]
      }
    },
    {
      "step_id": "step_6_get_text",
      "description": "Get text chunks for entities",
      "action": {
        "tools": [{
          "tool_id": "Chunk.GetTextForEntities",
          "inputs": {
            "graph_reference_id": {"from_step_id": "step_2_build_er_graph", "named_output_key": "er_graph_id"},
            "entity_ids": {"from_step_id": "step_4_search_entities", "named_output_key": "search_results"}
          },
          "named_outputs": {}
        }]
      }
    }
  ]
}""",
            "",
            f'Now, based on the user query: "{user_query}" and the available tools, generate the JSON ExecutionPlan.',
            "Return ONLY the JSON plan object, starting with '{' and ending with '}'. Do not include any other text before or after the JSON."
        ]
        
        user_prompt_content = "\n".join(user_prompt_parts)
        
        execution_plan = await self._generate_plan_with_llm(system_prompt=system_prompt, user_prompt_content=user_prompt_content)

        if execution_plan:
            logger.info(f"PlanningAgent: Successfully generated ExecutionPlan object directly via instructor.")
            logger.info(f"Generated Plan (Pydantic model):\n{execution_plan.model_dump_json(indent=2)}")
        else:
            logger.error("PlanningAgent: Failed to generate ExecutionPlan object with LLM and instructor.")

        return execution_plan

    async def process_query(self, user_query: str, actual_corpus_name: Optional[str] = None) -> Any | None:
        """
        Takes a natural language query, generates a plan, executes it,
        generates a natural language answer based on the retrieved context,
        and returns the answer and context.
        """
        if not self.orchestrator:
            logger.error("Orchestrator not initialized in PlanningAgent. Cannot process query.")
            return {"error": "Orchestrator not initialized.", "generated_answer": "Error: Orchestrator not initialized."}

        if not self.llm_provider:
            logger.error("LLM Provider not initialized in PlanningAgent. Cannot generate plan or answer.")
            return {"error": "LLM Provider not initialized.", "generated_answer": "Error: LLM Provider not initialized."}

        logger.info(f"PlanningAgent: Processing query: {user_query} with corpus: {actual_corpus_name}")
        execution_plan = await self.generate_plan(user_query, actual_corpus_name)

        retrieved_context = None
        generated_answer = "No answer generated." # Default message
        final_result = {}

        if execution_plan:
            logger.info(f"PlanningAgent: Executing generated plan ID: {execution_plan.plan_id}")
            try:
                current_plan: Optional[ExecutionPlan] = execution_plan
                orchestrator_step_outputs: Dict[str, Any] = {}
                serializable_context_for_prompt: Dict[str, Any] = {}
                
                log_with_context(logger, "INFO", "Starting plan execution",
                               plan_id=execution_plan.plan_id,
                               total_steps=len(execution_plan.steps))
                
                retrieved_context = await self.orchestrator.execute_plan(plan=execution_plan)
                logger.info(f"PlanningAgent: Plan execution finished. Retrieved context: {retrieved_context}")
                
                log_with_context(logger, "INFO", "Plan execution completed",
                               plan_id=execution_plan.plan_id,
                               context_keys=list(retrieved_context.keys()) if retrieved_context else [])
                final_result["retrieved_context"] = retrieved_context

                is_retrieval_error = isinstance(retrieved_context, dict) and retrieved_context.get("error")

                if retrieved_context and not is_retrieval_error:
                    original_user_query = current_plan.plan_inputs.get("main_query", "the user's query")
                    all_step_outputs = self.orchestrator.step_outputs

                    logger.debug(f"PlanningAgent: Full retrieved context from orchestrator: {all_step_outputs}")

                    # Build comprehensive context from all steps
                    prompt_context_summary = ""
                    
                    # 1. VDB Search Results
                    found_entities_data = None
                    search_step_id_in_plan = None
                    search_results_alias_in_plan = None

                    for step_definition in current_plan.steps:
                        if step_definition.action and step_definition.action.tools:
                            for tool_call_def in step_definition.action.tools:
                                if tool_call_def.tool_id == "Entity.VDBSearch":
                                    search_step_id_in_plan = step_definition.step_id
                                    if tool_call_def.named_outputs:
                                        # Find the alias for 'similar_entities' field
                                        # named_outputs maps alias -> field_name, so we need to find where value is 'similar_entities'
                                        for alias, pydantic_field in tool_call_def.named_outputs.items():
                                            if pydantic_field == "similar_entities":
                                                search_results_alias_in_plan = alias
                                                break
                                    break
                            if search_step_id_in_plan and search_results_alias_in_plan:
                                break

                    if search_step_id_in_plan and search_results_alias_in_plan:
                        search_step_actual_outputs = all_step_outputs.get(search_step_id_in_plan, {})
                        logger.debug(f"PlanningAgent: Outputs from VDB search step ('{search_step_id_in_plan}'): {search_step_actual_outputs}")
                        found_entities_data = search_step_actual_outputs.get(search_results_alias_in_plan)
                    else:
                        logger.warning("PlanningAgent: Could not find 'Entity.VDBSearch' step or its 'similar_entities' named output alias in the plan.")

                    logger.debug(f"PlanningAgent: Extracted 'found_entities_data' for final prompt: {found_entities_data}")

                    # 2. One-hop Relationships
                    onehop_step_outputs = None
                    for step_id, outputs in all_step_outputs.items():
                        if "onehop" in step_id.lower():
                            onehop_step_outputs = outputs
                            break
                    
                    if onehop_step_outputs and isinstance(onehop_step_outputs, dict):
                        relationships = onehop_step_outputs.get('one_hop_relationships', onehop_step_outputs.get('one_hop_neighbors', []))
                        if relationships:
                            prompt_context_summary += "**Related Information (Graph Relationships):**\n"
                            for rel in relationships[:10]:  # Limit to first 10 relationships
                                if isinstance(rel, dict):
                                    src = rel.get('src_id', 'Unknown')
                                    tgt = rel.get('tgt_id', 'Unknown')
                                    desc = rel.get('description', '')
                                    if desc:
                                        prompt_context_summary += f"- {src} → {tgt}: {desc}\n"
                                    else:
                                        prompt_context_summary += f"- {src} → {tgt}\n"
                            prompt_context_summary += "\n"
                    
                    # 3. Text Chunks
                    text_step_outputs = None
                    retrieved_chunks = []
                    
                    # Look for retrieved chunks in any step
                    for step_id, outputs in all_step_outputs.items():
                        if isinstance(outputs, dict):
                            # Check for 'retrieved_chunks' key
                            if 'retrieved_chunks' in outputs:
                                chunks = outputs['retrieved_chunks']
                                if isinstance(chunks, list):
                                    retrieved_chunks.extend(chunks)
                                    logger.debug(f"Found {len(chunks)} chunks in step '{step_id}'")
                    
                    # If we found chunks, use them
                    if retrieved_chunks:
                        text_step_outputs = {'retrieved_chunks': retrieved_chunks}
                    
                    if text_step_outputs and isinstance(text_step_outputs, dict):
                        # Handle different possible output formats
                        text_content = text_step_outputs.get('retrieved_chunks', text_step_outputs.get('relevant_chunks', text_step_outputs.get('text_chunks', text_step_outputs.get('chunks', []))))
                        if isinstance(text_content, list) and text_content:
                            prompt_context_summary += "**Relevant Text Content:**\n"
                            for i, chunk in enumerate(text_content[:3]):  # Limit to first 3 chunks
                                if isinstance(chunk, dict):
                                    content = chunk.get('text_content', chunk.get('content', chunk.get('text', str(chunk))))
                                elif hasattr(chunk, 'text_content'):
                                    content = chunk.text_content
                                elif hasattr(chunk, 'content'):
                                    content = chunk.content
                                else:
                                    content = str(chunk)
                                prompt_context_summary += f"[Chunk {i+1}]: {content[:500]}...\n\n"  # Limit each chunk to 500 chars
                        elif isinstance(text_content, str) and text_content:
                            prompt_context_summary += f"**Relevant Text Content:**\n{text_content[:1500]}...\n\n"
                        else:
                            prompt_context_summary += "**Relevant Text Content:** No text chunks found. Please use one-hop relationships for context.\n"
                    
                    if found_entities_data and isinstance(found_entities_data, list) and len(found_entities_data) > 0:
                        prompt_context_summary += "**Relevant Entities Found:**\n"
                        for item_data in found_entities_data:
                            entity_name = "Unknown Entity"
                            score_val = "N/A"
                            if isinstance(item_data, dict):
                                entity_name = item_data.get('entity_name', item_data.get('node_id'))
                                score_val = item_data.get('score')
                            elif hasattr(item_data, 'entity_name'):
                                entity_name = getattr(item_data, 'entity_name', getattr(item_data, 'node_id', None))
                                score_val = getattr(item_data, 'score', None)
                            
                            score_str = f"{score_val:.3f}" if isinstance(score_val, float) else str(score_val)
                            prompt_context_summary += f"- {entity_name} (Similarity: {score_str})\n"
                        prompt_context_summary += "\n"
                    
                    if not prompt_context_summary:
                        prompt_context_summary = "No specific entities or information were found that seem relevant to the query."
                    else:
                        prompt_context_summary += "\nPlease synthesize the above information to answer the user's query."
                    
                    system_prompt = f"""You are a helpful assistant. Your task is to answer the user's query based *only* on the 'Retrieved Context' provided below.
User's Query: "{original_user_query}"

Retrieved Context:
{prompt_context_summary}

Instructions:
1. If text content is available, use it as the primary source for your answer.
2. If text content is not available but entity relationships are shown, synthesize an answer from the entity names and their relationships.
3. Look for entities that directly answer the query - for example, if asked about causes, look for entities named after causes or events.
4. Pay special attention to relationships between entities as they often contain the answer.
5. Do NOT use any general knowledge. Base your answer ONLY on the provided context.
6. If the context doesn't contain information to answer the query, explicitly state that.

Based on the entities and relationships found, provide your answer."""

                    messages = [{"role": "system", "content": system_prompt}]
                    # You might add: messages.append({"role": "user", "content": original_user_query}) if your LLM provider/model benefits from it.

                    logger.info("PlanningAgent: Generating final natural language answer.")
                    logger.debug(f"PlanningAgent: Data for final LLM prompt: {found_entities_data}")
                    
                    try:
                        llm_response = await self.llm_provider.acompletion(
                            messages=messages
                        )
                        generated_answer = self.llm_provider.get_choice_text(llm_response)
                        logger.info(f"PlanningAgent: Generated answer: {generated_answer}")
                    except Exception as e:
                        logger.error(f"PlanningAgent: Error during answer generation LLM call: {e}", exc_info=True)
                        final_result["generation_error"] = str(e)
                        generated_answer = "Error: Could not generate an answer due to an internal issue during LLM call."
                
                elif is_retrieval_error:
                    logger.warning(f"PlanningAgent: Context retrieval resulted in an error: {retrieved_context.get('error')}. Skipping answer generation.")
                    generated_answer = f"Could not generate an answer because context retrieval failed: {retrieved_context.get('error')}"
                else: # Context is None or empty but not an error dict
                    logger.info("PlanningAgent: Retrieved context is empty or not suitable for answer generation. Skipping.")
                    generated_answer = "Could not find specific information to answer the query based on the retrieved context."

            except Exception as e: # This catches errors from orchestrator.execute_plan OR the context serialization
                logger.error(f"PlanningAgent: Exception during plan execution or context processing: {e}", exc_info=True)
                final_result["execution_error"] = str(e)
                if retrieved_context is None:
                    retrieved_context = {"error": f"Exception during plan execution or context processing: {str(e)}"}
                # If retrieved_context was successfully fetched but serialization failed, it's already in final_result
                # Ensure final_result["retrieved_context"] reflects the state
                if "retrieved_context" not in final_result:
                     final_result["retrieved_context"] = retrieved_context if retrieved_context is not None else {"error": str(e)}

                generated_answer = "Error: Could not process context for answer generation due to an exception."
        else:
            logger.error("PlanningAgent: Failed to generate a valid execution plan. Cannot execute or generate answer.")
            final_result["error"] = "Failed to generate a valid execution plan."
            generated_answer = "Error: Could not generate an execution plan."

        final_result["generated_answer"] = generated_answer
        return final_result

    async def process_query_react(self, query: str, actual_corpus_name: str = None) -> dict:
        """
        Process a query using the ReAct (Reason-Act-Observe) paradigm.
        Iteratively reasons about the query, takes actions, observes results, and decides next steps.
        """
        logger.info(f"PlanningAgent (ReAct): Processing query: {query}")
        
        # Initialize ReAct state
        react_state = {
            "original_query": query,
            "current_context": {},
            "observations": [],
            "executed_steps": [],
            "remaining_steps": [],
            "iterations": 0,
            "max_iterations": 10,
            "corpus_name": actual_corpus_name or (self.graphrag_context.target_dataset_name if self.graphrag_context else None),
            "final_answer": None,
            "reasoning_history": []
        }
        
        # Check what's already available in the context
        available_resources = {
            "corpus_prepared": False,
            "graphs": [],
            "vdbs": []
        }
        
        if self.graphrag_context:
            try:
                available_resources["graphs"] = self.graphrag_context.list_graphs()
                logger.debug(f"ReAct: Fetched available graphs: {available_resources['graphs']}")
            except Exception as e:
                logger.error(f"ReAct: Error fetching available graphs: {e}", exc_info=True)
                available_resources["graphs"] = []
                
            try:
                available_resources["vdbs"] = self.graphrag_context.list_vdbs()
                logger.debug(f"ReAct: Fetched available VDBs: {available_resources['vdbs']}")
            except Exception as e:
                logger.error(f"ReAct: Error fetching available VDBs: {e}", exc_info=True)
                available_resources["vdbs"] = []
        
        # Check if corpus is prepared (Results directory exists)
        import os
        corpus_results_path = f"Results/{actual_corpus_name}" if actual_corpus_name else None
        if corpus_results_path and os.path.exists(corpus_results_path):
            available_resources["corpus_prepared"] = True
            
        # Add available resources to initial context
        react_state["current_context"]["available_resources"] = available_resources
        logger.info(f"Available resources: {available_resources}")
        
        # Generate initial natural language plan with context about available resources
        nl_plan_prompt = f"""Given the query: "{query}"

Available resources:
- Corpus prepared: {available_resources['corpus_prepared']}
- Existing graphs: {available_resources['graphs']}
- Existing VDBs: {available_resources['vdbs']}
- Target corpus name: {actual_corpus_name or 'Not specified'}

Generate a step-by-step plan to answer this query. If resources already exist (corpus prepared, graphs built, etc.), 
start from the appropriate point rather than recreating them."""
        
        nl_plan = await self._generate_natural_language_plan(nl_plan_prompt)
        
        if not nl_plan:
            return {"error": "Failed to generate natural language plan."}
            
        logger.info(f"ReAct: Generated initial NL plan with {len(nl_plan)} steps:")
        for i, step in enumerate(nl_plan):
            logger.info(f"  {i+1}. {step}")
        
        react_state["initial_plan"] = nl_plan
        react_state["remaining_steps"] = list(nl_plan)  # Create a copy
        
        # ReACT loop - execute steps one at a time with reasoning
        max_iterations = 10  # Prevent infinite loops
        iteration = 0
        
        while iteration < max_iterations and not react_state["final_answer"]:
            iteration += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"ReAct Iteration {iteration}")
            logger.info(f"{'='*60}")
            
            # THINK: Reason about what to do next
            reasoning = await self._react_reason(react_state)
            react_state["reasoning_history"].append(reasoning)
            logger.info(f"ReAct REASONING: {reasoning['thought']}")
            
            if reasoning["should_answer"]:
                # Generate final answer based on current context
                logger.info("ReAct: Sufficient information gathered. Generating final answer.")
                final_answer = await self._generate_final_answer(
                    query,
                    react_state["current_context"],
                    react_state["observations"]
                )
                react_state["final_answer"] = final_answer
                break
                
            if reasoning["should_stop"]:
                logger.info("ReAct: Stopping due to reasoning decision.")
                break
                
            # ACT: Execute the next step
            next_step = reasoning.get("next_step")
            if not next_step:
                logger.warning("ReAct: No next step determined. Stopping.")
                break
                
            logger.info(f"ReAct ACTION: Executing step: {next_step}")
            
            # Check if the step requires tool execution
            is_tool_step = any(tool in next_step for tool in ["corpus.", "graph.", "Entity.", "Relationship.", "Chunk."]) or next_step.strip().startswith("Use ")
            
            if is_tool_step:
                # Translate the step to executable format
                step_plan = await self._translate_nl_step_to_pydantic(
                    next_step,
                    query,
                    react_state["corpus_name"],
                    previous_context=react_state["current_context"]
                )
                
                if not step_plan:
                    logger.error(f"ReAct: Failed to translate step: {next_step}")
                    react_state["observations"].append({
                        "step": next_step,
                        "error": "Failed to translate to executable format",
                        "success": False
                    })
                    continue
                    
                # Execute the step
                try:
                    results = await self.orchestrator.execute_plan(step_plan)
                    
                    # Extract and observe results
                    observation = await self._react_observe(next_step, results)
                    react_state["observations"].append(observation)
                    
                    # Update executed steps
                    react_state["executed_steps"].append(next_step)
                    
                    # Update context with results (results is a dict of step outputs)
                    if results and isinstance(results, dict):
                        for step_id, step_output in results.items():
                            if step_output:
                                react_state["current_context"][step_id] = {
                                    "output": step_output,
                                    "status": "success"
                                }
                                
                except Exception as e:
                    logger.error(f"ReAct: Error executing step: {e}")
                    react_state["observations"].append({
                        "step": next_step,
                        "error": str(e),
                        "success": False
                    })
            else:
                # Non-tool step (e.g., asking user, manual analysis)
                logger.info(f"ReAct: Non-tool step identified: {next_step}")
                react_state["observations"].append({
                    "step": next_step,
                    "note": "This step requires manual intervention or is not a tool execution",
                    "success": False
                })
                react_state["executed_steps"].append(next_step)
                
            # Remove executed step from remaining steps
            if next_step in react_state["remaining_steps"]:
                react_state["remaining_steps"].remove(next_step)
                
        # If we didn't generate a final answer yet, do it now
        if not react_state["final_answer"]:
            logger.info("ReAct: Max iterations reached or stopped. Generating answer with available context.")
            react_state["final_answer"] = await self._generate_final_answer(
                query,
                react_state["current_context"],
                react_state["observations"]
            )
            
        # Return comprehensive results
        return {
            "react_mode": True,
            "generated_answer": react_state["final_answer"],
            "retrieved_context": react_state["current_context"],
            "initial_plan": react_state["initial_plan"],
            "executed_steps": react_state["executed_steps"],
            "observations": react_state["observations"],
            "reasoning_history": react_state["reasoning_history"],
            "iterations": iteration
        }
    
    async def _react_reason(self, react_state: dict) -> dict:
        """
        Reason about the current state and decide what to do next.
        Returns a reasoning decision including whether to continue, what step to take, etc.
        """
        system_prompt = """You are a reasoning agent in a ReAct (Reason-Act-Observe) loop.
Based on the current state, decide what to do next.

Your options:
1. Execute the next planned step
2. Skip a planned step if it's no longer needed
3. Create a new step based on observations
4. Decide we have enough information to answer the query
5. Stop due to errors or inability to proceed

Respond in JSON format:
{
    "thought": "Your reasoning about the current situation",
    "should_answer": true/false,  // Do we have enough info to answer?
    "should_stop": true/false,     // Should we stop trying?
    "next_step": "Description of next step to execute. If this step involves calling a tool, it MUST start with 'Use ' followed by the general action, e.g., 'Use search_vector_db with index_name=Fictional_Test_entities, query=Zorathian Empire' or 'Use graph.GetOneHopNeighbors for entity_id=some_id'.",  // null if should_answer or should_stop
    "reasoning": "Why this decision makes sense"
}"""
        
        user_prompt = f"""Current State:
Original Query: {react_state['original_query']}
Executed Steps: {json.dumps(react_state['executed_steps'], indent=2)}
Remaining Planned Steps: {json.dumps(react_state['remaining_steps'], indent=2)}

Recent Observations:
{json.dumps(react_state['observations'][-3:] if react_state['observations'] else [], indent=2)}

Context Keys Available: {list(react_state['current_context'].keys())}

What should we do next?"""
        
        response = await self.llm_provider.acompletion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        try:
            # Extract JSON from response
            content = response.choices[0].message.content
            # Find JSON block in response
            import re
            json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
            if json_match:
                reasoning = json.loads(json_match.group())
            else:
                # Fallback if no JSON found
                reasoning = {
                    "thought": content,
                    "should_answer": False,
                    "should_stop": False,
                    "next_step": react_state['remaining_steps'][0] if react_state['remaining_steps'] else None,
                    "reasoning": "Continuing with planned steps"
                }
        except Exception as e:
            logger.error(f"Error parsing reasoning response: {e}")
            # Fallback reasoning
            reasoning = {
                "thought": "Error in reasoning, continuing with plan",
                "should_answer": False,
                "should_stop": False,
                "next_step": react_state['remaining_steps'][0] if react_state['remaining_steps'] else None,
                "reasoning": "Fallback to planned execution"
            }
            
        return reasoning
    
    async def _react_observe(self, step: str, results: Any) -> dict:
        """
        Process the results of a step execution into an observation.
        
        Args:
            step: The natural language step that was executed
            results: Results from orchestrator (can be dict or object)
        """
        # Handle both dict and object formats
        if hasattr(results, '__dict__') and not isinstance(results, dict):
            # Convert object to dict if needed
            results_dict = results.__dict__ if hasattr(results, '__dict__') else {}
        else:
            results_dict = results if isinstance(results, dict) else {}
            
        observation = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "success": True,
            "data_keys": list(results_dict.keys()) if results_dict else [],
            "summary": ""
        }
        
        # Generate a summary based on the results
        summaries = []
        
        # Check each step result
        for step_id, step_result in results_dict.items():
            if isinstance(step_result, dict):
                if "error" in step_result:
                    observation["success"] = False
                    summaries.append(f"{step_id}: Error - {step_result['error']}")
                elif "message" in step_result:
                    summaries.append(f"{step_id}: {step_result['message']}")
                else:
                    # Summarize the output
                    output_summary = self._summarize_output(step_result)
                    summaries.append(f"{step_id}: {output_summary}")
            else:
                summaries.append(f"{step_id}: {str(step_result)[:100]}")
                    
        observation["summary"] = "; ".join(summaries) if summaries else "No output"
        logger.info(f"ReAct OBSERVATION: {observation['summary']}")
        return observation
        
    def _summarize_output(self, output: Any) -> str:
        """Helper to summarize step output for observations."""
        if isinstance(output, dict):
            if "message" in output:
                return output["message"]
            elif "results" in output:
                return f"{len(output['results'])} results"
            elif "entities" in output:
                return f"{len(output['entities'])} entities"
            elif "relationships" in output:
                return f"{len(output['relationships'])} relationships"
            else:
                return f"{len(output)} fields"
        elif isinstance(output, list):
            return f"{len(output)} items"
        else:
            return str(output)[:100]
    
    async def _generate_final_answer(
        self, 
        query: str, 
        context: dict, 
        observations: List[dict]
    ) -> str:
        """
        Generate a final answer based on the context and observations.
        """
        system_prompt = """You are a helpful assistant. Your task is to answer the user's query based *only* on the 'Retrieved Context' provided below.
User's Query: "{query}"

Retrieved Context:
{context}

Observations:
{observations}

Instructions:
1. If text content is available, use it as the primary source for your answer.
2. If text content is not available but entity relationships are shown, synthesize an answer from the entity names and their relationships.
3. Look for entities that directly answer the query - for example, if asked about causes, look for entities named after causes or events.
4. Pay special attention to relationships between entities as they often contain the answer.
5. Do NOT use any general knowledge. Base your answer ONLY on the provided context.
6. If the context doesn't contain information to answer the query, explicitly state that.

Based on the entities and relationships found, provide your answer."""

        messages = [{"role": "system", "content": system_prompt.format(query=query, context=context, observations=observations)}]
        
        try:
            response = await self.llm_provider.acompletion(messages=messages)
            answer = self.llm_provider.get_choice_text(response)
            return answer
        except Exception as e:
            logger.error(f"Error generating final answer: {e}")
            return "Error: Could not generate a final answer."

# END: /home/brian/digimon/Core/AgentBrain/agent_brain.py