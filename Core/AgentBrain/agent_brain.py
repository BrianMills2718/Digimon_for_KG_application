# START: /home/brian/digimon/Core/AgentBrain/agent_brain.py
import json
from typing import Dict, Any, List, Tuple, Type, get_origin, get_args, Optional # Added Optional

from pydantic import BaseModel, Field # Ensure Field is imported if used explicitly in models, though not directly here
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from Core.AgentSchema.plan import ExecutionPlan
from Core.AgentSchema.context import GraphRAGContext
from Core.AgentOrchestrator.orchestrator import AgentOrchestrator
from Core.Common.Logger import logger
from Option.Config2 import Config
from Config.LLMConfig import LLMConfig, LLMType # LLMType likely used by create_llm_instance
from Core.Provider.BaseLLM import BaseLLM # Import BaseLLM for type hinting
from Core.Provider.LiteLLMProvider import LiteLLMProvider # Needed for isinstance check
from Core.Provider.LLMProviderRegister import create_llm_instance

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

## Guidelines:
1. **Corpus and Graph**: Most queries need corpus preparation and ER graph building as initial steps
2. **Entity Search**: For informational queries, use Entity.VDBSearch to find relevant entities
3. **Data References**: 
   - Plan inputs: Use `"plan_inputs.main_query"` string format for user query reference
   - Step outputs: Use `{{"from_step_id": "step_id", "named_output_key": "alias"}}` for step references
   - Named outputs: `{{"alias": "actual_tool_field"}}` (alias as key, actual field name as value)

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
        ]

        for tool_info in tools_to_document:
            docs_parts.append(f"\n### Tool: {tool_info['tool_id']}")
            docs_parts.append(f"  Description: {tool_info['description']}")

            docs_parts.append("  Input Schema:")
            docs_parts.append(_format_pydantic_model_for_prompt(tool_info['inputs_model']))

            docs_parts.append("  Output Schema:")
            docs_parts.append(_format_pydantic_model_for_prompt(tool_info['outputs_model']))

        return "\n".join(docs_parts)

    async def _translate_nl_step_to_pydantic(
        self, 
        nl_step: str, 
        user_query: str,
        actual_corpus_name: Optional[str] = None
    ) -> Optional[ExecutionPlan]:
        """
        Translate a single natural language step into a Pydantic ExecutionPlan.
        """
        if not self.llm_provider:
            logger.error("LLM provider not initialized. Cannot translate NL step.")
            return None
            
        tool_docs = self._get_tool_documentation_for_prompt()
        
        system_prompt = """You are an expert at translating natural language instructions into precise tool execution plans.

Given a single natural language step and the available tools, create a JSON ExecutionPlan that executes ONLY that specific step.

## Rules:
1. Create a plan with exactly ONE step that implements the given natural language instruction
2. Select the most appropriate tool from the available tools
3. Set all required parameters based on context
4. Use these default IDs when relevant:
   - Graph ID: "kg_graph"
   - Entity VDB ID: "entities_vdb"
   - Relationship VDB ID: "relationships_vdb"
5. For named_outputs, use the exact field names from the tool's output model

## ExecutionPlan Schema:
{
  "plan_id": "string",
  "plan_description": "string", 
  "target_dataset_name": "string",
  "plan_inputs": { "main_query": "string" },
  "steps": [
    {
      "step_id": "string",
      "description": "string",
      "action": {
        "tools": [
          {
            "tool_id": "string",
            "description": "string",
            "parameters": { },
            "inputs": { },
            "named_outputs": { }
          }
        ]
      }
    }
  ]
}"""

        corpus_instruction = ""
        if actual_corpus_name:
            corpus_instruction = f"\nTarget dataset: {actual_corpus_name}"
            
        user_prompt = f"""Natural Language Step: "{nl_step}"
Original User Query: "{user_query}"{corpus_instruction}

Available Tools:
{tool_docs}

Translate this single step into a JSON ExecutionPlan. Return ONLY the JSON."""

        try:
            execution_plan = await self.llm_provider.instructor_async_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_model=ExecutionPlan,
                temperature=0.0
            )
            
            if execution_plan:
                logger.info(f"Successfully translated NL step to Pydantic: {nl_step}")
                return execution_plan
            else:
                logger.error(f"Failed to translate NL step: {nl_step}")
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
            execution_plan: Optional[ExecutionPlan] = await self.llm_provider.async_instructor_completion(
                messages=messages,
                response_model=ExecutionPlan,
                max_retries=2,
                max_tokens=4000,  # Add max_tokens to handle longer plans
            )

            if execution_plan:
                logger.info(f"PlanningAgent: Successfully received and parsed ExecutionPlan from LLM.")
            else:
                logger.error(f"PlanningAgent: LLM call for plan generation returned None or failed parsing with instructor.")
            return execution_plan
        except Exception as e:
            logger.error(f"PlanningAgent: Error during LLM call for plan generation with instructor: {e}", exc_info=True)
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

        system_prompt = self._get_system_task_prompt_for_planning(tool_docs, actual_corpus_name)

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
                
                retrieved_context = await self.orchestrator.execute_plan(plan=execution_plan)
                logger.info(f"PlanningAgent: Plan execution finished. Retrieved context: {retrieved_context}")
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
                    for step_id, outputs in all_step_outputs.items():
                        if "get_text" in step_id.lower() or "chunk" in step_id.lower():
                            text_step_outputs = outputs
                            break
                    
                    if text_step_outputs and isinstance(text_step_outputs, dict):
                        # Handle different possible output formats
                        text_content = text_step_outputs.get('text_chunks', text_step_outputs.get('chunks', text_step_outputs.get('text', [])))
                        if isinstance(text_content, list) and text_content:
                            prompt_context_summary += "**Relevant Text Content:**\n"
                            for i, chunk in enumerate(text_content[:3]):  # Limit to first 3 chunks
                                if isinstance(chunk, dict):
                                    content = chunk.get('content', chunk.get('text', str(chunk)))
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
                        llm_response = await self.llm_provider.acompletion(messages=messages)
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

    async def process_query_react(self, user_query: str, actual_corpus_name: Optional[str] = None) -> Any | None:
        """
        Experimental ReAct-style query processing.
        Plans in natural language, executes one step at a time, observes, and adapts.
        """
        if not self.orchestrator:
            logger.error("Orchestrator not initialized. Cannot process query.")
            return {"error": "Orchestrator not initialized."}
            
        if not self.llm_provider:
            logger.error("LLM Provider not initialized. Cannot process query.")
            return {"error": "LLM Provider not initialized."}
            
        logger.info(f"PlanningAgent (ReAct): Processing query: {user_query}")
        
        # Step 1: Generate natural language plan
        nl_plan = await self._generate_natural_language_plan(user_query, actual_corpus_name)
        if not nl_plan:
            return {"error": "Failed to generate natural language plan."}
            
        logger.info(f"ReAct: Generated NL plan with {len(nl_plan)} steps:")
        for i, step in enumerate(nl_plan):
            logger.info(f"  {i+1}. {step}")
            
        # For this initial implementation, execute only the first step
        if nl_plan:
            first_step = nl_plan[0]
            logger.info(f"ReAct: Translating first step to Pydantic: {first_step}")
            
            # Step 2: Translate first NL step to Pydantic
            single_step_plan = await self._translate_nl_step_to_pydantic(
                first_step, 
                user_query,
                actual_corpus_name
            )
            
            if not single_step_plan:
                return {"error": f"Failed to translate step: {first_step}"}
                
            logger.info(f"ReAct: Executing plan for first step")
            
            # Step 3: Execute the single step
            try:
                step_results = await self.orchestrator.execute_plan(
                    plan=single_step_plan,
                    graphrag_context=self.graphrag_context,
                    main_config=self.config
                )
                logger.info(f"ReAct: First step execution complete. Results: {step_results}")
                
                # Step 4: Basic observation and response
                # In a full implementation, we would:
                # - Analyze results
                # - Update context
                # - Decide next step
                # - Loop until goal achieved
                
                return {
                    "react_mode": True,
                    "natural_language_plan": nl_plan,
                    "executed_steps": [first_step],
                    "step_results": step_results,
                    "message": f"Executed first step: '{first_step}'. Full ReAct loop not yet implemented.",
                    "next_steps": nl_plan[1:] if len(nl_plan) > 1 else []
                }
                
            except Exception as e:
                logger.error(f"ReAct: Error executing first step: {e}")
                return {"error": f"Error executing first step: {str(e)}"}
                
        return {"error": "No steps in natural language plan."}

# END: /home/brian/digimon/Core/AgentBrain/agent_brain.py  