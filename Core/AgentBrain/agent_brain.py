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

        if graphrag_context:
            self.orchestrator: Optional[AgentOrchestrator] = AgentOrchestrator(graphrag_context=graphrag_context) # Use Optional
        else:
            self.orchestrator = None
            logger.warning("PlanningAgent initialized without GraphRAGContext. Orchestrator will be None and execution will fail.")

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
            )

            if execution_plan:
                logger.info(f"PlanningAgent: Successfully received and parsed ExecutionPlan from LLM.")
            else:
                logger.error(f"PlanningAgent: LLM call for plan generation returned None or failed parsing with instructor.")
            return execution_plan
        except Exception as e:
            logger.error(f"PlanningAgent: Error during LLM call for plan generation with instructor: {e}", exc_info=True)
            return None

    async def generate_plan(self, user_query: str) -> Optional[ExecutionPlan]: # Use Optional
        """
        Generates an ExecutionPlan based on the user_query using an LLM.
        """
        tool_docs = self._get_tool_documentation_for_prompt()

        system_prompt = """You are an expert planning agent. Your task is to create a valid JSON execution plan
to answer the user's query by selecting and orchestrating a sequence of available tools.

## Constraints:
- The output MUST be a valid JSON object that conforms to the ExecutionPlan schema provided below.
- Refer to tools by their exact `tool_id` as listed in the 'Available Tools' section.
- For tool inputs:
    - If the input value comes directly from the overall plan's inputs (like the user's main query),
      use the format: "plan_inputs.your_plan_input_key" (e.g., "plan_inputs.main_query").
    - If the input value comes from the output of a PREVIOUS step in the plan, use the ToolInputSource object format:
      {"from_step_id": "id_of_previous_step", "named_output_key": "key_of_output_from_that_step"}
- Ensure `named_outputs` for each tool call have unique and descriptive keys (e.g., "vdb_search_results", "ranked_ppr_entities"). These keys are used by subsequent steps to reference the output.
- The `target_dataset_name` in the plan should typically be "MySampleTexts".
- Provide a brief `plan_description` summarizing the plan.
- Each `step` must have a `step_id` (e.g., "step_1_vdb", "step_2_ppr") and a `description`.
- Each `tool_call` within a step's `action.tools` list must have a `tool_id` and a `description`.
- Only include parameters in the `parameters` dictionary if you are overriding their default values or if they are required and have no defaults. Do not include optional parameters if they are not needed for the query.

## ExecutionPlan Schema Outline (Simplified for understanding, actual output must be full JSON):
{
  "plan_id": "string (e.g., llm_generated_plan_unique_id)",
  "plan_description": "string",
  "target_dataset_name": "string (e.g., MySampleTexts)",
  "plan_inputs": { "main_query": "string (user's original query)" },
  "steps": [
    {
      "step_id": "string (e.g., step_1_name)",
      "description": "string",
      "action": {
        "tools": [
          {
            "tool_id": "string (e.g., Entity.VDBSearch)",
            "description": "string (tool call description)",
            "parameters": { /* tool-specific parameters, e.g., "top_k_results": 5 */ },
            "inputs": { /* mapping of tool's input fields to sources */ },
            "named_outputs": { /* mapping of output keys to descriptions, e.g., "vdb_output": "Results from VDB Search" */ }
          }
        ]
      }
    }
  ]
}
"""

        user_prompt_content = f"""
## Available Tools:
{tool_docs}

## User Query:
"{user_query}"

## Example of a 2-step plan JSON (for query: "Find entities similar to 'AI impact' and then rank them with PPR"):
{{
  "plan_id": "llm_plan_example_vdb_ppr",
  "plan_description": "Find entities with VDB for 'AI impact' and then run PPR on the results.",
  "target_dataset_name": "MySampleTexts",
  "plan_inputs": {{ "main_query": "AI impact" }},
  "steps": [
    {{
      "step_id": "step_1_vdb_search_ai",
      "description": "Perform VDB search for entities related to 'AI impact'.",
      "action": {{
        "tools": [
          {{
            "tool_id": "Entity.VDBSearch",
            "description": "Find initial relevant entities using VDB for 'AI impact'.",
            "parameters": {{
              "vdb_reference_id": "entities_vdb",
              "top_k_results": 3
            }},
            "inputs": {{
              "query_text": "plan_inputs.main_query"
            }},
            "named_outputs": {{
              "vdb_search_results_object": "Raw EntityVDBSearchOutputs object from VDB search for AI impact."
            }}
          }}
        ]
      }}
    }},
    {{
      "step_id": "step_2_ppr_on_ai_results",
      "description": "Run Personalized PageRank on entities found by VDB search.",
      "action": {{
        "tools": [
          {{
            "tool_id": "Entity.PPR",
            "description": "Calculate PPR scores for seed entities related to AI impact.",
            "parameters": {{
              "graph_reference_id": "kg_graph",
              "personalization_weight_alpha": 0.15,
              "top_k_results": 5
            }},
            "inputs": {{
              "seed_entity_ids": {{
                "from_step_id": "step_1_vdb_search_ai",
                "named_output_key": "vdb_search_results_object"
              }}
            }},
            "named_outputs": {{
              "ppr_ai_ranked_entities": "List of (entity_id, ppr_score) tuples from PPR for AI impact entities."
            }}
          }}
        ]
      }}
    }}
  ]
}}

Now, based on the user query: "{user_query}" and the available tools, generate the JSON ExecutionPlan.
Return ONLY the JSON plan object, starting with `{{` and ending with `}}`. Do not include any other text before or after the JSON.
"""

        execution_plan = await self._generate_plan_with_llm(system_prompt=system_prompt, user_prompt_content=user_prompt_content)

        if execution_plan:
            logger.info(f"PlanningAgent: Successfully generated ExecutionPlan object directly via instructor.")
            logger.info(f"Generated Plan (Pydantic model):\n{execution_plan.model_dump_json(indent=2)}")
        else:
            logger.error("PlanningAgent: Failed to generate ExecutionPlan object with LLM and instructor.")

        return execution_plan

    async def process_query(self, user_query: str) -> Any | None:
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

        logger.info(f"PlanningAgent: Processing query: {user_query}")
        execution_plan = await self.generate_plan(user_query)

        retrieved_context = None
        generated_answer = "No answer generated." # Default message
        final_result = {}

        if execution_plan:
            logger.info(f"PlanningAgent: Executing generated plan ID: {execution_plan.plan_id}")
            try:
                retrieved_context = await self.orchestrator.execute_plan(plan=execution_plan)
                logger.info(f"PlanningAgent: Plan execution finished. Retrieved context: {retrieved_context}")
                final_result["retrieved_context"] = retrieved_context

                is_retrieval_error = isinstance(retrieved_context, dict) and retrieved_context.get("error")

                if retrieved_context and not is_retrieval_error:
                    # Prepare context for JSON serialization, handling Pydantic models
                    serializable_context_for_prompt = {}
                    if isinstance(retrieved_context, dict):
                        for key, value in retrieved_context.items():
                            if isinstance(value, BaseModel):  # BaseModel from Pydantic
                                serializable_context_for_prompt[key] = value.model_dump()
                            else:
                                serializable_context_for_prompt[key] = value
                    elif isinstance(retrieved_context, BaseModel):
                        serializable_context_for_prompt = retrieved_context.model_dump()
                    elif isinstance(retrieved_context, list):
                        serializable_context_for_prompt = [
                            item.model_dump() if isinstance(item, BaseModel) else item
                            for item in retrieved_context
                        ]
                    else:
                        serializable_context_for_prompt = retrieved_context # Fallback

                    context_str_for_prompt = json.dumps(serializable_context_for_prompt, indent=2)

                    generation_prompt_messages = [
                        {
                            "role": "system",
                            "content": "You are a helpful AI assistant. Based on the provided user query and the retrieved context, generate a concise and informative answer to the user's query. If the context is empty or does not seem relevant to the query, state that you could not find specific information to answer the query based on the context."
                        },
                        {
                            "role": "user",
                            "content": f"Original User Query:\n\"\"\"\n{user_query}\n\"\"\"\n\nRetrieved Context:\n\"\"\"\n{context_str_for_prompt}\n\"\"\"\n\nPlease provide a natural language answer to the original user query based on the retrieved context."
                        }
                    ]

                    logger.info("PlanningAgent: Generating natural language answer based on retrieved context...")
                    try:
                        llm_response = await self.llm_provider.acompletion(messages=generation_prompt_messages)
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

# END: /home/brian/digimon/Core/AgentBrain/agent_brain.py