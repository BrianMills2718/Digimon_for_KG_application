# Core/AgentSchema/plan.py

import uuid
from enum import Enum
from typing import List, Optional, Dict, Any, Union, Tuple
from pydantic import BaseModel, Field

# --- Ontology Configuration ---
class OntologySourceType(str, Enum):
    """Specifies the source or method for defining the ontology for a plan."""
    DEFAULT_INTERNAL = "default_internal" # Uses GraphRAG's internal default or last globally set ontology
    FILE_PATH = "file_path"            # Specifies a path to a custom_ontology.json file
    INLINE_DEFINITION = "inline_definition"  # The LLM provides the ontology structure directly in the plan

class OntologyConfig(BaseModel):
    """Configuration for how the ontology should be set up for this execution plan."""
    source_type: OntologySourceType = OntologySourceType.DEFAULT_INTERNAL
    path_or_identifier: Optional[str] = None 
        # Description: If source_type is FILE_PATH, this is the path to the ontology file (e.g., "./Config/custom_ontology.json").
        # If source_type is INLINE_DEFINITION, this could be an optional ID for the inline definition.
    definition: Optional[Dict[str, Any]] = None 
        # Description: If source_type is INLINE_DEFINITION, this holds the actual ontology structure.

# --- Defining an Input Source for a Tool ---
class ToolInputSource(BaseModel):
    """Specifies that an input for a tool comes from a previous step's named output."""
    from_step_id: str      # The 'step_id' of the preceding ExecutionStep that produced this input.
    named_output_key: str  # The logical key (name) under which the specific output was stored by the source_step.

# --- Defining a Single Tool Call (Granular Operator) ---
class ToolCall(BaseModel):
    """Defines a call to a single granular tool (one of your ~16 operators from README.md)."""
    tool_id: str  # Unique string identifying the tool. 
                  # Examples: "Entity.PPR", "Chunk.FromRelationships", "Subgraph.KHopPaths", "web.Google Search"
    
    description: Optional[str] = None # Optional human-readable description or the LLM's reasoning for this tool call.

    parameters: Optional[Dict[str, Any]] = None 
                  # Direct literal parameters for THIS tool call.
                  # e.g., for "Entity.PPR": {"personalization_weight": 0.15, "top_k_results": 10}
                  # These are parameters whose values are known when the plan is created, not sourced from other steps.

    # How this tool's inputs are sourced.
    # Key: The logical input parameter name the tool's underlying implementation expects.
    # Value can be:
    #   - A literal value (str, int, float, bool, list, dict) if not specified in 'parameters'.
    #   - A string like "plan_inputs.user_query" to reference a global input defined in the ExecutionPlan.
    #   - A ToolInputSource instance to dynamically link to a previous step's named output.
    inputs: Optional[Dict[str, Union[Any, ToolInputSource, str]]] = None

    # How the outputs of THIS tool call are named for use by subsequent steps or as final plan results.
    # Key: A logical name chosen for this output within the plan (e.g., "retrieved_entities_with_scores").
    # Value: A string describing the expected type or nature of the output (e.g., "List[Tuple[str, float]]", "List[TextChunk]").
    #        This helps in understanding the plan and for potential type checking by the orchestrator.
    named_outputs: Optional[Dict[str, str]] = None 

# --- Configuration for different types of Execution Steps ---
class DynamicToolChainConfig(BaseModel):
    """Defines a step that executes a dynamic chain of granular tools."""
    chain_description: Optional[str] = None
    tools: List[ToolCall] # The sequence of tool calls that make up this step.
    # A patch here could apply to some overarching context for this chain if necessary, 
    # though most parameters should be within individual ToolCalls.
    # yaml_patch: Optional[Dict[str, Any]] = None 

class PredefinedMethodConfig(BaseModel):
    """Defines a step that executes a full, pre-defined RAG method via its YAML configuration."""
    method_yaml_name: str # Name of the YAML file in Option/Method/, e.g., "HippoRAG.yaml", "RAPTOR.yaml".
    
    # Patch to apply to the loaded method YAML.
    # This allows overriding specific configurations within the chosen method YAML.
    # Could be a generic Dict, or refined later into specific Pydantic "Patch" models.
    yaml_patch: Optional[Dict[str, Any]] = None 

# --- A Single Step in the Execution Plan ---
class ExecutionStep(BaseModel):
    """A single, discrete step in the overall execution plan."""
    step_id: str = Field(default_factory=lambda: f"step_{uuid.uuid4().hex[:8]}") # Auto-generated unique ID for referencing.
    description: Optional[str] = None # What this step aims to achieve.
    
    # The core action of this step. The LLM agent chooses one type of action.
    # Other action types like "kg_build_config" (for graph building steps) or "conditional_branch" could be added later.
    action: Union[DynamicToolChainConfig, PredefinedMethodConfig]

# --- The Main Execution Plan ---
class ExecutionPlan(BaseModel):
    """The overall plan generated by the LLM agent to fulfill a user's request."""
    plan_id: str = Field(default_factory=lambda: f"plan_{uuid.uuid4().hex[:8]}") # Auto-generated unique ID.
    plan_description: str # Human-readable summary of the plan's overall goal.
    
    target_dataset_name: str # Specifies which dataset this plan operates on (influences artifact paths).
                             # Refers to datasets like those in Data/MySampleTexts/ or Data/HotpotQA/.
    
    ontology_setup: Optional[OntologyConfig] = None # Specifies ontology for KG creation or interaction within this plan.
    
    # Optional global configurations/patches that can apply to all steps unless overridden at the step/tool level.
    global_llm_config_patch: Optional[Dict[str, Any]] = None # e.g., to set a specific LLM model for all LLM-dependent tools.
                                                              # Maps to Core/Config/LLMConfig.py parameters.
    global_embedding_config_patch: Optional[Dict[str, Any]] = None # Maps to Core/Config/EmbConfig.py parameters.

    # Inputs that are global to the entire plan (e.g., the user's initial query, references to existing VDBs or graphs).
    # Individual ToolCalls within steps can refer to these using a prefix like "plan_inputs.user_query".
    plan_inputs: Optional[Dict[str, Any]] = None 
                        # e.g., {"user_query": "What were the main causes of the French Revolution?", 
                        #        "main_graph_reference_id": "my_sample_er_graph_v1",
                        #        "document_vdb_id": "my_sample_docs_vdb_v1"}

    steps: List[ExecutionStep] # The ordered sequence of steps the Agent Orchestrator will execute.

# Explanation of these models:
#
# These Pydantic models define the expected JSON structure for an "Execution Plan."
# ExecutionPlan is the top-level object.
# It contains a list of ExecutionSteps.
# Each ExecutionStep has an action which can either be a DynamicToolChainConfig (a list of ToolCalls for your granular operators) or a PredefinedMethodConfig (to run a full method YAML like Dalk.yaml). This Union gives the agent flexibility.
# The ToolCall model is key for dynamic chaining. Its tool_id will map to one of your ~16 KG operators (or new tools like web search). The parameters field holds direct values for the tool, while the inputs field allows sourcing data from plan_inputs or named_outputs of previous steps, enabling data flow. named_outputs declares what a tool call produces.
# OntologyConfig allows the plan to specify how the ontology should be handled, referencing your Config/custom_ontology.json or even allowing the LLM to define one inline.
