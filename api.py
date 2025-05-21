import asyncio
import os
import json 
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
from pathlib import Path

# --- LLM Chat Guidance Imports (commented out, adjust as needed) ---
# from Core.Common.LLM import LLM # Assuming LLM class is in Core.Common.LLM
# from Core.Config.LLMConfig import LLMConfig # Assuming LLMConfig is here
# from Core.Provider.LLMProviderRegister import LLMProviderRegister # If you use this
# from Core.Provider.OpenaiApi import OpenaiApi # If using OpenAI directly

from Core.GraphRAG import GraphRAG
from Option.Config2 import Config # Assuming Config class is in Option.Config2
from Core.Common.Logger import logger 
# from Core.Utils.Exceptions import EmptyNetworkError # Removed as it's not defined in user's Exceptions.py

# Ontology management paths
ONTOLOGY_DIR = Path("Config")
CUSTOM_ONTOLOGY_FILE = ONTOLOGY_DIR / "custom_ontology.json"

# High-level and low-level methods/operators description for analysis guidance
available_techniques_and_methods_description = """
High-Level Methods (UI):
- Dalk: Narrative transformation, path analysis.
- GR: Global retrieval, subgraph discovery.
- LGraphRAG: Local graph retrieval, entity-centric analysis.
- GGraphRAG: Global graph retrieval, cross-community analysis.
- HippoRAG: Personalized PageRank, influential user detection.
- KGP: Knowledge graph pathfinding, relationship tracing.
- LightRAG: Lightweight retrieval, fast neighborhood exploration.
- RAPTOR: Tree-based aggregation, hierarchical exploration.
- ToG: Agent-based traversal, dynamic subgraph extraction.

Low-Level Operators:
- Node Retrieval: By type, property, or relationship.
- Edge Retrieval: By type or property.
- Subgraph Extraction: k-hop, path, Steiner tree.
- Centrality Measures: Degree, betweenness, PageRank.
- Community Detection: Cluster assignment, modularity.
- Aggregation: Node/edge attribute aggregation, group-by.
"""


# Initialize Flask app
app = Flask(__name__)
CORS(app) 

graphrag_instances_cache = {}

async def get_or_create_graphrag_instance(dataset_name: str, method_config_path_stem: str, setup_for_querying_required: bool = True):
    """
    Retrieves a GraphRAG instance from cache or creates and initializes it.
    If setup_for_querying_required is True, it will also load artifacts.
    Uses method_config_path_stem directly as part of the cache key AND for exp_name.
    """
    cache_key = (dataset_name, method_config_path_stem)

    if setup_for_querying_required and cache_key in graphrag_instances_cache:
        logger.info(f"Using cached GraphRAG instance for {cache_key}")
        return graphrag_instances_cache[cache_key]
    
    logger.info(f"Creating new GraphRAG instance for {cache_key} (method stem/exp_name: {method_config_path_stem})")
    try:
        options_file_path = Path("Option/Method") / f"{method_config_path_stem}.yaml"
        if not options_file_path.exists():
            logger.error(f"Method configuration file not found: {options_file_path}")
            return None
            
        # *** MODIFIED LINE: Pass method_config_path_stem as exp_name ***
        config_options = Config.parse(
            options_file_path, 
            dataset_name=dataset_name, 
            exp_name=method_config_path_stem # This ensures self.config.exp_name is the method name
        )
        graphrag_instance = GraphRAG(config=config_options)

        if setup_for_querying_required:
            logger.info(f"Setting up GraphRAG for querying (loading artifacts) for {cache_key}...")
            if not await graphrag_instance.setup_for_querying():
                logger.error(f"Failed to setup GraphRAG for querying for {cache_key}.")
                if cache_key in graphrag_instances_cache: 
                    del graphrag_instances_cache[cache_key]
                return None
            graphrag_instances_cache[cache_key] = graphrag_instance 
            logger.info(f"GraphRAG instance for {cache_key} initialized, setup for querying, and cached.")
        else: # Typically for build, where we don't need to load artifacts yet from this instance
            logger.info(f"GraphRAG instance for {cache_key} created (not setup for querying, exp_name set to method).")

        return graphrag_instance
    except Exception as e:
        logger.error(f"Error creating or setting up GraphRAG instance for {cache_key}: {e}", exc_info=True)
        if cache_key in graphrag_instances_cache: 
            del graphrag_instances_cache[cache_key]
        return None

@app.route('/api/query', methods=['POST'])
def handle_query():
    try:
        data = request.get_json()
        logger.info(f"Received /api/query request: {data}")

        dataset_name = data.get('datasetName')
        selected_method_stem = data.get('selectedMethod')
        question_str = data.get('question')

        if not all([dataset_name, selected_method_stem, question_str]):
            logger.error("Missing parameters in /api/query request")
            return jsonify({"error": "Missing parameters: datasetName, selectedMethod, and question are required."}), 400

        # Pass selected_method_stem to get_or_create_graphrag_instance
        graphrag_instance = asyncio.run(get_or_create_graphrag_instance(dataset_name, selected_method_stem, setup_for_querying_required=True))

        if graphrag_instance is None:
            logger.error(f"Could not initialize GraphRAG instance for query: {dataset_name}, {selected_method_stem}")
            return jsonify({"error": "Failed to initialize RAG system for querying. Check backend logs."}), 500

        logger.info(f"Processing query with {selected_method_stem} on {dataset_name}: '{question_str[:50]}...'")
        answer = asyncio.run(graphrag_instance.query(question_str))
        logger.info(f"Generated answer snippet: {str(answer)[:100]}...")

        return jsonify({"answer": answer})

    except Exception as e:
        logger.error(f"Error in /api/query: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/build', methods=['POST'])
def handle_build():
    data = request.get_json()
    logger.info(f"Received /api/build request: {data}")

    dataset_name = data.get('datasetName')
    selected_method_stem = data.get('selectedMethod')

    if not all([dataset_name, selected_method_stem]):
        logger.error("Missing parameters in /api/build request")
        return jsonify({"error": "Missing parameters: datasetName and selectedMethod are required."}), 400

    cache_key = (dataset_name, selected_method_stem)

    try:
        logger.info(f"Creating GraphRAG instance for build: {dataset_name}, {selected_method_stem}")
        # For build, exp_name in config should be the method stem.
        graphrag_instance_for_build = asyncio.run(get_or_create_graphrag_instance(dataset_name, selected_method_stem, setup_for_querying_required=False))
        
        if graphrag_instance_for_build is None:
            logger.error(f"Failed to create GraphRAG instance for build: {dataset_name}, {selected_method_stem}")
            return jsonify({"error": "Failed to create RAG system for build. Check backend logs."}), 500

        docs_path = Path("Data") / dataset_name 
        logger.info(f"Document path for build: {str(docs_path)}")
        
        logger.info(f"Starting artifact build for {dataset_name} with {selected_method_stem}...")
        
        build_result = asyncio.run(graphrag_instance_for_build.build_and_persist_artifacts(str(docs_path))) 

        if isinstance(build_result, dict) and "error" in build_result: # Check if build method returned an error
            logger.error(f"Build process reported an error: {build_result['error']}")
            return jsonify({"error": f"Failed to build artifacts: {build_result['error']}"}), 500

        if cache_key in graphrag_instances_cache:
            del graphrag_instances_cache[cache_key]
            logger.info(f"Cache invalidated for {cache_key} after successful build.")

        logger.info(f"Build completed for {dataset_name} with {selected_method_stem}.")
        # Ensure build_result is a dict with 'message' for consistency
        response_message = build_result.get("message", f"Build process completed for {dataset_name} using {selected_method_stem}.") if isinstance(build_result, dict) else str(build_result)

        return jsonify({"message": response_message, "details": "Build completed."}), 200
    
    except Exception as e: # General exception for other unexpected errors
        logger.error(f"Error in /api/build for {dataset_name}, {selected_method_stem}: {e}", exc_info=True)
        return jsonify({"error": f"Failed to build artifacts: {str(e)}"}), 500


@app.route('/api/evaluate', methods=['POST'])
def handle_evaluate():
    data = request.get_json()
    logger.info(f"Received /api/evaluate request: {data}")

    dataset_name = data.get('datasetName')
    selected_method_stem = data.get('selectedMethod')

    if not all([dataset_name, selected_method_stem]):
        logger.error("Missing parameters in /api/evaluate request")
        return jsonify({"error": "Missing parameters: datasetName and selectedMethod are required for evaluation."}), 400

    try:
        # Pass selected_method_stem to get_or_create_graphrag_instance
        graphrag_instance = asyncio.run(get_or_create_graphrag_instance(dataset_name, selected_method_stem, setup_for_querying_required=True))

        if graphrag_instance is None:
            logger.error(f"Could not initialize GraphRAG instance for evaluation: {dataset_name}, {selected_method_stem}")
            return jsonify({"error": "Failed to initialize RAG system for evaluation. Ensure artifacts are built. Check backend logs."}), 500

        logger.info(f"Starting evaluation for {dataset_name} with {selected_method_stem} (exp_name in config: {graphrag_instance.config.exp_name})...")
        
        evaluation_result_data = asyncio.run(graphrag_instance.evaluate_model())

        if isinstance(evaluation_result_data, dict) and "error" in evaluation_result_data:
            logger.error(f"Evaluation process reported an error: {evaluation_result_data['error']}")
            return jsonify({"error": evaluation_result_data['error'], "metrics": evaluation_result_data.get("metrics", {})}), 500
        
        response_payload = {
            "message": evaluation_result_data.get("message", "Evaluation completed."),
            "metrics": evaluation_result_data.get("metrics", {}), 
            "results_file_path": evaluation_result_data.get("results_file_path"),
            "metrics_file_path": evaluation_result_data.get("metrics_file_path")
        }
        
        logger.info(f"Evaluation completed for {dataset_name} with {selected_method_stem}. Response payload: {response_payload}")
        return jsonify(response_payload), 200

    except AttributeError as e:
        if 'evaluate_model' in str(e): 
            logger.error(f"GraphRAG instance does not have an 'evaluate_model' method: {e}", exc_info=True)
            return jsonify({"error": "Evaluation functionality (evaluate_model method) not found in GraphRAG instance."}), 501
        else:
            logger.error(f"AttributeError in /api/evaluate: {e}", exc_info=True)
            return jsonify({"error": f"An attribute error occurred during evaluation: {str(e)}"}), 500
    except Exception as e:
        logger.error(f"Error in /api/evaluate for {dataset_name}, {selected_method_stem}: {e}", exc_info=True)
        return jsonify({"error": f"Failed to run evaluation: {str(e)}"}), 500

# --- Chat Guidance LLM Placeholder Function ---
# Replace this with actual LLM logic as needed.
import logging

def get_llm_chat_response(messages_for_llm):
    """
    Calls Gemini API (google-genai) in JSON mode for guidance.
    Uses schema-based output for robust, structured replies.
    """
    import os
    import logging
    logger = logging.getLogger("GeminiLLM")
    try:
        from google import genai
        try:
            import typing_extensions as typing
        except ImportError:
            import typing as typing
    except ImportError:
        logger.error("google-genai package not installed. Please install with: pip install google-genai")
        return "[Backend error: Gemini API package not installed. Please contact admin.]"

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY environment variable not set.")
        return "[Backend error: Gemini API key not set. Please contact admin.]"

    client = genai.Client(api_key=api_key)
    MODEL_ID = "gemini-2.5-flash-preview-04-17"

    # Compose the chat as a single string prompt (simulate chat history)
    system_prompt = ""
    user_turns = []
    for msg in messages_for_llm:
        if msg["role"] == "system":
            system_prompt = msg["content"]
        elif msg["role"] == "user":
            user_turns.append(f"User: {msg['content']}")
        elif msg["role"] == "assistant":
            user_turns.append(f"Assistant: {msg['content']}")
    prompt = system_prompt + "\n" + "\n".join(user_turns)

    # Define a simple schema for guidance replies
    class GuidanceReply(typing.TypedDict):
        reply: str

    try:
        result = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
            config={
                'response_mime_type': 'application/json',
                'response_schema': GuidanceReply,
            },
        )
        logger.info(f"Gemini raw response: {result.text}")
        import json
        response_obj = json.loads(result.text)
        return response_obj.get("reply", "[No reply returned by Gemini API]")
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return f"[Backend error: Gemini API call failed: {e}]"

# --- Chat Guidance Endpoint ---
from flask import request, jsonify

from flask import request, jsonify
import json
from pathlib import Path
from Core.Common.Logger import logger
# Make sure get_llm_chat_response is imported/defined

# Ensure ONTOLOGY_DIR and CUSTOM_ONTOLOGY_FILE are defined globally
# ONTOLOGY_DIR = Path("Config")
# CUSTOM_ONTOLOGY_FILE = ONTOLOGY_DIR / "custom_ontology.json"

@app.route('/api/ontology_chat', methods=['POST'])
def handle_ontology_chat():
    try:
        data = request.get_json()
        user_message_content = data.get('message')
        history = data.get('history', [])

        if not user_message_content:
            logger.error("No message provided in /api/ontology_chat")
            return jsonify({"error": "No message provided"}), 400

        logger.info(f"Received /api/ontology_chat with message: {user_message_content[:100]}...")

        # --- Prompt for Human-Readable Description ---
        system_prompt_ontology_description = (
            "You are an expert AI assistant for the 'Knowledge Graph Analysis Framework' (internally named 'Digimon'). "
            "Your current specific role is to help Subject Matter Experts (SMEs) CONVERSATIONALLY DESIGN the ONTOLOGY for their knowledge graph. "
            "When a user describes their research question or analytical goal (e.g., 'I want to analyze how narratives transform', 'I am trying to build a KG of social media discourse'):\n"
            "1. Focus SOLELY on discussing and explaining a suitable ontology structure. Do NOT discuss analysis methods or operators. "
            "2. Help them define relevant entity types, key properties for those entities, and important relationship types (including typical source/target entities and key properties). "
            "3. Provide a COMPREHENSIVE, HUMAN-READABLE EXPLANATION of your suggested ontology design, including your reasoning and why each part is helpful for their stated goal. Use bullet points for clarity. "
            "Do NOT include any JSON in THIS response. This is only for the descriptive part. "
            "If the user provides feedback, help them refine this descriptive suggestion."
        )

        messages_for_description_llm = [{"role": "system", "content": system_prompt_ontology_description}]
        for msg in history:
            if msg.get("text") and isinstance(msg.get("isUser"), bool):
                messages_for_description_llm.append({
                    "role": "user" if msg.get("isUser") else "assistant",
                    "content": msg.get("text")
                })
        messages_for_description_llm.append({"role": "user", "content": user_message_content})

        descriptive_reply_text = get_llm_chat_response(messages_for_description_llm)
        logger.info(f"LLM ontology chat (descriptive part) reply: {descriptive_reply_text[:100]}...")

        # --- Prompt for JSON Conversion of the Description ---
        system_prompt_ontology_json_conversion = (
            "You are a data structuring AI. Your task is to convert a textual description of a knowledge graph ontology into a **strict JSON format**. "
            "The JSON output MUST have a top-level object with two keys: 'entities' and 'relations'.\n"
            "Each item in the 'entities' list MUST be an object with a 'name' (string) and a 'properties' list. Each item in 'properties' MUST be an object with 'name' (string), 'type' (string, e.g., 'string', 'integer', 'text', 'datetime', 'boolean'), and 'description' (string).\n"
            "Each item in the 'relations' list MUST be an object with a 'name' (string), 'source_entity' (string, matching an entity name), 'target_entity' (string, matching an entity name), and a 'properties' list (structured like entity properties). A top-level 'description' for the relation itself is also good if appropriate.\n"
            "Output ONLY the valid JSON object. Do NOT include any explanations, apologies, or surrounding text like ```json or ```."
        )

        user_prompt_for_json = (
            f"Based on our prior discussion and the user's goal, the following ontology was described:\n--- BEGIN ONTOLOGY DESCRIPTION ---\n{descriptive_reply_text}\n--- END ONTOLOGY DESCRIPTION ---\n\n"
            f"Now, please convert this description into the strict JSON format as specified. User's original goal was: '{user_message_content}'"
        )

        messages_for_json_llm = [
            {"role": "system", "content": system_prompt_ontology_json_conversion},
            {"role": "user", "content": user_prompt_for_json}
        ]

        json_reply_str = get_llm_chat_response(messages_for_json_llm)
        logger.info(f"LLM ontology chat (JSON part) raw string: {json_reply_str[:200]}...")

        suggested_ontology_object = None
        try:
            cleaned_json_str = json_reply_str.strip()
            if cleaned_json_str.startswith("```json"):
                cleaned_json_str = cleaned_json_str[7:]
            if cleaned_json_str.endswith("```"):
                cleaned_json_str = cleaned_json_str[:-3]

            suggested_ontology_object = json.loads(cleaned_json_str.strip())
            if not isinstance(suggested_ontology_object, dict) or \
               not isinstance(suggested_ontology_object.get("entities"), list) or \
               not isinstance(suggested_ontology_object.get("relations"), list):
                logger.error("Generated JSON does not have the root 'entities' and 'relations' lists.")
                suggested_ontology_object = {"error": "Generated JSON structure is invalid."}
            else:
                logger.info("Successfully parsed LLM response into ontology JSON object.")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON output from LLM for ontology: {e}. Raw string: {json_reply_str}")
            suggested_ontology_object = {"error": f"Failed to parse suggested ontology as JSON: {e}", "raw_output": json_reply_str}
        except Exception as e_gen:
            logger.error(f"Generic error processing JSON output from LLM: {e_gen}. Raw string: {json_reply_str}")
            suggested_ontology_object = {"error": f"Error processing suggested ontology: {e_gen}", "raw_output": json_reply_str}

        return jsonify({
            "descriptive_reply": descriptive_reply_text,
            "suggested_ontology_json": suggested_ontology_object 
        }), 200



        messages_for_llm = [{"role": "system", "content": system_prompt_ontology}]
        for msg in history:
            if msg.get("text") and isinstance(msg.get("isUser"), bool):
                messages_for_llm.append({
                    "role": "user" if msg.get("isUser") else "assistant",
                    "content": msg.get("text")
                })
        messages_for_llm.append({"role": "user", "content": user_message_content})

        llm_reply = get_llm_chat_response(messages_for_llm)

        logger.info(f"LLM ontology chat reply: {llm_reply[:100]}...")
        return jsonify({"reply": llm_reply})

    except Exception as e:
        logger.error(f"Error in /api/ontology_chat: {e}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/api/chat_guidance', methods=['POST'])
def handle_chat_guidance():
    try:
        data = request.get_json()
        user_message_content = data.get('message')
        history = data.get('history', [])

        if not user_message_content:
            logger.error("No message provided in /api/chat_guidance")
            return jsonify({"error": "No message provided"}), 400

        logger.info(f"Received /api/chat_guidance (analysis) with message: {user_message_content[:100]}...")

        # Load current custom ontology to provide context to the LLM
        current_ontology_str = "No custom ontology is currently defined."
        if CUSTOM_ONTOLOGY_FILE.exists() and CUSTOM_ONTOLOGY_FILE.is_file():
            try:
                with open(CUSTOM_ONTOLOGY_FILE, 'r') as f_ont:
                    ontology_data = json.load(f_ont)
                    current_ontology_str = json.dumps(ontology_data, indent=2)
                logger.info(f"Loaded custom ontology for analysis chat context from {CUSTOM_ONTOLOGY_FILE}")
            except Exception as e:
                logger.error(f"Failed to load or parse custom ontology for analysis chat: {e}")
                current_ontology_str = "Error loading custom ontology. Please check its format."

        # available_techniques_and_methods_description should be defined globally or above this function
        system_prompt_analysis = (
            "You are an expert AI assistant for the 'Knowledge Graph Analysis Framework' (internally named 'Digimon'). "
            "Your current specific role is to help Subject Matter Experts (SMEs) with ANALYSIS and RETRIEVAL from their knowledge graph. "
            "Assume an ontology for the graph is already defined or being designed separately. "
            f"The currently defined custom ontology for the graph is:\n```json\n{current_ontology_str}\n```\n\n"
            "Your main task is to guide users on how to achieve their analytical goals using this framework and the defined ontology. "
            "You have access to a list of high-level analysis 'Methods' available in the UI, and the underlying low-level 'Operators' they use. "
            f"Reference information on these techniques and methods: {available_techniques_and_methods_description}"

            "\n\nUSER GUIDANCE TASK (linking to UI methods & operators based on ontology):\n"
            "When a user asks how to achieve a specific research goal (e.g., 'how to analyze narrative transformation', 'how to find influential users'):\n"
            "1. Consider their goal in the context of the provided current ontology. "
            "2. First, explain the general analytical principle or relevant low-level KG operator(s) that would be effective with the current ontology. "
            "3. Second, and most importantly, recommend one or more of the 9 specific High-Level UI Methods (Dalk, GR, LGraphRAG, GGraphRAG, HippoRAG, KGP, LightRAG, RAPTOR, ToG) that implement or are well-suited for that principle/operator, especially considering the current ontology. Explain your recommendation. "
            "For example, if the ontology defines 'Tweets' and 'Users' with a 'RETWEETED' relationship, and the user wants to find influential users, you might suggest PPR and then the HippoRAG method, explaining how it applies to 'Users' and 'RETWEETED' relationships from their ontology. "
            "If the current ontology seems unsuitable or missing key elements for their stated goal, you can politely point this out and suggest that they might first need to refine their ontology (perhaps using the dedicated ontology design chat). "
            "If the user's query is a general question or greeting, respond normally. "
            "If a question is too complex or outside your scope, politely say so."
        )
        """
        Low-Level Operator Categories (these are components within the methods above):
        I. RETRIEVAL OPERATORS:
            A. Node/Entity Retrieval:
                - Vector Search (VDB): Find nodes by semantic similarity. (Used in GR, LightRAG, RAPTOR)
                - Personalized PageRank (PPR): Identify influential/central nodes. (Used in HippoRAG, FastGraphRAG - though FastGraphRAG is not in the UI list)
                - Keyword Search / TF-IDF: Find nodes by keyword matching. (Used in KGP)
                - Entity Occurrence: Retrieve items where specific entities co-occur.
                - By Relationship (RelNode): Find nodes connected by certain relationships. (Used in LightRAG)
                - OneHop: Get direct neighbors. (Used in LightRAG, LGraphRAG, ToG)
                - Agent: LLM selects relevant nodes/entities. (Used in ToG)
            B. Relationship/Edge Retrieval:
                - VDB, OneHop, Aggregator, Agent.
            C. Subgraph Retrieval:
                - Path Finding (KhopPath, AgentPath): Find paths between nodes. (Used in Dalk, ToG)
                - Steiner Tree: Minimal connecting subgraph. (Used in GR)
            D. Community Retrieval:
                - By Entity, By Layer: Retrieve communities based on content or hierarchy. (Used in LGraphRAG, GGraphRAG)
            E. Chunk/Document Retrieval:
                - Based on associated entities, relationships, or vector similarity. (Used in many methods like LightRAG, LGraphRAG, HippoRAG)

        II. TRANSFORMATION & ANALYSIS OPERATORS (often chained after retrieval):
            - extract_categorical_value: Classify text (e.g., narrative frames, sentiment).
            - generate_vector_representation: Create semantic embeddings.
            - generate_text_summary: Summarize text.
            - to_categorical_distribution: Quantify frequencies of categories.
            - to_graph_projection: Create simplified graph views.
            - predict_edge_weight: Forecast future connections.
            - find_causal_paths: Trace influence.

        User Guidance Task:
        When a user describes an analytical goal (e.g., 'how to analyze narrative transformation', 'how to find influential users'):
        1. Identify the core analytical task.
        2. Explain the underlying low-level KG operator(s) or principle(s) best suited for this task.
        3. Crucially, then recommend which of the 9 High-Level UI Methods (Dalk, GR, LGraphRAG, GGraphRAG, HippoRAG, KGP, LightRAG, RAPTOR, ToG) would be most appropriate to use to implement that principle/operator. Explain why that Method is a good fit.
        4. If multiple methods could apply, you can mention them.
        Example: If recommending PPR for influence, also mention that 'HippoRAG' is a UI method that utilizes PPR.
        """

        messages_for_llm = []
        messages_for_llm.append({
            "role": "system",
            "content": (
                "You are an expert AI assistant for the 'Knowledge Graph Analysis Framework' (internally named 'Digimon'). "
                "Your role is to provide clear, concise, and helpful guidance to Subject Matter Experts (SMEs) "
                "on using the framework to achieve their analytical goals. "
                "You have access to a list of high-level analysis 'Methods' available in the UI, and the underlying low-level 'Operators' they use. "
                f"Here is the information about available techniques and methods: {available_techniques_and_methods_description}"

                "\n\nONTOLOGY SUGGESTION TASK:\n"
                "When a user describes their research question or analytical goal (e.g., 'I want to analyze how narratives transform', 'I need to understand influence networks for specific topics'), "
                "in addition to recommending analysis methods/operators: \n"
                "1. Think about what kinds of entities, properties, and relationships would be essential or helpful to capture in the knowledge graph to address their goal. "
                "2. Suggest a basic ontology structure. This suggestion should be formatted clearly, perhaps using bullet points for entity types and their key properties, and relation types with their typical source/target entities and key properties. "
                "3. Explain briefly why this ontology structure is suitable for their stated goal. "
                "4. Present this ontology suggestion *before* or *alongside* the method/operator recommendations. "
                "Example Ontology Suggestion Format for 'analyzing narrative transformation between communities':\n"
                "   'To analyze narrative transformation, it would be beneficial to structure your graph with an ontology like this:\n"
                "   Entities:\n"
                "     - User: (Properties: user_id, community_id, other_demographics)\n"
                "     - Post: (Properties: post_id, text_content, timestamp, author_user_id)\n"
                "     - NarrativeFrame: (Properties: frame_name, description, keywords)\n"
                "     - Community: (Properties: community_id, platform, name)\n"
                "   Relationships:\n"
                "     - POSTED: (Source: User, Target: Post, Properties: timestamp)\n"
                "     - USES_FRAME: (Source: Post, Target: NarrativeFrame, Properties: explicitness_score)\n"
                "     - MEMBER_OF: (Source: User, Target: Community)\n"
                "   This structure helps capture who says what, what narrative elements are used, and in which communities, which is key for tracking transformations.'\n\n"

                "USER GUIDANCE TASK (linking to UI methods):\n"
                "When a user asks how to achieve a specific research goal: "
                "First, explain the general analytical principle or relevant low-level KG operator(s). "
                "Second, and most importantly, recommend one or more of the 9 specific High-Level UI Methods (Dalk, GR, LGraphRAG, GGraphRAG, HippoRAG, KGP, LightRAG, RAPTOR, ToG) that implement or are well-suited for that principle/operator, and explain why. "
                "If the user's query is a general question or greeting, respond normally. "
                "If a question is too complex or outside your scope, politely say so."
            )
        })

        for msg in history:
            if msg.get("text") and isinstance(msg.get("isUser"), bool):
                messages_for_llm.append({
                    "role": "user" if msg.get("isUser") else "assistant",
                    "content": msg.get("text")
                })

        messages_for_llm.append({"role": "user", "content": user_message_content})

        llm_reply = get_llm_chat_response(messages_for_llm)

        logger.info(f"LLM recommendation reply: {llm_reply[:100]}...")
        return jsonify({"reply": llm_reply})

    except Exception as e:
        logger.error(f"Error in /api/chat_guidance: {e}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/api/ontology', methods=['GET'])
def get_ontology():
    try:
        if CUSTOM_ONTOLOGY_FILE.exists() and CUSTOM_ONTOLOGY_FILE.is_file():
            with open(CUSTOM_ONTOLOGY_FILE, 'r') as f:
                ontology_data = json.load(f)
            logger.info(f"Retrieved custom ontology from {CUSTOM_ONTOLOGY_FILE}")
            return jsonify(ontology_data), 200
        else:
            logger.info(f"Custom ontology file not found at {CUSTOM_ONTOLOGY_FILE}. Returning empty ontology.")
            return jsonify({"entities": [], "relations": [], "message": "No custom ontology defined."}), 200
    except Exception as e:
        logger.error(f"Error in /api/ontology GET: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/ontology', methods=['POST'])
def save_ontology():
    try:
        ontology_data = request.get_json()
        if not ontology_data:
            logger.error("No ontology data provided in POST request to /api/ontology")
            return jsonify({"error": "No data provided"}), 400
        ONTOLOGY_DIR.mkdir(parents=True, exist_ok=True)
        with open(CUSTOM_ONTOLOGY_FILE, 'w') as f:
            json.dump(ontology_data, f, indent=2)
        logger.info(f"Custom ontology saved to {CUSTOM_ONTOLOGY_FILE}")
        # Optionally clear cache or signal reload here
        return jsonify({"message": "Ontology saved successfully."}), 200
    except Exception as e:
        logger.error(f"Error in /api/ontology POST: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Flask API server for GraphRAG...")
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)

