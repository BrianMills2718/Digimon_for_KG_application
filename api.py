import asyncio
import os
import json 
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS

from Core.GraphRAG import GraphRAG
from Option.Config2 import Config # Assuming Config class is in Option.Config2
from Core.Common.Logger import logger 
# from Core.Utils.Exceptions import EmptyNetworkError # Removed as it's not defined in user's Exceptions.py

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

if __name__ == '__main__':
    logger.info("Starting Flask API server for GraphRAG...")
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)

