import asyncio
import os
import json # Make sure json is imported
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS

from Core.GraphRAG import GraphRAG
from Option.Config2 import Config
# Assuming your logger is configured in a way that it can be imported and used
# If not, you might need to initialize it here or remove detailed logging for the API
from Core.Common.Logger import logger 

# Initialize Flask app
app = Flask(__name__)
CORS(app) # Enable CORS for all routes, allowing requests from your React app

# Global cache for GraphRAG instances to avoid re-initializing on every request
# Key: (dataset_name, method_config_path_stem)
# Value: graphrag_instance
graphrag_instances_cache = {}

async def get_or_create_graphrag_instance(dataset_name: str, method_config_path: str):
    """
    Retrieves a GraphRAG instance from cache or creates and initializes it.
    This function is async because setup_for_querying is async.
    """
    method_config_stem = Path(method_config_path).stem
    cache_key = (dataset_name, method_config_stem)

    if cache_key in graphrag_instances_cache:
        logger.info(f"Using cached GraphRAG instance for {cache_key}")
        return graphrag_instances_cache[cache_key]

    logger.info(f"Creating new GraphRAG instance for {cache_key}")
    try:
        # Construct the full path to the method's option YAML file
        # Assuming 'Option/Method/' is relative to the project root where api.py is
        options_file_path = Path("Option/Method") / f"{method_config_stem}.yaml"

        config_options = Config.parse(options_file_path, dataset_name=dataset_name)
        graphrag_instance = GraphRAG(config=config_options)

        logger.info(f"Setting up GraphRAG for querying (loading artifacts) for {cache_key}...")
        if not await graphrag_instance.setup_for_querying():
            logger.error(f"Failed to setup GraphRAG for querying for {cache_key}.")
            # Remove from cache if setup failed to avoid using a bad instance
            if cache_key in graphrag_instances_cache:
                del graphrag_instances_cache[cache_key]
            return None

        graphrag_instances_cache[cache_key] = graphrag_instance
        logger.info(f"GraphRAG instance for {cache_key} initialized and cached.")
        return graphrag_instance
    except Exception as e:
        logger.error(f"Error creating or setting up GraphRAG instance for {cache_key}: {e}")
        if cache_key in graphrag_instances_cache: # Should not happen if error during creation
            del graphrag_instances_cache[cache_key]
        return None

@app.route('/api/query', methods=['POST'])
def handle_query():
    try:
        data = request.get_json()
        logger.info(f"Received /api/query request: {data}")

        dataset_name = data.get('datasetName')
        selected_method_stem = data.get('selectedMethod') # e.g., "LGraphRAG"
        question_str = data.get('question')

        if not all([dataset_name, selected_method_stem, question_str]):
            logger.error("Missing parameters in /api/query request")
            return jsonify({"error": "Missing parameters: datasetName, selectedMethod, and question are required."}), 400

        method_config_path = f"Option/Method/{selected_method_stem}.yaml" # Construct path

        # Run async code in Flask sync route
        # For a production app, consider an async framework like FastAPI or Quart,
        # or use Flask's async capabilities if available and suitable.
        # For now, asyncio.run() is a simple way to bridge this.

        # Get or create GraphRAG instance
        graphrag_instance = asyncio.run(get_or_create_graphrag_instance(dataset_name, method_config_path))

        if graphrag_instance is None:
            logger.error(f"Could not initialize GraphRAG instance for {dataset_name}, {selected_method_stem}")
            return jsonify({"error": "Failed to initialize RAG system. Check backend logs."}), 500

        logger.info(f"Processing query with {selected_method_stem} on {dataset_name}: '{question_str[:50]}...'")

        answer = asyncio.run(graphrag_instance.query(question_str))
        logger.info(f"Generated answer: {answer[:100]}...") # Log a snippet of the answer

        return jsonify({"answer": answer})

    except Exception as e:
        logger.error(f"Error in /api/query: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# Placeholder for build endpoint
@app.route('/api/build', methods=['POST'])
def handle_build():
    data = request.get_json()
    logger.info(f"Received /api/build request: {data}")
    # TODO: Implement logic similar to handle_query to get config,
    # then call graphrag_instance.build_and_persist_artifacts()
    # This will be more complex due to the long-running nature of build.
    # For now, just a placeholder.
    return jsonify({"message": "Build endpoint called (not yet implemented)", "data_received": data}), 202

# Placeholder for evaluate endpoint
@app.route('/api/evaluate', methods=['POST'])
def handle_evaluate():
    data = request.get_json()
    logger.info(f"Received /api/evaluate request: {data}")
    # TODO: Implement logic similar to handle_query to get config,
    # then call the evaluation flow (which itself calls query multiple times).
    # This will also be complex. Placeholder for now.
    return jsonify({"message": "Evaluate endpoint called (not yet implemented)", "data_received": data}), 202


if __name__ == '__main__':
    # Make sure to run with a single worker for simplicity with asyncio.run and caching
    # For production, a proper ASGI server (like Gunicorn with Uvicorn workers for FastAPI/Quart)
    # or a WSGI server that handles async well would be needed.
    logger.info("Starting Flask API server for GraphRAG...")
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)