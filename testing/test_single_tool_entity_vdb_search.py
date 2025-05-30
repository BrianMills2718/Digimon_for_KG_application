# testing/test_single_tool_entity_vdb_search.py

import asyncio
import os
from typing import List, Optional, Any, Dict, Tuple

# Imports for the Pydantic models we are testing
from Core.AgentSchema.context import GraphRAGContext
from Core.AgentSchema.tool_contracts import EntityVDBSearchInputs, EntityVDBSearchOutputs

# Import the tool function we want to test
from Core.AgentTools.entity_tools import entity_vdb_search_tool

from Core.Index.EmbeddingFactory import get_rag_embedding
from Option.Config2 import Config
from Config.EmbConfig import EmbeddingType
from Core.Storage.NameSpace import NameSpace

# --- Mock or Simplified Components for Testing ---

    # Implement other abstract methods from BaseEmb as needed for compatibility

async def main_test():
    print("--- Starting Test for entity_vdb_search_tool ---")

    # --- 1. Setup Test Parameters ---
    target_dataset = "MySampleTexts"
    vdb_ref_id = "entities_vdb" # This is the VDB name created by the LGraphRAG build
    results_root_dir = "./results"
    # Directly construct the expected VDB path
    expected_vdb_path = os.path.join(results_root_dir, target_dataset, "kg_graph", vdb_ref_id)
    faiss_file = os.path.join(expected_vdb_path, "vector_index.faiss")

    print(f"Expected VDB path: {expected_vdb_path}")
    if not os.path.exists(faiss_file):
        print(f"WARNING: Test Faiss VDB not found at '{faiss_file}'.")
        print(f"Please ensure a Faiss index (vector_index.faiss and id_mapping.pkl) exists at this location for dataset '{target_dataset}' and VDB reference '{vdb_ref_id}'.")
        print("The tool will likely return empty or fail to load the VDB.")

    # --- Load Main Configuration ---
    main_config: Optional[Config] = None
    try:
        main_config = Config.from_yaml_file("Option/Config2.yaml")
        print(f"INFO: Loaded main config from 'Option/Config2.yaml'. Embedding type: {getattr(main_config.embedding, 'api_type', 'N/A')}")
        # Optionally check for specific embedding type/model here
        # if main_config.embedding.api_type != EmbeddingType.OPENAI or main_config.embedding.model != "text-embedding-3-small":
        #     print("WARNING: Embedding config in Option/Config2.yaml does not match expected OpenAI 'text-embedding-3-small'.")
        #     print(f"Actual api_type: {main_config.embedding.api_type}, model: {main_config.embedding.model}")
    except FileNotFoundError:
        print(f"ERROR: Main configuration file 'Option/Config2.yaml' not found. Please ensure it exists.")
        print("Test cannot proceed without main_config.")
        return
    except Exception as e:
        print(f"ERROR: Could not load or parse main configuration from 'Option/Config2.yaml'. Error: {e}")
        print("Please ensure 'Option/Config2.yaml' is correctly formatted and all necessary API keys (e.g., OpenAI) are set in your environment if the config expects them.")
        import traceback
        traceback.print_exc()
        return
    if main_config is None:
        print("ERROR: main_config was not loaded. Exiting test.")
        return

    try:
        llama_index_embedding_provider = get_rag_embedding(config=main_config)
        print(f"Successfully created LlamaIndex embedding provider: {type(llama_index_embedding_provider)}")
    except Exception as e:
        print(f"ERROR: Failed to create LlamaIndex embedding provider using EmbeddingFactory: {e}")
        print("Please check your embedding configuration in 'Option/Config2.yaml' and ensure any necessary services (like Ollama) are running or API keys are set.")
        import traceback
        traceback.print_exc()
        return

    test_context = GraphRAGContext(
        target_dataset_name=target_dataset,
        resolved_configs={
            "storage_root_dir": results_root_dir,
            "main_config_dict": main_config.model_dump()
        },
        embedding_provider=llama_index_embedding_provider,
    )

    # --- 3. Prepare Tool Inputs ---
    tool_inputs = EntityVDBSearchInputs(
        vdb_reference_id=vdb_ref_id,
        query_text="What were the key causes of the American Revolution?",
        embedding_model_id="mock_embedding_model",
        top_k_results=3
    )

    # --- 4. Call the Tool ---
    print(f"\nCalling entity_vdb_search_tool with inputs: {tool_inputs}")
    try:
        result_outputs: EntityVDBSearchOutputs = await entity_vdb_search_tool(
            params=tool_inputs,
            graphrag_context=test_context
        )
        print("\n--- Tool Execution Finished ---")
        print("Tool Outputs:")
        if result_outputs and result_outputs.similar_entities:
            for entity_id, score in result_outputs.similar_entities:
                print(f"  Entity ID: {entity_id}, Score: {score:.4f}")
        else:
            print("  No similar entities returned or an error occurred.")

    except Exception as e:
        print(f"\n--- Tool Execution Error ---")
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Test Script Finished ---")

if __name__ == "__main__":
    asyncio.run(main_test())
