from Core.GraphRAG import GraphRAG
from Option.Config2 import Config
import argparse
import asyncio
import os
import pandas as pd
import json
from pathlib import Path
from shutil import copyfile
from Core.GraphRAG import GraphRAG
from Option.Config2 import Config
from Data.QueryDataset import RAGQueryDataset
from Core.Utils.Evaluation import Evaluator
from Core.Common.Logger import logger # Assuming logger is used

# --- Helper Functions (modified or kept from original) ---
def check_and_create_dirs(opt_config, cmd_args_opt_path):
    """
    Creates directories for results, configs, and metrics.
    Copies relevant configuration files.
    Now takes opt_config and cmd_args_opt_path as parameters.
    """
    # Base directory for the specific dataset/experiment
    base_exp_dir = os.path.join(opt_config.working_dir, opt_config.exp_name) # opt_config.exp_name is dataset_name

    result_dir = os.path.join(base_exp_dir, "Results")
    config_out_dir = os.path.join(base_exp_dir, "Configs") # Renamed to avoid conflict with Config module
    metric_dir = os.path.join(base_exp_dir, "Metrics")

    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(config_out_dir, exist_ok=True)
    os.makedirs(metric_dir, exist_ok=True)

    # Copy the specific method YAML
    if cmd_args_opt_path and os.path.exists(cmd_args_opt_path):
        opt_name = os.path.basename(cmd_args_opt_path)
        copyfile(cmd_args_opt_path, os.path.join(config_out_dir, opt_name))
    else:
        logger.warning(f"Specific option file path not provided or not found: {cmd_args_opt_path}")

    # Copy the base Config2.yaml
    # Assuming Config2.yaml is in the same directory as the specific method YAML's parent 'Option' folder
    if cmd_args_opt_path:
        base_config_path_parts = Path(cmd_args_opt_path).parent.parent / "Config2.yaml" # e.g. Option/Config2.yaml
        if os.path.exists(base_config_path_parts):
            copyfile(base_config_path_parts, os.path.join(config_out_dir, "Config2.yaml"))
        else:
            logger.warning(f"Base Config2.yaml not found at expected location: {base_config_path_parts}")
    
    logger.info(f"Output directories created/verified under: {base_exp_dir}")
    return result_dir, metric_dir # Return metric_dir as well

async def wrapper_query_async(query_dataset, digimon_instance, result_dir_path):
    """
    Asynchronously queries the dataset and saves results.
    Now an async function and takes digimon_instance.
    """
    all_res = []
    # Limiting dataset length for testing, remove or adjust for production
    dataset_len = min(len(query_dataset), 10) 
    
    logger.info(f"Starting queries for {dataset_len} items...")
    for i in range(dataset_len):
        query_item = query_dataset[i]
        try:
            # Assuming digimon_instance is already setup for querying
            res = await digimon_instance.query(query_item["question"])
            query_item["output"] = res
        except Exception as e:
            logger.error(f"Error querying for question ID {query_item.get('id', i)}: {e}")
            query_item["output"] = f"Error: {e}"
        all_res.append(query_item)

    all_res_df = pd.DataFrame(all_res)
    save_path = os.path.join(result_dir_path, "results.json")
    all_res_df.to_json(save_path, orient="records", lines=True)
    logger.info(f"Query results saved to: {save_path}")
    return save_path

async def wrapper_evaluation_async(results_json_path, opt_config, metric_dir_path):
    """
    Asynchronously evaluates results.
    Now an async function.
    """
    if not os.path.exists(results_json_path):
        logger.error(f"Results JSON file not found for evaluation: {results_json_path}")
        return

    eval_instance = Evaluator(results_json_path, opt_config.dataset_name) # opt_config.dataset_name is exp_name
    res_dict = await eval_instance.evaluate()
    
    save_path = os.path.join(metric_dir_path, "metrics.json")
    try:
        with open(save_path, "w", encoding="utf-8") as f:
            # Ensure res_dict is serializable, e.g. convert to string or use json.dump
            f.write(str(res_dict)) 
        logger.info(f"Evaluation metrics saved to: {save_path}")
    except Exception as e:
        logger.error(f"Failed to save metrics: {e}")

# --- Mode Specific Handlers ---
async def handle_build_mode(config_obj, dataset_name_str, graphrag_instance):
    """Handles the 'build' mode: constructs and persists graph artifacts."""
    logger.info(f"Starting 'build' mode for dataset: {dataset_name_str}...")
    
    query_dataset = RAGQueryDataset(
        data_dir=os.path.join(config_obj.data_root, dataset_name_str)
    )
    corpus = query_dataset.get_corpus()
    if not corpus:
        logger.error(f"No corpus data found for {dataset_name_str}. Build process cannot continue.")
        return

    # The 'insert' method will be refactored to 'build_and_persist_artifacts'
    # For now, it performs the build steps.
    await graphrag_instance.build_and_persist_artifacts(corpus) 
    logger.info(f"Build process completed for dataset: {dataset_name_str}.")
    logger.info(f"Artifacts should be saved in: {os.path.join(config_obj.working_dir, dataset_name_str)}")

async def handle_query_mode(config_obj, dataset_name_str, question_str, graphrag_instance):
    """Handles the 'query' mode: loads artifacts and answers a question."""
    logger.info(f"Starting 'query' mode for dataset: {dataset_name_str}...")
    logger.info(f"Question: {question_str}")

    # Placeholder: In future, GraphRAG.py will have a method to load artifacts.
    # await graphrag_instance.setup_for_querying() # This method needs to be implemented
    logger.info("Attempting to setup GraphRAG for querying (loading artifacts)...")
    await graphrag_instance.setup_for_querying() # Call the new setup method

    answer = await graphrag_instance.query(question_str)
    logger.info(f"Answer: {answer}")
    # print(f"Answer: {answer}") # Also print to stdout for direct user feedback

async def handle_evaluate_mode(config_options: Config, dataset_name: str, method_config_path: str):
    logger.info(f"Starting 'evaluate' mode for method config: {method_config_path} on dataset: {dataset_name}...")

    # Initialize GraphRAG instance for the method to be evaluated
    graphrag_instance = GraphRAG(config=config_options)

    # Setup for querying (load artifacts)
    logger.info("Setting up GraphRAG for querying (loading artifacts)...")
    if not await graphrag_instance.setup_for_querying():
        logger.error("Failed to setup GraphRAG for querying. Ensure 'build' mode was run successfully for this configuration.")
        print("Error: Failed to load necessary artifacts for evaluation. Please run 'build' mode first for this method configuration.")
        return

    # Load the query dataset
    query_dataset_path = Path(config_options.data_root) / dataset_name
    if not query_dataset_path.exists():
        logger.error(f"Dataset path not found: {query_dataset_path}")
        print(f"Error: Dataset path not found: {query_dataset_path}")
        return

    logger.info(f"Loading query dataset from: {query_dataset_path}")
    query_dataset = RAGQueryDataset(data_dir=str(query_dataset_path))
    if len(query_dataset) == 0:
        logger.error(f"No questions found in dataset: {dataset_name}")
        print(f"Error: No questions found in dataset: {dataset_name}")
        return

    # --- Querying Phase ---
    all_query_results = []
    num_questions_to_evaluate = len(query_dataset) # Evaluate all for now
    logger.info(f"Processing {num_questions_to_evaluate} questions for evaluation...")

    for i in range(num_questions_to_evaluate):
        query_item = query_dataset[i]
        question_str = query_item["question"]
        logger.info(f"Querying for question #{i+1}: {question_str[:100]}...")

        try:
            generated_answer = await graphrag_instance.query(question_str)
        except Exception as e:
            logger.error(f"Error querying for question '{question_str}': {e}")
            generated_answer = "Error: Query failed."

        query_output_data = query_item.copy() # Start with all original data from Question.json
        query_output_data["output"] = generated_answer # Add the generated answer
        all_query_results.append(query_output_data)

    # Save query results to a temporary JSON file
    method_name = Path(method_config_path).stem # e.g., "LGraphRAG"
    eval_results_base_dir = Path(config_options.working_dir) / config_options.exp_name
    evaluation_output_dir = eval_results_base_dir / "Evaluation_Outputs" / method_name
    os.makedirs(evaluation_output_dir, exist_ok=True)
    query_results_save_path = evaluation_output_dir / f"{dataset_name}_query_outputs_for_eval.json"

    try:
        all_results_df = pd.DataFrame(all_query_results)
        all_results_df.to_json(query_results_save_path, orient="records", lines=True)
        logger.info(f"Query outputs for evaluation saved to: {query_results_save_path}")
    except Exception as e:
        logger.error(f"Failed to save query outputs to JSON: {e}")
        print(f"Error: Failed to save query outputs: {e}")
        return

    # --- Evaluation Phase ---
    logger.info(f"Starting evaluation using results from: {query_results_save_path}")
    try:
        evaluator = Evaluator(str(query_results_save_path), dataset_name)
        metrics_dict = await evaluator.evaluate()
        metrics_save_path = evaluation_output_dir / f"{dataset_name}_evaluation_metrics.json"
        with open(metrics_save_path, "w") as f:
            json.dump(metrics_dict, f, indent=4)
        logger.info(f"Evaluation metrics saved to: {metrics_save_path}")
        print(f"Evaluation complete. Metrics saved to: {metrics_save_path}")
        print("Metrics:", metrics_dict)
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        print(f"Error during evaluation: {e}")

# --- Main Orchestration ---
async def main():
    parser = argparse.ArgumentParser(description="GraphRAG CLI - Build, Query, or Evaluate.")
    parser.add_argument(
        "mode", 
        choices=["build", "query", "evaluate"], 
        help="The operational mode: 'build' to create artifacts, 'query' to ask a question, 'evaluate' to run evaluations."
    )
    parser.add_argument(
        "-opt", 
        type=str, 
        required=True, 
        help="Path to the primary option YAML file (e.g., specific method config)."
    )
    parser.add_argument(
        "-dataset_name", 
        type=str, 
        required=True, 
        help="Name of the dataset/experiment. This will also be used as opt.exp_name."
    )
    parser.add_argument(
        "-question", 
        type=str, 
        help="The question to ask in 'query' mode."
    )
    # Potentially add an argument for corpus_dir if it's different from dataset_name structure
    # parser.add_argument("-corpus_data_dir", type=str, help="Path to the directory containing Corpus.json for 'build' mode, if different from dataset_name convention.")

    args = parser.parse_args()

    # --- Configuration Loading ---
    # dataset_name is also used as opt.exp_name for directory structuring
    opt = Config.parse(Path(args.opt), dataset_name=args.dataset_name) 
    
    # --- GraphRAG Instance ---
    # The GraphRAG instance is created once. Its internal state will be managed by new methods.
    digimon = GraphRAG(config=opt)

    # --- Mode Dispatch ---
    if args.mode == "build":
        # corpus_data_dir = args.corpus_data_dir or os.path.join(opt.data_root, args.dataset_name)
        await handle_build_mode(opt, args.dataset_name, digimon)
    elif args.mode == "query":
        if not args.question:
            parser.error("-question is required for 'query' mode.")
        await handle_query_mode(opt, args.dataset_name, args.question, digimon)
    elif args.mode == "evaluate":
        # Pass the method_config_path for directory naming purposes
        await handle_evaluate_mode(opt, args.dataset_name, args.opt)

if __name__ == "__main__":
    asyncio.run(main())
