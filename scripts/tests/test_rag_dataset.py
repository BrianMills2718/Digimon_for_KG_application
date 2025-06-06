import os
from Data.QueryDataset import RAGQueryDataset # Make sure this import path is correct from your project root
import traceback

def run_test():
    # --- Configuration for the test ---
    # Set this to the directory where your Corpus.json and Question.json are located.
    DATASET_NAME_FOR_TEST = "MySampleTexts" # This should match the directory name you used
    
    base_data_dir = "Data" 
    data_directory_path = os.path.join(base_data_dir, DATASET_NAME_FOR_TEST)
    # --- End Configuration ---

    print(f"Attempting to load RAGQueryDataset from: {data_directory_path}")

    corpus_file_path = os.path.join(data_directory_path, "Corpus.json")
    question_file_path = os.path.join(data_directory_path, "Question.json")

    if not os.path.isdir(data_directory_path):
        print(f"ERROR: The directory '{data_directory_path}' does not exist. Please create it and place your files there.")
        return
    if not os.path.exists(corpus_file_path):
        print(f"ERROR: '{corpus_file_path}' does not exist. Make sure you ran prepare_corpus.py correctly.")
        return
    if not os.path.exists(question_file_path):
        print(f"ERROR: '{question_file_path}' does not exist. Please create the Question.json file in that directory using the content from the Canvas.")
        return
    
    try:
        query_dataset_instance = RAGQueryDataset(data_dir=data_directory_path)
        corpus_list = query_dataset_instance.get_corpus()
        
        print("\n--- Corpus Loaded ---")
        if corpus_list:
            for i, doc in enumerate(corpus_list):
                print(f"Document {i}:")
                print(f"  Title: {doc.get('title')}")
                content_preview = doc.get('content', '')[:50] + "..." if doc.get('content') else "N/A"
                print(f"  Content (first 50 chars): {content_preview}")
                print(f"  Doc ID: {doc.get('doc_id')}")
        else:
            print("Corpus is empty or failed to load.")

        print("\n--- RAGQueryDataset Test ---")
        print(f"Number of items in dataset (from Question.json): {len(query_dataset_instance)}")
        if len(query_dataset_instance) > 0:
            print(f"First item (from Question.json): {query_dataset_instance[0]}")
        else:
            print("No items loaded from Question.json.")

    except Exception as e:
        print(f"An error occurred during the RAGQueryDataset test: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    run_test()
