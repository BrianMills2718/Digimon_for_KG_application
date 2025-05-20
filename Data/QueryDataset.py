import pandas as pd
from torch.utils.data import Dataset
import os
from pathlib import Path # <<< IMPORT ADDED HERE

class RAGQueryDataset(Dataset):
    def __init__(self,data_dir):
        super().__init__()
      
        self.corpus_path = os.path.join(data_dir, "Corpus.json")
        self.qa_path = os.path.join(data_dir, "Question.json")
        
        # Check if QA path exists, if not, create a dummy one for corpus loading to work
        if not os.path.exists(self.qa_path):
            print(f"Warning: '{self.qa_path}' not found. Creating a dummy file for testing corpus loading.")
            # Ensure the directory exists
            # Get the directory part of qa_path
            qa_dir = os.path.dirname(self.qa_path)
            if qa_dir and not os.path.exists(qa_dir): # Check if qa_dir is not empty
                os.makedirs(qa_dir, exist_ok=True)
            with open(self.qa_path, 'w') as f:
                # Write a dummy JSON line if the file needs to exist and be non-empty
                # This allows get_corpus() to be tested independently if Question.json isn't the focus
                f.write('{"question": "dummy", "answer": "dummy", "id": -1}\n')
        
        self.dataset = pd.read_json(self.qa_path, lines=True, orient="records")

    def get_corpus(self):
        # Check if corpus_path exists
        if not os.path.exists(self.corpus_path):
            print(f"Error: Corpus file '{self.corpus_path}' not found.")
            return []
            
        corpus_df = pd.read_json(self.corpus_path, lines=True)
        corpus_list = []
        for i in range(len(corpus_df)):
            corpus_list.append(
                {
                    "title": corpus_df.iloc[i]["title"],
                    # Corrected key from "context" to "content"
                    "content": corpus_df.iloc[i]["content"], 
                    "doc_id": corpus_df.iloc[i]["doc_id"], # Use doc_id from the file
                }
            )
        return corpus_list

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        question = self.dataset.iloc[idx]["question"]
        answer = self.dataset.iloc[idx]["answer"]
        # other_attrs = self.dataset.iloc[idx].drop(["answer", "question"])
        # Ensure 'id' column exists or handle its absence
        item_id = self.dataset.iloc[idx].get("id", idx) # Use 'id' if present, else fallback to idx
        
        # Create a copy to avoid modifying the original DataFrame slice
        other_attrs_series = self.dataset.iloc[idx].copy()
        # Remove 'answer' and 'question' if they exist
        if "answer" in other_attrs_series:
            other_attrs_series = other_attrs_series.drop("answer")
        if "question" in other_attrs_series:
            other_attrs_series = other_attrs_series.drop("question")
        if "id" in other_attrs_series: # Also remove 'id' if it was part of other_attrs
             other_attrs_series = other_attrs_series.drop("id")
        
        other_attrs_dict = other_attrs_series.to_dict()

        return {"id": item_id, "question": question, "answer": answer, **other_attrs_dict}


if __name__ == "__main__":
    # This block is for testing Data/QueryDataset.py directly.
    # It should NOT be part of test_rag_dataset.py.
    print("--- Running Data/QueryDataset.py directly for self-test ---")
    
    test_data_dir_name = "TestDatasetInternal" # Use a different name to avoid conflict
    # Assuming this script (QueryDataset.py) is in a subdirectory like 'Data'
    # and the project root is one level up.
    current_script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    project_root_for_test = current_script_dir.parent 
    
    test_data_full_path = project_root_for_test / "Data" / test_data_dir_name
    
    if not os.path.exists(test_data_full_path):
        os.makedirs(test_data_full_path, exist_ok=True)
        print(f"Created directory: {test_data_full_path}")

    dummy_corpus_path = os.path.join(test_data_full_path, "Corpus.json")
    with open(dummy_corpus_path, 'w') as f:
        f.write('{"title": "SelfTest Title 1", "content": "SelfTest content 1.", "doc_id": 0}\n')
        f.write('{"title": "SelfTest Title 2", "content": "SelfTest content 2.", "doc_id": 1}\n')
    print(f"Created dummy corpus for self-test: {dummy_corpus_path}")

    dummy_qa_path = os.path.join(test_data_full_path, "Question.json")
    with open(dummy_qa_path, 'w') as f:
        f.write('{"id": 200, "question": "What is self-test 1?", "answer": "Content SelfTest 1"}\n')
        f.write('{"id": 201, "question": "What is self-test 2?", "answer": "Content SelfTest 2"}\n')
    print(f"Created dummy questions for self-test: {dummy_qa_path}")
        
    print(f"Initializing RAGQueryDataset with data_dir='{test_data_full_path}' for self-test")
    query_dataset = RAGQueryDataset(data_dir=test_data_full_path)
    
    print("\n--- Self-Testing get_corpus() ---")
    corpus = query_dataset.get_corpus()
    print(f"Loaded corpus for self-test: {corpus}")
    
    print("\n--- Self-Testing __len__() ---")
    print(f"Dataset length for self-test: {len(query_dataset)}")
    
    print("\n--- Self-Testing __getitem__(0) ---")
    if len(query_dataset) > 0:
        print(f"First item for self-test: {query_dataset[0]}")
    else:
        print("Dataset is empty for self-test.")
    print("--- End of Data/QueryDataset.py self-test ---")

