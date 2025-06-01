"""
Pydantic contracts for corpus management and manipulation tools.
"""
from pydantic import BaseModel, Field
from typing import Optional

class PrepareCorpusInputs(BaseModel):
    input_directory_path: str = Field(description="The directory path containing .txt files to process.")
    output_directory_path: str = Field(description="The directory path where the Corpus.json (JSON Lines format) will be saved. The Corpus.json will be created directly inside this directory.")
    target_corpus_name: Optional[str] = Field(default=None, description="An optional logical name for this corpus, which might be used for dataset registration or naming derived artifacts later. If None, can be derived from output_directory_path's basename.")

class PrepareCorpusOutputs(BaseModel):
    corpus_json_path: Optional[str] = Field(default=None, description="The full path to the generated Corpus.json file.")
    document_count: int = Field(default=0, description="Number of .txt documents successfully processed and added to the corpus.")
    status: str = Field(description="Status of the operation, e.g., 'success', 'failure'.")
    message: str = Field(description="A descriptive message about the outcome, including any errors.")
