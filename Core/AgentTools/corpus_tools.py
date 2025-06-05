"""
Agent tool functions for corpus manipulation and management.
Each function takes its Pydantic input model and processes the corpus accordingly.
"""
import os
import json
import asyncio
from pathlib import Path
from typing import Any, List, Dict, Optional

from Core.AgentSchema.corpus_tool_contracts import PrepareCorpusInputs, PrepareCorpusOutputs
from Core.Common.Logger import logger

async def prepare_corpus_from_directory(
    tool_input: PrepareCorpusInputs,
    main_config=None
) -> PrepareCorpusOutputs:
    """
    Process a directory of .txt files and create a Corpus.json file in JSON Lines format.
    
    This function reads all .txt files from the input directory, processes them into a corpus format,
    and writes the output to Corpus.json in the output directory.
    
    Each document in the corpus will have:
    - 'title' (derived from filename)
    - 'content' (text content of the file)
    - 'doc_id' (sequential identifier)
    
    Args:
        tool_input: PrepareCorpusInputs object containing input_directory_path, output_directory_path,
                   and optionally target_corpus_name
        
    Returns:
        PrepareCorpusOutputs with processing results and corpus file path
    """
    try:
        input_dir = Path(tool_input.input_directory_path)
        output_dir = Path(tool_input.output_directory_path)
        
        # If input_dir is not absolute and doesn't exist, try common data directories
        if not input_dir.is_absolute() and not input_dir.exists():
            # Try under Data/ directory first
            data_path = Path("Data") / input_dir
            if data_path.exists():
                input_dir = data_path
                logger.info(f"Resolved input directory to: {input_dir}")
            else:
                # Try current directory
                if input_dir.exists():
                    input_dir = input_dir.absolute()
                else:
                    logger.warning(f"Could not find input directory: {tool_input.input_directory_path}")
        
        # Handle optional corpus name
        corpus_name = tool_input.target_corpus_name
        if not corpus_name:
            # Derive from output directory name if not provided
            corpus_name = output_dir.name
            logger.info(f"No target_corpus_name provided, using derived name: {corpus_name}")
        
        # Validate input directory
        if not input_dir.is_dir():
            error_msg = f"Input directory '{input_dir}' not found or is not a directory."
            logger.error(error_msg)
            return PrepareCorpusOutputs(
                status="failure",
                message=error_msg,
                document_count=0
            )
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file_path = output_dir / "Corpus.json"
        
        corpus_data = []
        doc_id_counter = 0
        
        logger.info(f"Reading text files from: {input_dir} for corpus: {corpus_name}")
        # Iterate over .txt files in the input directory
        txt_files = sorted(input_dir.glob("*.txt"))  # Sort for consistent doc_ids
        
        if not txt_files:
            warning_msg = f"No .txt files found in the input directory: {input_dir}"
            logger.warning(warning_msg)
            return PrepareCorpusOutputs(
                status="failure",
                message=warning_msg,
                document_count=0,
                corpus_json_path=None
            )
        
        for filepath in txt_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Use filename (without extension) as title
                title = filepath.stem 
                
                corpus_entry = {
                    "title": title,
                    "content": content,
                    "doc_id": doc_id_counter
                }
                corpus_data.append(corpus_entry)
                doc_id_counter += 1
                logger.info(f"Processed: {filepath.name} (doc_id: {doc_id_counter-1})")
            except Exception as e:
                logger.error(f"Error processing file {filepath.name}: {e}")
        
        if not corpus_data:
            warning_msg = "Failed to process any .txt files in the input directory due to errors."
            logger.warning(warning_msg)
            return PrepareCorpusOutputs(
                status="failure",
                message=warning_msg,
                document_count=0,
                corpus_json_path=None
            )
        
        # Write to Corpus.json in JSON Lines format
        try:
            with open(output_file_path, 'w', encoding='utf-8') as outfile:
                for entry in corpus_data:
                    json.dump(entry, outfile)
                    outfile.write('\n')
            
            success_msg = f"Successfully created Corpus.json with {len(corpus_data)} documents for corpus '{corpus_name}'"
            logger.info(f"{success_msg} at: {output_file_path}")
            
            return PrepareCorpusOutputs(
                corpus_json_path=str(output_file_path),
                document_count=len(corpus_data),
                status="success",
                message=success_msg
            )
        except Exception as e:
            error_msg = f"Error writing Corpus.json: {e}"
            logger.error(error_msg)
            return PrepareCorpusOutputs(
                status="failure",
                message=error_msg,
                document_count=0,
                corpus_json_path=None
            )
            
    except Exception as e:
        error_msg = f"Unexpected error in prepare_corpus_from_directory: {e}"
        logger.exception(error_msg)
        return PrepareCorpusOutputs(
            status="failure",
            message=error_msg,
            document_count=0,
            corpus_json_path=None
        )
