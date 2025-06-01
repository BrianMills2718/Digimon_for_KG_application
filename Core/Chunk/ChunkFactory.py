from typing import Any, List, Tuple, Dict, Optional
from Core.Common.Utils import mdhash_id
from collections import defaultdict
import json
import os
from pathlib import Path
import logging
from loguru import logger

from Core.Schema.ChunkSchema import TextChunk


class ChunkingFactory:
    chunk_methods: dict = defaultdict(Any)

    def register_chunking_method(
            self,
            method_name: str,
            method_func=None  # can be any classes or functions
    ):
        if self.has_chunk_method(method_name):
            return

        self.chunk_methods[method_name] = method_func

    def has_chunk_method(self, key: str) -> Any:
        return key in self.chunk_methods

    def get_method(self, key) -> Any:
        return self.chunk_methods.get(key)


# Registry instance
CHUNKING_REGISTRY = ChunkingFactory()


def register_chunking_method(method_name):
    """ Register a new chunking method
    
    This is a decorator that can be used to register a new chunking method.
    The method will be stored in the self.methods dictionary.
    
    Parameters
    ----------
    method_name: str
        The name of the chunking method.
    """

    def decorator(func):
        """ Register a new chunking method """
        CHUNKING_REGISTRY.register_chunking_method(method_name, func)

    return decorator


def create_chunk_method(method_name):
    chunking_method = CHUNKING_REGISTRY.get_method(method_name)
    return chunking_method


class ChunkFactory:
    """
    A factory class for loading and processing text chunks from corpus files.
    Supports JSONL (JSON Lines) format where each line contains a separate JSON object.
    """
    
    def __init__(self, config):
        """
        Initialize the ChunkFactory with configuration.
        
        Args:
            config: Main configuration object containing settings such as working_dir and data_root
        """
        self.main_config = config
        self.workspaces = {}
        logger.info(f"ChunkFactory initialized with working_dir: {self.main_config.working_dir}")
    
    def get_namespace(self, dataset_name, graph_type="er_graph"):
        """
        Get or create a namespace for the given dataset and graph type.
        
        Args:
            dataset_name: Name of the dataset
            graph_type: Type of graph, defaults to 'er_graph'
            
        Returns:
            Namespace object for the dataset and graph type
        """
        # Ensure dataset_name is not empty
        if not dataset_name:
            raise ValueError("dataset_name cannot be empty for creating a namespace.")
        
        # Create namespace path based on working_dir and dataset_name
        class Namespace:
            def __init__(self, path):
                self.path = path
                
            def get_save_path(self, suffix=None):
                # Return a path for saving the graph file
                if suffix:
                    return str(Path(self.path) / suffix)
                return str(Path(self.path))
        
        # Check both possible locations for the dataset
        # First try working_dir (for intermediate output from prepare corpus tool)
        working_dir_path = Path(self.main_config.working_dir) / dataset_name / graph_type
        # Alternatively check data_root (for pre-existing datasets)
        data_root_path = Path(self.main_config.data_root) / dataset_name / graph_type
        
        # Choose the path that exists, prioritize working_dir
        if working_dir_path.exists():
            namespace_path = working_dir_path
        elif data_root_path.exists():
            namespace_path = data_root_path
        else:
            # Default to working_dir and create the directory
            namespace_path = working_dir_path
            
        # Create the directory if it doesn't exist
        namespace_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"ChunkFactory: Created/ensured namespace for dataset '{dataset_name}', type '{graph_type}' at {namespace_path}")
        
        return Namespace(str(namespace_path))
    
    async def get_chunks_for_dataset(self, dataset_name) -> List[Tuple[str, TextChunk]]:
        """
        Load corpus file for the given dataset and convert to chunks.
        Handles JSONL format (one JSON object per line).
        
        Args:
            dataset_name: Name of the dataset to load
            
        Returns:
            List of tuples (chunk_key, TextChunk)
        """
        logger.info(f"ChunkFactory: Getting chunks for dataset '{dataset_name}'")
        
        # Look for Corpus.json in multiple possible locations
        possible_corpus_paths = [
            # Check working_dir first (for output from PrepareCorpusFromDirectoryTool)
            Path(self.main_config.working_dir) / dataset_name / "Corpus.json",
            # Then check data_root (for pre-existing datasets)
            Path(self.main_config.data_root) / dataset_name / "Corpus.json"
        ]
        
        corpus_path = None
        for path in possible_corpus_paths:
            if path.exists():
                corpus_path = path
                logger.info(f"Found corpus file at {corpus_path}")
                break
        
        if not corpus_path:
            logger.error(f"Corpus file not found for dataset '{dataset_name}'")
            return []
        
        try:
            # Process the Corpus.json file in JSONL format (one JSON object per line)
            documents = []
            with open(corpus_path, 'r', encoding='utf-8') as f:
                for line_number, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    try:
                        doc = json.loads(line)
                        documents.append(doc)
                    except json.JSONDecodeError as e:
                        logger.error(f"Error decoding JSON object on line {line_number} in {corpus_path}: {e} - Line content: '{line[:100]}...'")
                        # Skip problematic line and continue with next
            
            logger.info(f"Successfully loaded {len(documents)} documents from JSONL file: {corpus_path}")
            
            # Convert documents to TextChunk objects
            chunks = []
            for i, doc in enumerate(documents):
                # Extract document properties - format may vary
                doc_id = doc.get("doc_id", f"doc_{i}")
                title = doc.get("title", f"Document {doc_id}")
                content = doc.get("content", "")
                
                # Use a hash-based chunk_id if none is provided
                chunk_id = doc.get("chunk_id", f"chunk_{doc_id}")
                
                # Create TextChunk object
                chunk = TextChunk(
                    tokens=len(content.split()),  # Estimate token count
                    chunk_id=chunk_id,
                    content=content,
                    doc_id=doc_id,
                    index=i,
                    title=title
                )
                
                # Add tuple of (chunk_key, TextChunk) to the result list
                chunks.append((chunk_id, chunk))
            
            logger.info(f"Created {len(chunks)} TextChunk objects for dataset '{dataset_name}'")
            return chunks
            
        except Exception as e:
            logger.error(f"Error loading chunks from corpus: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
