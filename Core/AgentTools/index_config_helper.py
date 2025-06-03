"""
Helper module for creating proper index configurations in agent tools.
"""

import os
from typing import Optional
from pathlib import Path

from Core.Index.Schema import FAISSIndexConfig, VectorIndexConfig
from Core.Index.EmbeddingFactory import get_rag_embedding
from Core.Common.Logger import logger
from llama_index.core.embeddings import BaseEmbedding


def create_faiss_index_config(
    persist_path: str,
    embed_model: Optional[BaseEmbedding] = None,
    config: Optional[any] = None,
    name: Optional[str] = None
) -> FAISSIndexConfig:
    """
    Create a proper FAISSIndexConfig for use in agent tools.
    
    Args:
        persist_path: Path where the index will be persisted
        embed_model: Embedding model to use (if None, will create from config)
        config: Optional config object with embedding settings
        name: Optional name for the index
        
    Returns:
        FAISSIndexConfig properly configured
    """
    # Ensure persist path exists
    persist_path = Path(persist_path)
    persist_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get embedding model if not provided
    if embed_model is None and config is not None:
        try:
            embed_model = get_rag_embedding(config=config)
            logger.info("Created embedding model from config")
        except Exception as e:
            logger.error(f"Failed to create embedding model: {e}")
            raise
    
    # Create the config
    faiss_config = FAISSIndexConfig(
        persist_path=str(persist_path),
        embed_model=embed_model
    )
    
    # Add optional name as metadata if needed
    if name:
        # Store name in a separate attribute or use for logging
        logger.info(f"Created FAISS index config with name: {name}")
    
    return faiss_config


def create_vector_index_config(
    persist_path: str,
    embed_model: Optional[BaseEmbedding] = None,
    config: Optional[any] = None
) -> VectorIndexConfig:
    """
    Create a proper VectorIndexConfig for use in agent tools.
    
    Args:
        persist_path: Path where the index will be persisted
        embed_model: Embedding model to use (if None, will create from config)
        config: Optional config object with embedding settings
        
    Returns:
        VectorIndexConfig properly configured
    """
    # Ensure persist path exists
    persist_path = Path(persist_path)
    persist_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get embedding model if not provided
    if embed_model is None and config is not None:
        try:
            embed_model = get_rag_embedding(config=config)
            logger.info("Created embedding model from config")
        except Exception as e:
            logger.error(f"Failed to create embedding model: {e}")
            raise
    
    # Create the config
    vector_config = VectorIndexConfig(
        persist_path=str(persist_path),
        embed_model=embed_model
    )
    
    return vector_config