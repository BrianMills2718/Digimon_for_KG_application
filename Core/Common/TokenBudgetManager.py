"""
Token Budget Manager for handling model token limits gracefully
"""

from typing import Dict, List, Tuple, Optional
from Core.Utils.TokenCounter import TOKEN_MAX, count_input_tokens
from Core.Common.Logger import logger


class TokenBudgetManager:
    """Manages token budgets to prevent exceeding model limits"""
    
    # Model-specific completion token limits
    COMPLETION_LIMITS = {
        "o4-mini": 100000,
        "gpt-4o-mini": 16384,
        "gpt-4-turbo": 4096,
        "claude-3-opus": 4096,
        "claude-3-sonnet": 4096,
    }
    
    # Safety buffers for different operations
    SAFETY_BUFFERS = {
        "default": 1000,
        "ontology_generation": 2000,
        "graph_extraction": 1500,
        "summarization": 500,
    }
    
    @staticmethod
    def get_model_limits(model: str) -> Tuple[int, int]:
        """Get total context and completion limits for a model
        
        Returns:
            (total_context_limit, max_completion_limit)
        """
        total_limit = TOKEN_MAX.get(model, 128000)
        
        # Handle model name variations
        model_key = model
        if "/" in model:
            model_key = model.split("/")[-1]
        
        completion_limit = TokenBudgetManager.COMPLETION_LIMITS.get(
            model_key, 
            min(total_limit // 2, 100000)  # Default to half of total or 100k
        )
        
        return total_limit, completion_limit
    
    @staticmethod
    def calculate_safe_tokens(
        messages: List[Dict[str, str]], 
        model: str, 
        operation: str = "default",
        requested_tokens: Optional[int] = None
    ) -> int:
        """Calculate safe max tokens for an operation
        
        Args:
            messages: Input messages
            model: Model name
            operation: Type of operation (for safety buffer)
            requested_tokens: Optional requested token count
            
        Returns:
            Safe number of completion tokens to request
        """
        try:
            # Get model limits
            total_limit, completion_limit = TokenBudgetManager.get_model_limits(model)
            
            # Count input tokens
            input_tokens = count_input_tokens(messages, model)
            
            # Get safety buffer
            safety_buffer = TokenBudgetManager.SAFETY_BUFFERS.get(
                operation, 
                TokenBudgetManager.SAFETY_BUFFERS["default"]
            )
            
            # Calculate available tokens
            available_in_context = total_limit - input_tokens - safety_buffer
            
            # Apply completion limit
            safe_tokens = min(available_in_context, completion_limit)
            
            # Apply requested limit if provided
            if requested_tokens:
                safe_tokens = min(safe_tokens, requested_tokens)
            
            # Ensure positive
            safe_tokens = max(safe_tokens, 100)  # Minimum 100 tokens
            
            logger.debug(
                f"Token budget for {operation}: "
                f"input={input_tokens}, "
                f"available={available_in_context}, "
                f"completion_limit={completion_limit}, "
                f"safe={safe_tokens}"
            )
            
            return safe_tokens
            
        except Exception as e:
            logger.warning(f"Error calculating token budget: {e}, using default 4000")
            return 4000
    
    @staticmethod
    def chunk_messages_for_model(
        messages: List[Dict[str, str]], 
        model: str,
        target_chunks: int = None
    ) -> List[List[Dict[str, str]]]:
        """Split messages into chunks that fit within model limits
        
        Args:
            messages: Messages to chunk
            model: Model name
            target_chunks: Target number of chunks (None = auto)
            
        Returns:
            List of message chunks
        """
        total_limit, _ = TokenBudgetManager.get_model_limits(model)
        
        # Reserve space for system messages and safety
        available_per_chunk = int(total_limit * 0.7)  # Use 70% of limit
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for msg in messages:
            msg_tokens = count_input_tokens([msg], model)
            
            if current_tokens + msg_tokens > available_per_chunk and current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_tokens = 0
            
            current_chunk.append(msg)
            current_tokens += msg_tokens
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    @staticmethod
    def should_chunk_content(content: str, model: str, operation: str = "default") -> bool:
        """Check if content needs to be chunked
        
        Args:
            content: Content to check
            model: Model name
            operation: Type of operation
            
        Returns:
            True if content should be chunked
        """
        # Create a simple message to test
        test_message = [{"role": "user", "content": content}]
        
        try:
            input_tokens = count_input_tokens(test_message, model)
            total_limit, _ = TokenBudgetManager.get_model_limits(model)
            safety_buffer = TokenBudgetManager.SAFETY_BUFFERS.get(
                operation, 
                TokenBudgetManager.SAFETY_BUFFERS["default"]
            )
            
            # Check if we're using more than 50% of context
            return input_tokens > (total_limit - safety_buffer) * 0.5
            
        except Exception as e:
            logger.warning(f"Error checking if chunking needed: {e}")
            # Err on the side of caution
            return len(content) > 10000  # Chunk if over 10k chars
    
    @staticmethod
    def chunk_text_by_tokens(
        text: str, 
        model: str, 
        max_tokens_per_chunk: int = None
    ) -> List[str]:
        """Chunk text to fit within token limits
        
        Args:
            text: Text to chunk
            model: Model name
            max_tokens_per_chunk: Max tokens per chunk (None = auto)
            
        Returns:
            List of text chunks
        """
        if max_tokens_per_chunk is None:
            total_limit, _ = TokenBudgetManager.get_model_limits(model)
            max_tokens_per_chunk = int(total_limit * 0.5)  # Use 50% of limit
        
        # Simple character-based chunking (rough approximation)
        # Approximately 4 characters per token
        chars_per_chunk = max_tokens_per_chunk * 4
        
        chunks = []
        for i in range(0, len(text), chars_per_chunk):
            chunk = text[i:i + chars_per_chunk]
            chunks.append(chunk)
        
        return chunks