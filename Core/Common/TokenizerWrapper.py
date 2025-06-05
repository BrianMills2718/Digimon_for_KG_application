"""Simple tokenizer wrapper for BaseGraph compatibility"""

from tiktoken import get_encoding

class TokenizerWrapper:
    """
    Wrapper to provide encode/decode methods for BaseGraph compatibility.
    Uses tiktoken for tokenization.
    """
    
    def __init__(self, model_name="gpt-3.5-turbo"):
        # Use cl100k_base encoding which is compatible with most models
        self.encoding = get_encoding("cl100k_base")
    
    def encode(self, text: str) -> list:
        """Encode text to tokens"""
        return self.encoding.encode(text)
    
    def decode(self, tokens: list) -> str:
        """Decode tokens to text"""
        return self.encoding.decode(tokens)