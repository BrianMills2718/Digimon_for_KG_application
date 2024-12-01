from dataclasses import dataclass, asdict

@dataclass
class TextChunk:
    def __init__(self, tokens, content: str, doc_id: str, chunk_order_index: int):
        self.tokens: int = tokens
        self.content: str  = content
        self.doc_id: str = doc_id
        self.chunk_order_index: int = chunk_order_index

    @property
    def as_dict(self):
        return asdict(self)