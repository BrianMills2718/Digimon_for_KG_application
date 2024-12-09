from dataclasses import dataclass, asdict

@dataclass
class TextChunk:
    def __init__(self, tokens, chunk_id: str, content: str, doc_id: str, index: int):
        self.tokens: int = tokens
        self.chunk_id: str = chunk_id   
        self.content: str  = content
        self.doc_id: str = doc_id
        self.index: int = index

    @property
    def as_dict(self):
        return asdict(self)