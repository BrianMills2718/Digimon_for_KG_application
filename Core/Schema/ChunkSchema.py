from dataclasses import dataclass, asdict

@dataclass
class TextChunk:
    def __init__(self, tokens, content: str, doc_id: str, index: int):
        self.tokens: int = tokens
        self.content: str  = content
        self.doc_id: str = doc_id
        self.index: int = index

    @property
    def as_dict(self):
        return asdict(self)