from abc import ABC, abstractmethod
from typing import Iterable, Callable

from semchunk import chunkerify


class Chunker(ABC):
    """
    Interface for a class which splits a document into chunks.
    """
    
    @abstractmethod
    def chunk(self, text: str) -> Iterable[str]:
        """
        Break input text into a sequence of strings.
        """
        raise NotImplementedError()
 

class SemchunkChunker:
    def __init__(self, 
                 tokenizer: str | Callable[[str], int] = lambda text: len(text.split()),
                 chunk_size: int = 120,
                 overlap: float = 0.15
                 ):
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunker = chunkerify(self.tokenizer, self.chunk_size)


    def chunk(self, text: str) -> list[str]:
        return self.chunker(text, overlap=self.overlap) # type: ignore
