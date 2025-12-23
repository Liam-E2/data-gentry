import json
from typing import List
from abc import ABC, abstractmethod
from collections import OrderedDict

from types_boto3_bedrock_runtime import BedrockRuntimeClient
from .settings import settings


class EmbeddingSource(ABC):
    """
    Interface for a source of text embeddings with LRU caching.

    Methods:
        get_embedding: (text: str) => List[float]  # Public method with caching
        _get_embedding: (text: str) => List[float]  # Override this in subclasses
    """

    def __init__(self, cache_size: int = 10000):
        """Initialize with an LRU cache for embeddings."""
        self._cache: OrderedDict[str, List[float]] = OrderedDict()
        self._cache_size = cache_size


    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text, using cache if available."""
        if text in self._cache:
            self._cache.move_to_end(text)
            return self._cache[text]

        embedding = self._get_embedding(text)
        self._cache[text] = embedding
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

        return embedding


    @abstractmethod
    def _get_embedding(self, text: str) -> List[float]:
        """Implement this method to generate embeddings."""
        raise NotImplementedError()


class TestEmbeddingSource(EmbeddingSource):
    def __init__(self, cache_size: int = 10_000, **kwargs):
        super().__init__(cache_size=cache_size, **kwargs)

    def _get_embedding(self, text: str) -> List[float]:
        return [1.0 for _ in range(settings.vec_size)]


class BedrockEmbeddingSource(EmbeddingSource):
    """
    Embedding source for AWS Bedrock.
    """
    def __init__(self, client: BedrockRuntimeClient, model_id: str = "amazon.titan-embed-text-v2:0", cache_size: int = 10_000, **kwargs) -> None:
        super().__init__(cache_size=cache_size, **kwargs)
        self.client = client
        self.model_id = model_id


    def _get_embedding(self, text: str) -> List[float]:
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps({"inputText": text}),
            contentType="application/json"
        )
        response_body = json.loads(response['body'].read())
        embedding = response_body.get('embedding')

        if len(embedding) != settings.vec_size:
            raise ValueError(f"Got embeddings of length {len(embedding)}; DATAGENT_VEC_SIZE={settings.vec_size}")

        return embedding
