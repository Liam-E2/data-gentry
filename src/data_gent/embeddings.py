import json
from typing import List
from abc import ABC, abstractmethod

from botocore.client import BaseClient


class EmbeddingSource(ABC):
    """
    Interface for a source of text embeddings.
    
    Methods:
        get_embedding: (text: str) => List[float]
    """

    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        pass


class TestEmbeddingSource(EmbeddingSource):
    def get_embedding(self, text: str) -> List[float]:
        return [1.0, 2.0, 3.0]


class BedrockEmbeddingSource(EmbeddingSource):
    """
    Embedding source for AWS Bedrock.
    """
    def __init__(self, client: BaseClient, model_id: str = "amazon.titan-embed-text-v2:0") -> None:
        self.client = client
        self.model_id = model_id


    def get_embedding(self, text: str) -> List[float]:
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps({"inputText": text}),
            contentType="application/json"
        )
        response_body = json.loads(response['body'].read())
        embedding = response_body.get('embedding')
        
        return embedding
