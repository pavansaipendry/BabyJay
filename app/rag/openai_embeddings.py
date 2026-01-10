# app/rag/openai_embeddings.py
"""
OpenAI Embeddings for ChromaDB - No PyTorch needed!
"""
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class OpenAIEmbeddingFunction:
    """
    Custom embedding function using OpenAI API.
    Compatible with ChromaDB's embedding_function interface.
    """
    
    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
    
    def __call__(self, input: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        if not input:
            return []
        
        # OpenAI API call
        response = client.embeddings.create(
            model=self.model,
            input=input
        )
        
        # Extract embeddings in order
        embeddings = [item.embedding for item in response.data]
        return embeddings


# Singleton instance
openai_ef = OpenAIEmbeddingFunction()


def get_embedding_function():
    """Get the OpenAI embedding function for ChromaDB."""
    return openai_ef
