#!/usr/bin/env python3
"""
Example usage of the Qwen3 Embedding Server
"""

import requests
import numpy as np
from typing import List

class EmbeddingClient:
    """Client for Qwen3 Embedding Server"""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        
    def embed(self, text: str, normalize: bool = True) -> np.ndarray:
        """Get embedding for a single text"""
        response = requests.post(
            f"{self.base_url}/embed",
            json={"text": text, "normalize": normalize}
        )
        response.raise_for_status()
        data = response.json()
        return np.array(data["embedding"], dtype=np.float32)
    
    def embed_batch(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """Get embeddings for multiple texts"""
        response = requests.post(
            f"{self.base_url}/embed_batch",
            json={"texts": texts, "normalize": normalize}
        )
        response.raise_for_status()
        data = response.json()
        return np.array(data["embeddings"], dtype=np.float32)
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def main():
    # Initialize client
    client = EmbeddingClient()
    
    # Example 1: Single embedding
    print("=== Single Embedding Example ===")
    text = "Machine learning is transforming the world"
    embedding = client.embed(text)
    print(f"Text: '{text}'")
    print(f"Embedding shape: {embedding.shape}")
    print(f"First 5 values: {embedding[:5]}")
    
    # Example 2: Batch embeddings
    print("\n=== Batch Embedding Example ===")
    texts = [
        "Python is a great programming language",
        "JavaScript runs in the browser",
        "Machine learning uses neural networks"
    ]
    embeddings = client.embed_batch(texts)
    print(f"Batch shape: {embeddings.shape}")
    
    # Example 3: Semantic similarity
    print("\n=== Semantic Similarity Example ===")
    query = "artificial intelligence"
    candidates = [
        "machine learning and deep learning",
        "neural networks and AI",
        "cooking pasta recipes",
        "weather forecast for tomorrow"
    ]
    
    # Get embeddings
    query_emb = client.embed(query)
    candidate_embs = client.embed_batch(candidates)
    
    # Calculate similarities
    similarities = []
    for i, candidate in enumerate(candidates):
        sim = client.cosine_similarity(query_emb, candidate_embs[i])
        similarities.append((candidate, sim))
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    print(f"Query: '{query}'")
    print("\nMost similar texts:")
    for text, sim in similarities:
        print(f"  {sim:.4f}: {text}")
    
    # Example 4: Document search simulation
    print("\n=== Document Search Example ===")
    documents = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning algorithms can learn patterns from data",
        "Python is widely used in data science and AI",
        "The weather today is sunny and warm",
        "Neural networks are inspired by biological neurons"
    ]
    
    search_query = "artificial intelligence and programming"
    
    # Embed all documents and query
    doc_embeddings = client.embed_batch(documents)
    query_embedding = client.embed(search_query)
    
    # Find most relevant documents
    results = []
    for i, doc in enumerate(documents):
        score = client.cosine_similarity(query_embedding, doc_embeddings[i])
        results.append((doc, score))
    
    results.sort(key=lambda x: x[1], reverse=True)
    
    print(f"Search query: '{search_query}'")
    print("\nTop 3 relevant documents:")
    for doc, score in results[:3]:
        print(f"  Score {score:.4f}: {doc}")

if __name__ == "__main__":
    main()