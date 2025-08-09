#!/usr/bin/env python3
"""
API Tests for Qwen3 Embedding Server

Run with: python tests/test_api.py
Or with pytest: pytest tests/test_api.py -v
"""

import sys
import time
import json
from typing import List, Dict, Any
import requests
import numpy as np

# Configuration
BASE_URL = "http://localhost:8000"
EMBEDDING_DIM = 1024
TOLERANCE = 0.01  # For float comparisons

class TestClient:
    """Test client for Qwen3 Embedding Server"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        
    def check_server(self) -> bool:
        """Check if server is running"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=2)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def test_health(self) -> Dict[str, Any]:
        """Test health endpoint"""
        response = self.session.get(f"{self.base_url}/health")
        assert response.status_code == 200, f"Health check failed: {response.status_code}"
        
        data = response.json()
        assert "status" in data
        assert "model_status" in data
        assert "embedding_dim" in data
        assert data["embedding_dim"] == EMBEDDING_DIM
        
        return data
    
    def test_single_embedding(self) -> Dict[str, Any]:
        """Test single text embedding"""
        test_text = "Machine learning is transforming the world"
        
        response = self.session.post(
            f"{self.base_url}/embed",
            json={"text": test_text, "normalize": True}
        )
        
        assert response.status_code == 200, f"Embedding failed: {response.text}"
        
        data = response.json()
        assert "embedding" in data
        assert "dim" in data
        assert "normalized" in data
        assert "processing_time_ms" in data
        
        # Validate embedding
        embedding = np.array(data["embedding"])
        assert embedding.shape == (EMBEDDING_DIM,), f"Wrong dimension: {embedding.shape}"
        
        # Check normalization
        if data["normalized"]:
            norm = np.linalg.norm(embedding)
            assert abs(norm - 1.0) < TOLERANCE, f"Not normalized: norm={norm}"
        
        return data
    
    def test_batch_embedding(self) -> Dict[str, Any]:
        """Test batch embedding"""
        test_texts = [
            "Python is a great programming language",
            "FastAPI makes building APIs easy",
            "MLX is optimized for Apple Silicon"
        ]
        
        response = self.session.post(
            f"{self.base_url}/embed_batch",
            json={"texts": test_texts, "normalize": True}
        )
        
        assert response.status_code == 200, f"Batch embedding failed: {response.text}"
        
        data = response.json()
        assert "embeddings" in data
        assert "count" in data
        assert "dim" in data
        assert "normalized" in data
        
        # Validate embeddings
        embeddings = np.array(data["embeddings"])
        assert embeddings.shape == (len(test_texts), EMBEDDING_DIM)
        assert data["count"] == len(test_texts)
        
        # Check normalization
        if data["normalized"]:
            for emb in embeddings:
                norm = np.linalg.norm(emb)
                assert abs(norm - 1.0) < TOLERANCE, f"Not normalized: norm={norm}"
        
        return data
    
    def test_empty_text(self) -> None:
        """Test handling of empty text"""
        response = self.session.post(
            f"{self.base_url}/embed",
            json={"text": ""}
        )
        
        assert response.status_code == 422, "Empty text should be rejected"
    
    def test_large_batch(self) -> None:
        """Test handling of large batch"""
        large_texts = ["Test text"] * 100  # Exceeds typical max_batch_size
        
        response = self.session.post(
            f"{self.base_url}/embed_batch",
            json={"texts": large_texts}
        )
        
        # Should either succeed or return 422 for exceeding limit
        assert response.status_code in [200, 422]
    
    def test_similarity(self) -> None:
        """Test semantic similarity"""
        pairs = [
            ("dog", "puppy", 0.3),  # Should be similar
            ("dog", "car", 0.1),     # Should be dissimilar
            ("AI", "artificial intelligence", 0.2),  # Should be similar
        ]
        
        for text1, text2, min_similarity in pairs:
            response = self.session.post(
                f"{self.base_url}/embed_batch",
                json={"texts": [text1, text2]}
            )
            
            assert response.status_code == 200
            embeddings = np.array(response.json()["embeddings"])
            
            # Calculate cosine similarity
            similarity = np.dot(embeddings[0], embeddings[1])
            
            if min_similarity > 0:
                assert similarity >= min_similarity, \
                    f"'{text1}' and '{text2}' similarity {similarity:.3f} < {min_similarity}"
    
    def test_performance(self) -> Dict[str, float]:
        """Test performance metrics"""
        metrics = {}
        
        # Single embedding latency
        times = []
        for _ in range(5):
            start = time.time()
            response = self.session.post(
                f"{self.base_url}/embed",
                json={"text": "Performance test"}
            )
            times.append((time.time() - start) * 1000)
            assert response.status_code == 200
        
        metrics["single_embed_ms"] = np.mean(times[1:])  # Skip first (warmup)
        
        # Batch embedding latency
        times = []
        for _ in range(3):
            start = time.time()
            response = self.session.post(
                f"{self.base_url}/embed_batch",
                json={"texts": ["Test"] * 10}
            )
            times.append((time.time() - start) * 1000)
            assert response.status_code == 200
        
        metrics["batch_10_ms"] = np.mean(times)
        metrics["throughput_per_sec"] = 10000 / metrics["batch_10_ms"]
        
        return metrics

def run_tests():
    """Run all tests"""
    print("🧪 Qwen3 Embedding Server - Test Suite")
    print("=" * 50)
    
    client = TestClient()
    
    # Check server
    if not client.check_server():
        print("❌ Server is not running. Start with: python server.py")
        return False
    
    results = {"passed": 0, "failed": 0}
    
    # Test suite
    tests = [
        ("Health Check", client.test_health),
        ("Single Embedding", client.test_single_embedding),
        ("Batch Embedding", client.test_batch_embedding),
        ("Empty Text Validation", client.test_empty_text),
        ("Large Batch Handling", client.test_large_batch),
        ("Semantic Similarity", client.test_similarity),
        ("Performance Metrics", client.test_performance),
    ]
    
    for test_name, test_func in tests:
        try:
            print(f"\n📋 {test_name}")
            result = test_func()
            
            if result:
                if isinstance(result, dict):
                    for key, value in result.items():
                        if isinstance(value, float):
                            print(f"  ✓ {key}: {value:.2f}")
                        else:
                            print(f"  ✓ {key}: {value}")
            
            print(f"  ✅ Passed")
            results["passed"] += 1
            
        except AssertionError as e:
            print(f"  ❌ Failed: {e}")
            results["failed"] += 1
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results["failed"] += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"Results: {results['passed']} passed, {results['failed']} failed")
    
    return results["failed"] == 0

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)