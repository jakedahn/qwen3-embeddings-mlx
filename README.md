# Qwen3 Embeddings Server

A high-performance text embedding server using the Qwen3 model on Apple Silicon. Built with FastAPI and MLX for optimal performance on M1/M2/M3 Macs.

## 🌟 Features

- **🚀 Optimized for Apple Silicon**: Leverages MLX framework for Metal acceleration
- **⚡ Fast Inference**: ~10-20ms per embedding after warmup
- **🎯 Multiple Model Support**: Choose between 0.6B, 4B, and 8B models
- **📦 Simple Deployment**: Single Python file, minimal dependencies
- **🔄 Batch Processing**: Efficient batch embedding with automatic chunking
- **💾 Smart Caching**: LRU cache for frequently requested embeddings
- **📊 Production Ready**: Health checks, metrics, proper error handling
- **🔒 CORS Support**: Configurable CORS for web applications
- **📝 Full Documentation**: Interactive API docs at `/docs`

## 📋 Requirements

- Apple Silicon Mac (M1/M2/M3)
- Python 3.9+
- ~1GB free RAM for 4-bit quantized model
- macOS 15.0+ (Monterey or later)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/qwen3-embeddings.git
cd qwen3-embeddings
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or for exact versions:

```bash
pip install -r requirements-lock.txt
```

### 3. Run the Server

```bash
python server.py
```

The server will:

1. Download the Qwen3 model (~900MB) on first run
2. Warm up the model (compile Metal kernels)
3. Start serving at `http://localhost:8000`

### 4. Test the API

```bash
# Single embedding
curl -X POST "http://localhost:8000/embed" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!"}'

# Batch embeddings
curl -X POST "http://localhost:8000/embed_batch" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello", "World"]}'

# Health check
curl http://localhost:8000/health
```

## 📖 API Documentation

### Interactive Documentation

Once the server is running, visit:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Available Models

The server supports three Qwen3 embedding models:

| Model | Alias | Parameters | Embedding Dim | Description |
|-------|-------|------------|---------------|-------------|
| Qwen3-Embedding-0.6B | `small`, `0.6b`, `default` | 0.6B | 1024 | Fast and efficient |
| Qwen3-Embedding-4B | `medium`, `4b` | 4B | 2560 | Balanced performance |
| Qwen3-Embedding-8B | `large`, `8b` | 8B | 4096 | Higher quality embeddings |

### Endpoints

#### `POST /embed`

Generate embedding for a single text.

**Request:**

```json
{
  "text": "Your text here",
  "model": "medium",  // Optional: "small", "medium", "large", or full model name
  "normalize": true
}
```

**Response:**

```json
{
  "embedding": [0.123, -0.456, ...],
  "model": "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
  "dim": 1024,
  "normalized": true,
  "processing_time_ms": 15.2
}
```

#### `POST /embed_batch`

Generate embeddings for multiple texts.

**Request:**

```json
{
  "texts": ["Text 1", "Text 2", "Text 3"],
  "model": "medium",  // Optional: "small", "medium", "large", or full model name
  "normalize": true
}
```

**Response:**

```json
{
  "embeddings": [[...], [...], [...]],
  "model": "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
  "dim": 1024,
  "count": 3,
  "normalized": true,
  "processing_time_ms": 42.1
}
```

#### `GET /health`

Health check endpoint for monitoring.

**Response:**

```json
{
  "status": "healthy",
  "model_status": "ready",
  "model_name": "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
  "embedding_dim": 1024,
  "memory_usage_mb": 920.5,
  "uptime_seconds": 3600.0
}
```

#### `GET /metrics`

Detailed metrics and configuration.

#### `GET /models`

List available models and their current status.

## 🔧 Configuration

Configure the server using environment variables:

| Variable          | Description             | Default                                       |
| ----------------- | ----------------------- | --------------------------------------------- |
| `MODEL_NAME`      | MLX model to use        | `mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ` |
| `PORT`            | Server port             | `8000`                                        |
| `HOST`            | Server host             | `0.0.0.0`                                     |
| `MAX_BATCH_SIZE`  | Maximum texts per batch | `32`                                          |
| `MAX_TEXT_LENGTH` | Maximum tokens per text | `8192`                                        |
| `LOG_LEVEL`       | Logging level           | `INFO`                                        |
| `ENABLE_CORS`     | Enable CORS             | `true`                                        |
| `CORS_ORIGINS`    | Allowed CORS origins    | `*`                                           |
| `DEV_MODE`        | Enable auto-reload      | `false`                                       |

### Example with Custom Configuration

```bash
MODEL_NAME=mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ \
PORT=8080 \
MAX_BATCH_SIZE=64 \
LOG_LEVEL=DEBUG \
python server.py
```

## 💻 Usage Examples

### Python Client

```python
import requests
import numpy as np

class EmbeddingClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url

    def embed(self, text: str, model: str = None) -> np.ndarray:
        response = requests.post(
            f"{self.base_url}/embed",
            json={"text": text, "model": model}
        )
        return np.array(response.json()["embedding"])

    def embed_batch(self, texts: list, model: str = None) -> np.ndarray:
        response = requests.post(
            f"{self.base_url}/embed_batch",
            json={"texts": texts, "model": model}
        )
        return np.array(response.json()["embeddings"])

    def list_models(self):
        response = requests.get(f"{self.base_url}/models")
        return response.json()

# Usage
client = EmbeddingClient()

# Use default model (small)
embedding = client.embed("Machine learning is amazing")
print(f"Shape: {embedding.shape}")  # (1024,)

# Use medium model
embedding_medium = client.embed("Machine learning is amazing", model="medium")
print(f"Shape: {embedding_medium.shape}")  # (2560,)

# Use large model for higher quality
embedding_large = client.embed("Machine learning is amazing", model="large")
print(f"Shape: {embedding_large.shape}")  # (4096,)

# Check available models
models = client.list_models()
print(f"Available models: {models['loaded_models']}")
```

### JavaScript/TypeScript Client

```javascript
async function getEmbedding(text, model = null) {
  const response = await fetch("http://localhost:8000/embed", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, model }),
  });
  const data = await response.json();
  return data.embedding;
}

async function listModels() {
  const response = await fetch("http://localhost:8000/models");
  return await response.json();
}

// Usage
// Use default model
const embedding = await getEmbedding("Hello, world!");
console.log(`Dimension: ${embedding.length}`); // 1024

// Use medium model
const embeddingMedium = await getEmbedding("Hello, world!", "medium");
console.log(`Dimension: ${embeddingMedium.length}`); // 2560

// Check available models
const models = await listModels();
console.log("Available models:", models.models);
```

### Semantic Search Example

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Your document corpus
documents = [
    "Machine learning is a subset of artificial intelligence",
    "Python is a popular programming language",
    "The weather today is sunny and warm",
    "Neural networks are inspired by biological neurons"
]

# Generate embeddings for all documents
doc_embeddings = client.embed_batch(documents)

# Search query
query = "AI and deep learning"
query_embedding = client.embed(query)

# Calculate similarities
similarities = cosine_similarity([query_embedding], doc_embeddings)[0]

# Get top results
top_indices = np.argsort(similarities)[-3:][::-1]
for idx in top_indices:
    print(f"Score: {similarities[idx]:.3f} - {documents[idx]}")
```

## 🧪 Testing

Run the test suite:

```bash
# Run API tests
python tests/test_api.py

# Run with pytest (if installed)
pytest tests/
```

## 🚀 Production Deployment

### Using systemd (Linux/macOS)

Create `/etc/systemd/system/qwen3-embeddings.service`:

```ini
[Unit]
Description=Qwen3 Embeddings Server
After=network.target

[Service]
Type=simple
User=youruser
WorkingDirectory=/path/to/qwen3-embeddings
Environment="PATH=/usr/local/bin:/usr/bin"
ExecStart=/usr/bin/python3 /path/to/qwen3-embeddings/server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable qwen3-embeddings
sudo systemctl start qwen3-embeddings
```

### Using Docker (Experimental)

While this project is optimized for native Apple Silicon execution, you can containerize it:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .
EXPOSE 8000

CMD ["python", "server.py"]
```

Note: Docker on macOS doesn't have direct Metal access, so performance will be degraded.

### Reverse Proxy with nginx

```nginx
server {
    listen 80;
    server_name embeddings.yourdomain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 300s;
    }
}
```

## 📊 Performance

### Running Benchmarks

The project includes a comprehensive benchmark script in the tests directory:

```bash
# Quick benchmark
python tests/benchmark.py --quick

# Full benchmark with all tests
python tests/benchmark.py --iterations 100 --workers 10

# Or use the Makefile (recommended)
make benchmark       # Quick benchmark
make benchmark-full  # Comprehensive benchmark
```

Benchmark results are saved to `tests/benchmarks/` which is gitignored.

### Benchmark Results

On M2 Pro MacBook Pro:

| Metric                  | Performance     | Notes              |
| ----------------------- | --------------- | ------------------ |
| **Single Embedding**    |                 |                    |
| - Short text            | ~1.4ms          | 2-3 words          |
| - Medium text           | ~1.3ms          | 10-15 words        |
| - Long text             | ~1.3ms          | 50+ words          |
| **Batch Processing**    |                 |                    |
| - Batch size 1          | 726 texts/sec   |                    |
| - Batch size 10         | 1,887 texts/sec |                    |
| - Batch size 32         | 2,117 texts/sec | Optimal batch size |
| **Concurrent Requests** |                 |                    |
| - 10 workers            | 207 req/sec     |                    |
| - P95 latency           | 7.4ms           |                    |
| - P99 latency           | 8.6ms           |                    |
| **Cache Performance**   |                 |                    |
| - Speedup               | 13.6x           |                    |
| - Cached latency        | ~1.4ms          |                    |
| **Resource Usage**      |                 |                    |
| - Model load time       | ~1s             | Per model          |
| - Memory usage (0.6B)   | ~900MB          |                    |
| - Memory usage (4B)     | ~2.5GB          |                    |
| - Memory usage (8B)     | ~4.5GB          |                    |

### Optimization Tips

1. **Batch Processing**: Group requests for better throughput
2. **Text Length**: Shorter texts process faster
3. **Caching**: Frequently requested embeddings are cached
4. **Warm Start**: Keep the server running to avoid cold starts

## 🔍 Troubleshooting

| Issue                | Solution                                         |
| -------------------- | ------------------------------------------------ |
| Model download fails | Check internet connection and HuggingFace access |
| Out of memory        | Reduce batch size or use system with more RAM    |
| Slow performance     | Ensure running on Apple Silicon, not Intel Mac   |
| Import errors        | Update mlx-lm: `pip install --upgrade mlx-lm`    |

## 🛠️ Development

### Running in Development Mode

```bash
DEV_MODE=true LOG_LEVEL=DEBUG python server.py
```

### Code Structure

```
qwen3-embeddings/
├── server.py           # Main server implementation
├── requirements.txt    # Python dependencies
├── tests/
│   └── test_api.py    # API tests
├── examples/
│   └── client_example.py  # Usage examples
└── README.md          # This file
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [MLX](https://github.com/ml-explore/mlx) - Apple's machine learning framework
- [Qwen](https://github.com/QwenLM/Qwen) - The Qwen model family
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework for Python
- The MLX Community for model conversions

## 📮 Support

For issues and questions:

- Open an issue on GitHub
- Check existing issues for solutions
- Consult the API documentation at `/docs`
