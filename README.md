# Qwen3 Embeddings Server for Mac

**Lightning-fast text embeddings on your Mac.** No cloud, no GPU needed – just Apple Silicon magic. 🚀

Works great on all Apple Silicon Macs including the latest M4 chips!

![0.6B Speed](https://img.shields.io/badge/0.6B-44K_tokens/sec-green)
![4B Speed](https://img.shields.io/badge/4B-18K_tokens/sec-blue)
![8B Speed](https://img.shields.io/badge/8B-11K_tokens/sec-purple)
![Platform](https://img.shields.io/badge/Platform-Apple_Silicon-black)

## ✨ What is this?

A simple, fast API server that runs state-of-the-art text embedding models locally on your Mac. Perfect for:

- 🔍 Semantic search
- 🤖 RAG applications
- 📊 Document clustering
- 🎯 Similarity matching

**Performance**: Process 44,000+ tokens/second with the small model, or get higher quality with larger models.

## 🏃 Quick Start (2 minutes)

### Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- Python 3.9+
- 1-5GB free space (depending on model)

### Install & Run

```bash
# Clone and enter directory
git clone https://github.com/yourusername/qwen3-embeddings.git
cd qwen3-embeddings

# Install (one-time, ~30 seconds)
pip install -r requirements.txt

# Run! 🎉
python server.py
```

That's it! The server is now running at `http://localhost:8000`

On first run, it will download the model (~900MB) which takes about a minute.

### Test it works

```bash
# Generate an embedding
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}'
```

## 🎮 Choose Your Model

Three models available, from fast to powerful:

| Model               | Speed            | Quality  | Memory | Use When             |
| ------------------- | ---------------- | -------- | ------ | -------------------- |
| **Small** (default) | ⚡⚡⚡ 44K tok/s | ⭐⭐     | 900MB  | Speed matters most   |
| **Medium**          | ⚡⚡ 18K tok/s   | ⭐⭐⭐   | 2.5GB  | **Best balance** ✨  |
| **Large**           | ⚡ 11K tok/s     | ⭐⭐⭐⭐ | 4.5GB  | Quality matters most |

Use different models per request:

```python
# Python example
import requests

# Fast model for high-volume
requests.post("http://localhost:8000/embed",
    json={"text": "Quick search", "model": "small"})

# Quality model for important documents
requests.post("http://localhost:8000/embed",
    json={"text": "Important document", "model": "large"})
```

## 📖 API Reference

**Interactive docs**: Visit http://localhost:8000/docs when server is running

### Core Endpoints

#### Generate Single Embedding

```bash
POST /embed
{
  "text": "Your text here",
  "model": "small|medium|large"  # optional
}
```

#### Generate Multiple Embeddings

```bash
POST /embed_batch
{
  "texts": ["Text 1", "Text 2", "Text 3"],
  "model": "small|medium|large"  # optional
}
```

#### List Available Models

```bash
GET /models
```

#### Health Check

```bash
GET /health
```

## 💻 Client Examples

### Python

```python
import requests
import numpy as np

def get_embedding(text, model="medium"):
    response = requests.post(
        "http://localhost:8000/embed",
        json={"text": text, "model": model}
    )
    return np.array(response.json()["embedding"])

# Use it
embedding = get_embedding("Machine learning is amazing")
print(f"Shape: {embedding.shape}")  # (2560,) for medium model
```

### JavaScript

```javascript
async function getEmbedding(text, model = "medium") {
  const response = await fetch("http://localhost:8000/embed", {
    method: "POST",
    headers: { "Content-Type": application/json" },
    body: JSON.stringify({ text, model })
  });
  const data = await response.json();
  return data.embedding;
}

// Use it
const embedding = await getEmbedding("Hello world");
console.log(`Dimensions: ${embedding.length}`);
```

### Semantic Search Example

```python
from sklearn.metrics.pairwise import cosine_similarity

# Your documents
docs = [
    "Machine learning is a subset of AI",
    "Python is a programming language",
    "Neural networks are inspired by the brain"
]

# Get embeddings for all docs
doc_embeddings = requests.post(
    "http://localhost:8000/embed_batch",
    json={"texts": docs}
).json()["embeddings"]

# Search
query = "artificial intelligence"
query_embedding = requests.post(
    "http://localhost:8000/embed",
    json={"text": query}
).json()["embedding"]

# Find most similar
similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
best_match = docs[similarities.argmax()]
print(f"Best match: {best_match}")
```

## 🛠️ Advanced Usage

### Configuration

Set environment variables to customize:

```bash
# Use a specific model by default
MODEL_NAME=mlx-community/Qwen3-Embedding-4B-4bit-DWQ python server.py

# Change port
PORT=8080 python server.py

# Development mode with auto-reload
DEV_MODE=true python server.py

# Increase batch size limit
MAX_BATCH_SIZE=128 python server.py
```

### Performance Tuning

```bash
# Run benchmarks
make benchmark

# See what models are loaded
curl http://localhost:8000/models

# Check performance metrics
curl http://localhost:8000/metrics
```

### Production Deployment

For production, use a process manager:

```bash
# Install PM2
npm install -g pm2

# Start server
pm2 start server.py --interpreter python3 --name embeddings

# Auto-start on boot
pm2 startup
pm2 save
```

Or use the included systemd service file for Linux servers.

## 📊 Performance

Real-world benchmarks from a 16" MacBook Pro (2023) with M2 Max chip and 32GB RAM:

| Operation           | Performance       |
| ------------------- | ----------------- |
| Single embedding    | 1-3ms             |
| Batch (32 texts)    | 44,000 tokens/sec |
| Concurrent requests | 200+ req/sec      |
| Cache speedup       | 13x faster        |

The medium model offers the best quality/speed balance with 0.65 semantic coherence score.

*Performance scales with Apple Silicon generation - expect even better results on M3/M4 chips!*

## 🎯 Use Cases

### RAG (Retrieval Augmented Generation)

```python
# 1. Embed your documents
embeddings = embed_batch(documents)
store_in_vector_db(embeddings)

# 2. Embed user query
query_embedding = embed(user_question)

# 3. Find relevant docs
relevant_docs = vector_db.search(query_embedding, top_k=5)

# 4. Pass to LLM
llm_response = llm.generate(user_question, context=relevant_docs)
```

### Semantic Deduplication

```python
# Find duplicate content
embeddings = embed_batch(articles)
similarity_matrix = cosine_similarity(embeddings)
duplicates = np.where(similarity_matrix > 0.95)
```

### Content Recommendation

```python
# Find similar items
user_liked_embedding = embed(user_liked_item)
all_embeddings = embed_batch(all_items)
similarities = cosine_similarity([user_liked_embedding], all_embeddings)
recommendations = all_items[similarities.argsort()[-10:]]
```

## 🐛 Troubleshooting

| Issue                   | Solution                                        |
| ----------------------- | ----------------------------------------------- |
| "Out of memory"         | Use smaller model or reduce batch size          |
| "Slow on first request" | Normal - model warming up. Keep server running. |
| "Can't connect"         | Check firewall, ensure port 8000 is free        |
| "Module not found"      | Run `pip install -r requirements.txt` again     |

## 🚀 Why Use This?

- **Privacy**: Your data never leaves your machine
- **Speed**: Faster than cloud APIs (no network latency)
- **Cost**: Free after initial setup (no API fees)
- **Reliability**: No internet required, no rate limits
- **Quality**: State-of-the-art Qwen3 models with 4-bit quantization

## 📦 What's Included

```
qwen3-embeddings/
├── server.py           # The entire server (one file!)
├── requirements.txt    # Dependencies
├── Makefile           # Convenience commands
├── tests/             # Benchmarks and tests
│   ├── test_api.py    # API tests
│   └── benchmark.py   # Performance benchmarks
└── examples/          # Usage examples
    └── visualize_embeddings.py  # Embedding visualization
```

## 🤝 Contributing

Contributions welcome! This is a simple, focused project:

1. Fork the repo
2. Create your feature branch
3. Make your changes
4. Run tests: `make test`
5. Submit a PR

## 📄 License

MIT License - use it however you want!

## 🙏 Credits

Built with:

- [MLX](https://github.com/ml-explore/mlx) - Apple's ML framework
- [Qwen](https://github.com/QwenLM/Qwen) - The embedding models
- [FastAPI](https://fastapi.tiangolo.com/) - The web framework

---

**Questions?** Open an issue on GitHub or check the [interactive docs](http://localhost:8000/docs).

**Ready to start?** Just run `python server.py` 🎉
