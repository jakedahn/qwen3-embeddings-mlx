# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a high-performance text embedding server that runs multiple Qwen3 models (0.6B, 4B, 8B) on Apple Silicon using the MLX framework. The server provides REST API endpoints for generating text embeddings optimized for M1/M2/M3 Macs with dynamic model selection per request.

## Key Commands

### Server Operations
```bash
# Install dependencies
make install

# Run the server
make run                # Production mode
make dev               # Development mode with DEBUG logging and auto-reload

# Health check
make health            # Check if server is running
curl http://localhost:8000/health
```

### Testing and Benchmarking
```bash
# Run tests
make test              # Run API tests
python tests/test_api.py

# Benchmarking
make benchmark         # Quick benchmark
make benchmark-full    # Comprehensive benchmark (100 iterations, 10 workers)
python tests/benchmark.py --quick
python tests/benchmark.py --iterations 100 --workers 10
```

### Development
```bash
# Code quality
make lint             # Run flake8 and mypy
make format           # Format with black
make clean            # Remove cache and temp files

# Install dev dependencies
make install-dev      # Installs pytest, black, flake8, mypy, httpx
```

## Architecture

### Core Components

1. **server.py** - Single-file FastAPI server with three main components:
   - `ModelManager`: Handles MLX model lifecycle, caching, and inference for multiple models
   - `ServerConfig`: Dataclass for configuration via environment variables
   - FastAPI app with endpoints: `/embed`, `/embed_batch`, `/health`, `/metrics`, `/models`

2. **Multi-Model Loading Strategy**:
   - Default model loads at startup, others load on-demand
   - LRU-style memory management with configurable max models (default: 2)
   - Model aliases for convenience: "small" (0.6B), "medium" (4B), "large" (8B)
   - Per-model caching with model-aware cache keys
   - Warmup phase compiles Metal kernels for faster inference
   - Hidden states extracted manually from transformer layers (not using logits)
   - Mean pooling applied across sequence dimension for embeddings

3. **Critical Model Implementation Detail**:
   The Qwen3 model outputs logits (vocabulary size), not embeddings directly. The correct embedding extraction requires:
   ```python
   # Get hidden states BEFORE output projection
   h = model.model.embed_tokens(input_ids)
   for layer in model.model.layers:
       h = layer(h, mask=None, cache=None)
   h = model.model.norm(h)  # Shape varies by model
   pooled = mx.mean(h, axis=1)   # Mean pooling → [1, embedding_dim]
   # embedding_dim: 1024 (0.6B), 2560 (4B), 4096 (8B)
   ```

### Configuration

Environment variables control server behavior:
- `MODEL_NAME`: MLX model to use (default: `mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ`)
- `PORT`: Server port (default: 8000)
- `HOST`: Server host (default: 0.0.0.0)
- `MAX_BATCH_SIZE`: Max texts per batch (default: 32)
- `MAX_TEXT_LENGTH`: Max tokens per text (default: 8192)
- `LOG_LEVEL`: Logging level (default: INFO)
- `ENABLE_CORS`: Enable CORS (default: true)
- `DEV_MODE`: Enable auto-reload (default: false)

### Performance Characteristics

| Model | Embedding Dim | Model Size | Latency | Description |
|-------|---------------|------------|---------|-------------|
| 0.6B  | 1024         | ~900MB     | ~1.3ms  | Fast and efficient |
| 4B    | 2560         | ~2.5GB     | ~3-5ms  | Balanced performance |
| 8B    | 4096         | ~4.5GB     | ~8-12ms | Higher quality |

- **Throughput**: 2,100+ texts/sec (0.6B model, batch size 32)
- **Cache speedup**: 13.6x for repeated queries
- **Memory management**: Auto-eviction when max models exceeded

### MLX-Specific Considerations

1. **Tokenizer API**: MLX's `tokenizer.encode()` returns `List[int]`, not dict
2. **Array conversion**: Use `mx.array([tokens])` to add batch dimension
3. **Evaluation**: Call `mx.eval()` before numpy conversion to avoid dtype issues
4. **No attention_mask**: MLX models don't use attention_mask parameter directly

### Testing Structure

- `tests/test_api.py`: Comprehensive API tests including validation, performance, and similarity
- `tests/benchmark.py`: Performance benchmarking with rich output visualization
- Tests check for proper normalization (L2 norm ~1.0) and correct dimensions per model

### Common Issues and Solutions

1. **Empty `cors_origins` list error**: Fixed by using `None` default in dataclass and setting in `__post_init__`
2. **Wrong embedding dimension (151669)**: Model outputs logits, need to extract hidden states before projection
3. **Numpy conversion errors**: Use `pooled.tolist()[0]` then `np.array()` with explicit dtype
4. **Warmup failures**: Non-critical, happens when model status check occurs during loading

### Development Workflow

When modifying embeddings extraction:
1. Test with single text first: `curl -X POST localhost:8000/embed -d '{"text":"test", "model":"small"}'`
2. Verify dimension matches model: 1024 (small), 2560 (medium), 4096 (large)
3. Check L2 norm is ~1.0 for normalized embeddings
4. Test model switching: `curl -X POST localhost:8000/embed -d '{"text":"test", "model":"large"}'`
5. Run benchmark to ensure performance hasn't regressed

When adding new endpoints:
1. Add Pydantic models for request/response validation
2. Include proper error handling and status codes
3. Update `/metrics` endpoint if tracking new metrics
4. Add corresponding tests in `tests/test_api.py`

### Multi-Model API Usage

```bash
# List available models
curl http://localhost:8000/models

# Use specific model by alias
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello world", "model":"medium"}'

# Use model by full name
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello world", "model":"mlx-community/Qwen3-Embedding-4B-4bit-DWQ"}'
```