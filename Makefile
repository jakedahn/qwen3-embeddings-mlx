.PHONY: help install install-dev run test clean lint format health benchmark dev

# Default target
help:
	@echo "Qwen3 Embeddings Server - Available Commands:"
	@echo ""
	@echo "  make install          Install production dependencies"
	@echo "  make install-dev      Install development dependencies"
	@echo "  make run              Run the server"
	@echo "  make dev              Run in development mode with auto-reload"
	@echo "  make test             Run test suite"
	@echo "  make lint             Run code linting"
	@echo "  make format           Format code with black"
	@echo "  make clean            Remove cache and temporary files"
	@echo "  make health           Check server health"
	@echo "  make benchmark        Run quick benchmark (all models)"
	@echo "  make benchmark-full   Run comprehensive benchmark (all models)"
	@echo "  make benchmark-small  Quick benchmark (0.6B model)"
	@echo "  make benchmark-medium Quick benchmark (4B model)"
	@echo "  make benchmark-large  Quick benchmark (8B model)"
	@echo "  make benchmark-stress Stress test with large batches (up to 512)"
	@echo "  make benchmark-extreme EXTREME stress test (batches up to 1024!)"
	@echo ""

# Install production dependencies
install:
	pip install -r requirements.txt

# Install development dependencies
install-dev: install
	pip install pytest black flake8 mypy httpx

# Run the server
run:
	python server.py

# Run in development mode
dev:
	DEV_MODE=true LOG_LEVEL=DEBUG python server.py

# Run tests
test:
	@if [ -f tests/test_api.py ]; then \
		python tests/test_api.py; \
	else \
		echo "No tests found"; \
	fi

# Run pytest if available
pytest:
	@which pytest > /dev/null && pytest tests/ -v || echo "pytest not installed"

# Lint code
lint:
	@which flake8 > /dev/null && flake8 server.py --max-line-length=120 || echo "flake8 not installed"
	@which mypy > /dev/null && mypy server.py --ignore-missing-imports || echo "mypy not installed"

# Format code
format:
	@which black > /dev/null && black server.py tests/ examples/ || echo "black not installed"

# Clean cache and temporary files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.log" -delete
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info

# Check server health
health:
	@curl -s http://localhost:8000/health | python -m json.tool || echo "Server not running"

# Run benchmarks
benchmark:
	@if [ -f tests/benchmark.py ]; then \
		echo "🚀 Running comprehensive benchmark..."; \
		python tests/benchmark.py --quick; \
	else \
		echo "Running simple benchmark..."; \
		python -c "import time, requests, json; \
			start = time.time(); \
			r = requests.post('http://localhost:8000/embed', json={'text': 'test'}); \
			print(f'Single embed: {(time.time()-start)*1000:.2f}ms'); \
			start = time.time(); \
			r = requests.post('http://localhost:8000/embed_batch', json={'texts': ['test']*10}); \
			print(f'Batch (10): {(time.time()-start)*1000:.2f}ms')"; \
	fi || echo "Server not running"

# Run full benchmark on all models
benchmark-full:
	@if [ -f tests/benchmark.py ]; then \
		python tests/benchmark.py --iterations 100 --workers 10 --model all; \
	else \
		echo "benchmark.py not found"; \
	fi

# Benchmark specific model
benchmark-small:
	python tests/benchmark.py --quick --model small

benchmark-medium:
	python tests/benchmark.py --quick --model medium

benchmark-large:
	python tests/benchmark.py --quick --model large

# Stress tests (make your fans spin!)
benchmark-stress:
	@echo "⚡ Running stress test with large batches..."
	python tests/benchmark.py --stress --model all

benchmark-extreme:
	@echo "🔥 EXTREME STRESS TEST - This will push your system hard!"
	@echo "Press Ctrl+C to cancel..."
	@sleep 3
	python tests/benchmark.py --extreme --model all

# Docker build (experimental)
docker-build:
	docker build -t qwen3-embeddings .

# Docker run (experimental)
docker-run:
	docker run -p 8000:8000 qwen3-embeddings