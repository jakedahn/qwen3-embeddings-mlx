#!/usr/bin/env python3
"""
Benchmark script for Qwen3 Embedding Server

Tests various aspects of performance including:
- Single embedding latency
- Batch processing throughput
- Concurrent request handling
- Text length impact on performance
- Cache effectiveness
"""

import time
import json
import asyncio
import statistics
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import numpy as np
import argparse
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

# Configuration
BASE_URL = "http://localhost:8000"
EMBEDDING_DIM = 1024

# Test data
SHORT_TEXTS = [
    "Hello world",
    "Machine learning",
    "Apple Silicon",
    "Fast inference",
    "Text embedding"
]

MEDIUM_TEXTS = [
    "The quick brown fox jumps over the lazy dog in the sunny afternoon",
    "Machine learning models are transforming how we process and understand data",
    "Apple Silicon provides unprecedented performance for machine learning workloads",
    "FastAPI makes it easy to build high-performance REST APIs in Python",
    "Text embeddings capture semantic meaning in high-dimensional vector spaces"
]

LONG_TEXTS = [
    "Natural language processing has evolved significantly over the past decade, with transformer models revolutionizing how we approach text understanding. These models can capture complex semantic relationships and contextual nuances that were previously impossible to model effectively.",
    "The development of specialized hardware for machine learning, such as Apple's M-series chips with their Neural Engine, has democratized access to powerful AI capabilities. This hardware acceleration enables developers to run sophisticated models locally without relying on cloud infrastructure.",
    "Modern embedding models go beyond simple word representations to capture entire sentence and paragraph meanings. They can understand context, sentiment, and even subtle implications in text, making them invaluable for search, recommendation, and analysis applications.",
    "The intersection of efficient model architectures and optimized hardware has created new possibilities for edge AI applications. We can now run models that previously required data center resources on personal devices, enabling privacy-preserving and low-latency applications.",
    "Vector databases have emerged as a critical infrastructure component for AI applications, enabling efficient similarity search across millions or billions of embeddings. This technology powers everything from semantic search to recommendation systems and content moderation."
]

console = Console()

class BenchmarkClient:
    """Client for running benchmarks"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        
    def check_server(self) -> bool:
        """Check if server is available"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def single_embed(self, text: str) -> float:
        """Measure single embedding time"""
        start = time.perf_counter()
        response = self.session.post(
            f"{self.base_url}/embed",
            json={"text": text}
        )
        latency = (time.perf_counter() - start) * 1000
        
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}")
        
        return latency
    
    def batch_embed(self, texts: List[str]) -> float:
        """Measure batch embedding time"""
        start = time.perf_counter()
        response = self.session.post(
            f"{self.base_url}/embed_batch",
            json={"texts": texts}
        )
        latency = (time.perf_counter() - start) * 1000
        
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}")
        
        return latency

def benchmark_single_latency(client: BenchmarkClient, iterations: int = 100) -> Dict[str, Any]:
    """Benchmark single embedding latency"""
    console.print("\n[bold cyan]📊 Single Embedding Latency Test[/bold cyan]")
    
    results = {"short": [], "medium": [], "long": []}
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        
        # Test different text lengths
        for text_type, texts in [("short", SHORT_TEXTS), ("medium", MEDIUM_TEXTS), ("long", LONG_TEXTS)]:
            task = progress.add_task(f"Testing {text_type} texts...", total=iterations)
            
            for i in range(iterations):
                text = texts[i % len(texts)]
                try:
                    latency = client.single_embed(text)
                    results[text_type].append(latency)
                except Exception as e:
                    console.print(f"[red]Error: {e}[/red]")
                
                progress.update(task, advance=1)
    
    # Calculate statistics
    stats = {}
    for text_type, latencies in results.items():
        if latencies:
            stats[text_type] = {
                "mean": statistics.mean(latencies),
                "median": statistics.median(latencies),
                "stdev": statistics.stdev(latencies) if len(latencies) > 1 else 0,
                "min": min(latencies),
                "max": max(latencies),
                "p95": np.percentile(latencies, 95),
                "p99": np.percentile(latencies, 99)
            }
    
    return stats

def benchmark_batch_throughput(client: BenchmarkClient) -> Dict[str, Any]:
    """Benchmark batch processing throughput"""
    console.print("\n[bold cyan]📊 Batch Processing Throughput Test[/bold cyan]")
    
    batch_sizes = [1, 5, 10, 20, 32]
    results = {}
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Testing batch sizes...", total=len(batch_sizes))
        
        for batch_size in batch_sizes:
            texts = MEDIUM_TEXTS * (batch_size // len(MEDIUM_TEXTS) + 1)
            texts = texts[:batch_size]
            
            latencies = []
            for _ in range(10):  # 10 iterations per batch size
                try:
                    latency = client.batch_embed(texts)
                    latencies.append(latency)
                except Exception as e:
                    console.print(f"[red]Error with batch size {batch_size}: {e}[/red]")
            
            if latencies:
                avg_latency = statistics.mean(latencies)
                throughput = (batch_size / avg_latency) * 1000  # texts per second
                
                results[batch_size] = {
                    "latency_ms": avg_latency,
                    "throughput_per_sec": throughput,
                    "ms_per_text": avg_latency / batch_size
                }
            
            progress.update(task, advance=1)
    
    return results

def benchmark_concurrent_requests(client: BenchmarkClient, num_workers: int = 10) -> Dict[str, Any]:
    """Benchmark concurrent request handling"""
    console.print("\n[bold cyan]📊 Concurrent Request Test[/bold cyan]")
    
    def make_request(text: str) -> float:
        """Make a single request"""
        start = time.perf_counter()
        response = requests.post(
            f"{BASE_URL}/embed",
            json={"text": text}
        )
        latency = (time.perf_counter() - start) * 1000
        return latency if response.status_code == 200 else None
    
    results = []
    texts = MEDIUM_TEXTS * 20  # 100 total requests
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        task = progress.add_task(f"Running {len(texts)} concurrent requests...", total=len(texts))
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(make_request, text): text for text in texts}
            
            for future in as_completed(futures):
                latency = future.result()
                if latency:
                    results.append(latency)
                progress.update(task, advance=1)
    
    if results:
        return {
            "total_requests": len(texts),
            "successful_requests": len(results),
            "avg_latency_ms": statistics.mean(results),
            "median_latency_ms": statistics.median(results),
            "p95_latency_ms": np.percentile(results, 95),
            "p99_latency_ms": np.percentile(results, 99),
            "requests_per_sec": len(results) / (sum(results) / 1000)
        }
    
    return {}

def benchmark_cache_effectiveness(client: BenchmarkClient) -> Dict[str, Any]:
    """Test cache effectiveness with repeated queries"""
    console.print("\n[bold cyan]📊 Cache Effectiveness Test[/bold cyan]")
    
    test_text = "This is a test text for cache benchmarking"
    results = {"first_calls": [], "cached_calls": []}
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Testing cache performance...", total=40)
        
        # First calls (cold cache)
        for i in range(20):
            text = f"{test_text} variation {i}"
            latency = client.single_embed(text)
            results["first_calls"].append(latency)
            progress.update(task, advance=1)
        
        # Repeated calls (should hit cache)
        for i in range(20):
            text = f"{test_text} variation {i}"
            latency = client.single_embed(text)
            results["cached_calls"].append(latency)
            progress.update(task, advance=1)
    
    return {
        "avg_first_call_ms": statistics.mean(results["first_calls"]),
        "avg_cached_call_ms": statistics.mean(results["cached_calls"]),
        "cache_speedup": statistics.mean(results["first_calls"]) / statistics.mean(results["cached_calls"]),
        "cache_time_saved_ms": statistics.mean(results["first_calls"]) - statistics.mean(results["cached_calls"])
    }

def print_results(results: Dict[str, Any]):
    """Print benchmark results in a nice table format"""
    
    # Single latency results
    if "single_latency" in results:
        console.print("\n[bold green]📈 Single Embedding Latency Results[/bold green]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Text Length", style="cyan")
        table.add_column("Mean (ms)", justify="right")
        table.add_column("Median (ms)", justify="right")
        table.add_column("P95 (ms)", justify="right")
        table.add_column("P99 (ms)", justify="right")
        
        for text_type, stats in results["single_latency"].items():
            table.add_row(
                text_type.capitalize(),
                f"{stats['mean']:.2f}",
                f"{stats['median']:.2f}",
                f"{stats['p95']:.2f}",
                f"{stats['p99']:.2f}"
            )
        
        console.print(table)
    
    # Batch throughput results
    if "batch_throughput" in results:
        console.print("\n[bold green]📈 Batch Processing Throughput Results[/bold green]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Batch Size", style="cyan", justify="center")
        table.add_column("Latency (ms)", justify="right")
        table.add_column("ms/text", justify="right")
        table.add_column("Throughput (texts/sec)", justify="right")
        
        for batch_size, stats in sorted(results["batch_throughput"].items()):
            table.add_row(
                str(batch_size),
                f"{stats['latency_ms']:.2f}",
                f"{stats['ms_per_text']:.2f}",
                f"{stats['throughput_per_sec']:.1f}"
            )
        
        console.print(table)
    
    # Concurrent requests results
    if "concurrent" in results:
        console.print("\n[bold green]📈 Concurrent Request Handling Results[/bold green]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        
        table.add_row("Total Requests", str(results["concurrent"]["total_requests"]))
        table.add_row("Successful", str(results["concurrent"]["successful_requests"]))
        table.add_row("Avg Latency (ms)", f"{results['concurrent']['avg_latency_ms']:.2f}")
        table.add_row("P95 Latency (ms)", f"{results['concurrent']['p95_latency_ms']:.2f}")
        table.add_row("P99 Latency (ms)", f"{results['concurrent']['p99_latency_ms']:.2f}")
        table.add_row("Throughput (req/sec)", f"{results['concurrent']['requests_per_sec']:.1f}")
        
        console.print(table)
    
    # Cache effectiveness results
    if "cache" in results:
        console.print("\n[bold green]📈 Cache Effectiveness Results[/bold green]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        
        table.add_row("Avg First Call (ms)", f"{results['cache']['avg_first_call_ms']:.2f}")
        table.add_row("Avg Cached Call (ms)", f"{results['cache']['avg_cached_call_ms']:.2f}")
        table.add_row("Cache Speedup", f"{results['cache']['cache_speedup']:.2f}x")
        table.add_row("Time Saved (ms)", f"{results['cache']['cache_time_saved_ms']:.2f}")
        
        console.print(table)

def main():
    parser = argparse.ArgumentParser(description="Benchmark Qwen3 Embedding Server")
    parser.add_argument("--url", default=BASE_URL, help="Server URL")
    parser.add_argument("--iterations", type=int, default=50, help="Iterations for latency test")
    parser.add_argument("--workers", type=int, default=10, help="Concurrent workers")
    parser.add_argument("--skip-cache", action="store_true", help="Skip cache test")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark")
    args = parser.parse_args()
    
    console.print("[bold cyan]🚀 Qwen3 Embedding Server Benchmark[/bold cyan]")
    console.print("=" * 50)
    
    client = BenchmarkClient(args.url)
    
    # Check server
    if not client.check_server():
        console.print("[bold red]❌ Server is not running![/bold red]")
        console.print("Start the server with: [cyan]python server.py[/cyan]")
        return
    
    console.print("[green]✓ Server is running[/green]")
    
    results = {}
    
    try:
        # Run benchmarks
        if args.quick:
            # Quick benchmark - fewer iterations
            results["single_latency"] = benchmark_single_latency(client, iterations=20)
            results["batch_throughput"] = benchmark_batch_throughput(client)
        else:
            # Full benchmark
            results["single_latency"] = benchmark_single_latency(client, iterations=args.iterations)
            results["batch_throughput"] = benchmark_batch_throughput(client)
            results["concurrent"] = benchmark_concurrent_requests(client, num_workers=args.workers)
            
            if not args.skip_cache:
                results["cache"] = benchmark_cache_effectiveness(client)
        
        # Print results
        print_results(results)
        
        # Save results to file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Create benchmarks directory if it doesn't exist
        import os
        benchmarks_dir = os.path.join(os.path.dirname(__file__), "benchmarks")
        os.makedirs(benchmarks_dir, exist_ok=True)
        
        filename = os.path.join(benchmarks_dir, f"benchmark_results_{timestamp}.json")
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
        
        console.print(f"\n[green]✓ Results saved to {os.path.relpath(filename)}[/green]")
        
        # Print summary
        console.print("\n[bold cyan]📊 Summary[/bold cyan]")
        console.print("=" * 50)
        
        if "single_latency" in results and "medium" in results["single_latency"]:
            avg_latency = results["single_latency"]["medium"]["mean"]
            console.print(f"Average latency (medium text): [bold green]{avg_latency:.2f}ms[/bold green]")
            console.print(f"Requests per second (single): [bold green]{1000/avg_latency:.1f}[/bold green]")
        
        if "batch_throughput" in results and 32 in results["batch_throughput"]:
            throughput = results["batch_throughput"][32]["throughput_per_sec"]
            console.print(f"Batch throughput (32 texts): [bold green]{throughput:.1f} texts/sec[/bold green]")
        
        if "cache" in results:
            speedup = results["cache"]["cache_speedup"]
            console.print(f"Cache speedup: [bold green]{speedup:.2f}x[/bold green]")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Benchmark interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Benchmark failed: {e}[/red]")

if __name__ == "__main__":
    main()