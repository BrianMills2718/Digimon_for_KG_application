"""
DIGIMON Performance Benchmarking Suite

Measures performance improvements from optimizations:
- Caching effectiveness
- Batch processing speedup
- Error handling overhead
- Memory usage
"""

import asyncio
import time
import statistics
from typing import List, Dict, Any, Callable
import json
from pathlib import Path
from datetime import datetime

from Core.Common.PerformanceMonitor import get_performance_monitor, monitor_performance
from Core.Common.CacheManager import get_cache_manager
from Core.Common.BatchProcessor import BatchProcessor, BatchConfig
from Core.Common.LoggerConfig import get_logger
from Core.AgentBrain.agent_brain import PlanningAgent
from Core.AgentOrchestrator.orchestrator import AgentOrchestrator
from Core.AgentSchema.context import GraphRAGContext
from Option.Config2 import Config

logger = get_logger(__name__)


class BenchmarkResult:
    """Container for benchmark results."""
    
    def __init__(self, name: str):
        self.name = name
        self.durations: List[float] = []
        self.memory_deltas: List[float] = []
        self.cache_hits = 0
        self.cache_misses = 0
        self.errors = 0
        
    def add_run(self, duration: float, memory_delta: float):
        """Add a benchmark run result."""
        self.durations.append(duration)
        self.memory_deltas.append(memory_delta)
        
    def get_stats(self) -> Dict[str, float]:
        """Get statistical summary."""
        if not self.durations:
            return {}
            
        return {
            "runs": len(self.durations),
            "avg_duration": statistics.mean(self.durations),
            "min_duration": min(self.durations),
            "max_duration": max(self.durations),
            "std_duration": statistics.stdev(self.durations) if len(self.durations) > 1 else 0,
            "avg_memory_mb": statistics.mean(self.memory_deltas),
            "max_memory_mb": max(self.memory_deltas),
            "cache_hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
            "error_rate": self.errors / len(self.durations) if self.durations else 0
        }


class PerformanceBenchmark:
    """Main benchmarking class."""
    
    def __init__(self, output_file: str = "benchmark_results.json"):
        self.output_file = output_file
        self.results: Dict[str, BenchmarkResult] = {}
        self.perf_monitor = get_performance_monitor()
        self.cache_manager = get_cache_manager()
        
    async def benchmark_cache_effectiveness(self, runs: int = 10):
        """Benchmark caching effectiveness for LLM calls."""
        logger.info(f"Benchmarking cache effectiveness with {runs} runs")
        
        # Prepare test data
        test_queries = [
            "What are the main entities in the Zorathian Empire?",
            "How does crystal technology work?",
            "What caused the fall of Aerophantis?"
        ]
        
        # Test without cache
        result_no_cache = BenchmarkResult("llm_no_cache")
        self.cache_manager.clear()
        
        for _ in range(runs):
            for query in test_queries:
                start = time.time()
                # Simulate LLM call (in real test, use actual agent)
                await asyncio.sleep(0.1)  # Simulate API latency
                duration = time.time() - start
                result_no_cache.add_run(duration, 0)
        
        self.results["llm_no_cache"] = result_no_cache
        
        # Test with cache
        result_with_cache = BenchmarkResult("llm_with_cache")
        
        for i in range(runs):
            for query in test_queries:
                start = time.time()
                
                # Check cache
                cached = self.cache_manager.get("llm_response", {"query": query})
                if cached:
                    result_with_cache.cache_hits += 1
                else:
                    result_with_cache.cache_misses += 1
                    # Simulate LLM call
                    await asyncio.sleep(0.1)
                    # Cache the result
                    self.cache_manager.set("llm_response", {"query": query}, "response")
                
                duration = time.time() - start
                result_with_cache.add_run(duration, 0)
        
        self.results["llm_with_cache"] = result_with_cache
        
        # Calculate speedup
        no_cache_avg = result_no_cache.get_stats()["avg_duration"]
        with_cache_avg = result_with_cache.get_stats()["avg_duration"]
        speedup = no_cache_avg / with_cache_avg if with_cache_avg > 0 else 0
        
        logger.info(f"Cache speedup: {speedup:.2f}x")
        logger.info(f"Cache hit rate: {result_with_cache.get_stats()['cache_hit_rate']:.2%}")
        
    async def benchmark_batch_processing(self, item_counts: List[int] = [10, 50, 100]):
        """Benchmark batch processing effectiveness."""
        logger.info(f"Benchmarking batch processing with item counts: {item_counts}")
        
        async def process_items(items: List[int]) -> List[int]:
            """Simulate processing items."""
            await asyncio.sleep(0.01 * len(items))  # Simulate batch processing
            return [i * 2 for i in items]
        
        for count in item_counts:
            items = list(range(count))
            
            # Test sequential processing
            result_seq = BenchmarkResult(f"sequential_{count}")
            start = time.time()
            
            results = []
            for item in items:
                result = await process_items([item])
                results.extend(result)
            
            duration = time.time() - start
            result_seq.add_run(duration, 0)
            self.results[f"sequential_{count}"] = result_seq
            
            # Test batch processing
            result_batch = BenchmarkResult(f"batch_{count}")
            processor = BatchProcessor(BatchConfig(batch_size=32))
            
            start = time.time()
            results = await processor.process_async(items, process_items)
            duration = time.time() - start
            
            result_batch.add_run(duration, 0)
            self.results[f"batch_{count}"] = result_batch
            
            # Calculate speedup
            speedup = result_seq.durations[0] / result_batch.durations[0] if result_batch.durations[0] > 0 else 0
            logger.info(f"Batch processing speedup for {count} items: {speedup:.2f}x")
    
    @monitor_performance(name="benchmark_error_handling")
    async def benchmark_error_handling(self, runs: int = 20):
        """Benchmark error handling and retry overhead."""
        logger.info(f"Benchmarking error handling with {runs} runs")
        
        from Core.Common.RetryUtils import retry_llm_call
        
        # Test successful calls
        result_success = BenchmarkResult("no_errors")
        
        @retry_llm_call(max_attempts=3)
        async def successful_call():
            await asyncio.sleep(0.01)
            return "success"
        
        for _ in range(runs):
            start = time.time()
            await successful_call()
            duration = time.time() - start
            result_success.add_run(duration, 0)
        
        self.results["no_errors"] = result_success
        
        # Test with retries
        result_retry = BenchmarkResult("with_retries")
        call_count = 0
        
        @retry_llm_call(max_attempts=3)
        async def flaky_call():
            nonlocal call_count
            call_count += 1
            if call_count % 3 != 0:  # Fail 2 out of 3 times
                raise Exception("429 Too Many Requests")
            await asyncio.sleep(0.01)
            return "success"
        
        call_count = 0
        for _ in range(runs):
            start = time.time()
            try:
                await flaky_call()
            except Exception:
                result_retry.errors += 1
            duration = time.time() - start
            result_retry.add_run(duration, 0)
        
        self.results["with_retries"] = result_retry
        
        # Calculate overhead
        no_error_avg = result_success.get_stats()["avg_duration"]
        retry_avg = result_retry.get_stats()["avg_duration"]
        overhead = (retry_avg - no_error_avg) / no_error_avg if no_error_avg > 0 else 0
        
        logger.info(f"Retry overhead: {overhead:.2%}")
        
    async def benchmark_memory_usage(self):
        """Benchmark memory usage patterns."""
        logger.info("Benchmarking memory usage")
        
        # Get performance stats from monitor
        perf_stats = self.perf_monitor.get_summary()
        
        for func_name, stats in perf_stats.items():
            result = BenchmarkResult(f"memory_{func_name}")
            
            # Extract memory stats
            if "memory_delta_mb" in stats:
                result.memory_deltas = [
                    stats["memory_delta_mb"]["mean"],
                    stats["memory_delta_mb"]["max"]
                ]
                result.durations = [stats["duration"]["mean"]]
                
            self.results[f"memory_{func_name}"] = result
            
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "benchmarks": {}
        }
        
        for name, result in self.results.items():
            report["benchmarks"][name] = result.get_stats()
            
        # Calculate overall improvements
        improvements = {}
        
        # Cache improvement
        if "llm_no_cache" in self.results and "llm_with_cache" in self.results:
            no_cache = self.results["llm_no_cache"].get_stats()["avg_duration"]
            with_cache = self.results["llm_with_cache"].get_stats()["avg_duration"]
            improvements["cache_speedup"] = no_cache / with_cache if with_cache > 0 else 0
            
        # Batch processing improvement
        batch_speedups = []
        for count in [10, 50, 100]:
            if f"sequential_{count}" in self.results and f"batch_{count}" in self.results:
                seq = self.results[f"sequential_{count}"].get_stats()["avg_duration"]
                batch = self.results[f"batch_{count}"].get_stats()["avg_duration"]
                batch_speedups.append(seq / batch if batch > 0 else 0)
        
        if batch_speedups:
            improvements["avg_batch_speedup"] = statistics.mean(batch_speedups)
            
        report["improvements"] = improvements
        
        return report
        
    def save_report(self):
        """Save benchmark report to file."""
        report = self.generate_report()
        
        with open(self.output_file, "w") as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Benchmark report saved to {self.output_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        
        for name, stats in report["benchmarks"].items():
            if stats:
                print(f"\n{name}:")
                print(f"  Avg Duration: {stats.get('avg_duration', 0):.3f}s")
                if 'cache_hit_rate' in stats:
                    print(f"  Cache Hit Rate: {stats['cache_hit_rate']:.2%}")
                if 'avg_memory_mb' in stats:
                    print(f"  Avg Memory: {stats['avg_memory_mb']:.1f}MB")
                    
        if report["improvements"]:
            print("\nIMPROVEMENTS:")
            for key, value in report["improvements"].items():
                print(f"  {key}: {value:.2f}x")
        
        print("="*60)


async def run_benchmarks():
    """Run all benchmarks."""
    benchmark = PerformanceBenchmark()
    
    # Run benchmarks
    await benchmark.benchmark_cache_effectiveness(runs=5)
    await benchmark.benchmark_batch_processing()
    await benchmark.benchmark_error_handling(runs=10)
    await benchmark.benchmark_memory_usage()
    
    # Generate and save report
    benchmark.save_report()


if __name__ == "__main__":
    asyncio.run(run_benchmarks())