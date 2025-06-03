"""
Performance monitoring utilities for tracking execution time and resource usage.
"""

import time
import asyncio
import functools
import psutil
import threading
from typing import Dict, Any, Callable, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

from .LoggerConfig import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    function_name: str
    start_time: float
    end_time: float
    duration: float
    cpu_percent_start: float
    cpu_percent_end: float
    memory_mb_start: float
    memory_mb_end: float
    memory_delta_mb: float
    exception: Optional[str] = None
    additional_metrics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def cpu_delta(self) -> float:
        """CPU usage change during execution."""
        return self.cpu_percent_end - self.cpu_percent_start


class PerformanceMonitor:
    """Monitors and tracks performance metrics."""
    
    def __init__(self):
        self._metrics: Dict[str, List[PerformanceMetrics]] = defaultdict(list)
        self._active_monitors: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        
    def start_monitoring(self, operation_id: str) -> Dict[str, Any]:
        """Start monitoring an operation."""
        process = psutil.Process()
        
        monitor_data = {
            "start_time": time.time(),
            "cpu_percent_start": process.cpu_percent(interval=0.1),
            "memory_mb_start": process.memory_info().rss / 1024 / 1024
        }
        
        with self._lock:
            self._active_monitors[operation_id] = monitor_data
            
        return monitor_data
    
    def stop_monitoring(
        self, 
        operation_id: str, 
        function_name: str,
        exception: Optional[Exception] = None,
        **additional_metrics
    ) -> PerformanceMetrics:
        """Stop monitoring and record metrics."""
        process = psutil.Process()
        end_time = time.time()
        
        with self._lock:
            if operation_id not in self._active_monitors:
                logger.warning(f"No active monitoring for operation: {operation_id}")
                return None
                
            monitor_data = self._active_monitors.pop(operation_id)
        
        metrics = PerformanceMetrics(
            function_name=function_name,
            start_time=monitor_data["start_time"],
            end_time=end_time,
            duration=end_time - monitor_data["start_time"],
            cpu_percent_start=monitor_data["cpu_percent_start"],
            cpu_percent_end=process.cpu_percent(interval=0.1),
            memory_mb_start=monitor_data["memory_mb_start"],
            memory_mb_end=process.memory_info().rss / 1024 / 1024,
            memory_delta_mb=(process.memory_info().rss / 1024 / 1024) - monitor_data["memory_mb_start"],
            exception=str(exception) if exception else None,
            additional_metrics=additional_metrics
        )
        
        with self._lock:
            self._metrics[function_name].append(metrics)
            
        return metrics
    
    def get_metrics(self, function_name: Optional[str] = None) -> Dict[str, List[PerformanceMetrics]]:
        """Get recorded metrics."""
        with self._lock:
            if function_name:
                return {function_name: self._metrics.get(function_name, [])}
            return dict(self._metrics)
    
    def get_summary(self, function_name: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """Get summary statistics for metrics."""
        metrics = self.get_metrics(function_name)
        summary = {}
        
        for func_name, func_metrics in metrics.items():
            if not func_metrics:
                continue
                
            durations = [m.duration for m in func_metrics]
            memory_deltas = [m.memory_delta_mb for m in func_metrics]
            cpu_deltas = [m.cpu_delta for m in func_metrics]
            error_count = sum(1 for m in func_metrics if m.exception)
            
            summary[func_name] = {
                "call_count": len(func_metrics),
                "error_count": error_count,
                "error_rate": error_count / len(func_metrics) if func_metrics else 0,
                "duration": {
                    "mean": sum(durations) / len(durations),
                    "min": min(durations),
                    "max": max(durations),
                    "total": sum(durations)
                },
                "memory_delta_mb": {
                    "mean": sum(memory_deltas) / len(memory_deltas),
                    "min": min(memory_deltas),
                    "max": max(memory_deltas),
                    "total": sum(memory_deltas)
                },
                "cpu_delta": {
                    "mean": sum(cpu_deltas) / len(cpu_deltas),
                    "min": min(cpu_deltas),
                    "max": max(cpu_deltas)
                }
            }
            
        return summary
    
    def clear_metrics(self, function_name: Optional[str] = None):
        """Clear recorded metrics."""
        with self._lock:
            if function_name:
                self._metrics.pop(function_name, None)
            else:
                self._metrics.clear()
    
    def log_summary(self, function_name: Optional[str] = None):
        """Log performance summary."""
        summary = self.get_summary(function_name)
        
        for func_name, stats in summary.items():
            logger.info(
                f"Performance summary for {func_name}:\n"
                f"  Calls: {stats['call_count']} (errors: {stats['error_count']})\n"
                f"  Duration: mean={stats['duration']['mean']:.3f}s, "
                f"min={stats['duration']['min']:.3f}s, max={stats['duration']['max']:.3f}s\n"
                f"  Memory: mean={stats['memory_delta_mb']['mean']:.1f}MB, "
                f"max={stats['memory_delta_mb']['max']:.1f}MB\n"
                f"  CPU: mean={stats['cpu_delta']['mean']:.1f}%, "
                f"max={stats['cpu_delta']['max']:.1f}%"
            )
    
    def measure_operation(self, operation_name: str):
        """Context manager for measuring operation performance."""
        from contextlib import contextmanager
        
        @contextmanager
        def _measure():
            operation_id = f"{operation_name}_{id(self)}_{time.time()}"
            self.start_monitoring(operation_id)
            exception = None
            try:
                yield
            except Exception as e:
                exception = e
                raise
            finally:
                self.stop_monitoring(
                    operation_id,
                    operation_name,
                    exception=exception
                )
        
        return _measure()


# Global performance monitor instance
_global_monitor = PerformanceMonitor()


def monitor_performance(
    name: Optional[str] = None,
    log_metrics: bool = True,
    track_memory: bool = True,
    track_cpu: bool = True
):
    """
    Decorator to monitor function performance.
    
    Args:
        name: Custom name for the operation (defaults to function name)
        log_metrics: Whether to log metrics after execution
        track_memory: Whether to track memory usage
        track_cpu: Whether to track CPU usage
    """
    def decorator(func: Callable) -> Callable:
        func_name = name or func.__name__
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            operation_id = f"{func_name}_{time.time()}"
            
            # Start monitoring
            _global_monitor.start_monitoring(operation_id)
            
            exception = None
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                exception = e
                raise
            finally:
                # Stop monitoring
                metrics = _global_monitor.stop_monitoring(
                    operation_id, 
                    func_name,
                    exception=exception
                )
                
                if log_metrics and metrics:
                    logger.info(
                        f"{func_name} completed in {metrics.duration:.3f}s "
                        f"(memory: {metrics.memory_delta_mb:+.1f}MB, "
                        f"cpu: {metrics.cpu_delta:+.1f}%)"
                    )
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            operation_id = f"{func_name}_{time.time()}"
            
            # Start monitoring
            _global_monitor.start_monitoring(operation_id)
            
            exception = None
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                exception = e
                raise
            finally:
                # Stop monitoring
                metrics = _global_monitor.stop_monitoring(
                    operation_id,
                    func_name,
                    exception=exception
                )
                
                if log_metrics and metrics:
                    logger.info(
                        f"{func_name} completed in {metrics.duration:.3f}s "
                        f"(memory: {metrics.memory_delta_mb:+.1f}MB, "
                        f"cpu: {metrics.cpu_delta:+.1f}%)"
                    )
        
        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    return _global_monitor


# Context manager for monitoring code blocks
class monitor_block:
    """Context manager for monitoring performance of code blocks."""
    
    def __init__(self, name: str, log_metrics: bool = True):
        self.name = name
        self.log_metrics = log_metrics
        self.operation_id = f"{name}_{time.time()}"
        
    def __enter__(self):
        _global_monitor.start_monitoring(self.operation_id)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        metrics = _global_monitor.stop_monitoring(
            self.operation_id,
            self.name,
            exception=exc_val
        )
        
        if self.log_metrics and metrics:
            logger.info(
                f"{self.name} completed in {metrics.duration:.3f}s "
                f"(memory: {metrics.memory_delta_mb:+.1f}MB)"
            )