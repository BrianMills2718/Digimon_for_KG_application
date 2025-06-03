"""
Performance monitoring dashboard for DIGIMON system.
"""

import asyncio
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict

from .PerformanceMonitor import get_performance_monitor
from .LLMEnhancements import get_llm_performance_tracker, get_llm_cache, get_adaptive_timeout
from .LoggerConfig import get_logger

logger = get_logger(__name__)


class PerformanceDashboard:
    """Central dashboard for system performance monitoring."""
    
    def __init__(self):
        self.performance_monitor = get_performance_monitor()
        self.llm_tracker = get_llm_performance_tracker()
        self.llm_cache = get_llm_cache()
        self.adaptive_timeout = get_adaptive_timeout()
        
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system performance metrics."""
        
        # Get general performance metrics
        general_metrics = self.performance_monitor.get_summary()
        
        # Get LLM-specific metrics
        llm_metrics = await self.llm_tracker.get_performance_summary()
        
        # Get cache statistics
        cache_stats = await self.llm_cache.get_stats()
        
        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(general_metrics, llm_metrics)
        
        # Generate optimization suggestions
        optimizations = self._generate_optimizations(
            general_metrics, llm_metrics, cache_stats, bottlenecks
        )
        
        return {
            "timestamp": datetime.now().isoformat(),
            "general_performance": general_metrics,
            "llm_performance": llm_metrics,
            "cache_performance": cache_stats,
            "bottlenecks": bottlenecks,
            "optimizations": optimizations,
            "system_health": self._calculate_health_score(
                general_metrics, llm_metrics, cache_stats
            )
        }
    
    def _identify_bottlenecks(
        self,
        general_metrics: Dict[str, Any],
        llm_metrics: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        
        # Check for slow operations
        for func_name, stats in general_metrics.items():
            if isinstance(stats, dict) and "duration" in stats:
                avg_duration = stats["duration"]["mean"]
                if avg_duration > 10:  # Operations taking more than 10 seconds
                    bottlenecks.append({
                        "type": "slow_operation",
                        "component": func_name,
                        "severity": "high" if avg_duration > 30 else "medium",
                        "details": {
                            "avg_duration": avg_duration,
                            "max_duration": stats["duration"]["max"],
                            "call_count": stats["call_count"]
                        }
                    })
        
        # Check for high error rates
        for func_name, stats in general_metrics.items():
            if isinstance(stats, dict) and "error_rate" in stats:
                error_rate = stats["error_rate"]
                if error_rate > 0.1:  # More than 10% errors
                    bottlenecks.append({
                        "type": "high_error_rate",
                        "component": func_name,
                        "severity": "high" if error_rate > 0.3 else "medium",
                        "details": {
                            "error_rate": error_rate,
                            "error_count": stats["error_count"],
                            "total_calls": stats["call_count"]
                        }
                    })
        
        # Check LLM-specific bottlenecks
        if llm_metrics.get("avg_duration_seconds", 0) > 20:
            bottlenecks.append({
                "type": "slow_llm_responses",
                "component": "llm_provider",
                "severity": "high",
                "details": {
                    "avg_duration": llm_metrics["avg_duration_seconds"],
                    "total_requests": llm_metrics.get("total_requests", 0)
                }
            })
        
        # Check for memory issues
        for func_name, stats in general_metrics.items():
            if isinstance(stats, dict) and "memory_delta_mb" in stats:
                max_memory = stats["memory_delta_mb"]["max"]
                if max_memory > 500:  # Using more than 500MB
                    bottlenecks.append({
                        "type": "high_memory_usage",
                        "component": func_name,
                        "severity": "medium",
                        "details": {
                            "max_memory_mb": max_memory,
                            "avg_memory_mb": stats["memory_delta_mb"]["mean"]
                        }
                    })
        
        return bottlenecks
    
    def _generate_optimizations(
        self,
        general_metrics: Dict[str, Any],
        llm_metrics: Dict[str, Any],
        cache_stats: Dict[str, Any],
        bottlenecks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate optimization suggestions."""
        optimizations = []
        
        # Cache optimizations
        if cache_stats["hit_rate"] < 0.3 and cache_stats["size"] < cache_stats["max_size"] / 2:
            optimizations.append({
                "category": "cache",
                "priority": "medium",
                "suggestion": "Increase cache TTL to improve hit rate",
                "expected_impact": "Reduce LLM calls by up to 50%",
                "implementation": {
                    "param": "ttl_seconds",
                    "current": cache_stats["ttl_seconds"],
                    "recommended": cache_stats["ttl_seconds"] * 2
                }
            })
        
        # LLM optimizations
        if llm_metrics.get("retry_requests", 0) > llm_metrics.get("total_requests", 1) * 0.1:
            optimizations.append({
                "category": "llm",
                "priority": "high",
                "suggestion": "Implement request throttling to reduce rate limit errors",
                "expected_impact": "Reduce retry overhead by 80%",
                "implementation": {
                    "strategy": "exponential_backoff",
                    "initial_delay": 1,
                    "max_delay": 60
                }
            })
        
        # Bottleneck-specific optimizations
        for bottleneck in bottlenecks:
            if bottleneck["type"] == "slow_operation":
                optimizations.append({
                    "category": "performance",
                    "priority": bottleneck["severity"],
                    "suggestion": f"Optimize {bottleneck['component']} operation",
                    "expected_impact": f"Reduce execution time from {bottleneck['details']['avg_duration']:.1f}s",
                    "implementation": {
                        "strategies": [
                            "Add caching layer",
                            "Implement batch processing",
                            "Use async operations"
                        ]
                    }
                })
            elif bottleneck["type"] == "high_memory_usage":
                optimizations.append({
                    "category": "memory",
                    "priority": "medium",
                    "suggestion": f"Reduce memory usage in {bottleneck['component']}",
                    "expected_impact": f"Reduce memory by {bottleneck['details']['max_memory_mb']:.0f}MB",
                    "implementation": {
                        "strategies": [
                            "Process data in chunks",
                            "Clear intermediate results",
                            "Use generators instead of lists"
                        ]
                    }
                })
        
        # Batch processing optimization
        total_llm_requests = llm_metrics.get("total_requests", 0)
        if total_llm_requests > 100:
            optimizations.append({
                "category": "batch_processing",
                "priority": "medium",
                "suggestion": "Implement batch processing for similar requests",
                "expected_impact": "Reduce total processing time by 30-40%",
                "implementation": {
                    "batch_size": 10,
                    "grouping_strategy": "similarity_based"
                }
            })
        
        return optimizations
    
    def _calculate_health_score(
        self,
        general_metrics: Dict[str, Any],
        llm_metrics: Dict[str, Any],
        cache_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate overall system health score."""
        scores = {
            "performance": 100,
            "reliability": 100,
            "efficiency": 100
        }
        
        # Performance score
        avg_durations = []
        for stats in general_metrics.values():
            if isinstance(stats, dict) and "duration" in stats:
                avg_durations.append(stats["duration"]["mean"])
        
        if avg_durations:
            avg_system_duration = sum(avg_durations) / len(avg_durations)
            if avg_system_duration > 30:
                scores["performance"] -= 50
            elif avg_system_duration > 10:
                scores["performance"] -= 20
            elif avg_system_duration > 5:
                scores["performance"] -= 10
        
        # Reliability score
        error_rates = []
        for stats in general_metrics.values():
            if isinstance(stats, dict) and "error_rate" in stats:
                error_rates.append(stats["error_rate"])
        
        if error_rates:
            avg_error_rate = sum(error_rates) / len(error_rates)
            scores["reliability"] -= int(avg_error_rate * 100)
        
        llm_success_rate = llm_metrics.get("success_rate", 1.0)
        scores["reliability"] -= int((1 - llm_success_rate) * 50)
        
        # Efficiency score
        cache_hit_rate = cache_stats.get("hit_rate", 0)
        scores["efficiency"] = int(50 + cache_hit_rate * 50)
        
        # Overall score
        overall = sum(scores.values()) / len(scores)
        
        return {
            "overall": overall,
            "components": scores,
            "status": "healthy" if overall > 80 else "degraded" if overall > 60 else "unhealthy"
        }
    
    async def generate_report(self, output_file: Optional[str] = None) -> str:
        """Generate a comprehensive performance report."""
        metrics = await self.get_system_metrics()
        
        report_lines = [
            "# DIGIMON Performance Report",
            f"Generated at: {metrics['timestamp']}",
            f"System Health: {metrics['system_health']['status'].upper()} ({metrics['system_health']['overall']:.1f}/100)",
            "",
            "## Executive Summary",
            f"- Overall Health Score: {metrics['system_health']['overall']:.1f}/100",
            f"- Performance Score: {metrics['system_health']['components']['performance']}/100",
            f"- Reliability Score: {metrics['system_health']['components']['reliability']}/100",
            f"- Efficiency Score: {metrics['system_health']['components']['efficiency']}/100",
            "",
            "## Key Metrics",
            "",
            "### LLM Performance",
            f"- Total Requests: {metrics['llm_performance'].get('total_requests', 0)}",
            f"- Success Rate: {metrics['llm_performance'].get('success_rate', 0):.1%}",
            f"- Average Response Time: {metrics['llm_performance'].get('avg_duration_seconds', 0):.2f}s",
            f"- Cache Hit Rate: {metrics['cache_performance']['hit_rate']:.1%}",
            "",
            "### Identified Bottlenecks",
        ]
        
        if metrics['bottlenecks']:
            for bottleneck in metrics['bottlenecks']:
                report_lines.append(
                    f"- **{bottleneck['type']}** in {bottleneck['component']} "
                    f"(Severity: {bottleneck['severity']})"
                )
        else:
            report_lines.append("- No significant bottlenecks identified")
        
        report_lines.extend([
            "",
            "## Optimization Recommendations",
            ""
        ])
        
        if metrics['optimizations']:
            for i, opt in enumerate(metrics['optimizations'], 1):
                report_lines.extend([
                    f"### {i}. {opt['suggestion']}",
                    f"- Category: {opt['category']}",
                    f"- Priority: {opt['priority']}",
                    f"- Expected Impact: {opt['expected_impact']}",
                    ""
                ])
        else:
            report_lines.append("No optimizations recommended at this time.")
        
        report = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            logger.info(f"Performance report saved to {output_file}")
        
        return report
    
    def start_monitoring_loop(self, interval_seconds: int = 300):
        """Start a background monitoring loop."""
        async def monitoring_loop():
            while True:
                try:
                    metrics = await self.get_system_metrics()
                    
                    # Log critical issues
                    if metrics['system_health']['status'] == 'unhealthy':
                        logger.error(
                            f"System health is UNHEALTHY! "
                            f"Score: {metrics['system_health']['overall']:.1f}/100"
                        )
                    
                    # Log bottlenecks
                    for bottleneck in metrics['bottlenecks']:
                        if bottleneck['severity'] == 'high':
                            logger.warning(
                                f"High severity bottleneck: {bottleneck['type']} "
                                f"in {bottleneck['component']}"
                            )
                    
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                
                await asyncio.sleep(interval_seconds)
        
        # Start the loop in the background
        asyncio.create_task(monitoring_loop())
        logger.info(f"Started performance monitoring loop (interval: {interval_seconds}s)")


# Global dashboard instance
_dashboard = PerformanceDashboard()


def get_performance_dashboard() -> PerformanceDashboard:
    """Get the global performance dashboard."""
    return _dashboard