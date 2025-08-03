"""Enhanced logging utilities with structured logging and monitoring support"""

import sys
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime
from contextlib import contextmanager
from functools import wraps

from loguru import logger
import pandas as pd

from ..config import get_settings


class StructuredLogger:
    """Structured logging with JSON output support"""
    
    def __init__(self, name: str, enable_json: bool = False):
        """
        Initialize structured logger
        
        Args:
            name: Logger name
            enable_json: Enable JSON output format
        """
        self.name = name
        self.enable_json = enable_json
        self.context: Dict[str, Any] = {}
        
    def set_context(self, **kwargs):
        """Set persistent context for all log messages"""
        self.context.update(kwargs)
        
    def clear_context(self):
        """Clear persistent context"""
        self.context.clear()
        
    def _format_message(self, level: str, message: str, **kwargs) -> Dict[str, Any]:
        """Format message with structure"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "logger": self.name,
            "message": message,
            **self.context,
            **kwargs
        }
        return log_entry
        
    def _log(self, level: str, message: str, **kwargs):
        """Internal logging method"""
        if self.enable_json:
            log_entry = self._format_message(level, message, **kwargs)
            getattr(logger, level.lower())(json.dumps(log_entry))
        else:
            # Format extra fields for human-readable output
            extra_str = " ".join(f"{k}={v}" for k, v in {**self.context, **kwargs}.items())
            if extra_str:
                message = f"{message} | {extra_str}"
            getattr(logger, level.lower())(message)
            
    def debug(self, message: str, **kwargs):
        self._log("DEBUG", message, **kwargs)
        
    def info(self, message: str, **kwargs):
        self._log("INFO", message, **kwargs)
        
    def warning(self, message: str, **kwargs):
        self._log("WARNING", message, **kwargs)
        
    def error(self, message: str, **kwargs):
        self._log("ERROR", message, **kwargs)
        
    def critical(self, message: str, **kwargs):
        self._log("CRITICAL", message, **kwargs)


class PerformanceMonitor:
    """Monitor performance metrics and resource usage"""
    
    def __init__(self):
        """Initialize performance monitor"""
        self.metrics: Dict[str, List[float]] = {}
        self.active_timers: Dict[str, float] = {}
        
    @contextmanager
    def timer(self, operation: str, **tags):
        """
        Context manager for timing operations
        
        Args:
            operation: Name of the operation
            **tags: Additional tags for the metric
        """
        start_time = time.time()
        
        try:
            yield
        finally:
            duration = time.time() - start_time
            self._record_metric(f"{operation}_duration_seconds", duration, **tags)
            logger.debug(f"Operation '{operation}' took {duration:.2f} seconds", **tags)
            
    def start_timer(self, name: str):
        """Start a named timer"""
        self.active_timers[name] = time.time()
        
    def stop_timer(self, name: str) -> float:
        """Stop a named timer and return duration"""
        if name not in self.active_timers:
            logger.warning(f"Timer '{name}' was not started")
            return 0.0
            
        duration = time.time() - self.active_timers.pop(name)
        self._record_metric(f"{name}_duration_seconds", duration)
        return duration
        
    def record_metric(self, name: str, value: float, **tags):
        """Record a custom metric"""
        self._record_metric(name, value, **tags)
        
    def _record_metric(self, name: str, value: float, **tags):
        """Internal method to record metrics"""
        key = f"{name}_{json.dumps(tags, sort_keys=True)}" if tags else name
        
        if key not in self.metrics:
            self.metrics[key] = []
        self.metrics[key].append(value)
        
    def get_metrics_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all metrics"""
        summary = {}
        
        for key, values in self.metrics.items():
            if not values:
                continue
                
            series = pd.Series(values)
            summary[key] = {
                "count": len(values),
                "mean": series.mean(),
                "std": series.std(),
                "min": series.min(),
                "max": series.max(),
                "p50": series.quantile(0.5),
                "p95": series.quantile(0.95),
                "p99": series.quantile(0.99),
            }
            
        return summary
        
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format"""
        summary = self.get_metrics_summary()
        
        if format == "json":
            return json.dumps(summary, indent=2)
        elif format == "prometheus":
            lines = []
            for metric_name, stats in summary.items():
                base_name = metric_name.split("_")[0]
                for stat_name, value in stats.items():
                    lines.append(f"{base_name}_{stat_name} {value}")
            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported format: {format}")


class LoggingConfigurator:
    """Configure application-wide logging"""
    
    @staticmethod
    def configure(
        log_level: str = "INFO",
        log_file: Optional[Path] = None,
        json_logs: bool = False,
        enable_rotation: bool = True
    ):
        """
        Configure loguru logger
        
        Args:
            log_level: Logging level
            log_file: Optional log file path
            json_logs: Enable JSON logging
            enable_rotation: Enable log rotation
        """
        # Remove default handler
        logger.remove()
        
        # Configure format
        if json_logs:
            log_format = "{message}"
        else:
            log_format = (
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            )
        
        # Add console handler
        logger.add(
            sys.stderr,
            format=log_format,
            level=log_level,
            colorize=not json_logs
        )
        
        # Add file handler if specified
        if log_file:
            rotation = "10 MB" if enable_rotation else None
            logger.add(
                log_file,
                format=log_format,
                level=log_level,
                rotation=rotation,
                retention="7 days",
                compression="gz"
            )
            
    @staticmethod
    def get_logger(name: str, structured: bool = False) -> StructuredLogger:
        """Get a logger instance"""
        settings = get_settings()
        return StructuredLogger(
            name, 
            enable_json=structured or settings.log_format == "json"
        )


# Performance monitoring decorator
def monitor_performance(operation: str = None):
    """
    Decorator to monitor function performance
    
    Args:
        operation: Operation name (defaults to function name)
    """
    def decorator(func: Callable) -> Callable:
        op_name = operation or func.__name__
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            monitor = PerformanceMonitor()
            
            with monitor.timer(op_name):
                try:
                    result = func(*args, **kwargs)
                    monitor.record_metric(f"{op_name}_success", 1)
                    return result
                except Exception as e:
                    monitor.record_metric(f"{op_name}_failure", 1)
                    logger.error(f"Error in {op_name}: {str(e)}")
                    raise
                    
        return wrapper
    return decorator


# Global instances
performance_monitor = PerformanceMonitor()
structured_logger = StructuredLogger("fhfa_pipeline")


def setup_logging(settings: Optional[Any] = None):
    """Setup logging based on settings"""
    if settings is None:
        settings = get_settings()
        
    LoggingConfigurator.configure(
        log_level=settings.log_level,
        log_file=Path(settings.log_file) if settings.log_file else None,
        json_logs=getattr(settings, 'json_logs', False),
        enable_rotation=True
    )