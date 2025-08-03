"""Monitoring and telemetry utilities for tracking pipeline performance"""

import os
import psutil
import time
import json
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from pathlib import Path
import threading
from queue import Queue
import sqlite3

from loguru import logger
import pandas as pd


@dataclass
class MetricPoint:
    """Single metric data point"""
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""


@dataclass
class ResourceUsage:
    """System resource usage snapshot"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    
    
class MetricsCollector:
    """Collect and store metrics"""
    
    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize metrics collector
        
        Args:
            storage_path: Path to SQLite database for metrics storage
        """
        self.storage_path = storage_path or Path(".metrics/metrics.db")
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_database()
        self.metrics_queue: Queue[MetricPoint] = Queue()
        self._stop_event = threading.Event()
        self._writer_thread = None
        
    def _init_database(self):
        """Initialize SQLite database for metrics"""
        with sqlite3.connect(str(self.storage_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    value REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    tags TEXT,
                    unit TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_timestamp 
                ON metrics(timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_name 
                ON metrics(name)
            """)
            
    def start(self):
        """Start background metrics writer"""
        if self._writer_thread is None or not self._writer_thread.is_alive():
            self._stop_event.clear()
            self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
            self._writer_thread.start()
            logger.info("Metrics collector started")
            
    def stop(self):
        """Stop background metrics writer"""
        self._stop_event.set()
        if self._writer_thread:
            self._writer_thread.join(timeout=5)
        logger.info("Metrics collector stopped")
        
    def _writer_loop(self):
        """Background loop to write metrics to database"""
        batch = []
        last_flush = time.time()
        
        while not self._stop_event.is_set():
            try:
                # Collect metrics for up to 1 second
                deadline = time.time() + 1.0
                while time.time() < deadline:
                    try:
                        metric = self.metrics_queue.get(timeout=0.1)
                        batch.append(metric)
                    except:
                        break
                        
                # Flush if batch is large enough or time elapsed
                if len(batch) > 100 or (time.time() - last_flush) > 10:
                    self._flush_metrics(batch)
                    batch.clear()
                    last_flush = time.time()
                    
            except Exception as e:
                logger.error(f"Error in metrics writer: {e}")
                
        # Final flush
        if batch:
            self._flush_metrics(batch)
            
    def _flush_metrics(self, metrics: List[MetricPoint]):
        """Write metrics batch to database"""
        if not metrics:
            return
            
        try:
            with sqlite3.connect(str(self.storage_path)) as conn:
                data = [
                    (
                        m.name,
                        m.value,
                        m.timestamp.isoformat(),
                        json.dumps(m.tags) if m.tags else None,
                        m.unit
                    )
                    for m in metrics
                ]
                
                conn.executemany(
                    "INSERT INTO metrics (name, value, timestamp, tags, unit) VALUES (?, ?, ?, ?, ?)",
                    data
                )
                
            logger.debug(f"Flushed {len(metrics)} metrics to database")
        except Exception as e:
            logger.error(f"Failed to flush metrics: {e}")
            
    def record(self, name: str, value: float, unit: str = "", **tags):
        """Record a metric"""
        metric = MetricPoint(
            name=name,
            value=value,
            unit=unit,
            tags=tags
        )
        self.metrics_queue.put(metric)
        
    def query_metrics(
        self,
        name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Query metrics from database
        
        Args:
            name: Metric name pattern (supports wildcards)
            start_time: Start time filter
            end_time: End time filter  
            tags: Tag filters
            
        Returns:
            DataFrame with metrics
        """
        query = "SELECT * FROM metrics WHERE name LIKE ?"
        params = [name.replace("*", "%")]
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
            
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
            
        with sqlite3.connect(str(self.storage_path)) as conn:
            df = pd.read_sql_query(query, conn, params=params)
            
        # Parse timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Parse tags
        df['tags'] = df['tags'].apply(lambda x: json.loads(x) if x else {})
        
        # Filter by tags if specified
        if tags:
            mask = df['tags'].apply(
                lambda x: all(x.get(k) == v for k, v in tags.items())
            )
            df = df[mask]
            
        return df


class ResourceMonitor:
    """Monitor system resource usage"""
    
    def __init__(self, interval_seconds: int = 60):
        """
        Initialize resource monitor
        
        Args:
            interval_seconds: Sampling interval
        """
        self.interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._monitor_thread = None
        self.metrics_collector = MetricsCollector()
        
        # Get initial disk IO counters
        self._last_disk_io = psutil.disk_io_counters()
        self._last_sample_time = time.time()
        
    def start(self):
        """Start resource monitoring"""
        if self._monitor_thread is None or not self._monitor_thread.is_alive():
            self._stop_event.clear()
            self.metrics_collector.start()
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()
            logger.info("Resource monitor started")
            
    def stop(self):
        """Stop resource monitoring"""
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        self.metrics_collector.stop()
        logger.info("Resource monitor stopped")
        
    def _monitor_loop(self):
        """Background monitoring loop"""
        while not self._stop_event.is_set():
            try:
                usage = self._sample_resources()
                self._record_usage(usage)
            except Exception as e:
                logger.error(f"Error sampling resources: {e}")
                
            # Wait for next interval
            self._stop_event.wait(self.interval_seconds)
            
    def _sample_resources(self) -> ResourceUsage:
        """Sample current resource usage"""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_mb = memory.used / (1024 * 1024)
        
        # Disk IO
        current_disk_io = psutil.disk_io_counters()
        current_time = time.time()
        time_delta = current_time - self._last_sample_time
        
        if time_delta > 0:
            read_mb = (current_disk_io.read_bytes - self._last_disk_io.read_bytes) / (1024 * 1024)
            write_mb = (current_disk_io.write_bytes - self._last_disk_io.write_bytes) / (1024 * 1024)
            
            # Convert to MB/s
            disk_io_read_mb = read_mb / time_delta
            disk_io_write_mb = write_mb / time_delta
        else:
            disk_io_read_mb = 0
            disk_io_write_mb = 0
            
        self._last_disk_io = current_disk_io
        self._last_sample_time = current_time
        
        return ResourceUsage(
            timestamp=datetime.utcnow(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_mb=memory_mb,
            disk_io_read_mb=disk_io_read_mb,
            disk_io_write_mb=disk_io_write_mb
        )
        
    def _record_usage(self, usage: ResourceUsage):
        """Record resource usage metrics"""
        self.metrics_collector.record("system.cpu.percent", usage.cpu_percent, unit="%")
        self.metrics_collector.record("system.memory.percent", usage.memory_percent, unit="%")
        self.metrics_collector.record("system.memory.mb", usage.memory_mb, unit="MB")
        self.metrics_collector.record("system.disk.read_mb_s", usage.disk_io_read_mb, unit="MB/s")
        self.metrics_collector.record("system.disk.write_mb_s", usage.disk_io_write_mb, unit="MB/s")
        
    def get_resource_summary(self, last_minutes: int = 60) -> Dict[str, Any]:
        """Get resource usage summary"""
        start_time = datetime.utcnow() - timedelta(minutes=last_minutes)
        
        summary = {}
        metrics = ["cpu.percent", "memory.percent", "memory.mb", "disk.read_mb_s", "disk.write_mb_s"]
        
        for metric in metrics:
            df = self.metrics_collector.query_metrics(
                f"system.{metric}",
                start_time=start_time
            )
            
            if not df.empty:
                summary[metric] = {
                    "mean": df['value'].mean(),
                    "max": df['value'].max(),
                    "min": df['value'].min(),
                    "p95": df['value'].quantile(0.95)
                }
                
        return summary


# Health check utilities
class HealthChecker:
    """Application health checks"""
    
    def __init__(self):
        """Initialize health checker"""
        self.checks: Dict[str, Callable[[], Dict[str, Any]]] = {}
        
    def register_check(self, name: str, check_func: Callable[[], Dict[str, Any]]):
        """Register a health check"""
        self.checks[name] = check_func
        
    def run_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "status": "healthy",
            "checks": {}
        }
        
        for name, check_func in self.checks.items():
            try:
                result = check_func()
                results["checks"][name] = {
                    "status": "healthy",
                    **result
                }
            except Exception as e:
                results["checks"][name] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                results["status"] = "unhealthy"
                
        return results


# Global instances
metrics_collector = MetricsCollector()
resource_monitor = ResourceMonitor()
health_checker = HealthChecker()


# Convenience decorators
def track_metrics(metric_name: str = None, include_args: bool = False):
    """Decorator to track function metrics"""
    def decorator(func: Callable) -> Callable:
        name = metric_name or f"function.{func.__name__}"
        
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                tags = {}
                if include_args and args:
                    tags["arg0"] = str(args[0])[:50]  # First arg, truncated
                    
                metrics_collector.record(f"{name}.duration", duration, unit="seconds", **tags)
                metrics_collector.record(f"{name}.success", 1, **tags)
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                metrics_collector.record(f"{name}.duration", duration, unit="seconds")
                metrics_collector.record(f"{name}.error", 1, error_type=type(e).__name__)
                raise
                
        return wrapper
    return decorator