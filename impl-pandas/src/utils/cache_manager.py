import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union
import logging
from pathlib import Path
import pickle
import hashlib
import time
from datetime import datetime, timedelta
import json

logger = logging.getLogger('seasonal_adjustment.cache_manager')


class CacheManager:
    """Manages caching of computed results for performance optimization."""
    
    def __init__(self, cache_dir: Optional[Union[str, Path]] = None,
                 max_cache_size_mb: float = 1000,
                 ttl_hours: float = 24):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory for cache storage. Uses temp if None.
            max_cache_size_mb: Maximum cache size in MB
            ttl_hours: Time to live for cache entries in hours
        """
        if cache_dir is None:
            self.cache_dir = Path.home() / '.seasonal_adjustment_cache'
        else:
            self.cache_dir = Path(cache_dir)
            
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_cache_size = max_cache_size_mb * 1024 * 1024  # Convert to bytes
        self.ttl = timedelta(hours=ttl_hours)
        
        # In-memory cache for frequently accessed items
        self.memory_cache = {}
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
        
        # Load cache index
        self._load_cache_index()
        
    def _load_cache_index(self):
        """Load or create cache index."""
        self.index_path = self.cache_dir / 'cache_index.json'
        
        if self.index_path.exists():
            try:
                with open(self.index_path, 'r') as f:
                    self.cache_index = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache index: {str(e)}")
                self.cache_index = {}
        else:
            self.cache_index = {}
            
    def _save_cache_index(self):
        """Save cache index to disk."""
        try:
            with open(self.index_path, 'w') as f:
                json.dump(self.cache_index, f)
        except Exception as e:
            logger.error(f"Failed to save cache index: {str(e)}")
            
    def _generate_cache_key(self, key_data: Union[str, Dict]) -> str:
        """Generate hash-based cache key."""
        if isinstance(key_data, dict):
            # Sort dict for consistent hashing
            key_str = json.dumps(key_data, sort_keys=True)
        else:
            key_str = str(key_data)
            
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def cache_seasonal_factors(self, geography_id: str, 
                             factors: pd.Series,
                             model_params: Optional[Dict] = None) -> None:
        """Cache computed seasonal factors."""
        logger.debug(f"Caching seasonal factors for {geography_id}")
        
        cache_key = self._generate_cache_key({
            'type': 'seasonal_factors',
            'geography_id': geography_id,
            'model_params': model_params or {}
        })
        
        cache_data = {
            'factors': factors,
            'geography_id': geography_id,
            'model_params': model_params,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save to disk
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
            
        # Update index
        self.cache_index[cache_key] = {
            'path': str(cache_path),
            'type': 'seasonal_factors',
            'geography_id': geography_id,
            'timestamp': cache_data['timestamp'],
            'size': cache_path.stat().st_size
        }
        
        # Store in memory cache
        self.memory_cache[cache_key] = cache_data
        
        self._save_cache_index()
        self._check_cache_size()
        
    def get_cached_results(self, geography_id: str,
                         result_type: str = 'seasonal_factors',
                         model_params: Optional[Dict] = None) -> Optional[Dict]:
        """Retrieve cached results if available."""
        cache_key = self._generate_cache_key({
            'type': result_type,
            'geography_id': geography_id,
            'model_params': model_params or {}
        })
        
        # Check memory cache first
        if cache_key in self.memory_cache:
            self.cache_stats['hits'] += 1
            logger.debug(f"Memory cache hit for {geography_id}")
            return self.memory_cache[cache_key]
            
        # Check disk cache
        if cache_key in self.cache_index:
            cache_info = self.cache_index[cache_key]
            
            # Check if expired
            cache_time = datetime.fromisoformat(cache_info['timestamp'])
            if datetime.now() - cache_time > self.ttl:
                logger.debug(f"Cache expired for {geography_id}")
                self._remove_cache_entry(cache_key)
                self.cache_stats['misses'] += 1
                return None
                
            # Load from disk
            try:
                cache_path = Path(cache_info['path'])
                with open(cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                    
                # Store in memory cache
                self.memory_cache[cache_key] = cache_data
                
                self.cache_stats['hits'] += 1
                logger.debug(f"Disk cache hit for {geography_id}")
                return cache_data
                
            except Exception as e:
                logger.error(f"Failed to load cache for {geography_id}: {str(e)}")
                self._remove_cache_entry(cache_key)
                
        self.cache_stats['misses'] += 1
        return None
    
    def cache_model_results(self, geography_id: str,
                          results: Dict[str, Any]) -> None:
        """Cache complete model results."""
        logger.debug(f"Caching model results for {geography_id}")
        
        cache_key = self._generate_cache_key({
            'type': 'model_results',
            'geography_id': geography_id
        })
        
        # Extract cacheable parts (avoid large model objects)
        cacheable_results = {
            'geography_id': geography_id,
            'status': results.get('status'),
            'model_spec': results.get('model_spec'),
            'diagnostics': results.get('diagnostics'),
            'stability': results.get('stability'),
            'validation': results.get('validation'),
            'timestamp': datetime.now().isoformat()
        }
        
        # Handle series data separately
        if 'series' in results:
            series_data = results['series']
            cacheable_results['series'] = {
                'original': series_data.get('original'),
                'seasonally_adjusted': series_data.get('seasonally_adjusted'),
                'seasonal_factors': series_data.get('seasonal_factors')
            }
            
        # Save to disk
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        with open(cache_path, 'wb') as f:
            pickle.dump(cacheable_results, f)
            
        # Update index
        self.cache_index[cache_key] = {
            'path': str(cache_path),
            'type': 'model_results',
            'geography_id': geography_id,
            'timestamp': cacheable_results['timestamp'],
            'size': cache_path.stat().st_size
        }
        
        self._save_cache_index()
        self._check_cache_size()
        
    def invalidate_cache(self, geography_id: Optional[str] = None,
                        cache_type: Optional[str] = None) -> None:
        """Invalidate cache entries."""
        logger.info(f"Invalidating cache: geography={geography_id}, type={cache_type}")
        
        keys_to_remove = []
        
        for cache_key, cache_info in self.cache_index.items():
            if geography_id and cache_info.get('geography_id') != geography_id:
                continue
            if cache_type and cache_info.get('type') != cache_type:
                continue
                
            keys_to_remove.append(cache_key)
            
        for key in keys_to_remove:
            self._remove_cache_entry(key)
            
        self._save_cache_index()
        
    def _remove_cache_entry(self, cache_key: str) -> None:
        """Remove a cache entry."""
        if cache_key in self.cache_index:
            cache_info = self.cache_index[cache_key]
            cache_path = Path(cache_info['path'])
            
            # Remove file
            if cache_path.exists():
                cache_path.unlink()
                
            # Remove from index
            del self.cache_index[cache_key]
            
            # Remove from memory cache
            if cache_key in self.memory_cache:
                del self.memory_cache[cache_key]
                
            self.cache_stats['evictions'] += 1
            
    def _check_cache_size(self) -> None:
        """Check and enforce cache size limits."""
        total_size = sum(info['size'] for info in self.cache_index.values())
        
        if total_size > self.max_cache_size:
            logger.info("Cache size exceeded, performing cleanup")
            
            # Sort by timestamp (oldest first)
            sorted_entries = sorted(
                self.cache_index.items(),
                key=lambda x: x[1]['timestamp']
            )
            
            # Remove oldest entries until under limit
            while total_size > self.max_cache_size * 0.8:  # Keep 20% buffer
                if not sorted_entries:
                    break
                    
                cache_key, cache_info = sorted_entries.pop(0)
                total_size -= cache_info['size']
                self._remove_cache_entry(cache_key)
                
    def clear_memory_cache(self) -> None:
        """Clear in-memory cache while preserving disk cache."""
        self.memory_cache.clear()
        logger.info("Cleared memory cache")
        
    def clear_all_cache(self) -> None:
        """Clear all cache (memory and disk)."""
        logger.info("Clearing all cache")
        
        # Remove all cache files
        for cache_info in self.cache_index.values():
            cache_path = Path(cache_info['path'])
            if cache_path.exists():
                cache_path.unlink()
                
        # Clear indices
        self.cache_index.clear()
        self.memory_cache.clear()
        
        # Reset stats
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
        
        self._save_cache_index()
        
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_size = sum(info['size'] for info in self.cache_index.values())
        
        return {
            'hits': self.cache_stats['hits'],
            'misses': self.cache_stats['misses'],
            'hit_rate': self.cache_stats['hits'] / max(1, self.cache_stats['hits'] + self.cache_stats['misses']),
            'evictions': self.cache_stats['evictions'],
            'entries': len(self.cache_index),
            'memory_entries': len(self.memory_cache),
            'total_size_mb': total_size / (1024 * 1024),
            'max_size_mb': self.max_cache_size / (1024 * 1024)
        }
        
    def optimize_cache(self) -> None:
        """Optimize cache by removing expired entries and defragmenting."""
        logger.info("Optimizing cache")
        
        expired_keys = []
        current_time = datetime.now()
        
        # Find expired entries
        for cache_key, cache_info in self.cache_index.items():
            cache_time = datetime.fromisoformat(cache_info['timestamp'])
            if current_time - cache_time > self.ttl:
                expired_keys.append(cache_key)
                
        # Remove expired entries
        for key in expired_keys:
            self._remove_cache_entry(key)
            
        # Clear least recently used items from memory cache if too large
        if len(self.memory_cache) > 100:
            # Keep only most recent 50 items
            sorted_items = sorted(
                self.memory_cache.items(),
                key=lambda x: x[1].get('timestamp', ''),
                reverse=True
            )
            self.memory_cache = dict(sorted_items[:50])
            
        self._save_cache_index()
        
        logger.info(f"Cache optimization complete: removed {len(expired_keys)} expired entries")