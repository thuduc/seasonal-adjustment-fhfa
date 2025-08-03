"""Configuration loader with support for multiple formats and environments"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger

from .settings import Settings, get_settings


class ConfigLoader:
    """Load configuration from multiple sources with priority order"""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize configuration loader
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = config_dir or Path(__file__).parent.parent.parent / "config"
        self.environment = os.getenv("FHFA_ENV", "development")
        
    def load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        filepath = self.config_dir / filename
        if not filepath.exists():
            logger.debug(f"Config file not found: {filepath}")
            return {}
            
        try:
            with open(filepath, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Failed to load YAML config from {filepath}: {e}")
            return {}
    
    def load_json(self, filename: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        filepath = self.config_dir / filename
        if not filepath.exists():
            logger.debug(f"Config file not found: {filepath}")
            return {}
            
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load JSON config from {filepath}: {e}")
            return {}
    
    def load_environment_config(self) -> Dict[str, Any]:
        """Load environment-specific configuration"""
        config = {}
        
        # Load base configuration
        base_config = self.load_yaml("config.yaml")
        config.update(base_config)
        
        # Load environment-specific config
        env_config = self.load_yaml(f"config.{self.environment}.yaml")
        config.update(env_config)
        
        # Load JSON overrides if exist
        json_overrides = self.load_json(f"config.{self.environment}.json")
        config.update(json_overrides)
        
        return config
    
    def merge_configs(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge multiple configuration dictionaries with deep merge
        
        Args:
            configs: Configuration dictionaries to merge
            
        Returns:
            Merged configuration
        """
        result = {}
        
        for config in configs:
            for key, value in config.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    # Deep merge dictionaries
                    result[key] = self.merge_configs(result[key], value)
                else:
                    result[key] = value
                    
        return result
    
    def get_full_config(self) -> Dict[str, Any]:
        """
        Get full configuration from all sources
        
        Priority order (highest to lowest):
        1. Environment variables (via pydantic Settings)
        2. Environment-specific config files
        3. Base config files
        4. Default values in Settings
        
        Returns:
            Complete configuration dictionary
        """
        # Load file-based configs
        file_config = self.load_environment_config()
        
        # Get settings (includes env vars)
        settings = get_settings()
        settings_dict = settings.to_dict()
        
        # Merge with file config taking precedence over settings defaults
        full_config = self.merge_configs(file_config, settings_dict)
        
        # Add computed values
        full_config['environment'] = self.environment
        full_config['config_sources'] = {
            'environment': self.environment,
            'config_dir': str(self.config_dir),
            'env_file': '.env' if Path('.env').exists() else None
        }
        
        return full_config


def load_model_config(model_type: str) -> Dict[str, Any]:
    """
    Load model-specific configuration
    
    Args:
        model_type: Type of model (arima, panel, quantile, etc.)
        
    Returns:
        Model configuration
    """
    loader = ConfigLoader()
    full_config = loader.get_full_config()
    
    # Extract model-specific config
    model_config = full_config.get('models', {}).get(model_type, {})
    
    # Add global model defaults
    model_config.setdefault('seasonal_period', full_config.get('seasonal_period', 4))
    model_config.setdefault('min_observations', full_config.get('min_observations', 20))
    
    return model_config


def get_pipeline_config() -> Dict[str, Any]:
    """Get pipeline-specific configuration"""
    loader = ConfigLoader()
    full_config = loader.get_full_config()
    
    return {
        'parallel_enabled': full_config.get('parallel_enabled', True),
        'n_jobs': full_config.get('n_jobs', -1),
        'chunk_size': full_config.get('chunk_size', 1000),
        'memory_limit_gb': full_config.get('memory_limit_gb', 16.0),
        'enable_caching': full_config.get('enable_caching', True),
        'cache_ttl_hours': full_config.get('cache_ttl_hours', 24),
        'output_dir': full_config.get('output_dir', './output'),
        'log_level': full_config.get('log_level', 'INFO')
    }