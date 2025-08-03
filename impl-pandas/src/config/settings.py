"""Application settings and configuration management"""

from typing import Optional, Dict, Any
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from functools import lru_cache
import os


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Project paths
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent
    )
    data_dir: Path = Field(default=None)
    output_dir: Path = Field(default=None)
    cache_dir: Path = Field(default=None)
    
    # Model parameters
    default_ar_order: int = Field(default=1, ge=0, le=5)
    default_diff_order: int = Field(default=1, ge=0, le=2)
    default_ma_order: int = Field(default=2, ge=0, le=5)
    seasonal_period: int = Field(default=4, description="Quarterly data")
    
    # Processing settings
    parallel_enabled: bool = Field(default=True)
    n_jobs: int = Field(default=-1, description="Number of parallel jobs (-1 for all cores)")
    chunk_size: int = Field(default=1000)
    memory_limit_gb: float = Field(default=16.0)
    
    # Data quality thresholds
    min_observations: int = Field(default=20, description="Minimum observations for ARIMA")
    max_missing_rate: float = Field(default=0.2, description="Maximum missing data rate")
    outlier_threshold: float = Field(default=3.0, description="Standard deviations for outlier detection")
    
    # Logging
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    log_file: Optional[str] = Field(default=None)
    log_format: str = Field(
        default="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} | {message}"
    )
    
    # Performance
    enable_caching: bool = Field(default=True)
    cache_ttl_hours: int = Field(default=24)
    
    # Validation
    validation_tolerance: float = Field(default=0.001)
    
    @field_validator("data_dir", "output_dir", "cache_dir", mode='before')
    @classmethod
    def set_directories(cls, v, info):
        if v is None:
            project_root = Path(__file__).parent.parent.parent
            field_name = info.field_name
            if field_name == "data_dir":
                return project_root / "data"
            elif field_name == "output_dir":
                return project_root / "output"
            elif field_name == "cache_dir":
                return project_root / ".cache"
        return v
    
    @field_validator("n_jobs", mode='before')
    @classmethod
    def validate_n_jobs(cls, v):
        if v == -1:
            return os.cpu_count() or 1
        return min(v, os.cpu_count() or 1)
    
    model_config = {
        "env_prefix": "FHFA_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False
    }
        
    def to_dict(self) -> Dict[str, Any]:
        """Export settings as dictionary"""
        return {
            k: str(v) if isinstance(v, Path) else v 
            for k, v in self.model_dump().items()
        }


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Model-specific constants
class ModelConstants:
    """Constants for model specifications"""
    
    # Geography levels
    GEOGRAPHY_LEVELS = ["national", "state", "msa", "zip3"]
    
    # Index types
    INDEX_TYPES = [
        "purchase_only",
        "all_transactions",
        "expanded_data",
        "distress_free",
        "manufactured_homes"
    ]
    
    # Time frequencies
    FREQUENCIES = {
        "monthly": "M",
        "quarterly": "Q",
        "annual": "A"
    }
    
    # Variable names
    WEATHER_VARS = [
        "temp_range_q1", "temp_range_q2", "temp_range_q3", "temp_range_q4",
        "avg_temp", "precipitation"
    ]
    
    HOUSEHOLD_VARS = [
        "avg_household_income",
        "pct_white",
        "pct_over_65",
        "pct_bachelors_plus",
        "pct_single_family"
    ]
    
    INDUSTRY_VARS = [
        "healthcare_share",
        "manufacturing_share",
        "professional_share",
        "retail_share",
        "education_share",
        "fire_share",  # Finance, Insurance, Real Estate
        "public_admin_share",
        "construction_share",
        "transportation_share",
        "agriculture_share"
    ]
    
    # Time trend breakpoints
    TIME_BREAKPOINTS = [1998, 2007, 2011, 2020]
    
    # State codes (excluding Hawaii for weather analysis)
    STATES_WITH_WEATHER = [
        "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
        "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA",
        "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM",
        "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD",
        "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"
    ]