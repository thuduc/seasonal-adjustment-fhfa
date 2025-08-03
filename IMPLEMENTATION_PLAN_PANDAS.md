# FHFA Seasonal Adjustment Implementation Plan - Python/Pandas

## Executive Summary

This document outlines the implementation plan for the FHFA House Price Index Seasonal Adjustment system using Python, Pandas, and machine learning libraries. The implementation follows the requirements specified in the PRD and ensures comprehensive testing with at least 80% code coverage for core models.

## 1. Technical Stack

### 1.1 Core Libraries
```python
# Data manipulation
pandas==2.1.0
numpy==1.24.0

# Time series and statistical modeling
statsmodels==0.14.0  # For ARIMA, regression, and X-13ARIMA-SEATS wrapper
scipy==1.11.0       # Statistical tests
scikit-learn==1.3.0 # Machine learning utilities

# Econometrics
linearmodels==5.3   # Panel data and fixed effects models
quantreg==0.1.0     # Quantile regression (or use statsmodels)

# Visualization
matplotlib==3.7.0
seaborn==0.12.0
plotly==5.17.0     # Interactive diagnostics

# Testing
pytest==7.4.0
pytest-cov==4.1.0
pytest-mock==3.11.0
hypothesis==6.82.0  # Property-based testing

# Code quality
black==23.7.0
flake8==6.1.0
mypy==1.5.0
pre-commit==3.3.0

# Performance
numba==0.58.0      # JIT compilation for computationally intensive parts
dask==2023.8.0     # Parallel processing for large datasets

# Additional utilities
pydantic==2.3.0    # Data validation
click==8.1.0       # CLI interface
python-dotenv==1.0.0
loguru==0.7.0      # Advanced logging
```

### 1.2 Development Environment
- Python 3.10+ (for modern type hints and performance)
- Virtual environment management (venv/conda)
- Docker for containerized deployment
- Git for version control

## 2. Project Structure

```
fhfa-seasonal-adjustment/
├── src/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py          # Configuration management
│   │   └── constants.py         # Model constants and parameters
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loaders.py          # Data loading interfaces
│   │   ├── preprocessors.py    # Data cleaning and preparation
│   │   ├── validators.py       # Data validation
│   │   └── transformers.py     # Feature engineering
│   ├── models/
│   │   ├── __init__.py
│   │   ├── arima/
│   │   │   ├── __init__.py
│   │   │   ├── regarima.py    # RegARIMA implementation
│   │   │   ├── x13wrapper.py  # X-13ARIMA-SEATS wrapper
│   │   │   └── diagnostics.py # ARIMA diagnostics
│   │   ├── regression/
│   │   │   ├── __init__.py
│   │   │   ├── linear.py      # Linear regression models
│   │   │   ├── quantile.py    # Quantile regression
│   │   │   └── panel.py       # Fixed effects models
│   │   └── seasonal/
│   │       ├── __init__.py
│   │       ├── adjuster.py    # Main seasonal adjustment class
│   │       └── outliers.py    # Outlier detection
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── weather_impact.py  # Weather variable analysis
│   │   ├── geographic.py      # Geographic analysis
│   │   └── temporal.py        # Time trend analysis
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── orchestrator.py    # Main pipeline orchestrator
│   │   ├── stages.py          # Pipeline stages
│   │   └── parallel.py        # Parallel processing
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── diagnostics.py     # Diagnostic plots
│   │   ├── reports.py         # Report generation
│   │   └── interactive.py     # Interactive dashboards
│   └── utils/
│       ├── __init__.py
│       ├── validation.py      # Input validation
│       ├── metrics.py         # Performance metrics
│       └── logging.py         # Logging utilities
├── tests/
│   ├── __init__.py
│   ├── conftest.py            # Pytest fixtures
│   ├── unit/
│   │   ├── test_arima.py
│   │   ├── test_regression.py
│   │   ├── test_data_loaders.py
│   │   └── ...
│   ├── integration/
│   │   ├── test_pipeline.py
│   │   ├── test_seasonal_adjustment.py
│   │   └── ...
│   ├── fixtures/
│   │   ├── sample_data.py
│   │   └── mock_data.py
│   └── performance/
│       └── test_benchmarks.py
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   ├── model_development.ipynb
│   └── validation_results.ipynb
├── scripts/
│   ├── run_seasonal_adjustment.py
│   ├── generate_reports.py
│   └── validate_results.py
├── data/
│   ├── raw/               # Original data files
│   ├── processed/         # Processed data
│   └── results/          # Output results
├── docs/
│   ├── api/              # API documentation
│   ├── user_guide.md     # User documentation
│   └── development.md    # Developer documentation
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── .github/
│   └── workflows/
│       ├── tests.yml     # CI/CD pipeline
│       └── quality.yml   # Code quality checks
├── requirements.txt
├── requirements-dev.txt
├── setup.py
├── pyproject.toml
├── .env.example
├── .gitignore
├── README.md
└── Makefile
```

## 3. Implementation Phases

### Phase 1: Core Infrastructure (Week 1-2)

#### 3.1.1 Project Setup
- Initialize repository with structure
- Configure development environment
- Set up CI/CD pipeline
- Implement logging and configuration management

#### 3.1.2 Data Layer
```python
# src/data/loaders.py
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Optional

class DataLoader(ABC):
    """Abstract base class for data loaders"""
    
    @abstractmethod
    def load(self, **kwargs) -> pd.DataFrame:
        pass

class HPIDataLoader(DataLoader):
    """Load FHFA HPI repeat-sales data"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
    
    def load(self, start_date: str, end_date: str, 
             geography: str = "state") -> pd.DataFrame:
        # Implementation here
        pass

class WeatherDataLoader(DataLoader):
    """Load NOAA weather data"""
    
    def load(self, geography: str, variables: List[str]) -> pd.DataFrame:
        # Implementation here
        pass

# src/data/validators.py
class DataValidator:
    """Validate input data quality"""
    
    @staticmethod
    def validate_time_series(df: pd.DataFrame) -> bool:
        # Check for gaps, outliers, consistency
        pass
    
    @staticmethod
    def validate_panel_data(df: pd.DataFrame) -> bool:
        # Check panel structure, missing values
        pass
```

### Phase 2: Model Implementation (Week 3-4)

#### 3.2.1 RegARIMA Model
```python
# src/models/arima/regarima.py
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.x13 import x13_arima_analysis
from typing import Tuple, Dict, Optional

class RegARIMA:
    """RegARIMA model implementation following FHFA specification"""
    
    def __init__(self, 
                 ar_order: int = 1,
                 diff_order: int = 1, 
                 ma_order: int = 2,
                 seasonal_order: Tuple[int, int, int, int] = (0, 1, 1, 4),
                 exog_vars: Optional[List[str]] = None):
        self.order = (ar_order, diff_order, ma_order)
        self.seasonal_order = seasonal_order
        self.exog_vars = exog_vars
        self.model = None
        self.results = None
    
    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> 'RegARIMA':
        """Fit RegARIMA model"""
        # Implement model fitting with proper error handling
        pass
    
    def seasonal_adjust(self, method: str = "x13") -> pd.Series:
        """Apply seasonal adjustment"""
        if method == "x13":
            return self._x13_adjust()
        else:
            return self._manual_adjust()
    
    def _x13_adjust(self) -> pd.Series:
        """Use X-13ARIMA-SEATS for adjustment"""
        # Wrapper around statsmodels x13_arima_analysis
        pass
    
    def diagnostics(self) -> Dict[str, float]:
        """Run diagnostic tests"""
        return {
            "ljung_box": self._ljung_box_test(),
            "normality": self._normality_test(),
            "heteroscedasticity": self._het_test()
        }
```

#### 3.2.2 Regression Models
```python
# src/models/regression/panel.py
import linearmodels as lm
from linearmodels.panel import PanelOLS, RandomEffects
import pandas as pd

class SeasonalityImpactModel:
    """Panel regression for seasonality impact analysis"""
    
    def __init__(self, model_type: str = "fixed_effects"):
        self.model_type = model_type
        self.model = None
        self.results = None
    
    def prepare_panel_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for panel regression"""
        # Set multi-index (entity, time)
        # Calculate dependent variable (|NSA - SA|)
        # Engineer features
        pass
    
    def fit(self, data: pd.DataFrame, formula: str) -> 'SeasonalityImpactModel':
        """Fit panel regression model"""
        if self.model_type == "fixed_effects":
            self.model = PanelOLS.from_formula(formula, data)
        elif self.model_type == "random_effects":
            self.model = RandomEffects.from_formula(formula, data)
        
        self.results = self.model.fit(cov_type='clustered', cluster_entity=True)
        return self
    
    def quantile_regression(self, data: pd.DataFrame, quantiles: List[float]):
        """Implement quantile regression for distributional analysis"""
        # Use statsmodels QuantReg
        pass
```

### Phase 3: Pipeline Integration (Week 5-6)

#### 3.3.1 Main Pipeline
```python
# src/pipeline/orchestrator.py
from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd
from loguru import logger

@dataclass
class PipelineConfig:
    """Configuration for seasonal adjustment pipeline"""
    geography_level: str  # "state", "msa", "national"
    start_date: str
    end_date: str
    index_type: str  # "purchase_only", "all_transactions", etc.
    parallel: bool = True
    n_jobs: int = -1

class SeasonalAdjustmentPipeline:
    """Main orchestrator for seasonal adjustment process"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.data_loader = self._init_data_loader()
        self.models = {}
        self.results = {}
    
    def run(self) -> Dict[str, pd.DataFrame]:
        """Execute full pipeline"""
        logger.info(f"Starting pipeline for {self.config.geography_level}")
        
        # Stage 1: Data loading
        data = self._load_data()
        
        # Stage 2: Data validation and preprocessing
        validated_data = self._validate_and_preprocess(data)
        
        # Stage 3: Model fitting (parallel if configured)
        if self.config.parallel:
            adjusted_series = self._parallel_adjustment(validated_data)
        else:
            adjusted_series = self._sequential_adjustment(validated_data)
        
        # Stage 4: Diagnostics
        diagnostics = self._run_diagnostics(adjusted_series)
        
        # Stage 5: Impact analysis
        impact_results = self._analyze_impacts(adjusted_series, validated_data)
        
        # Stage 6: Generate outputs
        outputs = self._generate_outputs(adjusted_series, diagnostics, impact_results)
        
        return outputs
    
    def _parallel_adjustment(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        """Parallel processing for multiple geographies"""
        from dask import delayed, compute
        
        tasks = []
        for geo_id, geo_data in data.items():
            task = delayed(self._adjust_single_geography)(geo_id, geo_data)
            tasks.append(task)
        
        results = compute(*tasks)
        return dict(results)
```

### Phase 4: Testing Implementation (Week 7-8)

#### 3.4.1 Unit Tests
```python
# tests/unit/test_arima.py
import pytest
import numpy as np
import pandas as pd
from src.models.arima import RegARIMA
from hypothesis import given, strategies as st

class TestRegARIMA:
    """Unit tests for RegARIMA model"""
    
    @pytest.fixture
    def sample_time_series(self):
        """Generate sample time series with known seasonality"""
        np.random.seed(42)
        n = 100
        t = np.arange(n)
        trend = 0.5 * t
        seasonal = 10 * np.sin(2 * np.pi * t / 4)  # Quarterly seasonality
        noise = np.random.normal(0, 1, n)
        y = trend + seasonal + noise
        
        dates = pd.date_range('2010-01-01', periods=n, freq='Q')
        return pd.Series(y, index=dates)
    
    def test_model_initialization(self):
        """Test model can be initialized with correct parameters"""
        model = RegARIMA(ar_order=1, diff_order=1, ma_order=1)
        assert model.order == (1, 1, 1)
        assert model.seasonal_order == (0, 1, 1, 4)
    
    def test_model_fitting(self, sample_time_series):
        """Test model fitting on sample data"""
        model = RegARIMA()
        model.fit(sample_time_series)
        
        assert model.results is not None
        assert hasattr(model.results, 'aic')
        assert hasattr(model.results, 'bic')
    
    def test_seasonal_adjustment(self, sample_time_series):
        """Test seasonal adjustment removes seasonality"""
        model = RegARIMA()
        model.fit(sample_time_series)
        adjusted = model.seasonal_adjust()
        
        # Check that variance is reduced
        assert adjusted.std() < sample_time_series.std()
        
        # Check that seasonal pattern is removed
        from statsmodels.tsa.seasonal import seasonal_decompose
        decomp = seasonal_decompose(adjusted, model='additive', period=4)
        assert decomp.seasonal.std() < 0.1  # Minimal seasonality remaining
    
    @given(st.integers(min_value=0, max_value=3),
           st.integers(min_value=0, max_value=2),
           st.integers(min_value=0, max_value=3))
    def test_model_order_validation(self, p, d, q):
        """Property-based test for model order validation"""
        if p + d + q == 0:
            with pytest.raises(ValueError):
                RegARIMA(ar_order=p, diff_order=d, ma_order=q)
        else:
            model = RegARIMA(ar_order=p, diff_order=d, ma_order=q)
            assert model.order == (p, d, q)
    
    def test_diagnostics(self, sample_time_series):
        """Test diagnostic outputs"""
        model = RegARIMA()
        model.fit(sample_time_series)
        diagnostics = model.diagnostics()
        
        assert 'ljung_box' in diagnostics
        assert 'normality' in diagnostics
        assert diagnostics['ljung_box'] > 0.05  # No autocorrelation

# tests/unit/test_regression.py
class TestSeasonalityImpact:
    """Unit tests for seasonality impact models"""
    
    @pytest.fixture
    def panel_data(self):
        """Generate panel data for testing"""
        # Create multi-index data with states and time
        states = ['CA', 'TX', 'FL', 'NY'] * 20
        dates = pd.date_range('2010-01-01', periods=20, freq='Q').tolist() * 4
        
        data = pd.DataFrame({
            'state': states,
            'date': dates,
            'nsa_hpi': np.random.uniform(100, 200, 80),
            'sa_hpi': np.random.uniform(100, 200, 80),
            'temp_range_q1': np.random.uniform(10, 30, 80),
            'temp_range_q2': np.random.uniform(15, 35, 80),
            'temp_range_q3': np.random.uniform(20, 40, 80),
            'temp_range_q4': np.random.uniform(5, 25, 80),
            'avg_sales': np.random.uniform(1000, 5000, 80)
        })
        
        data['seasonality'] = np.abs(data['nsa_hpi'] - data['sa_hpi'])
        return data.set_index(['state', 'date'])
    
    def test_fixed_effects_model(self, panel_data):
        """Test fixed effects regression"""
        model = SeasonalityImpactModel(model_type="fixed_effects")
        formula = 'seasonality ~ temp_range_q1 + temp_range_q2 + temp_range_q3 + temp_range_q4 + avg_sales'
        
        model.fit(panel_data, formula)
        
        assert model.results is not None
        assert model.results.rsquared > 0  # Some explanatory power
        assert len(model.results.params) == 5  # All coefficients estimated
```

#### 3.4.2 Integration Tests
```python
# tests/integration/test_pipeline.py
import pytest
from src.pipeline import SeasonalAdjustmentPipeline, PipelineConfig

class TestPipeline:
    """Integration tests for full pipeline"""
    
    @pytest.fixture
    def test_config(self):
        return PipelineConfig(
            geography_level="state",
            start_date="2010-01-01",
            end_date="2020-12-31",
            index_type="purchase_only",
            parallel=False  # Easier to debug
        )
    
    def test_end_to_end_pipeline(self, test_config, tmp_path):
        """Test complete pipeline execution"""
        # Use mock data for testing
        pipeline = SeasonalAdjustmentPipeline(test_config)
        
        # Mock data loading
        with pytest.mock.patch.object(pipeline, '_load_data') as mock_load:
            mock_load.return_value = self._generate_mock_data()
            
            results = pipeline.run()
            
            assert 'adjusted_indices' in results
            assert 'diagnostics' in results
            assert 'impact_analysis' in results
            
            # Check output formats
            assert isinstance(results['adjusted_indices'], pd.DataFrame)
            assert all(col in results['adjusted_indices'].columns 
                      for col in ['nsa', 'sa', 'adjustment_factor'])
    
    def test_parallel_processing(self, test_config):
        """Test parallel processing capabilities"""
        test_config.parallel = True
        test_config.n_jobs = 2
        
        pipeline = SeasonalAdjustmentPipeline(test_config)
        # Test execution time is reasonable
        # Test results match sequential processing
```

#### 3.4.3 Test Coverage Strategy
```python
# tests/conftest.py
import pytest
from pytest_cov import plugin

def pytest_configure(config):
    """Configure pytest with coverage requirements"""
    config.option.cov_source = ['src']
    config.option.cov_branch = True
    config.option.cov_report = {
        'term-missing': True,
        'html': 'htmlcov',
        'xml': 'coverage.xml'
    }
    config.option.cov_fail_under = 80  # Minimum 80% coverage

# Makefile targets for testing
test:
	pytest tests/ -v

test-coverage:
	pytest tests/ --cov=src --cov-report=term-missing --cov-report=html --cov-fail-under=80

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-performance:
	pytest tests/performance/ -v -m performance
```

### Phase 5: Performance Optimization (Week 9)

#### 3.5.1 Performance Optimizations
```python
# src/utils/performance.py
from functools import lru_cache
import numba as nb
import numpy as np

@nb.jit(nopython=True)
def fast_seasonal_decomposition(y: np.ndarray, period: int) -> tuple:
    """Numba-optimized seasonal decomposition"""
    # Implement fast moving average and decomposition
    pass

class PerformanceMonitor:
    """Monitor and log performance metrics"""
    
    def __init__(self):
        self.metrics = {}
    
    def time_operation(self, operation_name: str):
        """Decorator to time operations"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start = time.time()
                result = func(*args, **kwargs)
                elapsed = time.time() - start
                
                self.metrics[operation_name] = {
                    'elapsed_time': elapsed,
                    'timestamp': datetime.now()
                }
                logger.info(f"{operation_name} completed in {elapsed:.2f} seconds")
                
                return result
            return wrapper
        return decorator

# Caching for repeated calculations
@lru_cache(maxsize=1000)
def cached_weather_calculations(state: str, quarter: int) -> float:
    """Cache weather calculations to avoid recomputation"""
    pass
```

### Phase 6: Validation and Documentation (Week 10)

#### 3.6.1 Validation Suite
```python
# scripts/validate_results.py
import click
import pandas as pd
from src.validation import ResultValidator

@click.command()
@click.option('--baseline-path', required=True, help='Path to baseline results')
@click.option('--new-path', required=True, help='Path to new results')
@click.option('--tolerance', default=0.001, help='Tolerance for comparison')
def validate_results(baseline_path: str, new_path: str, tolerance: float):
    """Validate new results against baseline"""
    validator = ResultValidator(tolerance=tolerance)
    
    baseline = pd.read_csv(baseline_path)
    new_results = pd.read_csv(new_path)
    
    validation_report = validator.compare(baseline, new_results)
    
    if validation_report['all_tests_passed']:
        click.echo("✅ All validation tests passed!")
    else:
        click.echo("❌ Validation failed:")
        for test, result in validation_report['tests'].items():
            if not result['passed']:
                click.echo(f"  - {test}: {result['message']}")

if __name__ == '__main__':
    validate_results()
```

## 4. Testing Strategy

### 4.1 Test Coverage Requirements
- **Core Models (src/models/)**: Minimum 90% coverage
- **Data Processing (src/data/)**: Minimum 85% coverage
- **Pipeline (src/pipeline/)**: Minimum 80% coverage
- **Overall Project**: Minimum 80% coverage

### 4.2 Test Types
1. **Unit Tests**: Test individual functions and classes
2. **Integration Tests**: Test component interactions
3. **End-to-End Tests**: Test complete workflows
4. **Performance Tests**: Benchmark against requirements
5. **Property-Based Tests**: Use Hypothesis for edge cases
6. **Regression Tests**: Ensure changes don't break existing functionality

### 4.3 Test Data Strategy
```python
# tests/fixtures/test_data_generator.py
class TestDataGenerator:
    """Generate realistic test data for various scenarios"""
    
    @staticmethod
    def generate_time_series(n_periods: int, 
                           seasonality: bool = True,
                           trend: bool = True,
                           outliers: bool = False) -> pd.Series:
        """Generate time series with known properties"""
        pass
    
    @staticmethod
    def generate_panel_data(n_entities: int,
                          n_periods: int,
                          missing_rate: float = 0.0) -> pd.DataFrame:
        """Generate panel data for regression tests"""
        pass
    
    @staticmethod
    def generate_edge_cases() -> Dict[str, pd.DataFrame]:
        """Generate edge cases for robustness testing"""
        return {
            'missing_values': self._missing_values_case(),
            'extreme_outliers': self._outliers_case(),
            'short_series': self._short_series_case(),
            'structural_break': self._structural_break_case()
        }
```

## 5. Performance Benchmarks

### 5.1 Target Performance Metrics
Based on PRD requirements:
- United States level: < 15 minutes
- State level (50 states): < 90 minutes
- Top 100 MSAs: < 150 minutes
- Memory usage: < 16GB RAM for largest dataset

### 5.2 Performance Testing
```python
# tests/performance/test_benchmarks.py
import pytest
import time
from memory_profiler import profile

@pytest.mark.performance
class TestPerformance:
    
    def test_state_processing_time(self, large_state_dataset):
        """Test state-level processing meets time requirements"""
        start = time.time()
        
        pipeline = SeasonalAdjustmentPipeline(
            PipelineConfig(geography_level="state", ...)
        )
        pipeline.run()
        
        elapsed = time.time() - start
        assert elapsed < 90 * 60  # 90 minutes in seconds
    
    @profile
    def test_memory_usage(self, large_dataset):
        """Test memory usage stays within limits"""
        # Memory profiler will track usage
        pass
```

## 6. Deployment Strategy

### 6.1 Docker Configuration
```dockerfile
# docker/Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ ./src/
COPY scripts/ ./scripts/

# Set environment variables
ENV PYTHONPATH=/app
ENV FHFA_CONFIG_PATH=/app/config/production.yaml

CMD ["python", "scripts/run_seasonal_adjustment.py"]
```

### 6.2 CI/CD Pipeline
```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run linting
      run: |
        flake8 src/ tests/
        mypy src/
    
    - name: Run tests with coverage
      run: |
        pytest tests/ --cov=src --cov-report=xml --cov-fail-under=80
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

## 7. Monitoring and Maintenance

### 7.1 Logging Strategy
```python
# src/utils/logging.py
from loguru import logger
import sys

def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Configure application logging"""
    logger.remove()  # Remove default handler
    
    # Console logging
    logger.add(
        sys.stderr,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} | {message}",
        level=log_level
    )
    
    # File logging
    if log_file:
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} | {message}",
            rotation="100 MB",
            retention="30 days",
            level="DEBUG"
        )
    
    return logger
```

### 7.2 Model Monitoring
```python
# src/monitoring/model_monitor.py
class ModelMonitor:
    """Monitor model performance and drift"""
    
    def __init__(self, baseline_metrics: Dict[str, float]):
        self.baseline_metrics = baseline_metrics
        self.alerts = []
    
    def check_model_drift(self, current_metrics: Dict[str, float], 
                         threshold: float = 0.1) -> bool:
        """Check if model performance has drifted"""
        for metric, baseline_value in self.baseline_metrics.items():
            current_value = current_metrics.get(metric)
            if current_value is None:
                continue
                
            drift = abs(current_value - baseline_value) / baseline_value
            if drift > threshold:
                self.alerts.append({
                    'metric': metric,
                    'baseline': baseline_value,
                    'current': current_value,
                    'drift_percentage': drift * 100
                })
        
        return len(self.alerts) == 0
```

## 8. Documentation Requirements

### 8.1 API Documentation
- Use docstrings following Google style
- Generate API docs with Sphinx
- Include examples in docstrings

### 8.2 User Guide
- Installation instructions
- Quick start guide
- Complete parameter reference
- Troubleshooting section

### 8.3 Developer Documentation
- Architecture overview
- Contributing guidelines
- Testing guide
- Performance optimization tips

## 9. Timeline Summary

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| Phase 1: Infrastructure | 2 weeks | Project setup, data layer, CI/CD |
| Phase 2: Models | 2 weeks | RegARIMA, regression models |
| Phase 3: Pipeline | 2 weeks | Orchestration, parallel processing |
| Phase 4: Testing | 2 weeks | Unit/integration tests, 80%+ coverage |
| Phase 5: Optimization | 1 week | Performance tuning, benchmarks |
| Phase 6: Validation | 1 week | Validation suite, documentation |
| **Total** | **10 weeks** | **Production-ready system** |

## 10. Success Criteria

1. **Functional Requirements**: All PRD requirements implemented
2. **Test Coverage**: Core models ≥ 90%, overall ≥ 80%
3. **Performance**: Meets PRD processing time requirements
4. **Code Quality**: Passes all linting, type checking
5. **Documentation**: Complete API and user documentation
6. **Validation**: Results match FHFA baseline within tolerance

## 11. Risk Mitigation

| Risk | Mitigation Strategy |
|------|-------------------|
| X-13ARIMA-SEATS integration issues | Implement fallback to statsmodels ARIMA |
| Performance bottlenecks | Use Dask for parallelization, Numba for optimization |
| Data quality issues | Comprehensive validation layer, automated alerts |
| Model drift over time | Continuous monitoring, automated retraining pipeline |
| Dependency conflicts | Use Docker containers, pin versions |

## 12. Next Steps

1. Set up development environment
2. Create repository with initial structure
3. Implement data loaders and validators
4. Begin RegARIMA model development
5. Set up continuous integration pipeline

This implementation plan provides a comprehensive roadmap for building a production-ready FHFA seasonal adjustment system with robust testing and maintainability.