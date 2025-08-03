# Developer Guide - FHFA Seasonal Adjustment Pipeline

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Development Setup](#development-setup)
3. [Code Structure](#code-structure)
4. [Adding New Features](#adding-new-features)
5. [Testing Guidelines](#testing-guidelines)
6. [Performance Optimization](#performance-optimization)
7. [Contributing](#contributing)

## Architecture Overview

### Design Principles

The pipeline follows these core design principles:

1. **Modularity**: Each component has a single responsibility
2. **Extensibility**: Easy to add new models and methods
3. **Testability**: Comprehensive test coverage with clear interfaces
4. **Performance**: Optimized for large-scale data processing
5. **Maintainability**: Clear code structure with proper documentation

### Component Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Data Layer    │────▶│  Model Layer     │────▶│ Pipeline Layer  │
│                 │     │                  │     │                 │
│ - Loaders       │     │ - RegARIMA       │     │ - Orchestrator  │
│ - Validators    │     │ - Panel Models   │     │ - Stages        │
│ - Preprocessors │     │ - Adjusters      │     │ - Parallel Proc │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         │                       │                         │
         ▼                       ▼                         ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Validation Layer│     │ Visualization    │     │  Config/Utils   │
│                 │     │                  │     │                 │
│ - Result Valid. │     │ - Plots          │     │ - Settings      │
│ - Data Quality  │     │ - Reports        │     │ - Logging       │
│ - Model Valid.  │     │ - Interactive    │     │ - Monitoring    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

### Data Flow

1. **Input Stage**: Data loaders read and validate raw data
2. **Processing Stage**: Preprocessors clean and transform data
3. **Modeling Stage**: Models fit and generate predictions
4. **Analysis Stage**: Impact analysis and diagnostics
5. **Output Stage**: Results validation and report generation

## Development Setup

### Environment Setup

1. **Clone and setup development environment:**
   ```bash
   git clone <repository-url>
   cd fhfa-seasonal-adjustment/impl-pandas
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt -r requirements-dev.txt
   ```

2. **Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

3. **Configure IDE (VS Code example):**
   ```json
   {
     "python.linting.enabled": true,
     "python.linting.flake8Enabled": true,
     "python.formatting.provider": "black",
     "python.testing.pytestEnabled": true,
     "python.testing.pytestArgs": ["tests"]
   }
   ```

### Development Dependencies

```txt
# requirements-dev.txt
pytest>=7.4.0
pytest-cov>=4.1.0
pytest-mock>=3.11.0
hypothesis>=6.82.0
black>=23.7.0
flake8>=6.1.0
mypy>=1.5.0
pre-commit>=3.3.0
ipdb>=0.13.13
jupyter>=1.0.0
```

## Code Structure

### Module Organization

```
src/
├── config/           # Configuration management
│   ├── settings.py   # Pydantic settings
│   └── constants.py  # Model constants
├── data/            # Data handling
│   ├── loaders.py   # Data loading interfaces
│   ├── validators.py # Data validation
│   └── preprocessors.py # Data cleaning
├── models/          # Statistical models
│   ├── arima/       # ARIMA models
│   ├── regression/  # Panel regression
│   └── seasonal/    # Seasonal adjustment
├── pipeline/        # Pipeline orchestration
│   ├── orchestrator.py # Main pipeline
│   └── stages.py    # Pipeline stages
├── validation/      # Result validation
│   ├── result_validator.py
│   └── model_validator.py
└── utils/           # Utilities
    ├── logging.py   # Logging setup
    └── monitoring.py # Performance monitoring
```

### Key Base Classes

```python
# src/models/base.py
from abc import ABC, abstractmethod

class BaseModel(ABC):
    """Abstract base class for all models"""
    
    @abstractmethod
    def fit(self, data, **kwargs):
        """Fit model to data"""
        pass
    
    @abstractmethod
    def predict(self, **kwargs):
        """Generate predictions"""
        pass
    
    @abstractmethod
    def get_diagnostics(self):
        """Return model diagnostics"""
        pass

# src/data/base.py
class BaseLoader(ABC):
    """Abstract base class for data loaders"""
    
    @abstractmethod
    def load(self, **kwargs):
        """Load data from source"""
        pass
    
    @abstractmethod
    def validate(self, data):
        """Validate loaded data"""
        pass
```

## Adding New Features

### Adding a New Model

1. **Create model class inheriting from BaseModel:**

```python
# src/models/custom/my_model.py
from src.models.base import BaseModel
import pandas as pd

class MyCustomModel(BaseModel):
    """Custom model implementation"""
    
    def __init__(self, param1: float = 1.0, param2: str = 'default'):
        self.param1 = param1
        self.param2 = param2
        self.is_fitted = False
        
    def fit(self, data: pd.DataFrame, **kwargs):
        """Fit custom model"""
        # Implementation
        self.is_fitted = True
        return self
        
    def predict(self, n_periods: int = 1):
        """Generate predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        # Implementation
        return predictions
        
    def get_diagnostics(self):
        """Return diagnostics"""
        return {
            'metric1': self.calculate_metric1(),
            'metric2': self.calculate_metric2()
        }
```

2. **Add comprehensive tests:**

```python
# tests/unit/test_my_model.py
import pytest
from src.models.custom.my_model import MyCustomModel

class TestMyCustomModel:
    
    @pytest.fixture
    def sample_data(self):
        # Create sample data
        return pd.DataFrame(...)
    
    def test_initialization(self):
        model = MyCustomModel(param1=2.0)
        assert model.param1 == 2.0
        assert not model.is_fitted
    
    def test_fit(self, sample_data):
        model = MyCustomModel()
        model.fit(sample_data)
        assert model.is_fitted
        
    def test_predict_before_fit(self):
        model = MyCustomModel()
        with pytest.raises(ValueError):
            model.predict()
```

3. **Integrate with pipeline:**

```python
# src/pipeline/stages.py
def model_selection_stage(config):
    """Select appropriate model based on config"""
    if config.model_type == 'my_custom':
        from src.models.custom.my_model import MyCustomModel
        return MyCustomModel(**config.model_params)
```

### Adding a New Data Source

1. **Implement loader interface:**

```python
# src/data/loaders.py
class MyDataLoader(BaseLoader):
    """Custom data source loader"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        
    def load(self, query: str) -> pd.DataFrame:
        """Load data from custom source"""
        # Connect to data source
        # Execute query
        # Return DataFrame
        pass
        
    def validate(self, data: pd.DataFrame) -> bool:
        """Validate data structure"""
        required_columns = ['date', 'value']
        return all(col in data.columns for col in required_columns)
```

2. **Add to pipeline configuration:**

```python
# src/config/settings.py
class DataSourceSettings(BaseSettings):
    source_type: str = 'csv'
    connection_string: Optional[str] = None
    
    @validator('source_type')
    def validate_source_type(cls, v):
        valid_types = ['csv', 'database', 'api', 'my_custom']
        if v not in valid_types:
            raise ValueError(f"Invalid source type: {v}")
        return v
```

## Testing Guidelines

### Test Structure

```
tests/
├── unit/           # Unit tests for individual components
├── integration/    # Integration tests for workflows
├── fixtures/       # Test data and fixtures
├── performance/    # Performance benchmarks
└── conftest.py     # Shared pytest fixtures
```

### Writing Tests

1. **Follow AAA pattern (Arrange, Act, Assert):**

```python
def test_seasonal_adjustment():
    # Arrange
    data = generate_test_data()
    adjuster = SeasonalAdjuster(method='x11')
    
    # Act
    result = adjuster.adjust(data)
    
    # Assert
    assert len(result) == len(data)
    assert result.std() < data.std()  # Variance reduced
```

2. **Use fixtures for reusable test data:**

```python
@pytest.fixture
def quarterly_series():
    """Generate quarterly time series for testing"""
    dates = pd.date_range('2010-01-01', periods=40, freq='QE')
    values = 100 + np.cumsum(np.random.randn(40))
    return pd.Series(values, index=dates)
```

3. **Test edge cases:**

```python
def test_empty_series():
    """Test handling of empty input"""
    adjuster = SeasonalAdjuster()
    with pytest.raises(ValueError, match="Empty series"):
        adjuster.adjust(pd.Series())

def test_short_series():
    """Test handling of insufficient data"""
    short_series = pd.Series([1, 2, 3])
    adjuster = SeasonalAdjuster()
    with pytest.raises(ValueError, match="Insufficient data"):
        adjuster.adjust(short_series)
```

### Property-Based Testing

Use Hypothesis for comprehensive testing:

```python
from hypothesis import given, strategies as st

@given(
    n_obs=st.integers(min_value=20, max_value=1000),
    trend=st.floats(min_value=-1, max_value=1),
    noise_level=st.floats(min_value=0.1, max_value=10)
)
def test_model_robustness(n_obs, trend, noise_level):
    """Test model handles various data characteristics"""
    data = generate_series(n_obs, trend, noise_level)
    model = RegARIMA()
    
    # Should not raise exceptions
    model.fit(data)
    diagnostics = model.get_diagnostics()
    
    # Basic sanity checks
    assert diagnostics['aic'] is not None
    assert 0 <= diagnostics['ljung_box_pvalue'] <= 1
```

### Test Coverage

Maintain minimum coverage requirements:

```bash
# Run tests with coverage
pytest --cov=src --cov-report=html --cov-fail-under=80

# View coverage report
open htmlcov/index.html
```

Focus coverage on:
- Core models: 90%+ coverage
- Data processing: 85%+ coverage
- Pipeline logic: 80%+ coverage

## Performance Optimization

### Profiling Code

1. **Use cProfile for function-level profiling:**

```python
import cProfile
import pstats

def profile_function():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Code to profile
    pipeline.run_full_pipeline()
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions
```

2. **Memory profiling:**

```python
from memory_profiler import profile

@profile
def memory_intensive_function():
    # Function implementation
    pass
```

### Optimization Strategies

1. **Vectorization with NumPy/Pandas:**

```python
# Bad: Loop-based calculation
def calculate_returns_slow(prices):
    returns = []
    for i in range(1, len(prices)):
        returns.append((prices[i] - prices[i-1]) / prices[i-1])
    return returns

# Good: Vectorized calculation
def calculate_returns_fast(prices):
    return prices.pct_change()
```

2. **Caching expensive computations:**

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_calculation(param1, param2):
    # Expensive computation
    return result
```

3. **Parallel processing with Dask:**

```python
import dask.dataframe as dd

def process_large_dataset(df):
    # Convert to Dask DataFrame
    ddf = dd.from_pandas(df, npartitions=10)
    
    # Apply operations in parallel
    result = ddf.groupby('state').apply(
        lambda x: seasonal_adjustment(x),
        meta=('adjusted', 'float64')
    )
    
    return result.compute()
```

4. **Numba for computational bottlenecks:**

```python
import numba as nb

@nb.jit(nopython=True)
def fast_moving_average(x, window):
    """Numba-optimized moving average"""
    n = len(x)
    result = np.empty(n)
    
    for i in range(n):
        if i < window - 1:
            result[i] = np.nan
        else:
            result[i] = np.mean(x[i-window+1:i+1])
    
    return result
```

### Monitoring Performance

1. **Add performance tracking decorators:**

```python
from src.utils.monitoring import track_metrics

@track_metrics()
def critical_function():
    # Function that needs monitoring
    pass
```

2. **Set up alerts for performance degradation:**

```python
# src/monitoring/alerts.py
def check_performance_thresholds():
    metrics = collector.get_recent_metrics(hours=1)
    
    for metric in metrics:
        if metric['duration'] > THRESHOLD:
            send_alert(f"Performance degradation in {metric['function']}")
```

## Contributing

### Development Workflow

1. **Create feature branch:**
   ```bash
   git checkout -b feature/my-new-feature
   ```

2. **Make changes following style guide:**
   - Use Black for formatting
   - Follow PEP 8
   - Add type hints
   - Write docstrings

3. **Run tests and checks:**
   ```bash
   # Format code
   black src/ tests/
   
   # Run linting
   flake8 src/ tests/
   
   # Type checking
   mypy src/
   
   # Run tests
   pytest tests/
   ```

4. **Submit pull request:**
   - Clear description of changes
   - Link to related issues
   - Include test results
   - Update documentation

### Code Review Checklist

- [ ] Tests pass with adequate coverage
- [ ] Code follows project style guide
- [ ] Documentation is updated
- [ ] No performance regressions
- [ ] Security considerations addressed
- [ ] Backward compatibility maintained

### Release Process

1. **Update version:**
   ```python
   # src/__init__.py
   __version__ = "1.2.0"
   ```

2. **Update changelog:**
   ```markdown
   ## [1.2.0] - 2024-01-15
   ### Added
   - New feature X
   ### Fixed
   - Bug in Y
   ```

3. **Tag release:**
   ```bash
   git tag -a v1.2.0 -m "Release version 1.2.0"
   git push origin v1.2.0
   ```

### Getting Help

- **Documentation**: Start with user guide and API reference
- **Issues**: Check existing issues before creating new ones
- **Discussions**: Use GitHub discussions for questions
- **Contact**: Reach maintainers via email for sensitive issues