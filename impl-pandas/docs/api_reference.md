# API Reference

This document provides a comprehensive API reference for the FHFA Seasonal Adjustment Pipeline.

## Core Models

### RegARIMA Model

```python
from src.models.arima.regarima import RegARIMA

# Initialize model
model = RegARIMA(
    ar_order=1,           # AR order
    diff_order=1,         # Differencing order
    ma_order=2,           # MA order
    seasonal_order=(0, 1, 1, 4)  # Seasonal ARIMA order
)

# Fit model
model.fit(y_series, X=None)

# Generate predictions
predictions = model.predict(n_periods=4)

# Get diagnostics
diagnostics = model.diagnostics()
```

**Parameters:**
- `ar_order` (int): Autoregressive order (default: 1)
- `diff_order` (int): Differencing order (default: 1)
- `ma_order` (int): Moving average order (default: 2)
- `seasonal_order` (tuple): Seasonal ARIMA (P,D,Q,s) order

**Methods:**
- `fit(y, X=None)`: Fit the model to time series data
- `predict(n_periods)`: Generate forecasts
- `diagnostics()`: Return model diagnostics

### Seasonal Adjuster

```python
from src.models.seasonal.adjuster import SeasonalAdjuster

# Create adjuster
adjuster = SeasonalAdjuster(
    method='classical',      # Adjustment method
    outlier_detection=True,  # Enable outlier detection
    forecast_years=1         # Years to forecast
)

# Run adjustment
adjusted_series = adjuster.adjust(
    series,
    frequency=4  # Quarterly data
)

# Get components
components = adjuster.get_components()
```

**Methods:**
- `adjust(series, frequency)`: Perform seasonal adjustment
- `get_components()`: Get trend, seasonal, and irregular components
- `get_results()`: Get detailed adjustment results

### Panel Regression Models

```python
from src.models.regression.panel import SeasonalityImpactModel

# Initialize model
model = SeasonalityImpactModel(
    model_type='fixed_effects',
    entity_effects=True,
    time_effects=True,
    clustered_errors=True
)

# Fit model
model.fit(
    data=panel_data,
    y_col='hpi_growth',
    weather_col='temperature',
    sales_col='sales_volume',
    char_cols=['characteristic_1', 'characteristic_2'],
    industry_cols=['industry_share_1'],
    entity_col='cbsa',
    time_col='date'
)

# Get coefficients
coefficients = model.get_coefficients()

# Test coefficient equality across quarters
test_result = model.test_coefficient_equality(
    feature='temperature',
    quarter1=1,
    quarter2=3
)
```

**Model Types:**
- `'fixed_effects'`: Entity and/or time fixed effects
- `'random_effects'`: Random effects specification
- `'between'`: Between-effects model

### Quantile Regression

```python
from src.models.regression.quantile import QuantileRegressionModel

# Initialize model
model = QuantileRegressionModel(
    quantiles=[0.25, 0.5, 0.75],
    fit_intercept=True
)

# Fit model
model.fit(
    data=df,
    y_col='target',
    x_cols=['feature1', 'feature2']
)

# Get predictions for specific quantiles
predictions = model.predict(new_data)
```

## Data Processing

### Data Loaders

```python
from src.data.loaders import HPIDataLoader, WeatherDataLoader

# Load HPI data
hpi_loader = HPIDataLoader()
hpi_data = hpi_loader.load(
    start_date='2010-01-01',
    end_date='2023-12-31',
    geography='state'  # or 'msa', 'national'
)

# Load weather data
weather_loader = WeatherDataLoader()
weather_data = weather_loader.load(
    geography='state',
    variables=['temperature', 'precipitation'],
    start_date='2010-01-01',
    end_date='2023-12-31'
)
```

### Data Validators

```python
from src.data.validators import TimeSeriesValidator, PanelDataValidator

# Validate time series
ts_validator = TimeSeriesValidator()
validation_result = ts_validator.validate(series)

# Validate panel data
panel_validator = PanelDataValidator()
validation_result = panel_validator.validate(
    panel_data,
    entity_col='cbsa',
    time_col='date'
)
```

## Pipeline Orchestration

### Main Pipeline

```python
from src.pipeline.orchestrator import SeasonalAdjustmentPipeline

# Initialize pipeline
pipeline = SeasonalAdjustmentPipeline()

# Run full pipeline
results = pipeline.run_full_pipeline(
    hpi_data_path='data/hpi.csv',
    weather_data_path='data/weather.csv',
    output_dir='./output',
    generate_report=True,
    report_formats=['html', 'excel', 'json']
)

# Run seasonal adjustment only
adjusted_data = pipeline.run_seasonal_adjustment_only(series)

# Run impact analysis only
impact_results = pipeline.run_impact_analysis_only(
    adjusted_data,
    weather_data,
    entity_col='cbsa',
    time_col='date'
)
```

**Pipeline Methods:**
- `run_full_pipeline()`: Execute complete analysis pipeline
- `run_seasonal_adjustment_only()`: Run only seasonal adjustment
- `run_impact_analysis_only()`: Run only regression analysis
- `get_summary_report()`: Get pipeline execution summary

## Visualization

### Plotting Functions

```python
from src.visualization.plots import SeasonalAdjustmentPlotter, DiagnosticPlotter

# Create seasonal adjustment plots
sa_plotter = SeasonalAdjustmentPlotter()

# Plot original vs adjusted
sa_plotter.plot_adjustment_comparison(
    original=original_series,
    adjusted=adjusted_series,
    title='HPI Seasonal Adjustment'
)

# Plot decomposition
sa_plotter.plot_decomposition(
    decomposition=components,
    title='Seasonal Decomposition'
)

# Create diagnostic plots
diag_plotter = DiagnosticPlotter()

# Plot residual diagnostics
diag_plotter.plot_residual_diagnostics(residuals)

# Plot coefficient analysis
diag_plotter.plot_coefficient_analysis(
    coefficients=coef_df,
    ci_lower=ci_lower,
    ci_upper=ci_upper
)
```

### Report Generation

```python
from src.visualization.reports import ReportGenerator

# Initialize report generator
generator = ReportGenerator(
    template_dir='templates',
    output_dir='output'
)

# Generate HTML report
html_path = generator.generate_html_report(
    results=pipeline_results,
    title='Seasonal Adjustment Report'
)

# Generate Excel report
excel_path = generator.generate_excel_report(
    results=pipeline_results,
    include_charts=True
)
```

## Validation

### Result Validation

```python
from src.validation import ResultValidator

# Initialize validator
validator = ResultValidator(
    tolerance=0.001,
    strict_mode=True
)

# Compare results
report = validator.compare(
    baseline=baseline_df,
    new_results=new_results_df
)

# Validate model outputs
model_report = validator.validate_model_outputs(
    model_results=model.get_metrics(),
    expected_metrics={
        'aic': (-1000, 1000),
        'rsquared': (0.0, 1.0)
    }
)
```

### Data Quality Validation

```python
from src.validation import DataQualityValidator

# Initialize validator
dq_validator = DataQualityValidator()

# Validate time series
ts_report = dq_validator.validate_time_series(
    data=series,
    min_length=8,
    max_missing_ratio=0.2
)

# Validate panel data
panel_report = dq_validator.validate_panel_data(
    data=panel_df,
    entity_col='cbsa',
    time_col='date'
)
```

### Model Validation

```python
from src.validation import ModelValidator

# Initialize validator
model_validator = ModelValidator(significance_level=0.05)

# Validate ARIMA assumptions
arima_report = model_validator.validate_arima_assumptions(
    residuals=model.residuals,
    fitted_values=model.fitted_values
)

# Validate seasonal adjustment
sa_report = model_validator.validate_seasonal_adjustment(
    original=original_series,
    adjusted=adjusted_series,
    frequency=4
)
```

## Configuration

### Settings Management

```python
from src.config import Settings, get_settings

# Get current settings
settings = get_settings()

# Access configuration values
data_dir = settings.data_dir
output_dir = settings.output_dir
log_level = settings.log_level

# Override settings with environment variables
# FHFA_DATA_DIR=/custom/path
# FHFA_LOG_LEVEL=DEBUG
```

### Available Settings:
- `data_dir`: Directory for input data
- `output_dir`: Directory for results
- `log_level`: Logging verbosity
- `n_jobs`: Number of parallel jobs
- `x13_path`: Path to X-13ARIMA binary
- `cache_results`: Enable result caching
- `enable_monitoring`: Enable metrics collection

## Utilities

### Logging

```python
from src.utils.logging import setup_logging
from loguru import logger

# Setup logging
setup_logging(
    log_level='INFO',
    log_file='logs/pipeline.log',
    rotation='100 MB'
)

# Use logger
logger.info("Starting analysis")
logger.warning("Missing data detected")
logger.error("Model fitting failed")
```

### Monitoring

```python
from src.utils.monitoring import MetricsCollector, track_metrics

# Initialize metrics collector
collector = MetricsCollector(db_path='metrics.db')

# Track function metrics
@track_metrics(collector)
def expensive_operation():
    # Function implementation
    pass

# Query metrics
recent_metrics = collector.get_recent_metrics(hours=24)
aggregated = collector.get_aggregated_metrics()
```

## Error Handling

All functions raise appropriate exceptions:

- `ValueError`: Invalid parameter values
- `KeyError`: Missing required columns
- `RuntimeError`: Model fitting failures
- `FileNotFoundError`: Missing data files

Example error handling:

```python
try:
    model.fit(data)
except ValueError as e:
    logger.error(f"Invalid model parameters: {e}")
except RuntimeError as e:
    logger.error(f"Model fitting failed: {e}")
```

## Performance Considerations

- Use `n_jobs=-1` for parallel processing
- Enable caching for repeated analyses
- Consider using Dask for very large datasets
- Monitor memory usage with the metrics collector

## Examples

See the `notebooks/` directory for complete examples:
- `exploratory_analysis.ipynb`: Data exploration workflow
- `seasonal_adjustment_example.ipynb`: Step-by-step adjustment
- `panel_regression_example.ipynb`: Impact analysis tutorial