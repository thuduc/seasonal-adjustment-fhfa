# User Guide - FHFA Seasonal Adjustment Pipeline

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Data Preparation](#data-preparation)
5. [Running Analyses](#running-analyses)
6. [Understanding Results](#understanding-results)
7. [Common Workflows](#common-workflows)
8. [Troubleshooting](#troubleshooting)

## Introduction

The FHFA Seasonal Adjustment Pipeline implements the methodology described in the FHFA whitepaper for seasonally adjusting house price indices. It provides:

- **RegARIMA modeling** for time series decomposition
- **X-13ARIMA-SEATS** integration for robust seasonal adjustment
- **Panel regression analysis** to understand weather and market impacts
- **Comprehensive reporting** with visualizations and diagnostics

### Key Concepts

- **Seasonal Adjustment**: Removing predictable seasonal patterns from time series
- **RegARIMA**: Regression with ARIMA errors for modeling time series
- **Panel Data**: Data with both cross-sectional (geographic) and time dimensions
- **Fixed Effects**: Controlling for entity-specific characteristics

## Installation

### Prerequisites

- Python 3.10 or higher
- 8GB RAM minimum (16GB recommended for large datasets)
- X-13ARIMA-SEATS binary (optional, for advanced adjustment)

### Step-by-Step Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd fhfa-seasonal-adjustment/impl-pandas
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

5. **Verify installation:**
   ```bash
   python -c "from src.pipeline import SeasonalAdjustmentPipeline; print('Installation successful!')"
   ```

### Docker Installation

For containerized deployment:

```bash
docker build -t fhfa-seasonal-adjustment .
docker run -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output fhfa-seasonal-adjustment
```

## Quick Start

### Running with Default Settings

The simplest way to run the pipeline:

```bash
python main.py
```

This will:
- Use synthetic data for demonstration
- Run seasonal adjustment
- Perform impact analysis
- Generate HTML and Excel reports

### Using Your Own Data

```bash
python main.py \
  --hpi-data data/my_hpi_data.csv \
  --weather-data data/my_weather_data.csv \
  --output-dir results/
```

### Adjustment Only Mode

If you only need seasonal adjustment:

```bash
python main.py --mode adjustment-only --hpi-data data/hpi.csv
```

## Data Preparation

### HPI Data Format

Your HPI data should be in CSV format with:
- Date index (quarterly or monthly)
- Columns named `hpi_state_XX` or `hpi_cbsa_XXXXX`

Example structure:
```csv
date,hpi_state_CA,hpi_state_TX,hpi_state_FL
2010-01-01,100.0,95.5,98.2
2010-04-01,101.2,96.1,99.0
...
```

### Weather Data Format

Weather data should include:
- Geographic identifiers matching HPI data
- Temperature, precipitation, and other weather variables
- Same time period as HPI data

Example structure:
```csv
date,geography,temperature,precipitation
2010-01-01,CA,65.2,3.4
2010-01-01,TX,72.1,2.1
...
```

### Data Quality Requirements

- Minimum 2 years of data (8 quarters)
- Less than 20% missing values
- No gaps in time series
- Positive HPI values

## Running Analyses

### Command Line Interface

The main CLI provides several options:

```bash
python main.py [OPTIONS]

Options:
  --hpi-data PATH              Path to HPI data file
  --weather-data PATH          Path to weather data file
  --demographics-data PATH     Path to demographics data file
  --output-dir PATH           Output directory [default: ./output]
  --mode [full|adjustment-only|impact-only]
                              Analysis mode [default: full]
  --geography [national|state|msa]
                              Geographic level [default: state]
  --start-date DATE           Start date (YYYY-MM-DD)
  --end-date DATE             End date (YYYY-MM-DD)
  --report-formats TEXT       Report formats (comma-separated)
  --parallel / --no-parallel  Enable parallel processing
  --n-jobs INTEGER            Number of parallel jobs
  --verbose                   Verbose output
  --help                      Show this message and exit
```

### Python API

For programmatic use:

```python
from src.pipeline.orchestrator import SeasonalAdjustmentPipeline

# Initialize pipeline
pipeline = SeasonalAdjustmentPipeline()

# Run analysis
results = pipeline.run_full_pipeline(
    hpi_data_path="data/hpi.csv",
    weather_data_path="data/weather.csv",
    output_dir="./output",
    generate_report=True
)

# Access results
adjusted_series = results['adjusted_indices']
coefficients = results['impact_analysis']['coefficients']
```

### Validation Tools

Validate your data before processing:

```bash
# Validate data quality
python scripts/validate_results.py data \
  --data-path data/hpi.csv \
  --data-type time_series

# Compare results against baseline
python scripts/validate_results.py results \
  --baseline-path baseline/results.csv \
  --new-path output/adjusted_series.csv \
  --tolerance 0.001
```

## Understanding Results

### Output Files

The pipeline generates several output files:

1. **adjusted_series.csv**: Seasonally adjusted time series
   - Original values
   - Adjusted values
   - Seasonal factors
   - Trend component

2. **fixed_effects_coefficients.csv**: Panel regression results
   - Weather impact coefficients by quarter
   - Standard errors and p-values
   - Entity fixed effects

3. **diagnostics.json**: Model diagnostics
   - Ljung-Box test results
   - Normality tests
   - Model fit statistics

4. **Report files**:
   - `seasonal_adjustment_report_YYYYMMDD.html`: Interactive HTML report
   - `seasonal_adjustment_results_YYYYMMDD.xlsx`: Excel workbook with charts

### Interpreting Results

#### Seasonal Adjustment Quality

Good seasonal adjustment should show:
- Reduced seasonal variation in adjusted series
- Preserved long-term trend
- Stable seasonal factors over time

Warning signs:
- Adjusted series still shows clear seasonal patterns
- Extreme seasonal factors (>20% of original values)
- Unstable decomposition

#### Regression Coefficients

Temperature coefficients interpretation:
- Positive coefficient: Higher temperatures associated with higher seasonality
- Magnitude: Percentage point change in seasonality per degree
- Significance: p-value < 0.05 indicates statistical significance

Expected patterns:
- Q1 & Q2: Small positive coefficients (0.011, 0.014)
- Q3: Larger positive coefficient (0.027)
- Q4: Negative coefficient (-0.018)

### Visualization Guide

The HTML report includes interactive plots:

1. **Time Series Plots**: Original vs adjusted series
   - Hover for exact values
   - Zoom to examine specific periods
   - Toggle series on/off

2. **Decomposition Plots**: Trend, seasonal, irregular components
   - Check seasonal pattern stability
   - Identify structural breaks
   - Examine residuals

3. **Diagnostic Plots**: Model validation
   - Q-Q plots for normality
   - ACF plots for autocorrelation
   - Residual scatter plots

4. **Coefficient Plots**: Regression results
   - Confidence intervals
   - Quarter-by-quarter comparison
   - Geographic variations

## Common Workflows

### Workflow 1: Monthly National Analysis

```bash
# Prepare monthly national data
python scripts/prepare_national_data.py \
  --input data/raw/hpi_monthly.csv \
  --output data/processed/hpi_national.csv

# Run adjustment with monthly frequency
python main.py \
  --hpi-data data/processed/hpi_national.csv \
  --geography national \
  --mode adjustment-only
```

### Workflow 2: State-Level Impact Analysis

```bash
# Run full pipeline for states
python main.py \
  --hpi-data data/hpi_states.csv \
  --weather-data data/weather_states.csv \
  --geography state \
  --start-date 2010-01-01 \
  --end-date 2023-12-31

# Generate additional visualizations
python scripts/generate_state_maps.py \
  --results output/fixed_effects_coefficients.csv \
  --output output/maps/
```

### Workflow 3: Custom Seasonal Adjustment

```python
from src.models.seasonal.adjuster import SeasonalAdjuster
import pandas as pd

# Load your data
data = pd.read_csv('my_data.csv', index_col=0, parse_dates=True)

# Configure custom adjustment
adjuster = SeasonalAdjuster(
    method='x11',              # Use X-11 method
    outlier_detection=True,    # Enable outlier handling
    forecast_years=2,          # Extend series for stability
    trading_day_adjustment=True # Adjust for trading days
)

# Run adjustment
adjusted = adjuster.adjust(data['hpi'], frequency=4)

# Save results
results_df = pd.DataFrame({
    'original': data['hpi'],
    'adjusted': adjusted,
    'seasonal_factor': data['hpi'] / adjusted
})
results_df.to_csv('custom_adjustment.csv')
```

### Workflow 4: Batch Processing Multiple Geographies

```python
from pathlib import Path
import pandas as pd
from src.pipeline.orchestrator import SeasonalAdjustmentPipeline

# Process each state separately
pipeline = SeasonalAdjustmentPipeline()
results_dict = {}

for state_file in Path('data/states/').glob('hpi_state_*.csv'):
    state = state_file.stem.split('_')[-1]
    print(f"Processing {state}...")
    
    results = pipeline.run_seasonal_adjustment_only(
        pd.read_csv(state_file, index_col=0, parse_dates=True)
    )
    results_dict[state] = results

# Combine results
combined_results = pd.DataFrame(results_dict)
combined_results.to_csv('output/all_states_adjusted.csv')
```

## Troubleshooting

### Common Issues and Solutions

#### Issue: "Insufficient data for seasonal adjustment"
**Solution**: Ensure you have at least 2 years (8 quarters) of data. For shorter series, try:
```python
adjuster = SeasonalAdjuster(method='classical')  # Less data requirements
```

#### Issue: "Model convergence failed"
**Solution**: Check for:
- Extreme outliers in data
- Too many missing values
- Structural breaks

Clean data before processing:
```python
from src.data.preprocessors import DataPreprocessor
preprocessor = DataPreprocessor()
clean_data = preprocessor.clean(data, handle_outliers=True)
```

#### Issue: "Memory error with large datasets"
**Solution**: Enable chunked processing:
```bash
python main.py --parallel --n-jobs 4 --chunk-size 1000
```

Or use Dask for very large data:
```python
import dask.dataframe as dd
ddf = dd.from_pandas(large_df, npartitions=10)
```

#### Issue: "X-13ARIMA-SEATS not found"
**Solution**: The pipeline will fall back to statsmodels. To use X-13:
1. Download X-13 binary from Census Bureau
2. Set path in .env: `X13_PATH=/path/to/x13as`
3. Or set environment variable: `export X13_PATH=/path/to/x13as`

### Performance Tips

1. **Enable parallel processing** for multiple geographies:
   ```bash
   python main.py --parallel --n-jobs -1  # Use all CPUs
   ```

2. **Cache intermediate results**:
   ```python
   pipeline = SeasonalAdjustmentPipeline(cache_results=True)
   ```

3. **Optimize data types**:
   ```python
   df = df.astype({'hpi': 'float32'})  # Reduce memory usage
   ```

4. **Profile performance**:
   ```bash
   python main.py --enable-monitoring --metrics-export metrics.json
   ```

### Getting Help

1. **Check logs**: Detailed error information in `logs/pipeline.log`
2. **Run diagnostics**: `python scripts/diagnose_data.py --data-path your_data.csv`
3. **Enable debug mode**: `python main.py --verbose`
4. **Review test data**: Examples in `tests/fixtures/`

### Reporting Issues

When reporting issues, include:
- Error message and stack trace
- Data characteristics (shape, time range)
- Command or code used
- Environment details (Python version, OS)

## Next Steps

- Explore advanced options in the [API Reference](api_reference.md)
- Read the [Developer Guide](developer_guide.md) for customization
- Check example notebooks in `notebooks/`
- Review the original [FHFA whitepaper](../references/) for methodology details