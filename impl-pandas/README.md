# FHFA Seasonal Adjustment Implementation (Pandas)

This is the Pandas-based implementation of the FHFA seasonal adjustment methodology for housing price indices, based on the FHFA whitepaper "Applying Seasonal Adjustments to Housing Markets".

## Overview

This pipeline implements:
- **RegARIMA models** for time series decomposition
- **X-13ARIMA-SEATS** seasonal adjustment methodology
- **Panel regression models** with fixed/random effects for impact analysis
- **Quantile regression** for distributional analysis
- **Comprehensive visualization and reporting** functionality

## Features

### Phase 1: Core Infrastructure ✅
- Data loaders for HPI, weather, and demographic data
- Time series and panel data validators
- Data preprocessing and transformation utilities
- Configuration management with Pydantic v2

### Phase 2: Model Implementation ✅
- RegARIMA implementation with seasonal components
- X-13ARIMA wrapper (placeholder for actual X-13 integration)
- Panel regression models with entity/time effects
- Quantile regression for multiple quantiles
- Seasonal adjustment orchestrator

### Phase 3: Visualization & Reporting ✅
- Interactive plots for seasonal adjustment results
- Diagnostic visualizations for model validation
- HTML/Excel/JSON report generation
- Enhanced pipeline orchestration

### Phase 4: Testing Framework ✅
- Property-based testing with Hypothesis
- Performance benchmarks and profiling
- Comprehensive test data generators
- 64% test coverage (exceeds 60% target)

### Phase 5: Production Features ✅
- CLI interface with Click
- Environment-based configuration
- Structured logging and monitoring
- Docker containerization
- Resource usage tracking

## Installation

### Local Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd impl-pandas
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment:
```bash
cp .env.example .env
# Edit .env with your settings
```

5. (Optional) Install X-13ARIMA-SEATS binary for advanced seasonal adjustment

### Docker Installation

1. Build the Docker image:
```bash
docker build -t fhfa-seasonal-adjustment .
```

2. Run with docker-compose:
```bash
docker-compose up fhfa-pipeline
```

3. Or run standalone:
```bash
docker run -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output fhfa-seasonal-adjustment python main.py

## Quick Start

### Using the CLI

Run with synthetic data:
```bash
python main.py
```

Run with your own data:
```bash
python main.py --hpi-data data/hpi.csv --weather-data data/weather.csv
```

Run adjustment only:
```bash
python main.py --mode adjustment-only
```

Generate different report formats:
```bash
python main.py --report-formats html excel json
```

### Monitoring and Metrics

Enable resource monitoring:
```bash
python main.py --enable-monitoring --metrics-export metrics.json
```

View metrics dashboard:
```bash
docker-compose up metrics-viewer
# Access at http://localhost:3000
```

### Using the API

```python
from src.pipeline.orchestrator import SeasonalAdjustmentPipeline

# Initialize pipeline
pipeline = SeasonalAdjustmentPipeline()

# Run full pipeline
results = pipeline.run_full_pipeline(
    hpi_data_path="data/hpi.csv",
    weather_data_path="data/weather.csv",
    output_dir="./output",
    generate_report=True,
    report_formats=['html', 'excel']
)

# Get summary
print(pipeline.get_summary_report())
```

## Model Specifications

### RegARIMA Model
The core model follows the specification:

```
φ(B)Φ(Bs)(1 - B)^d(yt - Σ βixit) = θ(B)Θ(Bs)εt
```

Where:
- φ(B): AR polynomial
- Φ(Bs): Seasonal AR polynomial  
- θ(B): MA polynomial
- Θ(Bs): Seasonal MA polynomial
- d: Differencing order
- xit: Exogenous variables

### Panel Regression Model
```
Yit = α + βXit + γZi + δt + εit
```

With temperature coefficients by quarter:
- Q1: 0.011
- Q2: 0.014
- Q3: 0.027
- Q4: -0.018

## Output Structure

```
output/
├── adjusted_series.csv          # Seasonally adjusted series
├── fixed_effects_coefficients.csv   # Panel regression results
├── quantile_coefficients.csv    # Quantile regression results
├── diagnostics.json             # Model diagnostics
├── full_results.json            # Complete results
├── seasonal_adjustment_report_YYYYMMDD_HHMMSS.html
└── seasonal_adjustment_results_YYYYMMDD_HHMMSS.xlsx
```

## Advanced Usage

### Custom Seasonal Adjustment

```python
from src.models.seasonal.adjuster import SeasonalAdjuster

# Create adjuster
adjuster = SeasonalAdjuster(
    method='classical',
    outlier_detection=True,
    forecast_years=1
)

# Run adjustment
adjusted = adjuster.adjust(series, frequency=4)

# Get diagnostics
results = adjuster.get_results()
```

### Panel Regression Analysis

```python
from src.models.regression.panel import SeasonalityImpactModel

# Create model
model = SeasonalityImpactModel(
    model_type='fixed_effects',
    entity_effects=True,
    time_effects=True
)

# Fit model
model.fit(
    panel_data,
    y_col='hpi_growth',
    weather_col='temperature',
    entity_col='cbsa',
    time_col='date'
)

# Get results
coefficients = model.get_coefficients()
```

## Data Requirements

### HPI Data
- CSV format with date index
- Columns: `hpi_state_XX` or `hpi_cbsa_XXXXX`
- Quarterly or monthly frequency

### Weather Data
- Temperature, precipitation, and other weather variables
- Matched to geographic identifiers

### Demographic Data
- Population, income, industry shares
- Annual frequency, matched to geographic areas

## Testing

Run all tests:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=src --cov-report=term-missing
```

Current test coverage: **64%** (exceeds 60% target)

## Configuration

Settings can be configured via environment variables or `.env` file:

```env
FHFA_DATA_DIR=/path/to/data
FHFA_OUTPUT_DIR=/path/to/output
FHFA_LOG_LEVEL=INFO
FHFA_N_JOBS=-1  # Use all CPU cores
```

## Limitations

- X-13ARIMA integration requires external X-13 binary (not included)
- Synthetic data generator for testing purposes only
- Some advanced seasonal adjustment methods require additional dependencies

## Future Enhancements

- [ ] Full X-13ARIMA-SEATS integration
- [ ] Real-time data streaming support
- [ ] Interactive dashboard
- [ ] Cloud deployment options
- [ ] GPU acceleration for large datasets

## License

This implementation is for demonstration purposes based on public FHFA methodology.