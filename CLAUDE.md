# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements a seasonal adjustment model for house price indices using the X-13ARIMA-SEATS methodology. The implementation is based on pandas/numpy and processes housing price index (HPI) data to remove seasonal patterns, identify outliers, and analyze causation factors.

## Development Commands

### Setup and Installation
```bash
cd impl-pandas
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Running Tests
```bash
# Run all tests with coverage
pytest

# Run specific test file
pytest tests/unit/test_x13_engine.py

# Run specific test
pytest tests/unit/test_x13_engine.py::TestX13ARIMAEngine::test_pre_adjustment

# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Generate HTML coverage report
pytest --cov=src --cov-report=html
```

### Development Setup
```bash
# Install development dependencies
pip install -e ".[dev]"

# Format code with black
black src/ tests/

# Run linting
flake8 src/ tests/
```

## Architecture Overview

The codebase follows a modular architecture with clear separation of concerns:

### Core Components

1. **Data Layer** (`src/data/`)
   - `data_loader.py`: Handles loading and validation of HPI, weather, demographic, and industry data

2. **Models Layer** (`src/models/`)
   - `x13_engine.py`: Core X-13ARIMA-SEATS implementation for seasonal adjustment
   - `model_selection.py`: Automatic ARIMA model selection with grid search and cross-validation
   - `pipeline.py`: End-to-end pipeline orchestrating the entire adjustment process

3. **Analysis Layer** (`src/analysis/`)
   - `causation_analysis.py`: Analyzes weather and demographic impacts on seasonality
   - `feature_engineering.py`: Creates features for regression analysis

4. **Output Layer** (`src/output/`)
   - `report_generator.py`: Creates comprehensive PDF/Excel reports
   - `results_manager.py`: Manages and exports adjustment results

5. **Utilities** (`src/utils/`)
   - `cache_manager.py`: Caching for improved performance
   - `parallel_processor.py`: Parallel processing for batch operations

### Key Processing Flow

1. Data is loaded and validated via `DataLoader`
2. `SeasonalAdjustmentPipeline` orchestrates the process:
   - Pre-adjustment for trading day effects
   - Outlier detection using Modified Z-score with MAD
   - ARIMA model selection (automatic or manual)
   - Seasonal factor extraction
   - Diagnostics (Ljung-Box, Jarque-Bera, ADF tests)
3. Results are cached and can be exported in various formats

### Important Implementation Details

- Minimum 20 observations required per geography
- Default ARIMA model: (1,1,1)(1,1,1)4 with quarterly seasonality
- Parallel processing uses ProcessPoolExecutor for batch operations
- Test coverage is 100% with both unit and integration tests
- Pre-generated sample data in `data/` directory speeds up test execution

## Data Requirements

### HPI Data Format
- `geography_id`: Geographic identifier
- `geography_type`: Type (STATE, MSA, etc.)
- `period`: Date in quarterly format
- `nsa_index`: Non-seasonally adjusted index
- `num_transactions`: Number of transactions

### Optional Data Formats
Weather, demographic, and industry data follow similar CSV structures with geography_id as the key field.