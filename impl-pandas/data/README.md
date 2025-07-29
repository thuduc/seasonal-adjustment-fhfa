# Pre-generated Sample Data

This directory contains pre-generated sample data files used for unit and integration tests. These files help speed up test execution by avoiding the need to regenerate sample data on every test run.

## Files

- `sample_hpi_data.csv` - Sample House Price Index data with geography IDs, periods, and NSA indices
- `sample_weather_data.csv` - Sample weather data including temperature ranges and precipitation
- `sample_demographics_data.csv` - Sample demographic data with population statistics
- `sample_industry_data.csv` - Sample industry composition data

## Usage

### Running Tests with Pre-generated Data (Default)

By default, tests will use the pre-generated data from this directory:

```bash
pytest
```

### Running Tests with Fresh Data

To force tests to generate new data instead of using pre-generated files:

```bash
USE_GENERATED_DATA=false pytest
```

### Regenerating Sample Data

To regenerate the sample data files:

```bash
source venv/bin/activate
python generate_sample_data.py
```

This will overwrite the existing sample data files in this directory.

## Benefits

1. **Faster Test Execution**: Tests run ~30% faster by skipping data generation
2. **Consistent Test Data**: All test runs use the same baseline data
3. **Offline Testing**: Tests can run without needing to generate data
4. **Debugging**: Easier to debug issues with known, stable test data