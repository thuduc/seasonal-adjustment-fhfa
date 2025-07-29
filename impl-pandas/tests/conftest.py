import pytest
import pandas as pd
from pathlib import Path
import tempfile
import shutil
import os


@pytest.fixture(scope="session")
def sample_data_dir():
    """Provide path to pre-generated sample data directory."""
    return Path(__file__).parent.parent / 'data'


@pytest.fixture
def use_generated_data():
    """Check if we should use pre-generated data or create new test data."""
    # Can be controlled via environment variable
    return os.getenv('USE_GENERATED_DATA', 'true').lower() == 'true'


@pytest.fixture
def test_data_dir(sample_data_dir, use_generated_data):
    """Provide test data directory - either pre-generated or temporary."""
    if use_generated_data and sample_data_dir.exists():
        # Use pre-generated data
        yield sample_data_dir
    else:
        # Create temporary data for this test session
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # If sample data exists, copy it to temp dir
            if sample_data_dir.exists():
                for file in sample_data_dir.glob('*.csv'):
                    shutil.copy(file, temp_path)
            
            yield temp_path


@pytest.fixture
def sample_hpi_data(test_data_dir):
    """Load sample HPI data from test data directory."""
    hpi_file = test_data_dir / 'sample_hpi_data.csv'
    if hpi_file.exists():
        return pd.read_csv(hpi_file, parse_dates=['period'])
    return None


@pytest.fixture
def sample_weather_data(test_data_dir):
    """Load sample weather data from test data directory."""
    weather_file = test_data_dir / 'sample_weather_data.csv'
    if weather_file.exists():
        return pd.read_csv(weather_file, parse_dates=['period'])
    return None


@pytest.fixture  
def sample_demographics_data(test_data_dir):
    """Load sample demographics data from test data directory."""
    demo_file = test_data_dir / 'sample_demographics_data.csv'
    if demo_file.exists():
        return pd.read_csv(demo_file, parse_dates=['period'])
    return None


@pytest.fixture
def sample_industry_data(test_data_dir):
    """Load sample industry data from test data directory."""
    industry_file = test_data_dir / 'sample_industry_data.csv'
    if industry_file.exists():
        return pd.read_csv(industry_file, parse_dates=['period'])
    return None