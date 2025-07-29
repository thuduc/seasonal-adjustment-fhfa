import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

from src.data.data_loader import DataLoader


class TestDataLoader:
    """Unit tests for DataLoader class."""
    
    @pytest.fixture
    def data_loader(self):
        """Create DataLoader instance."""
        return DataLoader()
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
            
    def create_test_hpi_data(self, temp_dir):
        """Create test HPI data file."""
        data = {
            'geography_id': ['CA'] * 20 + ['NY'] * 20,
            'geography_type': ['STATE'] * 40,
            'period': pd.date_range('2018-01-01', periods=20, freq='Q').tolist() * 2,
            'nsa_index': np.random.uniform(100, 120, 40),
            'num_transactions': np.random.randint(1000, 5000, 40)
        }
        df = pd.DataFrame(data)
        filepath = Path(temp_dir) / 'test_hpi.csv'
        df.to_csv(filepath, index=False)
        return filepath
    
    def create_test_weather_data(self, temp_dir):
        """Create test weather data file."""
        data = {
            'geography_id': ['CA'] * 20 + ['NY'] * 20,
            'period': pd.date_range('2018-01-01', periods=20, freq='Q').tolist() * 2,
            'temp_range': np.random.uniform(10, 30, 40),
            'avg_temp': np.random.uniform(40, 80, 40),
            'precipitation': np.random.uniform(0, 10, 40)
        }
        df = pd.DataFrame(data)
        filepath = Path(temp_dir) / 'test_weather.csv'
        df.to_csv(filepath, index=False)
        return filepath
    
    def test_load_hpi_data_success(self, data_loader, temp_dir):
        """Test successful HPI data loading."""
        filepath = self.create_test_hpi_data(temp_dir)
        
        df = data_loader.load_hpi_data(filepath)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 40
        assert 'geography_id' in df.columns
        assert 'nsa_index' in df.columns
        assert df['period'].dtype == 'datetime64[ns]'
        
    def test_load_weather_data_success(self, data_loader, temp_dir):
        """Test successful weather data loading."""
        filepath = self.create_test_weather_data(temp_dir)
        
        df = data_loader.load_weather_data(filepath)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 40
        assert 'temp_range' in df.columns
        assert 'precipitation' in df.columns
        
    def test_validate_hpi_data(self, data_loader):
        """Test HPI data validation."""
        # Valid data
        valid_df = pd.DataFrame({
            'geography_id': ['CA'] * 20,
            'geography_type': ['STATE'] * 20,
            'period': pd.date_range('2018-01-01', periods=20, freq='Q'),
            'nsa_index': np.random.uniform(100, 120, 20),
            'num_transactions': np.random.randint(1000, 5000, 20)
        })
        
        assert data_loader.validate_data(valid_df, 'hpi') == True
        
        # Invalid geography type
        invalid_df = valid_df.copy()
        invalid_df['geography_type'] = 'INVALID'
        assert data_loader.validate_data(invalid_df, 'hpi') == False
        
        # Insufficient history
        short_df = valid_df.iloc[:10]
        result = data_loader.validate_data(short_df, 'hpi')
        assert result == True  # Should pass but with warning
        
    def test_validate_weather_data(self, data_loader):
        """Test weather data validation."""
        # Valid data
        valid_df = pd.DataFrame({
            'geography_id': ['CA'] * 10,
            'period': pd.date_range('2020-01-01', periods=10, freq='Q'),
            'temp_range': np.random.uniform(10, 30, 10),
            'avg_temp': np.random.uniform(40, 80, 10),
            'precipitation': np.random.uniform(0, 10, 10)
        })
        
        assert data_loader.validate_data(valid_df, 'weather') == True
        
        # Invalid temperature range
        invalid_df = valid_df.copy()
        invalid_df.loc[0, 'temp_range'] = -10
        result = data_loader.validate_data(invalid_df, 'weather')
        assert result == True  # Should pass but with warning
        
    def test_validate_demographics_data(self, data_loader):
        """Test demographics data validation."""
        # Valid data
        valid_df = pd.DataFrame({
            'geography_id': ['CA', 'NY'],
            'pct_over_65': [15.0, 14.0],
            'pct_white': [60.0, 65.0],
            'pct_bachelors': [35.0, 40.0],
            'pct_single_family': [65.0, 55.0],
            'avg_income': [75000, 80000]
        })
        
        assert data_loader.validate_data(valid_df, 'demographics') == True
        
        # Invalid percentage
        invalid_df = valid_df.copy()
        invalid_df.loc[0, 'pct_over_65'] = 150.0
        assert data_loader.validate_data(invalid_df, 'demographics') == False
        
    def test_merge_datasets(self, data_loader, temp_dir):
        """Test merging multiple datasets."""
        # Create and load test data
        hpi_path = self.create_test_hpi_data(temp_dir)
        weather_path = self.create_test_weather_data(temp_dir)
        
        data_loader.load_hpi_data(hpi_path)
        data_loader.load_weather_data(weather_path)
        
        # Create demographics data
        data_loader.demographics_data = pd.DataFrame({
            'geography_id': ['CA', 'NY'],
            'pct_over_65': [15.0, 14.0],
            'pct_bachelors': [35.0, 40.0]
        })
        
        # Merge datasets
        master_df = data_loader.merge_datasets()
        
        assert isinstance(master_df, pd.DataFrame)
        assert 'nsa_index' in master_df.columns
        assert 'temp_range' in master_df.columns
        assert 'pct_over_65' in master_df.columns
        
    def test_create_sample_data(self, data_loader, temp_dir):
        """Test sample data creation."""
        data_loader.create_sample_data(temp_dir)
        
        # Check files were created
        expected_files = [
            'sample_hpi_data.csv',
            'sample_weather_data.csv',
            'sample_demographics_data.csv',
            'sample_industry_data.csv'
        ]
        
        for filename in expected_files:
            filepath = Path(temp_dir) / filename
            assert filepath.exists()
            
            # Load and validate each file
            df = pd.read_csv(filepath)
            assert len(df) > 0
            
    def test_empty_data_validation(self, data_loader):
        """Test validation with empty DataFrame."""
        empty_df = pd.DataFrame()
        assert data_loader.validate_data(empty_df, 'hpi') == False
        
    def test_missing_required_columns(self, data_loader):
        """Test validation with missing required columns."""
        incomplete_df = pd.DataFrame({
            'geography_id': ['CA'] * 10,
            'period': pd.date_range('2020-01-01', periods=10, freq='Q')
            # Missing nsa_index and other required columns
        })
        
        assert data_loader.validate_data(incomplete_df, 'hpi') == False