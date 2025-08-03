"""Tests for data validators"""

import pytest
import pandas as pd
import numpy as np

from src.data.validators import TimeSeriesValidator, PanelDataValidator


class TestTimeSeriesValidator:
    """Test time series validator"""
    
    def setup_method(self):
        """Set up test data"""
        self.validator = TimeSeriesValidator()
        
        # Create valid time series
        dates = pd.date_range('2015-01-01', '2023-12-31', freq='QE')
        self.valid_series = pd.Series(
            np.random.uniform(100, 200, len(dates)),
            index=dates,
            name='test_series'
        )
        
    def test_validate_valid_series(self):
        """Test validation of valid series"""
        result = self.validator.validate(self.valid_series)
        
        assert result['overall_valid'] == True
        assert result['has_datetime_index'] == True
        assert result['has_frequency'] == True
        assert result['missing_ratio'] == 0.0
        assert result['min_length_check'] == True
        
    def test_validate_missing_values(self):
        """Test validation with missing values"""
        # Add missing values
        series_with_missing = self.valid_series.copy()
        series_with_missing.iloc[5:10] = np.nan
        
        result = self.validator.validate(series_with_missing)
        
        assert result['missing_ratio'] > 0
        assert result['missing_ratio'] < 0.2  # Should be around 5/48 ≈ 0.104
        
    def test_validate_no_datetime_index(self):
        """Test validation without datetime index"""
        series = pd.Series([1, 2, 3, 4, 5])
        result = self.validator.validate(series)
        
        assert result['has_datetime_index'] == False
        assert result['overall_valid'] == False
        
    def test_validate_outliers(self):
        """Test outlier detection"""
        # Add outliers
        series_with_outliers = self.valid_series.copy()
        series_with_outliers.iloc[10] = 1000  # Extreme value
        
        result = self.validator.validate(series_with_outliers)
        
        assert result['outlier_ratio'] > 0
        assert len(result['outlier_indices']) > 0
        # Check that an outlier was detected around the inserted position
        # The indices are Timestamps, so check by position
        outlier_positions = [self.valid_series.index.get_loc(idx) for idx in result['outlier_indices']]
        assert any(abs(pos - 10) <= 1 for pos in outlier_positions)  # Allow some tolerance
        
    def test_stationarity_test(self):
        """Test stationarity testing"""
        # Create non-stationary series (trending)
        dates = pd.date_range('2015-01-01', '2023-12-31', freq='QE')
        trend = np.linspace(100, 200, len(dates))
        noise = np.random.normal(0, 5, len(dates))
        trending_series = pd.Series(trend + noise, index=dates)
        
        result = self.validator.test_stationarity(trending_series)
        
        assert 'adf_statistic' in result
        assert 'adf_pvalue' in result
        assert 'is_stationary' in result
        

class TestPanelDataValidator:
    """Test panel data validator"""
    
    def setup_method(self):
        """Set up test data"""
        self.validator = PanelDataValidator()
        
        # Create valid panel data
        dates = pd.date_range('2015-01-01', '2023-12-31', freq='QE')
        cbsas = ['12345', '67890', '11111']
        
        panel_data = []
        for cbsa in cbsas:
            for date in dates:
                panel_data.append({
                    'date': date,
                    'cbsa_code': cbsa,
                    'hpi': np.random.uniform(100, 200),
                    'temperature': np.random.normal(65, 10),
                    'population': np.random.uniform(100000, 1000000)
                })
        
        self.valid_panel = pd.DataFrame(panel_data)
        
    def test_validate_valid_panel(self):
        """Test validation of valid panel data"""
        result = self.validator.validate(
            self.valid_panel,
            entity_col='cbsa_code',
            time_col='date'
        )
        
        assert result['overall_valid'] == True
        assert result['is_balanced'] == True
        assert result['n_entities'] == 3
        assert result['n_periods'] > 0
        
    def test_validate_unbalanced_panel(self):
        """Test validation of unbalanced panel"""
        # Remove some observations
        unbalanced = self.valid_panel.iloc[5:]  # Remove first 5 obs
        
        result = self.validator.validate(
            unbalanced,
            entity_col='cbsa_code',
            time_col='date'
        )
        
        assert result['is_balanced'] == False
        assert result['overall_valid'] == True  # Can still be valid if unbalanced
        
    def test_validate_missing_columns(self):
        """Test validation with missing required columns"""
        # Remove a column
        incomplete = self.valid_panel.drop('temperature', axis=1)
        
        result = self.validator.validate(
            incomplete,
            entity_col='cbsa_code',
            time_col='date',
            required_cols=['temperature', 'hpi']
        )
        
        assert result['overall_valid'] == False
        # Check that temperature is mentioned in validation errors
        assert any('temperature' in error for error in result['validation_errors'])
        
    def test_multicollinearity_check(self):
        """Test multicollinearity detection"""
        # Add highly correlated variable
        panel_with_collinear = self.valid_panel.copy()
        panel_with_collinear['hpi_duplicate'] = panel_with_collinear['hpi'] * 1.1 + 5
        
        numeric_cols = ['hpi', 'hpi_duplicate', 'temperature']
        result = self.validator.check_multicollinearity(
            panel_with_collinear[numeric_cols]
        )
        
        assert result['high_vif_vars'] is not None
        assert len(result['high_vif_vars']) > 0
        assert result['max_vif'] > 10  # Should detect high VIF