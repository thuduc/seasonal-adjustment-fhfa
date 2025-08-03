"""Tests for validation components"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import json

from src.validation import ResultValidator, DataQualityValidator, ModelValidator
from src.validation.result_validator import ValidationResult


class TestResultValidator:
    """Test result validation functionality"""
    
    @pytest.fixture
    def sample_baseline(self):
        """Create sample baseline data"""
        dates = pd.date_range('2010-01-01', periods=20, freq='QE')
        return pd.DataFrame({
            'original': 100 + np.cumsum(np.random.randn(20)),
            'adjusted': 100 + np.cumsum(np.random.randn(20) * 0.8),
            'seasonal_factor': 1 + 0.1 * np.sin(np.arange(20) * np.pi / 2)
        }, index=dates)
    
    @pytest.fixture
    def sample_new_results(self, sample_baseline):
        """Create new results similar to baseline"""
        # Add small random noise
        return sample_baseline * (1 + np.random.randn(*sample_baseline.shape) * 0.0001)
    
    def test_validator_initialization(self):
        """Test validator initialization"""
        validator = ResultValidator(tolerance=0.01, strict_mode=True)
        assert validator.tolerance == 0.01
        assert validator.strict_mode is True
        assert validator.validation_results == []
    
    def test_compare_identical_results(self, sample_baseline):
        """Test comparing identical results"""
        validator = ResultValidator(tolerance=0.001)
        report = validator.compare(sample_baseline, sample_baseline.copy())
        
        assert report['all_tests_passed'] is True
        assert report['passed_tests'] == report['total_tests']
        assert report['failed_tests'] == 0
    
    def test_compare_within_tolerance(self, sample_baseline, sample_new_results):
        """Test comparing results within tolerance"""
        validator = ResultValidator(tolerance=0.01)
        report = validator.compare(sample_baseline, sample_new_results)
        
        assert report['all_tests_passed'] is True
        
        # Check individual tests
        assert report['tests']['shape_consistency']['passed'] is True
        assert report['tests']['column_consistency']['passed'] is True
        assert report['tests']['index_consistency']['passed'] is True
    
    def test_compare_outside_tolerance(self, sample_baseline):
        """Test comparing results outside tolerance"""
        # Create results with large differences
        new_results = sample_baseline * 1.1  # 10% difference
        
        validator = ResultValidator(tolerance=0.001)
        report = validator.compare(sample_baseline, new_results)
        
        assert report['all_tests_passed'] is False
        assert report['failed_tests'] > 0
        
        # At least value tolerance tests should fail
        value_tests = [k for k in report['tests'] if 'value_tolerance' in k]
        assert any(not report['tests'][test]['passed'] for test in value_tests)
    
    def test_structural_mismatch(self, sample_baseline):
        """Test detecting structural mismatches"""
        # Different shape
        new_results = sample_baseline.iloc[:-5]  # Remove rows
        
        validator = ResultValidator()
        report = validator.compare(sample_baseline, new_results)
        
        assert report['tests']['shape_consistency']['passed'] is False
        assert 'Shape mismatch' in report['tests']['shape_consistency']['message']
    
    def test_column_mismatch(self, sample_baseline):
        """Test detecting column mismatches"""
        # Add extra column
        new_results = sample_baseline.copy()
        new_results['extra_column'] = 1.0
        
        validator = ResultValidator()
        report = validator.compare(sample_baseline, new_results)
        
        assert report['tests']['column_consistency']['passed'] is False
        assert 'extra' in report['tests']['column_consistency']['details']
    
    def test_seasonal_pattern_validation(self):
        """Test seasonal pattern validation"""
        dates = pd.date_range('2010-01-01', periods=40, freq='QE')
        
        # Create data with clear seasonal pattern
        seasonal = 10 * np.sin(2 * np.pi * np.arange(40) / 4)
        original = pd.DataFrame({
            'sa_hpi': 100 + np.cumsum(np.random.randn(40)) + seasonal
        }, index=dates)
        
        # Adjusted should have less seasonality
        adjusted = pd.DataFrame({
            'sa_hpi': 100 + np.cumsum(np.random.randn(40)) + seasonal * 0.1
        }, index=dates)
        
        validator = ResultValidator()
        report = validator.compare(original, adjusted)
        
        # Should detect seasonal adjustment effectiveness
        seasonal_test = report['tests'].get('seasonal_adjustment_sa_hpi')
        if seasonal_test:
            assert seasonal_test['passed'] is True
    
    def test_save_report(self, tmp_path, sample_baseline):
        """Test saving validation report"""
        validator = ResultValidator()
        report = validator.compare(sample_baseline, sample_baseline)
        
        output_path = tmp_path / "validation_report.json"
        validator.save_report(report, str(output_path))
        
        assert output_path.exists()
        
        # Load and verify
        with open(output_path) as f:
            loaded_report = json.load(f)
        
        assert loaded_report['all_tests_passed'] == report['all_tests_passed']
        assert loaded_report['total_tests'] == report['total_tests']


class TestDataQualityValidator:
    """Test data quality validation"""
    
    @pytest.fixture
    def sample_time_series(self):
        """Create sample time series"""
        dates = pd.date_range('2010-01-01', periods=100, freq='D')
        return pd.Series(
            100 + np.cumsum(np.random.randn(100)),
            index=dates,
            name='test_series'
        )
    
    @pytest.fixture
    def sample_panel_data(self):
        """Create sample panel data"""
        entities = ['A', 'B', 'C'] * 20
        dates = pd.date_range('2010-01-01', periods=20, freq='QE').tolist() * 3
        
        return pd.DataFrame({
            'entity': entities,
            'time': dates,
            'value': np.random.randn(60) * 10 + 100,
            'feature1': np.random.randn(60),
            'feature2': np.random.randn(60)
        })
    
    def test_validator_initialization(self):
        """Test validator initialization"""
        validator = DataQualityValidator(error_threshold=0.1)
        assert validator.error_threshold == 0.1
        assert validator.issues == []
    
    def test_validate_good_time_series(self, sample_time_series):
        """Test validation of good time series"""
        validator = DataQualityValidator()
        report = validator.validate_time_series(sample_time_series)
        
        assert report['valid'] is True
        assert report['error_count'] == 0
        assert isinstance(report['issues'], list)
    
    def test_validate_short_time_series(self):
        """Test validation of too short series"""
        short_series = pd.Series([1, 2, 3], name='short')
        
        validator = DataQualityValidator()
        report = validator.validate_time_series(short_series, min_length=8)
        
        assert report['valid'] is False
        assert report['error_count'] > 0
        
        # Check for insufficient data issue
        issues = [i for i in report['issues'] if i['type'] == 'insufficient_data']
        assert len(issues) > 0
    
    def test_validate_missing_values(self, sample_time_series):
        """Test detection of missing values"""
        # Add missing values
        series_with_missing = sample_time_series.copy()
        series_with_missing.iloc[10:20] = np.nan
        
        validator = DataQualityValidator()
        report = validator.validate_time_series(series_with_missing, max_missing_ratio=0.05)
        
        assert report['valid'] is False
        
        # Check for missing value issue
        issues = [i for i in report['issues'] if 'missing' in i['type']]
        assert len(issues) > 0
    
    def test_validate_outliers(self):
        """Test outlier detection"""
        dates = pd.date_range('2010-01-01', periods=100, freq='D')
        values = np.random.randn(100) * 10 + 100
        values[50] = 1000  # Add outlier
        
        series = pd.Series(values, index=dates)
        
        validator = DataQualityValidator()
        report = validator.validate_time_series(series)
        
        # Should detect outlier
        outlier_issues = [i for i in report['issues'] if i['type'] == 'outliers']
        assert len(outlier_issues) > 0
        # Check that the outlier date is detected
        assert dates[50] in outlier_issues[0]['affected_rows']
    
    def test_validate_constant_series(self):
        """Test detection of constant series"""
        constant_series = pd.Series([100] * 20, name='constant')
        
        validator = DataQualityValidator()
        report = validator.validate_time_series(constant_series)
        
        assert report['valid'] is False
        
        # Check for constant series issue
        issues = [i for i in report['issues'] if i['type'] == 'constant_series']
        assert len(issues) > 0
    
    def test_validate_panel_data(self, sample_panel_data):
        """Test panel data validation"""
        validator = DataQualityValidator()
        report = validator.validate_panel_data(
            sample_panel_data,
            entity_col='entity',
            time_col='time'
        )
        
        assert report['valid'] is True
        assert report['error_count'] == 0
    
    def test_validate_panel_duplicates(self, sample_panel_data):
        """Test detection of duplicate observations"""
        # Add duplicate
        dup_data = sample_panel_data.copy()
        dup_data = pd.concat([dup_data, dup_data.iloc[[0]]], ignore_index=True)
        
        validator = DataQualityValidator()
        report = validator.validate_panel_data(
            dup_data,
            entity_col='entity',
            time_col='time'
        )
        
        assert report['valid'] is False
        
        # Check for duplicate issue
        issues = [i for i in report['issues'] if i['type'] == 'duplicate_observations']
        assert len(issues) > 0
    
    def test_validate_weather_data(self):
        """Test weather data specific validation"""
        weather_data = pd.DataFrame({
            'temperature': [70, 75, -100, 80],  # Unreasonable temp
            'precipitation': [2.5, 3.0, -1.0, 4.0]  # Negative precip
        })
        
        validator = DataQualityValidator()
        report = validator.validate_weather_data(weather_data)
        
        assert report['valid'] is False
        
        # Should detect both issues
        temp_issues = [i for i in report['issues'] if 'temperature' in i['description']]
        precip_issues = [i for i in report['issues'] if 'precipitation' in i['description']]
        
        assert len(temp_issues) > 0
        assert len(precip_issues) > 0


class TestModelValidator:
    """Test model validation functionality"""
    
    @pytest.fixture
    def sample_residuals(self):
        """Create sample residuals"""
        np.random.seed(42)
        return pd.Series(np.random.normal(0, 1, 100))
    
    @pytest.fixture
    def sample_fitted_values(self):
        """Create sample fitted values"""
        return pd.Series(100 + np.cumsum(np.random.randn(100) * 0.5))
    
    def test_validator_initialization(self):
        """Test validator initialization"""
        validator = ModelValidator(significance_level=0.10)
        assert validator.significance_level == 0.10
    
    def test_validate_arima_assumptions_good(self, sample_residuals):
        """Test ARIMA validation with good residuals"""
        validator = ModelValidator()
        report = validator.validate_arima_assumptions(sample_residuals)
        
        assert 'assumptions_met' in report
        assert 'tests' in report
        
        # Should have standard tests
        assert 'normality' in report['tests']
        assert 'autocorrelation' in report['tests']
        assert 'zero_mean' in report['tests']
    
    def test_validate_arima_assumptions_bad(self):
        """Test ARIMA validation with problematic residuals"""
        # Create autocorrelated residuals
        residuals = pd.Series([1, -1] * 50)  # Perfect autocorrelation
        
        validator = ModelValidator()
        report = validator.validate_arima_assumptions(residuals)
        
        assert report['assumptions_met'] is False
        assert report['tests']['autocorrelation']['passed'] is False
    
    def test_validate_regression_assumptions(self, sample_residuals):
        """Test regression assumption validation"""
        X = pd.DataFrame({
            'x1': np.random.randn(100),
            'x2': np.random.randn(100)
        })
        y = pd.Series(np.random.randn(100))
        
        validator = ModelValidator()
        report = validator.validate_regression_assumptions(
            sample_residuals, X, y
        )
        
        assert 'assumptions_met' in report
        assert 'tests' in report
        
        # Should have regression-specific tests
        assert 'linearity' in report['tests']
        assert 'multicollinearity' in report['tests']
        assert 'normality' in report['tests']
    
    def test_validate_seasonal_adjustment(self):
        """Test seasonal adjustment validation"""
        dates = pd.date_range('2010-01-01', periods=40, freq='QE')
        
        # Original with strong seasonality
        seasonal = 10 * np.sin(2 * np.pi * np.arange(40) / 4)
        original = pd.Series(100 + seasonal + np.random.randn(40), index=dates)
        
        # Adjusted with reduced seasonality
        adjusted = pd.Series(100 + seasonal * 0.1 + np.random.randn(40), index=dates)
        
        validator = ModelValidator()
        report = validator.validate_seasonal_adjustment(original, adjusted)
        
        assert 'adjustment_effective' in report
        assert 'metrics' in report
        
        # Should show variance reduction
        if 'seasonal_variance_reduction' in report['metrics']:
            assert report['metrics']['seasonal_variance_reduction']['effective'] is True
    
    def test_vif_calculation(self):
        """Test VIF calculation for multicollinearity"""
        # Create correlated features
        X = pd.DataFrame({
            'x1': np.random.randn(100),
            'x2': np.random.randn(100)
        })
        X['x3'] = X['x1'] + X['x2'] + np.random.randn(100) * 0.1  # High correlation
        
        validator = ModelValidator()
        vif_results = validator._calculate_vif(X)
        
        assert 'x1' in vif_results
        assert 'x2' in vif_results
        assert 'x3' in vif_results
        
        # x3 should have high VIF
        assert vif_results['x3'] > 5
    
    def test_diagnostic_plots_data(self, sample_residuals):
        """Test diagnostic plot data generation"""
        validator = ModelValidator()
        plot_data = validator.generate_diagnostic_plots(sample_residuals)
        
        assert 'qq_plot' in plot_data
        assert 'acf_plot' in plot_data
        assert 'histogram' in plot_data
        
        # Check data structure
        assert 'theoretical' in plot_data['qq_plot']
        assert 'empirical' in plot_data['qq_plot']
        assert len(plot_data['qq_plot']['theoretical']) == len(sample_residuals)
    
    def test_insufficient_data_handling(self):
        """Test handling of insufficient data"""
        short_residuals = pd.Series([1, 2, 3])
        
        validator = ModelValidator()
        report = validator.validate_arima_assumptions(short_residuals)
        
        assert report['assumptions_met'] is False
        assert 'error' in report['tests']