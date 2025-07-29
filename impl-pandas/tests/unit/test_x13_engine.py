import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from src.models.x13_engine import X13ARIMAEngine


class TestX13ARIMAEngine:
    """Unit tests for X13ARIMAEngine class."""
    
    @pytest.fixture
    def engine(self):
        """Create X13ARIMAEngine instance."""
        return X13ARIMAEngine()
    
    @pytest.fixture
    def sample_series(self):
        """Create sample time series data."""
        dates = pd.date_range('2015-01-01', '2023-12-31', freq='QE')
        
        # Create series with trend and seasonal pattern
        trend = np.linspace(100, 120, len(dates))
        seasonal = 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 4)
        noise = np.random.normal(0, 1, len(dates))
        
        values = trend + seasonal + noise
        return pd.Series(values, index=dates)
    
    def test_init(self):
        """Test engine initialization."""
        engine = X13ARIMAEngine()
        assert engine.max_ar_order == 4
        assert engine.max_ma_order == 4
        assert engine.max_seasonal_order == 2
        assert engine.seasonal_period == 4
        
        # Test with custom config
        config = {'max_ar_order': 3, 'seasonal_period': 12}
        engine = X13ARIMAEngine(config)
        assert engine.max_ar_order == 3
        assert engine.seasonal_period == 12
        
    def test_pre_adjustment(self, engine, sample_series):
        """Test pre-adjustment for trading day effects."""
        adjusted = engine.pre_adjustment(sample_series)
        
        assert isinstance(adjusted, pd.Series)
        assert len(adjusted) == len(sample_series)
        assert adjusted.index.equals(sample_series.index)
        
        # Check that adjustment factors are reasonable
        adjustment_factors = adjusted / sample_series
        assert adjustment_factors.min() > 0.95
        assert adjustment_factors.max() < 1.05
        
    def test_detect_outliers(self, engine, sample_series):
        """Test outlier detection."""
        # Add artificial outliers - make them extremely obvious
        series_with_outliers = sample_series.copy()
        # Create very extreme outliers
        series_with_outliers.iloc[10] = 1000  # Much higher than typical ~110 values
        series_with_outliers.iloc[20] = -1000  # Much lower
        
        outlier_df = engine.detect_outliers(series_with_outliers, threshold=2.0)
        
        assert isinstance(outlier_df, pd.DataFrame)
        assert 'is_outlier' in outlier_df.columns
        assert 'adjusted_value' in outlier_df.columns
        # With such extreme values, we should detect outliers
        num_outliers = outlier_df['is_outlier'].sum()
        # Check if outliers were interpolated (adjusted values should be different)
        outlier_adjusted = (outlier_df['value'] != outlier_df['adjusted_value']).sum()
        assert num_outliers >= 1 or outlier_adjusted >= 1  # Should detect and adjust at least one outlier
        
    def test_fit_arima_model(self, engine, sample_series):
        """Test ARIMA model fitting."""
        order = (1, 1, 1)
        seasonal_order = (1, 1, 1, 4)
        
        model = engine.fit_arima_model(sample_series, order, seasonal_order)
        
        assert model is not None
        assert hasattr(model, 'params')
        assert hasattr(model, 'resid')
        assert hasattr(model, 'fittedvalues')
        
    def test_extract_seasonal_factors(self, engine, sample_series):
        """Test seasonal factor extraction."""
        # Test without model (uses decomposition)
        factors = engine.extract_seasonal_factors(sample_series)
        
        assert isinstance(factors, pd.Series)
        assert len(factors) == len(sample_series)
        assert factors.index.equals(sample_series.index)
        
        # Check that factors average to 1.0
        assert np.abs(factors.mean() - 1.0) < 0.01
        
        # Check seasonal pattern
        q1_factors = factors[factors.index.quarter == 1]
        q3_factors = factors[factors.index.quarter == 3]
        assert q1_factors.mean() != q3_factors.mean()  # Should be different
        
    def test_calculate_adjusted_series(self, engine, sample_series):
        """Test seasonal adjustment calculation."""
        # Create seasonal factors with the same length as sample_series
        n_periods = len(sample_series)
        pattern = [1.02, 0.98, 1.01, 0.99]
        seasonal_factors = pd.Series(
            [pattern[i % 4] for i in range(n_periods)],
            index=sample_series.index
        )
        
        adjusted = engine.calculate_adjusted_series(sample_series, seasonal_factors)
        
        assert isinstance(adjusted, pd.Series)
        assert len(adjusted) == len(sample_series)
        
        # Check that adjustment was applied (values should be different)
        assert not adjusted.equals(sample_series)
        
    def test_decompose_series(self, engine, sample_series):
        """Test complete series decomposition."""
        components = engine.decompose_series(sample_series)
        
        assert isinstance(components, dict)
        assert 'original' in components
        assert 'trend' in components
        assert 'seasonal' in components
        assert 'irregular' in components
        assert 'seasonally_adjusted' in components
        
        # Check component properties
        assert components['original'].equals(sample_series)
        assert len(components['trend']) == len(sample_series)
        assert len(components['seasonal']) == len(sample_series)
        
    @patch('statsmodels.stats.diagnostic.acorr_ljungbox')
    @patch('scipy.stats.jarque_bera')
    @patch('statsmodels.tsa.stattools.adfuller')
    def test_run_diagnostics(self, mock_adf, mock_jb, mock_lb, engine):
        """Test diagnostic tests."""
        # Mock test results
        mock_lb.return_value = pd.DataFrame({
            'lb_stat': [5.0] * 10,
            'lb_pvalue': [0.1] * 10
        })
        mock_jb.return_value = (2.0, 0.3)
        mock_adf.return_value = (-3.0, 0.03, 0, 100, {}, {})
        
        # Create mock model
        mock_model = Mock()
        mock_model.aic = 100.0
        mock_model.bic = 110.0
        mock_model.llf = -50.0
        
        # Create mock residuals
        residuals = pd.Series(np.random.normal(0, 1, 100))
        
        diagnostics = engine.run_diagnostics(mock_model, residuals)
        
        assert isinstance(diagnostics, dict)
        assert 'ljung_box' in diagnostics
        assert 'jarque_bera' in diagnostics
        assert 'adf_test' in diagnostics
        assert 'aic' in diagnostics
        assert 'residual_stats' in diagnostics
        
        # Check diagnostic values
        assert diagnostics['jarque_bera']['normal'] == True
        assert diagnostics['adf_test']['stationary'] == True
        
    def test_stability_test(self, engine, sample_series):
        """Test seasonal pattern stability test."""
        # Create series with stable seasonal pattern
        dates = pd.date_range('2015-01-01', '2023-12-31', freq='QE')
        stable_pattern = []
        
        for date in dates:
            quarter = date.quarter
            if quarter == 1:
                stable_pattern.append(105)
            elif quarter == 2:
                stable_pattern.append(100)
            elif quarter == 3:
                stable_pattern.append(95)
            else:
                stable_pattern.append(100)
                
        stable_series = pd.Series(stable_pattern, index=dates)
        
        stability_results = engine.stability_test(stable_series, window_size=20)
        
        assert isinstance(stability_results, dict)
        assert 'cv_by_quarter' in stability_results
        assert 'max_cv' in stability_results
        assert 'stable' in stability_results
        assert 'seasonal_patterns' in stability_results
        
        # Should be stable
        assert stability_results['stable'] == True
        assert stability_results['max_cv'] < 0.1
        
    def test_calculate_seasonal_factors_from_model(self, engine, sample_series):
        """Test seasonal factor calculation from ARIMA model."""
        # Fit a simple model first
        order = (1, 0, 1)
        seasonal_order = (1, 0, 1, 4)
        
        model = engine.fit_arima_model(sample_series, order, seasonal_order)
        factors = engine._calculate_seasonal_factors_from_model(sample_series, model)
        
        assert isinstance(factors, pd.Series)
        assert len(factors) == len(sample_series)
        
        # Check quarterly pattern exists
        q_factors = {}
        for q in range(1, 5):
            q_factors[q] = factors[factors.index.quarter == q].mean()
            
        # Should have some variation across quarters
        factor_std = np.std(list(q_factors.values()))
        assert factor_std > 0