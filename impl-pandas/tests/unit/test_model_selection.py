import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import warnings

from src.models.model_selection import ModelSelector

warnings.filterwarnings('ignore')


class TestModelSelector:
    """Unit tests for ModelSelector class."""
    
    @pytest.fixture
    def selector(self):
        """Create ModelSelector instance."""
        return ModelSelector(max_p=2, max_q=2, max_P=1, max_Q=1)
    
    @pytest.fixture
    def sample_series(self):
        """Create sample time series data."""
        dates = pd.date_range('2015-01-01', '2023-12-31', freq='Q')
        
        # Create non-stationary series with seasonal pattern
        trend = np.linspace(100, 150, len(dates))
        seasonal = 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 4)
        noise = np.random.normal(0, 2, len(dates))
        
        values = trend + seasonal + noise
        return pd.Series(values, index=dates)
    
    def test_init(self):
        """Test selector initialization."""
        selector = ModelSelector()
        assert selector.max_p == 4
        assert selector.max_d == 2
        assert selector.max_q == 4
        assert selector.seasonal_period == 4
        
        # Test custom parameters
        selector = ModelSelector(max_p=3, max_P=1, seasonal_period=12)
        assert selector.max_p == 3
        assert selector.max_P == 1
        assert selector.seasonal_period == 12
        
    @patch('statsmodels.tsa.stattools.adfuller')
    @patch('statsmodels.tsa.stattools.kpss')
    def test_determine_differencing_order(self, mock_kpss, mock_adf, selector, sample_series):
        """Test differencing order determination."""
        # Mock test results - first call non-stationary, second stationary
        mock_adf.side_effect = [
            (-1.5, 0.5, 0, 100, {}, {}),  # Non-stationary
            (-4.0, 0.01, 0, 100, {}, {})  # Stationary
        ]
        mock_kpss.side_effect = [
            (0.8, 0.01, 0, {}),  # Reject stationarity
            (0.3, 0.1, 0, {})    # Accept stationarity
        ]
        
        d, D = selector.determine_differencing_order(sample_series)
        
        assert isinstance(d, int)
        assert isinstance(D, int)
        assert 0 <= d <= selector.max_d
        assert 0 <= D <= selector.max_D
        
    def test_grid_search_arima_simple(self, selector):
        """Test ARIMA grid search with simple data."""
        # Create simple AR(1) series
        np.random.seed(42)
        n = 100
        series_values = [100]
        
        for i in range(1, n):
            series_values.append(0.8 * series_values[-1] + 20 + np.random.normal(0, 1))
            
        dates = pd.date_range('2015-01-01', periods=n, freq='Q')
        series = pd.Series(series_values, index=dates)
        
        # Run grid search with fixed differencing
        results = selector.grid_search_arima(series, d=0, D=0)
        
        assert isinstance(results, dict)
        assert 'best_model' in results
        assert 'top_models' in results
        assert 'all_results' in results
        
        # Check best model structure
        best = results['best_model']
        assert 'order' in best
        assert 'seasonal_order' in best
        assert 'aic' in best
        assert 'model' in best
        
    def test_validate_model(self, selector):
        """Test model validation."""
        # Create simple series
        np.random.seed(42)
        dates = pd.date_range('2015-01-01', '2023-12-31', freq='Q')
        series = pd.Series(
            100 + np.cumsum(np.random.normal(0, 1, len(dates))),
            index=dates
        )
        
        # Fit a simple model
        from statsmodels.tsa.arima.model import ARIMA
        model = ARIMA(series, order=(1, 1, 1), seasonal_order=(0, 0, 0, 4))
        fitted_model = model.fit()
        
        # Validate
        validation_results = selector.validate_model(fitted_model, series, test_size=4)
        
        assert isinstance(validation_results, dict)
        assert 'mae' in validation_results
        assert 'rmse' in validation_results
        assert 'mape' in validation_results
        assert 'direction_accuracy' in validation_results
        assert 'forecasts' in validation_results
        assert 'actuals' in validation_results
        
        # Check metric ranges
        assert validation_results['mae'] >= 0
        assert validation_results['rmse'] >= 0
        assert 0 <= validation_results['direction_accuracy'] <= 1
        
    def test_cross_validate_model(self, selector):
        """Test cross-validation."""
        # Create longer series for CV
        np.random.seed(42)
        dates = pd.date_range('2010-01-01', '2023-12-31', freq='Q')
        series = pd.Series(
            100 + np.cumsum(np.random.normal(0, 1, len(dates))),
            index=dates
        )
        
        model_spec = {
            'order': (1, 1, 1),
            'seasonal_order': (0, 0, 0, 4)
        }
        
        cv_results = selector.cross_validate_model(
            series, model_spec, n_splits=3, test_size=4
        )
        
        assert isinstance(cv_results, dict)
        assert 'mean_mae' in cv_results
        assert 'mean_rmse' in cv_results
        assert 'mean_mape' in cv_results
        assert 'std_rmse' in cv_results
        assert 'fold_results' in cv_results
        
        # Check that we got results from multiple folds
        assert len(cv_results['fold_results']) > 0
        assert cv_results['mean_rmse'] > 0
        
    def test_select_best_model(self, selector):
        """Test complete model selection process."""
        # Create series with clear ARIMA structure
        np.random.seed(42)
        dates = pd.date_range('2010-01-01', '2023-12-31', freq='Q')
        
        # AR(1) with drift
        series_values = [100]
        for i in range(1, len(dates)):
            series_values.append(
                0.1 + 0.9 * series_values[-1] + np.random.normal(0, 1)
            )
            
        series = pd.Series(series_values, index=dates)
        
        # Run selection
        best_model_info = selector.select_best_model(
            series, criterion='aic', validate=True
        )
        
        assert isinstance(best_model_info, dict)
        assert 'order' in best_model_info
        assert 'seasonal_order' in best_model_info
        assert 'aic' in best_model_info
        assert 'model' in best_model_info
        assert 'validation' in best_model_info
        assert 'cross_validation' in best_model_info
        
    def test_select_best_model_bic_criterion(self, selector):
        """Test model selection with BIC criterion."""
        np.random.seed(42)
        dates = pd.date_range('2015-01-01', '2023-12-31', freq='Q')
        series = pd.Series(
            100 + np.cumsum(np.random.normal(0, 1, len(dates))),
            index=dates
        )
        
        best_model_info = selector.select_best_model(
            series, criterion='bic', validate=False
        )
        
        assert isinstance(best_model_info, dict)
        assert 'bic' in best_model_info
        
    def test_invalid_criterion(self, selector):
        """Test with invalid selection criterion."""
        # Create a valid time series
        dates = pd.date_range('2020-01-01', periods=40, freq='QE')
        series = pd.Series(100 + np.cumsum(np.random.normal(0, 1, 40)), index=dates)
        
        with pytest.raises(ValueError, match="Unknown criterion"):
            selector.select_best_model(series, criterion='invalid')
            
    def test_insufficient_data(self, selector):
        """Test with insufficient data for model fitting."""
        # Very short series
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='Q')
        series = pd.Series([100, 101, 102, 103], index=dates)
        
        with pytest.raises(ValueError):
            selector.grid_search_arima(series)
            
    def test_grid_search_no_valid_models(self, selector):
        """Test grid search when no models can be fitted."""
        # Create constant series
        series = pd.Series([100] * 10)
        
        # Should return default result for constant series
        result = selector.grid_search_arima(series, d=1, D=0)
        
        # Check that it returns the expected structure
        assert 'best_model' in result
        assert 'top_models' in result
        assert 'all_results' in result
        assert result['best_model']['aic'] == np.inf  # Indicates no valid model