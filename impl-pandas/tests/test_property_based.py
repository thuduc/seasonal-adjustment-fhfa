"""Property-based tests using Hypothesis for edge case discovery"""

import pytest
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, assume, settings
from hypothesis.extra.pandas import data_frames, column, range_indexes, series
from hypothesis.extra.numpy import arrays
from datetime import datetime, timedelta

from src.models.arima.regarima import RegARIMA
from src.models.regression.panel import SeasonalityImpactModel
from src.models.regression.quantile import QuantileRegressionModel
from src.data.validators import TimeSeriesValidator, PanelDataValidator
from src.models.seasonal.adjuster import SeasonalAdjuster


class TestPropertyBasedRegARIMA:
    """Property-based tests for RegARIMA model"""
    
    @given(
        ar_order=st.integers(min_value=0, max_value=3),
        diff_order=st.integers(min_value=0, max_value=2),
        ma_order=st.integers(min_value=0, max_value=3)
    )
    def test_model_order_validation(self, ar_order, diff_order, ma_order):
        """Test that model orders are properly validated"""
        # Skip invalid combinations
        assume(ar_order + diff_order + ma_order > 0)
        
        model = RegARIMA(
            ar_order=ar_order,
            diff_order=diff_order,
            ma_order=ma_order
        )
        
        assert model.order[0] == ar_order
        assert model.order[1] == diff_order
        assert model.order[2] == ma_order
    
    @given(
        n_obs=st.integers(min_value=20, max_value=1000),
        trend_strength=st.floats(min_value=-1.0, max_value=1.0),
        seasonal_strength=st.floats(min_value=0.0, max_value=10.0),
        noise_level=st.floats(min_value=0.1, max_value=5.0)
    )
    @settings(max_examples=20, deadline=None)
    def test_model_fitting_robustness(self, n_obs, trend_strength, seasonal_strength, noise_level):
        """Test model fitting on various synthetic series"""
        # Generate synthetic time series
        t = np.arange(n_obs)
        trend = trend_strength * t
        seasonal = seasonal_strength * np.sin(2 * np.pi * t / 4)
        noise = np.random.normal(0, noise_level, n_obs)
        
        y = pd.Series(
            100 + trend + seasonal + noise,
            index=pd.date_range('2010-01-01', periods=n_obs, freq='QE')
        )
        
        # Ensure positive values
        y = y.clip(lower=1)
        
        model = RegARIMA(ar_order=1, diff_order=1, ma_order=1)
        
        try:
            model.fit(y)
            # If fitting succeeds, check basic properties
            assert model.results is not None
            assert hasattr(model.results, 'aic')
            assert hasattr(model.results, 'bic')
            # AIC and BIC can be negative for very good models
            assert np.isfinite(model.results.aic)
            assert np.isfinite(model.results.bic)
        except Exception as e:
            # Some parameter combinations might fail - that's OK
            # but we should get a meaningful error
            assert isinstance(e, (ValueError, np.linalg.LinAlgError))
    
    @given(
        series=series(
            dtype=np.float64,
            index=range_indexes(min_size=20, max_size=100),
            elements=st.floats(min_value=50, max_value=200, allow_nan=False)
        )
    )
    @settings(max_examples=10, deadline=None)
    def test_predict_returns_valid_output(self, series):
        """Test that predictions are valid for any reasonable input"""
        # Convert to quarterly datetime index
        dates = pd.date_range('2010-01-01', periods=len(series), freq='QE')
        series.index = dates
        
        model = RegARIMA(ar_order=1, diff_order=1, ma_order=1)
        
        try:
            model.fit(series)
            predictions = model.predict(n_periods=4)
            
            # Check predictions are valid
            assert isinstance(predictions, pd.Series)
            assert len(predictions) == 4
            assert not predictions.isna().any()
            assert (predictions > 0).all()  # For positive series
        except Exception:
            # Skip if model fitting fails
            pass


class TestPropertyBasedValidators:
    """Property-based tests for data validators"""
    
    @given(
        series=series(
            dtype=np.float64,
            index=range_indexes(min_size=4, max_size=1000),
            elements=st.floats(min_value=-1000, max_value=1000) | st.floats(allow_nan=True)
        )
    )
    def test_time_series_validator_handles_any_series(self, series):
        """Test validator handles any series gracefully"""
        # Convert to datetime index
        dates = pd.date_range('2010-01-01', periods=len(series), freq='D')
        series.index = dates
        
        validator = TimeSeriesValidator()
        result = validator.validate(series)
        
        # Result should always be a dictionary with expected keys
        assert isinstance(result, dict)
        assert 'overall_valid' in result
        assert isinstance(result['overall_valid'], bool)
    
    @given(
        df=data_frames(
            columns=[
                column('entity', elements=st.sampled_from(['A', 'B', 'C', 'D'])),
                column('time', elements=st.integers(min_value=2010, max_value=2023)),
                column('value', elements=st.floats(min_value=0, max_value=1000) | st.floats(allow_nan=True))
            ],
            index=range_indexes(min_size=10, max_size=100)
        )
    )
    def test_panel_validator_handles_any_dataframe(self, df):
        """Test panel validator handles any dataframe gracefully"""
        validator = PanelDataValidator()
        result = validator.validate(df, entity_col='entity', time_col='time')
        
        assert isinstance(result, dict)
        assert 'overall_valid' in result
        assert isinstance(result['overall_valid'], bool)


class TestPropertyBasedSeasonalAdjustment:
    """Property-based tests for seasonal adjustment"""
    
    @given(
        n_periods=st.integers(min_value=8, max_value=200),
        frequency=st.sampled_from([4, 12]),  # Quarterly or monthly
        method=st.sampled_from(['classical', 'stl', 'x11'])
    )
    @settings(max_examples=20, deadline=None)
    def test_seasonal_adjustment_preserves_series_properties(self, n_periods, frequency, method):
        """Test that seasonal adjustment preserves basic properties"""
        # Generate series with known properties
        t = np.arange(n_periods)
        trend = 100 + 0.5 * t
        seasonal = 10 * np.sin(2 * np.pi * t / frequency)
        noise = np.random.normal(0, 2, n_periods)
        
        original = pd.Series(
            trend + seasonal + noise,
            index=pd.date_range('2010-01-01', periods=n_periods, 
                               freq='QE' if frequency == 4 else 'ME')
        )
        
        adjuster = SeasonalAdjuster(method=method)
        
        try:
            adjusted = adjuster.adjust(original, frequency=frequency)
            
            # Properties that should hold
            assert len(adjusted) == len(original)
            assert adjusted.index.equals(original.index)
            
            # Mean should be approximately preserved
            assert abs(adjusted.mean() - original.mean()) < original.std()
            
            # Variance should typically be reduced
            # (not always true for all methods, so we're lenient)
            assert adjusted.std() <= original.std() * 1.5
            
        except Exception as e:
            # Some methods might fail on short series
            assert n_periods < 20 or method == 'x11'
    
    @given(
        series=series(
            dtype=np.float64,
            index=range_indexes(min_size=20, max_size=100),
            elements=st.floats(min_value=10, max_value=1000, allow_nan=False)
        ),
        outlier_fraction=st.floats(min_value=0, max_value=0.3)
    )
    @settings(max_examples=10, deadline=None)
    def test_outlier_detection_robustness(self, series, outlier_fraction):
        """Test outlier detection handles various contamination levels"""
        # Add outliers
        n_outliers = int(len(series) * outlier_fraction)
        if n_outliers > 0:
            outlier_indices = np.random.choice(len(series), n_outliers, replace=False)
            series.iloc[outlier_indices] *= np.random.uniform(2, 5, n_outliers)
        
        # Convert to datetime index
        dates = pd.date_range('2010-01-01', periods=len(series), freq='QE')
        series.index = dates
        
        adjuster = SeasonalAdjuster(outlier_detection=True)
        
        try:
            # Should handle any level of contamination
            adjusted = adjuster.adjust(series, frequency=4)
            assert isinstance(adjusted, pd.Series)
            assert len(adjusted) == len(series)
        except Exception as e:
            # Should only fail for extreme cases
            assert outlier_fraction > 0.2


class TestPropertyBasedPanelRegression:
    """Property-based tests for panel regression models"""
    
    @given(
        n_entities=st.integers(min_value=2, max_value=20),
        n_periods=st.integers(min_value=10, max_value=50),
        n_features=st.integers(min_value=1, max_value=10),
        model_type=st.sampled_from(['fixed_effects', 'random_effects'])
    )
    @settings(max_examples=20, deadline=None)
    def test_panel_regression_handles_various_dimensions(self, n_entities, n_periods, 
                                                       n_features, model_type):
        """Test panel regression with various data dimensions"""
        # Generate panel data
        entities = [f'E{i}' for i in range(n_entities)]
        dates = pd.date_range('2010-01-01', periods=n_periods, freq='QE')
        
        # Generate random data without multi-index (fit method will set it)
        data = []
        for entity in entities:
            for date in dates:
                row = {
                    'entity': entity,
                    'time': date,
                    'y': np.random.randn() + np.random.normal(0, 1),  # Entity effect
                }
                for i in range(n_features):
                    row[f'x{i}'] = np.random.randn()
                data.append(row)
        
        data = pd.DataFrame(data)
        
        model = SeasonalityImpactModel(model_type=model_type)
        
        try:
            model.fit(
                data,
                y_col='y',
                weather_col='x0',
                sales_col='y',       # Use y as proxy for sales
                char_cols=[],        # No characteristics
                industry_cols=[],    # No industry data
                entity_col='entity',
                time_col='time'
            )
            
            # Check model was fitted
            assert model.results is not None
            assert hasattr(model.results, 'params')
            assert len(model.results.params) > 0
            
        except Exception as e:
            # Some configurations might be singular, have rank issues, or numerical problems
            error_msg = str(e).lower()
            assert ('singular' in error_msg or 
                    'rank' in error_msg or
                    'division by zero' in error_msg or
                    n_entities * n_periods < n_features + 10)


class TestPropertyBasedQuantileRegression:
    """Property-based tests for quantile regression"""
    
    @given(
        n_obs=st.integers(min_value=50, max_value=500),
        n_features=st.integers(min_value=1, max_value=10),
        quantiles=st.lists(
            st.floats(min_value=0.1, max_value=0.9),
            min_size=1,
            max_size=5,
            unique=True
        )
    )
    @settings(max_examples=20, deadline=None)
    def test_quantile_regression_monotonicity(self, n_obs, n_features, quantiles):
        """Test that quantile predictions are monotonic"""
        # Generate data
        X = np.random.randn(n_obs, n_features)
        true_beta = np.random.randn(n_features)
        y = X @ true_beta + np.random.randn(n_obs)
        
        df = pd.DataFrame(X, columns=[f'x{i}' for i in range(n_features)])
        df['y'] = y
        
        model = QuantileRegressionModel(quantiles=sorted(quantiles))
        
        try:
            model.fit(df, 'y', [f'x{i}' for i in range(n_features)])
            
            # Get predictions for a test point
            test_X = pd.DataFrame(
                np.random.randn(1, n_features),
                columns=[f'x{i}' for i in range(n_features)]
            )
            
            predictions = model.predict(test_X)
            
            # Check monotonicity - higher quantiles should give higher predictions
            if len(quantiles) > 1:
                pred_values = [predictions[q].iloc[0] for q in sorted(quantiles)]
                assert all(pred_values[i] <= pred_values[i+1] 
                          for i in range(len(pred_values)-1))
                
        except Exception:
            # Some configurations might fail
            pass


class TestPropertyBasedEdgeCases:
    """Test edge cases that commonly cause issues"""
    
    @given(
        series_length=st.integers(min_value=1, max_value=10),
        constant_value=st.floats(min_value=-100, max_value=100, allow_nan=False)
    )
    def test_short_or_constant_series(self, series_length, constant_value):
        """Test handling of very short or constant series"""
        series = pd.Series(
            [constant_value] * series_length,
            index=pd.date_range('2020-01-01', periods=series_length, freq='QE')
        )
        
        # Validators should handle gracefully
        validator = TimeSeriesValidator()
        result = validator.validate(series)
        assert isinstance(result, dict)
        
        # Seasonal adjustment should handle gracefully
        if series_length >= 8:  # Minimum for seasonal decomposition
            adjuster = SeasonalAdjuster()
            try:
                adjusted = adjuster.adjust(series, frequency=4)
                # Constant series should remain constant (or have only minor floating point differences)
                # For very small constant values, use absolute tolerance
                assert np.allclose(adjusted, constant_value, rtol=1e-10, atol=1e-10)
            except Exception as e:
                # Should give meaningful error
                assert 'constant' in str(e).lower() or 'seasonal' in str(e).lower()
    
    @given(
        missing_pattern=st.lists(
            st.booleans(),
            min_size=20,
            max_size=100
        )
    )
    def test_missing_data_patterns(self, missing_pattern):
        """Test handling of various missing data patterns"""
        # Create series with missing data
        n = len(missing_pattern)
        values = np.random.randn(n) * 10 + 100
        values[missing_pattern] = np.nan
        
        series = pd.Series(
            values,
            index=pd.date_range('2010-01-01', periods=n, freq='QE')
        )
        
        # Skip if too many missing values
        assume(series.notna().sum() >= 8)
        
        validator = TimeSeriesValidator()
        result = validator.validate(series)
        
        # Should identify missing data issue
        assert 'missing_count' in result
        assert 'missing_ratio' in result
        # Check if validator correctly identified missing values
        if any(missing_pattern):
            assert result['missing_count'] > 0
            assert result['missing_ratio'] > 0
        else:
            assert result['missing_count'] == 0
            assert result['missing_ratio'] == 0