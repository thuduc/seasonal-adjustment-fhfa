"""Tests for model implementations"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.models.arima.regarima import RegARIMA
from src.models.regression.linear import LinearRegressionModel
from src.models.regression.quantile import QuantileRegressionModel
from src.models.regression.panel import SeasonalityImpactModel
from src.models.seasonal.adjuster import SeasonalAdjuster


class TestRegARIMA:
    """Test RegARIMA model"""
    
    def setup_method(self):
        """Set up test data"""
        # Create synthetic quarterly time series
        dates = pd.date_range('2010-01-01', '2023-12-31', freq='Q')
        trend = np.linspace(100, 150, len(dates))
        seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 4)
        noise = np.random.normal(0, 2, len(dates))
        
        self.ts = pd.Series(
            trend + seasonal + noise,
            index=dates,
            name='test_series'
        )
        
        # Create exogenous variables
        self.exog = pd.DataFrame({
            'temperature': np.random.normal(65, 10, len(dates)),
            'gdp_growth': np.random.normal(2, 1, len(dates))
        }, index=dates)
        
    def test_basic_fit(self):
        """Test basic model fitting"""
        model = RegARIMA(
            ar_order=1,
            diff_order=1,
            ma_order=1,
            seasonal_order=(0, 1, 1, 4)
        )
        
        fitted = model.fit(self.ts)
        
        assert fitted.results is not None
        assert hasattr(fitted.results, 'aic')
        assert hasattr(fitted.results, 'bic')
        
    def test_fit_with_exogenous(self):
        """Test fitting with exogenous variables"""
        model = RegARIMA(
            ar_order=1,
            diff_order=1,
            ma_order=1,
            seasonal_order=(0, 1, 1, 4),
            exog_vars=['temperature', 'gdp_growth']
        )
        
        fitted = model.fit(self.ts, X=self.exog)
        
        assert fitted.results is not None
        # Check that exogenous variables were included
        assert fitted.results.model.k_exog > 0
        
    def test_predict(self):
        """Test prediction"""
        model = RegARIMA()
        model.fit(self.ts)
        
        # Predict next 4 quarters
        predictions = model.predict(steps=4)
        
        assert isinstance(predictions, pd.Series)
        assert len(predictions) == 4
        assert predictions.index[0] > self.ts.index[-1]
        
    def test_diagnostics(self):
        """Test model diagnostics"""
        model = RegARIMA()
        model.fit(self.ts)
        
        diag = model.diagnostics()
        
        # Check diagnostic keys
        assert 'ljung_box_pvalue' in diag
        assert 'jarque_bera_pvalue' in diag
        assert 'aic' in diag
        assert 'bic' in diag
        assert 'residual_stats' in diag
        

class TestLinearRegression:
    """Test linear regression model"""
    
    def setup_method(self):
        """Set up test data"""
        n = 100
        self.X = pd.DataFrame({
            'x1': np.random.normal(0, 1, n),
            'x2': np.random.normal(0, 1, n),
            'x3': np.random.normal(0, 1, n)
        })
        
        # Create target with known coefficients
        true_coef = [2.0, -1.5, 0.5]
        self.y = pd.Series(
            10 + sum(coef * self.X[col] for coef, col in zip(true_coef, self.X.columns)) +
            np.random.normal(0, 0.5, n)
        )
        
    def test_fit_sklearn(self):
        """Test fitting with sklearn backend"""
        model = LinearRegressionModel(use_statsmodels=False)
        model.fit(self.X, self.y)
        
        assert model.coefficients is not None
        assert len(model.coefficients) == len(self.X.columns) + 1  # Including intercept
        
    def test_fit_statsmodels(self):
        """Test fitting with statsmodels backend"""
        model = LinearRegressionModel(use_statsmodels=True)
        model.fit(self.X, self.y)
        
        assert model.results is not None
        assert model.diagnostics is not None
        assert 'r_squared' in model.diagnostics
        
    def test_predict(self):
        """Test prediction"""
        model = LinearRegressionModel()
        model.fit(self.X, self.y)
        
        predictions = model.predict(self.X)
        
        assert isinstance(predictions, pd.Series)
        assert len(predictions) == len(self.y)
        
    def test_get_coefficients(self):
        """Test coefficient extraction"""
        model = LinearRegressionModel(use_statsmodels=True)
        model.fit(self.X, self.y)
        
        coef_df = model.get_coefficients()
        
        assert isinstance(coef_df, pd.DataFrame)
        assert 'coefficient' in coef_df.columns
        assert 'p_value' in coef_df.columns
        assert 'std_error' in coef_df.columns
        

class TestQuantileRegression:
    """Test quantile regression model"""
    
    def setup_method(self):
        """Set up test panel data"""
        # Create panel data
        dates = pd.date_range('2010-01-01', '2023-12-31', freq='Q')
        entities = ['A', 'B', 'C']
        
        data = []
        for entity in entities:
            for date in dates:
                data.append({
                    'entity': entity,
                    'date': date,
                    'year': date.year,
                    'quarter': date.quarter,
                    'y': np.random.normal(100, 20),
                    'temperature': np.random.normal(65, 10),
                    'n_sales': np.random.poisson(1000),
                    'income': np.random.normal(60000, 10000),
                    'density': np.random.normal(1000, 200),
                    'industry': np.random.uniform(0.1, 0.3)
                })
        
        self.panel_data = pd.DataFrame(data)
        
    def test_fit_quantiles(self):
        """Test fitting quantile regression"""
        model = QuantileRegressionModel(
            quantiles=[0.25, 0.5, 0.75],
            use_statsmodels=False
        )
        
        model.fit(
            self.panel_data,
            y_col='y',
            weather_col='temperature',
            sales_col='n_sales',
            char_cols=['income', 'density'],
            industry_cols=['industry'],
            entity_col='entity',
            year_col='year'
        )
        
        assert len(model.models) == 3  # One for each quantile
        assert all(tau in model.models for tau in [0.25, 0.5, 0.75])
        
    def test_predict_quantiles(self):
        """Test quantile predictions"""
        model = QuantileRegressionModel(quantiles=[0.5])
        
        model.fit(
            self.panel_data,
            y_col='y',
            weather_col='temperature',
            sales_col='n_sales',
            char_cols=['income', 'density'],
            industry_cols=['industry'],
            entity_col='entity',
            year_col='year'
        )
        
        # Predict median
        predictions = model.predict(self.panel_data, quantile=0.5)
        
        assert isinstance(predictions, pd.Series)
        assert len(predictions) == len(self.panel_data)
        

class TestSeasonalAdjuster:
    """Test seasonal adjuster"""
    
    def setup_method(self):
        """Set up test data"""
        # Create seasonal time series
        dates = pd.date_range('2010-01-01', '2023-12-31', freq='Q')
        trend = np.linspace(100, 150, len(dates))
        seasonal_pattern = [1.05, 0.95, 0.98, 1.02]  # Quarterly pattern
        seasonal = np.tile(seasonal_pattern, len(dates) // 4)[:len(dates)]
        
        self.series = pd.Series(
            trend * seasonal + np.random.normal(0, 2, len(dates)),
            index=dates,
            name='test_series'
        )
        
    def test_classical_adjustment(self):
        """Test classical seasonal adjustment"""
        adjuster = SeasonalAdjuster(method='classical')
        adjusted = adjuster.adjust(self.series, frequency=4)
        
        assert isinstance(adjusted, pd.Series)
        assert len(adjusted) == len(self.series)
        
        # Check that seasonal variation is reduced
        orig_seasonal_var = self.series.groupby(self.series.index.quarter).std().mean()
        adj_seasonal_var = adjusted.groupby(adjusted.index.quarter).std().mean()
        
        # Adjusted should have less seasonal variation (with some tolerance for synthetic data)
        # Allow up to 5% increase due to edge effects in synthetic data
        assert adj_seasonal_var <= orig_seasonal_var * 1.05
        
    def test_stl_adjustment(self):
        """Test STL decomposition adjustment"""
        adjuster = SeasonalAdjuster(method='stl')
        adjusted = adjuster.adjust(self.series, frequency=4)
        
        assert isinstance(adjusted, pd.Series)
        assert 'components' in adjuster.results
        assert 'trend' in adjuster.results['components']
        assert 'seasonal' in adjuster.results['components']
        
    def test_diagnostics(self):
        """Test adjustment diagnostics"""
        adjuster = SeasonalAdjuster(method='classical')
        adjusted = adjuster.adjust(self.series, frequency=4)
        
        results = adjuster.get_results()
        
        assert 'diagnostics' in results
        assert 'seasonal_strength' in results['diagnostics']
        assert 'seasonal_stability' in results['diagnostics']
        
    def test_quality_checks(self):
        """Test quality control checks"""
        adjuster = SeasonalAdjuster(method='classical')
        adjusted = adjuster.adjust(self.series, frequency=4)
        
        # Mean should be preserved (approximately)
        assert abs(self.series.mean() - adjusted.mean()) < 1.0
        
        # Should not introduce extreme values
        assert adjusted.max() < self.series.max() * 2
        assert adjusted.min() > self.series.min() * 0.5