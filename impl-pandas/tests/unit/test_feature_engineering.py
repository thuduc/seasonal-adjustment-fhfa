import pytest
import pandas as pd
import numpy as np

from src.analysis.feature_engineering import FeatureEngineer


class TestFeatureEngineer:
    """Unit tests for FeatureEngineer class."""
    
    @pytest.fixture
    def engineer(self):
        """Create FeatureEngineer instance."""
        return FeatureEngineer()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='Q')
        
        data = pd.DataFrame({
            'nsa_index': 100 + np.random.normal(0, 5, len(dates)),
            'sa_index': 100 + np.random.normal(0, 3, len(dates)),
            'temp_range': 20 + np.random.normal(0, 5, len(dates)),
            'avg_temp': 60 + np.random.normal(0, 10, len(dates)),
            'precipitation': 3 + np.random.normal(0, 1, len(dates)),
            'num_transactions': np.random.randint(1000, 5000, len(dates)),
            'pct_over_65': 15.0,
            'pct_white': 65.0,
            'pct_bachelors': 35.0,
            'pct_single_family': 60.0,
            'avg_income': 75000,
            'healthcare_share': 15.0,
            'manufacturing_share': 10.0,
            'retail_share': 12.0
        }, index=dates)
        
        return data
    
    def test_create_seasonal_dummies(self, engineer, sample_data):
        """Test seasonal dummy variable creation."""
        result = engineer.create_seasonal_dummies(sample_data.copy())
        
        assert 'quarter' in result.columns
        assert 'quarter_2' in result.columns
        assert 'quarter_3' in result.columns
        assert 'quarter_4' in result.columns
        
        # Check dummy encoding
        assert result['quarter_2'].sum() == (result['quarter'] == 2).sum()
        assert set(result['quarter_2'].unique()) == {0, 1}
        
        # Check feature names updated
        assert 'quarter_2' in engineer.feature_names
        assert 'quarter_3' in engineer.feature_names
        assert 'quarter_4' in engineer.feature_names
        
    def test_create_weather_features(self, engineer, sample_data):
        """Test weather feature creation."""
        # First create seasonal dummies
        data_with_dummies = engineer.create_seasonal_dummies(sample_data.copy())
        result = engineer.create_weather_features(data_with_dummies)
        
        # Check squared terms
        assert 'temp_range_squared' in result.columns
        assert 'avg_temp_squared' in result.columns
        assert 'precipitation_squared' in result.columns
        
        # Check interaction terms
        assert 'quarter_2_x_temp_range' in result.columns
        assert 'quarter_3_x_avg_temp' in result.columns
        
        # Check lagged features
        assert 'temp_range_lag1' in result.columns
        assert 'avg_temp_lag1' in result.columns
        
        # Verify calculations
        np.testing.assert_array_almost_equal(
            result['temp_range_squared'].values,
            result['temp_range'].values ** 2
        )
        
    def test_create_weather_features_no_data(self, engineer):
        """Test weather features when no weather data available."""
        data = pd.DataFrame({
            'value': [1, 2, 3, 4]
        }, index=pd.date_range('2020-01-01', periods=4, freq='Q'))
        
        result = engineer.create_weather_features(data)
        
        # Should return unchanged
        assert len(result.columns) == len(data.columns)
        
    def test_create_time_trends(self, engineer, sample_data):
        """Test time trend feature creation."""
        result = engineer.create_time_trends(sample_data.copy(), n_knots=3)
        
        assert 'time_index' in result.columns
        assert 'linear_trend' in result.columns
        assert 'quadratic_trend' in result.columns
        
        # Check spline features
        assert 'spline_knot_1' in result.columns
        assert 'spline_knot_2' in result.columns
        assert 'spline_knot_3' in result.columns
        
        # Verify trends
        assert result['linear_trend'].min() >= 0
        assert result['linear_trend'].max() <= 1
        assert result['quadratic_trend'].min() >= 0
        
    def test_calculate_seasonality_measure(self, engineer, sample_data):
        """Test seasonality measure calculation."""
        data_copy = sample_data.copy()
        seasonality = engineer.calculate_seasonality_measure(data_copy)
        
        assert isinstance(seasonality, pd.Series)
        assert len(seasonality) == len(sample_data)
        assert (seasonality >= 0).all()  # Absolute values
        
        # Check percentage calculation - it's added to the dataframe copy
        assert 'seasonality_pct' in data_copy.columns
        
    def test_calculate_seasonality_measure_missing_columns(self, engineer):
        """Test seasonality calculation with missing columns."""
        data = pd.DataFrame({'value': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="Both NSA and SA indices required"):
            engineer.calculate_seasonality_measure(data)
            
    def test_create_market_features(self, engineer, sample_data):
        """Test market-related feature creation."""
        result = engineer.create_market_features(sample_data.copy())
        
        assert 'log_transactions' in result.columns
        assert 'transactions_ma4' in result.columns
        assert 'transactions_yoy' in result.columns
        
        # Verify calculations
        expected_log = np.log1p(result['num_transactions'])
        np.testing.assert_array_almost_equal(
            result['log_transactions'].values,
            expected_log.values
        )
        
    def test_create_demographic_features(self, engineer, sample_data):
        """Test demographic feature processing."""
        result = engineer.create_demographic_features(sample_data.copy())
        
        assert 'log_avg_income' in result.columns
        assert 'elderly_single_family' in result.columns
        
        # Check income transformation
        expected_log_income = np.log(result['avg_income'])
        np.testing.assert_array_almost_equal(
            result['log_avg_income'].values,
            expected_log_income.values
        )
        
        # Check interaction
        expected_interaction = result['pct_over_65'] * result['pct_single_family'] / 100
        np.testing.assert_array_almost_equal(
            result['elderly_single_family'].values,
            expected_interaction.values
        )
        
    def test_create_industry_features(self, engineer, sample_data):
        """Test industry feature processing."""
        result = engineer.create_industry_features(sample_data.copy())
        
        assert 'dominant_industry' in result.columns
        assert 'industry_concentration' in result.columns
        
        # Check that industry shares are in feature names
        assert 'healthcare_share' in engineer.feature_names
        assert 'manufacturing_share' in engineer.feature_names
        
    def test_prepare_regression_data(self, engineer, sample_data):
        """Test complete regression data preparation."""
        # Add geography for testing
        sample_data['geography_id'] = 'CA'
        
        result = engineer.prepare_regression_data(sample_data.copy())
        
        assert 'target' in result.columns
        assert len(engineer.feature_names) > 0
        
        # Check all feature types were created
        feature_types = {
            'quarter_': any('quarter_' in f for f in engineer.feature_names),
            'squared': any('squared' in f for f in engineer.feature_names),
            'lag': any('lag' in f for f in engineer.feature_names),
            'trend': any('trend' in f for f in engineer.feature_names)
        }
        
        assert all(feature_types.values())
        
    def test_prepare_regression_data_percentage_target(self, engineer, sample_data):
        """Test regression data with percentage target."""
        result = engineer.prepare_regression_data(
            sample_data.copy(), 
            target_type='percentage'
        )
        
        assert 'target' in result.columns
        assert 'seasonality_pct' in result.columns
        assert result['target'].equals(result['seasonality_pct'])
        
    def test_prepare_regression_data_invalid_target(self, engineer, sample_data):
        """Test with invalid target type."""
        with pytest.raises(ValueError, match="Unknown target type"):
            engineer.prepare_regression_data(sample_data, target_type='invalid')
            
    def test_prepare_regression_data_multiple_geographies(self, engineer):
        """Test with multiple geographies."""
        dates = pd.date_range('2020-01-01', '2021-12-31', freq='Q')
        
        data = pd.DataFrame({
            'geography_id': ['CA'] * 4 + ['NY'] * 4,
            'nsa_index': 100 + np.random.normal(0, 5, 8),
            'sa_index': 100 + np.random.normal(0, 3, 8),
            'period': dates[:4].tolist() + dates[:4].tolist()
        })
        
        data.set_index('period', inplace=True)
        
        result = engineer.prepare_regression_data(data)
        
        # Should have geographic fixed effects
        geo_dummies = [f for f in result.columns if f.startswith('geo_')]
        assert len(geo_dummies) > 0
        
    def test_get_feature_names(self, engineer, sample_data):
        """Test feature name retrieval."""
        engineer.prepare_regression_data(sample_data.copy())
        
        feature_names = engineer.get_feature_names()
        
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
        assert all(isinstance(name, str) for name in feature_names)