import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import statsmodels.api as sm

from src.analysis.causation_analysis import CausationAnalysis


class TestCausationAnalysis:
    """Unit tests for CausationAnalysis class."""
    
    @pytest.fixture
    def analyzer(self):
        """Create CausationAnalysis instance."""
        return CausationAnalysis()
    
    @pytest.fixture
    def sample_master_data(self):
        """Create sample master dataset."""
        np.random.seed(42)
        dates = pd.date_range('2015-01-01', '2023-12-31', freq='Q')
        
        # Create data for multiple geographies
        data_list = []
        for geo in ['CA', 'NY', 'TX']:
            geo_data = pd.DataFrame({
                'geography_id': geo,
                'period': dates,
                'nsa_index': 100 + np.random.normal(0, 5, len(dates)),
                'sa_index': 100 + np.random.normal(0, 3, len(dates)),
                'temp_range': 20 + np.random.normal(0, 5, len(dates)),
                'avg_temp': 60 + np.random.normal(0, 10, len(dates)),
                'precipitation': 3 + np.random.normal(0, 1, len(dates)),
                'num_transactions': np.random.randint(1000, 5000, len(dates)),
                'pct_over_65': 15 + np.random.uniform(-3, 3),
                'pct_white': 65 + np.random.uniform(-10, 10),
                'pct_bachelors': 35 + np.random.uniform(-5, 5),
                'pct_single_family': 60 + np.random.uniform(-5, 5),
                'avg_income': 75000 + np.random.uniform(-10000, 20000),
                'healthcare_share': 15 + np.random.uniform(-2, 2),
                'manufacturing_share': 10 + np.random.uniform(-2, 2)
            })
            data_list.append(geo_data)
            
        return pd.concat(data_list, ignore_index=True)
    
    def test_init(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.feature_engineer is not None
        assert analyzer.scaler is not None
        assert isinstance(analyzer.models, dict)
        assert isinstance(analyzer.results, dict)
        
    def test_prepare_regression_data(self, analyzer, sample_master_data):
        """Test regression data preparation."""
        regression_df = analyzer.prepare_regression_data(sample_master_data)
        
        assert isinstance(regression_df, pd.DataFrame)
        assert 'target' in regression_df.columns
        assert len(regression_df) > 0
        
    def test_fit_linear_model(self, analyzer, sample_master_data):
        """Test linear model fitting."""
        # Create sample data with both nsa and sa indices
        sample_data = sample_master_data.copy()
        # Add SA index (slightly modified NSA)
        sample_data['sa_index'] = sample_data['nsa_index'] * 0.98
        
        # Prepare data
        regression_df = analyzer.prepare_regression_data(sample_data)
        
        # Fit model
        results = analyzer.fit_linear_model(regression_df)
        
        assert isinstance(results, dict)
        assert 'model' in results
        assert 'coefficients' in results
        assert 'p_values' in results
        assert 'r_squared' in results
        assert 'significant_features' in results
        assert 'feature_importance' in results
        
        # Check model properties
        assert 0 <= results['r_squared'] <= 1
        assert results['n_observations'] == len(regression_df)
        assert isinstance(results['significant_features'], list)
        
    def test_fit_linear_model_no_features(self, analyzer):
        """Test linear model with no available features."""
        df = pd.DataFrame({
            'target': [1, 2, 3, 4, 5]
        })
        
        with pytest.raises(ValueError, match="No features available"):
            analyzer.fit_linear_model(df)
            
    def test_fit_quantile_regression(self, analyzer, sample_master_data):
        """Test quantile regression fitting."""
        # Create sample data with both nsa and sa indices
        sample_data = sample_master_data.copy()
        sample_data['sa_index'] = sample_data['nsa_index'] * 0.98
        
        # Prepare data
        regression_df = analyzer.prepare_regression_data(sample_data)
        
        # Fit quantile models
        quantiles = [0.25, 0.5, 0.75]
        results = analyzer.fit_quantile_regression(regression_df, quantiles=quantiles)
        
        assert isinstance(results, dict)
        assert 'quantile_models' in results
        assert 'coefficient_matrix' in results
        assert 'varying_features' in results
        
        # Check each quantile
        for q in quantiles:
            assert q in results['quantile_models']
            q_results = results['quantile_models'][q]
            assert 'model' in q_results
            assert 'coefficients' in q_results
            assert 'pseudo_r_squared' in q_results
            
    def test_analyze_geographic_patterns(self, analyzer, sample_master_data):
        """Test geographic pattern analysis."""
        # Create sample data with both nsa and sa indices
        sample_data = sample_master_data.copy()
        sample_data['sa_index'] = sample_data['nsa_index'] * 0.98
        
        # First fit a linear model
        regression_df = analyzer.prepare_regression_data(sample_data)
        linear_results = analyzer.fit_linear_model(regression_df)
        
        # Analyze patterns
        geo_patterns = analyzer.analyze_geographic_patterns(
            {'linear': linear_results},
            sample_data
        )
        
        assert isinstance(geo_patterns, pd.DataFrame)
        assert 'geography_id' in geo_patterns.columns
        assert 'avg_seasonality' in geo_patterns.columns
        assert 'seasonality_rank' in geo_patterns.columns
        assert 'high_seasonality' in geo_patterns.columns
        
        # Check rankings
        assert geo_patterns['seasonality_rank'].min() >= 1
        assert geo_patterns['seasonality_rank'].max() <= len(geo_patterns)
        
    def test_analyze_geographic_patterns_no_linear_model(self, analyzer, sample_master_data):
        """Test geographic analysis without linear model."""
        with pytest.raises(ValueError, match="Linear model results required"):
            analyzer.analyze_geographic_patterns({}, sample_master_data)
            
    def test_calculate_weather_impact(self, analyzer):
        """Test weather impact calculation."""
        # Create mock model results with both required keys
        model_results = {
            'linear': {
                'coefficients': {
                    'temp_range': 0.5,
                    'temp_range_squared': 0.2,
                    'avg_temp': 0.3,
                    'precipitation': 0.1,
                    'quarter_2_x_temp_range': 0.15
                },
                'feature_importance': [
                    ('temp_range', 0.5),
                    ('avg_temp', 0.3),
                    ('temp_range_squared', 0.2),
                    ('quarter_2_x_temp_range', 0.15),
                    ('precipitation', 0.1),
                    ('other_feature', 0.05)  # Add non-weather feature
                ]
            }
        }
        
        weather_impact = analyzer.calculate_weather_impact(model_results)
        
        assert isinstance(weather_impact, dict)
        assert 'total_impact' in weather_impact
        assert 'by_feature' in weather_impact
        assert 'by_type' in weather_impact
        assert 'top_weather_features' in weather_impact
        
        # Check calculations
        assert weather_impact['total_impact'] >= 0  # Can be 0 if no weather features
        if weather_impact['total_impact'] > 0:
            assert weather_impact['by_type']['temperature'] >= 0
        
    def test_run_full_analysis(self, analyzer, sample_master_data):
        """Test complete causation analysis pipeline."""
        # Create sample data with both nsa and sa indices
        sample_data = sample_master_data.copy()
        sample_data['sa_index'] = sample_data['nsa_index'] * 0.98
        
        results = analyzer.run_full_analysis(sample_data)
        
        assert isinstance(results, dict)
        assert 'linear_model' in results
        assert 'quantile_models' in results
        assert 'geographic_patterns' in results
        assert 'weather_impact' in results
        assert 'feature_names' in results
        assert 'n_observations' in results
        assert 'target_summary' in results
        
        # Check target summary
        target_summary = results['target_summary']
        assert 'mean' in target_summary
        assert 'std' in target_summary
        assert 'min' in target_summary
        assert 'max' in target_summary
        
    def test_fit_linear_model_with_nan_values(self, analyzer):
        """Test linear model fitting with NaN values."""
        # Create data with NaN values
        dates = pd.date_range('2020-01-01', '2021-12-31', freq='QE')
        df = pd.DataFrame({
            'target': [1, 2, np.nan, 4, 5, 6, 7, 8],
            'feature1': [1, np.nan, 3, 4, 5, 6, 7, 8],
            'feature2': [8, 7, 6, 5, 4, 3, 2, 1]
        }, index=dates)
        
        # Mock feature engineer
        analyzer.feature_engineer.get_feature_names = Mock(return_value=['feature1', 'feature2'])
        
        results = analyzer.fit_linear_model(df)
        
        # Should handle NaN values
        assert results['n_observations'] < len(df)  # Some rows dropped
        
    def test_quantile_regression_edge_cases(self, analyzer, sample_master_data):
        """Test quantile regression with edge case quantiles."""
        # Create sample data with both nsa and sa indices
        sample_data = sample_master_data.copy()
        sample_data['sa_index'] = sample_data['nsa_index'] * 0.98
        
        regression_df = analyzer.prepare_regression_data(sample_data)
        
        # Test with extreme quantiles
        results = analyzer.fit_quantile_regression(
            regression_df, 
            quantiles=[0.01, 0.99]
        )
        
        assert 0.01 in results['quantile_models']
        assert 0.99 in results['quantile_models']