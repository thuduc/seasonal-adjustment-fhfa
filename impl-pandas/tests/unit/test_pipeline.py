import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path

from src.models.pipeline import SeasonalAdjustmentPipeline


class TestSeasonalAdjustmentPipeline:
    """Unit tests for SeasonalAdjustmentPipeline class."""
    
    @pytest.fixture
    def pipeline(self):
        """Create pipeline instance."""
        return SeasonalAdjustmentPipeline()
    
    @pytest.fixture
    def sample_hpi_data(self):
        """Create sample HPI data."""
        np.random.seed(42)
        
        # Create data for 3 geographies
        geographies = ['CA', 'NY', 'TX']
        records = []
        
        for geo in geographies:
            dates = pd.date_range('2015-01-01', '2023-12-31', freq='Q')
            
            for i, date in enumerate(dates):
                # Create seasonal pattern
                quarter = date.quarter
                seasonal_factor = 1 + 0.05 * np.sin(2 * np.pi * quarter / 4)
                
                # Add trend
                trend = 100 * (1 + 0.02 * i / len(dates))
                
                # Add noise
                noise = np.random.normal(0, 2)
                
                records.append({
                    'geography_id': geo,
                    'geography_type': 'STATE',
                    'period': date,
                    'nsa_index': trend * seasonal_factor + noise,
                    'num_transactions': np.random.randint(1000, 5000)
                })
                
        return pd.DataFrame(records)
    
    def test_init(self):
        """Test pipeline initialization."""
        pipeline = SeasonalAdjustmentPipeline()
        assert pipeline.engine is not None
        assert pipeline.selector is not None
        assert isinstance(pipeline.results_cache, dict)
        
        # Test with custom config
        config = {
            'engine_config': {'max_ar_order': 3},
            'selector_config': {'max_p': 2}
        }
        pipeline = SeasonalAdjustmentPipeline(config)
        assert pipeline.engine.max_ar_order == 3
        
    def test_process_geography_success(self, pipeline, sample_hpi_data):
        """Test successful processing of single geography."""
        result = pipeline.process_geography('CA', sample_hpi_data, auto_select_model=False)
        
        assert isinstance(result, dict)
        assert result['geography_id'] == 'CA'
        assert result['status'] == 'success'
        assert 'series' in result
        assert 'diagnostics' in result
        assert 'stability' in result
        
        # Check series components
        series = result['series']
        assert 'original' in series
        assert 'seasonally_adjusted' in series
        assert 'seasonal_factors' in series
        
        # Check that results are cached
        assert 'CA' in pipeline.results_cache
        
    def test_process_geography_insufficient_data(self, pipeline):
        """Test processing with insufficient data."""
        # Create minimal data
        short_data = pd.DataFrame({
            'geography_id': ['CA'] * 10,
            'geography_type': ['STATE'] * 10,
            'period': pd.date_range('2022-01-01', periods=10, freq='Q'),
            'nsa_index': np.random.uniform(100, 110, 10),
            'num_transactions': [1000] * 10
        })
        
        result = pipeline.process_geography('CA', short_data)
        
        assert result['status'] == 'failed'
        assert result['error'] == 'Insufficient data'
        
    @patch('src.models.model_selection.ModelSelector.select_best_model')
    def test_process_geography_auto_model_selection(self, mock_select, pipeline, sample_hpi_data):
        """Test processing with automatic model selection."""
        # Mock the model selection with proper return structure
        mock_model = Mock()
        mock_model.resid = pd.Series(np.random.normal(0, 1, 36))
        mock_model.fittedvalues = pd.Series(np.random.uniform(100, 120, 36))
        mock_model.aic = 100.0
        mock_model.bic = 110.0
        mock_model.llf = -50.0
        mock_model.params = np.array([0.1, 0.2, 0.3])
        mock_model.pvalues = np.array([0.01, 0.02, 0.03])
        mock_model.conf_int = Mock(return_value=np.array([[0.05, 0.15], [0.15, 0.25], [0.25, 0.35]]))
        # Add nested model attribute for seasonal_order access
        mock_model.model = Mock()
        mock_model.model.seasonal_order = (1, 1, 1, 4)
        
        mock_select.return_value = {
            'model': mock_model,
            'order': (1, 1, 1),
            'seasonal_order': (1, 1, 1, 4),
            'aic': 100.0,
            'bic': 110.0,
            'validation': {'rmse': 1.5}
        }
        
        result = pipeline.process_geography('CA', sample_hpi_data, auto_select_model=True)
        
        assert result['status'] == 'success'
        assert mock_select.called
        
    def test_process_geography_error_handling(self, pipeline):
        """Test error handling in geography processing."""
        # Create data that will cause an error
        bad_data = pd.DataFrame({
            'geography_id': ['CA'] * 20,
            'period': pd.date_range('2020-01-01', periods=20, freq='Q'),
            # Missing required columns (nsa_index, geography_type, num_transactions)
        })
        
        result = pipeline.process_geography('CA', bad_data)
        
        assert result['status'] == 'failed'
        assert 'error' in result
        
    def test_batch_process(self, pipeline, sample_hpi_data):
        """Test batch processing of multiple geographies."""
        geography_ids = ['CA', 'NY', 'TX']
        
        summary_df = pipeline.batch_process(
            geography_ids, 
            sample_hpi_data,
            n_jobs=2,
            auto_select_model=False
        )
        
        assert isinstance(summary_df, pd.DataFrame)
        assert len(summary_df) == 3
        assert 'status' in summary_df.columns
        assert 'geography_id' in summary_df.columns
        
        # Check success rate
        success_rate = (summary_df['status'] == 'success').mean()
        assert success_rate > 0  # At least some should succeed
        
    def test_process_single_geography_helper(self, pipeline, sample_hpi_data):
        """Test the helper method for parallel processing."""
        result = pipeline._process_single_geography(
            'CA', sample_hpi_data, auto_select_model=False
        )
        
        assert isinstance(result, dict)
        assert result['geography_id'] == 'CA'
        
    def test_get_adjusted_series(self, pipeline, sample_hpi_data):
        """Test retrieving adjusted series."""
        # First process a geography
        pipeline.process_geography('CA', sample_hpi_data, auto_select_model=False)
        
        # Retrieve the series
        series_df = pipeline.get_adjusted_series('CA')
        
        assert isinstance(series_df, pd.DataFrame)
        assert 'nsa_index' in series_df.columns
        assert 'sa_index' in series_df.columns
        assert 'seasonal_factor' in series_df.columns
        
    def test_get_adjusted_series_not_found(self, pipeline):
        """Test retrieving series for non-existent geography."""
        series_df = pipeline.get_adjusted_series('INVALID')
        assert series_df is None
        
    def test_export_results(self, pipeline, sample_hpi_data):
        """Test exporting results to files."""
        # Process some data first
        pipeline.process_geography('CA', sample_hpi_data, auto_select_model=False)
        pipeline.process_geography('NY', sample_hpi_data, auto_select_model=False)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline.export_results(temp_dir)
            
            # Check files were created
            output_path = Path(temp_dir)
            assert (output_path / 'adjusted_series.csv').exists()
            assert (output_path / 'diagnostics.csv').exists()
            
            # Load and check content
            adj_series = pd.read_csv(output_path / 'adjusted_series.csv')
            assert len(adj_series) > 0
            assert 'geography_id' in adj_series.columns
            
    def test_batch_process_with_failures(self, pipeline):
        """Test batch processing with some failures."""
        # Create mixed data - some good, some bad
        good_data = pd.DataFrame({
            'geography_id': ['CA'] * 30,
            'geography_type': ['STATE'] * 30,
            'period': pd.date_range('2015-01-01', periods=30, freq='Q'),
            'nsa_index': np.random.uniform(100, 120, 30),
            'num_transactions': np.random.randint(1000, 5000, 30)
        })
        
        bad_data = pd.DataFrame({
            'geography_id': ['NY'] * 5,  # Too few observations
            'geography_type': ['STATE'] * 5,
            'period': pd.date_range('2023-01-01', periods=5, freq='Q'),
            'nsa_index': [100] * 5,
            'num_transactions': [1000] * 5
        })
        
        combined_data = pd.concat([good_data, bad_data])
        
        summary_df = pipeline.batch_process(
            ['CA', 'NY'], 
            combined_data,
            n_jobs=1,
            auto_select_model=False
        )
        
        assert len(summary_df) == 2
        assert (summary_df['status'] == 'success').sum() == 1
        assert (summary_df['status'] == 'failed').sum() == 1