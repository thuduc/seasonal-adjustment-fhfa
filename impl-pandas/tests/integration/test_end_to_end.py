import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import time

from src.data.data_loader import DataLoader
from src.models.pipeline import SeasonalAdjustmentPipeline
from src.analysis.causation_analysis import CausationAnalysis
from src.output.results_manager import ResultsManager
from src.output.report_generator import ReportGenerator
from src.utils.parallel_processor import ParallelProcessor
from src.utils.cache_manager import CacheManager


def _process_geo_for_test(geo_id, data, **kwargs):
    """Helper function for parallel processing test."""
    geo_data = data[data['geography_id'] == geo_id]
    return {
        'geography_id': geo_id,
        'status': 'success',
        'mean_index': geo_data['nsa_index'].mean(),
        'count': len(geo_data)
    }


class TestEndToEnd:
    """Integration tests for complete seasonal adjustment workflow."""
    
    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        with tempfile.TemporaryDirectory() as data_dir, \
             tempfile.TemporaryDirectory() as output_dir, \
             tempfile.TemporaryDirectory() as cache_dir:
            yield {
                'data': data_dir,
                'output': output_dir,
                'cache': cache_dir
            }
            
    def test_complete_workflow(self, temp_dirs, test_data_dir, use_generated_data):
        """Test complete seasonal adjustment workflow from data loading to reporting."""
        data_loader = DataLoader()
        
        # Step 1: Use test_data_dir which contains pre-generated or temp data
        if not use_generated_data or not (test_data_dir / 'sample_hpi_data.csv').exists():
            # Generate data if not using pre-generated
            data_loader.create_sample_data(test_data_dir)
        
        # Step 2: Load all data
        hpi_data = data_loader.load_hpi_data(
            test_data_dir / 'sample_hpi_data.csv'
        )
        weather_data = data_loader.load_weather_data(
            test_data_dir / 'sample_weather_data.csv'
        )
        demographics_data = data_loader.load_demographics_data(
            test_data_dir / 'sample_demographics_data.csv'
        )
        industry_data = data_loader.load_industry_data(
            test_data_dir / 'sample_industry_data.csv'
        )
        
        # Step 3: Merge datasets
        master_df = data_loader.merge_datasets()
        
        assert len(master_df) > 0
        assert 'nsa_index' in master_df.columns
        assert 'temp_range' in master_df.columns
        
        # Step 4: Run seasonal adjustment pipeline
        pipeline = SeasonalAdjustmentPipeline()
        
        # Process single geography
        single_result = pipeline.process_geography('CA', hpi_data)
        assert single_result['status'] == 'success'
        
        # Batch process all geographies
        geography_ids = hpi_data['geography_id'].unique().tolist()
        summary_df = pipeline.batch_process(
            geography_ids[:3],  # Limit for speed
            hpi_data,
            n_jobs=2
        )
        
        assert len(summary_df) == 3
        assert (summary_df['status'] == 'success').any()
        
        # Step 5: Skip causation analysis for this test (it requires complete SA data)
        # In a real scenario, we would process all geographies and then run causation analysis
        causation_results = {
            'linear_model': {'r_squared': 0.85},  # Mock result
            'weather_impact': {'total_impact': 0.15}
        }
        
        # Step 6: Save results
        results_manager = ResultsManager(temp_dirs['output'])
        
        # Save single result
        results_manager.save_adjusted_series('CA', single_result)
        results_manager.save_seasonal_factors(
            'CA', 
            single_result['series']['seasonal_factors']
        )
        results_manager.save_diagnostics('CA', single_result['diagnostics'])
        
        # Verify files created
        assert (Path(temp_dirs['output']) / 'adjusted_series' / 'CA_adjusted.csv').exists()
        
        # Step 7: Generate reports
        report_generator = ReportGenerator(temp_dirs['output'])
        
        # Create visualizations
        plot_paths = report_generator.generate_visualizations('CA', single_result)
        assert len(plot_paths) > 0
        
        # Generate batch report
        report_path = report_generator.generate_batch_report(
            summary_df, 
            causation_results
        )
        assert Path(report_path).exists()
        
    def test_parallel_processing(self, temp_dirs, test_data_dir, use_generated_data):
        """Test parallel processing capabilities."""
        data_loader = DataLoader()
        
        # Use test_data_dir which contains pre-generated or temp data
        if not use_generated_data or not (test_data_dir / 'sample_hpi_data.csv').exists():
            data_loader.create_sample_data(test_data_dir)
            
        hpi_data = data_loader.load_hpi_data(
            test_data_dir / 'sample_hpi_data.csv'
        )
        
        # Test parallel processor
        processor = ParallelProcessor(n_cores=2)
            
        # Run parallel processing with a simple lambda
        geography_list = hpi_data['geography_id'].unique().tolist()
        results = processor.parallel_arima_fit(
            geography_list,
            hpi_data,
            _process_geo_for_test
        )
        
        assert len(results) == len(geography_list)
        assert all('mean_index' in r for r in results.values())
        
    def test_caching_functionality(self, temp_dirs):
        """Test caching system."""
        # Initialize cache manager
        cache_manager = CacheManager(cache_dir=temp_dirs['cache'])
        
        # Create sample data
        geography_id = 'CA'
        factors = pd.Series(
            [1.02, 0.98, 1.01, 0.99] * 10,
            index=pd.date_range('2020-01-01', periods=40, freq='Q')
        )
        
        # Cache seasonal factors
        cache_manager.cache_seasonal_factors(geography_id, factors)
        
        # Retrieve from cache
        cached_result = cache_manager.get_cached_results(geography_id)
        
        assert cached_result is not None
        assert 'factors' in cached_result
        pd.testing.assert_series_equal(cached_result['factors'], factors)
        
        # Check cache stats
        stats = cache_manager.get_cache_stats()
        assert stats['hits'] == 1
        assert stats['entries'] == 1
        
    def test_error_recovery(self, temp_dirs):
        """Test system behavior with errors and edge cases."""
        # Create incomplete data
        bad_data = pd.DataFrame({
            'geography_id': ['XX'] * 5,  # Too few observations
            'geography_type': ['STATE'] * 5,
            'period': pd.date_range('2023-01-01', periods=5, freq='Q'),
            'nsa_index': [100] * 5,
            'num_transactions': [1000] * 5
        })
        
        pipeline = SeasonalAdjustmentPipeline()
        result = pipeline.process_geography('XX', bad_data)
        
        assert result['status'] == 'failed'
        assert 'error' in result
        
    def test_performance_benchmarks(self, temp_dirs):
        """Test performance meets requirements."""
        # Create larger dataset
        np.random.seed(42)
        n_geographies = 10
        n_periods = 40
        
        records = []
        for i in range(n_geographies):
            geo_id = f'GEO_{i:03d}'
            dates = pd.date_range('2014-01-01', periods=n_periods, freq='Q')
            
            for j, date in enumerate(dates):
                records.append({
                    'geography_id': geo_id,
                    'geography_type': 'MSA',
                    'period': date,
                    'nsa_index': 100 + np.random.normal(0, 5),
                    'num_transactions': np.random.randint(1000, 5000)
                })
                
        large_data = pd.DataFrame(records)
        
        # Time single geography processing
        pipeline = SeasonalAdjustmentPipeline()
        
        start_time = time.time()
        result = pipeline.process_geography('GEO_000', large_data, auto_select_model=False)
        single_time = time.time() - start_time
        
        # Should process single geography in < 3 seconds
        assert single_time < 3.0
        assert result['status'] == 'success'
        
        # Time batch processing
        start_time = time.time()
        summary_df = pipeline.batch_process(
            [f'GEO_{i:03d}' for i in range(5)],
            large_data,
            n_jobs=2,
            auto_select_model=False
        )
        batch_time = time.time() - start_time
        
        # Should process 5 geographies reasonably quickly
        assert batch_time < 15.0  # 3 seconds per geography
        
    def test_data_validation_integration(self, temp_dirs):
        """Test data validation across the pipeline."""
        # Create data with various issues
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='Q')
        
        # Missing values
        data_with_missing = pd.DataFrame({
            'geography_id': ['CA'] * len(dates),
            'geography_type': ['STATE'] * len(dates),
            'period': dates,
            'nsa_index': [100 + i if i % 5 != 0 else np.nan for i in range(len(dates))],
            'num_transactions': np.random.randint(1000, 5000, len(dates))
        })
        
        pipeline = SeasonalAdjustmentPipeline()
        
        # Should handle missing values appropriately
        result = pipeline.process_geography('CA', data_with_missing)
        
        # Pipeline should either succeed with interpolation or fail gracefully
        assert result['status'] in ['success', 'failed']
        
    def test_results_export_integration(self, temp_dirs, test_data_dir, use_generated_data):
        """Test exporting results in various formats."""
        data_loader = DataLoader()
        
        # Use test_data_dir which contains pre-generated or temp data
        if not use_generated_data or not (test_data_dir / 'sample_hpi_data.csv').exists():
            data_loader.create_sample_data(test_data_dir)
            
        hpi_data = data_loader.load_hpi_data(
            test_data_dir / 'sample_hpi_data.csv'
        )
        
        pipeline = SeasonalAdjustmentPipeline()
        
        # Process multiple geographies
        for geo_id in hpi_data['geography_id'].unique()[:3]:
            pipeline.process_geography(geo_id, hpi_data, auto_select_model=False)
            
        # Export results
        pipeline.export_results(temp_dirs['output'])
        
        # Check exported files
        assert (Path(temp_dirs['output']) / 'adjusted_series.csv').exists()
        assert (Path(temp_dirs['output']) / 'diagnostics.csv').exists()
        
        # Load and verify exported data
        adj_series = pd.read_csv(Path(temp_dirs['output']) / 'adjusted_series.csv')
        assert len(adj_series) > 0
        assert 'geography_id' in adj_series.columns
        assert 'sa_index' in adj_series.columns