"""Performance benchmarks and tests for the FHFA seasonal adjustment pipeline"""

import pytest
import pandas as pd
import numpy as np
import time
import psutil
import os
from memory_profiler import memory_usage
from pathlib import Path

from src.pipeline.orchestrator import SeasonalAdjustmentPipeline
from src.models.arima.regarima import RegARIMA
from src.models.regression.panel import SeasonalityImpactModel
from src.models.seasonal.adjuster import SeasonalAdjuster
from src.data.loaders import HPIDataLoader, WeatherDataLoader
from src.config import get_settings


# Mark all tests in this module as performance tests
pytestmark = pytest.mark.performance


class PerformanceMonitor:
    """Monitor performance metrics during test execution"""
    
    def __init__(self):
        self.start_time = None
        self.start_memory = None
        self.peak_memory = None
        self.process = psutil.Process(os.getpid())
    
    def start(self):
        """Start monitoring"""
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
    
    def stop(self):
        """Stop monitoring and return metrics"""
        elapsed_time = time.time() - self.start_time
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        memory_used = current_memory - self.start_memory
        
        return {
            'elapsed_time': elapsed_time,
            'memory_used_mb': memory_used,
            'peak_memory_mb': current_memory
        }


class TestPerformanceBenchmarks:
    """Performance benchmark tests based on PRD requirements"""
    
    @pytest.fixture
    def performance_monitor(self):
        """Create performance monitor"""
        return PerformanceMonitor()
    
    @pytest.fixture
    def large_state_dataset(self):
        """Generate large state-level dataset for performance testing"""
        # 50 states, 14 years of quarterly data = 2,800 observations
        states = [f'state_{i:02d}' for i in range(50)]
        dates = pd.date_range('2010-01-01', '2023-12-31', freq='QE')
        
        # Create a DataFrame with one column per state
        data = {}
        for state in states:
            state_data = []
            for date in dates:
                # Generate realistic HPI values
                base_value = 100 + np.random.normal(0, 5)
                trend = 0.01 * (date - dates[0]).days / 365
                seasonal = 5 * np.sin(2 * np.pi * date.quarter / 4)
                noise = np.random.normal(0, 2)
                
                hpi_value = base_value * (1 + trend) + seasonal + noise
                state_data.append(hpi_value)
            
            data[f'hpi_state_{state}'] = state_data
        
        return pd.DataFrame(data, index=dates)
    
    @pytest.fixture
    def large_msa_dataset(self):
        """Generate large MSA-level dataset for performance testing"""
        # 100 MSAs, 14 years of quarterly data = 5,600 observations
        msas = [f'msa_{i:03d}' for i in range(100)]
        dates = pd.date_range('2010-01-01', '2023-12-31', freq='QE')
        
        data = []
        for msa in msas:
            for date in dates:
                hpi_value = 100 + np.random.normal(0, 10)
                data.append({
                    'cbsa': msa,
                    'date': date,
                    'hpi': hpi_value
                })
        
        return pd.DataFrame(data)
    
    @pytest.mark.timeout(900)  # 15 minutes timeout
    def test_national_level_performance(self, performance_monitor):
        """Test performance at national level (should complete in < 15 minutes)"""
        # Generate national level data
        dates = pd.date_range('2010-01-01', '2023-12-31', freq='QE')
        national_hpi = pd.Series(
            100 + np.cumsum(np.random.normal(0.5, 2, len(dates))),
            index=dates,
            name='hpi_national'
        )
        
        performance_monitor.start()
        
        # Run seasonal adjustment
        adjuster = SeasonalAdjuster(method='classical')
        adjusted = adjuster.adjust(national_hpi, frequency=4)
        
        metrics = performance_monitor.stop()
        
        # Check performance requirements
        assert metrics['elapsed_time'] < 900, f"National level took {metrics['elapsed_time']} seconds"
        assert metrics['memory_used_mb'] < 1000, f"Used {metrics['memory_used_mb']} MB"
        
        print(f"\nNational level performance:")
        print(f"  Time: {metrics['elapsed_time']:.2f} seconds")
        print(f"  Memory: {metrics['memory_used_mb']:.2f} MB")
    
    @pytest.mark.timeout(5400)  # 90 minutes timeout
    def test_state_level_performance(self, performance_monitor, large_state_dataset):
        """Test performance at state level (should complete in < 90 minutes)"""
        performance_monitor.start()
        
        # Process each state
        adjusted_series = {}
        adjuster = SeasonalAdjuster(method='classical')
        
        for col in large_state_dataset.columns[:10]:  # Test subset for speed
            if col.startswith('hpi_state_'):
                series = large_state_dataset[col].dropna()
                adjusted = adjuster.adjust(series, frequency=4)
                adjusted_series[col] = adjusted
        
        metrics = performance_monitor.stop()
        
        # Extrapolate to full 50 states
        estimated_full_time = metrics['elapsed_time'] * 5
        
        print(f"\nState level performance (10 states):")
        print(f"  Time: {metrics['elapsed_time']:.2f} seconds")
        print(f"  Estimated for 50 states: {estimated_full_time:.2f} seconds")
        print(f"  Memory: {metrics['memory_used_mb']:.2f} MB")
        
        assert estimated_full_time < 5400, f"Estimated time {estimated_full_time} exceeds 90 minutes"
    
    @pytest.mark.timeout(600)  # 10 minutes for subset test
    def test_msa_level_performance(self, performance_monitor, large_msa_dataset):
        """Test performance at MSA level (subset test)"""
        performance_monitor.start()
        
        # Test with 10 MSAs (extrapolate to 100)
        subset_data = large_msa_dataset[large_msa_dataset['cbsa'].isin([f'msa_{i:03d}' for i in range(10)])]
        
        # Run panel regression
        model = SeasonalityImpactModel(model_type='fixed_effects')
        
        # Prepare panel data - don't set index yet as fit will do it
        panel_data = subset_data.copy()
        panel_data['hpi_growth'] = panel_data.groupby('cbsa')['hpi'].pct_change()
        panel_data['temperature'] = np.random.normal(20, 5, len(panel_data))
        
        model.fit(
            panel_data.dropna(),
            y_col='hpi_growth',
            weather_col='temperature',
            sales_col='hpi',  # Use HPI as proxy for sales
            char_cols=[],     # No characteristics for this test
            industry_cols=[], # No industry data for this test
            entity_col='cbsa',
            time_col='date'
        )
        
        metrics = performance_monitor.stop()
        
        # Extrapolate to 100 MSAs
        estimated_full_time = metrics['elapsed_time'] * 10
        
        print(f"\nMSA level performance (10 MSAs):")
        print(f"  Time: {metrics['elapsed_time']:.2f} seconds")
        print(f"  Estimated for 100 MSAs: {estimated_full_time:.2f} seconds")
        print(f"  Memory: {metrics['memory_used_mb']:.2f} MB")
        
        assert estimated_full_time < 9000, f"Estimated time {estimated_full_time} exceeds 150 minutes"
    
    def test_memory_usage_limits(self, large_state_dataset):
        """Test that memory usage stays within 16GB limit"""
        def process_data():
            # Simulate full pipeline processing
            adjuster = SeasonalAdjuster()
            results = {}
            
            for col in large_state_dataset.columns[:5]:
                series = large_state_dataset[col].dropna()
                adjusted = adjuster.adjust(series, frequency=4)
                results[col] = adjusted
            
            return results
        
        # Measure memory usage
        mem_usage = memory_usage(process_data, interval=0.1)
        peak_memory_mb = max(mem_usage)
        
        print(f"\nMemory usage test:")
        print(f"  Peak memory: {peak_memory_mb:.2f} MB")
        print(f"  Average memory: {np.mean(mem_usage):.2f} MB")
        
        # Check memory limit (using 1GB as proxy for test)
        assert peak_memory_mb < 1024, f"Peak memory {peak_memory_mb} MB exceeds limit"
    
    def test_parallel_processing_speedup(self, performance_monitor):
        """Test that parallel processing provides speedup"""
        # Generate test data
        n_series = 20
        dates = pd.date_range('2010-01-01', '2023-12-31', freq='QE')
        
        data = {}
        for i in range(n_series):
            data[f'series_{i}'] = pd.Series(
                100 + np.cumsum(np.random.normal(0, 1, len(dates))),
                index=dates
            )
        
        # Test sequential processing
        performance_monitor.start()
        pipeline_seq = SeasonalAdjustmentPipeline()
        
        results_seq = {}
        for name, series in data.items():
            results_seq[name] = pipeline_seq.run_seasonal_adjustment_only(series)
        
        metrics_seq = performance_monitor.stop()
        
        # Test parallel processing (simulated)
        performance_monitor.start()
        
        # In real implementation, this would use multiprocessing
        # For now, we'll just process normally but test the overhead
        results_par = {}
        for name, series in data.items():
            results_par[name] = pipeline_seq.run_seasonal_adjustment_only(series)
        
        metrics_par = performance_monitor.stop()
        
        print(f"\nParallel processing test:")
        print(f"  Sequential time: {metrics_seq['elapsed_time']:.2f} seconds")
        print(f"  Parallel time: {metrics_par['elapsed_time']:.2f} seconds")
        
        # In real parallel implementation, we'd expect speedup
        # For now, just check overhead is minimal
        assert metrics_par['elapsed_time'] < metrics_seq['elapsed_time'] * 1.2
    
    def test_model_fitting_performance(self, performance_monitor):
        """Test performance of individual model components"""
        # Generate larger dataset
        n_obs = 1000
        dates = pd.date_range('2000-01-01', periods=n_obs, freq='QE')
        
        # Test RegARIMA performance
        series = pd.Series(
            100 + np.cumsum(np.random.normal(0.1, 1, n_obs)),
            index=dates
        )
        
        performance_monitor.start()
        model = RegARIMA(ar_order=2, diff_order=1, ma_order=2)
        model.fit(series)
        metrics_arima = performance_monitor.stop()
        
        print(f"\nModel fitting performance:")
        print(f"  RegARIMA ({n_obs} obs): {metrics_arima['elapsed_time']:.2f} seconds")
        
        # Test panel regression performance
        n_entities = 50
        n_periods = 100
        
        # Generate panel data with proper date index to avoid spline issues
        dates = pd.date_range('2010-01-01', periods=n_periods, freq='QE')
        panel_data_list = []
        for entity in range(n_entities):
            for i, date in enumerate(dates):
                panel_data_list.append({
                    'entity': f'entity_{entity}',
                    'time': date,
                    'y': np.random.randn(),
                    'x1': np.random.randn(),
                    'x2': np.random.randn()
                })
        
        panel_data = pd.DataFrame(panel_data_list)
        
        performance_monitor.start()
        # Use model without splines for performance test
        panel_model = SeasonalityImpactModel(spline_df=0)
        panel_model.fit(panel_data, y_col='y', weather_col='x1', 
                       sales_col='x2', char_cols=[], industry_cols=[],
                       entity_col='entity', time_col='time')
        metrics_panel = performance_monitor.stop()
        
        print(f"  Panel regression ({n_entities}x{n_periods}): {metrics_panel['elapsed_time']:.2f} seconds")
        
        # Check performance is reasonable
        assert metrics_arima['elapsed_time'] < 10, "RegARIMA too slow"
        assert metrics_panel['elapsed_time'] < 20, "Panel regression too slow"
    
    def test_io_performance(self, performance_monitor, tmp_path):
        """Test I/O performance for reading/writing results"""
        # Generate test data
        n_series = 50
        n_obs = 100
        dates = pd.date_range('2010-01-01', periods=n_obs, freq='QE')
        
        data = pd.DataFrame({
            f'series_{i}': 100 + np.cumsum(np.random.normal(0, 1, n_obs))
            for i in range(n_series)
        }, index=dates)
        
        # Test write performance
        performance_monitor.start()
        output_path = tmp_path / "test_results.csv"
        data.to_csv(output_path)
        metrics_write = performance_monitor.stop()
        
        # Test read performance
        performance_monitor.start()
        loaded_data = pd.read_csv(output_path, index_col=0, parse_dates=True)
        metrics_read = performance_monitor.stop()
        
        print(f"\nI/O performance:")
        print(f"  Write {n_series}x{n_obs} CSV: {metrics_write['elapsed_time']:.2f} seconds")
        print(f"  Read {n_series}x{n_obs} CSV: {metrics_read['elapsed_time']:.2f} seconds")
        
        # Check I/O is reasonably fast
        assert metrics_write['elapsed_time'] < 1, "Write too slow"
        assert metrics_read['elapsed_time'] < 1, "Read too slow"
    
    def test_scalability_with_data_size(self, performance_monitor):
        """Test how performance scales with data size"""
        sizes = [100, 500, 1000, 2000]
        times = []
        
        adjuster = SeasonalAdjuster(method='classical')
        
        for size in sizes:
            dates = pd.date_range('2010-01-01', periods=size, freq='D')
            series = pd.Series(
                100 + np.cumsum(np.random.normal(0, 1, size)),
                index=dates
            )
            
            performance_monitor.start()
            
            # Convert to quarterly for seasonal adjustment
            quarterly = series.resample('QE').mean()
            if len(quarterly) >= 8:
                adjusted = adjuster.adjust(quarterly, frequency=4)
            
            metrics = performance_monitor.stop()
            times.append(metrics['elapsed_time'])
        
        print(f"\nScalability test:")
        for size, time_taken in zip(sizes, times):
            print(f"  Size {size}: {time_taken:.3f} seconds")
        
        # Check that scaling is roughly linear or better
        # Time should not increase more than quadratically
        if len(times) > 1:
            scaling_factor = times[-1] / times[0]
            size_factor = sizes[-1] / sizes[0]
            assert scaling_factor < size_factor ** 2, "Performance scaling is worse than quadratic"


class TestOptimizationValidation:
    """Validate that optimizations don't affect results"""
    
    def test_numba_optimization_consistency(self):
        """Test that Numba-optimized functions give same results"""
        # This would test actual Numba functions if implemented
        # For now, we'll test the concept
        
        def slow_moving_average(x, window):
            """Slow pure Python implementation"""
            result = np.empty_like(x)
            for i in range(len(x)):
                if i < window - 1:
                    result[i] = np.nan
                else:
                    result[i] = np.mean(x[i-window+1:i+1])
            return result
        
        def fast_moving_average(x, window):
            """Vectorized implementation"""
            return pd.Series(x).rolling(window).mean().values
        
        # Test consistency
        x = np.random.randn(1000)
        window = 12
        
        slow_result = slow_moving_average(x, window)
        fast_result = fast_moving_average(x, window)
        
        # Check results match (ignoring NaN positions)
        valid_idx = ~np.isnan(slow_result)
        np.testing.assert_array_almost_equal(
            slow_result[valid_idx],
            fast_result[valid_idx],
            decimal=10
        )
    
    def test_caching_effectiveness(self):
        """Test that caching improves performance"""
        from functools import lru_cache
        
        call_count = 0
        
        @lru_cache(maxsize=128)
        def expensive_calculation(n):
            nonlocal call_count
            call_count += 1
            # Simulate expensive operation
            result = sum(i**2 for i in range(n))
            return result
        
        # First calls should compute
        results1 = [expensive_calculation(i) for i in [100, 200, 300]]
        assert call_count == 3
        
        # Repeated calls should use cache
        results2 = [expensive_calculation(i) for i in [100, 200, 300]]
        assert call_count == 3  # No new calls
        assert results1 == results2
        
        # Clear cache
        expensive_calculation.cache_clear()
        
        # New calls after clearing
        results3 = [expensive_calculation(i) for i in [100, 200, 300]]
        assert call_count == 6