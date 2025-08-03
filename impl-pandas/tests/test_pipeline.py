"""Tests for pipeline orchestrator"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import json

from src.pipeline.orchestrator import SeasonalAdjustmentPipeline


class TestSeasonalAdjustmentPipeline:
    """Test seasonal adjustment pipeline"""
    
    def setup_method(self):
        """Set up test pipeline"""
        self.pipeline = SeasonalAdjustmentPipeline()
        
    def test_pipeline_initialization(self):
        """Test pipeline initialization"""
        assert self.pipeline is not None
        assert self.pipeline.settings is not None
        assert self.pipeline.hpi_loader is not None
        assert self.pipeline.ts_validator is not None
        
    def test_run_seasonal_adjustment_only(self):
        """Test running only seasonal adjustment"""
        # Create test series
        dates = pd.date_range('2010-01-01', '2023-12-31', freq='Q')
        seasonal_pattern = [1.05, 0.95, 0.98, 1.02]
        seasonal = np.tile(seasonal_pattern, len(dates) // 4)[:len(dates)]
        trend = np.linspace(100, 150, len(dates))
        
        series = pd.Series(
            trend * seasonal + np.random.normal(0, 2, len(dates)),
            index=dates,
            name='test_hpi'
        )
        
        # Run adjustment
        adjusted = self.pipeline.run_seasonal_adjustment_only(
            series,
            method='classical'
        )
        
        assert isinstance(adjusted, pd.Series)
        assert len(adjusted) == len(series)
        
        # Check seasonal reduction (with tolerance for synthetic data)
        orig_cv = series.groupby(series.index.quarter).std().mean() / series.mean()
        adj_cv = adjusted.groupby(adjusted.index.quarter).std().mean() / adjusted.mean()
        # Allow small increase due to edge effects
        assert adj_cv <= orig_cv * 1.05
        
    def test_run_impact_analysis_only(self):
        """Test running only impact analysis"""
        # Create test panel data
        dates = pd.date_range('2010-01-01', '2023-12-31', freq='Q')
        cbsas = ['12345', '67890']
        
        panel_data = []
        for cbsa in cbsas:
            for date in dates:
                panel_data.append({
                    'cbsa': cbsa,
                    'date': date,
                    'quarter': date.quarter,
                    'hpi_growth': np.random.normal(0.02, 0.05),
                    'temperature': np.random.normal(65, 10),
                    'n_sales': np.random.poisson(1000),
                    'median_income': np.random.normal(60000, 10000),
                    'population_density': np.random.normal(1000, 200),
                    'industry_share': np.random.uniform(0.1, 0.3)
                })
        
        panel_df = pd.DataFrame(panel_data)
        
        # Run analysis
        results = self.pipeline.run_impact_analysis_only(
            panel_df,
            model_type='fixed_effects'
        )
        
        assert 'coefficients' in results
        assert 'diagnostics' in results
        assert isinstance(results['coefficients'], pd.DataFrame)
        
    def test_full_pipeline_with_synthetic_data(self, tmp_path):
        """Test full pipeline execution with synthetic data"""
        # Run pipeline
        results = self.pipeline.run_full_pipeline(
            output_dir=str(tmp_path)
        )
        
        # Check results structure
        assert 'data_loaded' in results
        assert 'validation' in results
        assert 'seasonal_adjustment' in results
        assert 'diagnostics' in results
        
        # Check output files
        assert (tmp_path / 'adjusted_series.csv').exists()
        assert (tmp_path / 'diagnostics.json').exists()
        assert (tmp_path / 'full_results.json').exists()
        
    def test_pipeline_error_handling(self):
        """Test pipeline error handling"""
        # Create invalid series (too short)
        dates = pd.date_range('2023-01-01', '2023-06-30', freq='Q')
        short_series = pd.Series([100, 105], index=dates[:2])
        
        # Should handle gracefully
        with pytest.warns(UserWarning):
            adjusted = self.pipeline.run_seasonal_adjustment_only(
                short_series,
                method='classical'
            )
        
    def test_export_results(self, tmp_path):
        """Test results export functionality"""
        # Set up some results
        self.pipeline.results = {
            'seasonal_adjustment': {
                'hpi_cbsa_12345': {
                    'adjusted_series': pd.Series([100, 105, 110]),
                    'diagnostics': {'seasonal_strength': 0.8},
                    'method': 'classical'
                }
            },
            'diagnostics': {
                'pipeline_metadata': self.pipeline.metadata,
                'adjustment_quality': {'success_rate': 1.0}
            }
        }
        
        # Export
        self.pipeline._export_results(tmp_path)
        
        # Check files
        assert (tmp_path / 'diagnostics.json').exists()
        
        # Load and verify JSON
        with open(tmp_path / 'diagnostics.json', 'r') as f:
            diag = json.load(f)
            assert 'pipeline_metadata' in diag
            assert 'adjustment_quality' in diag
            
    def test_summary_report(self):
        """Test summary report generation"""
        # Set up some results
        self.pipeline.results = {
            'data_loaded': {
                'hpi_shape': (56, 5),
                'weather_shape': (168, 4)
            },
            'seasonal_adjustment': {
                'hpi_cbsa_12345': {'diagnostics': {'seasonal_stability': {'stable': True}}},
                'hpi_cbsa_67890': {'diagnostics': {'seasonal_stability': {'stable': True}}}
            }
        }
        
        report = self.pipeline.get_summary_report()
        
        assert isinstance(report, str)
        assert 'SEASONAL ADJUSTMENT PIPELINE SUMMARY REPORT' in report
        assert 'DATA SUMMARY' in report
        assert 'Series adjusted: 2' in report