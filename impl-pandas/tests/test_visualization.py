"""Tests for visualization components"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import tempfile
import json
from datetime import datetime

from src.visualization.plots import SeasonalAdjustmentPlotter, DiagnosticPlotter
from src.visualization.reports import ReportGenerator


class TestSeasonalAdjustmentPlotter:
    """Test seasonal adjustment plotting functionality"""
    
    @pytest.fixture
    def sample_series(self):
        """Create sample time series data"""
        dates = pd.date_range('2010-01-01', '2023-12-31', freq='QE')
        n_periods = len(dates)
        
        # Create trend + seasonal + noise
        trend = np.linspace(100, 150, n_periods)
        seasonal = 10 * np.sin(2 * np.pi * np.arange(n_periods) / 4)
        noise = np.random.normal(0, 2, n_periods)
        
        original = pd.Series(trend + seasonal + noise, index=dates, name='original')
        adjusted = pd.Series(trend + noise * 0.5, index=dates, name='adjusted')
        
        return original, adjusted
    
    @pytest.fixture
    def plotter(self):
        """Create plotter instance"""
        return SeasonalAdjustmentPlotter()
    
    def test_plot_adjustment_comparison(self, plotter, sample_series, tmp_path):
        """Test adjustment comparison plot"""
        original, adjusted = sample_series
        
        # Test plot creation
        fig = plotter.plot_adjustment_comparison(
            original, adjusted,
            title="Test Adjustment"
        )
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2
        
        # Test save functionality
        save_path = tmp_path / "test_adjustment.png"
        fig = plotter.plot_adjustment_comparison(
            original, adjusted,
            save_path=str(save_path)
        )
        
        assert save_path.exists()
        plt.close(fig)
    
    def test_plot_decomposition(self, plotter, sample_series, tmp_path):
        """Test decomposition plot"""
        original, adjusted = sample_series
        
        # Create components
        components = {
            'trend': adjusted,
            'seasonal': original - adjusted,
            'residual': pd.Series(np.random.normal(0, 0.5, len(original)), 
                                 index=original.index)
        }
        
        # Test plot creation
        fig = plotter.plot_decomposition(components)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 3
        
        plt.close(fig)
    
    def test_plot_seasonal_patterns(self, plotter, sample_series):
        """Test seasonal pattern plot"""
        original, adjusted = sample_series
        
        # Test quarterly patterns
        fig = plotter.plot_seasonal_patterns(
            original, adjusted,
            period='quarter'
        )
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2
        
        plt.close(fig)
    
    def test_plot_multiple_series(self, plotter, sample_series):
        """Test multiple series plot"""
        original, adjusted = sample_series
        
        series_dict = {
            'Original': original,
            'Adjusted': adjusted,
            'Difference': original - adjusted
        }
        
        # Test without normalization
        fig = plotter.plot_multiple_series(
            series_dict,
            normalize=False
        )
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes[0].lines) == 3
        
        # Test with normalization
        fig2 = plotter.plot_multiple_series(
            series_dict,
            normalize=True
        )
        
        assert isinstance(fig2, plt.Figure)
        
        plt.close(fig)
        plt.close(fig2)


class TestDiagnosticPlotter:
    """Test diagnostic plotting functionality"""
    
    @pytest.fixture
    def sample_residuals(self):
        """Create sample residuals"""
        dates = pd.date_range('2010-01-01', '2023-12-31', freq='QE')
        residuals = pd.Series(
            np.random.normal(0, 1, len(dates)),
            index=dates,
            name='residuals'
        )
        return residuals
    
    @pytest.fixture
    def plotter(self):
        """Create diagnostic plotter"""
        return DiagnosticPlotter()
    
    def test_plot_residual_diagnostics(self, plotter, sample_residuals, tmp_path):
        """Test residual diagnostic plots"""
        fig = plotter.plot_residual_diagnostics(
            sample_residuals,
            title="Residual Diagnostics"
        )
        
        assert isinstance(fig, plt.Figure)
        # Should have 5 subplots
        assert len(fig.axes) == 5
        
        # Test save
        save_path = tmp_path / "residual_diagnostics.png"
        fig = plotter.plot_residual_diagnostics(
            sample_residuals,
            save_path=str(save_path)
        )
        
        assert save_path.exists()
        plt.close(fig)
    
    def test_plot_model_fit(self, plotter, sample_residuals):
        """Test model fit plot"""
        # Create actual and fitted series
        actual = sample_residuals + 100
        fitted = actual + np.random.normal(0, 0.5, len(actual))
        residuals = actual - fitted
        
        # Test with residuals
        fig = plotter.plot_model_fit(
            actual, fitted, residuals,
            title="Model Fit"
        )
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 3
        
        # Test without residuals
        fig2 = plotter.plot_model_fit(
            actual, fitted,
            title="Model Fit (No Residuals)"
        )
        
        assert isinstance(fig2, plt.Figure)
        assert len(fig2.axes) == 2
        
        plt.close(fig)
        plt.close(fig2)
    
    def test_plot_coefficient_analysis(self, plotter):
        """Test coefficient analysis plot"""
        # Create sample coefficient data
        coefficients = pd.DataFrame({
            'coefficient': [0.5, -0.3, 0.8, -0.1, 0.4],
            'std_error': [0.1, 0.15, 0.2, 0.05, 0.12],
            'p_value': [0.001, 0.05, 0.0001, 0.4, 0.01]
        }, index=['var1', 'var2', 'var3', 'var4', 'var5'])
        
        fig = plotter.plot_coefficient_analysis(
            coefficients,
            title="Coefficient Analysis"
        )
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2
        
        plt.close(fig)
    
    def test_plot_stability_analysis(self, plotter):
        """Test stability analysis plot"""
        # Create time-varying results
        periods = ['2010-2012', '2013-2015', '2016-2018', '2019-2021']
        results_over_time = {}
        
        for i, period in enumerate(periods):
            results_over_time[period] = pd.DataFrame({
                'coefficient': [0.5 + i*0.1, -0.3 + i*0.05, 0.8 - i*0.1],
                'p_value': [0.01, 0.05, 0.001],
                't_stat': [5.0, -2.0, 8.0]
            }, index=['temperature', 'sales', 'income'])
        
        fig = plotter.plot_stability_analysis(
            results_over_time,
            metric='coefficient',
            title="Parameter Stability"
        )
        
        assert isinstance(fig, plt.Figure)
        # Should have at least 3 lines for parameters (might have more due to legend)
        assert len(fig.axes[0].lines) >= 3
        
        plt.close(fig)


class TestReportGenerator:
    """Test report generation functionality"""
    
    @pytest.fixture
    def sample_pipeline_results(self):
        """Create sample pipeline results"""
        # Create sample data
        dates = pd.date_range('2010-01-01', '2023-12-31', freq='QE')
        
        # Sample adjusted series
        adjustment_results = {}
        for state in ['CA', 'TX', 'FL']:
            series = pd.Series(
                np.random.normal(100, 10, len(dates)),
                index=dates
            )
            adjustment_results[f'hpi_state_{state}'] = {
                'adjusted_series': series,
                'method': 'classical',
                'diagnostics': {
                    'seasonal_strength': np.random.uniform(0.5, 0.9),
                    'seasonal_stability': {'stable': True, 'cv': 0.05}
                }
            }
        
        # Sample coefficients
        coefficients = pd.DataFrame({
            'coefficient': [0.011, 0.5, -0.3],
            'std_error': [0.002, 0.1, 0.15],
            'p_value': [0.001, 0.01, 0.05],
            't_stat': [5.5, 5.0, -2.0]
        }, index=['temperature', 'income', 'unemployment'])
        
        return {
            'data_loaded': {
                'hpi_shape': (len(dates), 10),
                'weather_shape': (len(dates), 5),
                'demographic_shape': (100, 20)
            },
            'validation': {
                'hpi_state_CA': {'overall_valid': True},
                'hpi_state_TX': {'overall_valid': True},
                'hpi_state_FL': {'overall_valid': False, 'missing_data': True}
            },
            'seasonal_adjustment': adjustment_results,
            'impact_analysis': {
                'fixed_effects': {
                    'coefficients': coefficients,
                    'diagnostics': {'r_squared': 0.85}
                },
                'quantile': {
                    'coefficients': coefficients * 0.9
                }
            },
            'diagnostics': {
                'pipeline_metadata': {
                    'pipeline_version': '1.0.0',
                    'created_at': datetime.now().isoformat()
                },
                'data_quality': {
                    'validation_summary': {
                        'total_series': 3,
                        'valid_series': 2
                    }
                },
                'adjustment_quality': {
                    'success_rate': 0.67
                },
                'model_quality': {
                    'fixed_effects_r2': 0.85
                }
            }
        }
    
    @pytest.fixture
    def report_generator(self):
        """Create report generator"""
        return ReportGenerator()
    
    def test_generate_html_report(self, report_generator, sample_pipeline_results, tmp_path):
        """Test HTML report generation"""
        output_path = tmp_path / "test_report.html"
        
        report_generator._generate_html_report(
            sample_pipeline_results,
            output_path
        )
        
        assert output_path.exists()
        
        # Check content
        with open(output_path, 'r') as f:
            content = f.read()
            
        assert "FHFA Seasonal Adjustment Analysis Report" in content
        assert "Executive Summary" in content
        assert "Data Quality Assessment" in content
        assert "Seasonal Adjustment Results" in content
    
    def test_generate_excel_report(self, report_generator, sample_pipeline_results, tmp_path):
        """Test Excel report generation"""
        output_path = tmp_path / "test_report.xlsx"
        
        report_generator._generate_excel_report(
            sample_pipeline_results,
            output_path
        )
        
        assert output_path.exists()
        
        # Check sheets
        xl_file = pd.ExcelFile(output_path)
        sheets = xl_file.sheet_names
        
        assert 'Summary' in sheets
        assert 'Adjusted Series' in sheets
        assert 'Diagnostics' in sheets
        assert 'FE Coefficients' in sheets
    
    def test_generate_json_export(self, report_generator, sample_pipeline_results, tmp_path):
        """Test JSON export generation"""
        output_path = tmp_path / "test_results.json"
        
        report_generator._generate_json_export(
            sample_pipeline_results,
            output_path
        )
        
        assert output_path.exists()
        
        # Check content
        with open(output_path, 'r') as f:
            data = json.load(f)
            
        assert 'data_loaded' in data
        assert 'seasonal_adjustment' in data
        assert 'impact_analysis' in data
        assert 'diagnostics' in data
    
    def test_generate_full_report(self, report_generator, sample_pipeline_results, tmp_path):
        """Test full report generation"""
        report_paths = report_generator.generate_full_report(
            sample_pipeline_results,
            tmp_path,
            formats=['html', 'excel', 'json']
        )
        
        assert 'html' in report_paths
        assert 'excel' in report_paths
        assert 'json' in report_paths
        
        # Check all files exist
        for format_type, path in report_paths.items():
            assert Path(path).exists()
    
    def test_generate_summary_statistics(self, report_generator):
        """Test summary statistics generation"""
        # Create sample adjusted series
        dates = pd.date_range('2010-01-01', '2023-12-31', freq='QE')
        adjusted_series = {
            f'series_{i}': pd.Series(
                np.random.normal(100 + i*10, 5, len(dates)),
                index=dates
            )
            for i in range(3)
        }
        
        stats_df = report_generator.generate_summary_statistics(adjusted_series)
        
        assert isinstance(stats_df, pd.DataFrame)
        assert len(stats_df) == 3
        assert 'Mean' in stats_df.columns
        assert 'Std' in stats_df.columns
        assert 'Mean Growth' in stats_df.columns