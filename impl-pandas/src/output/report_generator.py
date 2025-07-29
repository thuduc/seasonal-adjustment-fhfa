import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger('seasonal_adjustment.report_generator')


class ReportGenerator:
    """Generates reports and visualizations for seasonal adjustment results."""
    
    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.plots_dir = self.output_dir / 'plots'
        self.reports_dir = self.output_dir / 'reports'
        
        for dir_path in [self.plots_dir, self.reports_dir]:
            dir_path.mkdir(exist_ok=True)
            
        # Set plot style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
    def create_adjustment_summary(self, geography_id: str, 
                                results: Dict[str, any]) -> Dict[str, any]:
        """Generate summary of seasonal adjustments for a geography."""
        logger.debug(f"Creating adjustment summary for {geography_id}")
        
        if results.get('status') != 'success':
            return {
                'geography_id': geography_id,
                'status': 'failed',
                'error': results.get('error', 'Unknown error')
            }
            
        series_data = results['series']
        diagnostics = results.get('diagnostics', {})
        stability = results.get('stability', {})
        
        # Calculate summary statistics
        nsa_series = series_data['original']
        sa_series = series_data['seasonally_adjusted']
        seasonal_factors = series_data['seasonal_factors']
        
        # Seasonal pattern by quarter
        seasonal_pattern = {}
        for q in range(1, 5):
            q_factors = seasonal_factors[seasonal_factors.index.quarter == q]
            seasonal_pattern[f'Q{q}_factor'] = q_factors.mean()
            
        # Calculate adjustment impact
        adjustment_impact = {
            'mean_abs_adjustment': np.abs(nsa_series - sa_series).mean(),
            'max_abs_adjustment': np.abs(nsa_series - sa_series).max(),
            'mean_pct_adjustment': (np.abs(nsa_series - sa_series) / nsa_series * 100).mean(),
            'volatility_reduction': (nsa_series.std() - sa_series.std()) / nsa_series.std() * 100
        }
        
        summary = {
            'geography_id': geography_id,
            'model_spec': results.get('model_spec'),
            'seasonal_pattern': seasonal_pattern,
            'adjustment_impact': adjustment_impact,
            'model_fit': {
                'aic': diagnostics.get('aic'),
                'bic': diagnostics.get('bic'),
                'r_squared': results.get('validation', {}).get('r_squared')
            },
            'stability': {
                'stable': stability.get('stable'),
                'max_cv': stability.get('max_cv')
            },
            'validation_metrics': {
                'rmse': results.get('validation', {}).get('rmse'),
                'mape': results.get('validation', {}).get('mape'),
                'direction_accuracy': results.get('validation', {}).get('direction_accuracy')
            }
        }
        
        return summary
    
    def create_diagnostic_report(self, model_results: Dict[str, any]) -> pd.DataFrame:
        """Generate model fit statistics and diagnostics report."""
        logger.debug("Creating diagnostic report")
        
        diagnostics = model_results.get('diagnostics', {})
        
        # Create diagnostic summary
        diag_data = {
            'Test': [],
            'Statistic': [],
            'P-Value': [],
            'Result': [],
            'Interpretation': []
        }
        
        # Ljung-Box test
        if 'ljung_box' in diagnostics:
            lb = diagnostics['ljung_box']
            diag_data['Test'].append('Ljung-Box Q-test')
            diag_data['Statistic'].append(lb['statistic'][0] if isinstance(lb['statistic'], (list, np.ndarray)) else lb['statistic'])
            diag_data['P-Value'].append(lb['p_value'][0] if isinstance(lb['p_value'], (list, np.ndarray)) else lb['p_value'])
            diag_data['Result'].append('Pass' if not lb['significant'] else 'Fail')
            diag_data['Interpretation'].append(
                'No serial correlation' if not lb['significant'] 
                else 'Serial correlation detected'
            )
            
        # Jarque-Bera test
        if 'jarque_bera' in diagnostics:
            jb = diagnostics['jarque_bera']
            diag_data['Test'].append('Jarque-Bera test')
            diag_data['Statistic'].append(jb['statistic'])
            diag_data['P-Value'].append(jb['p_value'])
            diag_data['Result'].append('Pass' if jb['normal'] else 'Fail')
            diag_data['Interpretation'].append(
                'Residuals are normal' if jb['normal'] 
                else 'Residuals are not normal'
            )
            
        # ADF test
        if 'adf_test' in diagnostics:
            adf = diagnostics['adf_test']
            diag_data['Test'].append('ADF test')
            diag_data['Statistic'].append(adf['statistic'])
            diag_data['P-Value'].append(adf['p_value'])
            diag_data['Result'].append('Pass' if adf['stationary'] else 'Fail')
            diag_data['Interpretation'].append(
                'Residuals are stationary' if adf['stationary'] 
                else 'Residuals are non-stationary'
            )
            
        diag_df = pd.DataFrame(diag_data)
        
        return diag_df
    
    def create_causation_report(self, regression_results: Dict[str, any]) -> Dict[str, any]:
        """Summarize weather and demographic impacts on seasonality."""
        logger.debug("Creating causation report")
        
        linear_results = regression_results.get('linear_model', {})
        quantile_results = regression_results.get('quantile_models', {})
        weather_impact = regression_results.get('weather_impact', {})
        
        report = {
            'model_performance': {
                'r_squared': linear_results.get('r_squared'),
                'adj_r_squared': linear_results.get('adj_r_squared'),
                'n_observations': linear_results.get('n_observations'),
                'n_significant_features': len(linear_results.get('significant_features', []))
            },
            'top_predictors': linear_results.get('feature_importance', [])[:10],
            'weather_contribution': {
                'total_impact': weather_impact.get('total_impact'),
                'temperature_impact': weather_impact.get('by_type', {}).get('temperature'),
                'precipitation_impact': weather_impact.get('by_type', {}).get('precipitation'),
                'top_weather_features': weather_impact.get('top_weather_features', [])
            },
            'geographic_insights': {
                'high_seasonality_regions': regression_results.get('geographic_patterns', pd.DataFrame()).head(10).to_dict('records')
            }
        }
        
        # Add quantile regression insights
        if quantile_results:
            varying_features = quantile_results.get('varying_features', [])
            report['quantile_insights'] = {
                'features_with_varying_effects': [f['feature'] for f in varying_features[:5]],
                'strongest_effects_at_extremes': []
            }
            
        return report
    
    def generate_visualizations(self, geography_id: str, 
                              results: Dict[str, any]) -> Dict[str, str]:
        """Create charts for seasonal patterns."""
        logger.debug(f"Generating visualizations for {geography_id}")
        
        if results.get('status') != 'success':
            logger.warning(f"Cannot generate visualizations for failed geography {geography_id}")
            return {}
            
        series_data = results['series']
        plot_paths = {}
        
        # 1. Time series comparison plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        nsa_series = series_data['original']
        sa_series = series_data['seasonally_adjusted']
        
        ax.plot(nsa_series.index, nsa_series.values, label='NSA', alpha=0.7, linewidth=2)
        ax.plot(sa_series.index, sa_series.values, label='SA', alpha=0.9, linewidth=2)
        
        ax.set_title(f'Housing Price Index - {geography_id}', fontsize=14)
        ax.set_xlabel('Period', fontsize=12)
        ax.set_ylabel('Index Value', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ts_path = self.plots_dir / f"{geography_id}_timeseries.png"
        plt.tight_layout()
        plt.savefig(ts_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        plot_paths['timeseries'] = str(ts_path)
        
        # 2. Seasonal factors plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        seasonal_factors = series_data['seasonal_factors']
        
        # Calculate average seasonal pattern
        seasonal_pattern = []
        quarters = []
        for q in range(1, 5):
            q_factors = seasonal_factors[seasonal_factors.index.quarter == q]
            seasonal_pattern.append(q_factors.mean())
            quarters.append(f'Q{q}')
            
        ax.bar(quarters, seasonal_pattern, alpha=0.7)
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
        ax.set_title(f'Average Seasonal Factors - {geography_id}', fontsize=14)
        ax.set_xlabel('Quarter', fontsize=12)
        ax.set_ylabel('Seasonal Factor', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, v in enumerate(seasonal_pattern):
            ax.text(i, v + 0.002, f'{v:.3f}', ha='center', va='bottom')
            
        sf_path = self.plots_dir / f"{geography_id}_seasonal_factors.png"
        plt.tight_layout()
        plt.savefig(sf_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        plot_paths['seasonal_factors'] = str(sf_path)
        
        # 3. Adjustment impact plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Absolute adjustment
        adjustment = nsa_series - sa_series
        ax1.plot(adjustment.index, adjustment.values, alpha=0.7, linewidth=1.5)
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax1.set_title('Seasonal Adjustment Impact (NSA - SA)', fontsize=12)
        ax1.set_ylabel('Index Points', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Percentage adjustment
        pct_adjustment = (adjustment / nsa_series) * 100
        ax2.plot(pct_adjustment.index, pct_adjustment.values, alpha=0.7, linewidth=1.5, color='green')
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax2.set_title('Seasonal Adjustment Impact (%)', fontsize=12)
        ax2.set_xlabel('Period', fontsize=10)
        ax2.set_ylabel('Percentage', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        adj_path = self.plots_dir / f"{geography_id}_adjustment_impact.png"
        plt.tight_layout()
        plt.savefig(adj_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        plot_paths['adjustment_impact'] = str(adj_path)
        
        # 4. Residuals diagnostic plot (if available)
        if 'diagnostics' in results and 'residual_stats' in results['diagnostics']:
            # This would require access to the actual residuals
            # For now, we'll skip this plot
            pass
            
        return plot_paths
    
    def generate_batch_report(self, summary_df: pd.DataFrame, 
                            causation_results: Optional[Dict] = None) -> str:
        """Generate comprehensive report for batch processing."""
        logger.info("Generating batch processing report")
        
        report_path = self.reports_dir / f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("SEASONAL ADJUSTMENT BATCH PROCESSING REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Processing summary
            f.write("PROCESSING SUMMARY\n")
            f.write("-"*40 + "\n")
            
            total_geographies = len(summary_df)
            successful = (summary_df['status'] == 'success').sum() if 'status' in summary_df else 0
            failed = total_geographies - successful
            
            f.write(f"Total geographies processed: {total_geographies}\n")
            f.write(f"Successful: {successful} ({successful/total_geographies*100:.1f}%)\n")
            f.write(f"Failed: {failed} ({failed/total_geographies*100:.1f}%)\n\n")
            
            # Model selection summary
            if 'model_order' in summary_df.columns:
                f.write("MODEL SELECTION SUMMARY\n")
                f.write("-"*40 + "\n")
                
                model_counts = summary_df['model_order'].value_counts()
                f.write("Most common ARIMA orders:\n")
                for order, count in model_counts.head(5).items():
                    f.write(f"  {order}: {count} geographies\n")
                f.write("\n")
                
            # Model fit statistics
            if 'aic' in summary_df.columns:
                f.write("MODEL FIT STATISTICS\n")
                f.write("-"*40 + "\n")
                
                f.write(f"Average AIC: {summary_df['aic'].mean():.2f}\n")
                f.write(f"Average BIC: {summary_df['bic'].mean():.2f}\n")
                
                if 'stable' in summary_df.columns:
                    stable_count = summary_df['stable'].sum()
                    f.write(f"Stable seasonal patterns: {stable_count} ({stable_count/successful*100:.1f}%)\n")
                f.write("\n")
                
            # Validation metrics
            if 'validation_rmse' in summary_df.columns:
                f.write("VALIDATION METRICS\n")
                f.write("-"*40 + "\n")
                
                f.write(f"Average validation RMSE: {summary_df['validation_rmse'].mean():.4f}\n")
                f.write(f"Min RMSE: {summary_df['validation_rmse'].min():.4f}\n")
                f.write(f"Max RMSE: {summary_df['validation_rmse'].max():.4f}\n\n")
                
            # Causation analysis results
            if causation_results:
                f.write("CAUSATION ANALYSIS RESULTS\n")
                f.write("-"*40 + "\n")
                
                linear_model = causation_results.get('linear_model', {})
                f.write(f"Model R-squared: {linear_model.get('r_squared', 'N/A'):.4f}\n")
                f.write(f"Significant predictors: {len(linear_model.get('significant_features', []))}\n")
                
                weather_impact = causation_results.get('weather_impact', {})
                f.write(f"\nWeather contribution: {weather_impact.get('total_impact', 0):.3f}\n")
                
                f.write("\nTop 5 predictors:\n")
                for feat, importance in linear_model.get('feature_importance', [])[:5]:
                    f.write(f"  {feat}: {importance:.4f}\n")
                f.write("\n")
                
            # Failed geographies
            if failed > 0:
                f.write("FAILED GEOGRAPHIES\n")
                f.write("-"*40 + "\n")
                
                failed_df = summary_df[summary_df['status'] == 'failed']
                for _, row in failed_df.iterrows():
                    f.write(f"  {row['geography_id']}: {row.get('error', 'Unknown error')}\n")
                    
        logger.info(f"Batch report saved to {report_path}")
        
        return str(report_path)
    
    def generate_html_report(self, geography_id: str, 
                           results: Dict[str, any],
                           plot_paths: Dict[str, str]) -> str:
        """Generate HTML report for a geography."""
        logger.debug(f"Generating HTML report for {geography_id}")
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Seasonal Adjustment Report - {geography_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .plot {{ margin: 20px 0; text-align: center; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; 
                          background-color: #f9f9f9; border: 1px solid #ddd; }}
            </style>
        </head>
        <body>
            <h1>Seasonal Adjustment Report - {geography_id}</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Model Specification</h2>
            <p>ARIMA{results.get('model_spec', {}).get('order', 'N/A')} x 
               {results.get('model_spec', {}).get('seasonal_order', 'N/A')}</p>
            
            <h2>Key Metrics</h2>
            <div class="metric">
                <strong>AIC:</strong> {results.get('diagnostics', {}).get('aic', 'N/A'):.2f}
            </div>
            <div class="metric">
                <strong>BIC:</strong> {results.get('diagnostics', {}).get('bic', 'N/A'):.2f}
            </div>
            <div class="metric">
                <strong>Stable:</strong> {results.get('stability', {}).get('stable', 'N/A')}
            </div>
            
            <h2>Visualizations</h2>
        """
        
        for plot_name, plot_path in plot_paths.items():
            html_content += f"""
            <div class="plot">
                <h3>{plot_name.replace('_', ' ').title()}</h3>
                <img src="{Path(plot_path).name}" width="800">
            </div>
            """
            
        html_content += """
        </body>
        </html>
        """
        
        html_path = self.reports_dir / f"{geography_id}_report.html"
        with open(html_path, 'w') as f:
            f.write(html_content)
            
        return str(html_path)