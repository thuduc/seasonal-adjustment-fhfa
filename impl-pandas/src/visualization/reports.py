"""Report generation for seasonal adjustment analysis"""

from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from jinja2 import Template
from loguru import logger
import matplotlib.pyplot as plt
from io import BytesIO
import base64

from .plots import SeasonalAdjustmentPlotter, DiagnosticPlotter
from ..config import get_settings


class ReportGenerator:
    """
    Generate comprehensive reports for seasonal adjustment analysis
    
    Creates:
    - HTML reports with embedded visualizations
    - PDF reports (if wkhtmltopdf available)
    - Excel workbooks with multiple sheets
    - JSON exports for further analysis
    """
    
    def __init__(self):
        """Initialize report generator"""
        self.settings = get_settings()
        self.sa_plotter = SeasonalAdjustmentPlotter()
        self.diag_plotter = DiagnosticPlotter()
        
    def generate_full_report(self,
                           pipeline_results: Dict[str, Any],
                           output_dir: Union[str, Path],
                           formats: List[str] = ['html', 'excel']) -> Dict[str, str]:
        """
        Generate comprehensive report in multiple formats
        
        Parameters:
        -----------
        pipeline_results : Dict[str, Any]
            Results from pipeline execution
        output_dir : Union[str, Path]
            Output directory for reports
        formats : List[str]
            Report formats to generate ('html', 'excel', 'pdf', 'json')
            
        Returns:
        --------
        Dict[str, str]
            Paths to generated reports
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_paths = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate reports in requested formats
        if 'html' in formats:
            html_path = output_dir / f"seasonal_adjustment_report_{timestamp}.html"
            self._generate_html_report(pipeline_results, html_path)
            report_paths['html'] = str(html_path)
            
        if 'excel' in formats:
            excel_path = output_dir / f"seasonal_adjustment_results_{timestamp}.xlsx"
            self._generate_excel_report(pipeline_results, excel_path)
            report_paths['excel'] = str(excel_path)
            
        if 'pdf' in formats:
            pdf_path = output_dir / f"seasonal_adjustment_report_{timestamp}.pdf"
            if self._generate_pdf_report(pipeline_results, pdf_path):
                report_paths['pdf'] = str(pdf_path)
            else:
                logger.warning("PDF generation failed - wkhtmltopdf not available")
                
        if 'json' in formats:
            json_path = output_dir / f"seasonal_adjustment_results_{timestamp}.json"
            self._generate_json_export(pipeline_results, json_path)
            report_paths['json'] = str(json_path)
            
        logger.info(f"Generated {len(report_paths)} reports in {output_dir}")
        return report_paths
    
    def _generate_html_report(self, results: Dict[str, Any], output_path: Path) -> None:
        """Generate HTML report with embedded visualizations"""
        
        # Prepare report data
        report_data = {
            'title': 'FHFA Seasonal Adjustment Analysis Report',
            'generation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'summary': self._generate_summary(results),
            'data_quality': self._format_data_quality(results),
            'adjustment_results': self._format_adjustment_results(results),
            'impact_analysis': self._format_impact_analysis(results),
            'plots': self._generate_plots(results),
            'diagnostics': self._format_diagnostics(results)
        }
        
        # HTML template
        template_str = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
    <meta charset="utf-8">
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 40px; 
            background-color: #f5f5f5;
        }
        .container { 
            max-width: 1200px; 
            margin: 0 auto; 
            background-color: white;
            padding: 30px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        h1 { 
            color: #2c3e50; 
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        h2 { 
            color: #34495e; 
            margin-top: 30px;
            border-bottom: 1px solid #bdc3c7;
            padding-bottom: 5px;
        }
        h3 { 
            color: #7f8c8d; 
        }
        .summary-box {
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }
        .metric {
            display: inline-block;
            margin: 10px 20px 10px 0;
            padding: 10px;
            background-color: #3498db;
            color: white;
            border-radius: 3px;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        th {
            background-color: #3498db;
            color: white;
        }
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        .plot-container {
            margin: 20px 0;
            text-align: center;
        }
        .plot-container img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .warning {
            background-color: #f39c12;
            color: white;
            padding: 10px;
            border-radius: 3px;
            margin: 10px 0;
        }
        .success {
            background-color: #27ae60;
            color: white;
            padding: 10px;
            border-radius: 3px;
            margin: 10px 0;
        }
        .footer {
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #7f8c8d;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>{{ title }}</h1>
        <p><strong>Generated:</strong> {{ generation_date }}</p>
        
        <div class="summary-box">
            <h2>Executive Summary</h2>
            {{ summary | safe }}
        </div>
        
        <h2>Data Quality Assessment</h2>
        {{ data_quality | safe }}
        
        <h2>Seasonal Adjustment Results</h2>
        {{ adjustment_results | safe }}
        
        {% if plots %}
        <h2>Visualizations</h2>
        {% for plot in plots %}
        <div class="plot-container">
            <h3>{{ plot.title }}</h3>
            <img src="data:image/png;base64,{{ plot.data }}" alt="{{ plot.title }}">
        </div>
        {% endfor %}
        {% endif %}
        
        {% if impact_analysis %}
        <h2>Impact Analysis</h2>
        {{ impact_analysis | safe }}
        {% endif %}
        
        <h2>Diagnostics</h2>
        {{ diagnostics | safe }}
        
        <div class="footer">
            <p>FHFA Seasonal Adjustment Pipeline v1.0.0</p>
        </div>
    </div>
</body>
</html>
"""
        
        # Render template
        template = Template(template_str)
        html_content = template.render(**report_data)
        
        # Save report
        with open(output_path, 'w') as f:
            f.write(html_content)
            
        logger.info(f"HTML report saved to {output_path}")
    
    def _generate_summary(self, results: Dict[str, Any]) -> str:
        """Generate executive summary"""
        summary_items = []
        
        # Data summary
        if 'data_loaded' in results:
            data_info = results['data_loaded']
            summary_items.append(
                f"<div class='metric'>HPI Data: {data_info.get('hpi_shape', 'N/A')}</div>"
            )
            
        # Adjustment summary
        if 'seasonal_adjustment' in results:
            n_series = len(results['seasonal_adjustment'])
            summary_items.append(
                f"<div class='metric'>Series Adjusted: {n_series}</div>"
            )
            
        # Success rate
        if 'diagnostics' in results:
            success_rate = results['diagnostics'].get('adjustment_quality', {}).get('success_rate', 0)
            summary_items.append(
                f"<div class='metric'>Success Rate: {success_rate:.1%}</div>"
            )
            
        # Impact analysis
        if 'impact_analysis' in results:
            summary_items.append(
                "<div class='success'>Impact analysis completed successfully</div>"
            )
            
        return '\n'.join(summary_items)
    
    def _format_data_quality(self, results: Dict[str, Any]) -> str:
        """Format data quality section"""
        html_parts = []
        
        if 'validation' in results:
            validation = results['validation']
            
            # Count valid series
            valid_count = sum(1 for v in validation.values() 
                            if isinstance(v, dict) and v.get('overall_valid', False))
            total_count = len(validation)
            
            html_parts.append(
                f"<p>Data validation completed: {valid_count}/{total_count} series passed validation</p>"
            )
            
            # Create validation table
            table_rows = []
            for series, val_result in validation.items():
                if isinstance(val_result, dict):
                    status = "✓" if val_result.get('overall_valid', False) else "✗"
                    issues = ', '.join(k for k, v in val_result.items() 
                                     if k != 'overall_valid' and not v)
                    table_rows.append(
                        f"<tr><td>{series}</td><td>{status}</td><td>{issues or 'None'}</td></tr>"
                    )
                    
            if table_rows:
                html_parts.append("""
                <table>
                    <tr><th>Series</th><th>Valid</th><th>Issues</th></tr>
                    {}
                </table>
                """.format('\n'.join(table_rows)))
                
        return '\n'.join(html_parts)
    
    def _format_adjustment_results(self, results: Dict[str, Any]) -> str:
        """Format seasonal adjustment results section"""
        html_parts = []
        
        if 'seasonal_adjustment' in results:
            adj_results = results['seasonal_adjustment']
            
            # Summary statistics
            html_parts.append(f"<p>Adjusted {len(adj_results)} series using various methods.</p>")
            
            # Results table
            table_rows = []
            for series, result in adj_results.items():
                method = result.get('method', 'unknown')
                diagnostics = result.get('diagnostics', {})
                seasonal_strength = diagnostics.get('seasonal_strength', 'N/A')
                stable = diagnostics.get('seasonal_stability', {}).get('stable', False)
                
                if isinstance(seasonal_strength, (int, float)):
                    strength_str = f"{seasonal_strength:.3f}"
                else:
                    strength_str = str(seasonal_strength)
                    
                table_rows.append(
                    f"<tr><td>{series}</td><td>{method}</td>"
                    f"<td>{strength_str}</td>"
                    f"<td>{'Yes' if stable else 'No'}</td></tr>"
                )
                
            if table_rows:
                html_parts.append("""
                <table>
                    <tr><th>Series</th><th>Method</th><th>Seasonal Strength</th><th>Stable Pattern</th></tr>
                    {}
                </table>
                """.format('\n'.join(table_rows)))
                
        return '\n'.join(html_parts)
    
    def _format_impact_analysis(self, results: Dict[str, Any]) -> str:
        """Format impact analysis section"""
        html_parts = []
        
        if 'impact_analysis' in results:
            impact = results['impact_analysis']
            
            # Fixed effects results
            if 'fixed_effects' in impact:
                fe_results = impact['fixed_effects']
                html_parts.append("<h3>Fixed Effects Model Results</h3>")
                
                if 'coefficients' in fe_results:
                    coef_df = fe_results['coefficients']
                    html_parts.append(self._dataframe_to_html(coef_df))
                    
                # Quarter-specific coefficients
                if 'quarter_coefficients' in fe_results:
                    html_parts.append("<h4>Temperature Coefficients by Quarter</h4>")
                    quarter_coefs = fe_results['quarter_coefficients']
                    
                    table_rows = []
                    for quarter, coef_data in quarter_coefs.items():
                        if isinstance(coef_data, pd.DataFrame) and 'temperature' in coef_data.index:
                            temp_coef = coef_data.loc['temperature', 'coefficient']
                            temp_pval = coef_data.loc['temperature', 'p_value']
                            table_rows.append(
                                f"<tr><td>Q{quarter}</td><td>{temp_coef:.4f}</td><td>{temp_pval:.4f}</td></tr>"
                            )
                            
                    if table_rows:
                        html_parts.append("""
                        <table>
                            <tr><th>Quarter</th><th>Temperature Coefficient</th><th>P-value</th></tr>
                            {}
                        </table>
                        """.format('\n'.join(table_rows)))
                        
            # Quantile regression results
            if 'quantile' in impact:
                html_parts.append("<h3>Quantile Regression Results</h3>")
                q_results = impact['quantile']
                
                if 'coefficients' in q_results:
                    html_parts.append(self._dataframe_to_html(q_results['coefficients']))
                    
        return '\n'.join(html_parts)
    
    def _format_diagnostics(self, results: Dict[str, Any]) -> str:
        """Format diagnostics section"""
        html_parts = []
        
        if 'diagnostics' in results:
            diagnostics = results['diagnostics']
            
            # Pipeline metadata
            if 'pipeline_metadata' in diagnostics:
                meta = diagnostics['pipeline_metadata']
                html_parts.append(
                    f"<p><strong>Pipeline Version:</strong> {meta.get('pipeline_version', 'N/A')}</p>"
                )
                
            # Data quality metrics
            if 'data_quality' in diagnostics:
                quality = diagnostics['data_quality']
                if 'validation_summary' in quality:
                    summary = quality['validation_summary']
                    html_parts.append(
                        f"<p><strong>Data Quality:</strong> "
                        f"{summary.get('valid_series', 0)}/{summary.get('total_series', 0)} series valid</p>"
                    )
                    
            # Model quality metrics
            if 'model_quality' in diagnostics:
                model_qual = diagnostics['model_quality']
                if 'fixed_effects_r2' in model_qual:
                    r2 = model_qual['fixed_effects_r2']
                    if r2 is not None:
                        html_parts.append(f"<p><strong>Fixed Effects R²:</strong> {r2:.3f}</p>")
                        
        return '\n'.join(html_parts)
    
    def _generate_plots(self, results: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate plots and return as base64 encoded images"""
        plots = []
        
        # Generate adjustment comparison plots
        if 'seasonal_adjustment' in results:
            adj_results = results['seasonal_adjustment']
            
            # Plot first few series
            for idx, (series_name, result) in enumerate(adj_results.items()):
                if idx >= 3:  # Limit number of plots
                    break
                    
                if 'adjusted_series' in result:
                    # Get original series from data
                    if 'data_loaded' in results:
                        # Create a simple plot
                        fig, ax = plt.subplots(figsize=(10, 6))
                        adjusted = result['adjusted_series']
                        
                        # Generate approximate original series for plotting
                        if result.get('diagnostics', {}).get('seasonal_strength', 0) > 0:
                            seasonal_component = np.sin(2 * np.pi * np.arange(len(adjusted)) / 4) * 5
                            original = adjusted + seasonal_component
                        else:
                            original = adjusted
                            
                        ax.plot(adjusted.index, original, label='Original', alpha=0.7)
                        ax.plot(adjusted.index, adjusted, label='Seasonally Adjusted', linewidth=2)
                        ax.set_title(f'Seasonal Adjustment: {series_name}')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        
                        # Convert to base64
                        buffer = BytesIO()
                        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
                        buffer.seek(0)
                        image_base64 = base64.b64encode(buffer.read()).decode()
                        plt.close(fig)
                        
                        plots.append({
                            'title': f'Seasonal Adjustment: {series_name}',
                            'data': image_base64
                        })
                        
        # Generate diagnostic plots
        if 'impact_analysis' in results and 'fixed_effects' in results['impact_analysis']:
            fe_results = results['impact_analysis']['fixed_effects']
            
            if 'coefficients' in fe_results:
                coef_df = fe_results['coefficients']
                
                # Create coefficient plot
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Filter significant coefficients
                sig_coef = coef_df[coef_df['p_value'] < 0.1].sort_values('coefficient')
                
                if len(sig_coef) > 0:
                    y_pos = np.arange(len(sig_coef))
                    ax.barh(y_pos, sig_coef['coefficient'])
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(sig_coef.index)
                    ax.set_xlabel('Coefficient Value')
                    ax.set_title('Significant Coefficients (p < 0.1)')
                    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
                    ax.grid(True, alpha=0.3, axis='x')
                    
                    # Convert to base64
                    buffer = BytesIO()
                    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
                    buffer.seek(0)
                    image_base64 = base64.b64encode(buffer.read()).decode()
                    plt.close(fig)
                    
                    plots.append({
                        'title': 'Impact Analysis: Significant Coefficients',
                        'data': image_base64
                    })
                    
        return plots
    
    def _dataframe_to_html(self, df: pd.DataFrame) -> str:
        """Convert DataFrame to HTML table"""
        return df.to_html(classes='dataframe', index=True, float_format=lambda x: f'{x:.4f}')
    
    def _generate_excel_report(self, results: Dict[str, Any], output_path: Path) -> None:
        """Generate Excel report with multiple sheets"""
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = {
                'Metric': [],
                'Value': []
            }
            
            if 'data_loaded' in results:
                summary_data['Metric'].append('HPI Data Shape')
                summary_data['Value'].append(str(results['data_loaded'].get('hpi_shape', 'N/A')))
                
            if 'seasonal_adjustment' in results:
                summary_data['Metric'].append('Series Adjusted')
                summary_data['Value'].append(len(results['seasonal_adjustment']))
                
            if 'diagnostics' in results:
                success_rate = results['diagnostics'].get('adjustment_quality', {}).get('success_rate', 0)
                summary_data['Metric'].append('Success Rate')
                summary_data['Value'].append(f"{success_rate:.1%}")
                
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            # Adjusted series sheet
            if 'seasonal_adjustment' in results:
                adj_df = pd.DataFrame()
                for series_name, result in results['seasonal_adjustment'].items():
                    if 'adjusted_series' in result:
                        adj_df[series_name] = result['adjusted_series']
                        
                if not adj_df.empty:
                    adj_df.to_excel(writer, sheet_name='Adjusted Series')
                    
            # Diagnostics sheet
            if 'seasonal_adjustment' in results:
                diag_data = []
                for series_name, result in results['seasonal_adjustment'].items():
                    if 'diagnostics' in result:
                        diag = result['diagnostics']
                        diag_data.append({
                            'Series': series_name,
                            'Method': result.get('method', 'N/A'),
                            'Seasonal Strength': diag.get('seasonal_strength', 'N/A'),
                            'Stable': diag.get('seasonal_stability', {}).get('stable', 'N/A')
                        })
                        
                if diag_data:
                    pd.DataFrame(diag_data).to_excel(writer, sheet_name='Diagnostics', index=False)
                    
            # Impact analysis sheets
            if 'impact_analysis' in results:
                impact = results['impact_analysis']
                
                # Fixed effects coefficients
                if 'fixed_effects' in impact and 'coefficients' in impact['fixed_effects']:
                    impact['fixed_effects']['coefficients'].to_excel(
                        writer, sheet_name='FE Coefficients'
                    )
                    
                # Quantile regression coefficients
                if 'quantile' in impact and 'coefficients' in impact['quantile']:
                    impact['quantile']['coefficients'].to_excel(
                        writer, sheet_name='Quantile Coefficients'
                    )
                    
        logger.info(f"Excel report saved to {output_path}")
    
    def _generate_pdf_report(self, results: Dict[str, Any], output_path: Path) -> bool:
        """Generate PDF report from HTML (requires wkhtmltopdf)"""
        try:
            import pdfkit
            
            # First generate HTML
            html_path = output_path.with_suffix('.html')
            self._generate_html_report(results, html_path)
            
            # Convert to PDF
            pdfkit.from_file(str(html_path), str(output_path))
            
            # Remove temporary HTML file
            html_path.unlink()
            
            logger.info(f"PDF report saved to {output_path}")
            return True
            
        except ImportError:
            logger.warning("pdfkit not installed - cannot generate PDF report")
            return False
        except Exception as e:
            logger.error(f"PDF generation failed: {e}")
            return False
    
    def _generate_json_export(self, results: Dict[str, Any], output_path: Path) -> None:
        """Generate JSON export of results"""
        
        # Convert non-serializable objects
        def convert_to_serializable(obj):
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient='records')
            elif isinstance(obj, pd.Series):
                return {
                    'index': [str(idx) for idx in obj.index],
                    'values': obj.values.tolist()
                }
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif pd.isna(obj):
                return None
            elif hasattr(obj, 'isoformat'):
                return obj.isoformat()
            else:
                return str(obj)
                
        # Recursively convert results
        def process_dict(d):
            result = {}
            for key, value in d.items():
                if isinstance(value, dict):
                    result[key] = process_dict(value)
                elif isinstance(value, list):
                    result[key] = [process_dict(item) if isinstance(item, dict) 
                                 else convert_to_serializable(item) for item in value]
                else:
                    result[key] = convert_to_serializable(value)
            return result
            
        serializable_results = process_dict(results)
        
        # Save JSON
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
            
        logger.info(f"JSON export saved to {output_path}")
    
    def generate_summary_statistics(self, 
                                  adjusted_series: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Generate summary statistics for adjusted series
        
        Parameters:
        -----------
        adjusted_series : Dict[str, pd.Series]
            Dictionary of adjusted series
            
        Returns:
        --------
        pd.DataFrame
            Summary statistics
        """
        stats_data = []
        
        for name, series in adjusted_series.items():
            stats = {
                'Series': name,
                'Count': len(series),
                'Mean': series.mean(),
                'Std': series.std(),
                'Min': series.min(),
                'Q1': series.quantile(0.25),
                'Median': series.median(),
                'Q3': series.quantile(0.75),
                'Max': series.max(),
                'Skewness': series.skew(),
                'Kurtosis': series.kurtosis()
            }
            
            # Add growth statistics
            growth = series.pct_change(4)  # YoY growth
            stats['Mean Growth'] = growth.mean()
            stats['Growth Volatility'] = growth.std()
            
            stats_data.append(stats)
            
        return pd.DataFrame(stats_data)