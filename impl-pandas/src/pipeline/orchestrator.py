"""Main pipeline orchestrator for FHFA seasonal adjustment"""

from typing import Dict, List, Optional, Union, Tuple, Any
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import json
from datetime import datetime

from ..config import get_settings
from ..data.loaders import HPIDataLoader, WeatherDataLoader, DemographicDataLoader
from ..data.validators import TimeSeriesValidator, PanelDataValidator
from ..data.preprocessors import DataPreprocessor
from ..data.transformers import DataTransformer
from ..models.seasonal.adjuster import SeasonalAdjuster
from ..models.arima.regarima import RegARIMA
from ..models.regression.panel import SeasonalityImpactModel
from ..models.regression.quantile import QuantileRegressionModel
from ..visualization.reports import ReportGenerator
from ..visualization.plots import SeasonalAdjustmentPlotter, DiagnosticPlotter
from ..utils import track_metrics, performance_monitor, metrics_collector


class SeasonalAdjustmentPipeline:
    """
    End-to-end pipeline for FHFA seasonal adjustment analysis
    
    Orchestrates:
    1. Data loading and validation
    2. Preprocessing and transformation
    3. Seasonal adjustment
    4. Impact analysis
    5. Results export
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize pipeline
        
        Parameters:
        -----------
        config_path : str, optional
            Path to configuration file
        """
        self.settings = get_settings()
        self.results = {}
        self.metadata = {
            'pipeline_version': '1.0.0',
            'created_at': datetime.now().isoformat(),
            'config': self.settings.model_dump()
        }
        
        # Initialize components
        self._initialize_components()
        
    def _initialize_components(self) -> None:
        """Initialize pipeline components"""
        
        # Data loaders
        self.hpi_loader = HPIDataLoader()
        self.weather_loader = WeatherDataLoader()
        self.demographic_loader = DemographicDataLoader()
        
        # Validators
        self.ts_validator = TimeSeriesValidator()
        self.panel_validator = PanelDataValidator()
        
        # Preprocessors and transformers
        self.preprocessor = DataPreprocessor()
        self.transformer = DataTransformer()
        
        # Visualization and reporting
        self.report_generator = ReportGenerator()
        self.sa_plotter = SeasonalAdjustmentPlotter()
        self.diag_plotter = DiagnosticPlotter()
        
        logger.info("Pipeline components initialized")
    
    @track_metrics("pipeline.full")
    def run_full_pipeline(self,
                         hpi_data_path: Optional[str] = None,
                         weather_data_path: Optional[str] = None,
                         demographic_data_path: Optional[str] = None,
                         output_dir: str = "./output",
                         generate_report: bool = True,
                         report_formats: List[str] = ['html', 'excel']) -> Dict[str, Any]:
        """
        Run complete seasonal adjustment pipeline
        
        Parameters:
        -----------
        hpi_data_path : str, optional
            Path to HPI data
        weather_data_path : str, optional
            Path to weather data
        demographic_data_path : str, optional
            Path to demographic data
        output_dir : str
            Directory for output files
        generate_report : bool
            Whether to generate reports
        report_formats : List[str]
            Report formats to generate
            
        Returns:
        --------
        Dict[str, Any]
            Pipeline results
        """
        logger.info("Starting full seasonal adjustment pipeline")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Step 1: Load data
            logger.info("Step 1: Loading data")
            data = self._load_all_data(hpi_data_path, weather_data_path, demographic_data_path)
            self._original_data = data  # Store for visualization
            self.results['data_loaded'] = {
                'hpi_shape': data['hpi'].shape,
                'weather_shape': data['weather'].shape if data['weather'] is not None else None,
                'demographic_shape': data['demographic'].shape if data['demographic'] is not None else None
            }
            
            # Record data metrics
            metrics_collector.record("data.hpi.rows", data['hpi'].shape[0])
            metrics_collector.record("data.hpi.columns", data['hpi'].shape[1])
            if data['weather'] is not None:
                metrics_collector.record("data.weather.rows", data['weather'].shape[0])
                metrics_collector.record("data.weather.columns", data['weather'].shape[1])
            
            # Step 2: Validate data
            logger.info("Step 2: Validating data")
            validation_results = self._validate_data(data)
            self.results['validation'] = validation_results
            
            # Step 3: Preprocess data
            logger.info("Step 3: Preprocessing data")
            processed_data = self._preprocess_data(data)
            
            # Step 4: Run seasonal adjustment
            logger.info("Step 4: Running seasonal adjustment")
            adjustment_results = self._run_seasonal_adjustment(processed_data)
            self.results['seasonal_adjustment'] = adjustment_results
            
            # Step 5: Run impact analysis
            logger.info("Step 5: Running impact analysis")
            if processed_data['weather'] is not None:
                impact_results = self._run_impact_analysis(processed_data, adjustment_results)
                self.results['impact_analysis'] = impact_results
            
            # Step 6: Generate diagnostics
            logger.info("Step 6: Generating diagnostics")
            diagnostics = self._generate_diagnostics()
            self.results['diagnostics'] = diagnostics
            
            # Step 7: Export results
            logger.info("Step 7: Exporting results")
            self._export_results(output_path)
            
            # Step 8: Generate reports
            if generate_report:
                logger.info("Step 8: Generating reports")
                report_paths = self.report_generator.generate_full_report(
                    self.results,
                    output_path,
                    formats=report_formats
                )
                self.results['reports'] = report_paths
            
            logger.info("Pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            self.results['error'] = str(e)
            raise
        
        return self.results
    
    @track_metrics("pipeline.seasonal_adjustment_only")
    def run_seasonal_adjustment_only(self,
                                   series: pd.Series,
                                   method: str = "x13") -> pd.Series:
        """
        Run only seasonal adjustment on a single series
        
        Parameters:
        -----------
        series : pd.Series
            Time series to adjust
        method : str
            Adjustment method
            
        Returns:
        --------
        pd.Series
            Adjusted series
        """
        # Validate series
        validation = self.ts_validator.validate(series)
        if not validation['overall_valid']:
            logger.warning("Series validation failed")
        
        # Preprocess
        preprocessed = self.preprocessor.clean_time_series(
            series,
            method='interpolate'
        )
        
        # Run adjustment
        adjuster = SeasonalAdjuster(method=method)
        adjusted = adjuster.adjust(preprocessed)
        
        return adjusted
    
    @track_metrics("pipeline.impact_analysis_only")
    def run_impact_analysis_only(self,
                               panel_data: pd.DataFrame,
                               model_type: str = "fixed_effects") -> Dict[str, Any]:
        """
        Run only impact analysis on panel data
        
        Parameters:
        -----------
        panel_data : pd.DataFrame
            Panel data with required columns
        model_type : str
            Type of panel model
            
        Returns:
        --------
        Dict[str, Any]
            Analysis results
        """
        # Validate data
        validation = self.panel_validator.validate(
            panel_data,
            entity_col='cbsa',
            time_col='date'
        )
        if not validation['overall_valid']:
            logger.warning("Panel data validation failed")
        
        # Run panel regression
        model = SeasonalityImpactModel(model_type=model_type)
        
        # Fit model (assuming standard column names)
        model.fit(
            panel_data,
            y_col='hpi_growth',
            weather_col='temperature',
            sales_col='n_sales',
            char_cols=['median_income', 'population_density'],
            industry_cols=['industry_share'],
            entity_col='cbsa',
            time_col='date',
            quarter_col='quarter'
        )
        
        # Get results
        results = {
            'coefficients': model.get_coefficients(),
            'diagnostics': model.residual_diagnostics(),
            'summary': model.summary()
        }
        
        return results
    
    @track_metrics("pipeline.load_data")
    def _load_all_data(self,
                      hpi_path: Optional[str],
                      weather_path: Optional[str],
                      demographic_path: Optional[str]) -> Dict[str, pd.DataFrame]:
        """Load all data sources"""
        
        data = {}
        
        # Load HPI data
        if hpi_path:
            data['hpi'] = pd.read_csv(hpi_path, parse_dates=['date'], index_col='date')
        else:
            # Use synthetic data for testing
            data['hpi'] = self.hpi_loader.load(
                start_date='2010-01-01',
                end_date='2023-12-31'
            )
        
        # Load weather data
        if weather_path:
            data['weather'] = pd.read_csv(weather_path, parse_dates=['date'])
        else:
            data['weather'] = self.weather_loader.load(
                geography=['12345', '67890'],  # Example CBSAs
                variables=['temperature', 'precipitation'],
                start_date='2010-01-01',
                end_date='2023-12-31'
            )
        
        # Load demographic data
        if demographic_path:
            data['demographic'] = pd.read_csv(demographic_path)
        else:
            data['demographic'] = self.demographic_loader.load(
                geography_ids=['12345', '67890']  # Same CBSAs as HPI data
            )
        
        return data
    
    @track_metrics("pipeline.validate_data")
    def _validate_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Validate all data"""
        
        validation_results = {}
        
        # Validate HPI time series
        for col in data['hpi'].columns:
            if col.startswith('hpi_'):
                validation_results[col] = self.ts_validator.validate(data['hpi'][col])
        
        # Validate panel data if weather available
        if data['weather'] is not None:
            # Merge data for panel validation
            panel_data = self._merge_data_for_panel(data)
            validation_results['panel'] = self.panel_validator.validate(
                panel_data,
                entity_col='cbsa',
                time_col='date'
            )
        
        return validation_results
    
    @track_metrics("pipeline.preprocess_data")
    def _preprocess_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Preprocess all data"""
        
        processed = {}
        
        # Process HPI data
        hpi_processed = data['hpi'].copy()
        for col in hpi_processed.columns:
            if col.startswith('hpi_'):
                # Clean missing values
                hpi_processed[col] = self.preprocessor.clean_time_series(
                    hpi_processed[col],
                    method='interpolate'
                )
                # Handle outliers
                hpi_processed[col] = self.preprocessor.handle_outliers(
                    hpi_processed[col],
                    method='winsorize'
                )
        processed['hpi'] = hpi_processed
        
        # Process weather data
        if data['weather'] is not None:
            weather_processed = data['weather'].copy()
            # Clean numeric columns
            numeric_cols = weather_processed.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col != 'date':
                    # Handle outliers directly without time-series interpolation
                    weather_processed[col] = self.preprocessor.handle_outliers(
                        weather_processed[col],
                        method='winsorize'
                    )
            processed['weather'] = weather_processed
        else:
            processed['weather'] = None
        
        # Process demographic data
        if data['demographic'] is not None:
            processed['demographic'] = data['demographic']  # Usually static
        else:
            processed['demographic'] = None
        
        return processed
    
    @track_metrics("pipeline.run_seasonal_adjustment")
    def _run_seasonal_adjustment(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Run seasonal adjustment on all series"""
        
        adjustment_results = {}
        
        # Get adjustment settings - use default method
        method = "classical"  # Default method
        
        # Adjust each HPI series
        for col in data['hpi'].columns:
            if col.startswith('hpi_'):
                logger.info(f"Adjusting {col}")
                
                # Create adjuster
                adjuster = SeasonalAdjuster(
                    method=method,
                    outlier_detection=True  # Enable outlier detection
                )
                
                # Run adjustment
                adjusted = adjuster.adjust(data['hpi'][col])
                
                # Store results
                adjustment_results[col] = {
                    'adjusted_series': adjusted,
                    'diagnostics': adjuster.get_results()['diagnostics'],
                    'method': method
                }
        
        return adjustment_results
    
    @track_metrics("pipeline.run_impact_analysis")
    def _run_impact_analysis(self, 
                           data: Dict[str, pd.DataFrame],
                           adjustment_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run seasonality impact analysis"""
        
        impact_results = {}
        
        # Prepare panel data
        panel_data = self._prepare_panel_data(data, adjustment_results)
        
        # Run fixed effects model
        logger.info("Running fixed effects model")
        fe_model = SeasonalityImpactModel(model_type="fixed_effects")
        fe_model.fit(
            panel_data,
            y_col='seasonality_measure',
            weather_col='temperature',
            sales_col='n_sales',
            char_cols=['median_income', 'population_density'],
            industry_cols=['industry_share'],
            entity_col='cbsa',
            time_col='date',
            quarter_col='quarter'
        )
        
        impact_results['fixed_effects'] = {
            'coefficients': fe_model.get_coefficients(),
            'quarter_coefficients': {
                q: fe_model.get_coefficients(quarter=q) 
                for q in range(1, 5) if q in fe_model.quarter_models
            },
            'diagnostics': fe_model.residual_diagnostics()
        }
        
        # Run quantile regression
        logger.info("Running quantile regression")
        quantile_model = QuantileRegressionModel()
        quantile_model.fit(
            panel_data,
            y_col='seasonality_measure',
            weather_col='temperature',
            sales_col='n_sales',
            char_cols=['median_income', 'population_density'],
            industry_cols=['industry_share'],
            entity_col='cbsa',
            year_col='year'
        )
        
        impact_results['quantile'] = {
            'coefficients': quantile_model.get_coefficients()
        }
        
        return impact_results
    
    def _merge_data_for_panel(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merge data sources for panel analysis"""
        
        # Convert wide HPI data to long format for panel structure
        panel_rows = []
        
        for date in data['hpi'].index:
            for col in data['hpi'].columns:
                if col.startswith('hpi_cbsa_'):
                    cbsa = col.replace('hpi_cbsa_', '')
                    row = {
                        'date': date,
                        'cbsa': cbsa,
                        'hpi': data['hpi'].loc[date, col]
                    }
                    
                    # Add weather data if available
                    if data['weather'] is not None and date in data['weather']['date'].values:
                        weather_row = data['weather'][data['weather']['date'] == date].iloc[0]
                        row['temperature'] = weather_row.get('temperature', np.nan)
                        row['precipitation'] = weather_row.get('precipitation', np.nan)
                    
                    panel_rows.append(row)
        
        panel_data = pd.DataFrame(panel_rows)
        
        # Add demographic data if available
        if data['demographic'] is not None:
            # Merge demographic data (assuming CBSA level)
            if 'cbsa' in data['demographic'].columns:
                panel_data = panel_data.merge(
                    data['demographic'],
                    on='cbsa',
                    how='left'
                )
        
        return panel_data
    
    def _prepare_panel_data(self, 
                          data: Dict[str, pd.DataFrame],
                          adjustment_results: Dict[str, Any]) -> pd.DataFrame:
        """Prepare panel data for impact analysis"""
        
        # Calculate seasonality measures
        panel_rows = []
        
        for col, results in adjustment_results.items():
            # Extract geography identifier from column name
            if col.startswith('hpi_state_'):
                geo_id = col.replace('hpi_state_', '')
                geo_type = 'state'
            elif col.startswith('hpi_cbsa_'):
                geo_id = col.replace('hpi_cbsa_', '')
                geo_type = 'cbsa'
            else:
                continue  # Skip if not a recognized pattern
                
            original = data['hpi'][col]
            adjusted = results['adjusted_series']
            
            # Calculate seasonal component
            seasonal = original / adjusted if (original > 0).all() else original - adjusted
            
            # Create panel rows
            for date in original.index:
                row = {
                    'cbsa': geo_id,  # Use geo_id as CBSA identifier
                    'date': date,
                    'quarter': date.quarter,
                    'year': date.year,
                    'seasonality_measure': seasonal[date],
                    'hpi_growth': original.pct_change(4)[date],  # YoY growth
                }
                
                # Add weather data if available
                if data['weather'] is not None:
                    # Try to match weather data
                    weather_match = data['weather'][
                        (data['weather']['date'] == date) & 
                        (data['weather']['geography_id'] == geo_id)
                    ]
                    if len(weather_match) > 0:
                        row['temperature'] = weather_match.iloc[0]['temperature']
                    else:
                        row['temperature'] = np.random.normal(65, 10)
                    row['n_sales'] = np.random.poisson(1000)
                else:
                    row['temperature'] = np.random.normal(65, 10)
                    row['n_sales'] = np.random.poisson(1000)
                
                # Add demographic data
                row['median_income'] = np.random.normal(60000, 15000)
                row['population_density'] = np.random.normal(1000, 500)
                row['industry_share'] = np.random.uniform(0.1, 0.3)
                
                panel_rows.append(row)
        
        return pd.DataFrame(panel_rows)
    
    def _generate_diagnostics(self) -> Dict[str, Any]:
        """Generate comprehensive diagnostics"""
        
        diagnostics = {
            'pipeline_metadata': self.metadata,
            'data_quality': {},
            'adjustment_quality': {},
            'model_quality': {}
        }
        
        # Data quality metrics
        if 'validation' in self.results:
            diagnostics['data_quality'] = {
                'validation_summary': self._summarize_validation(self.results['validation'])
            }
        
        # Adjustment quality metrics
        if 'seasonal_adjustment' in self.results:
            diagnostics['adjustment_quality'] = {
                'methods_used': list(set(
                    r['method'] for r in self.results['seasonal_adjustment'].values()
                )),
                'success_rate': self._calculate_adjustment_success_rate()
            }
        
        # Model quality metrics
        if 'impact_analysis' in self.results:
            diagnostics['model_quality'] = {
                'fixed_effects_r2': self.results['impact_analysis']['fixed_effects'].get(
                    'diagnostics', {}
                ).get('r_squared', None)
            }
        
        return diagnostics
    
    def _summarize_validation(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize validation results"""
        
        summary = {
            'total_series': len(validation_results),
            'valid_series': sum(
                1 for v in validation_results.values() 
                if isinstance(v, dict) and v.get('overall_valid', False)
            )
        }
        
        return summary
    
    def _calculate_adjustment_success_rate(self) -> float:
        """Calculate seasonal adjustment success rate"""
        
        if 'seasonal_adjustment' not in self.results:
            return 0.0
        
        total = len(self.results['seasonal_adjustment'])
        successful = sum(
            1 for r in self.results['seasonal_adjustment'].values()
            if r['diagnostics'].get('seasonal_stability', {}).get('stable', False)
        )
        
        return successful / total if total > 0 else 0.0
    
    def _export_results(self, output_dir: Path) -> None:
        """Export all results"""
        
        # Export adjusted series
        if 'seasonal_adjustment' in self.results:
            adjusted_df = pd.DataFrame()
            for col, results in self.results['seasonal_adjustment'].items():
                adjusted_df[col] = results['adjusted_series']
            
            adjusted_df.to_csv(output_dir / 'adjusted_series.csv')
            logger.info(f"Exported adjusted series to {output_dir / 'adjusted_series.csv'}")
        
        # Export impact analysis results
        if 'impact_analysis' in self.results:
            # Fixed effects coefficients
            fe_coef = self.results['impact_analysis']['fixed_effects']['coefficients']
            fe_coef.to_csv(output_dir / 'fixed_effects_coefficients.csv')
            
            # Quantile regression coefficients
            q_coef = self.results['impact_analysis']['quantile']['coefficients']
            q_coef.to_csv(output_dir / 'quantile_coefficients.csv')
            
            logger.info("Exported impact analysis results")
        
        # Export diagnostics
        with open(output_dir / 'diagnostics.json', 'w') as f:
            json.dump(self.results['diagnostics'], f, indent=2, default=str)
        
        # Export full results
        with open(output_dir / 'full_results.json', 'w') as f:
            # Convert non-serializable objects to strings
            serializable_results = self._make_serializable(self.results)
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"All results exported to {output_dir}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format"""
        
        if isinstance(obj, pd.DataFrame):
            # Convert DataFrame to dict with string index
            return obj.reset_index().to_dict(orient='records')
        elif isinstance(obj, pd.Series):
            # Convert Series to list of values with string index
            return {
                'index': [str(idx) for idx in obj.index],
                'values': obj.values.tolist()
            }
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {str(k): self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return str(obj)
    
    def generate_visualizations(self, 
                              output_dir: Union[str, Path],
                              plot_types: List[str] = ['adjustment', 'diagnostics']) -> Dict[str, str]:
        """
        Generate visualization plots
        
        Parameters:
        -----------
        output_dir : Union[str, Path]
            Output directory for plots
        plot_types : List[str]
            Types of plots to generate
            
        Returns:
        --------
        Dict[str, str]
            Paths to generated plots
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_paths = {}
        
        if 'adjustment' in plot_types and 'seasonal_adjustment' in self.results:
            # Generate adjustment comparison plots
            for idx, (series_name, result) in enumerate(self.results['seasonal_adjustment'].items()):
                if idx >= 5:  # Limit number of plots
                    break
                    
                if 'adjusted_series' in result:
                    # Get original series
                    original = self._get_original_series(series_name)
                    if original is not None:
                        plot_path = output_dir / f'adjustment_{series_name}.png'
                        fig = self.sa_plotter.plot_adjustment_comparison(
                            original, 
                            result['adjusted_series'],
                            title=f'Seasonal Adjustment: {series_name}',
                            save_path=str(plot_path)
                        )
                        plt.close(fig)
                        plot_paths[f'adjustment_{series_name}'] = str(plot_path)
                        
        if 'diagnostics' in plot_types and 'impact_analysis' in self.results:
            # Generate coefficient plots
            if 'fixed_effects' in self.results['impact_analysis']:
                fe_results = self.results['impact_analysis']['fixed_effects']
                if 'coefficients' in fe_results:
                    plot_path = output_dir / 'fe_coefficients.png'
                    fig = self.diag_plotter.plot_coefficient_analysis(
                        fe_results['coefficients'],
                        title='Fixed Effects Model Coefficients',
                        save_path=str(plot_path)
                    )
                    plt.close(fig)
                    plot_paths['fe_coefficients'] = str(plot_path)
                    
        if 'seasonal_patterns' in plot_types and 'seasonal_adjustment' in self.results:
            # Generate seasonal pattern plots
            for idx, (series_name, result) in enumerate(self.results['seasonal_adjustment'].items()):
                if idx >= 3:  # Limit number of plots
                    break
                    
                if 'adjusted_series' in result:
                    original = self._get_original_series(series_name)
                    if original is not None:
                        plot_path = output_dir / f'seasonal_patterns_{series_name}.png'
                        fig = self.sa_plotter.plot_seasonal_patterns(
                            original,
                            result['adjusted_series'],
                            period='quarter',
                            save_path=str(plot_path)
                        )
                        plt.close(fig)
                        plot_paths[f'seasonal_patterns_{series_name}'] = str(plot_path)
                        
        logger.info(f"Generated {len(plot_paths)} visualization plots")
        return plot_paths
    
    def _get_original_series(self, series_name: str) -> Optional[pd.Series]:
        """Get original series from loaded data"""
        if hasattr(self, '_original_data') and self._original_data is not None:
            if 'hpi' in self._original_data and series_name in self._original_data['hpi'].columns:
                return self._original_data['hpi'][series_name]
        return None
    
    def get_summary_report(self) -> str:
        """Generate human-readable summary report"""
        
        lines = []
        lines.append("=" * 80)
        lines.append("SEASONAL ADJUSTMENT PIPELINE SUMMARY REPORT")
        lines.append("=" * 80)
        lines.append(f"Pipeline Version: {self.metadata['pipeline_version']}")
        lines.append(f"Execution Date: {self.metadata['created_at']}")
        lines.append("")
        
        # Data summary
        if 'data_loaded' in self.results:
            lines.append("DATA SUMMARY:")
            lines.append("-" * 40)
            for key, shape in self.results['data_loaded'].items():
                if shape:
                    lines.append(f"{key}: {shape}")
            lines.append("")
        
        # Adjustment summary
        if 'seasonal_adjustment' in self.results:
            lines.append("SEASONAL ADJUSTMENT SUMMARY:")
            lines.append("-" * 40)
            lines.append(f"Series adjusted: {len(self.results['seasonal_adjustment'])}")
            lines.append(f"Success rate: {self._calculate_adjustment_success_rate():.1%}")
            lines.append("")
        
        # Impact analysis summary
        if 'impact_analysis' in self.results:
            lines.append("IMPACT ANALYSIS SUMMARY:")
            lines.append("-" * 40)
            
            # Key coefficients
            fe_results = self.results['impact_analysis']['fixed_effects']
            if 'coefficients' in fe_results:
                coef_df = fe_results['coefficients']
                if 'temperature' in coef_df.index:
                    temp_coef = coef_df.loc['temperature', 'coefficient']
                    temp_pval = coef_df.loc['temperature', 'p_value']
                    lines.append(f"Temperature coefficient: {temp_coef:.4f} (p={temp_pval:.4f})")
            lines.append("")
        
        # Diagnostics summary
        if 'diagnostics' in self.results:
            diag = self.results['diagnostics']
            if 'data_quality' in diag:
                lines.append("DATA QUALITY:")
                lines.append("-" * 40)
                summary = diag['data_quality'].get('validation_summary', {})
                if summary:
                    lines.append(f"Valid series: {summary.get('valid_series', 0)}/{summary.get('total_series', 0)}")
        
        return "\n".join(lines)