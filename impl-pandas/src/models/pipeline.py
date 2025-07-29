import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path

from .x13_engine import X13ARIMAEngine
from .model_selection import ModelSelector

logger = logging.getLogger('seasonal_adjustment.pipeline')


class SeasonalAdjustmentPipeline:
    """End-to-end pipeline for seasonal adjustment processing."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.engine = X13ARIMAEngine(self.config.get('engine_config', {}))
        self.selector = ModelSelector(**self.config.get('selector_config', {}))
        self.results_cache = {}
        
    def process_geography(self, geography_id: str, 
                         hpi_data: pd.DataFrame,
                         auto_select_model: bool = True) -> Dict[str, any]:
        """Complete seasonal adjustment for one geography."""
        logger.info(f"Processing geography: {geography_id}")
        
        # Filter data for this geography
        geo_data = hpi_data[hpi_data['geography_id'] == geography_id].copy()
        
        if len(geo_data) < 20:
            logger.warning(f"Insufficient data for {geography_id}: {len(geo_data)} observations")
            return {
                'geography_id': geography_id,
                'status': 'failed',
                'error': 'Insufficient data'
            }
            
        try:
            # Sort by period and set as index
            geo_data = geo_data.sort_values('period')
            geo_data.set_index('period', inplace=True)
            
            # Get the HPI series
            hpi_series = geo_data['nsa_index']
            # Pre-process the series
            pre_adjusted = self.engine.pre_adjustment(hpi_series)
            
            # Detect and handle outliers
            outlier_df = self.engine.detect_outliers(pre_adjusted)
            clean_series = outlier_df['adjusted_value']
            
            # Select best model if auto-selection is enabled
            if auto_select_model:
                model_info = self.selector.select_best_model(
                    clean_series, 
                    criterion='aic',
                    validate=True
                )
                fitted_model = model_info['model']
            else:
                # Use default ARIMA(1,1,1)(1,1,1)4
                order = (1, 1, 1)
                seasonal_order = (1, 1, 1, 4)
                try:
                    fitted_model = self.engine.fit_arima_model(
                        clean_series, order, seasonal_order
                    )
                    model_info = {
                        'order': order,
                        'seasonal_order': seasonal_order,
                        'model': fitted_model
                    }
                except Exception as e:
                    logger.warning(f"Failed to fit default ARIMA model: {e}")
                    # Use simple seasonal decomposition as fallback
                    fitted_model = None
                    model_info = {
                        'order': order,
                        'seasonal_order': seasonal_order,
                        'model': None
                    }
                
            # Extract seasonal factors
            seasonal_factors = self.engine.extract_seasonal_factors(
                clean_series, fitted_model
            )
            
            # Calculate seasonally adjusted series
            sa_series = self.engine.calculate_adjusted_series(
                hpi_series, seasonal_factors
            )
            
            # Run diagnostics
            residuals = fitted_model.resid
            diagnostics = self.engine.run_diagnostics(fitted_model, residuals)
            
            # Test stability
            stability_results = self.engine.stability_test(hpi_series)
            
            # Compile results
            results = {
                'geography_id': geography_id,
                'status': 'success',
                'model_spec': {
                    'order': model_info.get('order'),
                    'seasonal_order': model_info.get('seasonal_order')
                },
                'series': {
                    'original': hpi_series,
                    'seasonally_adjusted': sa_series,
                    'seasonal_factors': seasonal_factors,
                    'outliers': outlier_df
                },
                'diagnostics': diagnostics,
                'stability': stability_results,
                'validation': model_info.get('validation', {}),
                'cross_validation': model_info.get('cross_validation', {})
            }
            
            # Cache results
            self.results_cache[geography_id] = results
            
            logger.info(f"Successfully processed {geography_id}")
            return results
            
        except Exception as e:
            logger.error(f"Error processing {geography_id}: {str(e)}")
            return {
                'geography_id': geography_id,
                'status': 'failed',
                'error': str(e)
            }
    
    def batch_process(self, geography_ids: List[str], 
                     hpi_data: pd.DataFrame,
                     n_jobs: Optional[int] = None,
                     auto_select_model: bool = True) -> pd.DataFrame:
        """Process multiple geographies in parallel."""
        logger.info(f"Starting batch processing for {len(geography_ids)} geographies")
        
        if n_jobs is None:
            n_jobs = min(mp.cpu_count() - 1, 4)
            
        results = []
        
        # Process in parallel
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Submit all tasks
            future_to_geo = {
                executor.submit(
                    self._process_single_geography,
                    geo_id, hpi_data, auto_select_model
                ): geo_id 
                for geo_id in geography_ids
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_geo):
                geo_id = future_to_geo[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Completed {geo_id}: {result['status']}")
                except Exception as e:
                    logger.error(f"Failed to process {geo_id}: {str(e)}")
                    results.append({
                        'geography_id': geo_id,
                        'status': 'failed',
                        'error': str(e)
                    })
                    
        # Create summary DataFrame
        summary_data = []
        for result in results:
            if result['status'] == 'success':
                summary_data.append({
                    'geography_id': result['geography_id'],
                    'status': result['status'],
                    'model_order': str(result['model_spec']['order']),
                    'seasonal_order': str(result['model_spec']['seasonal_order']),
                    'aic': result['diagnostics']['aic'],
                    'bic': result['diagnostics']['bic'],
                    'stable': result['stability']['stable'],
                    'validation_rmse': result.get('validation', {}).get('rmse', np.nan)
                })
            else:
                summary_data.append({
                    'geography_id': result['geography_id'],
                    'status': result['status'],
                    'error': result.get('error', 'Unknown error')
                })
                
        summary_df = pd.DataFrame(summary_data)
        
        logger.info(f"Batch processing complete. Success rate: "
                   f"{(summary_df['status'] == 'success').mean():.1%}")
        
        return summary_df
    
    def _process_single_geography(self, geography_id: str, 
                                 hpi_data: pd.DataFrame,
                                 auto_select_model: bool) -> Dict:
        """Helper method for parallel processing."""
        # Create new pipeline instance for each process
        pipeline = SeasonalAdjustmentPipeline(self.config)
        return pipeline.process_geography(geography_id, hpi_data, auto_select_model)
    
    def get_adjusted_series(self, geography_id: str) -> Optional[pd.DataFrame]:
        """Retrieve adjusted series for a geography."""
        if geography_id not in self.results_cache:
            logger.warning(f"No results found for {geography_id}")
            return None
            
        result = self.results_cache[geography_id]
        if result['status'] != 'success':
            return None
            
        # Create DataFrame with all series
        series_df = pd.DataFrame({
            'nsa_index': result['series']['original'],
            'sa_index': result['series']['seasonally_adjusted'],
            'seasonal_factor': result['series']['seasonal_factors']
        })
        
        return series_df
    
    def export_results(self, output_dir: Union[str, Path]) -> None:
        """Export all results to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export adjusted series
        all_series = []
        for geo_id, result in self.results_cache.items():
            if result['status'] == 'success':
                series_df = self.get_adjusted_series(geo_id)
                series_df['geography_id'] = geo_id
                all_series.append(series_df)
                
        if all_series:
            combined_series = pd.concat(all_series)
            combined_series.to_csv(output_dir / 'adjusted_series.csv')
            
        # Export diagnostics
        diagnostics_data = []
        for geo_id, result in self.results_cache.items():
            if result['status'] == 'success':
                diag = result['diagnostics']
                diagnostics_data.append({
                    'geography_id': geo_id,
                    'aic': diag['aic'],
                    'bic': diag['bic'],
                    'ljung_box_pvalue': diag['ljung_box']['p_value'].min(),
                    'residual_normality': diag['jarque_bera']['normal'],
                    'stable': result['stability']['stable']
                })
                
        if diagnostics_data:
            diag_df = pd.DataFrame(diagnostics_data)
            diag_df.to_csv(output_dir / 'diagnostics.csv', index=False)
            
        logger.info(f"Results exported to {output_dir}")