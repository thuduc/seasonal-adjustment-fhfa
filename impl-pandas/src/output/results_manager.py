import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging
from pathlib import Path
import json
import pickle

logger = logging.getLogger('seasonal_adjustment.results_manager')


class ResultsManager:
    """Manages storage and retrieval of seasonal adjustment results."""
    
    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.series_dir = self.output_dir / 'adjusted_series'
        self.factors_dir = self.output_dir / 'seasonal_factors'
        self.diagnostics_dir = self.output_dir / 'diagnostics'
        self.models_dir = self.output_dir / 'models'
        
        for dir_path in [self.series_dir, self.factors_dir, 
                        self.diagnostics_dir, self.models_dir]:
            dir_path.mkdir(exist_ok=True)
            
    def save_adjusted_series(self, geography_id: str, 
                           results: Dict[str, any]) -> None:
        """Store adjusted index series for a geography."""
        logger.debug(f"Saving adjusted series for {geography_id}")
        
        if 'series' not in results:
            logger.warning(f"No series data found for {geography_id}")
            return
            
        series_data = results['series']
        
        # Create DataFrame with all series
        df = pd.DataFrame({
            'nsa_index': series_data['original'],
            'sa_index': series_data['seasonally_adjusted'],
            'seasonal_factor': series_data['seasonal_factors']
        })
        
        # Add metadata
        df['geography_id'] = geography_id
        
        # Save to CSV
        filepath = self.series_dir / f"{geography_id}_adjusted.csv"
        df.to_csv(filepath)
        
        logger.info(f"Saved adjusted series to {filepath}")
        
    def save_seasonal_factors(self, geography_id: str, 
                            factors: pd.Series) -> None:
        """Store seasonal adjustment factors."""
        logger.debug(f"Saving seasonal factors for {geography_id}")
        
        # Create DataFrame with quarterly patterns
        quarterly_factors = {}
        for quarter in range(1, 5):
            quarter_mask = factors.index.quarter == quarter
            quarterly_factors[f'Q{quarter}'] = factors[quarter_mask].mean()
            
        # Save quarterly pattern
        factors_df = pd.DataFrame([quarterly_factors])
        factors_df['geography_id'] = geography_id
        
        filepath = self.factors_dir / f"{geography_id}_factors.csv"
        factors_df.to_csv(filepath, index=False)
        
        # Also save full series
        full_filepath = self.factors_dir / f"{geography_id}_factors_full.csv"
        factors.to_csv(full_filepath)
        
    def save_diagnostics(self, geography_id: str, 
                        diagnostics: Dict[str, any]) -> None:
        """Store model diagnostics and fit statistics."""
        logger.debug(f"Saving diagnostics for {geography_id}")
        
        # Flatten diagnostics for saving
        flat_diagnostics = {
            'geography_id': geography_id,
            'aic': diagnostics.get('aic'),
            'bic': diagnostics.get('bic'),
            'log_likelihood': diagnostics.get('log_likelihood'),
            'ljung_box_pvalue': diagnostics.get('ljung_box', {}).get('p_value', [None])[0],
            'ljung_box_significant': diagnostics.get('ljung_box', {}).get('significant'),
            'jarque_bera_pvalue': diagnostics.get('jarque_bera', {}).get('p_value'),
            'jarque_bera_normal': diagnostics.get('jarque_bera', {}).get('normal'),
            'adf_pvalue': diagnostics.get('adf_test', {}).get('p_value'),
            'adf_stationary': diagnostics.get('adf_test', {}).get('stationary'),
            'residual_mean': diagnostics.get('residual_stats', {}).get('mean'),
            'residual_std': diagnostics.get('residual_stats', {}).get('std'),
            'residual_skewness': diagnostics.get('residual_stats', {}).get('skewness'),
            'residual_kurtosis': diagnostics.get('residual_stats', {}).get('kurtosis')
        }
        
        # Convert numpy types to Python types for JSON serialization
        for key, value in flat_diagnostics.items():
            if isinstance(value, (np.bool_, np.bool)):
                flat_diagnostics[key] = bool(value)
            elif isinstance(value, (np.integer, np.floating)):
                flat_diagnostics[key] = float(value)
        
        # Save to JSON
        filepath = self.diagnostics_dir / f"{geography_id}_diagnostics.json"
        with open(filepath, 'w') as f:
            json.dump(flat_diagnostics, f, indent=2)
            
    def save_model(self, geography_id: str, model_info: Dict[str, any]) -> None:
        """Save fitted model object."""
        logger.debug(f"Saving model for {geography_id}")
        
        # Save model specification
        model_spec = {
            'geography_id': geography_id,
            'order': model_info.get('model_spec', {}).get('order'),
            'seasonal_order': model_info.get('model_spec', {}).get('seasonal_order')
        }
        
        spec_filepath = self.models_dir / f"{geography_id}_spec.json"
        with open(spec_filepath, 'w') as f:
            json.dump(model_spec, f, indent=2)
            
        # Save model object (if needed for future predictions)
        # Note: ARIMA models from statsmodels can be large
        # Consider saving only essential parameters
        
    def load_adjusted_series(self, geography_id: str) -> Optional[pd.DataFrame]:
        """Load adjusted series for a geography."""
        filepath = self.series_dir / f"{geography_id}_adjusted.csv"
        
        if not filepath.exists():
            logger.warning(f"No adjusted series found for {geography_id}")
            return None
            
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        return df
    
    def load_seasonal_factors(self, geography_id: str) -> Optional[pd.DataFrame]:
        """Load seasonal factors for a geography."""
        filepath = self.factors_dir / f"{geography_id}_factors.csv"
        
        if not filepath.exists():
            logger.warning(f"No seasonal factors found for {geography_id}")
            return None
            
        df = pd.read_csv(filepath)
        return df
    
    def load_diagnostics(self, geography_id: str) -> Optional[Dict]:
        """Load diagnostics for a geography."""
        filepath = self.diagnostics_dir / f"{geography_id}_diagnostics.json"
        
        if not filepath.exists():
            logger.warning(f"No diagnostics found for {geography_id}")
            return None
            
        with open(filepath, 'r') as f:
            diagnostics = json.load(f)
            
        return diagnostics
    
    def save_batch_results(self, results_list: List[Dict[str, any]]) -> None:
        """Save results from batch processing."""
        logger.info(f"Saving batch results for {len(results_list)} geographies")
        
        for result in results_list:
            if result['status'] == 'success':
                geo_id = result['geography_id']
                
                # Save all components
                self.save_adjusted_series(geo_id, result)
                
                if 'seasonal_factors' in result.get('series', {}):
                    self.save_seasonal_factors(
                        geo_id, 
                        result['series']['seasonal_factors']
                    )
                    
                if 'diagnostics' in result:
                    self.save_diagnostics(geo_id, result['diagnostics'])
                    
                if 'model_spec' in result:
                    self.save_model(geo_id, result)
                    
    def create_summary_report(self) -> pd.DataFrame:
        """Create summary report of all processed geographies."""
        logger.info("Creating summary report")
        
        summary_data = []
        
        # Scan diagnostics directory
        for diag_file in self.diagnostics_dir.glob("*_diagnostics.json"):
            geo_id = diag_file.stem.replace('_diagnostics', '')
            
            with open(diag_file, 'r') as f:
                diagnostics = json.load(f)
                
            summary_data.append({
                'geography_id': geo_id,
                'aic': diagnostics.get('aic'),
                'bic': diagnostics.get('bic'),
                'residuals_normal': diagnostics.get('jarque_bera_normal'),
                'residuals_stationary': diagnostics.get('adf_stationary'),
                'ljung_box_significant': diagnostics.get('ljung_box_significant')
            })
            
        if not summary_data:
            logger.warning("No results found for summary")
            return pd.DataFrame()
            
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('geography_id')
        
        # Save summary
        summary_path = self.output_dir / 'processing_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        
        logger.info(f"Summary report saved to {summary_path}")
        
        return summary_df
    
    def export_for_analysis(self, geography_ids: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """Export all results in a format suitable for further analysis."""
        logger.info("Exporting results for analysis")
        
        # If no geography IDs specified, get all available
        if geography_ids is None:
            geography_ids = [
                f.stem.replace('_adjusted', '') 
                for f in self.series_dir.glob("*_adjusted.csv")
            ]
            
        # Combine all adjusted series
        all_series = []
        for geo_id in geography_ids:
            series_df = self.load_adjusted_series(geo_id)
            if series_df is not None:
                all_series.append(series_df)
                
        if all_series:
            combined_series = pd.concat(all_series)
        else:
            combined_series = pd.DataFrame()
            
        # Combine all seasonal factors
        all_factors = []
        for geo_id in geography_ids:
            factors_df = self.load_seasonal_factors(geo_id)
            if factors_df is not None:
                all_factors.append(factors_df)
                
        if all_factors:
            combined_factors = pd.concat(all_factors)
        else:
            combined_factors = pd.DataFrame()
            
        # Create diagnostics summary
        all_diagnostics = []
        for geo_id in geography_ids:
            diag = self.load_diagnostics(geo_id)
            if diag is not None:
                all_diagnostics.append(diag)
                
        if all_diagnostics:
            diagnostics_df = pd.DataFrame(all_diagnostics)
        else:
            diagnostics_df = pd.DataFrame()
            
        export_data = {
            'adjusted_series': combined_series,
            'seasonal_factors': combined_factors,
            'diagnostics': diagnostics_df
        }
        
        # Save as Excel file with multiple sheets
        excel_path = self.output_dir / 'seasonal_adjustment_results.xlsx'
        with pd.ExcelWriter(excel_path) as writer:
            for sheet_name, df in export_data.items():
                if not df.empty:
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    
        logger.info(f"Results exported to {excel_path}")
        
        return export_data