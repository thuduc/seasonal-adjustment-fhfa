"""Data validation utilities"""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.outliers_influence import variance_inflation_factor
from loguru import logger


class TimeSeriesValidator:
    """Validate time series data quality and properties"""
    
    def __init__(self, min_length: int = 20, max_missing_ratio: float = 0.2):
        """
        Initialize time series validator
        
        Parameters:
        -----------
        min_length : int
            Minimum required series length
        max_missing_ratio : float
            Maximum allowed missing data ratio
        """
        self.min_length = min_length
        self.max_missing_ratio = max_missing_ratio
        
    def validate(self, series: pd.Series, check_stationarity: bool = True) -> Dict[str, Any]:
        """
        Comprehensive time series validation
        
        Parameters:
        -----------
        series : pd.Series
            Time series to validate
        check_stationarity : bool
            Whether to run stationarity tests
            
        Returns:
        --------
        Dict[str, Any]
            Validation results
        """
        results = {
            'series_name': series.name,
            'length': len(series),
            'overall_valid': True,
            'validation_errors': []
        }
        
        # Check if has datetime index
        results['has_datetime_index'] = isinstance(series.index, pd.DatetimeIndex)
        if not results['has_datetime_index']:
            results['overall_valid'] = False
            results['validation_errors'].append("Series must have DatetimeIndex")
        
        # Check frequency
        if results['has_datetime_index']:
            results['has_frequency'] = series.index.freq is not None
            results['frequency'] = str(series.index.freq) if series.index.freq else None
        else:
            results['has_frequency'] = False
            results['frequency'] = None
        
        # Check length
        results['min_length_check'] = len(series) >= self.min_length
        if not results['min_length_check']:
            results['overall_valid'] = False
            results['validation_errors'].append(f"Series too short ({len(series)} < {self.min_length})")
        
        # Check missing values
        results['missing_count'] = series.isna().sum()
        results['missing_ratio'] = series.isna().mean()
        results['missing_check'] = results['missing_ratio'] <= self.max_missing_ratio
        if not results['missing_check']:
            results['overall_valid'] = False
            results['validation_errors'].append(f"Too many missing values ({results['missing_ratio']:.2%})")
        
        # Check for outliers
        outlier_results = self.detect_outliers(series)
        results.update(outlier_results)
        
        # Check stationarity if requested
        if check_stationarity and len(series.dropna()) >= self.min_length:
            stationarity_results = self.test_stationarity(series)
            results['stationarity'] = stationarity_results
        
        return results
    
    def detect_outliers(self, series: pd.Series, method: str = 'iqr', threshold: float = 3.0) -> Dict[str, Any]:
        """
        Detect outliers in time series
        
        Parameters:
        -----------
        series : pd.Series
            Time series data
        method : str
            Detection method: 'iqr' or 'zscore'
        threshold : float
            Threshold for outlier detection
            
        Returns:
        --------
        Dict[str, Any]
            Outlier detection results
        """
        clean_series = series.dropna()
        
        if method == 'iqr':
            Q1 = clean_series.quantile(0.25)
            Q3 = clean_series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = (clean_series < lower_bound) | (clean_series > upper_bound)
        else:  # zscore
            mean = clean_series.mean()
            std = clean_series.std()
            z_scores = np.abs((clean_series - mean) / std)
            outliers = z_scores > threshold
        
        outlier_indices = list(clean_series[outliers].index)
        
        return {
            'outlier_count': outliers.sum(),
            'outlier_ratio': outliers.mean(),
            'outlier_indices': outlier_indices,
            'outlier_method': method
        }
    
    def test_stationarity(self, series: pd.Series) -> Dict[str, Any]:
        """
        Test time series stationarity
        
        Parameters:
        -----------
        series : pd.Series
            Time series data
            
        Returns:
        --------
        Dict[str, Any]
            Stationarity test results
        """
        clean_series = series.dropna()
        
        # Check if series is constant
        if len(clean_series) < 4 or clean_series.std() == 0:
            return {
                'adf_statistic': np.nan,
                'adf_pvalue': np.nan,
                'adf_critical_values': {},
                'kpss_statistic': np.nan,
                'kpss_pvalue': np.nan,
                'kpss_critical_values': {},
                'is_stationary': True,  # Constant series is technically stationary
                'constant_series': True
            }
        
        try:
            # ADF test
            adf_result = adfuller(clean_series, autolag='AIC')
            
            # KPSS test
            kpss_result = kpss(clean_series, regression='c', nlags='auto')
            
            return {
                'adf_statistic': adf_result[0],
                'adf_pvalue': adf_result[1],
                'adf_critical_values': adf_result[4],
                'kpss_statistic': kpss_result[0],
                'kpss_pvalue': kpss_result[1],
                'kpss_critical_values': kpss_result[3],
                'is_stationary': (adf_result[1] < 0.05) and (kpss_result[1] > 0.05),
                'constant_series': False
            }
        except Exception as e:
            # Handle any errors in stationarity tests
            return {
                'adf_statistic': np.nan,
                'adf_pvalue': np.nan,
                'adf_critical_values': {},
                'kpss_statistic': np.nan,
                'kpss_pvalue': np.nan,
                'kpss_critical_values': {},
                'is_stationary': None,
                'error': str(e)
            }


class PanelDataValidator:
    """Validate panel data structure and quality"""
    
    def __init__(self):
        """Initialize panel data validator"""
        pass
        
    def validate(self, 
                data: pd.DataFrame,
                entity_col: str,
                time_col: str,
                required_cols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Validate panel data structure
        
        Parameters:
        -----------
        data : pd.DataFrame
            Panel data
        entity_col : str
            Entity identifier column
        time_col : str
            Time identifier column
        required_cols : List[str], optional
            Required columns
            
        Returns:
        --------
        Dict[str, Any]
            Validation results
        """
        results = {
            'overall_valid': True,
            'validation_errors': [],
            'n_entities': 0,
            'n_periods': 0,
            'is_balanced': False
        }
        
        # Check required columns
        if entity_col not in data.columns:
            results['overall_valid'] = False
            results['validation_errors'].append(f"Entity column '{entity_col}' not found")
            
        if time_col not in data.columns:
            results['overall_valid'] = False
            results['validation_errors'].append(f"Time column '{time_col}' not found")
            
        if required_cols:
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                results['overall_valid'] = False
                for col in missing_cols:
                    results['validation_errors'].append(f"Required column '{col}' not found")
        
        if not results['overall_valid']:
            return results
            
        # Check panel structure
        entities = data[entity_col].unique()
        periods = data[time_col].unique()
        
        results['n_entities'] = len(entities)
        results['n_periods'] = len(periods)
        results['total_observations'] = len(data)
        
        # Check if balanced
        expected_obs = len(entities) * len(periods)
        results['is_balanced'] = len(data) == expected_obs
        results['missing_observations'] = expected_obs - len(data)
        
        # Check for duplicates
        duplicates = data.duplicated(subset=[entity_col, time_col])
        results['has_duplicates'] = duplicates.any()
        results['n_duplicates'] = duplicates.sum()
        
        if results['has_duplicates']:
            results['overall_valid'] = False
            results['validation_errors'].append(f"Found {results['n_duplicates']} duplicate observations")
        
        # Check numeric columns for basic statistics
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if entity_col in numeric_cols:
            numeric_cols.remove(entity_col)
        if time_col in numeric_cols:
            numeric_cols.remove(time_col)
            
        results['numeric_columns'] = numeric_cols
        
        # Check for multicollinearity if enough numeric columns
        if len(numeric_cols) >= 2:
            multicollinearity = self.check_multicollinearity(data[numeric_cols])
            results['multicollinearity'] = multicollinearity
        
        return results
    
    def check_multicollinearity(self, data: pd.DataFrame, vif_threshold: float = 10.0) -> Dict[str, Any]:
        """
        Check for multicollinearity using VIF
        
        Parameters:
        -----------
        data : pd.DataFrame
            Numeric data
        vif_threshold : float
            VIF threshold for flagging multicollinearity
            
        Returns:
        --------
        Dict[str, Any]
            Multicollinearity test results
        """
        # Drop any rows with missing values
        clean_data = data.dropna()
        
        if len(clean_data) < len(data.columns) * 2:
            return {
                'vif_calculated': False,
                'reason': 'Insufficient data for VIF calculation'
            }
        
        vif_data = pd.DataFrame()
        vif_data["Variable"] = data.columns
        
        try:
            vif_data["VIF"] = [variance_inflation_factor(clean_data.values, i) 
                              for i in range(len(data.columns))]
            
            high_vif = vif_data[vif_data["VIF"] > vif_threshold]
            
            return {
                'vif_calculated': True,
                'vif_data': vif_data.to_dict(),
                'max_vif': vif_data["VIF"].max(),
                'high_vif_vars': high_vif["Variable"].tolist() if len(high_vif) > 0 else None,
                'has_multicollinearity': len(high_vif) > 0
            }
        except Exception as e:
            return {
                'vif_calculated': False,
                'reason': str(e)
            }


class DataQualityChecker:
    """Comprehensive data quality checking"""
    
    def __init__(self):
        """Initialize data quality checker"""
        self.ts_validator = TimeSeriesValidator()
        self.panel_validator = PanelDataValidator()
        
    def check_hpi_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Check HPI data quality
        
        Parameters:
        -----------
        data : pd.DataFrame
            HPI data with datetime index
            
        Returns:
        --------
        Dict[str, Any]
            Quality check results
        """
        results = {
            'series_validations': {},
            'overall_quality': True
        }
        
        # Validate each HPI series
        for col in data.columns:
            if col.startswith('hpi_'):
                validation = self.ts_validator.validate(data[col])
                results['series_validations'][col] = validation
                
                if not validation['overall_valid']:
                    results['overall_quality'] = False
        
        # Check correlations between series
        if len(data.columns) > 1:
            corr_matrix = data.corr()
            results['average_correlation'] = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
            results['max_correlation'] = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].max()
        
        return results