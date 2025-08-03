"""Wrapper for X-13ARIMA-SEATS seasonal adjustment"""

from typing import Dict, Optional, Union
import pandas as pd
import numpy as np
from loguru import logger
import tempfile
import os

try:
    from statsmodels.tsa.x13 import x13_arima_analysis
    X13_AVAILABLE = True
except ImportError:
    X13_AVAILABLE = False
    logger.warning("X-13ARIMA-SEATS not available. Using fallback methods.")


class X13ARIMAWrapper:
    """
    Wrapper for X-13ARIMA-SEATS seasonal adjustment
    
    Note: Requires X-13ARIMA-SEATS binary to be installed
    """
    
    def __init__(self, x13_path: Optional[str] = None):
        """
        Initialize X-13ARIMA-SEATS wrapper
        
        Parameters:
        -----------
        x13_path : str, optional
            Path to X-13ARIMA-SEATS executable
        """
        self.x13_path = x13_path or os.environ.get("X13PATH")
        self.results = None
        
        if not X13_AVAILABLE:
            logger.warning("statsmodels X-13 interface not available")
    
    def seasonal_adjust(self,
                       series: pd.Series,
                       freq: int = 4,
                       outlier: bool = True,
                       trading: bool = False,
                       forecast_years: int = 1,
                       **kwargs) -> pd.Series:
        """
        Perform seasonal adjustment using X-13ARIMA-SEATS
        
        Parameters:
        -----------
        series : pd.Series
            Time series to adjust
        freq : int
            Frequency (4 for quarterly, 12 for monthly)
        outlier : bool
            Whether to perform outlier detection
        trading : bool
            Whether to adjust for trading days
        forecast_years : int
            Number of years to forecast
        **kwargs : dict
            Additional X-13 options
            
        Returns:
        --------
        pd.Series
            Seasonally adjusted series
        """
        if not X13_AVAILABLE:
            logger.warning("Using fallback seasonal adjustment method")
            return self._fallback_seasonal_adjust(series, freq)
        
        logger.info(f"Running X-13ARIMA-SEATS seasonal adjustment")
        
        try:
            # Ensure series has frequency
            if series.index.freq is None:
                if freq == 4:
                    series.index.freq = 'Q'
                elif freq == 12:
                    series.index.freq = 'M'
            
            # Run X-13
            self.results = x13_arima_analysis(
                series,
                x12path=self.x13_path,
                outlier=outlier,
                trading=trading,
                forecast_years=forecast_years,
                **kwargs
            )
            
            # Extract seasonally adjusted series
            adjusted = self.results.seasadj
            
            logger.info("X-13ARIMA-SEATS adjustment complete")
            
            return adjusted
            
        except Exception as e:
            logger.error(f"X-13ARIMA-SEATS failed: {e}")
            logger.warning("Falling back to alternative method")
            return self._fallback_seasonal_adjust(series, freq)
    
    def _fallback_seasonal_adjust(self, series: pd.Series, freq: int) -> pd.Series:
        """
        Fallback seasonal adjustment using classical decomposition
        
        Parameters:
        -----------
        series : pd.Series
            Time series to adjust
        freq : int
            Seasonal frequency
            
        Returns:
        --------
        pd.Series
            Seasonally adjusted series
        """
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        logger.info("Using classical seasonal decomposition as fallback")
        
        # Ensure we have enough data
        if len(series) < 2 * freq:
            logger.warning("Insufficient data for seasonal adjustment")
            return series
        
        # Check if series is constant or near-constant or too small for meaningful decomposition
        if series.std() < 1e-10 or series.abs().max() < 1e-10:
            logger.warning("Series is constant or near-constant, returning original series")
            return series
        
        # Perform decomposition
        decomposition = seasonal_decompose(
            series,
            model='multiplicative' if (series > 0).all() else 'additive',
            period=freq,
            extrapolate_trend='freq'
        )
        
        # Remove seasonal component
        if decomposition.seasonal is not None:
            if (series > 0).all():
                adjusted = series / decomposition.seasonal
            else:
                adjusted = series - decomposition.seasonal
        else:
            adjusted = series
        
        return adjusted
    
    def get_diagnostics(self) -> Dict[str, any]:
        """
        Get diagnostics from X-13 results
        
        Returns:
        --------
        Dict[str, any]
            Diagnostic information
        """
        if self.results is None:
            return {}
        
        diagnostics = {}
        
        # Extract available diagnostics
        if hasattr(self.results, 'Qs'):
            diagnostics['ljung_box_statistic'] = self.results.Qs
        
        if hasattr(self.results, 'Qsval'):
            diagnostics['ljung_box_pvalue'] = self.results.Qsval
        
        if hasattr(self.results, 'arima'):
            diagnostics['arima_order'] = self.results.arima
        
        if hasattr(self.results, 'outliers'):
            diagnostics['outliers'] = self.results.outliers
        
        return diagnostics
    
    def get_components(self) -> Dict[str, pd.Series]:
        """
        Get decomposition components
        
        Returns:
        --------
        Dict[str, pd.Series]
            Dictionary with trend, seasonal, irregular components
        """
        if self.results is None:
            return {}
        
        components = {}
        
        if hasattr(self.results, 'trend'):
            components['trend'] = self.results.trend
        
        if hasattr(self.results, 'seasonal'):
            components['seasonal'] = self.results.seasonal
        
        if hasattr(self.results, 'irregular'):
            components['irregular'] = self.results.irregular
        
        if hasattr(self.results, 'seasadj'):
            components['seasonally_adjusted'] = self.results.seasadj
        
        return components
    
    def plot_diagnostics(self) -> None:
        """Plot X-13 diagnostic charts"""
        if self.results is None:
            logger.warning("No results to plot")
            return
        
        try:
            self.results.plot()
        except Exception as e:
            logger.error(f"Error plotting diagnostics: {e}")


class CustomSeasonalAdjustment:
    """
    Custom seasonal adjustment implementation when X-13 is not available
    """
    
    def __init__(self):
        self.seasonal_factors = None
        self.trend = None
        
    def adjust(self, series: pd.Series, method: str = "ratio") -> pd.Series:
        """
        Perform custom seasonal adjustment
        
        Parameters:
        -----------
        series : pd.Series
            Time series to adjust
        method : str
            Adjustment method: 'ratio' or 'difference'
            
        Returns:
        --------
        pd.Series
            Seasonally adjusted series
        """
        # Calculate seasonal factors
        self.seasonal_factors = self._calculate_seasonal_factors(series, method)
        
        # Apply adjustment
        if method == "ratio":
            adjusted = series / self.seasonal_factors
        else:
            adjusted = series - self.seasonal_factors
        
        return adjusted
    
    def _calculate_seasonal_factors(self, 
                                  series: pd.Series, 
                                  method: str) -> pd.Series:
        """Calculate seasonal factors"""
        # Extract seasonal patterns
        if hasattr(series.index, 'quarter'):
            period_col = series.index.quarter
            n_periods = 4
        elif hasattr(series.index, 'month'):
            period_col = series.index.month
            n_periods = 12
        else:
            raise ValueError("Series must have quarterly or monthly frequency")
        
        # Calculate average for each period
        period_averages = {}
        for period in range(1, n_periods + 1):
            period_data = series[period_col == period]
            if method == "ratio":
                period_averages[period] = period_data.mean() / series.mean()
            else:
                period_averages[period] = period_data.mean() - series.mean()
        
        # Create seasonal factor series
        seasonal_factors = pd.Series(index=series.index, dtype=float)
        for period, factor in period_averages.items():
            seasonal_factors[period_col == period] = factor
        
        # Normalize factors
        if method == "ratio":
            seasonal_factors = seasonal_factors / seasonal_factors.mean()
        
        return seasonal_factors