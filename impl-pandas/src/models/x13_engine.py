import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
import logging
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox as ljungbox
from scipy import stats
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger('seasonal_adjustment.x13_engine')


class X13ARIMAEngine:
    """Implementation of X-13ARIMA-SEATS seasonal adjustment methodology."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.max_ar_order = self.config.get('max_ar_order', 4)
        self.max_ma_order = self.config.get('max_ma_order', 4)
        self.max_seasonal_order = self.config.get('max_seasonal_order', 2)
        self.seasonal_period = self.config.get('seasonal_period', 4)  # Quarterly data
        
    def pre_adjustment(self, series: pd.Series) -> pd.Series:
        """Remove trading day and holiday effects from series."""
        logger.debug("Performing pre-adjustment for trading day and holiday effects")
        
        # For quarterly data, trading day effects are minimal
        # Implement a simple adjustment based on number of days in quarter
        adjusted_series = series.copy()
        
        # Get the quarter start dates
        dates = pd.to_datetime(series.index)
        quarter_days = []
        
        for date in dates:
            # Calculate days in quarter
            quarter_start = pd.Timestamp(year=date.year, month=((date.quarter-1)*3)+1, day=1)
            if date.quarter == 4:
                quarter_end = pd.Timestamp(year=date.year+1, month=1, day=1)
            else:
                quarter_end = pd.Timestamp(year=date.year, month=((date.quarter)*3)+1, day=1)
            days_in_quarter = (quarter_end - quarter_start).days
            quarter_days.append(days_in_quarter)
            
        # Normalize by average quarter length (91.25 days)
        avg_days = 91.25
        adjustment_factors = pd.Series([avg_days / days for days in quarter_days], index=series.index)
        
        adjusted_series = series * adjustment_factors
        
        return adjusted_series
    
    def detect_outliers(self, series: pd.Series, threshold: float = 3.0) -> pd.DataFrame:
        """Identify and handle outliers in the series."""
        logger.debug("Detecting outliers in series")
        
        # Use median absolute deviation (MAD) for more robust outlier detection
        median = series.median()
        mad = np.median(np.abs(series - median))
        # Use a scaling factor to make MAD comparable to standard deviation
        mad_scaled = 1.4826 * mad if mad > 0 else 0.001
        
        # Calculate modified z-scores using median and MAD
        z_scores = np.abs((series - median) / mad_scaled)
        
        # Also keep rolling statistics for the outlier dataframe
        rolling_mean = series.rolling(window=4, center=True, min_periods=1).mean()
        rolling_std = series.rolling(window=4, center=True, min_periods=1).std()
        
        # Handle NaN z-scores (set them to 0)
        z_scores = z_scores.fillna(0)
        
        outliers = np.abs(z_scores) > threshold
        
        # Create outlier dataframe
        outlier_df = pd.DataFrame({
            'value': series,
            'rolling_mean': rolling_mean,
            'rolling_std': rolling_std,
            'z_score': z_scores,
            'is_outlier': outliers
        })
        
        # Handle outliers by interpolation
        adjusted_series = series.copy()
        if outliers.any():
            logger.info(f"Found {outliers.sum()} outliers")
            outlier_indices = series.index[outliers]
            for idx in outlier_indices:
                # Use linear interpolation
                adjusted_series.loc[idx] = np.nan
            adjusted_series = adjusted_series.interpolate(method='linear')
            
        outlier_df['adjusted_value'] = adjusted_series
        
        return outlier_df
    
    def fit_arima_model(self, series: pd.Series, order: Tuple[int, int, int], 
                       seasonal_order: Tuple[int, int, int, int]) -> ARIMA:
        """Fit ARIMA model with specified orders."""
        logger.debug(f"Fitting ARIMA model with order {order} and seasonal order {seasonal_order}")
        
        try:
            model = ARIMA(series, order=order, seasonal_order=seasonal_order)
            fitted_model = model.fit(method_kwargs={'maxiter': 1000})
            return fitted_model
        except Exception as e:
            logger.error(f"Error fitting ARIMA model: {str(e)}")
            raise
    
    def extract_seasonal_factors(self, series: pd.Series, 
                               fitted_model: Optional[ARIMA] = None) -> pd.Series:
        """Extract seasonal components from fitted model or using decomposition."""
        logger.debug("Extracting seasonal factors")
        
        if fitted_model is not None:
            # Extract seasonal component from ARIMA model
            # Get the seasonal AR and MA parameters
            seasonal_params = fitted_model.params[fitted_model.model.seasonal_order[0]:]
            
            # Calculate seasonal factors using the model's seasonal component
            seasonal_factors = self._calculate_seasonal_factors_from_model(
                series, fitted_model
            )
        else:
            # Use classical seasonal decomposition as fallback
            decomposition = seasonal_decompose(series, model='multiplicative', period=4)
            seasonal_factors = decomposition.seasonal
            
        # Normalize seasonal factors to average 1.0
        seasonal_factors = seasonal_factors / seasonal_factors.mean()
        
        return seasonal_factors
    
    def _calculate_seasonal_factors_from_model(self, series: pd.Series, 
                                             model: ARIMA) -> pd.Series:
        """Calculate seasonal factors from fitted ARIMA model."""
        
        # Get fitted values and residuals
        fitted_values = model.fittedvalues
        residuals = model.resid
        
        # Estimate seasonal pattern from residuals
        seasonal_pattern = []
        for quarter in range(1, 5):
            # Get residuals for specific quarter
            quarter_mask = series.index.quarter == quarter
            quarter_residuals = residuals[quarter_mask]
            
            # Calculate average seasonal effect for this quarter
            seasonal_effect = np.exp(quarter_residuals.mean())
            seasonal_pattern.append(seasonal_effect)
            
        # Create seasonal factors series
        seasonal_factors = pd.Series(index=series.index, dtype=float)
        for i, idx in enumerate(series.index):
            quarter = idx.quarter
            seasonal_factors.iloc[i] = seasonal_pattern[quarter - 1]
            
        return seasonal_factors
    
    def calculate_adjusted_series(self, original: pd.Series, 
                                seasonal: pd.Series) -> pd.Series:
        """Calculate seasonally adjusted series."""
        logger.debug("Calculating seasonally adjusted series")
        
        # Ensure same length and alignment
        if len(original) != len(seasonal):
            logger.warning(f"Length mismatch: original={len(original)}, seasonal={len(seasonal)}")
            # Align by index
            seasonal = seasonal.reindex(original.index, method='nearest')
        
        # For multiplicative model: SA = NSA / Seasonal
        adjusted = original / seasonal
        
        # Ensure no extreme adjustments
        ratio = adjusted / original
        if (ratio < 0.5).any() or (ratio > 2.0).any():
            logger.warning("Extreme seasonal adjustments detected")
            
        return adjusted
    
    def decompose_series(self, series: pd.Series) -> Dict[str, pd.Series]:
        """Perform complete decomposition into trend, seasonal, and irregular."""
        logger.debug("Performing series decomposition")
        
        # Remove outliers first
        outlier_df = self.detect_outliers(series)
        clean_series = outlier_df['adjusted_value']
        
        # Pre-adjust for trading days
        pre_adjusted = self.pre_adjustment(clean_series)
        
        # Perform decomposition
        decomposition = seasonal_decompose(pre_adjusted, model='multiplicative', period=4)
        
        components = {
            'original': series,
            'trend': decomposition.trend,
            'seasonal': decomposition.seasonal,
            'irregular': decomposition.resid,
            'seasonally_adjusted': series / decomposition.seasonal
        }
        
        return components
    
    def run_diagnostics(self, model: ARIMA, residuals: pd.Series) -> Dict[str, any]:
        """Run diagnostic tests on fitted model."""
        logger.debug("Running model diagnostics")
        
        diagnostics = {}
        
        # Ljung-Box test for residual autocorrelation
        lb_test = ljungbox(residuals.dropna(), lags=10, return_df=True)
        diagnostics['ljung_box'] = {
            'statistic': lb_test['lb_stat'].values,
            'p_value': lb_test['lb_pvalue'].values,
            'significant': (lb_test['lb_pvalue'] < 0.05).any()
        }
        
        # Normality tests
        jb_stat, jb_pvalue = stats.jarque_bera(residuals.dropna())
        diagnostics['jarque_bera'] = {
            'statistic': jb_stat,
            'p_value': jb_pvalue,
            'normal': jb_pvalue > 0.05
        }
        
        # Stationarity test (ADF)
        adf_result = adfuller(residuals.dropna())
        diagnostics['adf_test'] = {
            'statistic': adf_result[0],
            'p_value': adf_result[1],
            'stationary': adf_result[1] < 0.05
        }
        
        # Model fit statistics
        diagnostics['aic'] = model.aic
        diagnostics['bic'] = model.bic
        diagnostics['log_likelihood'] = model.llf
        
        # Residual statistics
        diagnostics['residual_stats'] = {
            'mean': residuals.mean(),
            'std': residuals.std(),
            'skewness': stats.skew(residuals.dropna()),
            'kurtosis': stats.kurtosis(residuals.dropna())
        }
        
        return diagnostics
    
    def stability_test(self, series: pd.Series, window_size: int = 20) -> Dict[str, any]:
        """Test stability of seasonal pattern over time."""
        logger.debug("Testing seasonal pattern stability")
        
        # Split series into overlapping windows
        stability_results = []
        
        for i in range(len(series) - window_size + 1):
            window = series.iloc[i:i+window_size]
            
            # Calculate seasonal factors for this window
            decomp = seasonal_decompose(window, model='multiplicative', period=4)
            seasonal = decomp.seasonal
            
            # Store seasonal pattern
            pattern = []
            for q in range(1, 5):
                q_values = seasonal[seasonal.index.quarter == q]
                if len(q_values) > 0:
                    pattern.append(q_values.mean())
                    
            if len(pattern) == 4:
                stability_results.append(pattern)
                
        # Calculate coefficient of variation for each quarter
        stability_results = np.array(stability_results)
        cv_by_quarter = np.std(stability_results, axis=0) / np.mean(stability_results, axis=0)
        
        return {
            'cv_by_quarter': cv_by_quarter,
            'max_cv': cv_by_quarter.max(),
            'stable': cv_by_quarter.max() < 0.1,
            'seasonal_patterns': stability_results
        }