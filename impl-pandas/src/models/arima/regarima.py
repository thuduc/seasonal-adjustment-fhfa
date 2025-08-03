"""RegARIMA model implementation following FHFA specification"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
from loguru import logger
from ...config import get_settings


class RegARIMA:
    """
    RegARIMA model implementation following FHFA specification
    
    Implements the general regression for a regARIMA model:
    φ(B)Φ(Bs)(1 - B)^d(yt - Σ βixit) = θ(B)Θ(Bs)εt
    """
    
    def __init__(self,
                 ar_order: int = 1,
                 diff_order: int = 1,
                 ma_order: int = 2,
                 seasonal_order: Tuple[int, int, int, int] = (0, 1, 1, 4),
                 exog_vars: Optional[List[str]] = None,
                 outlier_detection: bool = True):
        """
        Initialize RegARIMA model
        
        Parameters:
        -----------
        ar_order : int
            Order of autoregressive component (p)
        diff_order : int
            Order of differencing (d)
        ma_order : int
            Order of moving average component (q)
        seasonal_order : Tuple[int, int, int, int]
            Seasonal order (P, D, Q, s)
        exog_vars : List[str], optional
            Names of exogenous variables
        outlier_detection : bool
            Whether to detect and adjust for outliers
        """
        self.order = (ar_order, diff_order, ma_order)
        self.seasonal_order = seasonal_order
        self.exog_vars = exog_vars or []
        self.outlier_detection = outlier_detection
        
        self.model = None
        self.results = None
        self.outliers = {}
        self.seasonal_factors = None
        
        self.settings = get_settings()
        
        # Validate orders
        if ar_order + diff_order + ma_order == 0:
            raise ValueError("At least one of AR, I, or MA order must be non-zero")
    
    def fit(self, 
            y: pd.Series,
            X: Optional[pd.DataFrame] = None,
            **kwargs) -> 'RegARIMA':
        """
        Fit RegARIMA model
        
        Parameters:
        -----------
        y : pd.Series
            Dependent variable (time series)
        X : pd.DataFrame, optional
            Exogenous variables
        **kwargs : dict
            Additional arguments for SARIMAX
            
        Returns:
        --------
        self : RegARIMA
            Fitted model
        """
        logger.info(f"Fitting RegARIMA model with order {self.order} and "
                   f"seasonal order {self.seasonal_order}")
        
        # Ensure y is properly indexed
        if not isinstance(y.index, pd.DatetimeIndex):
            raise ValueError("Time series must have DatetimeIndex")
        
        # Handle missing values
        y_clean = y.dropna()
        
        # Detect outliers if requested
        if self.outlier_detection:
            self._detect_outliers(y_clean)
            # Add outlier dummies to exogenous variables
            if self.outliers and X is not None:
                X = self._add_outlier_dummies(X, y_clean.index)
        
        # Initialize SARIMAX model
        try:
            self.model = SARIMAX(
                y_clean,
                order=self.order,
                seasonal_order=self.seasonal_order,
                exog=X.loc[y_clean.index] if X is not None else None,
                **kwargs
            )
            
            # Fit model
            self.results = self.model.fit(disp=0)
            
            logger.info(f"Model fit complete. AIC: {self.results.aic:.2f}, "
                       f"BIC: {self.results.bic:.2f}")
            
        except Exception as e:
            logger.error(f"Error fitting RegARIMA model: {e}")
            raise
        
        return self
    
    def predict(self,
                steps: int = 1,
                exog: Optional[pd.DataFrame] = None,
                return_conf_int: bool = False) -> Union[pd.Series, Tuple[pd.Series, pd.DataFrame]]:
        """
        Generate forecasts
        
        Parameters:
        -----------
        steps : int
            Number of steps ahead to forecast
        exog : pd.DataFrame, optional
            Future values of exogenous variables
        return_conf_int : bool
            Whether to return confidence intervals
            
        Returns:
        --------
        pd.Series or Tuple[pd.Series, pd.DataFrame]
            Forecasts and optionally confidence intervals
        """
        if self.results is None:
            raise ValueError("Model must be fitted before prediction")
        
        forecast = self.results.forecast(steps=steps, exog=exog)
        
        if return_conf_int:
            # Get prediction intervals
            forecast_df = self.results.get_forecast(steps=steps, exog=exog)
            conf_int = forecast_df.conf_int(alpha=0.05)
            return forecast, conf_int
        
        return forecast
    
    def seasonal_adjust(self, method: str = "x13") -> pd.Series:
        """
        Apply seasonal adjustment
        
        Parameters:
        -----------
        method : str
            Method for seasonal adjustment: 'x13' or 'decomposition'
            
        Returns:
        --------
        pd.Series
            Seasonally adjusted series
        """
        if self.results is None:
            raise ValueError("Model must be fitted before seasonal adjustment")
        
        if method == "x13":
            # Use X-13ARIMA-SEATS wrapper
            from .x13wrapper import X13ARIMAWrapper
            x13 = X13ARIMAWrapper()
            return x13.seasonal_adjust(self.model.endog)
        else:
            # Use model-based decomposition
            return self._model_based_adjustment()
    
    def _model_based_adjustment(self) -> pd.Series:
        """Apply model-based seasonal adjustment"""
        # Get seasonal component from state space representation
        states = self.results.states.filtered
        
        # Extract seasonal component
        # This depends on the specific state space representation
        seasonal_component = self._extract_seasonal_component(states)
        
        # Remove seasonal component
        adjusted = pd.Series(
            self.model.endog - seasonal_component,
            index=self.model.endog.index
        )
        
        return adjusted
    
    def _extract_seasonal_component(self, states: np.ndarray) -> np.ndarray:
        """Extract seasonal component from state space"""
        # The seasonal component location depends on model specification
        # For SARIMAX with seasonal order (P,D,Q,s), it's typically
        # in the latter part of the state vector
        
        # This is a simplified extraction - actual implementation
        # depends on specific state space formulation
        seasonal_start_idx = self.order[0] + self.order[2]  # After AR and MA states
        seasonal_states = states[seasonal_start_idx:, :]
        
        # Sum seasonal states to get component
        seasonal_component = np.sum(seasonal_states, axis=0)
        
        return seasonal_component
    
    def diagnostics(self) -> Dict[str, Union[float, bool, pd.DataFrame]]:
        """
        Run diagnostic tests on fitted model
        
        Returns:
        --------
        Dict[str, Union[float, bool, pd.DataFrame]]
            Dictionary of diagnostic test results
        """
        if self.results is None:
            raise ValueError("Model must be fitted before diagnostics")
        
        residuals = self.results.resid
        
        # Ljung-Box test for autocorrelation
        lb_result = acorr_ljungbox(residuals, lags=10, return_df=True)
        ljung_box_pvalue = lb_result['lb_pvalue'].min()
        
        # Normality tests
        jb_stat, jb_pvalue = stats.jarque_bera(residuals)
        
        # Heteroscedasticity test (Breusch-Pagan)
        bp_stat, bp_pvalue = self._breusch_pagan_test(residuals)
        
        # ARCH test for conditional heteroscedasticity
        arch_stat, arch_pvalue = self._arch_test(residuals)
        
        diagnostics = {
            "ljung_box_pvalue": ljung_box_pvalue,
            "ljung_box_pass": ljung_box_pvalue > 0.05,
            "jarque_bera_pvalue": jb_pvalue,
            "normality_pass": jb_pvalue > 0.05,
            "breusch_pagan_pvalue": bp_pvalue,
            "homoscedasticity_pass": bp_pvalue > 0.05,
            "arch_pvalue": arch_pvalue,
            "arch_pass": arch_pvalue > 0.05,
            "aic": self.results.aic,
            "bic": self.results.bic,
            "log_likelihood": self.results.llf,
            "residual_stats": {
                "mean": residuals.mean(),
                "std": residuals.std(),
                "skew": stats.skew(residuals),
                "kurtosis": stats.kurtosis(residuals)
            }
        }
        
        return diagnostics
    
    def _detect_outliers(self, y: pd.Series) -> None:
        """Detect outliers in time series"""
        # Simple outlier detection using IQR method
        # More sophisticated methods would use intervention analysis
        
        residuals = y - y.rolling(window=4, center=True).mean()
        residuals = residuals.dropna()
        
        Q1 = residuals.quantile(0.25)
        Q3 = residuals.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        outlier_mask = (residuals < lower_bound) | (residuals > upper_bound)
        outlier_dates = residuals[outlier_mask].index
        
        # Classify outliers (simplified - could use more sophisticated methods)
        for date in outlier_dates:
            self.outliers[date] = {
                "type": "AO",  # Additive outlier
                "value": residuals[date]
            }
        
        if self.outliers:
            logger.info(f"Detected {len(self.outliers)} outliers")
    
    def _add_outlier_dummies(self, X: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
        """Add outlier dummy variables to exogenous matrix"""
        X_with_outliers = X.copy() if X is not None else pd.DataFrame(index=index)
        
        for date, outlier_info in self.outliers.items():
            if date in index:
                col_name = f"outlier_{date.strftime('%Y%m%d')}"
                X_with_outliers[col_name] = 0
                X_with_outliers.loc[date, col_name] = 1
        
        return X_with_outliers
    
    def _breusch_pagan_test(self, residuals: pd.Series) -> Tuple[float, float]:
        """Breusch-Pagan test for heteroscedasticity"""
        # Simplified implementation
        residuals_sq = residuals ** 2
        
        # Regress squared residuals on fitted values
        fitted = self.results.fittedvalues
        X = np.column_stack([np.ones(len(fitted)), fitted])
        
        # OLS regression
        beta = np.linalg.lstsq(X, residuals_sq, rcond=None)[0]
        predicted = X @ beta
        
        # Test statistic
        n = len(residuals)
        r_squared = 1 - np.sum((residuals_sq - predicted) ** 2) / np.sum((residuals_sq - residuals_sq.mean()) ** 2)
        lm_statistic = n * r_squared
        
        # Chi-square test with 1 degree of freedom
        p_value = 1 - stats.chi2.cdf(lm_statistic, df=1)
        
        return lm_statistic, p_value
    
    def _arch_test(self, residuals: pd.Series, lags: int = 5) -> Tuple[float, float]:
        """ARCH test for conditional heteroscedasticity"""
        residuals_sq = residuals ** 2
        n = len(residuals)
        
        # Create lagged squared residuals
        X = np.ones((n - lags, lags + 1))
        for i in range(1, lags + 1):
            X[:, i] = residuals_sq.iloc[lags-i:-i].values
        
        y = residuals_sq.iloc[lags:].values
        
        # OLS regression
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        predicted = X @ beta
        
        # Test statistic
        r_squared = 1 - np.sum((y - predicted) ** 2) / np.sum((y - y.mean()) ** 2)
        lm_statistic = n * r_squared
        
        # Chi-square test
        p_value = 1 - stats.chi2.cdf(lm_statistic, df=lags)
        
        return lm_statistic, p_value
    
    def summary(self) -> None:
        """Print model summary"""
        if self.results is None:
            raise ValueError("Model must be fitted before summary")
        
        print(self.results.summary())
        
        # Additional diagnostics
        print("\n" + "="*60)
        print("Additional Diagnostics:")
        print("="*60)
        
        diag = self.diagnostics()
        print(f"Ljung-Box p-value: {diag['ljung_box_pvalue']:.4f} "
              f"({'PASS' if diag['ljung_box_pass'] else 'FAIL'})")
        print(f"Normality p-value: {diag['jarque_bera_pvalue']:.4f} "
              f"({'PASS' if diag['normality_pass'] else 'FAIL'})")
        print(f"Homoscedasticity p-value: {diag['breusch_pagan_pvalue']:.4f} "
              f"({'PASS' if diag['homoscedasticity_pass'] else 'FAIL'})")
        print(f"ARCH test p-value: {diag['arch_pvalue']:.4f} "
              f"({'PASS' if diag['arch_pass'] else 'FAIL'})")
        
        if self.outliers:
            print(f"\nDetected {len(self.outliers)} outliers:")
            for date, info in list(self.outliers.items())[:5]:
                print(f"  - {date}: {info['type']} (value: {info['value']:.2f})")
            if len(self.outliers) > 5:
                print(f"  ... and {len(self.outliers) - 5} more")