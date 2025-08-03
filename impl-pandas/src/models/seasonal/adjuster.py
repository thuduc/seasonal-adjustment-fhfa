"""Main seasonal adjustment orchestrator"""

from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from loguru import logger
from ...config import get_settings
from ..arima.regarima import RegARIMA
from ..arima.x13wrapper import X13ARIMAWrapper
import warnings


class SeasonalAdjuster:
    """
    Main class for seasonal adjustment of housing price indices
    
    Orchestrates the full seasonal adjustment process:
    1. Data preparation and validation
    2. RegARIMA modeling
    3. Seasonal adjustment (X-13ARIMA-SEATS or alternatives)
    4. Quality control and diagnostics
    """
    
    def __init__(self,
                 method: str = "x13",
                 outlier_detection: bool = True,
                 trading_day_adjustment: bool = False,
                 forecast_years: int = 1):
        """
        Initialize seasonal adjuster
        
        Parameters:
        -----------
        method : str
            Adjustment method: 'x13', 'regarima', 'stl', 'classical'
        outlier_detection : bool
            Whether to detect and adjust for outliers
        trading_day_adjustment : bool
            Whether to adjust for trading day effects
        forecast_years : int
            Number of years to forecast
        """
        self.method = method
        self.outlier_detection = outlier_detection
        self.trading_day_adjustment = trading_day_adjustment
        self.forecast_years = forecast_years
        
        self.settings = get_settings()
        self.results = {}
        self.diagnostics = {}
        
    def adjust(self,
              series: pd.Series,
              exog: Optional[pd.DataFrame] = None,
              frequency: Optional[int] = None,
              **kwargs) -> pd.Series:
        """
        Perform seasonal adjustment
        
        Parameters:
        -----------
        series : pd.Series
            Time series to adjust
        exog : pd.DataFrame, optional
            Exogenous variables for RegARIMA
        frequency : int, optional
            Seasonal frequency (auto-detected if None)
        **kwargs : dict
            Additional method-specific parameters
            
        Returns:
        --------
        pd.Series
            Seasonally adjusted series
        """
        logger.info(f"Starting seasonal adjustment using {self.method} method")
        
        # Validate input
        self._validate_input(series, exog)
        
        # Detect frequency if not provided
        if frequency is None:
            frequency = self._detect_frequency(series)
            logger.info(f"Detected frequency: {frequency}")
        
        # Apply selected method
        if self.method == "x13":
            adjusted = self._adjust_x13(series, exog, frequency, **kwargs)
        elif self.method == "regarima":
            adjusted = self._adjust_regarima(series, exog, frequency, **kwargs)
        elif self.method == "stl":
            adjusted = self._adjust_stl(series, frequency, **kwargs)
        elif self.method == "classical":
            adjusted = self._adjust_classical(series, frequency, **kwargs)
        else:
            raise ValueError(f"Unknown adjustment method: {self.method}")
        
        # Run diagnostics
        self._run_diagnostics(series, adjusted)
        
        # Quality control
        if not self._quality_check(series, adjusted):
            warnings.warn("Quality checks failed. Results may be unreliable.")
        
        return adjusted
    
    def _adjust_x13(self,
                   series: pd.Series,
                   exog: Optional[pd.DataFrame],
                   frequency: int,
                   **kwargs) -> pd.Series:
        """Apply X-13ARIMA-SEATS adjustment"""
        
        # Initialize X-13 wrapper
        x13 = X13ARIMAWrapper(x13_path=getattr(self.settings, 'x13_path', None))
        
        # Prepare options
        x13_options = {
            'outlier': self.outlier_detection,
            'trading': self.trading_day_adjustment,
            'forecast_years': self.forecast_years
        }
        x13_options.update(kwargs)
        
        # Run adjustment
        adjusted = x13.seasonal_adjust(series, freq=frequency, **x13_options)
        
        # Store components
        self.results['components'] = x13.get_components()
        self.results['diagnostics'] = x13.get_diagnostics()
        
        return adjusted
    
    def _adjust_regarima(self,
                        series: pd.Series,
                        exog: Optional[pd.DataFrame],
                        frequency: int,
                        **kwargs) -> pd.Series:
        """Apply RegARIMA-based adjustment"""
        
        # Determine seasonal order based on frequency
        if frequency == 4:
            seasonal_order = (0, 1, 1, 4)
        elif frequency == 12:
            seasonal_order = (0, 1, 1, 12)
        else:
            seasonal_order = (0, 0, 0, 0)
        
        # Initialize RegARIMA model
        model = RegARIMA(
            ar_order=kwargs.get('ar_order', 1),
            diff_order=kwargs.get('diff_order', 1),
            ma_order=kwargs.get('ma_order', 1),
            seasonal_order=seasonal_order,
            exog_vars=exog.columns.tolist() if exog is not None else None,
            outlier_detection=self.outlier_detection
        )
        
        # Fit model
        model.fit(series, X=exog)
        
        # Get seasonally adjusted series
        adjusted = model.seasonal_adjust(method="decomposition")
        
        # Store results
        self.results['model'] = model
        self.results['diagnostics'] = model.diagnostics()
        
        return adjusted
    
    def _adjust_stl(self,
                   series: pd.Series,
                   frequency: int,
                   **kwargs) -> pd.Series:
        """Apply STL decomposition"""
        from statsmodels.tsa.seasonal import STL
        
        # Configure STL
        stl = STL(
            series,
            period=frequency,
            seasonal=kwargs.get('seasonal', 7),
            trend=kwargs.get('trend', None),
            robust=kwargs.get('robust', True)
        )
        
        # Perform decomposition
        decomposition = stl.fit()
        
        # Remove seasonal component
        adjusted = series - decomposition.seasonal
        
        # Store components
        self.results['components'] = {
            'trend': decomposition.trend,
            'seasonal': decomposition.seasonal,
            'residual': decomposition.resid
        }
        
        return adjusted
    
    def _adjust_classical(self,
                         series: pd.Series,
                         frequency: int,
                         **kwargs) -> pd.Series:
        """Apply classical seasonal decomposition"""
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        # Check if series is constant or near-constant or too small for meaningful decomposition
        if series.std() < 1e-10 or series.abs().max() < 1e-10:
            logger.warning("Series is constant or near-constant, returning original series")
            self.results['components'] = {
                'trend': series.copy(),
                'seasonal': pd.Series(0, index=series.index),
                'residual': pd.Series(0, index=series.index)
            }
            return series
        
        # Determine model type
        if (series > 0).all():
            model_type = 'multiplicative'
        else:
            model_type = 'additive'
        
        # Perform decomposition
        try:
            decomposition = seasonal_decompose(
                series,
                model=model_type,
                period=frequency,
                extrapolate_trend='freq'
            )
            
            # Remove seasonal component
            if model_type == 'multiplicative':
                adjusted = series / decomposition.seasonal
            else:
                adjusted = series - decomposition.seasonal
            
            # Store components
            self.results['components'] = {
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid
            }
            
            return adjusted
            
        except (ValueError, ZeroDivisionError) as e:
            # Handle short series or other errors
            logger.warning(f"Classical decomposition failed: {e}")
            warnings.warn(f"Cannot perform seasonal adjustment: {e}", UserWarning)
            # Return original series if decomposition fails
            return series
    
    def _validate_input(self, series: pd.Series, exog: Optional[pd.DataFrame]) -> None:
        """Validate input data"""
        
        # Check series
        if not isinstance(series, pd.Series):
            raise TypeError("Series must be a pandas Series")
        
        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError("Series must have DatetimeIndex")
        
        if series.isna().all():
            raise ValueError("Series contains only missing values")
        
        # Check for minimum length
        min_length = 2 * 12  # At least 2 years for monthly data
        if len(series) < min_length:
            warnings.warn(f"Series length ({len(series)}) may be too short for reliable adjustment")
        
        # Check exogenous variables
        if exog is not None:
            if not isinstance(exog, pd.DataFrame):
                raise TypeError("Exogenous variables must be a DataFrame")
            
            if not exog.index.equals(series.index):
                raise ValueError("Series and exogenous variables must have same index")
    
    def _detect_frequency(self, series: pd.Series) -> int:
        """Detect seasonal frequency"""
        
        # Try to infer from pandas frequency
        if series.index.freq is not None:
            freq_str = series.index.freq.name
            if freq_str.startswith('Q'):
                return 4
            elif freq_str == 'M' or freq_str.startswith('M'):
                return 12
            elif freq_str == 'D':
                return 7
        
        # Fall back to date differences
        date_diffs = series.index[1:] - series.index[:-1]
        median_diff = pd.Series(date_diffs).median()
        
        if median_diff.days >= 85 and median_diff.days <= 95:
            return 4  # Quarterly
        elif median_diff.days >= 28 and median_diff.days <= 31:
            return 12  # Monthly
        elif median_diff.days == 7:
            return 52  # Weekly
        else:
            logger.warning("Could not detect frequency, assuming quarterly")
            return 4
    
    def _run_diagnostics(self, original: pd.Series, adjusted: pd.Series) -> None:
        """Run diagnostic tests"""
        
        # Calculate seasonal factors
        if (original > 0).all() and (adjusted > 0).all():
            seasonal_factors = original / adjusted
        else:
            seasonal_factors = original - adjusted
        
        # Basic statistics
        self.diagnostics['seasonal_strength'] = 1 - (
            seasonal_factors.std() / original.std()
        )
        
        # Stability test
        self.diagnostics['seasonal_stability'] = self._test_seasonal_stability(
            seasonal_factors
        )
        
        # Residual diagnostics if available
        if 'components' in self.results and 'residual' in self.results['components']:
            residuals = self.results['components']['residual']
            
            # Normality test
            from scipy import stats
            jb_stat, jb_pvalue = stats.jarque_bera(residuals.dropna())
            self.diagnostics['residual_normality'] = {
                'statistic': jb_stat,
                'pvalue': jb_pvalue
            }
            
            # Autocorrelation test
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb_result = acorr_ljungbox(residuals.dropna(), lags=10, return_df=True)
            self.diagnostics['residual_autocorrelation'] = {
                'min_pvalue': lb_result['lb_pvalue'].min()
            }
    
    def _test_seasonal_stability(self, seasonal_factors: pd.Series) -> Dict[str, float]:
        """Test stability of seasonal pattern"""
        
        # Group by season
        if hasattr(seasonal_factors.index, 'quarter'):
            groups = seasonal_factors.groupby(seasonal_factors.index.quarter)
        elif hasattr(seasonal_factors.index, 'month'):
            groups = seasonal_factors.groupby(seasonal_factors.index.month)
        else:
            return {'cv': np.nan, 'stable': False}
        
        # Calculate coefficient of variation for each season
        cvs = []
        for name, group in groups:
            if len(group) > 1:
                cv = group.std() / group.mean() if group.mean() != 0 else np.nan
                cvs.append(cv)
        
        mean_cv = np.nanmean(cvs) if cvs else np.nan
        
        return {
            'cv': mean_cv,
            'stable': mean_cv < 0.1 if not np.isnan(mean_cv) else False
        }
    
    def _quality_check(self, original: pd.Series, adjusted: pd.Series) -> bool:
        """Perform quality checks"""
        
        checks_passed = True
        
        # Check 1: Adjusted series should have lower seasonal variation
        if hasattr(original.index, 'quarter'):
            orig_seasonal_var = original.groupby(original.index.quarter).std().mean()
            adj_seasonal_var = adjusted.groupby(adjusted.index.quarter).std().mean()
            
            if adj_seasonal_var >= orig_seasonal_var:
                logger.warning("Adjusted series has higher seasonal variation than original")
                checks_passed = False
        
        # Check 2: Mean preservation (for additive adjustment)
        if abs(original.mean() - adjusted.mean()) > 0.01 * original.mean():
            logger.warning("Mean not preserved in adjustment")
            checks_passed = False
        
        # Check 3: No extreme values introduced
        orig_range = original.max() - original.min()
        adj_range = adjusted.max() - adjusted.min()
        
        if adj_range > 2 * orig_range:
            logger.warning("Adjustment introduced extreme values")
            checks_passed = False
        
        # Check 4: Residual diagnostics (if available)
        if 'residual_normality' in self.diagnostics:
            if self.diagnostics['residual_normality']['pvalue'] < 0.01:
                logger.warning("Residuals show significant non-normality")
                checks_passed = False
        
        return checks_passed
    
    def get_results(self) -> Dict[str, any]:
        """Get adjustment results and diagnostics"""
        return {
            'method': self.method,
            'results': self.results,
            'diagnostics': self.diagnostics
        }
    
    def plot_results(self, 
                    original: pd.Series,
                    adjusted: pd.Series,
                    figsize: Tuple[int, int] = (15, 10)) -> None:
        """Plot adjustment results"""
        import matplotlib.pyplot as plt
        
        # Determine number of subplots based on available components
        n_plots = 2  # Original vs adjusted + seasonal factors
        if 'components' in self.results:
            n_plots = 4  # Add trend and residual
        
        fig, axes = plt.subplots(n_plots, 1, figsize=figsize)
        
        # Plot 1: Original vs Adjusted
        ax = axes[0]
        original.plot(ax=ax, label='Original', linewidth=2)
        adjusted.plot(ax=ax, label='Seasonally Adjusted', linewidth=2, alpha=0.8)
        ax.set_title('Original vs Seasonally Adjusted Series')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Seasonal Factors
        ax = axes[1]
        if (original > 0).all() and (adjusted > 0).all():
            seasonal_factors = original / adjusted
            ax.set_title('Seasonal Factors (Multiplicative)')
        else:
            seasonal_factors = original - adjusted
            ax.set_title('Seasonal Component (Additive)')
        
        seasonal_factors.plot(ax=ax, linewidth=1.5)
        ax.axhline(y=seasonal_factors.mean(), color='r', linestyle='--', 
                  label=f'Mean: {seasonal_factors.mean():.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Additional plots if components available
        if 'components' in self.results:
            components = self.results['components']
            
            # Plot 3: Trend
            if 'trend' in components:
                ax = axes[2]
                components['trend'].plot(ax=ax, linewidth=2, color='green')
                ax.set_title('Trend Component')
                ax.grid(True, alpha=0.3)
            
            # Plot 4: Residuals
            if 'residual' in components:
                ax = axes[3]
                components['residual'].plot(ax=ax, linewidth=1, color='red', alpha=0.7)
                ax.axhline(y=0, color='black', linestyle='--')
                ax.set_title('Residual Component')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def summary(self) -> str:
        """Generate summary report"""
        
        lines = []
        lines.append("=" * 60)
        lines.append("SEASONAL ADJUSTMENT SUMMARY")
        lines.append("=" * 60)
        lines.append(f"Method: {self.method.upper()}")
        lines.append(f"Outlier Detection: {self.outlier_detection}")
        lines.append(f"Trading Day Adjustment: {self.trading_day_adjustment}")
        
        if self.diagnostics:
            lines.append("\nDIAGNOSTICS:")
            lines.append("-" * 30)
            
            if 'seasonal_strength' in self.diagnostics:
                lines.append(f"Seasonal Strength: {self.diagnostics['seasonal_strength']:.3f}")
            
            if 'seasonal_stability' in self.diagnostics:
                stability = self.diagnostics['seasonal_stability']
                lines.append(f"Seasonal Stability CV: {stability['cv']:.3f}")
                lines.append(f"Stable Pattern: {'Yes' if stability['stable'] else 'No'}")
            
            if 'residual_normality' in self.diagnostics:
                norm = self.diagnostics['residual_normality']
                lines.append(f"Residual Normality p-value: {norm['pvalue']:.4f}")
            
            if 'residual_autocorrelation' in self.diagnostics:
                acorr = self.diagnostics['residual_autocorrelation']
                lines.append(f"Residual Autocorrelation min p-value: {acorr['min_pvalue']:.4f}")
        
        if 'diagnostics' in self.results:
            lines.append("\nMODEL DIAGNOSTICS:")
            lines.append("-" * 30)
            for key, value in self.results['diagnostics'].items():
                lines.append(f"{key}: {value}")
        
        return "\n".join(lines)