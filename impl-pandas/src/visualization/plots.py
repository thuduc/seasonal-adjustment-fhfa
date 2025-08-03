"""Plotting utilities for seasonal adjustment analysis"""

from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.dates as mdates
from loguru import logger
import warnings

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class SeasonalAdjustmentPlotter:
    """
    Create visualizations for seasonal adjustment results
    
    Generates plots for:
    - Original vs adjusted series
    - Seasonal components
    - Trend analysis
    - Residual diagnostics
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize plotter
        
        Parameters:
        -----------
        figsize : Tuple[int, int]
            Default figure size
        """
        self.figsize = figsize
        self.colors = sns.color_palette("husl", 8)
        
    def plot_adjustment_comparison(self,
                                 original: pd.Series,
                                 adjusted: pd.Series,
                                 title: Optional[str] = None,
                                 save_path: Optional[str] = None) -> Figure:
        """
        Plot original vs seasonally adjusted series
        
        Parameters:
        -----------
        original : pd.Series
            Original time series
        adjusted : pd.Series
            Seasonally adjusted series
        title : str, optional
            Plot title
        save_path : str, optional
            Path to save figure
            
        Returns:
        --------
        Figure
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, sharex=True)
        
        # Plot original vs adjusted
        ax1.plot(original.index, original.values, label='Original', 
                color=self.colors[0], linewidth=2, alpha=0.8)
        ax1.plot(adjusted.index, adjusted.values, label='Seasonally Adjusted', 
                color=self.colors[1], linewidth=2)
        ax1.set_ylabel('Index Value')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        if title:
            ax1.set_title(title)
        else:
            ax1.set_title('Original vs Seasonally Adjusted Series')
        
        # Plot seasonal component
        if (original > 0).all() and (adjusted > 0).all():
            seasonal = (original / adjusted - 1) * 100  # Percentage deviation
            ax2.set_ylabel('Seasonal Component (%)')
        else:
            seasonal = original - adjusted
            ax2.set_ylabel('Seasonal Component')
            
        ax2.plot(seasonal.index, seasonal.values, color=self.colors[2], linewidth=1.5)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax2.xaxis.set_major_locator(mdates.YearLocator(2))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
            
        return fig
    
    def plot_decomposition(self,
                         components: Dict[str, pd.Series],
                         title: Optional[str] = None,
                         save_path: Optional[str] = None) -> Figure:
        """
        Plot seasonal decomposition components
        
        Parameters:
        -----------
        components : Dict[str, pd.Series]
            Dictionary with 'trend', 'seasonal', 'residual' components
        title : str, optional
            Plot title
        save_path : str, optional
            Path to save figure
            
        Returns:
        --------
        Figure
            Matplotlib figure
        """
        n_components = len(components)
        fig, axes = plt.subplots(n_components, 1, figsize=(self.figsize[0], 3*n_components), 
                                sharex=True)
        
        if n_components == 1:
            axes = [axes]
            
        component_order = ['trend', 'seasonal', 'residual']
        plot_idx = 0
        
        for comp_name in component_order:
            if comp_name in components:
                ax = axes[plot_idx]
                comp_data = components[comp_name]
                
                ax.plot(comp_data.index, comp_data.values, 
                       color=self.colors[plot_idx], linewidth=1.5)
                ax.set_ylabel(comp_name.capitalize())
                ax.grid(True, alpha=0.3)
                
                if comp_name == 'residual':
                    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                    
                plot_idx += 1
        
        if title:
            axes[0].set_title(title)
        else:
            axes[0].set_title('Seasonal Decomposition')
            
        axes[-1].set_xlabel('Date')
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
            
        return fig
    
    def plot_seasonal_patterns(self,
                             series: pd.Series,
                             adjusted: pd.Series,
                             period: str = 'quarter',
                             save_path: Optional[str] = None) -> Figure:
        """
        Plot seasonal patterns by period
        
        Parameters:
        -----------
        series : pd.Series
            Original series
        adjusted : pd.Series
            Adjusted series
        period : str
            'quarter' or 'month'
        save_path : str, optional
            Path to save figure
            
        Returns:
        --------
        Figure
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.figsize[0]*1.2, self.figsize[1]*0.6))
        
        # Calculate seasonal factors
        if (series > 0).all() and (adjusted > 0).all():
            seasonal_factors = series / adjusted
        else:
            seasonal_factors = series - adjusted + 100  # Center around 100 for additive
            
        # Extract period
        if period == 'quarter':
            seasonal_factors = seasonal_factors.to_frame('value')
            seasonal_factors['period'] = seasonal_factors.index.quarter
        else:
            seasonal_factors = seasonal_factors.to_frame('value')
            seasonal_factors['period'] = seasonal_factors.index.month
            
        # Box plot
        seasonal_factors.boxplot(column='value', by='period', ax=ax1)
        ax1.set_xlabel(period.capitalize())
        ax1.set_ylabel('Seasonal Factor')
        ax1.set_title(f'Seasonal Factors by {period.capitalize()}')
        ax1.grid(True, alpha=0.3)
        
        # Time series of seasonal factors colored by period
        for p in seasonal_factors['period'].unique():
            mask = seasonal_factors['period'] == p
            data = seasonal_factors[mask]
            ax2.scatter(data.index, data['value'], 
                       label=f'{period.capitalize()} {p}',
                       alpha=0.6, s=30)
            
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Seasonal Factor')
        ax2.set_title('Seasonal Factors Over Time')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
            
        return fig
    
    def plot_multiple_series(self,
                           series_dict: Dict[str, pd.Series],
                           title: Optional[str] = None,
                           normalize: bool = True,
                           save_path: Optional[str] = None) -> Figure:
        """
        Plot multiple series for comparison
        
        Parameters:
        -----------
        series_dict : Dict[str, pd.Series]
            Dictionary of series to plot
        title : str, optional
            Plot title
        normalize : bool
            Whether to normalize series to 100 at start
        save_path : str, optional
            Path to save figure
            
        Returns:
        --------
        Figure
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for idx, (name, series) in enumerate(series_dict.items()):
            if normalize and len(series) > 0:
                series = series / series.iloc[0] * 100
                
            ax.plot(series.index, series.values, 
                   label=name, color=self.colors[idx % len(self.colors)],
                   linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Index Value' + (' (Normalized)' if normalize else ''))
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title('Series Comparison')
            
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
            
        return fig


class DiagnosticPlotter:
    """
    Create diagnostic plots for model validation
    
    Generates plots for:
    - Residual analysis
    - ACF/PACF plots
    - Q-Q plots
    - Model fit diagnostics
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize diagnostic plotter
        
        Parameters:
        -----------
        figsize : Tuple[int, int]
            Default figure size
        """
        self.figsize = figsize
        self.colors = sns.color_palette("husl", 8)
        
    def plot_residual_diagnostics(self,
                                residuals: pd.Series,
                                title: Optional[str] = None,
                                save_path: Optional[str] = None) -> Figure:
        """
        Create comprehensive residual diagnostic plots
        
        Parameters:
        -----------
        residuals : pd.Series
            Model residuals
        title : str, optional
            Plot title
        save_path : str, optional
            Path to save figure
            
        Returns:
        --------
        Figure
            Matplotlib figure
        """
        from scipy import stats
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        
        fig = plt.figure(figsize=(self.figsize[0], self.figsize[1]*1.5))
        
        # Create grid
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Residuals over time
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(residuals.index, residuals.values, color=self.colors[0], linewidth=1)
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax1.set_title('Residuals Over Time')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Residual')
        ax1.grid(True, alpha=0.3)
        
        # 2. Histogram with normal overlay
        ax2 = fig.add_subplot(gs[1, 0])
        n, bins, _ = ax2.hist(residuals.dropna(), bins=30, density=True, 
                             alpha=0.7, color=self.colors[1])
        
        # Fit normal distribution
        mu, sigma = stats.norm.fit(residuals.dropna())
        x = np.linspace(residuals.min(), residuals.max(), 100)
        ax2.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
                label=f'Normal(μ={mu:.2f}, σ={sigma:.2f})')
        ax2.set_title('Residual Distribution')
        ax2.set_xlabel('Residual')
        ax2.set_ylabel('Density')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Q-Q plot
        ax3 = fig.add_subplot(gs[1, 1])
        stats.probplot(residuals.dropna(), dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot')
        ax3.grid(True, alpha=0.3)
        
        # 4. ACF plot
        ax4 = fig.add_subplot(gs[2, 0])
        plot_acf(residuals.dropna(), lags=20, ax=ax4, alpha=0.05)
        ax4.set_title('Autocorrelation Function')
        ax4.grid(True, alpha=0.3)
        
        # 5. PACF plot
        ax5 = fig.add_subplot(gs[2, 1])
        try:
            plot_pacf(residuals.dropna(), lags=20, ax=ax5, alpha=0.05)
        except:
            # Handle case where PACF fails
            ax5.text(0.5, 0.5, 'PACF calculation failed', 
                    transform=ax5.transAxes, ha='center', va='center')
        ax5.set_title('Partial Autocorrelation Function')
        ax5.grid(True, alpha=0.3)
        
        if title:
            fig.suptitle(title, fontsize=14, y=0.98)
            
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
            
        return fig
    
    def plot_model_fit(self,
                      actual: pd.Series,
                      fitted: pd.Series,
                      residuals: Optional[pd.Series] = None,
                      title: Optional[str] = None,
                      save_path: Optional[str] = None) -> Figure:
        """
        Plot model fit diagnostics
        
        Parameters:
        -----------
        actual : pd.Series
            Actual values
        fitted : pd.Series
            Fitted values
        residuals : pd.Series, optional
            Model residuals
        title : str, optional
            Plot title
        save_path : str, optional
            Path to save figure
            
        Returns:
        --------
        Figure
            Matplotlib figure
        """
        n_plots = 3 if residuals is not None else 2
        fig, axes = plt.subplots(n_plots, 1, figsize=self.figsize, sharex=(n_plots==3))
        
        # 1. Actual vs Fitted
        ax1 = axes[0] if n_plots > 1 else axes
        ax1.plot(actual.index, actual.values, label='Actual', 
                color=self.colors[0], linewidth=2, alpha=0.8)
        ax1.plot(fitted.index, fitted.values, label='Fitted', 
                color=self.colors[1], linewidth=2, linestyle='--')
        ax1.set_ylabel('Value')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        if title:
            ax1.set_title(title)
        else:
            ax1.set_title('Model Fit: Actual vs Fitted')
            
        # 2. Scatter plot
        ax2 = axes[1] if n_plots > 1 else None
        if ax2 is not None:
            ax2.scatter(actual.values, fitted.values, alpha=0.6, color=self.colors[2])
            
            # Add 45-degree line
            min_val = min(actual.min(), fitted.min())
            max_val = max(actual.max(), fitted.max())
            ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            ax2.set_xlabel('Actual')
            ax2.set_ylabel('Fitted')
            ax2.set_title('Actual vs Fitted Values')
            ax2.grid(True, alpha=0.3)
            
            # Add R-squared
            from sklearn.metrics import r2_score
            r2 = r2_score(actual, fitted)
            ax2.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax2.transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 3. Residuals over time
        if residuals is not None and n_plots > 2:
            ax3 = axes[2]
            ax3.plot(residuals.index, residuals.values, color=self.colors[3], linewidth=1)
            ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Residual')
            ax3.set_title('Residuals Over Time')
            ax3.grid(True, alpha=0.3)
            
            # Format x-axis
            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax3.xaxis.set_major_locator(mdates.YearLocator(2))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
            
        return fig
    
    def plot_coefficient_analysis(self,
                                coefficients: pd.DataFrame,
                                title: Optional[str] = None,
                                save_path: Optional[str] = None) -> Figure:
        """
        Plot coefficient estimates with confidence intervals
        
        Parameters:
        -----------
        coefficients : pd.DataFrame
            DataFrame with columns: coefficient, std_error, p_value
        title : str, optional
            Plot title
        save_path : str, optional
            Path to save figure
            
        Returns:
        --------
        Figure
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.figsize[0]*1.2, self.figsize[1]*0.6))
        
        # Calculate confidence intervals
        coefficients = coefficients.copy()
        coefficients['ci_lower'] = coefficients['coefficient'] - 1.96 * coefficients['std_error']
        coefficients['ci_upper'] = coefficients['coefficient'] + 1.96 * coefficients['std_error']
        
        # Sort by coefficient magnitude
        coefficients = coefficients.sort_values('coefficient')
        
        # 1. Coefficient plot with error bars
        y_pos = np.arange(len(coefficients))
        ax1.errorbar(coefficients['coefficient'], y_pos,
                    xerr=[coefficients['coefficient'] - coefficients['ci_lower'],
                          coefficients['ci_upper'] - coefficients['coefficient']],
                    fmt='o', color=self.colors[0], capsize=5, capthick=2)
        ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(coefficients.index)
        ax1.set_xlabel('Coefficient Value')
        ax1.set_title('Coefficient Estimates (95% CI)')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # 2. P-value plot
        ax2.barh(y_pos, -np.log10(coefficients['p_value']), 
                color=self.colors[1], alpha=0.7)
        ax2.axvline(x=-np.log10(0.05), color='red', linestyle='--', 
                   label='p=0.05', alpha=0.8)
        ax2.axvline(x=-np.log10(0.01), color='darkred', linestyle='--', 
                   label='p=0.01', alpha=0.8)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(coefficients.index)
        ax2.set_xlabel('-log10(p-value)')
        ax2.set_title('Statistical Significance')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='x')
        
        if title:
            fig.suptitle(title, fontsize=14, y=0.98)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
            
        return fig
    
    def plot_stability_analysis(self,
                              results_over_time: Dict[str, pd.DataFrame],
                              metric: str = 'coefficient',
                              title: Optional[str] = None,
                              save_path: Optional[str] = None) -> Figure:
        """
        Plot parameter stability over time
        
        Parameters:
        -----------
        results_over_time : Dict[str, pd.DataFrame]
            Dictionary with time periods as keys and coefficient DataFrames as values
        metric : str
            Metric to plot ('coefficient', 'p_value', 't_stat')
        title : str, optional
            Plot title
        save_path : str, optional
            Path to save figure
            
        Returns:
        --------
        Figure
            Matplotlib figure
        """
        # Extract unique parameters
        all_params = set()
        for df in results_over_time.values():
            all_params.update(df.index)
        params = sorted(list(all_params))
        
        # Prepare data for plotting
        time_periods = sorted(results_over_time.keys())
        param_data = {param: [] for param in params}
        
        for period in time_periods:
            df = results_over_time[period]
            for param in params:
                if param in df.index:
                    param_data[param].append(df.loc[param, metric])
                else:
                    param_data[param].append(np.nan)
        
        # Create plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for idx, (param, values) in enumerate(param_data.items()):
            ax.plot(time_periods, values, marker='o', 
                   label=param, color=self.colors[idx % len(self.colors)],
                   linewidth=2, markersize=6)
        
        ax.set_xlabel('Time Period')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        if metric == 'coefficient':
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'{metric.replace("_", " ").title()} Stability Over Time')
            
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
            
        return fig