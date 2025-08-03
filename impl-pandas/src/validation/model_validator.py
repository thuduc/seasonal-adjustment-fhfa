"""Model validation for checking model assumptions and diagnostics"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.stats.stattools import jarque_bera
from loguru import logger


class ModelValidator:
    """Validate model assumptions and perform diagnostic tests"""
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize model validator
        
        Parameters
        ----------
        significance_level : float
            Significance level for hypothesis tests
        """
        self.significance_level = significance_level
        
    def validate_arima_assumptions(self, residuals: pd.Series,
                                 fitted_values: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Validate ARIMA model assumptions
        
        Parameters
        ----------
        residuals : pd.Series
            Model residuals
        fitted_values : Optional[pd.Series]
            Fitted values for additional tests
            
        Returns
        -------
        Dict[str, Any]
            Validation results
        """
        logger.info("Validating ARIMA model assumptions")
        
        results = {
            'assumptions_met': True,
            'tests': {}
        }
        
        # Remove NaN values
        clean_residuals = residuals.dropna()
        
        if len(clean_residuals) < 10:
            logger.warning("Insufficient residuals for diagnostic tests")
            return {
                'assumptions_met': False,
                'tests': {'error': 'Insufficient data for tests'}
            }
        
        # Test 1: Normality of residuals (Jarque-Bera test)
        jb_result = jarque_bera(clean_residuals)
        jb_stat = jb_result[0]
        jb_pvalue = jb_result[1]
        results['tests']['normality'] = {
            'test': 'Jarque-Bera',
            'statistic': float(jb_stat),
            'p_value': float(jb_pvalue),
            'passed': bool(jb_pvalue > self.significance_level),
            'interpretation': 'Residuals are normally distributed' if jb_pvalue > self.significance_level 
                            else 'Residuals are not normally distributed'
        }
        
        # Test 2: No autocorrelation (Ljung-Box test)
        lb_result = acorr_ljungbox(clean_residuals, lags=min(10, len(clean_residuals)//4), return_df=True)
        min_pvalue = lb_result['lb_pvalue'].min()
        
        results['tests']['autocorrelation'] = {
            'test': 'Ljung-Box',
            'min_p_value': float(min_pvalue),
            'passed': bool(min_pvalue > self.significance_level),
            'interpretation': 'No significant autocorrelation' if min_pvalue > self.significance_level
                            else 'Significant autocorrelation detected'
        }
        
        # Test 3: Homoscedasticity (if fitted values provided)
        if fitted_values is not None and len(fitted_values) == len(clean_residuals):
            try:
                # Create design matrix for Breusch-Pagan test
                X = np.column_stack([np.ones(len(fitted_values)), fitted_values])
                bp_stat, bp_pvalue, _, _ = het_breuschpagan(clean_residuals, X)
                
                results['tests']['homoscedasticity'] = {
                    'test': 'Breusch-Pagan',
                    'statistic': float(bp_stat),
                    'p_value': float(bp_pvalue),
                    'passed': bool(bp_pvalue > self.significance_level),
                    'interpretation': 'Constant variance (homoscedastic)' if bp_pvalue > self.significance_level
                                    else 'Non-constant variance (heteroscedastic)'
                }
            except Exception as e:
                logger.warning(f"Could not perform Breusch-Pagan test: {e}")
        
        # Test 4: Zero mean of residuals
        t_stat, t_pvalue = stats.ttest_1samp(clean_residuals, 0)
        results['tests']['zero_mean'] = {
            'test': 'One-sample t-test',
            'mean': float(clean_residuals.mean()),
            'statistic': float(t_stat),
            'p_value': float(t_pvalue),
            'passed': bool(t_pvalue > self.significance_level),
            'interpretation': 'Residuals have zero mean' if t_pvalue > self.significance_level
                            else 'Residuals do not have zero mean'
        }
        
        # Update overall result
        results['assumptions_met'] = all(test['passed'] for test in results['tests'].values() 
                                       if 'passed' in test)
        
        return results
    
    def validate_regression_assumptions(self, residuals: pd.Series,
                                      X: pd.DataFrame,
                                      y: pd.Series) -> Dict[str, Any]:
        """
        Validate regression model assumptions
        
        Parameters
        ----------
        residuals : pd.Series
            Model residuals
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
            
        Returns
        -------
        Dict[str, Any]
            Validation results
        """
        logger.info("Validating regression model assumptions")
        
        results = {
            'assumptions_met': True,
            'tests': {}
        }
        
        # Remove NaN values
        mask = ~(residuals.isna() | X.isna().any(axis=1) | y.isna())
        clean_residuals = residuals[mask]
        clean_X = X[mask]
        clean_y = y[mask]
        
        if len(clean_residuals) < 10:
            logger.warning("Insufficient data for diagnostic tests")
            return {
                'assumptions_met': False,
                'tests': {'error': 'Insufficient data for tests'}
            }
        
        # Test 1: Linearity (Rainbow test approximation)
        # Split data and test if model performs differently
        n_split = len(clean_residuals) // 2
        residuals_1 = clean_residuals.iloc[:n_split]
        residuals_2 = clean_residuals.iloc[n_split:]
        
        f_stat, f_pvalue = stats.f_oneway(residuals_1, residuals_2)
        results['tests']['linearity'] = {
            'test': 'F-test (split sample)',
            'statistic': float(f_stat),
            'p_value': float(f_pvalue),
            'passed': bool(f_pvalue > self.significance_level),
            'interpretation': 'Linear relationship' if f_pvalue > self.significance_level
                            else 'Potential non-linear relationship'
        }
        
        # Test 2: Multicollinearity (VIF)
        vif_results = self._calculate_vif(clean_X)
        max_vif = max(vif_results.values()) if vif_results else 0
        
        results['tests']['multicollinearity'] = {
            'test': 'Variance Inflation Factor',
            'max_vif': float(max_vif),
            'passed': bool(max_vif < 10),
            'interpretation': 'No severe multicollinearity' if max_vif < 10
                            else 'Severe multicollinearity detected',
            'details': vif_results
        }
        
        # Test 3: Normality of residuals
        jb_result = jarque_bera(clean_residuals)
        jb_stat = jb_result[0]
        jb_pvalue = jb_result[1]
        results['tests']['normality'] = {
            'test': 'Jarque-Bera',
            'statistic': float(jb_stat),
            'p_value': float(jb_pvalue),
            'passed': bool(jb_pvalue > self.significance_level),
            'interpretation': 'Residuals are normally distributed' if jb_pvalue > self.significance_level
                            else 'Residuals are not normally distributed'
        }
        
        # Test 4: Homoscedasticity
        try:
            X_with_const = np.column_stack([np.ones(len(clean_X)), clean_X])
            bp_stat, bp_pvalue, _, _ = het_breuschpagan(clean_residuals, X_with_const)
            
            results['tests']['homoscedasticity'] = {
                'test': 'Breusch-Pagan',
                'statistic': float(bp_stat),
                'p_value': float(bp_pvalue),
                'passed': bp_pvalue > self.significance_level,
                'interpretation': 'Constant variance' if bp_pvalue > self.significance_level
                                else 'Non-constant variance detected'
            }
        except Exception as e:
            logger.warning(f"Could not perform Breusch-Pagan test: {e}")
        
        # Update overall result
        results['assumptions_met'] = all(test['passed'] for test in results['tests'].values()
                                       if 'passed' in test)
        
        return results
    
    def validate_panel_regression(self, model_results: Any,
                                residuals: pd.Series) -> Dict[str, Any]:
        """
        Validate panel regression model
        
        Parameters
        ----------
        model_results : Any
            Panel model results object
        residuals : pd.Series
            Model residuals
            
        Returns
        -------
        Dict[str, Any]
            Validation results
        """
        logger.info("Validating panel regression model")
        
        results = {
            'model_valid': True,
            'tests': {}
        }
        
        # Test 1: F-test for model significance
        if hasattr(model_results, 'f_statistic'):
            results['tests']['model_significance'] = {
                'test': 'F-test',
                'statistic': float(model_results.f_statistic.stat),
                'p_value': float(model_results.f_statistic.pval),
                'passed': bool(model_results.f_statistic.pval < self.significance_level),
                'interpretation': 'Model is statistically significant' 
                                if model_results.f_statistic.pval < self.significance_level
                                else 'Model is not statistically significant'
            }
        
        # Test 2: Hausman test recommendation (if available)
        if hasattr(model_results, 'entity_effects') and hasattr(model_results, 'time_effects'):
            results['tests']['effects_structure'] = {
                'entity_effects': bool(model_results.entity_effects),
                'time_effects': bool(model_results.time_effects),
                'interpretation': 'Fixed effects model with appropriate structure'
            }
        
        # Test 3: Check for serial correlation in panel residuals
        if len(residuals) > 20:
            lb_result = acorr_ljungbox(residuals.dropna(), lags=5, return_df=True)
            min_pvalue = lb_result['lb_pvalue'].min()
            
            results['tests']['serial_correlation'] = {
                'test': 'Ljung-Box',
                'min_p_value': float(min_pvalue),
                'passed': bool(min_pvalue > self.significance_level),
                'interpretation': 'No serial correlation' if min_pvalue > self.significance_level
                                else 'Serial correlation detected'
            }
        
        # Update overall result
        results['model_valid'] = all(test.get('passed', True) for test in results['tests'].values())
        
        return results
    
    def validate_seasonal_adjustment(self, original: pd.Series,
                                   adjusted: pd.Series,
                                   frequency: int = 4) -> Dict[str, Any]:
        """
        Validate seasonal adjustment effectiveness
        
        Parameters
        ----------
        original : pd.Series
            Original series
        adjusted : pd.Series
            Seasonally adjusted series
        frequency : int
            Seasonal frequency (4 for quarterly)
            
        Returns
        -------
        Dict[str, Any]
            Validation results
        """
        logger.info("Validating seasonal adjustment effectiveness")
        
        results = {
            'adjustment_effective': True,
            'metrics': {}
        }
        
        # Calculate seasonal factors
        if len(original) >= frequency * 2:
            # Simple seasonal strength measure
            if hasattr(original.index, 'quarter'):
                orig_seasonal_var = original.groupby(original.index.quarter).mean().var()
                adj_seasonal_var = adjusted.groupby(adjusted.index.quarter).mean().var()
                
                reduction_ratio = 1 - (adj_seasonal_var / orig_seasonal_var) if orig_seasonal_var > 0 else 0
                
                results['metrics']['seasonal_variance_reduction'] = {
                    'original_seasonal_variance': float(orig_seasonal_var),
                    'adjusted_seasonal_variance': float(adj_seasonal_var),
                    'reduction_ratio': float(reduction_ratio),
                    'effective': bool(reduction_ratio > 0.5)
                }
        
        # Check if trend is preserved
        if len(original) > 10:
            orig_trend = np.polyfit(range(len(original)), original.values, 1)[0]
            adj_trend = np.polyfit(range(len(adjusted)), adjusted.values, 1)[0]
            
            trend_preserved = np.sign(orig_trend) == np.sign(adj_trend)
            
            results['metrics']['trend_preservation'] = {
                'original_trend': float(orig_trend),
                'adjusted_trend': float(adj_trend),
                'preserved': bool(trend_preserved)
            }
        
        # Check variance stability
        orig_cv = original.std() / original.mean() if original.mean() != 0 else 0
        adj_cv = adjusted.std() / adjusted.mean() if adjusted.mean() != 0 else 0
        
        results['metrics']['variance_stability'] = {
            'original_cv': float(orig_cv),
            'adjusted_cv': float(adj_cv),
            'stable': adj_cv <= orig_cv
        }
        
        # Update overall result
        results['adjustment_effective'] = all(
            metric.get('effective', metric.get('preserved', metric.get('stable', True)))
            for metric in results['metrics'].values()
        )
        
        return results
    
    def _calculate_vif(self, X: pd.DataFrame) -> Dict[str, float]:
        """Calculate Variance Inflation Factors"""
        vif_data = {}
        
        if X.shape[1] < 2:
            return vif_data
            
        for i, col in enumerate(X.columns):
            try:
                # Regress each variable on all others
                X_temp = X.drop(columns=[col])
                
                # Handle constant columns
                if X_temp.std().min() == 0:
                    continue
                    
                # Calculate R-squared
                from sklearn.linear_model import LinearRegression
                lr = LinearRegression()
                lr.fit(X_temp, X[col])
                r_squared = lr.score(X_temp, X[col])
                
                # Calculate VIF
                vif = 1 / (1 - r_squared) if r_squared < 0.999 else 999
                vif_data[col] = vif
                
            except Exception as e:
                logger.warning(f"Could not calculate VIF for {col}: {e}")
                
        return vif_data
    
    def generate_diagnostic_plots(self, residuals: pd.Series) -> Dict[str, Any]:
        """
        Generate diagnostic plot data
        
        Parameters
        ----------
        residuals : pd.Series
            Model residuals
            
        Returns
        -------
        Dict[str, Any]
            Plot data for visualization
        """
        clean_residuals = residuals.dropna()
        
        # Q-Q plot data
        theoretical_quantiles = np.linspace(0.01, 0.99, len(clean_residuals))
        theoretical_values = stats.norm.ppf(theoretical_quantiles)
        empirical_values = np.sort(clean_residuals)
        
        # ACF data
        from statsmodels.tsa.stattools import acf
        acf_values = acf(clean_residuals, nlags=min(20, len(clean_residuals)//4))
        
        return {
            'qq_plot': {
                'theoretical': theoretical_values.tolist(),
                'empirical': empirical_values.tolist()
            },
            'acf_plot': {
                'lags': list(range(len(acf_values))),
                'values': acf_values.tolist()
            },
            'histogram': {
                'values': clean_residuals.tolist(),
                'bins': 30
            }
        }