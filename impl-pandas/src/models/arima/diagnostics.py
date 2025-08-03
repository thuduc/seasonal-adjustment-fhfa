"""ARIMA model diagnostics"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
from loguru import logger


class ARIMADiagnostics:
    """Comprehensive diagnostics for ARIMA models"""
    
    def __init__(self, residuals: pd.Series, fitted_values: pd.Series):
        """
        Initialize diagnostics
        
        Parameters:
        -----------
        residuals : pd.Series
            Model residuals
        fitted_values : pd.Series
            Fitted values from model
        """
        self.residuals = residuals
        self.fitted_values = fitted_values
        self.diagnostics_results = {}
    
    def run_all_tests(self) -> Dict[str, any]:
        """Run all diagnostic tests"""
        logger.info("Running comprehensive ARIMA diagnostics")
        
        # Residual tests
        self.diagnostics_results['residual_tests'] = self.test_residuals()
        
        # Autocorrelation tests
        self.diagnostics_results['autocorrelation'] = self.test_autocorrelation()
        
        # Normality tests
        self.diagnostics_results['normality'] = self.test_normality()
        
        # Heteroscedasticity tests
        self.diagnostics_results['heteroscedasticity'] = self.test_heteroscedasticity()
        
        # Stationarity tests
        self.diagnostics_results['stationarity'] = self.test_stationarity()
        
        # Overall assessment
        self.diagnostics_results['overall_pass'] = self._overall_assessment()
        
        return self.diagnostics_results
    
    def test_residuals(self) -> Dict[str, float]:
        """Basic residual statistics"""
        return {
            'mean': self.residuals.mean(),
            'std': self.residuals.std(),
            'skewness': stats.skew(self.residuals),
            'kurtosis': stats.kurtosis(self.residuals),
            'min': self.residuals.min(),
            'max': self.residuals.max(),
            'n_observations': len(self.residuals)
        }
    
    def test_autocorrelation(self, lags: int = 20) -> Dict[str, any]:
        """Test for residual autocorrelation"""
        # Ljung-Box test
        lb_test = acorr_ljungbox(self.residuals, lags=lags, return_df=True)
        
        # Find first significant lag
        significant_lags = lb_test[lb_test['lb_pvalue'] < 0.05]
        first_significant = significant_lags.index[0] if len(significant_lags) > 0 else None
        
        return {
            'ljung_box_stats': lb_test['lb_stat'].to_dict(),
            'ljung_box_pvalues': lb_test['lb_pvalue'].to_dict(),
            'min_pvalue': lb_test['lb_pvalue'].min(),
            'pass_test': lb_test['lb_pvalue'].min() > 0.05,
            'first_significant_lag': first_significant
        }
    
    def test_normality(self) -> Dict[str, any]:
        """Test residual normality"""
        # Jarque-Bera test
        jb_stat, jb_pvalue = stats.jarque_bera(self.residuals)
        
        # Shapiro-Wilk test (for smaller samples)
        if len(self.residuals) <= 5000:
            sw_stat, sw_pvalue = stats.shapiro(self.residuals)
        else:
            sw_stat, sw_pvalue = np.nan, np.nan
        
        # Anderson-Darling test
        ad_result = stats.anderson(self.residuals, dist='norm')
        
        return {
            'jarque_bera': {
                'statistic': jb_stat,
                'pvalue': jb_pvalue,
                'pass': jb_pvalue > 0.05
            },
            'shapiro_wilk': {
                'statistic': sw_stat,
                'pvalue': sw_pvalue,
                'pass': sw_pvalue > 0.05 if not np.isnan(sw_pvalue) else None
            },
            'anderson_darling': {
                'statistic': ad_result.statistic,
                'critical_values': dict(zip(ad_result.significance_level, ad_result.critical_values)),
                'pass': ad_result.statistic < ad_result.critical_values[2]  # 5% level
            }
        }
    
    def test_heteroscedasticity(self) -> Dict[str, any]:
        """Test for heteroscedasticity"""
        # Breusch-Pagan test
        bp_stat, bp_pvalue = self._breusch_pagan_test()
        
        # White's test
        white_stat, white_pvalue = self._whites_test()
        
        # ARCH test
        arch_stat, arch_pvalue = self._arch_test()
        
        return {
            'breusch_pagan': {
                'statistic': bp_stat,
                'pvalue': bp_pvalue,
                'pass': bp_pvalue > 0.05
            },
            'white': {
                'statistic': white_stat,
                'pvalue': white_pvalue,
                'pass': white_pvalue > 0.05
            },
            'arch': {
                'statistic': arch_stat,
                'pvalue': arch_pvalue,
                'pass': arch_pvalue > 0.05
            }
        }
    
    def test_stationarity(self) -> Dict[str, any]:
        """Test residual stationarity"""
        from statsmodels.tsa.stattools import adfuller, kpss
        
        # ADF test
        adf_result = adfuller(self.residuals, autolag='AIC')
        
        # KPSS test
        kpss_result = kpss(self.residuals, regression='c', nlags='auto')
        
        return {
            'adf': {
                'statistic': adf_result[0],
                'pvalue': adf_result[1],
                'critical_values': adf_result[4],
                'pass': adf_result[1] < 0.05  # Reject null of unit root
            },
            'kpss': {
                'statistic': kpss_result[0],
                'pvalue': kpss_result[1],
                'critical_values': kpss_result[3],
                'pass': kpss_result[1] > 0.05  # Fail to reject null of stationarity
            }
        }
    
    def _breusch_pagan_test(self) -> Tuple[float, float]:
        """Breusch-Pagan test implementation"""
        residuals_sq = self.residuals ** 2
        
        # Simple regression of squared residuals on fitted values
        X = np.column_stack([np.ones(len(self.fitted_values)), self.fitted_values])
        beta = np.linalg.lstsq(X, residuals_sq, rcond=None)[0]
        predicted = X @ beta
        
        # Calculate R-squared
        ssr = np.sum((predicted - residuals_sq.mean()) ** 2)
        sst = np.sum((residuals_sq - residuals_sq.mean()) ** 2)
        r_squared = ssr / sst
        
        # Test statistic
        n = len(self.residuals)
        lm_statistic = n * r_squared
        p_value = 1 - stats.chi2.cdf(lm_statistic, df=1)
        
        return lm_statistic, p_value
    
    def _whites_test(self) -> Tuple[float, float]:
        """White's test for heteroscedasticity"""
        residuals_sq = self.residuals ** 2
        
        # Create regressors: fitted values and squared fitted values
        X = np.column_stack([
            np.ones(len(self.fitted_values)),
            self.fitted_values,
            self.fitted_values ** 2
        ])
        
        # Regression
        beta = np.linalg.lstsq(X, residuals_sq, rcond=None)[0]
        predicted = X @ beta
        
        # Calculate R-squared
        ssr = np.sum((predicted - residuals_sq.mean()) ** 2)
        sst = np.sum((residuals_sq - residuals_sq.mean()) ** 2)
        r_squared = ssr / sst
        
        # Test statistic
        n = len(self.residuals)
        lm_statistic = n * r_squared
        p_value = 1 - stats.chi2.cdf(lm_statistic, df=2)
        
        return lm_statistic, p_value
    
    def _arch_test(self, lags: int = 5) -> Tuple[float, float]:
        """ARCH test for conditional heteroscedasticity"""
        residuals_sq = self.residuals ** 2
        
        # Create lagged squared residuals
        n = len(residuals_sq)
        X = np.ones((n - lags, lags + 1))
        
        for i in range(1, lags + 1):
            X[:, i] = residuals_sq.iloc[:-i].values[:n-lags]
        
        y = residuals_sq.iloc[lags:].values
        
        # Regression
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        predicted = X @ beta
        
        # Calculate R-squared
        ssr = np.sum((predicted - y.mean()) ** 2)
        sst = np.sum((y - y.mean()) ** 2)
        r_squared = ssr / sst
        
        # Test statistic
        lm_statistic = (n - lags) * r_squared
        p_value = 1 - stats.chi2.cdf(lm_statistic, df=lags)
        
        return lm_statistic, p_value
    
    def _overall_assessment(self) -> bool:
        """Overall model assessment"""
        # Check key tests
        autocorr_pass = self.diagnostics_results['autocorrelation']['pass_test']
        normality_pass = self.diagnostics_results['normality']['jarque_bera']['pass']
        hetero_pass = self.diagnostics_results['heteroscedasticity']['breusch_pagan']['pass']
        
        return autocorr_pass and normality_pass and hetero_pass
    
    def plot_diagnostics(self, figsize: Tuple[int, int] = (15, 10)) -> None:
        """Create diagnostic plots"""
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('ARIMA Model Diagnostics', fontsize=16)
        
        # 1. Residuals over time
        ax = axes[0, 0]
        self.residuals.plot(ax=ax)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_title('Residuals')
        ax.set_xlabel('Time')
        ax.set_ylabel('Residual')
        
        # 2. Residual histogram
        ax = axes[0, 1]
        self.residuals.hist(bins=30, ax=ax, edgecolor='black')
        ax.set_title('Residual Distribution')
        ax.set_xlabel('Residual')
        ax.set_ylabel('Frequency')
        
        # Add normal distribution overlay
        x = np.linspace(self.residuals.min(), self.residuals.max(), 100)
        ax2 = ax.twinx()
        ax2.plot(x, stats.norm.pdf(x, self.residuals.mean(), self.residuals.std()), 
                'r-', label='Normal')
        ax2.set_ylabel('Density')
        ax2.legend()
        
        # 3. Q-Q plot
        ax = axes[0, 2]
        stats.probplot(self.residuals, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot')
        
        # 4. ACF of residuals
        ax = axes[1, 0]
        plot_acf(self.residuals, lags=20, ax=ax)
        ax.set_title('Residual ACF')
        
        # 5. PACF of residuals
        ax = axes[1, 1]
        plot_pacf(self.residuals, lags=20, ax=ax, method='ywm')
        ax.set_title('Residual PACF')
        
        # 6. Residuals vs Fitted
        ax = axes[1, 2]
        ax.scatter(self.fitted_values, self.residuals, alpha=0.5)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_title('Residuals vs Fitted Values')
        ax.set_xlabel('Fitted Values')
        ax.set_ylabel('Residuals')
        
        # Add lowess smoother
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smoothed = lowess(self.residuals, self.fitted_values, frac=0.3)
        ax.plot(smoothed[:, 0], smoothed[:, 1], 'r-', linewidth=2)
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self) -> str:
        """Generate text report of diagnostics"""
        if not self.diagnostics_results:
            self.run_all_tests()
        
        report = []
        report.append("="*60)
        report.append("ARIMA MODEL DIAGNOSTIC REPORT")
        report.append("="*60)
        
        # Residual statistics
        report.append("\n1. RESIDUAL STATISTICS")
        report.append("-"*30)
        res_stats = self.diagnostics_results['residual_tests']
        report.append(f"Mean: {res_stats['mean']:.6f}")
        report.append(f"Std Dev: {res_stats['std']:.4f}")
        report.append(f"Skewness: {res_stats['skewness']:.4f}")
        report.append(f"Kurtosis: {res_stats['kurtosis']:.4f}")
        
        # Autocorrelation
        report.append("\n2. AUTOCORRELATION TESTS")
        report.append("-"*30)
        autocorr = self.diagnostics_results['autocorrelation']
        report.append(f"Ljung-Box Test: {'PASS' if autocorr['pass_test'] else 'FAIL'}")
        report.append(f"Minimum p-value: {autocorr['min_pvalue']:.4f}")
        
        # Normality
        report.append("\n3. NORMALITY TESTS")
        report.append("-"*30)
        norm = self.diagnostics_results['normality']
        report.append(f"Jarque-Bera: {'PASS' if norm['jarque_bera']['pass'] else 'FAIL'} "
                     f"(p-value: {norm['jarque_bera']['pvalue']:.4f})")
        
        # Heteroscedasticity
        report.append("\n4. HETEROSCEDASTICITY TESTS")
        report.append("-"*30)
        hetero = self.diagnostics_results['heteroscedasticity']
        report.append(f"Breusch-Pagan: {'PASS' if hetero['breusch_pagan']['pass'] else 'FAIL'} "
                     f"(p-value: {hetero['breusch_pagan']['pvalue']:.4f})")
        report.append(f"ARCH: {'PASS' if hetero['arch']['pass'] else 'FAIL'} "
                     f"(p-value: {hetero['arch']['pvalue']:.4f})")
        
        # Overall assessment
        report.append("\n5. OVERALL ASSESSMENT")
        report.append("-"*30)
        report.append(f"Model Adequacy: {'ADEQUATE' if self.diagnostics_results['overall_pass'] else 'INADEQUATE'}")
        
        return "\n".join(report)