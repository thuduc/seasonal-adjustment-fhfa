"""Linear regression model implementation"""

from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm
from loguru import logger


class LinearRegressionModel:
    """
    Linear regression model for seasonality analysis
    
    Implements: yit = α + β1Wit + β2N.salesit + β3Chari,2010 + β4Industryi,2010 + γi + f(year) + εit
    """
    
    def __init__(self, use_statsmodels: bool = True):
        """
        Initialize linear regression model
        
        Parameters:
        -----------
        use_statsmodels : bool
            Whether to use statsmodels (for detailed statistics) or sklearn
        """
        self.use_statsmodels = use_statsmodels
        self.model = None
        self.results = None
        self.feature_names = None
        self.coefficients = None
        self.diagnostics = {}
    
    def fit(self,
            X: pd.DataFrame,
            y: pd.Series,
            add_constant: bool = True) -> 'LinearRegressionModel':
        """
        Fit linear regression model
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        add_constant : bool
            Whether to add intercept term
            
        Returns:
        --------
        self : LinearRegressionModel
            Fitted model
        """
        logger.info(f"Fitting linear regression with {X.shape[1]} features")
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        if self.use_statsmodels:
            # Statsmodels implementation
            if add_constant:
                X_with_const = sm.add_constant(X)
            else:
                X_with_const = X
            
            self.model = sm.OLS(y, X_with_const)
            self.results = self.model.fit()
            
            # Extract coefficients
            self.coefficients = self.results.params
            
            # Store diagnostics
            self._calculate_diagnostics()
            
        else:
            # Sklearn implementation
            self.model = LinearRegression(fit_intercept=add_constant)
            self.model.fit(X, y)
            
            # Extract coefficients
            if add_constant:
                self.coefficients = pd.Series(
                    [self.model.intercept_] + list(self.model.coef_),
                    index=['const'] + self.feature_names
                )
            else:
                self.coefficients = pd.Series(
                    self.model.coef_,
                    index=self.feature_names
                )
        
        return self
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Generate predictions
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
            
        Returns:
        --------
        pd.Series
            Predictions
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        if self.use_statsmodels:
            if 'const' in self.coefficients.index:
                X_with_const = sm.add_constant(X)
            else:
                X_with_const = X
            predictions = self.results.predict(X_with_const)
        else:
            predictions = pd.Series(
                self.model.predict(X),
                index=X.index
            )
        
        return predictions
    
    def get_coefficients(self, 
                        feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get model coefficients with statistics
        
        Parameters:
        -----------
        feature_names : List[str], optional
            Specific features to return
            
        Returns:
        --------
        pd.DataFrame
            Coefficient statistics
        """
        if self.use_statsmodels and self.results is not None:
            # Full statistics from statsmodels
            coef_df = pd.DataFrame({
                'coefficient': self.results.params,
                'std_error': self.results.bse,
                't_statistic': self.results.tvalues,
                'p_value': self.results.pvalues,
                'conf_lower': self.results.conf_int()[0],
                'conf_upper': self.results.conf_int()[1]
            })
        else:
            # Basic coefficients only
            coef_df = pd.DataFrame({
                'coefficient': self.coefficients
            })
        
        if feature_names:
            coef_df = coef_df.loc[feature_names]
        
        return coef_df
    
    def _calculate_diagnostics(self) -> None:
        """Calculate regression diagnostics"""
        if not self.use_statsmodels or self.results is None:
            return
        
        # Basic statistics
        self.diagnostics['r_squared'] = self.results.rsquared
        self.diagnostics['adjusted_r_squared'] = self.results.rsquared_adj
        self.diagnostics['f_statistic'] = self.results.fvalue
        self.diagnostics['f_pvalue'] = self.results.f_pvalue
        self.diagnostics['aic'] = self.results.aic
        self.diagnostics['bic'] = self.results.bic
        
        # Residual diagnostics
        residuals = self.results.resid
        self.diagnostics['residual_std'] = residuals.std()
        self.diagnostics['durbin_watson'] = sm.stats.durbin_watson(residuals)
        
        # Heteroscedasticity test
        bp_test = sm.stats.diagnostic.het_breuschpagan(
            residuals,
            self.results.model.exog
        )
        self.diagnostics['breusch_pagan_stat'] = bp_test[0]
        self.diagnostics['breusch_pagan_pvalue'] = bp_test[1]
        
        # Normality test
        jb_test = sm.stats.jarque_bera(residuals)
        self.diagnostics['jarque_bera_stat'] = jb_test[0]
        self.diagnostics['jarque_bera_pvalue'] = jb_test[1]
    
    def summary(self) -> Union[str, None]:
        """
        Get model summary
        
        Returns:
        --------
        str or None
            Model summary (if using statsmodels)
        """
        if self.use_statsmodels and self.results is not None:
            return str(self.results.summary())
        else:
            # Create custom summary for sklearn
            summary = []
            summary.append("Linear Regression Results")
            summary.append("=" * 50)
            summary.append(f"Number of observations: {len(self.model.coef_)}")
            summary.append(f"Number of features: {len(self.feature_names)}")
            summary.append("\nCoefficients:")
            summary.append("-" * 30)
            for name, coef in zip(self.feature_names, self.coefficients):
                summary.append(f"{name:20s}: {coef:10.4f}")
            
            return "\n".join(summary)
    
    def get_feature_importance(self) -> pd.Series:
        """
        Get feature importance (absolute standardized coefficients)
        
        Returns:
        --------
        pd.Series
            Feature importance scores
        """
        if self.coefficients is None:
            raise ValueError("Model must be fitted first")
        
        # Get absolute values of coefficients
        importance = self.coefficients.abs()
        
        # Remove intercept if present
        if 'const' in importance.index:
            importance = importance.drop('const')
        
        # Sort by importance
        importance = importance.sort_values(ascending=False)
        
        return importance
    
    def residual_analysis(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, any]:
        """
        Perform residual analysis
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            True values
            
        Returns:
        --------
        Dict[str, any]
            Residual statistics
        """
        predictions = self.predict(X)
        residuals = y - predictions
        
        analysis = {
            'mean': residuals.mean(),
            'std': residuals.std(),
            'min': residuals.min(),
            'max': residuals.max(),
            'skewness': residuals.skew(),
            'kurtosis': residuals.kurtosis(),
            'autocorrelation': residuals.autocorr() if len(residuals) > 1 else np.nan
        }
        
        return analysis