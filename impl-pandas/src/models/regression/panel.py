"""Panel data regression models for seasonality impact analysis"""

from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from linearmodels import PanelOLS, RandomEffects, BetweenOLS
from linearmodels.panel import compare
import statsmodels.api as sm
from sklearn.preprocessing import SplineTransformer
from scipy import stats
from loguru import logger


class SeasonalityImpactModel:
    """
    Panel data models for analyzing seasonality impact
    
    Implements various specifications:
    1. Fixed effects: yit = α + β1Wit + β2N.salesit + β3Chari,2010 + β4Industryi,2010 + γi + f(year) + εit
    2. Random effects variant
    3. Quarter-specific coefficients
    """
    
    def __init__(self,
                 model_type: str = "fixed_effects",
                 time_effects: bool = True,
                 entity_effects: bool = True,
                 clustered_errors: bool = True,
                 spline_df: int = 4):
        """
        Initialize panel regression model
        
        Parameters:
        -----------
        model_type : str
            Type of panel model: 'fixed_effects', 'random_effects', 'between'
        time_effects : bool
            Include time fixed effects
        entity_effects : bool
            Include entity fixed effects
        clustered_errors : bool
            Use clustered standard errors
        spline_df : int
            Degrees of freedom for year splines
        """
        self.model_type = model_type
        self.time_effects = time_effects
        self.entity_effects = entity_effects
        self.clustered_errors = clustered_errors
        self.spline_df = spline_df
        
        self.model = None
        self.results = None
        self.feature_names = None
        self.spline_transformer = None
        self.quarter_models = {}
        
    def fit(self,
            data: pd.DataFrame,
            y_col: str,
            weather_col: str,
            sales_col: str,
            char_cols: List[str],
            industry_cols: List[str],
            entity_col: str,
            time_col: str,
            quarter_col: Optional[str] = None) -> 'SeasonalityImpactModel':
        """
        Fit panel regression model
        
        Parameters:
        -----------
        data : pd.DataFrame
            Panel data
        y_col : str
            Dependent variable
        weather_col : str
            Weather variable
        sales_col : str
            Number of sales
        char_cols : List[str]
            2010 characteristics
        industry_cols : List[str]
            2010 industry shares
        entity_col : str
            Entity identifier
        time_col : str
            Time identifier
        quarter_col : str, optional
            Quarter identifier for quarter-specific models
            
        Returns:
        --------
        self : SeasonalityImpactModel
            Fitted model
        """
        logger.info(f"Fitting {self.model_type} panel model")
        
        # Set panel index
        panel_data = data.set_index([entity_col, time_col])
        
        # Prepare features
        X, y = self._prepare_panel_data(
            panel_data, y_col, weather_col, sales_col,
            char_cols, industry_cols
        )
        
        self.feature_names = list(X.columns)
        
        # Fit main model
        self._fit_panel_model(X, y, panel_data)
        
        # Fit quarter-specific models if requested
        if quarter_col is not None:
            self._fit_quarter_models(data, y_col, weather_col, sales_col,
                                   char_cols, industry_cols, entity_col,
                                   time_col, quarter_col)
        
        return self
    
    def _fit_panel_model(self, X: pd.DataFrame, y: pd.Series, 
                        panel_data: pd.DataFrame) -> None:
        """Fit the panel regression model"""
        
        if self.model_type == "fixed_effects":
            self.model = PanelOLS(
                y, X,
                entity_effects=self.entity_effects,
                time_effects=self.time_effects,
                drop_absorbed=True
            )
        elif self.model_type == "random_effects":
            self.model = RandomEffects(y, X)
        elif self.model_type == "between":
            self.model = BetweenOLS(y, X)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Fit model
        if self.clustered_errors and self.model_type == "fixed_effects":
            self.results = self.model.fit(cov_type='clustered', cluster_entity=True)
        else:
            self.results = self.model.fit()
        
        logger.info(f"Model fit complete. R-squared: {self.results.rsquared:.4f}")
    
    def _fit_quarter_models(self,
                           data: pd.DataFrame,
                           y_col: str,
                           weather_col: str,
                           sales_col: str,
                           char_cols: List[str],
                           industry_cols: List[str],
                           entity_col: str,
                           time_col: str,
                           quarter_col: str) -> None:
        """Fit separate models for each quarter"""
        
        logger.info("Fitting quarter-specific models")
        
        for quarter in [1, 2, 3, 4]:
            quarter_data = data[data[quarter_col] == quarter]
            
            if len(quarter_data) == 0:
                logger.warning(f"No data for quarter {quarter}")
                continue
            
            # Set panel index
            panel_data = quarter_data.set_index([entity_col, time_col])
            
            # Prepare features
            X, y = self._prepare_panel_data(
                panel_data, y_col, weather_col, sales_col,
                char_cols, industry_cols
            )
            
            # Fit model
            if self.model_type == "fixed_effects":
                model = PanelOLS(
                    y, X,
                    entity_effects=self.entity_effects,
                    time_effects=self.time_effects,
                    drop_absorbed=True
                )
            else:
                model = RandomEffects(y, X)
            
            if self.clustered_errors:
                results = model.fit(cov_type='clustered', cluster_entity=True)
            else:
                results = model.fit()
            
            self.quarter_models[quarter] = {
                'model': model,
                'results': results
            }
            
            logger.info(f"Quarter {quarter} R-squared: {results.rsquared:.4f}")
    
    def _prepare_panel_data(self,
                           panel_data: pd.DataFrame,
                           y_col: str,
                           weather_col: str,
                           sales_col: str,
                           char_cols: List[str],
                           industry_cols: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for panel regression"""
        
        # Extract features
        feature_cols = [weather_col, sales_col] + char_cols + industry_cols
        X = panel_data[feature_cols].copy()
        
        # Add year splines if requested
        if self.spline_df > 0:
            years = pd.to_datetime(panel_data.index.get_level_values(1)).year
            years_array = years.values.reshape(-1, 1)
            
            if self.spline_transformer is None:
                self.spline_transformer = SplineTransformer(
                    n_knots=self.spline_df,
                    degree=3
                )
                year_splines = self.spline_transformer.fit_transform(years_array)
            else:
                year_splines = self.spline_transformer.transform(years_array)
            
            # Add splines to feature matrix
            for i in range(year_splines.shape[1]):
                X[f'year_spline_{i}'] = year_splines[:, i]
        
        # Target variable
        y = panel_data[y_col]
        
        return X, y
    
    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate predictions
        
        Parameters:
        -----------
        data : pd.DataFrame
            Feature data with panel structure
            
        Returns:
        --------
        pd.Series
            Predictions
        """
        if self.results is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Ensure panel structure
        if not isinstance(data.index, pd.MultiIndex):
            raise ValueError("Data must have MultiIndex for panel structure")
        
        # Prepare features
        X = data[self.feature_names]
        
        # Generate predictions
        predictions = self.results.predict(X)
        
        return predictions
    
    def get_coefficients(self, quarter: Optional[int] = None) -> pd.DataFrame:
        """
        Get model coefficients
        
        Parameters:
        -----------
        quarter : int, optional
            Specific quarter (1-4) for quarter models
            
        Returns:
        --------
        pd.DataFrame
            Coefficient estimates with statistics
        """
        if quarter is not None:
            if quarter not in self.quarter_models:
                raise ValueError(f"No model for quarter {quarter}")
            results = self.quarter_models[quarter]['results']
        else:
            results = self.results
        
        coef_df = pd.DataFrame({
            'coefficient': results.params,
            'std_error': results.std_errors,
            't_statistic': results.tstats,
            'p_value': results.pvalues,
            'conf_lower': results.conf_int()['lower'],
            'conf_upper': results.conf_int()['upper']
        })
        
        return coef_df
    
    def test_coefficient_equality(self, 
                                 feature: str,
                                 quarter1: int,
                                 quarter2: int) -> Dict[str, float]:
        """
        Test equality of coefficients across quarters
        
        Parameters:
        -----------
        feature : str
            Feature to test
        quarter1 : int
            First quarter
        quarter2 : int
            Second quarter
            
        Returns:
        --------
        Dict[str, float]
            Test results
        """
        if quarter1 not in self.quarter_models or quarter2 not in self.quarter_models:
            raise ValueError("Both quarters must have fitted models")
        
        # Get coefficients and standard errors
        coef1 = self.quarter_models[quarter1]['results'].params[feature]
        coef2 = self.quarter_models[quarter2]['results'].params[feature]
        se1 = self.quarter_models[quarter1]['results'].std_errors[feature]
        se2 = self.quarter_models[quarter2]['results'].std_errors[feature]
        
        # Wald test
        diff = coef1 - coef2
        se_diff = np.sqrt(se1**2 + se2**2)
        z_stat = diff / se_diff
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        return {
            "coefficient_q1": coef1,
            "coefficient_q2": coef2,
            "difference": diff,
            "statistic": z_stat,
            "pvalue": p_value,
            "reject_equality": p_value < 0.05
        }
    
    def hausman_test(self, alternative_model: 'SeasonalityImpactModel') -> Dict[str, float]:
        """
        Perform Hausman test between two model specifications
        
        Parameters:
        -----------
        alternative_model : SeasonalityImpactModel
            Alternative model specification
            
        Returns:
        --------
        Dict[str, float]
            Test statistic and p-value
        """
        if self.results is None or alternative_model.results is None:
            raise ValueError("Both models must be fitted")
        
        # Use linearmodels compare function
        comparison = compare({
            self.model_type: self.results,
            alternative_model.model_type: alternative_model.results
        })
        
        return {
            "statistic": comparison.j_stat,
            "pvalue": comparison.pval,
            "reject_null": comparison.pval < 0.05,
            "preferred_model": self.model_type if comparison.pval < 0.05 else alternative_model.model_type
        }
    
    def residual_diagnostics(self) -> Dict[str, any]:
        """
        Perform residual diagnostics
        
        Returns:
        --------
        Dict[str, any]
            Diagnostic test results
        """
        if self.results is None:
            raise ValueError("Model must be fitted")
        
        residuals = self.results.resids
        
        # Basic statistics
        diagnostics = {
            'mean': residuals.mean(),
            'std': residuals.std(),
            'skewness': residuals.skew(),
            'kurtosis': residuals.kurtosis()
        }
        
        # Normality test
        from scipy import stats
        jb_stat, jb_pvalue = stats.jarque_bera(residuals)
        diagnostics['jarque_bera'] = {
            'statistic': jb_stat,
            'pvalue': jb_pvalue,
            'reject_normality': jb_pvalue < 0.05
        }
        
        # Serial correlation test (if time series)
        if len(residuals) > 20:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb_result = acorr_ljungbox(residuals, lags=10, return_df=True)
            diagnostics['serial_correlation'] = {
                'min_pvalue': lb_result['lb_pvalue'].min(),
                'reject_no_correlation': lb_result['lb_pvalue'].min() < 0.05
            }
        
        return diagnostics
    
    def summary(self, quarter: Optional[int] = None) -> str:
        """
        Get model summary
        
        Parameters:
        -----------
        quarter : int, optional
            Specific quarter for quarter models
            
        Returns:
        --------
        str
            Model summary
        """
        if quarter is not None:
            if quarter not in self.quarter_models:
                raise ValueError(f"No model for quarter {quarter}")
            return str(self.quarter_models[quarter]['results'])
        else:
            if self.results is None:
                raise ValueError("Model not fitted")
            return str(self.results)
    
    def plot_quarter_coefficients(self, 
                                 features: List[str],
                                 figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot coefficients across quarters
        
        Parameters:
        -----------
        features : List[str]
            Features to plot
        figsize : Tuple[int, int]
            Figure size
        """
        import matplotlib.pyplot as plt
        
        if not self.quarter_models:
            raise ValueError("Quarter models not fitted")
        
        quarters = sorted(self.quarter_models.keys())
        n_features = len(features)
        
        fig, axes = plt.subplots(1, n_features, figsize=figsize)
        if n_features == 1:
            axes = [axes]
        
        for idx, feature in enumerate(features):
            ax = axes[idx]
            
            coefficients = []
            lower_bounds = []
            upper_bounds = []
            
            for q in quarters:
                coef_df = self.get_coefficients(quarter=q)
                if feature in coef_df.index:
                    coefficients.append(coef_df.loc[feature, 'coefficient'])
                    lower_bounds.append(coef_df.loc[feature, 'conf_lower'])
                    upper_bounds.append(coef_df.loc[feature, 'conf_upper'])
                else:
                    coefficients.append(np.nan)
                    lower_bounds.append(np.nan)
                    upper_bounds.append(np.nan)
            
            # Plot coefficients with confidence intervals
            ax.plot(quarters, coefficients, 'o-', markersize=8, linewidth=2)
            ax.fill_between(quarters, lower_bounds, upper_bounds, alpha=0.3)
            
            ax.set_xlabel('Quarter')
            ax.set_ylabel('Coefficient')
            ax.set_title(f'{feature} Coefficient by Quarter')
            ax.grid(True, alpha=0.3)
            ax.set_xticks(quarters)
        
        plt.tight_layout()
        plt.show()
    
    def export_results(self, filepath: str) -> None:
        """
        Export results to file
        
        Parameters:
        -----------
        filepath : str
            Path to save results
        """
        results_dict = {
            'model_type': self.model_type,
            'main_results': self.get_coefficients().to_dict(),
            'diagnostics': self.residual_diagnostics()
        }
        
        if self.quarter_models:
            results_dict['quarter_results'] = {}
            for q, model_data in self.quarter_models.items():
                results_dict['quarter_results'][f'Q{q}'] = self.get_coefficients(quarter=q).to_dict()
        
        import json
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)