"""Quantile regression model implementation"""

from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import SplineTransformer
import statsmodels.formula.api as smf
from loguru import logger


class QuantileRegressionModel:
    """
    Quantile regression model for distributional analysis
    
    Implements: Qτ(yit) = ατ + β1τWit + β2τN.salesit + β3τChari,2010 + β4τIndustryi,2010 + γiτ + fτ(year) + εit
    """
    
    def __init__(self, 
                 quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9],
                 use_statsmodels: bool = True,
                 spline_df: int = 4):
        """
        Initialize quantile regression model
        
        Parameters:
        -----------
        quantiles : List[float]
            Quantiles to estimate
        use_statsmodels : bool
            Whether to use statsmodels (for formula interface) or sklearn
        spline_df : int
            Degrees of freedom for year splines
        """
        self.quantiles = quantiles
        self.use_statsmodels = use_statsmodels
        self.spline_df = spline_df
        
        self.models = {}
        self.results = {}
        self.feature_names = None
        self.spline_transformer = None
        
    def fit(self,
            data: pd.DataFrame,
            y_col: str,
            weather_col: str,
            sales_col: str,
            char_cols: List[str],
            industry_cols: List[str],
            entity_col: str,
            year_col: str) -> 'QuantileRegressionModel':
        """
        Fit quantile regression model
        
        Parameters:
        -----------
        data : pd.DataFrame
            Panel data
        y_col : str
            Dependent variable column
        weather_col : str
            Weather variable column
        sales_col : str
            Number of sales column
        char_cols : List[str]
            2010 characteristic columns
        industry_cols : List[str]
            2010 industry share columns
        entity_col : str
            Entity/location identifier
        year_col : str
            Year column for spline transformation
            
        Returns:
        --------
        self : QuantileRegressionModel
            Fitted model
        """
        logger.info(f"Fitting quantile regression for quantiles: {self.quantiles}")
        
        # Prepare data
        X, y, feature_info = self._prepare_data(
            data, y_col, weather_col, sales_col, 
            char_cols, industry_cols, entity_col, year_col
        )
        
        self.feature_names = list(X.columns)
        
        # Fit model for each quantile
        for tau in self.quantiles:
            logger.info(f"Fitting quantile {tau}")
            
            if self.use_statsmodels:
                # Build formula
                formula = self._build_formula(
                    y_col, weather_col, sales_col, 
                    char_cols, industry_cols, entity_col, year_col
                )
                
                # Fit using statsmodels
                model = smf.quantreg(formula, data=data)
                result = model.fit(q=tau)
                
                self.models[tau] = model
                self.results[tau] = result
                
            else:
                # Fit using sklearn
                model = QuantileRegressor(
                    quantile=tau,
                    alpha=0,
                    solver='highs'
                )
                model.fit(X, y)
                
                self.models[tau] = model
                self.results[tau] = self._extract_sklearn_results(model, X, y, tau)
        
        return self
    
    def predict(self, 
                data: pd.DataFrame,
                quantile: Optional[float] = None) -> Union[pd.Series, pd.DataFrame]:
        """
        Generate predictions
        
        Parameters:
        -----------
        data : pd.DataFrame
            Feature data
        quantile : float, optional
            Specific quantile to predict (if None, predict all)
            
        Returns:
        --------
        pd.Series or pd.DataFrame
            Predictions
        """
        if not self.models:
            raise ValueError("Model must be fitted before prediction")
        
        if quantile is not None:
            # Predict single quantile
            if quantile not in self.models:
                raise ValueError(f"Quantile {quantile} not fitted")
                
            if self.use_statsmodels:
                return self.results[quantile].predict(data)
            else:
                X = self._transform_features(data)
                return pd.Series(
                    self.models[quantile].predict(X),
                    index=data.index
                )
        else:
            # Predict all quantiles
            predictions = pd.DataFrame(index=data.index)
            
            for tau in self.quantiles:
                if self.use_statsmodels:
                    predictions[f'q{int(tau*100)}'] = self.results[tau].predict(data)
                else:
                    X = self._transform_features(data)
                    predictions[f'q{int(tau*100)}'] = self.models[tau].predict(X)
            
            return predictions
    
    def get_coefficients(self, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get coefficients across quantiles
        
        Parameters:
        -----------
        feature_names : List[str], optional
            Specific features to return
            
        Returns:
        --------
        pd.DataFrame
            Coefficients by quantile
        """
        coef_df = pd.DataFrame()
        
        for tau in self.quantiles:
            if self.use_statsmodels:
                coef = self.results[tau].params
                if feature_names:
                    coef = coef[coef.index.isin(feature_names)]
                coef_df[f'q{int(tau*100)}'] = coef
            else:
                coef = pd.Series(
                    self.models[tau].coef_,
                    index=self.feature_names
                )
                if feature_names:
                    coef = coef[coef.index.isin(feature_names)]
                coef_df[f'q{int(tau*100)}'] = coef
        
        return coef_df.T
    
    def test_coefficient_equality(self, 
                                 feature: str,
                                 quantile1: float,
                                 quantile2: float) -> Dict[str, float]:
        """
        Test equality of coefficients across quantiles
        
        Parameters:
        -----------
        feature : str
            Feature to test
        quantile1 : float
            First quantile
        quantile2 : float
            Second quantile
            
        Returns:
        --------
        Dict[str, float]
            Test statistic and p-value
        """
        if not self.use_statsmodels:
            logger.warning("Coefficient testing requires statsmodels")
            return {"statistic": np.nan, "pvalue": np.nan}
        
        # Get coefficients and standard errors
        coef1 = self.results[quantile1].params[feature]
        coef2 = self.results[quantile2].params[feature]
        se1 = self.results[quantile1].bse[feature]
        se2 = self.results[quantile2].bse[feature]
        
        # Wald test
        diff = coef1 - coef2
        se_diff = np.sqrt(se1**2 + se2**2)
        z_stat = diff / se_diff
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        return {
            "statistic": z_stat,
            "pvalue": p_value,
            "reject_equality": p_value < 0.05
        }
    
    def _prepare_data(self,
                     data: pd.DataFrame,
                     y_col: str,
                     weather_col: str,
                     sales_col: str,
                     char_cols: List[str],
                     industry_cols: List[str],
                     entity_col: str,
                     year_col: str) -> Tuple[pd.DataFrame, pd.Series, Dict]:
        """Prepare data for quantile regression"""
        
        # Create feature matrix
        feature_cols = [weather_col, sales_col] + char_cols + industry_cols
        X = data[feature_cols].copy()
        
        # Add entity fixed effects
        entity_dummies = pd.get_dummies(data[entity_col], prefix='entity')
        X = pd.concat([X, entity_dummies], axis=1)
        
        # Add year splines
        years = data[year_col].values.reshape(-1, 1)
        if self.spline_transformer is None:
            self.spline_transformer = SplineTransformer(
                n_knots=self.spline_df,
                degree=3
            )
            year_splines = self.spline_transformer.fit_transform(years)
        else:
            year_splines = self.spline_transformer.transform(years)
        
        spline_cols = [f'year_spline_{i}' for i in range(year_splines.shape[1])]
        year_spline_df = pd.DataFrame(
            year_splines,
            columns=spline_cols,
            index=data.index
        )
        X = pd.concat([X, year_spline_df], axis=1)
        
        # Target variable
        y = data[y_col]
        
        # Store feature information
        feature_info = {
            'weather': weather_col,
            'sales': sales_col,
            'characteristics': char_cols,
            'industry': industry_cols,
            'entities': list(entity_dummies.columns),
            'splines': spline_cols
        }
        
        return X, y, feature_info
    
    def _build_formula(self,
                      y_col: str,
                      weather_col: str,
                      sales_col: str,
                      char_cols: List[str],
                      industry_cols: List[str],
                      entity_col: str,
                      year_col: str) -> str:
        """Build formula for statsmodels"""
        
        # Base terms
        terms = [weather_col, sales_col]
        
        # Add characteristic and industry terms
        terms.extend(char_cols)
        terms.extend(industry_cols)
        
        # Add entity fixed effects
        terms.append(f'C({entity_col})')
        
        # Add year splines
        terms.append(f'bs({year_col}, df={self.spline_df})')
        
        # Build formula
        formula = f"{y_col} ~ " + " + ".join(terms)
        
        return formula
    
    def _transform_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform features for sklearn prediction"""
        # This would need to match the transformation done during fitting
        # For now, return as-is (assuming pre-transformed)
        return data[self.feature_names]
    
    def _extract_sklearn_results(self, 
                               model: QuantileRegressor,
                               X: pd.DataFrame,
                               y: pd.Series,
                               tau: float) -> Dict:
        """Extract results from sklearn model"""
        
        # Calculate residuals
        predictions = model.predict(X)
        residuals = y - predictions
        
        # Basic statistics
        results = {
            'coefficients': pd.Series(model.coef_, index=self.feature_names),
            'intercept': model.intercept_ if hasattr(model, 'intercept_') else 0,
            'quantile': tau,
            'n_obs': len(y),
            'residual_std': residuals.std(),
            'mae': np.abs(residuals).mean()
        }
        
        return results
    
    def plot_coefficients(self, 
                         features: Optional[List[str]] = None,
                         figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot coefficient evolution across quantiles
        
        Parameters:
        -----------
        features : List[str], optional
            Features to plot (default: weather and sales)
        figsize : Tuple[int, int]
            Figure size
        """
        import matplotlib.pyplot as plt
        
        if features is None:
            features = ['weather', 'sales']  # Default to main variables
        
        coef_df = self.get_coefficients(features)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for feature in features:
            if feature in coef_df.columns:
                ax.plot(self.quantiles, coef_df[feature], 
                       marker='o', label=feature, linewidth=2)
        
        ax.set_xlabel('Quantile')
        ax.set_ylabel('Coefficient')
        ax.set_title('Quantile Regression Coefficients')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def summary(self, quantile: Optional[float] = None) -> str:
        """
        Get model summary
        
        Parameters:
        -----------
        quantile : float, optional
            Specific quantile to summarize (default: median)
            
        Returns:
        --------
        str
            Model summary
        """
        if quantile is None:
            quantile = 0.5  # Default to median
        
        if quantile not in self.results:
            raise ValueError(f"Quantile {quantile} not fitted")
        
        if self.use_statsmodels:
            return str(self.results[quantile].summary())
        else:
            # Create custom summary for sklearn
            results = self.results[quantile]
            summary = []
            summary.append(f"Quantile Regression Results (τ = {quantile})")
            summary.append("=" * 50)
            summary.append(f"Number of observations: {results['n_obs']}")
            summary.append(f"Residual std: {results['residual_std']:.4f}")
            summary.append(f"MAE: {results['mae']:.4f}")
            summary.append("\nCoefficients:")
            summary.append("-" * 30)
            
            for name, coef in results['coefficients'].items():
                summary.append(f"{name:30s}: {coef:10.4f}")
            
            return "\n".join(summary)