import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm
import warnings

from .feature_engineering import FeatureEngineer

warnings.filterwarnings('ignore')
logger = logging.getLogger('seasonal_adjustment.causation_analysis')


class CausationAnalysis:
    """Analyzes causes of housing market seasonality using regression models."""
    
    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
    def prepare_regression_data(self, master_df: pd.DataFrame,
                              target_type: str = 'absolute') -> pd.DataFrame:
        """Prepare data for regression analysis."""
        return self.feature_engineer.prepare_regression_data(master_df, target_type)
    
    def fit_linear_model(self, df: pd.DataFrame, 
                        feature_cols: Optional[List[str]] = None) -> Dict[str, any]:
        """Fit OLS regression model to explain seasonality."""
        logger.info("Fitting linear regression model")
        
        if feature_cols is None:
            feature_cols = self.feature_engineer.get_feature_names()
            
        # Filter to available columns
        available_features = [col for col in feature_cols if col in df.columns]
        
        if not available_features:
            raise ValueError("No features available for regression")
            
        # Prepare data
        X = df[available_features].values
        y = df['target'].values
        
        # Remove any remaining NaN values
        # First convert to numeric arrays if needed
        if hasattr(X, 'values'):
            X_values = X.values
        else:
            X_values = X
            
        if hasattr(y, 'values'):
            y_values = y.values
        else:
            y_values = y
            
        # Create mask for non-NaN values
        mask = ~(np.isnan(X_values.astype(float)).any(axis=1) | np.isnan(y_values.astype(float)))
        X = X_values[mask]
        y = y_values[mask]
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Add constant
        X_with_const = sm.add_constant(X_scaled)
        
        # Fit model using statsmodels for detailed statistics
        model = sm.OLS(y, X_with_const)
        results = model.fit()
        
        # Store model
        self.models['linear'] = results
        
        # Extract key results
        coefficients = dict(zip(['const'] + available_features, results.params))
        p_values = dict(zip(['const'] + available_features, results.pvalues))
        conf_intervals = results.conf_int()
        
        # Identify significant predictors
        significant_features = [
            feat for feat, p_val in p_values.items() 
            if p_val < 0.05 and feat != 'const'
        ]
        
        # Calculate feature importance (absolute standardized coefficients)
        feature_importance = {}
        for i, feat in enumerate(available_features):
            # Standardized coefficient = beta * std(X) / std(y)
            std_coef = abs(results.params[i+1] * np.std(X_scaled[:, i]))
            feature_importance[feat] = std_coef
            
        # Sort by importance
        sorted_features = sorted(
            feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        results_dict = {
            'model': results,
            'coefficients': coefficients,
            'p_values': p_values,
            'confidence_intervals': {
                feat: tuple(conf_intervals[i])
                for i, feat in enumerate(['const'] + available_features)
            },
            'r_squared': results.rsquared,
            'adj_r_squared': results.rsquared_adj,
            'f_statistic': results.fvalue,
            'f_pvalue': results.f_pvalue,
            'significant_features': significant_features,
            'feature_importance': sorted_features[:10],  # Top 10
            'n_observations': len(y),
            'aic': results.aic,
            'bic': results.bic
        }
        
        self.results['linear'] = results_dict
        
        logger.info(f"Linear model R²: {results.rsquared:.4f}, "
                   f"Significant features: {len(significant_features)}")
        
        return results_dict
    
    def fit_quantile_regression(self, df: pd.DataFrame,
                              quantiles: List[float] = None,
                              feature_cols: Optional[List[str]] = None) -> Dict[str, any]:
        """Fit quantile regression models."""
        logger.info("Fitting quantile regression models")
        
        if quantiles is None:
            quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
            
        if feature_cols is None:
            feature_cols = self.feature_engineer.get_feature_names()
            
        # Filter to available columns
        available_features = [col for col in feature_cols if col in df.columns]
        
        # Prepare data
        X = df[available_features].values
        y = df['target'].values
        
        # Remove NaN values
        # First convert to numeric arrays if needed
        if hasattr(X, 'values'):
            X_values = X.values
        else:
            X_values = X
            
        if hasattr(y, 'values'):
            y_values = y.values
        else:
            y_values = y
            
        # Create mask for non-NaN values
        mask = ~(np.isnan(X_values.astype(float)).any(axis=1) | np.isnan(y_values.astype(float)))
        X = X_values[mask]
        y = y_values[mask]
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        quantile_results = {}
        
        for q in quantiles:
            logger.debug(f"Fitting quantile {q}")
            
            # Fit quantile regression
            qr = QuantileRegressor(quantile=q, alpha=0, solver='highs')
            qr.fit(X_scaled, y)
            
            # Calculate predictions and metrics
            y_pred = qr.predict(X_scaled)
            
            # Pseudo R-squared for quantile regression
            # 1 - (sum of weighted absolute deviations / total weighted deviations)
            weights = np.where(y >= y_pred, q, 1-q)
            wad = np.sum(weights * np.abs(y - y_pred))
            null_pred = np.quantile(y, q)
            total_wad = np.sum(weights * np.abs(y - null_pred))
            pseudo_r2 = 1 - wad / total_wad
            
            # Store results
            quantile_results[q] = {
                'model': qr,
                'coefficients': dict(zip(available_features, qr.coef_)),
                'intercept': qr.intercept_,
                'pseudo_r_squared': pseudo_r2,
                'predictions': y_pred
            }
            
        # Analyze how coefficients change across quantiles
        coef_by_quantile = pd.DataFrame({
            q: results['coefficients'] 
            for q, results in quantile_results.items()
        }).T
        
        # Identify features with varying effects across quantiles
        varying_features = []
        for feat in available_features:
            if feat in coef_by_quantile.columns:
                coef_range = coef_by_quantile[feat].max() - coef_by_quantile[feat].min()
                coef_std = coef_by_quantile[feat].std()
                if coef_std > 0.1:  # Threshold for variation
                    varying_features.append({
                        'feature': feat,
                        'range': coef_range,
                        'std': coef_std,
                        'coefficients': coef_by_quantile[feat].to_dict()
                    })
                    
        # Sort by variation
        varying_features.sort(key=lambda x: x['std'], reverse=True)
        
        results_dict = {
            'quantile_models': quantile_results,
            'coefficient_matrix': coef_by_quantile,
            'varying_features': varying_features[:10],  # Top 10
            'quantiles': quantiles
        }
        
        self.results['quantile'] = results_dict
        
        logger.info(f"Fitted {len(quantiles)} quantile models")
        
        return results_dict
    
    def analyze_geographic_patterns(self, results: Dict[str, any],
                                   geography_data: pd.DataFrame) -> pd.DataFrame:
        """Identify regions with high seasonality based on model results."""
        logger.info("Analyzing geographic patterns")
        
        if 'linear' not in results:
            raise ValueError("Linear model results required for geographic analysis")
            
        linear_model = results['linear']['model']
        
        # Get geographic fixed effects if available
        geo_effects = {}
        for var_name, coef in zip(linear_model.model.exog_names, linear_model.params):
            if var_name.startswith('geo_'):
                geo_id = var_name.replace('geo_', '')
                geo_effects[geo_id] = coef
                
        # Calculate predicted seasonality by geography
        geo_seasonality = []
        
        for geo_id in geography_data['geography_id'].unique():
            geo_data = geography_data[geography_data['geography_id'] == geo_id]
            
            if not geo_data.empty:
                # Calculate average seasonality
                if 'target' in geo_data.columns:
                    avg_seasonality = geo_data['target'].mean()
                else:
                    avg_seasonality = np.nan
                    
                # Get fixed effect
                fixed_effect = geo_effects.get(geo_id, 0)
                
                # Get average weather conditions
                weather_features = {}
                for col in ['temp_range', 'avg_temp', 'precipitation']:
                    if col in geo_data.columns:
                        weather_features[col] = geo_data[col].mean()
                        
                geo_seasonality.append({
                    'geography_id': geo_id,
                    'avg_seasonality': avg_seasonality,
                    'fixed_effect': fixed_effect,
                    'seasonality_rank': 0,  # Will be filled later
                    **weather_features
                })
                
        # Create DataFrame and rank
        geo_df = pd.DataFrame(geo_seasonality)
        geo_df['seasonality_rank'] = geo_df['avg_seasonality'].rank(ascending=False, na_option='bottom')
        
        # Identify high seasonality regions (top quartile)
        threshold = geo_df['avg_seasonality'].quantile(0.75)
        geo_df['high_seasonality'] = geo_df['avg_seasonality'] > threshold
        
        # Sort by seasonality
        geo_df = geo_df.sort_values('avg_seasonality', ascending=False)
        
        logger.info(f"Identified {geo_df['high_seasonality'].sum()} high seasonality regions")
        
        return geo_df
    
    def calculate_weather_impact(self, model_results: Dict[str, any]) -> Dict[str, float]:
        """Calculate the contribution of weather variables to seasonality."""
        logger.info("Calculating weather impact")
        
        if 'linear' not in model_results:
            raise ValueError("Linear model results required")
            
        coefficients = model_results['linear']['coefficients']
        
        # Identify weather-related features
        weather_features = [
            feat for feat in coefficients.keys()
            if any(weather_term in feat for weather_term in 
                  ['temp_', 'precipitation', 'weather'])
        ]
        
        # Calculate total and weather-specific R-squared
        # This is approximate - ideally would use partial R-squared
        weather_impact = {}
        
        for feat in weather_features:
            if feat in model_results['linear']['feature_importance']:
                importance = dict(model_results['linear']['feature_importance'])
                weather_impact[feat] = importance.get(feat, 0)
                
        # Calculate total weather contribution
        total_weather_impact = sum(weather_impact.values())
        
        # Separate by weather type
        impact_by_type = {
            'temperature': sum(v for k, v in weather_impact.items() if 'temp' in k),
            'precipitation': sum(v for k, v in weather_impact.items() if 'precip' in k),
            'interactions': sum(v for k, v in weather_impact.items() if '_x_' in k)
        }
        
        return {
            'total_impact': total_weather_impact,
            'by_feature': weather_impact,
            'by_type': impact_by_type,
            'top_weather_features': sorted(
                weather_impact.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
        }
    
    def run_full_analysis(self, master_df: pd.DataFrame,
                         quantiles: Optional[List[float]] = None) -> Dict[str, any]:
        """Run complete causation analysis pipeline."""
        logger.info("Running full causation analysis")
        
        # Prepare data
        regression_df = self.prepare_regression_data(master_df)
        
        # Fit linear model
        linear_results = self.fit_linear_model(regression_df)
        
        # Fit quantile regression
        quantile_results = self.fit_quantile_regression(regression_df, quantiles)
        
        # Analyze geographic patterns
        geo_patterns = self.analyze_geographic_patterns(
            {'linear': linear_results}, 
            master_df
        )
        
        # Calculate weather impact
        weather_impact = self.calculate_weather_impact({'linear': linear_results})
        
        # Compile all results
        full_results = {
            'linear_model': linear_results,
            'quantile_models': quantile_results,
            'geographic_patterns': geo_patterns,
            'weather_impact': weather_impact,
            'feature_names': self.feature_engineer.get_feature_names(),
            'n_observations': len(regression_df),
            'target_summary': {
                'mean': regression_df['target'].mean(),
                'std': regression_df['target'].std(),
                'min': regression_df['target'].min(),
                'max': regression_df['target'].max()
            }
        }
        
        logger.info("Causation analysis complete")
        
        return full_results