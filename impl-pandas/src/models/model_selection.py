import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
import logging
from itertools import product
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger('seasonal_adjustment.model_selection')


class ModelSelector:
    """Automatic ARIMA model order selection using grid search and information criteria."""
    
    def __init__(self, max_p: int = 4, max_d: int = 2, max_q: int = 4,
                 max_P: int = 2, max_D: int = 1, max_Q: int = 2,
                 seasonal_period: int = 4):
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.max_P = max_P
        self.max_D = max_D
        self.max_Q = max_Q
        self.seasonal_period = seasonal_period
        
    def determine_differencing_order(self, series: pd.Series) -> Tuple[int, int]:
        """Determine optimal differencing orders (d, D) using unit root tests."""
        logger.debug("Determining differencing order")
        
        # Test for non-seasonal differencing
        d = 0
        diff_series = series.copy()
        
        for i in range(self.max_d):
            adf_result = adfuller(diff_series.dropna())
            kpss_result = kpss(diff_series.dropna(), regression='c')
            
            # Series is stationary if ADF rejects null and KPSS fails to reject
            if adf_result[1] < 0.05 and kpss_result[1] > 0.05:
                break
            else:
                diff_series = diff_series.diff().dropna()
                d = i + 1
                
        # Test for seasonal differencing
        D = 0
        seasonal_diff = series.copy()
        
        for i in range(self.max_D):
            # Apply seasonal differencing
            seasonal_diff = seasonal_diff.diff(self.seasonal_period).dropna()
            
            adf_result = adfuller(seasonal_diff)
            
            if adf_result[1] < 0.05:
                D = i
                break
            else:
                D = i + 1
                
        logger.info(f"Selected differencing orders: d={d}, D={D}")
        return d, D
    
    def grid_search_arima(self, series: pd.Series, 
                         d: Optional[int] = None, 
                         D: Optional[int] = None) -> Dict[str, any]:
        """Test multiple ARIMA specifications and select best based on IC."""
        logger.debug("Starting ARIMA grid search")
        
        # Check if series is constant
        if series.std() == 0:
            # For constant series, return a simple model
            logger.warning("Series is constant, returning simple model")
            # Create a dummy model result
            return {
                'best_model': {
                    'order': (0, 0, 0),
                    'seasonal_order': (0, 0, 0, self.seasonal_period),
                    'aic': np.inf,
                    'bic': np.inf,
                    'rmse': 0,
                    'model': None
                },
                'top_models': [],
                'all_results': []
            }
        
        # Determine differencing if not provided
        if d is None or D is None:
            d_auto, D_auto = self.determine_differencing_order(series)
            d = d if d is not None else d_auto
            D = D if D is not None else D_auto
            
        # Generate parameter combinations
        p_values = range(0, self.max_p + 1)
        q_values = range(0, self.max_q + 1)
        P_values = range(0, self.max_P + 1)
        Q_values = range(0, self.max_Q + 1)
        
        results = []
        
        for p, q, P, Q in product(p_values, q_values, P_values, Q_values):
            # Skip invalid combinations
            if p == 0 and q == 0:
                continue
                
            order = (p, d, q)
            seasonal_order = (P, D, Q, self.seasonal_period)
            
            try:
                model = ARIMA(series, order=order, seasonal_order=seasonal_order)
                fitted = model.fit(method_kwargs={'maxiter': 500})
                
                # Calculate information criteria
                aic = fitted.aic
                bic = fitted.bic
                
                # Calculate in-sample RMSE
                residuals = fitted.resid
                rmse = np.sqrt(np.mean(residuals**2))
                
                results.append({
                    'order': order,
                    'seasonal_order': seasonal_order,
                    'aic': aic,
                    'bic': bic,
                    'rmse': rmse,
                    'model': fitted
                })
                
                logger.debug(f"Fitted ARIMA{order}x{seasonal_order}: AIC={aic:.2f}, BIC={bic:.2f}")
                
            except Exception as e:
                logger.debug(f"Failed to fit ARIMA{order}x{seasonal_order}: {str(e)}")
                continue
                
        if not results:
            # If no models could be fitted, return a default result
            logger.warning("No valid ARIMA models could be fitted")
            return {
                'best_model': {
                    'order': (1, 0, 1),
                    'seasonal_order': (0, 0, 0, self.seasonal_period),
                    'aic': np.inf,
                    'bic': np.inf,
                    'rmse': series.std(),
                    'model': None
                },
                'top_models': [],
                'all_results': []
            }
            
        # Sort by AIC
        results.sort(key=lambda x: x['aic'])
        
        # Return top 5 models
        top_models = results[:5]
        
        logger.info(f"Best model: ARIMA{top_models[0]['order']}x{top_models[0]['seasonal_order']} "
                   f"with AIC={top_models[0]['aic']:.2f}")
        
        return {
            'best_model': top_models[0],
            'top_models': top_models,
            'all_results': results
        }
    
    def validate_model(self, model: ARIMA, series: pd.Series, 
                      test_size: int = 8) -> Dict[str, any]:
        """Validate model using out-of-sample forecasts."""
        logger.debug(f"Validating model with {test_size} period forecast")
        
        # Split data
        train_size = len(series) - test_size
        train_data = series.iloc[:train_size]
        test_data = series.iloc[train_size:]
        
        # Refit model on training data
        train_model = ARIMA(train_data, 
                           order=model.model.order,
                           seasonal_order=model.model.seasonal_order)
        train_fitted = train_model.fit()
        
        # Generate forecasts
        forecasts = train_fitted.forecast(steps=test_size)
        
        # Calculate forecast accuracy metrics
        errors = test_data - forecasts
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors**2))
        mape = np.mean(np.abs(errors / test_data)) * 100
        
        # Direction accuracy
        actual_changes = test_data.diff().dropna()
        forecast_changes = forecasts.diff().dropna()
        direction_accuracy = np.mean(np.sign(actual_changes) == np.sign(forecast_changes))
        
        validation_results = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'direction_accuracy': direction_accuracy,
            'forecasts': forecasts,
            'actuals': test_data,
            'errors': errors
        }
        
        logger.info(f"Validation RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")
        
        return validation_results
    
    def cross_validate_model(self, series: pd.Series, model_spec: Dict,
                           n_splits: int = 5, test_size: int = 4) -> Dict[str, any]:
        """Perform time series cross-validation."""
        logger.debug(f"Starting {n_splits}-fold cross-validation")
        
        cv_results = []
        min_train_size = 20  # Minimum observations for training
        
        # Calculate split points
        total_size = len(series)
        step_size = (total_size - min_train_size - test_size) // n_splits
        
        for i in range(n_splits):
            train_end = min_train_size + i * step_size
            test_end = train_end + test_size
            
            if test_end > total_size:
                break
                
            # Split data
            train_data = series.iloc[:train_end]
            test_data = series.iloc[train_end:test_end]
            
            try:
                # Fit model
                model = ARIMA(train_data, 
                             order=model_spec['order'],
                             seasonal_order=model_spec['seasonal_order'])
                fitted = model.fit()
                
                # Forecast
                forecasts = fitted.forecast(steps=test_size)
                
                # Calculate metrics
                errors = test_data - forecasts
                fold_results = {
                    'fold': i,
                    'mae': np.mean(np.abs(errors)),
                    'rmse': np.sqrt(np.mean(errors**2)),
                    'mape': np.mean(np.abs(errors / test_data)) * 100
                }
                
                cv_results.append(fold_results)
                
            except Exception as e:
                logger.warning(f"CV fold {i} failed: {str(e)}")
                continue
                
        # Aggregate results
        if cv_results:
            cv_summary = {
                'mean_mae': np.mean([r['mae'] for r in cv_results]),
                'mean_rmse': np.mean([r['rmse'] for r in cv_results]),
                'mean_mape': np.mean([r['mape'] for r in cv_results]),
                'std_rmse': np.std([r['rmse'] for r in cv_results]),
                'fold_results': cv_results
            }
            
            logger.info(f"CV Mean RMSE: {cv_summary['mean_rmse']:.4f} "
                       f"(±{cv_summary['std_rmse']:.4f})")
            
            return cv_summary
        else:
            raise ValueError("Cross-validation failed for all folds")
    
    def select_best_model(self, series: pd.Series, 
                         criterion: str = 'aic',
                         validate: bool = True) -> Dict[str, any]:
        """Complete model selection process."""
        logger.info("Starting automatic model selection")
        
        try:
            # Grid search for best model
            search_results = self.grid_search_arima(series)
            
            # Get best model based on criterion
            if criterion == 'aic':
                best_model_info = search_results['best_model']
            elif criterion == 'bic':
                # Resort by BIC
                all_results = search_results['all_results']
                all_results.sort(key=lambda x: x['bic'])
                best_model_info = all_results[0]
            else:
                raise ValueError(f"Unknown criterion: {criterion}")
                
            # Validate if requested
            if validate:
                validation_results = self.validate_model(
                    best_model_info['model'], 
                    series
                )
                best_model_info['validation'] = validation_results
                
                # Cross-validation
                cv_results = self.cross_validate_model(
                    series,
                    best_model_info
                )
                best_model_info['cross_validation'] = cv_results
                
            return best_model_info
        except ValueError as e:
            # Check if it's the constant series error
            if "Invalid input" in str(e) or "constant" in str(e):
                raise ValueError("Invalid input, x is constant")
            else:
                raise