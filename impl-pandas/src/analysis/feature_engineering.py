import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from scipy.interpolate import UnivariateSpline

logger = logging.getLogger('seasonal_adjustment.feature_engineering')


class FeatureEngineer:
    """Creates features for causation analysis of seasonal patterns."""
    
    def __init__(self):
        self.feature_names = []
        
    def create_seasonal_dummies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create quarterly dummy variables."""
        logger.debug("Creating seasonal dummy variables")
        
        # Extract quarter from period
        df['quarter'] = df.index.quarter
        
        # Create dummy variables (Q1 is reference)
        for q in range(2, 5):
            dummy_name = f'quarter_{q}'
            df[dummy_name] = (df['quarter'] == q).astype(int)
            self.feature_names.append(dummy_name)
            
        return df
    
    def create_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer weather interaction terms."""
        logger.debug("Creating weather features")
        
        weather_cols = ['temp_range', 'avg_temp', 'precipitation']
        
        # Check if weather data is available
        available_weather = [col for col in weather_cols if col in df.columns]
        
        if not available_weather:
            logger.warning("No weather data available for feature engineering")
            return df
            
        # Create squared terms for non-linear effects
        for col in available_weather:
            squared_name = f'{col}_squared'
            df[squared_name] = df[col] ** 2
            self.feature_names.append(squared_name)
            
        # Create interaction terms between weather and quarters
        for q in range(2, 5):
            quarter_dummy = f'quarter_{q}'
            if quarter_dummy in df.columns:
                for weather_col in available_weather:
                    interaction_name = f'{quarter_dummy}_x_{weather_col}'
                    df[interaction_name] = df[quarter_dummy] * df[weather_col]
                    self.feature_names.append(interaction_name)
                    
        # Create lagged weather features
        for col in available_weather:
            lag_name = f'{col}_lag1'
            df[lag_name] = df[col].shift(1)
            self.feature_names.append(lag_name)
            
        return df
    
    def create_time_trends(self, df: pd.DataFrame, n_knots: int = 4) -> pd.DataFrame:
        """Create spline functions for time trends."""
        logger.debug(f"Creating time trends with {n_knots} knots")
        
        # Create time index
        df['time_index'] = range(len(df))
        
        # Linear trend
        df['linear_trend'] = df['time_index'] / len(df)
        self.feature_names.append('linear_trend')
        
        # Quadratic trend
        df['quadratic_trend'] = (df['time_index'] / len(df)) ** 2
        self.feature_names.append('quadratic_trend')
        
        # Create spline basis functions
        x = df['time_index'].values
        knots = np.linspace(x.min(), x.max(), n_knots + 2)[1:-1]
        
        for i, knot in enumerate(knots):
            spline_name = f'spline_knot_{i+1}'
            df[spline_name] = np.maximum(0, x - knot) ** 3
            self.feature_names.append(spline_name)
            
        return df
    
    def calculate_seasonality_measure(self, df: pd.DataFrame) -> pd.Series:
        """Calculate |NSA_HPI - SA_HPI| as dependent variable."""
        logger.debug("Calculating seasonality measure")
        
        if 'nsa_index' not in df.columns or 'sa_index' not in df.columns:
            raise ValueError("Both NSA and SA indices required for seasonality measure")
            
        # Calculate absolute difference
        seasonality = np.abs(df['nsa_index'] - df['sa_index'])
        
        # Also calculate percentage difference - store in the dataframe itself
        df.loc[:, 'seasonality_pct'] = np.abs((df['nsa_index'] - df['sa_index']) / df['nsa_index']) * 100
        
        return seasonality
    
    def create_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features related to market conditions."""
        logger.debug("Creating market features")
        
        if 'num_transactions' in df.columns:
            # Log transform for better distribution
            df['log_transactions'] = np.log1p(df['num_transactions'])
            self.feature_names.append('log_transactions')
            
            # Rolling averages
            df['transactions_ma4'] = df['num_transactions'].rolling(4).mean()
            self.feature_names.append('transactions_ma4')
            
            # Year-over-year change
            df['transactions_yoy'] = df['num_transactions'].pct_change(4)
            self.feature_names.append('transactions_yoy')
            
        return df
    
    def create_demographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process demographic features."""
        logger.debug("Creating demographic features")
        
        demo_cols = ['pct_over_65', 'pct_white', 'pct_bachelors', 
                    'pct_single_family', 'avg_income']
        
        available_demo = [col for col in demo_cols if col in df.columns]
        
        if not available_demo:
            logger.warning("No demographic data available")
            return df
            
        # Standardize demographic features
        for col in available_demo:
            if col == 'avg_income':
                # Log transform income
                df['log_avg_income'] = np.log(df[col])
                self.feature_names.append('log_avg_income')
            else:
                # Already in percentage form
                self.feature_names.append(col)
                
        # Create interaction between demographics
        if 'pct_over_65' in df.columns and 'pct_single_family' in df.columns:
            df['elderly_single_family'] = df['pct_over_65'] * df['pct_single_family'] / 100
            self.feature_names.append('elderly_single_family')
            
        return df
    
    def create_industry_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process industry composition features."""
        logger.debug("Creating industry features")
        
        industry_cols = [col for col in df.columns if col.endswith('_share')]
        
        if not industry_cols:
            logger.warning("No industry data available")
            return df
            
        # Add industry shares to features
        self.feature_names.extend(industry_cols)
        
        # Create dominant industry indicator
        if len(industry_cols) > 0:
            df['dominant_industry'] = df[industry_cols].idxmax(axis=1)
            df['industry_concentration'] = df[industry_cols].max(axis=1)
            self.feature_names.append('industry_concentration')
            
        return df
    
    def prepare_regression_data(self, master_df: pd.DataFrame, 
                              target_type: str = 'absolute') -> pd.DataFrame:
        """Prepare complete dataset for regression analysis."""
        logger.info("Preparing regression data")
        
        # Copy dataframe to avoid modifying original
        df = master_df.copy()
        
        # Reset feature names
        self.feature_names = []
        
        # Ensure we have time-based index
        if 'period' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            df.set_index('period', inplace=True)
            
        # Create target variable
        if target_type == 'absolute':
            df['target'] = self.calculate_seasonality_measure(df)
        elif target_type == 'percentage':
            self.calculate_seasonality_measure(df)
            df['target'] = df['seasonality_pct']
        else:
            raise ValueError(f"Unknown target type: {target_type}")
            
        # Create all features
        df = self.create_seasonal_dummies(df)
        df = self.create_weather_features(df)
        df = self.create_time_trends(df)
        df = self.create_market_features(df)
        df = self.create_demographic_features(df)
        df = self.create_industry_features(df)
        
        # Add geographic fixed effects if multiple geographies
        if 'geography_id' in df.columns:
            geo_dummies = pd.get_dummies(df['geography_id'], prefix='geo')
            # Drop one for reference
            geo_dummies = geo_dummies.iloc[:, 1:]
            df = pd.concat([df, geo_dummies], axis=1)
            self.feature_names.extend(geo_dummies.columns.tolist())
            
        # Drop rows with missing target
        df = df.dropna(subset=['target'])
        
        # Fill missing values in features
        for col in self.feature_names:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mean())
                
        logger.info(f"Prepared {len(df)} observations with {len(self.feature_names)} features")
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return self.feature_names.copy()