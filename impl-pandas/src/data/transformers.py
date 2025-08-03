"""Feature transformation utilities"""

from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from loguru import logger


class DataTransformer:
    """Transform features for modeling"""
    
    def __init__(self):
        self.scalers = {}
        self.transformations = {}
    
    def create_weather_features(self,
                              df: pd.DataFrame,
                              temp_cols: List[str]) -> pd.DataFrame:
        """
        Create weather-related features
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with weather variables
        temp_cols : List[str]
            Temperature column names
            
        Returns:
        --------
        pd.DataFrame
            Data with weather features
        """
        df = df.copy()
        
        # Calculate quarterly temperature ranges if individual quarters exist
        quarter_temps = [col for col in temp_cols if 'temp_range_q' in col]
        if quarter_temps:
            # Average temperature range
            df["temp_range_avg"] = df[quarter_temps].mean(axis=1)
            
            # Temperature variability across quarters
            df["temp_range_std"] = df[quarter_temps].std(axis=1)
            
            # Extreme temperature indicator
            df["extreme_temp"] = (
                (df[quarter_temps] > df[quarter_temps].mean() + 2 * df[quarter_temps].std()) |
                (df[quarter_temps] < df[quarter_temps].mean() - 2 * df[quarter_temps].std())
            ).any(axis=1).astype(int)
        
        # Create interaction terms
        if "avg_temp" in df.columns and "precipitation" in df.columns:
            df["temp_precip_interaction"] = df["avg_temp"] * df["precipitation"]
        
        return df
    
    def create_demographic_features(self,
                                  df: pd.DataFrame) -> pd.DataFrame:
        """
        Create demographic-related features
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with demographic variables
            
        Returns:
        --------
        pd.DataFrame
            Data with demographic features
        """
        df = df.copy()
        
        # Create demographic indices
        if "pct_bachelors_plus" in df.columns and "avg_household_income" in df.columns:
            # Education-income index
            df["education_income_index"] = (
                df["pct_bachelors_plus"] * np.log(df["avg_household_income"])
            )
        
        # Age structure features
        if "pct_over_65" in df.columns:
            df["working_age_pct"] = 1 - df["pct_over_65"]
        
        # Housing market features
        if "pct_single_family" in df.columns:
            df["multi_family_pct"] = 1 - df["pct_single_family"]
        
        # Industry concentration
        industry_cols = [col for col in df.columns if col.endswith("_share")]
        if industry_cols:
            # Herfindahl index for industry concentration
            df["industry_concentration"] = (df[industry_cols] ** 2).sum(axis=1)
            
            # Dominant industry indicator
            df["dominant_industry"] = df[industry_cols].idxmax(axis=1)
        
        return df
    
    def create_interaction_terms(self,
                               df: pd.DataFrame,
                               weather_vars: List[str],
                               demographic_vars: List[str]) -> pd.DataFrame:
        """
        Create interaction terms between weather and demographics
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with variables
        weather_vars : List[str]
            Weather variable names
        demographic_vars : List[str]
            Demographic variable names
            
        Returns:
        --------
        pd.DataFrame
            Data with interaction terms
        """
        df = df.copy()
        
        # Select key interactions based on domain knowledge
        key_interactions = [
            ("temp_range_avg", "pct_over_65"),  # Elderly sensitivity to temperature
            ("temp_range_q1", "construction_share"),  # Winter construction impact
            ("temp_range_q3", "agriculture_share"),  # Summer agriculture impact
            ("precipitation", "pct_single_family"),  # Rain impact on housing types
        ]
        
        for weather_var, demo_var in key_interactions:
            if weather_var in df.columns and demo_var in df.columns:
                interaction_name = f"{weather_var}_X_{demo_var}"
                df[interaction_name] = df[weather_var] * df[demo_var]
                logger.info(f"Created interaction: {interaction_name}")
        
        return df
    
    def scale_features(self,
                      df: pd.DataFrame,
                      feature_cols: List[str],
                      method: str = "standard",
                      fit: bool = True) -> pd.DataFrame:
        """
        Scale features for modeling
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with features
        feature_cols : List[str]
            Columns to scale
        method : str
            Scaling method: 'standard' or 'minmax'
        fit : bool
            Whether to fit the scaler (True for training)
            
        Returns:
        --------
        pd.DataFrame
            Data with scaled features
        """
        df = df.copy()
        
        # Initialize scaler
        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        # Fit or transform
        valid_cols = [col for col in feature_cols if col in df.columns]
        
        if fit:
            df[valid_cols] = scaler.fit_transform(df[valid_cols])
            self.scalers[method] = scaler
        else:
            if method in self.scalers:
                df[valid_cols] = self.scalers[method].transform(df[valid_cols])
            else:
                raise ValueError(f"Scaler not fitted for method: {method}")
        
        return df
    
    def create_lagged_features(self,
                             df: pd.DataFrame,
                             value_col: str,
                             entity_col: str,
                             time_col: str,
                             lags: List[int]) -> pd.DataFrame:
        """
        Create lagged features for time series
        
        Parameters:
        -----------
        df : pd.DataFrame
            Panel data
        value_col : str
            Column to lag
        entity_col : str
            Entity identifier
        time_col : str
            Time column
        lags : List[int]
            Lag periods to create
            
        Returns:
        --------
        pd.DataFrame
            Data with lagged features
        """
        df = df.copy()
        df = df.sort_values([entity_col, time_col])
        
        for lag in lags:
            lag_col = f"{value_col}_lag{lag}"
            df[lag_col] = df.groupby(entity_col)[value_col].shift(lag)
        
        # Create difference features
        if 1 in lags:
            df[f"{value_col}_diff"] = df[value_col] - df[f"{value_col}_lag1"]
        
        # Create moving averages
        for window in [4, 8]:  # Quarterly data: 1 year, 2 years
            ma_col = f"{value_col}_ma{window}"
            df[ma_col] = (
                df.groupby(entity_col)[value_col]
                .rolling(window=window, min_periods=window//2)
                .mean()
                .reset_index(0, drop=True)
            )
        
        return df
    
    def encode_categorical(self,
                         df: pd.DataFrame,
                         categorical_cols: List[str],
                         method: str = "onehot") -> pd.DataFrame:
        """
        Encode categorical variables
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with categorical variables
        categorical_cols : List[str]
            Columns to encode
        method : str
            Encoding method: 'onehot' or 'label'
            
        Returns:
        --------
        pd.DataFrame
            Data with encoded variables
        """
        df = df.copy()
        
        for col in categorical_cols:
            if col not in df.columns:
                continue
            
            if method == "onehot":
                # One-hot encode
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(columns=[col])
            elif method == "label":
                # Label encode
                df[col] = pd.Categorical(df[col]).codes
        
        return df
    
    def get_feature_names(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Get feature names by category
        
        Parameters:
        -----------
        df : pd.DataFrame
            Transformed data
            
        Returns:
        --------
        Dict[str, List[str]]
            Features grouped by category
        """
        features = {
            "weather": [],
            "demographic": [],
            "industry": [],
            "interaction": [],
            "temporal": [],
            "other": []
        }
        
        for col in df.columns:
            if any(term in col.lower() for term in ["temp", "precip", "weather"]):
                features["weather"].append(col)
            elif any(term in col.lower() for term in ["pct", "income", "education", "age"]):
                features["demographic"].append(col)
            elif col.endswith("_share"):
                features["industry"].append(col)
            elif "_X_" in col:
                features["interaction"].append(col)
            elif any(term in col.lower() for term in ["lag", "ma", "diff", "trend", "quarter"]):
                features["temporal"].append(col)
            else:
                features["other"].append(col)
        
        return features