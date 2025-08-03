"""Data preprocessing utilities"""

from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from loguru import logger
from scipy import stats


class DataPreprocessor:
    """Preprocess data for seasonal adjustment analysis"""
    
    def __init__(self):
        self.preprocessing_log = []
    
    def clean_time_series(self,
                         series: pd.Series,
                         method: str = "interpolate",
                         max_gap: int = 3) -> pd.Series:
        """
        Clean time series data
        
        Parameters:
        -----------
        series : pd.Series
            Time series to clean
        method : str
            Method for handling missing values: 'interpolate', 'forward_fill', 'drop'
        max_gap : int
            Maximum gap size to interpolate
            
        Returns:
        --------
        pd.Series
            Cleaned series
        """
        logger.info(f"Cleaning time series with {series.isna().sum()} missing values")
        
        cleaned = series.copy()
        
        # Handle missing values
        if method == "interpolate":
            cleaned = cleaned.interpolate(method="time", limit=max_gap)
        elif method == "forward_fill":
            cleaned = cleaned.fillna(method="ffill", limit=max_gap)
        elif method == "drop":
            cleaned = cleaned.dropna()
        
        # Log preprocessing
        self.preprocessing_log.append({
            "series_name": series.name,
            "original_missing": series.isna().sum(),
            "cleaned_missing": cleaned.isna().sum(),
            "method": method
        })
        
        return cleaned
    
    def handle_outliers(self,
                       series: pd.Series,
                       method: str = "winsorize",
                       threshold: float = 3.0) -> pd.Series:
        """
        Handle outliers in time series
        
        Parameters:
        -----------
        series : pd.Series
            Time series data
        method : str
            Method for handling outliers: 'winsorize', 'remove', 'cap'
        threshold : float
            Threshold for outlier detection (standard deviations)
            
        Returns:
        --------
        pd.Series
            Series with outliers handled
        """
        cleaned = series.copy()
        
        # Detect outliers
        mean = cleaned.mean()
        std = cleaned.std()
        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std
        
        outlier_mask = (cleaned < lower_bound) | (cleaned > upper_bound)
        n_outliers = outlier_mask.sum()
        
        if n_outliers > 0:
            logger.info(f"Found {n_outliers} outliers using {method} method")
            
            if method == "winsorize":
                # Cap outliers at threshold
                cleaned[cleaned < lower_bound] = lower_bound
                cleaned[cleaned > upper_bound] = upper_bound
            elif method == "remove":
                cleaned[outlier_mask] = np.nan
            elif method == "cap":
                # Cap at percentiles
                p_low = cleaned.quantile(0.01)
                p_high = cleaned.quantile(0.99)
                cleaned[cleaned < p_low] = p_low
                cleaned[cleaned > p_high] = p_high
        
        return cleaned
    
    def prepare_panel_data(self,
                          df: pd.DataFrame,
                          entity_col: str,
                          time_col: str,
                          value_col: str,
                          balance: bool = True) -> pd.DataFrame:
        """
        Prepare panel data for analysis
        
        Parameters:
        -----------
        df : pd.DataFrame
            Raw panel data
        entity_col : str
            Entity identifier column
        time_col : str
            Time column
        value_col : str
            Value column to analyze
        balance : bool
            Whether to balance the panel
            
        Returns:
        --------
        pd.DataFrame
            Prepared panel data
        """
        logger.info(f"Preparing panel data with {len(df)} observations")
        
        # Ensure proper types
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
        
        # Sort by entity and time
        df = df.sort_values([entity_col, time_col])
        
        # Balance panel if requested
        if balance:
            # Create complete index
            entities = df[entity_col].unique()
            time_periods = df[time_col].unique()
            
            full_index = pd.MultiIndex.from_product(
                [entities, time_periods],
                names=[entity_col, time_col]
            )
            
            # Reindex to balance
            df_indexed = df.set_index([entity_col, time_col])
            df_balanced = df_indexed.reindex(full_index)
            
            # Reset index
            df = df_balanced.reset_index()
            
            logger.info(f"Balanced panel: {len(df)} observations")
        
        return df
    
    def create_seasonal_dummies(self,
                              df: pd.DataFrame,
                              date_col: str,
                              frequency: str = "quarterly") -> pd.DataFrame:
        """
        Create seasonal dummy variables
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with date column
        date_col : str
            Name of date column
        frequency : str
            Frequency: 'quarterly' or 'monthly'
            
        Returns:
        --------
        pd.DataFrame
            Data with seasonal dummies added
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        if frequency == "quarterly":
            df["quarter"] = df[date_col].dt.quarter
            # Create dummies (Q4 as reference)
            for q in [1, 2, 3]:
                df[f"Q{q}"] = (df["quarter"] == q).astype(int)
        elif frequency == "monthly":
            df["month"] = df[date_col].dt.month
            # Create dummies (December as reference)
            for m in range(1, 12):
                df[f"M{m}"] = (df["month"] == m).astype(int)
        
        return df
    
    def create_time_trends(self,
                          df: pd.DataFrame,
                          date_col: str,
                          breakpoints: List[int]) -> pd.DataFrame:
        """
        Create time trend variables with breakpoints
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with date column
        date_col : str
            Name of date column
        breakpoints : List[int]
            Years for breakpoints (e.g., [1998, 2007, 2011, 2020])
            
        Returns:
        --------
        pd.DataFrame
            Data with time trend variables
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df["year"] = df[date_col].dt.year
        
        # Create time trend
        min_year = df["year"].min()
        df["time_trend"] = df["year"] - min_year
        
        # Create spline variables for each period
        breakpoints = [min_year] + breakpoints + [df["year"].max() + 1]
        
        for i in range(len(breakpoints) - 1):
            start_year = breakpoints[i]
            end_year = breakpoints[i + 1]
            
            period_name = f"trend_{start_year}_{end_year-1}"
            
            # Create spline: 0 before period, linear during, constant after
            df[period_name] = 0
            mask = (df["year"] >= start_year) & (df["year"] < end_year)
            df.loc[mask, period_name] = df.loc[mask, "year"] - start_year
            df.loc[df["year"] >= end_year, period_name] = end_year - start_year - 1
        
        return df
    
    def calculate_seasonality_measure(self,
                                    nsa_series: pd.Series,
                                    sa_series: pd.Series) -> pd.Series:
        """
        Calculate seasonality measure (absolute difference)
        
        Parameters:
        -----------
        nsa_series : pd.Series
            Non-seasonally adjusted series
        sa_series : pd.Series
            Seasonally adjusted series
            
        Returns:
        --------
        pd.Series
            Absolute difference between NSA and SA
        """
        if len(nsa_series) != len(sa_series):
            raise ValueError("NSA and SA series must have same length")
        
        return np.abs(nsa_series - sa_series)
    
    def merge_datasets(self,
                      hpi_data: pd.DataFrame,
                      weather_data: pd.DataFrame,
                      demographic_data: pd.DataFrame,
                      on_cols: List[str]) -> pd.DataFrame:
        """
        Merge multiple datasets for analysis
        
        Parameters:
        -----------
        hpi_data : pd.DataFrame
            HPI data
        weather_data : pd.DataFrame
            Weather data
        demographic_data : pd.DataFrame
            Demographic data
        on_cols : List[str]
            Columns to merge on
            
        Returns:
        --------
        pd.DataFrame
            Merged dataset
        """
        logger.info("Merging datasets")
        
        # Start with HPI data
        merged = hpi_data.copy()
        
        # Merge weather data
        if weather_data is not None:
            weather_merge_cols = [col for col in on_cols if col in weather_data.columns]
            merged = merged.merge(
                weather_data,
                on=weather_merge_cols,
                how="left",
                suffixes=("", "_weather")
            )
        
        # Merge demographic data
        if demographic_data is not None:
            # Demographics typically don't vary by time
            demo_merge_cols = [col for col in on_cols if col != "date" and col in demographic_data.columns]
            merged = merged.merge(
                demographic_data,
                on=demo_merge_cols,
                how="left",
                suffixes=("", "_demo")
            )
        
        logger.info(f"Merged dataset has {len(merged)} rows and {len(merged.columns)} columns")
        
        return merged