"""Test data generator for various scenarios including edge cases"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import random


class TestDataGenerator:
    """Generate realistic test data for various scenarios"""
    
    @staticmethod
    def generate_time_series(n_periods: int, 
                           frequency: str = 'QE',
                           start_date: str = '2010-01-01',
                           seasonality: bool = True,
                           trend: bool = True,
                           outliers: bool = False,
                           structural_break: bool = False,
                           missing_values: bool = False,
                           noise_level: float = 1.0) -> pd.Series:
        """
        Generate time series with known properties
        
        Parameters:
        -----------
        n_periods : int
            Number of time periods
        frequency : str
            Frequency of time series ('D', 'ME', 'QE', 'YE')
        start_date : str
            Start date for the series
        seasonality : bool
            Whether to include seasonal component
        trend : bool
            Whether to include trend component
        outliers : bool
            Whether to include outliers
        structural_break : bool
            Whether to include structural break
        missing_values : bool
            Whether to include missing values
        noise_level : float
            Standard deviation of noise component
            
        Returns:
        --------
        pd.Series
            Generated time series
        """
        # Create date index
        dates = pd.date_range(start=start_date, periods=n_periods, freq=frequency)
        
        # Initialize series
        values = np.zeros(n_periods)
        
        # Add trend component
        if trend:
            trend_strength = 0.1
            trend_component = trend_strength * np.arange(n_periods)
            values += trend_component
        
        # Add seasonal component
        if seasonality:
            if frequency in ['QE', 'Q']:
                period = 4
                seasonal_strength = 5.0
            elif frequency in ['ME', 'M']:
                period = 12
                seasonal_strength = 10.0
            else:
                period = 1
                seasonal_strength = 0.0
            
            if period > 1:
                seasonal_component = seasonal_strength * np.sin(2 * np.pi * np.arange(n_periods) / period)
                values += seasonal_component
        
        # Add structural break
        if structural_break and n_periods > 20:
            break_point = n_periods // 2
            values[break_point:] += 10.0  # Level shift
        
        # Add noise
        noise = np.random.normal(0, noise_level, n_periods)
        values += noise
        
        # Add base level
        values += 100
        
        # Add outliers
        if outliers and n_periods > 10:
            n_outliers = max(1, n_periods // 20)
            outlier_indices = np.random.choice(n_periods, n_outliers, replace=False)
            outlier_magnitude = np.random.choice([-1, 1], n_outliers) * np.random.uniform(3, 5, n_outliers) * noise_level
            values[outlier_indices] += outlier_magnitude
        
        # Create series
        series = pd.Series(values, index=dates, name='generated_series')
        
        # Add missing values
        if missing_values and n_periods > 10:
            n_missing = max(1, n_periods // 10)
            missing_indices = np.random.choice(n_periods, n_missing, replace=False)
            series.iloc[missing_indices] = np.nan
        
        return series
    
    @staticmethod
    def generate_panel_data(n_entities: int,
                          n_periods: int,
                          entity_prefix: str = 'entity',
                          start_date: str = '2010-01-01',
                          frequency: str = 'QE',
                          missing_rate: float = 0.0,
                          unbalanced: bool = False,
                          entity_effects: bool = True,
                          time_effects: bool = True,
                          n_features: int = 3) -> pd.DataFrame:
        """
        Generate panel data for regression tests
        
        Parameters:
        -----------
        n_entities : int
            Number of cross-sectional units
        n_periods : int
            Number of time periods
        entity_prefix : str
            Prefix for entity names
        start_date : str
            Start date for the panel
        frequency : str
            Frequency of observations
        missing_rate : float
            Fraction of missing observations
        unbalanced : bool
            Whether to create unbalanced panel
        entity_effects : bool
            Whether to include entity fixed effects
        time_effects : bool
            Whether to include time fixed effects
        n_features : int
            Number of feature variables
            
        Returns:
        --------
        pd.DataFrame
            Panel data with MultiIndex (entity, time)
        """
        # Create entities and time periods
        entities = [f'{entity_prefix}_{i:03d}' for i in range(n_entities)]
        dates = pd.date_range(start=start_date, periods=n_periods, freq=frequency)
        
        # Create multi-index
        if unbalanced:
            # Some entities have shorter time series
            index_tuples = []
            for entity in entities:
                if np.random.random() < 0.2:  # 20% chance of shorter series
                    entity_periods = np.random.randint(n_periods // 2, n_periods)
                    entity_dates = dates[:entity_periods]
                else:
                    entity_dates = dates
                
                for date in entity_dates:
                    index_tuples.append((entity, date))
            
            index = pd.MultiIndex.from_tuples(index_tuples, names=['entity', 'time'])
        else:
            index = pd.MultiIndex.from_product([entities, dates], names=['entity', 'time'])
        
        n_obs = len(index)
        
        # Generate features
        data = {}
        for i in range(n_features):
            data[f'x{i+1}'] = np.random.randn(n_obs)
        
        # Generate dependent variable with structure
        y = np.random.randn(n_obs)
        
        # Add entity effects
        if entity_effects:
            entity_fe = {entity: np.random.normal(0, 2) for entity in entities}
            for i, (entity, _) in enumerate(index):
                y[i] += entity_fe[entity]
        
        # Add time effects
        if time_effects:
            time_fe = {date: np.random.normal(0, 1) for date in dates}
            for i, (_, date) in enumerate(index):
                if date in time_fe:
                    y[i] += time_fe[date]
        
        # Add relationship with features
        for i in range(n_features):
            beta = np.random.normal(0.5, 0.2)
            y += beta * data[f'x{i+1}']
        
        data['y'] = y
        
        # Create DataFrame
        df = pd.DataFrame(data, index=index)
        
        # Add missing values
        if missing_rate > 0:
            n_missing = int(n_obs * missing_rate)
            missing_indices = np.random.choice(n_obs, n_missing, replace=False)
            
            # Randomly choose which variables to make missing
            for idx in missing_indices:
                missing_vars = np.random.choice(df.columns, 
                                              size=np.random.randint(1, len(df.columns)),
                                              replace=False)
                df.iloc[idx][missing_vars] = np.nan
        
        return df
    
    @staticmethod
    def generate_edge_cases() -> Dict[str, pd.DataFrame]:
        """
        Generate edge cases for robustness testing
        
        Returns:
        --------
        Dict[str, pd.DataFrame]
            Dictionary of edge case datasets
        """
        edge_cases = {}
        
        # 1. Very short series (minimum length for seasonal adjustment)
        edge_cases['short_series'] = pd.Series(
            np.random.randn(8) + 100,
            index=pd.date_range('2023-01-01', periods=8, freq='QE'),
            name='short_series'
        )
        
        # 2. Constant series (no variation)
        edge_cases['constant_series'] = pd.Series(
            [100.0] * 20,
            index=pd.date_range('2020-01-01', periods=20, freq='QE'),
            name='constant_series'
        )
        
        # 3. Series with extreme outliers
        values = np.random.randn(40) + 100
        values[[5, 15, 25]] = [500, -50, 300]  # Extreme outliers
        edge_cases['extreme_outliers'] = pd.Series(
            values,
            index=pd.date_range('2015-01-01', periods=40, freq='QE'),
            name='extreme_outliers'
        )
        
        # 4. Series with many missing values
        values = np.random.randn(50) + 100
        values[np.random.choice(50, 25, replace=False)] = np.nan
        edge_cases['many_missing'] = pd.Series(
            values,
            index=pd.date_range('2010-01-01', periods=50, freq='QE'),
            name='many_missing'
        )
        
        # 5. Series with structural break
        values = np.concatenate([
            np.random.randn(30) + 100,
            np.random.randn(30) + 150  # Level shift
        ])
        edge_cases['structural_break'] = pd.Series(
            values,
            index=pd.date_range('2010-01-01', periods=60, freq='QE'),
            name='structural_break'
        )
        
        # 6. High frequency noise
        t = np.arange(100)
        values = 100 + 0.1 * t + 10 * np.sin(2 * np.pi * t / 4) + 5 * np.random.randn(100)
        edge_cases['high_noise'] = pd.Series(
            values,
            index=pd.date_range('2010-01-01', periods=100, freq='QE'),
            name='high_noise'
        )
        
        # 7. Non-stationary series with strong trend
        t = np.arange(80)
        values = 100 * np.exp(0.02 * t) + np.random.randn(80)
        edge_cases['exponential_trend'] = pd.Series(
            values,
            index=pd.date_range('2010-01-01', periods=80, freq='QE'),
            name='exponential_trend'
        )
        
        # 8. Series with changing seasonality
        t = np.arange(100)
        seasonal_strength = 5 + 0.1 * t  # Increasing seasonal amplitude
        values = 100 + 0.5 * t + seasonal_strength * np.sin(2 * np.pi * t / 4) + np.random.randn(100)
        edge_cases['changing_seasonality'] = pd.Series(
            values,
            index=pd.date_range('2010-01-01', periods=100, freq='QE'),
            name='changing_seasonality'
        )
        
        # 9. Panel data with single entity
        panel_single = pd.DataFrame({
            'y': np.random.randn(20) + 100,
            'x1': np.random.randn(20),
            'x2': np.random.randn(20)
        }, index=pd.MultiIndex.from_product(
            [['entity_001'], pd.date_range('2020-01-01', periods=20, freq='QE')],
            names=['entity', 'time']
        ))
        edge_cases['panel_single_entity'] = panel_single
        
        # 10. Panel data with single time period
        panel_single_time = pd.DataFrame({
            'y': np.random.randn(10) + 100,
            'x1': np.random.randn(10),
            'x2': np.random.randn(10)
        }, index=pd.MultiIndex.from_product(
            [[f'entity_{i:03d}' for i in range(10)], [pd.Timestamp('2023-01-01')]],
            names=['entity', 'time']
        ))
        edge_cases['panel_single_period'] = panel_single_time
        
        # 11. Perfect multicollinearity in features
        n = 50
        x1 = np.random.randn(n)
        edge_cases['perfect_multicollinearity'] = pd.DataFrame({
            'y': 2 * x1 + np.random.randn(n) * 0.1,
            'x1': x1,
            'x2': 2 * x1,  # Perfect collinearity
            'x3': 3 * x1   # Another collinear variable
        })
        
        # 12. Zero variance in dependent variable
        edge_cases['zero_variance_y'] = pd.DataFrame({
            'y': [100.0] * 30,
            'x1': np.random.randn(30),
            'x2': np.random.randn(30)
        })
        
        return edge_cases
    
    @staticmethod
    def generate_realistic_hpi_data(geography: str = 'state',
                                  n_geographies: int = 10,
                                  start_date: str = '2010-01-01',
                                  end_date: str = '2023-12-31') -> pd.DataFrame:
        """
        Generate realistic HPI data similar to FHFA format
        
        Parameters:
        -----------
        geography : str
            'state', 'cbsa', or 'national'
        n_geographies : int
            Number of geographic units
        start_date : str
            Start date
        end_date : str
            End date
            
        Returns:
        --------
        pd.DataFrame
            HPI data in FHFA-like format
        """
        dates = pd.date_range(start=start_date, end=end_date, freq='QE')
        
        if geography == 'national':
            # National level data
            base_value = 100
            trend = np.linspace(0, 50, len(dates))  # 50% growth over period
            seasonal = 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 4)
            noise = np.random.normal(0, 2, len(dates))
            
            hpi_values = base_value + trend + seasonal + noise
            
            return pd.DataFrame({
                'hpi_national': hpi_values
            }, index=dates)
        
        elif geography == 'state':
            # State level data
            states = ['CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI'][:n_geographies]
            data = {}
            
            for state in states:
                # Each state has different growth rate and volatility
                base_value = np.random.uniform(90, 110)
                growth_rate = np.random.uniform(0.02, 0.08)  # 2-8% annual growth
                volatility = np.random.uniform(1, 3)
                
                trend = base_value * (1 + growth_rate) ** (np.arange(len(dates)) / 4)
                seasonal = volatility * np.sin(2 * np.pi * np.arange(len(dates)) / 4 + np.random.uniform(0, 2*np.pi))
                noise = np.random.normal(0, volatility, len(dates))
                
                data[f'hpi_state_{state}'] = trend + seasonal + noise
            
            return pd.DataFrame(data, index=dates)
        
        elif geography == 'cbsa':
            # CBSA/MSA level data
            cbsas = [f'{i:05d}' for i in np.random.choice(range(10000, 50000), n_geographies, replace=False)]
            data = []
            
            for cbsa in cbsas:
                for date in dates:
                    base_value = np.random.uniform(80, 120)
                    growth = (date - dates[0]).days / 365 * np.random.uniform(0.01, 0.1)
                    seasonal = 3 * np.sin(2 * np.pi * date.quarter / 4)
                    value = base_value * (1 + growth) + seasonal + np.random.normal(0, 2)
                    
                    data.append({
                        'cbsa': cbsa,
                        'date': date,
                        'hpi_cbsa': value
                    })
            
            return pd.DataFrame(data)
        
        else:
            raise ValueError(f"Unknown geography: {geography}")
    
    @staticmethod
    def generate_weather_data(geography: str,
                            variables: List[str],
                            start_date: str = '2010-01-01',
                            end_date: str = '2023-12-31') -> pd.DataFrame:
        """
        Generate realistic weather data
        
        Parameters:
        -----------
        geography : str
            Geographic level
        variables : List[str]
            Weather variables to generate
        start_date : str
            Start date
        end_date : str
            End date
            
        Returns:
        --------
        pd.DataFrame
            Weather data
        """
        dates = pd.date_range(start=start_date, end=end_date, freq='QE')
        data = {}
        
        for var in variables:
            if var == 'temperature':
                # Temperature with seasonal pattern
                base_temp = 60  # Fahrenheit
                seasonal_amp = 20
                values = base_temp + seasonal_amp * np.sin(2 * np.pi * np.arange(len(dates)) / 4 - np.pi/2)
                values += np.random.normal(0, 5, len(dates))
                data[var] = values
                
            elif var == 'precipitation':
                # Precipitation (inches per quarter)
                base_precip = 10
                seasonal_pattern = 3 * np.sin(2 * np.pi * np.arange(len(dates)) / 4)
                values = base_precip + seasonal_pattern + np.random.exponential(2, len(dates))
                data[var] = np.maximum(0, values)
                
            elif var == 'humidity':
                # Relative humidity (%)
                base_humidity = 65
                seasonal_pattern = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 4 + np.pi)
                values = base_humidity + seasonal_pattern + np.random.normal(0, 5, len(dates))
                data[var] = np.clip(values, 20, 95)
                
            else:
                # Generic weather variable
                data[var] = np.random.normal(50, 10, len(dates))
        
        df = pd.DataFrame(data, index=dates)
        
        # Add geographic dimension if needed
        if geography in ['state', 'cbsa']:
            df['geography'] = geography
        
        return df