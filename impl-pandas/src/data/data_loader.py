import pandas as pd
import numpy as np
from typing import Dict, Optional, Union
import logging
from pathlib import Path

logger = logging.getLogger('seasonal_adjustment.data_loader')


class DataLoader:
    """Handles loading and validation of all data types for seasonal adjustment model."""
    
    def __init__(self):
        self.hpi_data = None
        self.weather_data = None
        self.demographics_data = None
        self.industry_data = None
        
    def load_hpi_data(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """Load housing price index data from file.
        
        Expected columns:
        - geography_id: State/MSA identifier
        - geography_type: 'STATE' or 'MSA'
        - period: Date in YYYY-Q# format
        - nsa_index: Non-seasonally adjusted index value
        - num_transactions: Number of transactions
        """
        logger.info(f"Loading HPI data from {filepath}")
        
        df = pd.read_csv(filepath)
        
        # Convert period to datetime
        df['period'] = pd.to_datetime(df['period'])
        
        # Validate data
        if not self.validate_data(df, 'hpi'):
            raise ValueError("HPI data validation failed")
            
        self.hpi_data = df
        return df
    
    def load_weather_data(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """Load weather data from file.
        
        Expected columns:
        - geography_id: State/MSA identifier
        - period: Date
        - temp_range: Temperature range (max - min)
        - avg_temp: Average temperature
        - precipitation: Total precipitation
        """
        logger.info(f"Loading weather data from {filepath}")
        
        df = pd.read_csv(filepath)
        df['period'] = pd.to_datetime(df['period'])
        
        if not self.validate_data(df, 'weather'):
            raise ValueError("Weather data validation failed")
            
        self.weather_data = df
        return df
    
    def load_demographics_data(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """Load demographics data from file.
        
        Expected columns:
        - geography_id: State/MSA identifier
        - pct_over_65: Percentage over 65 years old
        - pct_white: Percentage of white population
        - pct_bachelors: Percentage with bachelor's degree or higher
        - pct_single_family: Percentage of single-family home sales
        - avg_income: Average household income
        """
        logger.info(f"Loading demographics data from {filepath}")
        
        df = pd.read_csv(filepath)
        
        if not self.validate_data(df, 'demographics'):
            raise ValueError("Demographics data validation failed")
            
        self.demographics_data = df
        return df
    
    def load_industry_data(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """Load industry composition data from file.
        
        Expected columns:
        - geography_id: State/MSA identifier
        - healthcare_share: Healthcare and Social Assistance employment share
        - manufacturing_share: Manufacturing employment share
        - professional_share: Professional Services employment share
        - retail_share: Retail employment share
        - education_share: Education employment share
        - fire_share: Finance, Insurance, Real Estate employment share
        - public_admin_share: Public Administration employment share
        - construction_share: Construction employment share
        - transportation_share: Transportation employment share
        - agriculture_share: Agriculture employment share
        """
        logger.info(f"Loading industry data from {filepath}")
        
        df = pd.read_csv(filepath)
        
        if not self.validate_data(df, 'industry'):
            raise ValueError("Industry data validation failed")
            
        self.industry_data = df
        return df
    
    def validate_data(self, df: pd.DataFrame, data_type: str) -> bool:
        """Validate data based on type-specific rules."""
        
        if df.empty:
            logger.error(f"{data_type} data is empty")
            return False
        
        # Check for missing values
        if df.isnull().any().any():
            logger.warning(f"{data_type} data contains missing values")
            
        if data_type == 'hpi':
            # Check required columns
            required_cols = ['geography_id', 'geography_type', 'period', 'nsa_index', 'num_transactions']
            if not all(col in df.columns for col in required_cols):
                logger.error(f"Missing required columns in HPI data")
                return False
                
            # Check for minimum history (5 years = 20 quarters)
            geo_counts = df.groupby('geography_id')['period'].count()
            if (geo_counts < 20).any():
                logger.warning("Some geographies have less than 5 years of history")
                
            # Check for valid geography types
            valid_types = ['STATE', 'MSA']
            if not df['geography_type'].isin(valid_types).all():
                logger.error("Invalid geography types found")
                return False
                
        elif data_type == 'weather':
            required_cols = ['geography_id', 'period', 'temp_range', 'avg_temp', 'precipitation']
            if not all(col in df.columns for col in required_cols):
                logger.error(f"Missing required columns in weather data")
                return False
                
            # Check for reasonable temperature ranges
            if (df['temp_range'] < 0).any() or (df['temp_range'] > 100).any():
                logger.warning("Unreasonable temperature ranges detected")
                
        elif data_type == 'demographics':
            required_cols = ['geography_id', 'pct_over_65', 'pct_white', 'pct_bachelors', 
                           'pct_single_family', 'avg_income']
            if not all(col in df.columns for col in required_cols):
                logger.error(f"Missing required columns in demographics data")
                return False
                
            # Check percentage columns are between 0 and 100
            pct_cols = [col for col in df.columns if col.startswith('pct_')]
            for col in pct_cols:
                if (df[col] < 0).any() or (df[col] > 100).any():
                    logger.error(f"Invalid percentage values in {col}")
                    return False
                    
        elif data_type == 'industry':
            # Check that shares sum to reasonable total (allow some flexibility)
            share_cols = [col for col in df.columns if col.endswith('_share')]
            if share_cols:
                total_shares = df[share_cols].sum(axis=1)
                if (total_shares < 50).any() or (total_shares > 150).any():
                    logger.warning("Industry shares may not sum correctly")
                    
        return True
    
    def merge_datasets(self) -> pd.DataFrame:
        """Merge all loaded datasets into a master DataFrame."""
        
        if self.hpi_data is None:
            raise ValueError("HPI data not loaded")
            
        # Start with HPI data
        master_df = self.hpi_data.copy()
        
        # Merge weather data if available
        if self.weather_data is not None:
            master_df = pd.merge(
                master_df,
                self.weather_data,
                on=['geography_id', 'period'],
                how='left'
            )
            
        # Merge demographics data if available
        if self.demographics_data is not None:
            master_df = pd.merge(
                master_df,
                self.demographics_data,
                on='geography_id',
                how='left'
            )
            
        # Merge industry data if available
        if self.industry_data is not None:
            master_df = pd.merge(
                master_df,
                self.industry_data,
                on='geography_id',
                how='left'
            )
            
        logger.info(f"Created master dataset with {len(master_df)} records")
        return master_df
    
    def create_sample_data(self, output_dir: Union[str, Path]) -> None:
        """Create sample data files for testing."""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sample HPI data
        np.random.seed(42)
        dates = pd.date_range('2015-01-01', '2023-12-31', freq='Q')
        geographies = ['CA', 'NY', 'TX', 'FL', 'IL']
        
        hpi_records = []
        for geo in geographies:
            base_index = 100
            for i, date in enumerate(dates):
                # Add seasonal pattern
                quarter = date.quarter
                seasonal_factor = 1 + 0.05 * np.sin(2 * np.pi * quarter / 4)
                
                # Add trend
                trend = 1 + 0.02 * i / len(dates)
                
                # Add noise
                noise = np.random.normal(0, 0.02)
                
                index_value = base_index * seasonal_factor * trend * (1 + noise)
                
                hpi_records.append({
                    'geography_id': geo,
                    'geography_type': 'STATE',
                    'period': f"{date.year}-Q{date.quarter}",
                    'nsa_index': round(index_value, 2),
                    'num_transactions': np.random.randint(1000, 5000)
                })
                
        hpi_df = pd.DataFrame(hpi_records)
        hpi_df.to_csv(output_dir / 'sample_hpi_data.csv', index=False)
        
        # Create sample weather data
        weather_records = []
        for geo in geographies:
            for date in dates:
                quarter = date.quarter
                base_temp = {'CA': 60, 'NY': 50, 'TX': 70, 'FL': 75, 'IL': 45}[geo]
                
                weather_records.append({
                    'geography_id': geo,
                    'period': date.strftime('%Y-%m-%d'),
                    'temp_range': round(20 + 10 * np.sin(2 * np.pi * quarter / 4) + np.random.normal(0, 2), 1),
                    'avg_temp': round(base_temp + 15 * np.sin(2 * np.pi * quarter / 4) + np.random.normal(0, 3), 1),
                    'precipitation': round(max(0, 3 + 2 * np.cos(2 * np.pi * quarter / 4) + np.random.normal(0, 1)), 1)
                })
                
        weather_df = pd.DataFrame(weather_records)
        weather_df.to_csv(output_dir / 'sample_weather_data.csv', index=False)
        
        # Create sample demographics data
        demo_records = []
        for geo in geographies:
            demo_records.append({
                'geography_id': geo,
                'pct_over_65': round(12 + np.random.uniform(-2, 5), 1),
                'pct_white': round(60 + np.random.uniform(-20, 20), 1),
                'pct_bachelors': round(30 + np.random.uniform(-10, 15), 1),
                'pct_single_family': round(65 + np.random.uniform(-10, 10), 1),
                'avg_income': round(50000 + np.random.uniform(-10000, 30000), 0)
            })
            
        demo_df = pd.DataFrame(demo_records)
        demo_df.to_csv(output_dir / 'sample_demographics_data.csv', index=False)
        
        # Create sample industry data
        industry_records = []
        for geo in geographies:
            shares = np.random.dirichlet(np.ones(10)) * 100
            
            industry_records.append({
                'geography_id': geo,
                'healthcare_share': round(shares[0], 1),
                'manufacturing_share': round(shares[1], 1),
                'professional_share': round(shares[2], 1),
                'retail_share': round(shares[3], 1),
                'education_share': round(shares[4], 1),
                'fire_share': round(shares[5], 1),
                'public_admin_share': round(shares[6], 1),
                'construction_share': round(shares[7], 1),
                'transportation_share': round(shares[8], 1),
                'agriculture_share': round(shares[9], 1)
            })
            
        industry_df = pd.DataFrame(industry_records)
        industry_df.to_csv(output_dir / 'sample_industry_data.csv', index=False)
        
        logger.info(f"Sample data files created in {output_dir}")