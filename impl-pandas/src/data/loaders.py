"""Data loading utilities"""

from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from ..config import get_settings


class DataLoader:
    """Base class for data loaders"""
    
    def __init__(self):
        self.settings = get_settings()
        self.data_path = self.settings.data_dir
        
        # Create data directories if they don't exist
        self.data_path.mkdir(parents=True, exist_ok=True)
    
    def load(self, **kwargs) -> pd.DataFrame:
        """Load data - to be implemented by subclasses"""
        raise NotImplementedError


class HPIDataLoader(DataLoader):
    """Load FHFA House Price Index data"""
    
    def __init__(self):
        super().__init__()
        # Define available index types
        self.index_types = {
            "purchase_only": "PO",
            "all_transactions": "AT",
            "expanded_data": "ED",
            "distress_free": "DF",
            "manufactured_homes": "MH"
        }
    
    def load(self, 
             start_date: str,
             end_date: str,
             geography: str = "state",
             index_type: str = "purchase_only",
             frequency: str = "quarterly") -> pd.DataFrame:
        """
        Load HPI data for specified parameters
        
        Parameters:
        -----------
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str
            End date in YYYY-MM-DD format
        geography : str
            Geographic level: 'state', 'msa', 'national', 'zip3'
        index_type : str
            Type of index: 'purchase_only', 'all_transactions', etc.
        frequency : str
            Data frequency: 'monthly', 'quarterly', 'annual'
            
        Returns:
        --------
        pd.DataFrame
            HPI data with DatetimeIndex and columns for each geography
        """
        logger.info(f"Loading HPI data: {geography} {index_type} from {start_date} to {end_date}")
        
        # Construct file path based on parameters
        file_pattern = f"hpi_{geography}_{self.index_types.get(index_type, 'PO')}.csv"
        file_path = self.data_path / "raw" / file_pattern
        
        if not file_path.exists():
            # Generate synthetic data for testing
            logger.warning(f"File not found: {file_path}. Generating synthetic data.")
            return self._generate_synthetic_hpi_data(
                start_date, end_date, geography, frequency
            )
        
        # Load actual data
        df = pd.read_csv(file_path, parse_dates=["date"])
        
        # Filter date range
        df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
        
        # Resample if needed
        if frequency == "quarterly" and "date" in df.columns:
            df = df.set_index("date").resample("QE").mean().reset_index()
        
        return df
    
    def _generate_synthetic_hpi_data(self,
                                    start_date: str,
                                    end_date: str,
                                    geography: str,
                                    frequency: str) -> pd.DataFrame:
        """Generate synthetic HPI data for testing"""
        # Date range
        freq_map = {"monthly": "ME", "quarterly": "QE", "annual": "YE"}
        dates = pd.date_range(start_date, end_date, freq=freq_map[frequency])
        
        # Geography IDs
        if geography == "state":
            geo_ids = ["CA", "TX", "FL", "NY", "PA", "IL", "OH", "GA", "NC", "MI"]
        elif geography == "msa":
            geo_ids = ["12345", "67890", "11111", "22222", "33333"]
        elif geography == "cbsa":
            geo_ids = ["12345", "67890", "11111", "22222", "33333"]
        else:
            geo_ids = ["USA"]
        
        # Generate data in wide format (columns for each geography)
        data = pd.DataFrame(index=dates)
        
        for geo_id in geo_ids:
            np.random.seed(hash(geo_id) % 2**32)
            
            # Base HPI with trend and seasonality
            n_periods = len(dates)
            trend = np.linspace(100, 150, n_periods)
            seasonal = 10 * np.sin(2 * np.pi * np.arange(n_periods) / 4)
            noise = np.random.normal(0, 2, n_periods)
            hpi = trend + seasonal + noise
            
            # Add as column
            col_name = f"hpi_{geography}_{geo_id}" if geography != "national" else "hpi_national"
            if geography == "cbsa" or geography == "msa":
                col_name = f"hpi_cbsa_{geo_id}"
            data[col_name] = hpi
        
        return data


class WeatherDataLoader(DataLoader):
    """Load NOAA weather data"""
    
    def load(self,
             geography: Union[str, List[str]],
             variables: List[str],
             start_date: str,
             end_date: str) -> pd.DataFrame:
        """
        Load weather data for specified geography and variables
        
        Parameters:
        -----------
        geography : str or List[str]
            Geographic identifier(s) (state code or CBSA code)
        variables : List[str]
            Weather variables to load
        start_date : str
            Start date
        end_date : str
            End date
            
        Returns:
        --------
        pd.DataFrame
            Weather data with requested variables
        """
        # Handle both string and list inputs
        if isinstance(geography, str):
            geography_list = [geography]
        else:
            geography_list = geography
            
        logger.info(f"Loading weather data for {geography_list}: {variables}")
        
        # Generate synthetic weather data for testing
        dates = pd.date_range(start_date, end_date, freq="QE")
        
        data = []
        for geo in geography_list:
            geo_data = pd.DataFrame({
                "date": dates,
                "cbsa_code": geo,
                "geography_id": geo
            })
            
            # Generate temperature with seasonal pattern
            np.random.seed(hash(geo) % 2**32)
            
            # Base temperature by quarter (average US temperatures)
            quarter_temps = {1: 40, 2: 65, 3: 75, 4: 45}
            temps = []
            for date in dates:
                base_temp = quarter_temps[date.quarter]
                temp = base_temp + np.random.normal(0, 10)
                temps.append(temp)
            
            if "temperature" in variables:
                geo_data["temperature"] = temps
            
            if "precipitation" in variables:
                # Generate precipitation (inches per quarter)
                geo_data["precipitation"] = np.random.gamma(2, 2, len(dates))
            
            if "humidity" in variables:
                geo_data["humidity"] = np.random.uniform(30, 80, len(dates))
            
            if "wind_speed" in variables:
                geo_data["wind_speed"] = np.random.gamma(3, 2, len(dates))
            
            # Add quarter-specific temperature ranges if requested
            for q in range(1, 5):
                var_name = f"temp_range_q{q}"
                if var_name in variables:
                    # Temperature range for specific quarter
                    quarter_mask = pd.Series([d.quarter == q for d in dates])
                    geo_data[var_name] = 0
                    geo_data.loc[quarter_mask, var_name] = np.random.uniform(10, 30, quarter_mask.sum())
            
            data.append(geo_data)
        
        return pd.concat(data, ignore_index=True)


class DemographicDataLoader(DataLoader):
    """Load Census demographic data"""
    
    def load(self,
             geography_ids: List[str],
             years: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Load demographic data for specified geographies
        
        Parameters:
        -----------
        geography_ids : List[str]
            List of geography identifiers (CBSA codes, state codes, etc.)
        years : List[int], optional
            Years to load data for (default: 2010-2023)
            
        Returns:
        --------
        pd.DataFrame
            Demographic data with population, income, and industry shares
        """
        logger.info(f"Loading demographic data for {len(geography_ids)} geographies")
        
        if years is None:
            years = list(range(2010, 2024))
        
        # Generate synthetic demographic data
        data = []
        
        for geo_id in geography_ids:
            np.random.seed(hash(geo_id) % 2**32)
            
            # Base values
            base_pop = np.random.uniform(50000, 5000000)
            base_income = np.random.uniform(40000, 100000)
            
            for year in years:
                # Population with growth
                pop_growth = 1 + 0.01 * (year - 2010)
                population = int(base_pop * pop_growth)
                
                # Median income with growth
                income_growth = 1 + 0.02 * (year - 2010)
                median_income = base_income * income_growth
                
                # Unemployment rate (cyclical)
                unemployment_rate = 5 + 2 * np.sin(2 * np.pi * (year - 2010) / 7) + np.random.normal(0, 0.5)
                unemployment_rate = max(2, min(15, unemployment_rate))
                
                # Industry shares (should sum to ~1 but we'll use main industry share)
                industry_share = np.random.uniform(0.1, 0.3)
                
                # Other characteristics
                pct_white = np.random.uniform(0.3, 0.9)
                pct_over_65 = np.random.uniform(0.1, 0.25)
                pct_bachelors_plus = np.random.uniform(0.15, 0.6)
                pct_single_family = np.random.uniform(0.4, 0.8)
                
                # Industry breakdown
                healthcare_share = np.random.uniform(0.1, 0.2)
                manufacturing_share = np.random.uniform(0.05, 0.25)
                professional_share = np.random.uniform(0.05, 0.15)
                retail_share = np.random.uniform(0.1, 0.15)
                
                row = {
                    "cbsa_code": geo_id,
                    "geography_id": geo_id,
                    "year": year,
                    "population": population,
                    "median_income": median_income,
                    "unemployment_rate": unemployment_rate,
                    "industry_share": industry_share,
                    "pct_white": pct_white,
                    "pct_over_65": pct_over_65,
                    "pct_bachelors_plus": pct_bachelors_plus,
                    "pct_single_family": pct_single_family,
                    "healthcare_share": healthcare_share,
                    "manufacturing_share": manufacturing_share,
                    "professional_share": professional_share,
                    "retail_share": retail_share
                }
                
                data.append(row)
        
        return pd.DataFrame(data)