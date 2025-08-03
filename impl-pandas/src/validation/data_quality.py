"""Data quality validation for input data integrity checks"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from loguru import logger


@dataclass
class DataQualityIssue:
    """Container for data quality issues"""
    issue_type: str
    severity: str  # 'error', 'warning', 'info'
    description: str
    affected_columns: Optional[List[str]] = None
    affected_rows: Optional[List[int]] = None
    details: Optional[Dict[str, Any]] = None


class DataQualityValidator:
    """Comprehensive data quality validation for time series and panel data"""
    
    def __init__(self, error_threshold: float = 0.05):
        """
        Initialize data quality validator
        
        Parameters
        ----------
        error_threshold : float
            Maximum allowed proportion of data quality issues
        """
        self.error_threshold = error_threshold
        self.issues = []
        
    def validate_time_series(self, data: pd.Series, 
                           min_length: int = 8,
                           max_missing_ratio: float = 0.2) -> Dict[str, Any]:
        """
        Validate time series data quality
        
        Parameters
        ----------
        data : pd.Series
            Time series to validate
        min_length : int
            Minimum required length
        max_missing_ratio : float
            Maximum allowed missing data ratio
            
        Returns
        -------
        Dict[str, Any]
            Validation report
        """
        logger.info(f"Validating time series with {len(data)} observations")
        self.issues = []
        
        # Check length
        if len(data) < min_length:
            self.issues.append(DataQualityIssue(
                issue_type="insufficient_data",
                severity="error",
                description=f"Series has {len(data)} observations, minimum {min_length} required"
            ))
        
        # Check missing values
        missing_count = data.isna().sum()
        missing_ratio = missing_count / len(data)
        
        if missing_ratio > max_missing_ratio:
            self.issues.append(DataQualityIssue(
                issue_type="excessive_missing",
                severity="error",
                description=f"Missing data ratio {missing_ratio:.2%} exceeds threshold {max_missing_ratio:.2%}",
                details={'missing_count': int(missing_count), 'total_count': len(data)}
            ))
        elif missing_count > 0:
            self.issues.append(DataQualityIssue(
                issue_type="missing_values",
                severity="warning",
                description=f"Series contains {missing_count} missing values",
                affected_rows=data[data.isna()].index.tolist()
            ))
        
        # Check for gaps in time index
        if isinstance(data.index, pd.DatetimeIndex):
            expected_freq = pd.infer_freq(data.index)
            if expected_freq:
                full_range = pd.date_range(data.index[0], data.index[-1], freq=expected_freq)
                missing_dates = full_range.difference(data.index)
                
                if len(missing_dates) > 0:
                    self.issues.append(DataQualityIssue(
                        issue_type="time_gaps",
                        severity="warning" if len(missing_dates) < 5 else "error",
                        description=f"Time series has {len(missing_dates)} gaps",
                        details={'missing_dates': missing_dates.tolist()}
                    ))
        
        # Check for outliers
        if len(data.dropna()) > 3:
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - 3 * iqr
            upper_bound = q3 + 3 * iqr
            
            outliers = data[(data < lower_bound) | (data > upper_bound)]
            
            if len(outliers) > 0:
                self.issues.append(DataQualityIssue(
                    issue_type="outliers",
                    severity="warning",
                    description=f"Detected {len(outliers)} potential outliers",
                    affected_rows=outliers.index.tolist(),
                    details={
                        'outlier_values': outliers.to_dict(),
                        'bounds': {'lower': lower_bound, 'upper': upper_bound}
                    }
                ))
        
        # Check for constant values
        if data.nunique() == 1:
            self.issues.append(DataQualityIssue(
                issue_type="constant_series",
                severity="error",
                description="Series contains only constant values"
            ))
        
        # Check for negative values (if should be positive)
        if data.name and 'price' in str(data.name).lower():
            negative_values = data[data < 0]
            if len(negative_values) > 0:
                self.issues.append(DataQualityIssue(
                    issue_type="negative_values",
                    severity="error",
                    description=f"Price series contains {len(negative_values)} negative values",
                    affected_rows=negative_values.index.tolist()
                ))
        
        # Compile report
        error_count = sum(1 for issue in self.issues if issue.severity == "error")
        warning_count = sum(1 for issue in self.issues if issue.severity == "warning")
        
        return {
            'valid': error_count == 0,
            'error_count': error_count,
            'warning_count': warning_count,
            'total_issues': len(self.issues),
            'issues': [self._issue_to_dict(issue) for issue in self.issues]
        }
    
    def validate_panel_data(self, data: pd.DataFrame,
                          entity_col: str,
                          time_col: str,
                          value_cols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Validate panel data structure and quality
        
        Parameters
        ----------
        data : pd.DataFrame
            Panel data to validate
        entity_col : str
            Column containing entity identifiers
        time_col : str
            Column containing time identifiers
        value_cols : Optional[List[str]]
            Columns to check for data quality
            
        Returns
        -------
        Dict[str, Any]
            Validation report
        """
        logger.info(f"Validating panel data with {len(data)} observations")
        self.issues = []
        
        # Check required columns exist
        required_cols = [entity_col, time_col]
        missing_cols = set(required_cols) - set(data.columns)
        
        if missing_cols:
            self.issues.append(DataQualityIssue(
                issue_type="missing_columns",
                severity="error",
                description=f"Required columns missing: {missing_cols}"
            ))
            return self._compile_report()
        
        # Check for duplicate entity-time combinations
        duplicates = data.duplicated(subset=[entity_col, time_col])
        if duplicates.any():
            self.issues.append(DataQualityIssue(
                issue_type="duplicate_observations",
                severity="error",
                description=f"Found {duplicates.sum()} duplicate entity-time combinations",
                affected_rows=data[duplicates].index.tolist()
            ))
        
        # Check panel balance
        entity_counts = data.groupby(entity_col)[time_col].count()
        time_counts = data.groupby(time_col)[entity_col].count()
        
        if entity_counts.std() / entity_counts.mean() > 0.1:
            self.issues.append(DataQualityIssue(
                issue_type="unbalanced_panel",
                severity="warning",
                description="Panel is unbalanced across entities",
                details={
                    'min_observations': int(entity_counts.min()),
                    'max_observations': int(entity_counts.max()),
                    'mean_observations': float(entity_counts.mean())
                }
            ))
        
        # Validate value columns
        if value_cols is None:
            value_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in value_cols:
            if col not in data.columns:
                continue
                
            # Check missing values by entity
            missing_by_entity = data.groupby(entity_col)[col].apply(lambda x: x.isna().sum())
            entities_with_missing = missing_by_entity[missing_by_entity > 0]
            
            if len(entities_with_missing) > 0:
                self.issues.append(DataQualityIssue(
                    issue_type="missing_values",
                    severity="warning",
                    description=f"{len(entities_with_missing)} entities have missing values in {col}",
                    affected_columns=[col],
                    details={'entities': entities_with_missing.to_dict()}
                ))
            
            # Check for zero variance within entities
            variance_by_entity = data.groupby(entity_col)[col].var()
            zero_var_entities = variance_by_entity[variance_by_entity == 0].index.tolist()
            
            if zero_var_entities:
                self.issues.append(DataQualityIssue(
                    issue_type="zero_variance",
                    severity="warning",
                    description=f"{len(zero_var_entities)} entities have zero variance in {col}",
                    affected_columns=[col],
                    details={'entities': zero_var_entities}
                ))
        
        return self._compile_report()
    
    def validate_weather_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate weather data specific requirements
        
        Parameters
        ----------
        data : pd.DataFrame
            Weather data to validate
            
        Returns
        -------
        Dict[str, Any]
            Validation report
        """
        logger.info("Validating weather data")
        self.issues = []
        
        # Check temperature columns
        temp_cols = [col for col in data.columns if 'temp' in col.lower()]
        
        for col in temp_cols:
            # Check reasonable temperature ranges (Fahrenheit assumed)
            unreasonable = data[(data[col] < -50) | (data[col] > 150)]
            if len(unreasonable) > 0:
                self.issues.append(DataQualityIssue(
                    issue_type="unreasonable_values",
                    severity="error",
                    description=f"Unreasonable temperature values in {col}",
                    affected_columns=[col],
                    affected_rows=unreasonable.index.tolist()
                ))
        
        # Check precipitation columns
        precip_cols = [col for col in data.columns if 'precip' in col.lower() or 'rain' in col.lower()]
        
        for col in precip_cols:
            # Check for negative precipitation
            negative = data[data[col] < 0]
            if len(negative) > 0:
                self.issues.append(DataQualityIssue(
                    issue_type="negative_values",
                    severity="error",
                    description=f"Negative precipitation values in {col}",
                    affected_columns=[col],
                    affected_rows=negative.index.tolist()
                ))
        
        return self._compile_report()
    
    def _issue_to_dict(self, issue: DataQualityIssue) -> Dict[str, Any]:
        """Convert issue to dictionary"""
        return {
            'type': issue.issue_type,
            'severity': issue.severity,
            'description': issue.description,
            'affected_columns': issue.affected_columns,
            'affected_rows': issue.affected_rows[:10] if issue.affected_rows else None,  # Limit size
            'details': issue.details
        }
    
    def _compile_report(self) -> Dict[str, Any]:
        """Compile validation report"""
        error_count = sum(1 for issue in self.issues if issue.severity == "error")
        warning_count = sum(1 for issue in self.issues if issue.severity == "warning")
        
        return {
            'valid': error_count == 0,
            'error_count': error_count,
            'warning_count': warning_count,
            'total_issues': len(self.issues),
            'issues': [self._issue_to_dict(issue) for issue in self.issues]
        }