"""Result validation for comparing seasonal adjustment outputs against baselines"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import json
from loguru import logger


@dataclass
class ValidationResult:
    """Container for validation test results"""
    test_name: str
    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None


class ResultValidator:
    """Validate seasonal adjustment results against baseline or expected values"""
    
    def __init__(self, tolerance: float = 0.001, strict_mode: bool = False):
        """
        Initialize result validator
        
        Parameters
        ----------
        tolerance : float
            Maximum allowed relative difference for numeric comparisons
        strict_mode : bool
            If True, all tests must pass; if False, warnings are issued
        """
        self.tolerance = tolerance
        self.strict_mode = strict_mode
        self.validation_results = []
        
    def compare(self, baseline: pd.DataFrame, new_results: pd.DataFrame) -> Dict[str, Any]:
        """
        Compare new results against baseline
        
        Parameters
        ----------
        baseline : pd.DataFrame
            Baseline results for comparison
        new_results : pd.DataFrame
            New results to validate
            
        Returns
        -------
        Dict[str, Any]
            Validation report with test results
        """
        logger.info("Starting result validation")
        
        # Run validation tests
        self._validate_structure(baseline, new_results)
        self._validate_values(baseline, new_results)
        self._validate_statistical_properties(baseline, new_results)
        self._validate_seasonal_patterns(baseline, new_results)
        
        # Compile report
        all_passed = all(result.passed for result in self.validation_results)
        
        report = {
            'all_tests_passed': all_passed,
            'total_tests': len(self.validation_results),
            'passed_tests': sum(1 for r in self.validation_results if r.passed),
            'failed_tests': sum(1 for r in self.validation_results if not r.passed),
            'tests': {r.test_name: {
                'passed': r.passed,
                'message': r.message,
                'details': r.details
            } for r in self.validation_results}
        }
        
        if not all_passed and self.strict_mode:
            logger.error(f"Validation failed: {report['failed_tests']} tests failed")
        elif not all_passed:
            logger.warning(f"Validation completed with warnings: {report['failed_tests']} tests failed")
        else:
            logger.success("All validation tests passed")
            
        return report
    
    def _validate_structure(self, baseline: pd.DataFrame, new_results: pd.DataFrame):
        """Validate structural consistency between datasets"""
        # Check shape
        if baseline.shape != new_results.shape:
            self.validation_results.append(ValidationResult(
                test_name="shape_consistency",
                passed=False,
                message=f"Shape mismatch: baseline {baseline.shape} vs new {new_results.shape}",
                details={'baseline_shape': baseline.shape, 'new_shape': new_results.shape}
            ))
        else:
            self.validation_results.append(ValidationResult(
                test_name="shape_consistency",
                passed=True,
                message="Shape matches"
            ))
        
        # Check columns
        missing_cols = set(baseline.columns) - set(new_results.columns)
        extra_cols = set(new_results.columns) - set(baseline.columns)
        
        if missing_cols or extra_cols:
            self.validation_results.append(ValidationResult(
                test_name="column_consistency",
                passed=False,
                message="Column mismatch detected",
                details={'missing': list(missing_cols), 'extra': list(extra_cols)}
            ))
        else:
            self.validation_results.append(ValidationResult(
                test_name="column_consistency",
                passed=True,
                message="All columns match"
            ))
        
        # Check index
        if not baseline.index.equals(new_results.index):
            self.validation_results.append(ValidationResult(
                test_name="index_consistency",
                passed=False,
                message="Index mismatch detected"
            ))
        else:
            self.validation_results.append(ValidationResult(
                test_name="index_consistency",
                passed=True,
                message="Index matches"
            ))
    
    def _validate_values(self, baseline: pd.DataFrame, new_results: pd.DataFrame):
        """Validate numeric values are within tolerance"""
        common_cols = list(set(baseline.columns) & set(new_results.columns))
        numeric_cols = baseline[common_cols].select_dtypes(include=[np.number]).columns
        
        # Only validate if shapes match
        if baseline.shape != new_results.shape:
            return
        
        for col in numeric_cols:
            # Calculate relative difference
            baseline_vals = baseline[col].values
            new_vals = new_results[col].values
            
            # Handle zeros and NaNs
            mask = ~(np.isnan(baseline_vals) | np.isnan(new_vals) | (baseline_vals == 0))
            
            if mask.sum() > 0:
                rel_diff = np.abs((new_vals[mask] - baseline_vals[mask]) / baseline_vals[mask])
                max_diff = np.max(rel_diff)
                mean_diff = np.mean(rel_diff)
                
                if max_diff > self.tolerance:
                    self.validation_results.append(ValidationResult(
                        test_name=f"value_tolerance_{col}",
                        passed=False,
                        message=f"Values exceed tolerance for {col}",
                        details={
                            'max_relative_diff': float(max_diff),
                            'mean_relative_diff': float(mean_diff),
                            'tolerance': self.tolerance
                        }
                    ))
                else:
                    self.validation_results.append(ValidationResult(
                        test_name=f"value_tolerance_{col}",
                        passed=True,
                        message=f"Values within tolerance for {col}",
                        details={'max_relative_diff': float(max_diff)}
                    ))
    
    def _validate_statistical_properties(self, baseline: pd.DataFrame, new_results: pd.DataFrame):
        """Validate statistical properties are preserved"""
        common_cols = list(set(baseline.columns) & set(new_results.columns))
        numeric_cols = baseline[common_cols].select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            baseline_stats = {
                'mean': baseline[col].mean(),
                'std': baseline[col].std(),
                'min': baseline[col].min(),
                'max': baseline[col].max()
            }
            
            new_stats = {
                'mean': new_results[col].mean(),
                'std': new_results[col].std(),
                'min': new_results[col].min(),
                'max': new_results[col].max()
            }
            
            # Check if basic statistics are similar
            stats_match = True
            for stat_name in ['mean', 'std']:
                if baseline_stats[stat_name] != 0:
                    rel_diff = abs(new_stats[stat_name] - baseline_stats[stat_name]) / abs(baseline_stats[stat_name])
                    if rel_diff > self.tolerance * 10:  # More lenient for statistics
                        stats_match = False
                        break
            
            self.validation_results.append(ValidationResult(
                test_name=f"statistical_properties_{col}",
                passed=stats_match,
                message=f"Statistical properties {'match' if stats_match else 'differ'} for {col}",
                details={'baseline': baseline_stats, 'new': new_stats}
            ))
    
    def _validate_seasonal_patterns(self, baseline: pd.DataFrame, new_results: pd.DataFrame):
        """Validate seasonal patterns are properly removed in adjusted series"""
        # Look for seasonally adjusted columns
        sa_cols = [col for col in baseline.columns if 'sa' in col.lower() or 'adjusted' in col.lower()]
        
        for col in sa_cols:
            if col not in new_results.columns:
                continue
                
            # Check if seasonal pattern is reduced
            if isinstance(baseline.index, pd.DatetimeIndex):
                # Calculate seasonal strength (simplified)
                baseline_seasonal_strength = self._calculate_seasonal_strength(baseline[col])
                new_seasonal_strength = self._calculate_seasonal_strength(new_results[col])
                
                # Adjusted series should have lower seasonal strength
                if new_seasonal_strength < baseline_seasonal_strength * 1.1:  # Allow 10% margin
                    self.validation_results.append(ValidationResult(
                        test_name=f"seasonal_adjustment_{col}",
                        passed=True,
                        message=f"Seasonal adjustment effective for {col}",
                        details={
                            'baseline_strength': baseline_seasonal_strength,
                            'new_strength': new_seasonal_strength
                        }
                    ))
                else:
                    self.validation_results.append(ValidationResult(
                        test_name=f"seasonal_adjustment_{col}",
                        passed=False,
                        message=f"Seasonal adjustment may be ineffective for {col}",
                        details={
                            'baseline_strength': baseline_seasonal_strength,
                            'new_strength': new_seasonal_strength
                        }
                    ))
    
    def _calculate_seasonal_strength(self, series: pd.Series) -> float:
        """Calculate a simple measure of seasonal strength"""
        if len(series) < 8:  # Need at least 2 years of quarterly data
            return 0.0
            
        try:
            # Simple seasonal strength: variance of quarterly means / total variance
            if hasattr(series.index, 'quarter'):
                quarterly_means = series.groupby(series.index.quarter).mean()
                seasonal_var = quarterly_means.var()
                total_var = series.var()
                
                if total_var > 0:
                    return seasonal_var / total_var
        except:
            pass
            
        return 0.0
    
    def validate_model_outputs(self, model_results: Dict[str, Any], 
                             expected_metrics: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """
        Validate model outputs against expected ranges
        
        Parameters
        ----------
        model_results : Dict[str, Any]
            Model output metrics
        expected_metrics : Dict[str, Tuple[float, float]]
            Expected ranges for each metric (min, max)
            
        Returns
        -------
        Dict[str, Any]
            Validation report
        """
        validation_results = []
        
        for metric_name, (min_val, max_val) in expected_metrics.items():
            if metric_name in model_results:
                value = model_results[metric_name]
                
                if min_val <= value <= max_val:
                    validation_results.append(ValidationResult(
                        test_name=f"metric_range_{metric_name}",
                        passed=True,
                        message=f"{metric_name} within expected range",
                        details={'value': value, 'range': (min_val, max_val)}
                    ))
                else:
                    validation_results.append(ValidationResult(
                        test_name=f"metric_range_{metric_name}",
                        passed=False,
                        message=f"{metric_name} outside expected range",
                        details={'value': value, 'range': (min_val, max_val)}
                    ))
            else:
                validation_results.append(ValidationResult(
                    test_name=f"metric_range_{metric_name}",
                    passed=False,
                    message=f"{metric_name} not found in results"
                ))
        
        all_passed = all(r.passed for r in validation_results)
        
        return {
            'all_tests_passed': all_passed,
            'tests': {r.test_name: {
                'passed': r.passed,
                'message': r.message,
                'details': r.details
            } for r in validation_results}
        }
    
    def save_report(self, report: Dict[str, Any], filepath: str):
        """Save validation report to file"""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Validation report saved to {filepath}")