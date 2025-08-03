#!/usr/bin/env python
"""Validate seasonal adjustment results against baseline"""

import click
import pandas as pd
import json
from pathlib import Path
import sys
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.validation import ResultValidator, DataQualityValidator, ModelValidator
from loguru import logger


@click.command()
@click.option('--baseline-path', required=True, help='Path to baseline results')
@click.option('--new-path', required=True, help='Path to new results')
@click.option('--tolerance', default=0.001, type=float, help='Tolerance for comparison')
@click.option('--output', default=None, help='Path to save validation report')
@click.option('--strict', is_flag=True, help='Fail on any validation errors')
@click.option('--validate-data-quality', is_flag=True, help='Also validate data quality')
def validate_results(baseline_path: str, new_path: str, tolerance: float, 
                    output: Optional[str], strict: bool, validate_data_quality: bool):
    """Validate new results against baseline"""
    
    logger.info(f"Validating results: {new_path} against baseline: {baseline_path}")
    
    # Initialize validator
    validator = ResultValidator(tolerance=tolerance, strict_mode=strict)
    
    try:
        # Load data
        baseline = pd.read_csv(baseline_path, index_col=0, parse_dates=True)
        new_results = pd.read_csv(new_path, index_col=0, parse_dates=True)
        
        # Run validation
        validation_report = validator.compare(baseline, new_results)
        
        # Optional data quality validation
        if validate_data_quality:
            dq_validator = DataQualityValidator()
            
            # Validate baseline quality
            baseline_quality = {}
            for col in baseline.select_dtypes(include=['float64', 'int64']).columns:
                baseline_quality[f'baseline_{col}'] = dq_validator.validate_time_series(baseline[col])
            
            # Validate new results quality
            new_quality = {}
            for col in new_results.select_dtypes(include=['float64', 'int64']).columns:
                new_quality[f'new_{col}'] = dq_validator.validate_time_series(new_results[col])
            
            validation_report['data_quality'] = {
                'baseline': baseline_quality,
                'new': new_quality
            }
        
        # Display results
        if validation_report['all_tests_passed']:
            click.echo(click.style("✅ All validation tests passed!", fg='green'))
        else:
            click.echo(click.style("❌ Validation failed:", fg='red'))
            
            # Show failed tests
            for test_name, test_result in validation_report['tests'].items():
                if not test_result['passed']:
                    click.echo(f"  - {test_name}: {test_result['message']}")
                    if test_result.get('details'):
                        click.echo(f"    Details: {json.dumps(test_result['details'], indent=4)}")
        
        # Summary statistics
        click.echo(f"\nValidation Summary:")
        click.echo(f"  Total tests: {validation_report['total_tests']}")
        click.echo(f"  Passed: {validation_report['passed_tests']}")
        click.echo(f"  Failed: {validation_report['failed_tests']}")
        
        # Save report if requested
        if output:
            validator.save_report(validation_report, output)
            click.echo(f"\nValidation report saved to: {output}")
        
        # Exit with appropriate code
        sys.exit(0 if validation_report['all_tests_passed'] else 1)
        
    except FileNotFoundError as e:
        click.echo(click.style(f"Error: File not found - {e}", fg='red'))
        sys.exit(1)
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg='red'))
        logger.exception("Validation failed with error")
        sys.exit(1)


@click.command()
@click.option('--data-path', required=True, help='Path to data file to validate')
@click.option('--data-type', 
              type=click.Choice(['time_series', 'panel', 'weather']), 
              default='time_series',
              help='Type of data to validate')
@click.option('--entity-col', default='entity', help='Entity column for panel data')
@click.option('--time-col', default='time', help='Time column for panel data')
@click.option('--output', default=None, help='Path to save validation report')
def validate_data(data_path: str, data_type: str, entity_col: str, 
                 time_col: str, output: Optional[str]):
    """Validate data quality before processing"""
    
    logger.info(f"Validating {data_type} data from: {data_path}")
    
    validator = DataQualityValidator()
    
    try:
        # Load data
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        
        # Run appropriate validation
        if data_type == 'time_series':
            if isinstance(data, pd.DataFrame):
                # Validate each column
                report = {}
                for col in data.columns:
                    report[col] = validator.validate_time_series(data[col])
            else:
                report = validator.validate_time_series(data)
                
        elif data_type == 'panel':
            report = validator.validate_panel_data(data, entity_col, time_col)
            
        elif data_type == 'weather':
            report = validator.validate_weather_data(data)
        
        # Display results
        if isinstance(report, dict) and report.get('valid', True):
            click.echo(click.style("✅ Data validation passed!", fg='green'))
        else:
            click.echo(click.style("⚠️  Data quality issues detected:", fg='yellow'))
            
            # Show issues
            if 'issues' in report:
                for issue in report['issues']:
                    severity_color = 'red' if issue['severity'] == 'error' else 'yellow'
                    click.echo(click.style(f"  [{issue['severity'].upper()}] {issue['description']}", 
                                         fg=severity_color))
        
        # Save report if requested
        if output:
            with open(output, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            click.echo(f"\nValidation report saved to: {output}")
        
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg='red'))
        logger.exception("Data validation failed")
        sys.exit(1)


@click.command()
@click.option('--model-type',
              type=click.Choice(['arima', 'regression', 'panel', 'seasonal']),
              required=True,
              help='Type of model to validate')
@click.option('--residuals-path', required=True, help='Path to residuals file')
@click.option('--original-path', help='Path to original series (for seasonal adjustment)')
@click.option('--adjusted-path', help='Path to adjusted series (for seasonal adjustment)')
@click.option('--output', default=None, help='Path to save validation report')
def validate_model(model_type: str, residuals_path: str, original_path: Optional[str],
                  adjusted_path: Optional[str], output: Optional[str]):
    """Validate model assumptions and diagnostics"""
    
    logger.info(f"Validating {model_type} model")
    
    validator = ModelValidator()
    
    try:
        # Load residuals
        residuals = pd.read_csv(residuals_path, index_col=0, parse_dates=True)
        if isinstance(residuals, pd.DataFrame):
            residuals = residuals.iloc[:, 0]  # Take first column
        
        # Run appropriate validation
        if model_type == 'arima':
            report = validator.validate_arima_assumptions(residuals)
            
        elif model_type == 'seasonal':
            if not original_path or not adjusted_path:
                click.echo(click.style("Error: Original and adjusted paths required for seasonal validation", 
                                     fg='red'))
                sys.exit(1)
                
            original = pd.read_csv(original_path, index_col=0, parse_dates=True)
            adjusted = pd.read_csv(adjusted_path, index_col=0, parse_dates=True)
            
            if isinstance(original, pd.DataFrame):
                original = original.iloc[:, 0]
            if isinstance(adjusted, pd.DataFrame):
                adjusted = adjusted.iloc[:, 0]
                
            report = validator.validate_seasonal_adjustment(original, adjusted)
        
        else:
            click.echo(click.style(f"Model type {model_type} validation not yet implemented", fg='yellow'))
            sys.exit(1)
        
        # Display results
        if report.get('assumptions_met', report.get('adjustment_effective', False)):
            click.echo(click.style("✅ Model validation passed!", fg='green'))
        else:
            click.echo(click.style("⚠️  Model validation issues detected:", fg='yellow'))
        
        # Show test results
        if 'tests' in report:
            for test_name, test_result in report['tests'].items():
                if isinstance(test_result, dict) and 'passed' in test_result:
                    status = '✓' if test_result['passed'] else '✗'
                    color = 'green' if test_result['passed'] else 'red'
                    click.echo(click.style(f"  {status} {test_name}: {test_result.get('interpretation', '')}",
                                         fg=color))
        
        # Save report if requested
        if output:
            with open(output, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            click.echo(f"\nValidation report saved to: {output}")
            
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg='red'))
        logger.exception("Model validation failed")
        sys.exit(1)


# Create CLI group
@click.group()
def cli():
    """FHFA Seasonal Adjustment Validation Tools"""
    pass


# Add commands to group
cli.add_command(validate_results, name='results')
cli.add_command(validate_data, name='data')
cli.add_command(validate_model, name='model')


if __name__ == '__main__':
    cli()