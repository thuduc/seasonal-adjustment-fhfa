#!/usr/bin/env python3
"""Main entry point for FHFA Seasonal Adjustment Pipeline"""

import click
from pathlib import Path
from loguru import logger
import sys
import json
from datetime import datetime

from src.pipeline.orchestrator import SeasonalAdjustmentPipeline
from src.config import get_settings
from src.utils import (
    setup_logging, 
    resource_monitor,
    metrics_collector,
    health_checker,
    performance_monitor
)


@click.command()
@click.option('--hpi-data', type=click.Path(exists=True), help='Path to HPI data file')
@click.option('--weather-data', type=click.Path(exists=True), help='Path to weather data file')
@click.option('--demographic-data', type=click.Path(exists=True), help='Path to demographic data file')
@click.option('--output-dir', type=click.Path(), default='./output', help='Output directory')
@click.option('--report/--no-report', default=True, help='Generate HTML/Excel reports')
@click.option('--report-formats', multiple=True, default=['html', 'excel'], 
              help='Report formats to generate (html, excel, json, pdf)')
@click.option('--log-level', default='INFO', 
              type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']))
@click.option('--mode', default='full',
              type=click.Choice(['full', 'adjustment-only', 'impact-only']),
              help='Pipeline mode')
@click.option('--enable-monitoring/--no-monitoring', default=True,
              help='Enable resource monitoring')
@click.option('--metrics-export', type=click.Path(),
              help='Export metrics to file after completion')
def main(hpi_data, weather_data, demographic_data, output_dir, report, 
         report_formats, log_level, mode, enable_monitoring, metrics_export):
    """
    FHFA Seasonal Adjustment Pipeline
    
    This pipeline performs seasonal adjustment on housing price indices
    and analyzes the impact of weather and other factors on seasonality.
    
    Example usage:
    
    \b
    # Run with synthetic data
    python main.py
    
    \b
    # Run with real data files
    python main.py --hpi-data data/hpi.csv --weather-data data/weather.csv
    
    \b
    # Run adjustment only
    python main.py --mode adjustment-only
    
    \b
    # Generate JSON report
    python main.py --report-formats json
    """
    
    # Configure logging
    setup_logging(get_settings())
    logger.remove()
    logger.add(sys.stderr, level=log_level)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.add(output_path / 'pipeline.log', 
              level=log_level,
              rotation="10 MB")
    
    # Start monitoring if enabled
    if enable_monitoring:
        resource_monitor.start()
        metrics_collector.start()
        logger.info("Resource monitoring enabled")
    
    # Initialize pipeline
    logger.info("Initializing FHFA Seasonal Adjustment Pipeline")
    pipeline = SeasonalAdjustmentPipeline()
    
    try:
        if mode == 'full':
            # Run full pipeline
            logger.info("Running full seasonal adjustment pipeline")
            results = pipeline.run_full_pipeline(
                hpi_data_path=hpi_data,
                weather_data_path=weather_data,
                demographic_data_path=demographic_data,
                output_dir=output_dir,
                generate_report=report,
                report_formats=list(report_formats)
            )
            
            # Print summary
            logger.info("\n" + pipeline.get_summary_report())
            
            # Generate visualizations
            if report:
                logger.info("Generating additional visualizations")
                plot_paths = pipeline.generate_visualizations(
                    output_dir,
                    plot_types=['adjustment', 'diagnostics', 'seasonal_patterns']
                )
                logger.info(f"Generated {len(plot_paths)} visualization plots")
                
        elif mode == 'adjustment-only':
            logger.info("Running seasonal adjustment only mode")
            
            # Load data
            if not hpi_data:
                logger.warning("No HPI data provided, using synthetic data")
                from src.data.loaders import HPIDataLoader
                loader = HPIDataLoader()
                data = loader.load(
                    start_date='2010-01-01',
                    end_date='2023-12-31',
                    geography='state'
                )
            else:
                import pandas as pd
                data = pd.read_csv(hpi_data, parse_dates=['date'], index_col='date')
            
            # Run adjustment for each series
            results = {}
            for col in data.columns:
                if col.startswith('hpi_'):
                    logger.info(f"Adjusting {col}")
                    adjusted = pipeline.run_seasonal_adjustment_only(
                        data[col],
                        method='classical'
                    )
                    results[col] = adjusted
                    
            # Save results
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            adjusted_df = pd.DataFrame(results)
            adjusted_df.to_csv(output_path / 'adjusted_series.csv')
            logger.info(f"Saved adjusted series to {output_path / 'adjusted_series.csv'}")
            
        elif mode == 'impact-only':
            logger.info("Running impact analysis only mode")
            
            if not hpi_data:
                logger.error("HPI data required for impact analysis mode")
                sys.exit(1)
                
            # Load panel data
            import pandas as pd
            panel_data = pd.read_csv(hpi_data)
            
            # Run impact analysis
            results = pipeline.run_impact_analysis_only(
                panel_data,
                model_type='fixed_effects'
            )
            
            # Display results
            logger.info("\nImpact Analysis Results:")
            logger.info("-" * 50)
            
            if 'coefficients' in results:
                logger.info("\nCoefficients:")
                logger.info(results['coefficients'].to_string())
                
            if 'summary' in results:
                logger.info("\nModel Summary:")
                logger.info(results['summary'])
                
        logger.success("Pipeline completed successfully!")
        
        # Export metrics if requested
        if metrics_export and enable_monitoring:
            try:
                # Get performance summary
                perf_summary = performance_monitor.get_metrics_summary()
                
                # Get resource summary
                resource_summary = resource_monitor.get_resource_summary(last_minutes=60)
                
                # Combine metrics
                all_metrics = {
                    "performance": perf_summary,
                    "resources": resource_summary,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                # Export
                export_path = Path(metrics_export)
                with open(export_path, 'w') as f:
                    json.dump(all_metrics, f, indent=2)
                    
                logger.info(f"Exported metrics to {export_path}")
            except Exception as e:
                logger.error(f"Failed to export metrics: {e}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.exception("Full traceback:")
        
        # Record failure metric
        if enable_monitoring:
            metrics_collector.record("pipeline.failure", 1, error_type=type(e).__name__)
            
        sys.exit(1)
    finally:
        # Stop monitoring
        if enable_monitoring:
            resource_monitor.stop()
            metrics_collector.stop()


if __name__ == '__main__':
    main()