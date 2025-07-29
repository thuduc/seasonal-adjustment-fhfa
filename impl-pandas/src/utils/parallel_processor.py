import pandas as pd
import numpy as np
from typing import Dict, List, Callable, Any, Optional
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial
import time

logger = logging.getLogger('seasonal_adjustment.parallel_processor')


class ParallelProcessor:
    """Handles parallel processing for seasonal adjustment tasks."""
    
    def __init__(self, n_cores: Optional[int] = None, backend: str = 'process'):
        """
        Initialize parallel processor.
        
        Args:
            n_cores: Number of cores to use. None for auto-detect.
            backend: 'process' for CPU-bound tasks, 'thread' for I/O-bound tasks
        """
        if n_cores is None:
            self.n_cores = min(mp.cpu_count() - 1, 8)
        else:
            self.n_cores = min(n_cores, mp.cpu_count())
            
        self.backend = backend
        logger.info(f"Initialized ParallelProcessor with {self.n_cores} cores, backend={backend}")
        
    def setup_multiprocessing(self, n_cores: Optional[int] = None):
        """Configure multiprocessing pool."""
        if n_cores is not None:
            self.n_cores = min(n_cores, mp.cpu_count())
            
        # Set multiprocessing start method for compatibility
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set
            
    def parallel_arima_fit(self, geography_list: List[str], 
                          data: pd.DataFrame,
                          fit_function: Callable,
                          **kwargs) -> Dict[str, Any]:
        """Fit ARIMA models in parallel for multiple geographies."""
        logger.info(f"Starting parallel ARIMA fitting for {len(geography_list)} geographies")
        
        start_time = time.time()
        results = {}
        
        # Create partial function with fixed arguments
        partial_fit = partial(self._fit_single_geography, 
                            data=data, 
                            fit_function=fit_function,
                            **kwargs)
        
        # Choose executor based on backend
        Executor = ProcessPoolExecutor if self.backend == 'process' else ThreadPoolExecutor
        
        with Executor(max_workers=self.n_cores) as executor:
            # Submit all tasks
            future_to_geo = {
                executor.submit(partial_fit, geo_id): geo_id 
                for geo_id in geography_list
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_geo):
                geo_id = future_to_geo[future]
                try:
                    result = future.result()
                    results[geo_id] = result
                    completed += 1
                    
                    if completed % 10 == 0:
                        logger.info(f"Progress: {completed}/{len(geography_list)} geographies completed")
                        
                except Exception as e:
                    logger.error(f"Failed to process {geo_id}: {str(e)}")
                    results[geo_id] = {
                        'status': 'failed',
                        'error': str(e),
                        'geography_id': geo_id
                    }
                    
        elapsed_time = time.time() - start_time
        logger.info(f"Parallel ARIMA fitting completed in {elapsed_time:.2f} seconds")
        logger.info(f"Average time per geography: {elapsed_time/len(geography_list):.2f} seconds")
        
        return results
    
    def _fit_single_geography(self, geo_id: str, data: pd.DataFrame, 
                            fit_function: Callable, **kwargs) -> Dict[str, Any]:
        """Helper to fit model for single geography."""
        try:
            # Filter data for this geography
            geo_data = data[data['geography_id'] == geo_id]
            
            if len(geo_data) < 20:
                return {
                    'geography_id': geo_id,
                    'status': 'failed',
                    'error': 'Insufficient data'
                }
                
            # Call the fitting function
            result = fit_function(geo_id, geo_data, **kwargs)
            return result
            
        except Exception as e:
            logger.error(f"Error in _fit_single_geography for {geo_id}: {str(e)}")
            return {
                'geography_id': geo_id,
                'status': 'failed',
                'error': str(e)
            }
            
    def parallel_regression(self, data_chunks: List[pd.DataFrame],
                          regression_function: Callable,
                          **kwargs) -> pd.DataFrame:
        """Run regression analysis in parallel on data chunks."""
        logger.info(f"Starting parallel regression on {len(data_chunks)} chunks")
        
        results = []
        
        # Create partial function
        partial_regression = partial(regression_function, **kwargs)
        
        with ProcessPoolExecutor(max_workers=self.n_cores) as executor:
            # Submit all chunks
            futures = [
                executor.submit(partial_regression, chunk) 
                for chunk in data_chunks
            ]
            
            # Collect results
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Regression chunk failed: {str(e)}")
                    
        # Combine results
        if results:
            combined_results = pd.concat(results, ignore_index=True)
        else:
            combined_results = pd.DataFrame()
            
        logger.info("Parallel regression completed")
        
        return combined_results
    
    def parallel_forecast(self, models: Dict[str, Any],
                         forecast_horizon: int,
                         forecast_function: Callable) -> Dict[str, pd.Series]:
        """Generate forecasts in parallel for multiple models."""
        logger.info(f"Generating parallel forecasts for {len(models)} models")
        
        forecasts = {}
        
        # Create partial function
        partial_forecast = partial(forecast_function, horizon=forecast_horizon)
        
        with ThreadPoolExecutor(max_workers=self.n_cores) as executor:
            # Submit forecast tasks
            future_to_geo = {
                executor.submit(partial_forecast, model): geo_id
                for geo_id, model in models.items()
            }
            
            # Collect forecasts
            for future in as_completed(future_to_geo):
                geo_id = future_to_geo[future]
                try:
                    forecast = future.result()
                    forecasts[geo_id] = forecast
                except Exception as e:
                    logger.error(f"Forecast failed for {geo_id}: {str(e)}")
                    
        return forecasts
    
    def chunk_data_for_parallel(self, data: pd.DataFrame, 
                              chunk_size: Optional[int] = None) -> List[pd.DataFrame]:
        """Split data into chunks for parallel processing."""
        
        if chunk_size is None:
            # Determine optimal chunk size based on data and cores
            total_rows = len(data)
            chunk_size = max(1000, total_rows // (self.n_cores * 4))
            
        chunks = []
        for i in range(0, len(data), chunk_size):
            chunk = data.iloc[i:i + chunk_size]
            chunks.append(chunk)
            
        logger.info(f"Split data into {len(chunks)} chunks of size ~{chunk_size}")
        
        return chunks
    
    def parallel_apply(self, data: pd.DataFrame, 
                      func: Callable, 
                      axis: int = 0,
                      **kwargs) -> pd.DataFrame:
        """Apply function to DataFrame in parallel."""
        logger.debug("Starting parallel apply operation")
        
        if axis == 0:
            # Apply to rows
            chunks = self.chunk_data_for_parallel(data)
            
            with ThreadPoolExecutor(max_workers=self.n_cores) as executor:
                results = list(executor.map(
                    lambda chunk: chunk.apply(func, axis=axis, **kwargs),
                    chunks
                ))
                
            return pd.concat(results)
            
        else:
            # Apply to columns - usually fewer, so thread-based is fine
            columns = data.columns.tolist()
            
            with ThreadPoolExecutor(max_workers=min(self.n_cores, len(columns))) as executor:
                results = {}
                
                futures = {
                    executor.submit(func, data[col], **kwargs): col
                    for col in columns
                }
                
                for future in as_completed(futures):
                    col = futures[future]
                    try:
                        results[col] = future.result()
                    except Exception as e:
                        logger.error(f"Failed to process column {col}: {str(e)}")
                        results[col] = pd.Series(index=data.index)
                        
            return pd.DataFrame(results)
    
    def parallel_io_operations(self, file_paths: List[str],
                             operation: str = 'read',
                             read_function: Optional[Callable] = None,
                             write_function: Optional[Callable] = None,
                             data_dict: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, pd.DataFrame]:
        """Perform I/O operations in parallel."""
        logger.info(f"Starting parallel {operation} for {len(file_paths)} files")
        
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.n_cores) as executor:
            if operation == 'read':
                if read_function is None:
                    read_function = pd.read_csv
                    
                future_to_path = {
                    executor.submit(read_function, path): path
                    for path in file_paths
                }
                
                for future in as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        data = future.result()
                        results[path] = data
                    except Exception as e:
                        logger.error(f"Failed to read {path}: {str(e)}")
                        
            elif operation == 'write' and data_dict is not None:
                if write_function is None:
                    write_function = lambda df, path: df.to_csv(path, index=False)
                    
                futures = []
                for path, data in data_dict.items():
                    future = executor.submit(write_function, data, path)
                    futures.append(future)
                    
                # Wait for all writes to complete
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Write operation failed: {str(e)}")
                        
        return results