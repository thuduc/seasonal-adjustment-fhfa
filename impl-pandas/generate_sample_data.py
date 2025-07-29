#!/usr/bin/env python
"""Script to generate and save sample data for tests."""

from pathlib import Path
from src.data.data_loader import DataLoader


def main():
    """Generate sample data and save to data folder."""
    # Create data directory if it doesn't exist
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # Create DataLoader and generate sample data
    loader = DataLoader()
    loader.create_sample_data(data_dir)
    
    print(f"Sample data generated successfully in {data_dir.absolute()}")
    print("\nGenerated files:")
    for file in data_dir.glob('*.csv'):
        print(f"  - {file.name}")


if __name__ == "__main__":
    main()