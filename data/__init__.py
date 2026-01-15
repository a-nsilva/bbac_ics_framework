"""
BBAC Framework - Data Management

This package contains data loading and processing utilities:
- Dataset Loader: Load and process bbac_ics_dataset
- Data Preprocessing: Feature engineering and normalization
"""

__version__ = "0.1.0"

from .dataset_loader import DatasetLoader

__all__ = [
    "DatasetLoader"
]
