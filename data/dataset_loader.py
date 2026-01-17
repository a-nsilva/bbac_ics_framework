"""
BBAC Framework - Dataset Loader

This module handles loading and preprocessing of the bbac_ics_dataset.
"""

# 1. Biblioteca padrão
import json
import logging
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# 2. Bibliotecas de terceiros
import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetLoader:
    """
    Load and preprocess data from bbac_ics_dataset repository.
    
    Expected dataset structure:
    - access_logs.csv: Historical access request logs
    - agent_profiles.json: Agent behavioral profiles
    - normal_patterns.csv: Normal behavior baselines
    - anomalies.csv: Known anomaly examples
    """
    
    def __init__(self, dataset_path: str):
        """
        Initialize the dataset loader.
        
        Args:
            dataset_path: Path to the dataset directory (default: data/data_100k)
        """
        self.dataset_path = Path(dataset_path)
        
        # Check if dataset exists
        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"\n{'='*70}\n"
                f"Dataset not found at: {self.dataset_path}\n"
                f"{'='*70}\n"
                f"Please ensure bbac_ics_dataset has been generated first.\n"
                f"Required files:\n"
                f"  - bbac_train.csv\n"
                f"  - bbac_val.csv\n"
                f"  - bbac_test.csv\n"
                f"  - agents.json\n"
                f"{'='*70}\n"
            )
        
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.agents = None
        
        logger.info(f"Initialized DatasetLoader with path: {self.dataset_path}")
    
    def load_all(self) -> bool:
        """
        Load all dataset files.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.load_train()
            self.load_val()
            self.load_test()
            self.load_agents()
            logger.info("All dataset files loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return False
    
    def load_train(self, filename: str = "bbac_train.csv") -> pd.DataFrame:
        """
        Load training dataset.
        
        Returns:
            DataFrame with training data
        """
        filepath = self.dataset_path / filename
        
        if not filepath.exists():
            raise FileNotFoundError(
                f"Training dataset not found: {filepath}\n"
                f"Run bbac_ics_dataset first to generate the data."
            )
        
        self.train_data = pd.read_csv(filepath)
        
        # Convert timestamp if exists
        if 'timestamp' in self.train_data.columns:
            self.train_data['timestamp'] = pd.to_datetime(self.train_data['timestamp'], format='mixed', errors='coerce')
        
        logger.info(f"Loaded {len(self.train_data)} training samples")
        return self.train_data
    
    def load_val(self, filename: str = "bbac_val.csv") -> pd.DataFrame:
        """
        Load validation dataset.
        
        Returns:
            DataFrame with validation data
        """
        filepath = self.dataset_path / filename
        
        if not filepath.exists():
            raise FileNotFoundError(
                f"Validation dataset not found: {filepath}\n"
                f"Run bbac_ics_dataset first to generate the data."
            )
        
        self.val_data = pd.read_csv(filepath)
        
        if 'timestamp' in self.val_data.columns:
            self.val_data['timestamp'] = pd.to_datetime(self.val_data['timestamp'], format='mixed', errors='coerce')
        
        logger.info(f"Loaded {len(self.val_data)} validation samples")
        return self.val_data
    
    def load_test(self, filename: str = "bbac_test.csv") -> pd.DataFrame:
        """
        Load test dataset.
        
        Returns:
            DataFrame with test data
        """
        filepath = self.dataset_path / filename
        
        if not filepath.exists():
            raise FileNotFoundError(
                f"Test dataset not found: {filepath}\n"
                f"Run bbac_ics_dataset first to generate the data."
            )
        
        self.test_data = pd.read_csv(filepath)
        
        if 'timestamp' in self.test_data.columns:
            self.test_data['timestamp'] = pd.to_datetime(self.test_data['timestamp'], format='mixed', errors='coerce')
        
        logger.info(f"Loaded {len(self.test_data)} test samples")
        return self.test_data
    
    def load_agents(self, filename: str = "agents.json") -> Dict:
        """
        Load agent profiles.
        
        Returns:
            Dictionary with agent profiles
        """
        filepath = self.dataset_path / filename
        
        if not filepath.exists():
            raise FileNotFoundError(
                f"Agents file not found: {filepath}\n"
                f"Run bbac_ics_dataset first to generate the data."
            )
        
        with open(filepath, 'r') as f:
            self.agents = json.load(f)
        
        logger.info(f"Loaded {len(self.agents)} agent profiles")
        return self.agents
    
    def get_statistics(self) -> Dict:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with statistics
        """
        if self.train_data is None:
            self.load_train()
        
        stats = {
            'train_samples': len(self.train_data) if self.train_data is not None else 0,
            'val_samples': len(self.val_data) if self.val_data is not None else 0,
            'test_samples': len(self.test_data) if self.test_data is not None else 0,
        }
        
        # Add column statistics if available
        if self.train_data is not None and len(self.train_data) > 0:
            stats.update({
                'unique_agents': self.train_data['agent_id'].nunique() if 'agent_id' in self.train_data.columns else 0,
                'unique_resources': self.train_data['resource_id'].nunique() if 'resource_id' in self.train_data.columns else 0,
                'columns': list(self.train_data.columns)
            })
        
        return stats


if __name__ == "__main__":
    # Test the dataset loader
    try:
        loader = DatasetLoader()
        loader.load_all()
        
        print("\nDataset Statistics:")
        stats = loader.get_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        if loader.train_data is not None:
            print("\nSample Training Data:")
            print(loader.train_data.head())
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease run bbac_ics_dataset first to generate the data.")
