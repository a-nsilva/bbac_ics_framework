"""
BBAC Framework - Dataset Loader

This module handles loading and preprocessing of the bbac_ics_dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import yaml
import logging

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
    
    def __init__(self, dataset_path: str = "data/bbac_ics_dataset"):
        """
        Initialize the dataset loader.
        
        Args:
            dataset_path: Path to the bbac_ics_dataset directory
        """
        self.dataset_path = Path(dataset_path)
        self.access_logs = None
        self.agent_profiles = None
        self.normal_patterns = None
        self.anomalies = None
        
        logger.info(f"Initialized DatasetLoader with path: {self.dataset_path}")
    
    def load_all(self) -> bool:
        """
        Load all dataset files.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.load_access_logs()
            self.load_agent_profiles()
            self.load_normal_patterns()
            self.load_anomalies()
            logger.info("All dataset files loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return False
    
    def load_access_logs(self, filename: str = "access_logs.csv") -> pd.DataFrame:
        """
        Load historical access logs.
        
        Expected columns:
        - timestamp: Access request timestamp
        - agent_id: ID of the requesting agent
        - agent_type: Type (robot/human)
        - resource_id: Target resource
        - action: Requested action (read/write/execute)
        - decision: Grant/Deny
        - context: Additional contextual information
        
        Returns:
            DataFrame with access logs
        """
        filepath = self.dataset_path / filename
        
        # If file doesn't exist, create sample data
        if not filepath.exists():
            logger.warning(f"{filename} not found. Creating sample data...")
            self.access_logs = self._create_sample_access_logs()
            return self.access_logs
        
        self.access_logs = pd.read_csv(filepath)
        self.access_logs['timestamp'] = pd.to_datetime(self.access_logs['timestamp'])
        
        logger.info(f"Loaded {len(self.access_logs)} access log entries")
        return self.access_logs
    
    def load_agent_profiles(self, filename: str = "agent_profiles.json") -> Dict:
        """
        Load agent behavioral profiles.
        
        Returns:
            Dictionary with agent profiles
        """
        filepath = self.dataset_path / filename
        
        if not filepath.exists():
            logger.warning(f"{filename} not found. Creating sample profiles...")
            self.agent_profiles = self._create_sample_agent_profiles()
            return self.agent_profiles
        
        with open(filepath, 'r') as f:
            self.agent_profiles = json.load(f)
        
        logger.info(f"Loaded profiles for {len(self.agent_profiles)} agents")
        return self.agent_profiles
    
    def load_normal_patterns(self, filename: str = "normal_patterns.csv") -> pd.DataFrame:
        """
        Load normal behavior patterns for training.
        
        Returns:
            DataFrame with normal patterns
        """
        filepath = self.dataset_path / filename
        
        if not filepath.exists():
            logger.warning(f"{filename} not found. Using access_logs as normal patterns...")
            # Use access logs with decision='grant' as normal patterns
            if self.access_logs is None:
                self.load_access_logs()
            self.normal_patterns = self.access_logs[
                self.access_logs['decision'] == 'grant'
            ].copy()
            return self.normal_patterns
        
        self.normal_patterns = pd.read_csv(filepath)
        self.normal_patterns['timestamp'] = pd.to_datetime(self.normal_patterns['timestamp'])
        
        logger.info(f"Loaded {len(self.normal_patterns)} normal pattern entries")
        return self.normal_patterns
    
    def load_anomalies(self, filename: str = "anomalies.csv") -> pd.DataFrame:
        """
        Load known anomaly examples.
        
        Returns:
            DataFrame with anomalies
        """
        filepath = self.dataset_path / filename
        
        if not filepath.exists():
            logger.warning(f"{filename} not found. Using denied requests as anomalies...")
            if self.access_logs is None:
                self.load_access_logs()
            self.anomalies = self.access_logs[
                self.access_logs['decision'] == 'deny'
            ].copy()
            return self.anomalies
        
        self.anomalies = pd.read_csv(filepath)
        self.anomalies['timestamp'] = pd.to_datetime(self.anomalies['timestamp'])
        
        logger.info(f"Loaded {len(self.anomalies)} anomaly entries")
        return self.anomalies
    
    def _create_sample_access_logs(self) -> pd.DataFrame:
        """
        Create sample access logs for testing.
        
        Returns:
            DataFrame with sample data
        """
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'timestamp': pd.date_range(start='2024-01-01', periods=n_samples, freq='5min'),
            'agent_id': np.random.choice(
                ['robot_001', 'robot_002', 'human_001', 'human_002'], 
                n_samples
            ),
            'agent_type': np.random.choice(['robot', 'human'], n_samples, p=[0.7, 0.3]),
            'resource_id': np.random.choice(
                ['assembly_station_A', 'assembly_station_B', 'material_storage', 'quality_db'],
                n_samples
            ),
            'action': np.random.choice(['read', 'write', 'execute'], n_samples, p=[0.5, 0.3, 0.2]),
            'decision': np.random.choice(['grant', 'deny'], n_samples, p=[0.9, 0.1]),
            'zone': np.random.choice(['production', 'quality_control', 'storage'], n_samples),
            'time_of_day': [pd.Timestamp(ts).hour for ts in pd.date_range(
                start='2024-01-01', periods=n_samples, freq='5min'
            )]
        }
        
        df = pd.DataFrame(data)
        
        # Save to file
        output_path = self.dataset_path / "sample_access_logs.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Created sample access logs: {output_path}")
        
        return df
    
    def _create_sample_agent_profiles(self) -> Dict:
        """
        Create sample agent profiles for testing.
        
        Returns:
            Dictionary with sample profiles
        """
        profiles = {
            "robot_001": {
                "type": "assembly_robot",
                "normal_behavior": {
                    "access_frequency": 120,
                    "typical_resources": ["assembly_station_A", "material_storage"],
                    "typical_actions": ["read", "write"],
                    "active_hours": [8, 9, 10, 11, 13, 14, 15, 16, 17]
                }
            },
            "robot_002": {
                "type": "camera_robot",
                "normal_behavior": {
                    "access_frequency": 240,
                    "typical_resources": ["quality_db", "inspection_station"],
                    "typical_actions": ["read", "write"],
                    "active_hours": list(range(6, 22))
                }
            },
            "human_001": {
                "type": "operator",
                "normal_behavior": {
                    "access_frequency": 80,
                    "typical_resources": ["assembly_station_A", "assembly_station_B"],
                    "typical_actions": ["read", "write", "monitor"],
                    "active_hours": [8, 9, 10, 11, 12, 13, 14, 15, 16]
                }
            },
            "human_002": {
                "type": "supervisor",
                "normal_behavior": {
                    "access_frequency": 50,
                    "typical_resources": ["*"],
                    "typical_actions": ["read", "write", "override"],
                    "active_hours": [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
                }
            }
        }
        
        # Save to file
        output_path = self.dataset_path / "sample_agent_profiles.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(profiles, f, indent=2)
        logger.info(f"Created sample agent profiles: {output_path}")
        
        return profiles
    
    def get_agent_features(self, agent_id: str) -> np.ndarray:
        """
        Extract features for a specific agent.
        
        Args:
            agent_id: Agent identifier
        
        Returns:
            Feature vector for the agent
        """
        if self.access_logs is None:
            self.load_access_logs()
        
        agent_logs = self.access_logs[self.access_logs['agent_id'] == agent_id]
        
        if len(agent_logs) == 0:
            return np.zeros(10)  # Return zero vector if no data
        
        # Extract simple features
        features = [
            len(agent_logs),  # Total requests
            (agent_logs['decision'] == 'grant').mean(),  # Grant rate
            (agent_logs['action'] == 'read').mean(),  # Read action rate
            (agent_logs['action'] == 'write').mean(),  # Write action rate
            (agent_logs['action'] == 'execute').mean(),  # Execute action rate
            agent_logs['time_of_day'].mean(),  # Average time of day
            agent_logs['time_of_day'].std(),  # Time variance
            agent_logs.groupby('resource_id').size().max() if len(agent_logs) > 0 else 0,  # Max resource usage
            len(agent_logs['resource_id'].unique()),  # Unique resources
            (agent_logs['agent_type'] == 'robot').iloc[0] if len(agent_logs) > 0 else 0  # Is robot
        ]
        
        return np.array(features)
    
    def get_statistics(self) -> Dict:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with statistics
        """
        if self.access_logs is None:
            self.load_access_logs()
        
        stats = {
            'total_logs': len(self.access_logs),
            'unique_agents': self.access_logs['agent_id'].nunique(),
            'unique_resources': self.access_logs['resource_id'].nunique(),
            'grant_rate': (self.access_logs['decision'] == 'grant').mean(),
            'robot_ratio': (self.access_logs['agent_type'] == 'robot').mean(),
            'action_distribution': self.access_logs['action'].value_counts().to_dict()
        }
        
        return stats


if __name__ == "__main__":
    # Test the dataset loader
    loader = DatasetLoader()
    loader.load_all()
    
    print("\nDataset Statistics:")
    stats = loader.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nSample Access Logs:")
    print(loader.access_logs.head())
