"""
BBAC Framework - Layer 3: ML-based Anomaly Detection

This module implements machine learning-based anomaly detection using:
- Isolation Forest algorithm
- Feature extraction from access patterns
- Adaptive learning and model updates
- Anomaly scoring and threshold-based decisions
"""

# 1. Biblioteca padrão
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# 2. Bibliotecas de terceiros
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLAnomalyDetector:
    """
    Layer 3: ML-based Anomaly Detection using Isolation Forest
    
    Detects anomalous access patterns using:
    - Isolation Forest algorithm
    - Feature engineering from access requests
    - Continuous learning and adaptation
    - Probabilistic anomaly scores
    """
    
    def __init__(self, contamination: float = 0.1, n_estimators: int = 100,
                 max_samples: int = 256, random_state: int = 42,
                 anomaly_threshold: float = 0.7):
        """
        Initialize the ML Anomaly Detector.
        
        Args:
            contamination: Expected proportion of outliers (0.0 to 0.5)
            n_estimators: Number of trees in the forest
            max_samples: Number of samples to draw to train each estimator
            random_state: Random seed for reproducibility
            anomaly_threshold: Threshold for classifying as anomaly
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.anomaly_threshold = anomaly_threshold
        
        # Models per agent type (robots have more predictable patterns)
        self.models = {}  # {agent_type: IsolationForest}
        self.scalers = {}  # {agent_type: StandardScaler}
        self.feature_names = []
        
        # Training statistics
        self.training_stats = {}
        
        logger.info(f"MLAnomalyDetector initialized (contamination={contamination}, "
                   f"n_estimators={n_estimators})")
    
    def train(self, training_data: pd.DataFrame, agent_type: Optional[str] = None) -> bool:
        """
        Train anomaly detection models on normal behavior data.
        
        Args:
            training_data: DataFrame with access logs
            agent_type: If specified, train only for this agent type
        
        Returns:
            True if training successful
        """
        try:
            if agent_type:
                type_data = training_data[training_data['agent_type'] == agent_type]
                self._train_model_for_type(agent_type, type_data)
            else:
                # Train models for each agent type
                for atype in training_data['agent_type'].unique():
                    type_data = training_data[training_data['agent_type'] == atype]
                    if len(type_data) >= 50:  # Minimum samples
                        self._train_model_for_type(atype, type_data)
                    else:
                        logger.warning(f"Insufficient data for agent type {atype}")
            
            logger.info(f"Trained models for {len(self.models)} agent types")
            return True
        
        except Exception as e:
            logger.error(f"Error during training: {e}")
            return False
    
    def _train_model_for_type(self, agent_type: str, data: pd.DataFrame):
        """
        Train an Isolation Forest model for a specific agent type.
        
        Args:
            agent_type: Type of agent (robot/human)
            data: Historical access data for this type
        """
        # Extract features
        features = self._extract_features(data)
        
        # Store feature names
        if not self.feature_names:
            self.feature_names = list(features.columns)
        
        # Initialize and fit scaler
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Initialize and fit Isolation Forest
        model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            max_samples=min(self.max_samples, len(features)),
            random_state=self.random_state,
            n_jobs=-1
        )
        model.fit(features_scaled)
        
        # Store model and scaler
        self.models[agent_type] = model
        self.scalers[agent_type] = scaler
        
        # Calculate training statistics
        scores = model.score_samples(features_scaled)
        self.training_stats[agent_type] = {
            'num_samples': len(data),
            'mean_score': float(np.mean(scores)),
            'std_score': float(np.std(scores)),
            'min_score': float(np.min(scores)),
            'max_score': float(np.max(scores))
        }
        
        logger.info(f"Trained model for {agent_type}: {len(data)} samples, "
                   f"mean score: {self.training_stats[agent_type]['mean_score']:.3f}")
    
    def _extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from access request data.
        
        Features include:
        - Time-based: hour, day of week, weekend flag
        - Frequency-based: access rate, action distribution
        - Sequence-based: state transitions, pattern consistency
        - Context-based: zone, resource type
        
        Args:
            data: DataFrame with access logs
        
        Returns:
            DataFrame with extracted features
        """
        features = pd.DataFrame()
        
        # Ensure timestamp is datetime
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            # Time-based features
            features['hour'] = data['timestamp'].dt.hour
            features['day_of_week'] = data['timestamp'].dt.dayofweek
            features['is_weekend'] = (data['timestamp'].dt.dayofweek >= 5).astype(int)
            features['is_work_hours'] = ((data['timestamp'].dt.hour >= 8) & 
                                        (data['timestamp'].dt.hour <= 17)).astype(int)
        else:
            # Default values if no timestamp
            features['hour'] = 12
            features['day_of_week'] = 0
            features['is_weekend'] = 0
            features['is_work_hours'] = 1
        
        # Action encoding
        action_map = {'read': 0, 'write': 1, 'execute': 2, 'delete': 3, 'transport': 4}
        features['action_code'] = data['action'].map(action_map).fillna(-1)
        
        # Resource encoding (hash-based for unknown resources)
        features['resource_hash'] = data['resource_id'].apply(lambda x: hash(x) % 1000)
        
        # Zone encoding
        if 'zone' in data.columns:
            zone_map = {'production': 0, 'quality_control': 1, 'storage': 2, 'maintenance': 3}
            features['zone_code'] = data['zone'].map(zone_map).fillna(-1)
        else:
            features['zone_code'] = 0
        
        # Decision encoding (for training data)
        if 'decision' in data.columns:
            features['decision_code'] = (data['decision'] == 'grant').astype(int)
        
        # Agent-specific features (for multiple agents)
        if len(data['agent_id'].unique()) > 1:
            features['agent_hash'] = data['agent_id'].apply(lambda x: hash(x) % 100)
        else:
            features['agent_hash'] = 0
        
        # Statistical features (if we have enough data)
        if len(data) > 10:
            # Rolling window features
            features['recent_access_count'] = range(len(data))
            features['recent_access_count'] = features['recent_access_count'].rolling(
                window=min(10, len(data)), min_periods=1
            ).count()
        
        return features
    
    def evaluate_access_request(self, request: Dict, 
                                context_data: Optional[pd.DataFrame] = None) -> Tuple[str, float, Dict]:
        """
        Evaluate access request using ML anomaly detection.
        
        Args:
            request: Dictionary containing access request details
            context_data: Optional recent access history for feature extraction
        
        Returns:
            Tuple of (decision, confidence, explanation)
        """
        agent_type = request.get('agent_type', 'unknown')
        
        explanation = {
            'layer': 'ml_detection',
            'agent_type': agent_type,
            'has_model': agent_type in self.models,
            'anomaly_score': 0.0,
            'normalized_score': 0.0,
            'decision_reason': None
        }
        
        # Check if we have a model for this agent type
        if agent_type not in self.models:
            explanation['decision_reason'] = 'no_ml_model'
            return ('uncertain', 0.5, explanation)
        
        # Extract features from request
        request_df = self._request_to_dataframe(request, context_data)
        features = self._extract_features(request_df)
        
        # Scale features
        scaler = self.scalers[agent_type]
        features_scaled = scaler.transform(features)
        
        # Get anomaly score
        model = self.models[agent_type]
        anomaly_score = model.score_samples(features_scaled)[0]
        
        # Normalize score to [0, 1] range
        # Lower score = more anomalous
        # Convert to probability: higher = more anomalous
        stats = self.training_stats[agent_type]
        normalized_score = self._normalize_score(anomaly_score, stats)
        
        explanation['anomaly_score'] = float(anomaly_score)
        explanation['normalized_score'] = float(normalized_score)
        
        # Decision logic
        if normalized_score < 0.3:
            # Low anomaly score - normal behavior
            decision = 'grant'
            confidence = 1.0 - normalized_score
            explanation['decision_reason'] = 'ml_normal_pattern'
        
        elif normalized_score < self.anomaly_threshold:
            # Moderate anomaly score - uncertain
            decision = 'uncertain'
            confidence = 0.5
            explanation['decision_reason'] = 'ml_moderately_unusual'
        
        else:
            # High anomaly score - anomalous
            decision = 'deny'
            confidence = normalized_score
            explanation['decision_reason'] = 'ml_anomaly_detected'
        
        return (decision, confidence, explanation)
    
    def _request_to_dataframe(self, request: Dict, 
                             context_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Convert a single request to DataFrame format.
        
        Args:
            request: Request dictionary
            context_data: Optional context data
        
        Returns:
            DataFrame with request data
        """
        df_dict = {
            'agent_id': [request.get('agent_id', 'unknown')],
            'agent_type': [request.get('agent_type', 'unknown')],
            'resource_id': [request.get('resource_id', 'unknown')],
            'action': [request.get('action', 'read')],
            'timestamp': [request.get('timestamp', datetime.now())]
        }
        
        # Add context fields if available
        context = request.get('context', {})
        df_dict['zone'] = [context.get('zone', 'production')]
        
        return pd.DataFrame(df_dict)
    
    def _normalize_score(self, score: float, stats: Dict) -> float:
        """
        Normalize anomaly score to [0, 1] range.
        
        Lower raw score = more anomalous
        Normalized score: 0 = normal, 1 = highly anomalous
        
        Args:
            score: Raw anomaly score from Isolation Forest
            stats: Training statistics
        
        Returns:
            Normalized score (0 to 1)
        """
        mean = stats['mean_score']
        std = stats['std_score']
        
        # Z-score normalization
        if std > 0:
            z_score = (score - mean) / std
        else:
            z_score = 0
        
        # Convert to probability (lower z-score = more anomalous)
        # Use sigmoid function
        normalized = 1.0 / (1.0 + np.exp(z_score))
        
        # Clip to [0, 1]
        return np.clip(normalized, 0.0, 1.0)
    
    def update_model(self, new_data: pd.DataFrame, agent_type: str) -> bool:
        """
        Update model with new data (online learning simulation).
        
        Args:
            new_data: New access log data
            agent_type: Agent type to update
        
        Returns:
            True if successful
        """
        # For Isolation Forest, we retrain with combined data
        # In production, consider incremental learning algorithms
        
        if agent_type not in self.models:
            logger.warning(f"No existing model for {agent_type}, training new model")
            return self._train_model_for_type(agent_type, new_data)
        
        # This is a simplified update - in practice, maintain a sliding window
        logger.info(f"Model update for {agent_type} with {len(new_data)} new samples")
        return self._train_model_for_type(agent_type, new_data)
    
    def save_models(self, filepath: str) -> bool:
        """
        Save trained models to disk.
        
        Args:
            filepath: Base filepath (will add .pkl extension)
        
        Returns:
            True if successful
        """
        try:
            model_data = {
                'models': self.models,
                'scalers': self.scalers,
                'feature_names': self.feature_names,
                'training_stats': self.training_stats,
                'config': {
                    'contamination': self.contamination,
                    'n_estimators': self.n_estimators,
                    'anomaly_threshold': self.anomaly_threshold
                }
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"Saved models to {filepath}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            return False
    
    def load_models(self, filepath: str) -> bool:
        """
        Load trained models from disk.
        
        Args:
            filepath: Path to model file
        
        Returns:
            True if successful
        """
        try:
            model_data = joblib.load(filepath)
            
            self.models = model_data['models']
            self.scalers = model_data['scalers']
            self.feature_names = model_data['feature_names']
            self.training_stats = model_data['training_stats']
            
            config = model_data['config']
            self.contamination = config['contamination']
            self.n_estimators = config['n_estimators']
            self.anomaly_threshold = config['anomaly_threshold']
            
            logger.info(f"Loaded models from {filepath}")
            return True
        
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def get_model_info(self) -> Dict:
        """
        Get information about trained models.
        
        Returns:
            Dictionary with model information
        """
        return {
            'agent_types': list(self.models.keys()),
            'num_models': len(self.models),
            'feature_names': self.feature_names,
            'training_stats': self.training_stats,
            'config': {
                'contamination': self.contamination,
                'n_estimators': self.n_estimators,
                'anomaly_threshold': self.anomaly_threshold
            }
        }


if __name__ == "__main__":
    print("MLAnomalyDetector - Layer 3")
    print("This module requires training data from bbac_ics_dataset.")
    print("Use train_models.py to train the models with real data.")
