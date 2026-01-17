"""
BBAC Framework - Model Training Script

Trains Layer 2 (Behavioral) and Layer 3 (ML) using real data from bbac_ics_dataset.
Models are saved to models/ directory for use by the BBAC Controller.
"""

# 1. Biblioteca padrão
import logging
import os
from pathlib import Path

# 2. Bibliotecas de terceiros
import pandas as pd

# 3. Imports locais
from core.behavioral_analysis import BehavioralAnalyzer
from core.ml_detection import MLAnomalyDetector
from data.dataset_loader import DatasetLoader

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Trains BBAC models with real dataset.
    """
    
    def __init__(self, data_path: str = "data/100k", models_path: str = "models"):
        """
        Initialize model trainer.
        
        Args:
            data_path: Path to dataset
            models_path: Path to save trained models
        """
        self.data_path = data_path
        self.models_path = Path(models_path)
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        self.loader = None
        self.behavioral_analyzer = None
        self.ml_detector = None
    
    def load_data(self) -> bool:
        """Load dataset."""
        try:
            logger.info("="*70)
            logger.info("LOADING DATASET")
            logger.info("="*70)
            
            self.loader = DatasetLoader(self.data_path)
            self.loader.load_all()
            
            logger.info(f"✓ Loaded {len(self.loader.train_data)} training samples")
            logger.info(f"✓ Loaded {len(self.loader.val_data)} validation samples")
            logger.info(f"✓ Loaded {len(self.loader.test_data)} test samples")
            logger.info(f"✓ Loaded {len(self.loader.agents)} agent profiles")
            
            return True
            
        except FileNotFoundError as e:
            logger.error(str(e))
            return False
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare dataset for training.
        
        Maps dataset columns to expected format:
        - user_id → agent_id
        - resource → resource_id
        - location → zone
        
        Args:
            df: Raw dataset
            
        Returns:
            Prepared DataFrame
        """
        prepared = df.copy()
        
        # Rename columns to match expected format
        prepared = prepared.rename(columns={
            'user_id': 'agent_id',
            'resource': 'resource_id',
            'location': 'zone'
        })
        
        # Ensure required columns exist
        if 'agent_id' not in prepared.columns:
            raise ValueError("Dataset missing 'user_id' column")
        if 'resource_id' not in prepared.columns:
            raise ValueError("Dataset missing 'resource' column")
        if 'action' not in prepared.columns:
            raise ValueError("Dataset missing 'action' column")
        
        return prepared
    
    def train_behavioral_layer(self) -> bool:
        """
        Train Layer 2: Behavioral Analyzer.
        
        Returns:
            True if successful
        """
        try:
            logger.info("")
            logger.info("="*70)
            logger.info("TRAINING LAYER 2: BEHAVIORAL ANALYZER")
            logger.info("="*70)
            
            # Initialize analyzer
            self.behavioral_analyzer = BehavioralAnalyzer(
                order=2,
                smoothing='laplace',
                min_samples=50
            )
            
            # Prepare data
            train_data = self.prepare_data(self.loader.train_data)
            
            # Get unique agents
            unique_agents = train_data['agent_id'].unique()
            logger.info(f"Training on {len(unique_agents)} unique agents...")
            
            # Train per agent
            trained_count = 0
            for agent_id in unique_agents:
                agent_data = train_data[train_data['agent_id'] == agent_id]
                
                if len(agent_data) >= 50:  # Minimum samples
                    success = self.behavioral_analyzer.train(agent_data, agent_id=agent_id)
                    if success:
                        trained_count += 1
                        if trained_count % 10 == 0:
                            logger.info(f"  Trained {trained_count}/{len(unique_agents)} agents...")
            
            logger.info(f"✓ Trained behavioral models for {trained_count} agents")
            
            # Save models
            models_file = self.models_path / "behavioral_models.json"
            self.behavioral_analyzer.save_models(str(models_file))
            logger.info(f"✓ Saved models to: {models_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training behavioral layer: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def train_ml_layer(self) -> bool:
        """
        Train Layer 3: ML Anomaly Detector.
        
        Returns:
            True if successful
        """
        try:
            logger.info("")
            logger.info("="*70)
            logger.info("TRAINING LAYER 3: ML ANOMALY DETECTOR")
            logger.info("="*70)
            
            # Initialize detector
            self.ml_detector = MLAnomalyDetector(
                contamination=0.1,
                n_estimators=100,
                anomaly_threshold=0.7
            )
            
            # Prepare data
            train_data = self.prepare_data(self.loader.train_data)
            
            # Train models per agent type
            agent_types = train_data['agent_type'].unique()
            logger.info(f"Training on {len(agent_types)} agent types: {list(agent_types)}")
            
            # Train
            success = self.ml_detector.train(train_data)
            
            if success:
                logger.info(f"✓ Trained ML models for {len(self.ml_detector.models)} agent types")
                
                # Print model info
                for agent_type, stats in self.ml_detector.training_stats.items():
                    logger.info(f"  {agent_type}: {stats['num_samples']} samples, "
                              f"mean score: {stats['mean_score']:.3f}")
                
                # Save models
                models_file = self.models_path / "ml_models.pkl"
                self.ml_detector.save_models(str(models_file))
                logger.info(f"✓ Saved models to: {models_file}")
                
                return True
            else:
                logger.error("Failed to train ML models")
                return False
            
        except Exception as e:
            logger.error(f"Error training ML layer: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def validate_models(self) -> bool:
        """
        Validate trained models on validation set.
        
        Returns:
            True if validation successful
        """
        try:
            logger.info("")
            logger.info("="*70)
            logger.info("VALIDATING MODELS")
            logger.info("="*70)
            
            val_data = self.prepare_data(self.loader.val_data)
            
            # Sample some validation requests
            sample_size = min(100, len(val_data))
            val_sample = val_data.sample(n=sample_size, random_state=42)
            
            behavioral_decisions = 0
            ml_decisions = 0
            
            logger.info(f"Testing on {sample_size} validation samples...")
            
            for _, row in val_sample.iterrows():
                request = {
                    'agent_id': row['agent_id'],
                    'resource_id': row['resource_id'],
                    'action': row['action'],
                    'timestamp': row['timestamp'] if 'timestamp' in row else None,
                    'agent_type': row['agent_type'],
                    'context': {'zone': row.get('zone', 'unknown')}
                }
                
                # Test behavioral
                try:
                    decision_b, _, _ = self.behavioral_analyzer.evaluate_access_request(request)
                    if decision_b in ['grant', 'deny']:
                        behavioral_decisions += 1
                except:
                    pass
                
                # Test ML
                try:
                    decision_ml, _, _ = self.ml_detector.evaluate_access_request(request)
                    if decision_ml in ['grant', 'deny']:
                        ml_decisions += 1
                except:
                    pass
            
            behavioral_rate = (behavioral_decisions / sample_size) * 100
            ml_rate = (ml_decisions / sample_size) * 100
            
            logger.info(f"Behavioral Layer: {behavioral_decisions}/{sample_size} decisions ({behavioral_rate:.1f}%)")
            logger.info(f"ML Layer: {ml_decisions}/{sample_size} decisions ({ml_rate:.1f}%)")
            
            if behavioral_rate > 50 and ml_rate > 50:
                logger.info("✓ Models validated successfully")
                return True
            else:
                logger.warning("⚠ Low decision rate - models may need review")
                return False
            
        except Exception as e:
            logger.error(f"Error during validation: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run(self) -> bool:
        """
        Run complete training pipeline.
        
        Returns:
            True if all steps successful
        """
        print("\n" + "="*70)
        print("BBAC FRAMEWORK - MODEL TRAINING")
        print("="*70)
        print(f"Data path: {self.data_path}")
        print(f"Models path: {self.models_path}")
        print("="*70)
        
        # Step 1: Load data
        if not self.load_data():
            logger.error("Failed to load dataset")
            return False
        
        # Step 2: Train behavioral layer
        if not self.train_behavioral_layer():
            logger.error("Failed to train behavioral layer")
            return False
        
        # Step 3: Train ML layer
        if not self.train_ml_layer():
            logger.error("Failed to train ML layer")
            return False
        
        # Step 4: Validate
        self.validate_models()
        
        # Summary
        print("\n" + "="*70)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"Models saved to: {self.models_path}")
        print(f"  - behavioral_models.json")
        print(f"  - ml_models.pkl")
        print("\nYou can now run experiments using main.py")
        print("="*70 + "\n")
        
        return True


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train BBAC models with real data')
    parser.add_argument('--data', type=str, default='data/data_100k',
                       help='Path to dataset directory')
    parser.add_argument('--output', type=str, default='models',
                       help='Path to save trained models')
    
    args = parser.parse_args()
    
    trainer = ModelTrainer(data_path=args.data, models_path=args.output)
    success = trainer.run()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
