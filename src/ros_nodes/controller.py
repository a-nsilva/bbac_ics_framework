"""
BBAC Framework - Main Controller Node

Integrates all three decision layers:
- Layer 1: Rule Engine
- Layer 2: Behavioral Analysis (Markov)
- Layer 3: ML Anomaly Detection (Isolation Forest)

ROS2 Humble compatible
"""

# 1. Biblioteca padrão
import json
import os
import sys 
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

# 2. Bibliotecas de terceiros
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# 3. Imports locais
from core.behavioral_analysis import BehavioralAnalyzer
from core.ml_detection import MLAnomalyDetector
from core.rule_engine import RuleEngine

class BBACController(Node):
    """
    Main BBAC Controller Node - Hybrid Decision System
    
    Integrates three layers of access control:
    1. Rule-based (deterministic policies)
    2. Behavioral (Markov chain analysis)
    3. ML-based (anomaly detection)
    
    Decision fusion strategy: Weighted voting with configurable thresholds
    """
    
    def __init__(self):
        super().__init__('bbac_controller')
        
        self.get_logger().info('Initializing BBAC Controller...')
        
        # Initialize the three layers
        self._initialize_layers()
        
        # ROS2 Publishers and Subscribers
        self.decision_publisher = self.create_publisher(
            String, 
            '/access_decisions', 
            10
        )
        
        self.request_subscriber = self.create_subscription(
            String,
            '/access_requests',
            self.handle_access_request,
            10
        )
        
        self.emergency_publisher = self.create_publisher(
            String,
            '/emergency_alerts',
            10
        )
        
        # Statistics
        self.requests_processed = 0
        self.decisions_granted = 0
        self.decisions_denied = 0
        self.total_latency_ms = 0.0
        
        # Performance timer
        self.stats_timer = self.create_timer(10.0, self.publish_statistics)
        
        self.get_logger().info('BBAC Controller initialized successfully')
        self.get_logger().info('Listening on /access_requests...')
    
    def _check_and_train_models(self):
        """
        Check if trained models exist. If not, train them automatically.
        
        Returns:
            Tuple of (behavioral_model_path, ml_model_path) or (None, None)
        """
        models_dir = Path('models')
        behavioral_path = models_dir / 'behavioral_models.json'
        ml_path = models_dir / 'ml_models.pkl'
        
        # Check if models exist
        if behavioral_path.exists() and ml_path.exists():
            self.get_logger().info('Found existing trained models')
            return str(behavioral_path), str(ml_path)
        
        # Models don't exist - need to train
        self.get_logger().warn('Trained models not found. Auto-training...')
        self.get_logger().warn('This is a one-time operation and may take a few minutes.')
        
        try:
            # Import trainer
            #sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            from core.train_models import ModelTrainer
            
            # Train models
            self.get_logger().info('Starting model training with dataset...')
            trainer = ModelTrainer(data_path='data/data_100k', models_path='models')
            
            # Run training
            success = trainer.run()
            
            if success:
                self.get_logger().info('Model training completed successfully')
                return str(behavioral_path), str(ml_path)
            else:
                self.get_logger().error('Model training failed')
                return None, None
                
        except FileNotFoundError as e:
            self.get_logger().error(f'Dataset not found: {e}')
            self.get_logger().error('Please ensure bbac_ics_dataset has been generated first.')
            return None, None
        except Exception as e:
            self.get_logger().error(f'Error during auto-training: {e}')
            import traceback
            traceback.print_exc()
            return None, None
    
    def _initialize_layers(self):
        """Initialize all three decision layers."""
        try:
            # Layer 1: Rule Engine (always uses config files)
            self.get_logger().info('Loading Layer 1: Rule Engine...')
            self.rule_engine = RuleEngine(
                policies_path='config/policies.json',
                emergency_rules_path='config/emergency_rules.json'
            )
            
            # Check and train models if needed
            behavioral_model, ml_model = self._check_and_train_models()
            
            # Layer 2: Behavioral Analyzer
            self.get_logger().info('Loading Layer 2: Behavioral Analyzer...')
            self.behavioral_analyzer = BehavioralAnalyzer(
                order=2,
                smoothing='laplace',
                min_samples=50
            )
            
            # Load trained model if available
            if behavioral_model:
                success = self.behavioral_analyzer.load_models(behavioral_model)
                if success:
                    self.get_logger().info(f'Loaded behavioral models from {behavioral_model}')
                else:
                    self.get_logger().warn('Failed to load behavioral models, using untrained model')
            else:
                self.get_logger().warn('No behavioral models available, using untrained model')
            
            # Layer 3: ML Anomaly Detector
            self.get_logger().info('Loading Layer 3: ML Anomaly Detector...')
            self.ml_detector = MLAnomalyDetector(
                contamination=0.1,
                n_estimators=100,
                anomaly_threshold=0.7
            )
            
            # Load trained model if available
            if ml_model:
                success = self.ml_detector.load_models(ml_model)
                if success:
                    self.get_logger().info(f'Loaded ML models from {ml_model}')
                else:
                    self.get_logger().warn('Failed to load ML models, using untrained model')
            else:
                self.get_logger().warn('No ML models available, using untrained model')
            
            self.get_logger().info('All layers initialized successfully')
            
        except Exception as e:
            self.get_logger().error(f'Error initializing layers: {e}')
            raise
    
    def handle_access_request(self, msg: String):
        """
        Handle incoming access request.
        
        Args:
            msg: ROS2 String message containing JSON access request
        """
        start_time = time.time()
        
        try:
            # Parse request
            request = json.loads(msg.data)
            request_id = request.get('request_id', 'unknown')
            agent_id = request.get('agent_id', 'unknown')
            action = request.get('action', 'unknown')
            resource = request.get('resource_id', 'unknown')
            
            self.get_logger().info(
                f'Request {request_id}: {agent_id} -> {action} on {resource}'
            )
            
            # Add timestamp if not present
            if 'timestamp' not in request:
                request['timestamp'] = datetime.now()
            
            # Make hybrid decision
            decision, confidence, explanation = self._make_hybrid_decision(request)
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Prepare decision message
            decision_msg = {
                'request_id': request_id,
                'decision': decision,
                'confidence': confidence,
                'latency_ms': round(latency_ms, 2),
                'timestamp': datetime.now().isoformat(),
                'explanation': explanation
            }
            
            # Publish decision
            response = String()
            response.data = json.dumps(decision_msg)
            self.decision_publisher.publish(response)
            
            # Update statistics
            self._update_statistics(decision, latency_ms)
            
            # Log decision
            status = "GRANTED" if decision == 'grant' else "DENIED"
            self.get_logger().info(
                f'{status} (confidence: {confidence:.2f}, '
                f'latency: {latency_ms:.2f}ms)'
            )
            
        except Exception as e:
            self.get_logger().error(f'Error handling request: {e}')
    
    def _make_hybrid_decision(self, request: Dict) -> Tuple[str, float, Dict]:
        """
        Make hybrid access control decision using all three layers.
        
        Decision fusion strategy:
        - If any layer has high confidence denial -> DENY
        - If all layers agree on grant -> GRANT
        - Otherwise -> Use weighted voting
        
        Args:
            request: Access request dictionary
        
        Returns:
            Tuple of (decision, confidence, explanation)
        """
        # Evaluate each layer
        l1_decision, l1_confidence, l1_explanation = self.rule_engine.evaluate_access_request(request)
        l2_decision, l2_confidence, l2_explanation = self.behavioral_analyzer.evaluate_access_request(request)
        l3_decision, l3_confidence, l3_explanation = self.ml_detector.evaluate_access_request(request)
        
        # Collect layer results
        layer_results = {
            'layer1_rules': {
                'decision': l1_decision,
                'confidence': l1_confidence,
                'explanation': l1_explanation
            },
            'layer2_behavioral': {
                'decision': l2_decision,
                'confidence': l2_confidence,
                'explanation': l2_explanation
            },
            'layer3_ml': {
                'decision': l3_decision,
                'confidence': l3_confidence,
                'explanation': l3_explanation
            }
        }
        
        # Decision fusion logic
        
        # Priority 1: Rule engine has absolute priority for explicit decisions
        #if l1_decision in ['grant', 'deny'] and l1_confidence == 1.0:
        #    return l1_decision, l1_confidence, {
        #        'fusion_strategy': 'rule_priority',
        #        'layers': layer_results,
        #        'reason': l1_explanation.get('decision_reason', 'rule_based')
        #    }
        if l1_decision in ['grant', 'deny', 'require_approval'] and l1_confidence == 1.0:
            # Tratar require_approval como deny com flag especial
            if l1_decision == 'require_approval':
                return 'deny', 0.8, {
                    'fusion_strategy': 'require_approval',
                    'layers': layer_results,
                    'reason': 'manual_approval_required'
                }
        
        # Priority 2: If any layer strongly denies (confidence > 0.8) -> DENY
        if (l1_decision == 'deny' and l1_confidence > 0.8) or \
           (l2_decision == 'deny' and l2_confidence > 0.8) or \
           (l3_decision == 'deny' and l3_confidence > 0.8):
            
            denying_layers = []
            if l1_decision == 'deny' and l1_confidence > 0.8:
                denying_layers.append('rules')
            if l2_decision == 'deny' and l2_confidence > 0.8:
                denying_layers.append('behavioral')
            if l3_decision == 'deny' and l3_confidence > 0.8:
                denying_layers.append('ml')
            
            return 'deny', max(l1_confidence, l2_confidence, l3_confidence), {
                'fusion_strategy': 'high_confidence_denial',
                'layers': layer_results,
                'denying_layers': denying_layers
            }
        
        # Priority 3: Weighted voting (all layers participate)
        # Weights: Rules=0.4, Behavioral=0.3, ML=0.3
        weights = {'grant': 0.0, 'deny': 0.0, 'uncertain': 0.0}
        
        # Layer 1 vote (weight: 0.4)
        if l1_decision != 'uncertain':
            weights[l1_decision] += 0.4 * l1_confidence
        
        # Layer 2 vote (weight: 0.3)
        if l2_decision != 'uncertain':
            weights[l2_decision] += 0.3 * l2_confidence
        else:
            weights['grant'] += 0.15  # Neutral uncertainty splits vote
            weights['deny'] += 0.15
        
        # Layer 3 vote (weight: 0.3)
        if l3_decision != 'uncertain':
            weights[l3_decision] += 0.3 * l3_confidence
        else:
            weights['grant'] += 0.15
            weights['deny'] += 0.15
        
        # Final decision based on weighted votes
        if weights['grant'] > weights['deny']:
            final_decision = 'grant'
            final_confidence = weights['grant'] / (weights['grant'] + weights['deny'])
        else:
            final_decision = 'deny'
            final_confidence = weights['deny'] / (weights['grant'] + weights['deny'])
        
        return final_decision, final_confidence, {
            'fusion_strategy': 'weighted_voting',
            'layers': layer_results,
            'votes': weights,
            'weights': {'rules': 0.4, 'behavioral': 0.3, 'ml': 0.3}
        }
    
    def _update_statistics(self, decision: str, latency_ms: float):
        """Update performance statistics."""
        self.requests_processed += 1
        self.total_latency_ms += latency_ms
        
        if decision == 'grant':
            self.decisions_granted += 1
        else:
            self.decisions_denied += 1
    
    def publish_statistics(self):
        """Publish periodic statistics."""
        if self.requests_processed > 0:
            avg_latency = self.total_latency_ms / self.requests_processed
            grant_rate = (self.decisions_granted / self.requests_processed) * 100
            
            self.get_logger().info(
                f'Stats: {self.requests_processed} requests, '
                f'Grant={self.decisions_granted} ({grant_rate:.1f}%), '
                f'Deny={self.decisions_denied}, '
                f'Avg Latency={avg_latency:.2f}ms'
            )
    
    def trigger_emergency(self, emergency_type: str, context: Dict = None):
        """
        Trigger an emergency state.
        
        Args:
            emergency_type: Type of emergency
            context: Additional context
        """
        # Trigger in rule engine
        self.rule_engine.trigger_emergency(emergency_type, context)
        
        # Publish emergency alert
        alert = {
            'emergency_type': emergency_type,
            'timestamp': datetime.now().isoformat(),
            'active': True,
            'context': context or {}
        }
        
        msg = String()
        msg.data = json.dumps(alert)
        self.emergency_publisher.publish(msg)
        
        self.get_logger().warn(f'EMERGENCY TRIGGERED: {emergency_type}')


def main(args=None):
    """Main function."""
    rclpy.init(args=args)
    
    try:
        controller = BBACController()
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {e}')
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
