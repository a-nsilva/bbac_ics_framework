"""
BBAC Framework - Layer 2: Behavioral Analysis using Markov Chains

This module implements behavioral pattern analysis using:
- Markov Chain models for sequence prediction
- Transition probability matrices
- Pattern learning and deviation detection
- Agent-specific behavioral profiles
"""

# Biblioteca padrão
import json
import logging
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional

# Bibliotecas de terceiros
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BehavioralAnalyzer:
    """
    Layer 2: Behavioral Analysis using Markov Chains
    
    Learns and analyzes agent behavior patterns using:
    - N-order Markov chains
    - Transition probability matrices
    - Sequence anomaly detection
    - Temporal pattern analysis
    """
    
    def __init__(self, order: int = 2, smoothing: str = 'laplace',
                 min_samples: int = 50, confidence_threshold: float = 0.6):
        """
        Initialize the Behavioral Analyzer.
        
        Args:
            order: Order of Markov chain (1 or 2)
            smoothing: Smoothing method ('laplace', 'none')
            min_samples: Minimum samples needed to build reliable model
            confidence_threshold: Threshold for confident predictions
        """
        self.order = order
        self.smoothing = smoothing
        self.min_samples = min_samples
        self.confidence_threshold = confidence_threshold
        
        # Agent-specific models
        self.agent_models = {}  # {agent_id: MarkovModel}
        self.agent_histories = defaultdict(deque)  # Recent history per agent
        self.max_history_length = 1000
        
        logger.info(f"BehavioralAnalyzer initialized (order={order}, smoothing={smoothing})")
    
    def train(self, training_data: pd.DataFrame, agent_id: Optional[str] = None) -> bool:
        """
        Train Markov models on historical access patterns.
        
        Args:
            training_data: DataFrame with columns: agent_id, resource_id, action, timestamp
            agent_id: If specified, train only for this agent
        
        Returns:
            True if training successful
        """
        try:
            if agent_id:
                agent_data = training_data[training_data['agent_id'] == agent_id]
                self._train_agent_model(agent_id, agent_data)
            else:
                # Train models for all agents
                for aid in training_data['agent_id'].unique():
                    agent_data = training_data[training_data['agent_id'] == aid]
                    if len(agent_data) >= self.min_samples:
                        self._train_agent_model(aid, agent_data)
                    else:
                        logger.warning(f"Insufficient data for agent {aid}: {len(agent_data)} samples")
            
            logger.info(f"Trained models for {len(self.agent_models)} agents")
            return True
        
        except Exception as e:
            logger.error(f"Error during training: {e}")
            return False
    
    def _train_agent_model(self, agent_id: str, data: pd.DataFrame):
        """
        Train a Markov model for a specific agent.
        
        Args:
            agent_id: Agent identifier
            data: Historical access data for this agent
        """
        # Sort by timestamp
        data = data.sort_values('timestamp')
        
        # Create state sequences (state = resource_id + action)
        states = [f"{row['resource_id']}:{row['action']}" 
                 for _, row in data.iterrows()]
        
        # Build transition matrix
        model = MarkovModel(order=self.order, smoothing=self.smoothing)
        model.fit(states)
        
        self.agent_models[agent_id] = model
        
        logger.info(f"Trained model for {agent_id}: {len(states)} sequences, "
                   f"{model.get_num_states()} unique states")
    
    def evaluate_access_request(self, request: Dict) -> Tuple[str, float, Dict]:
        """
        Evaluate access request based on behavioral patterns.
        
        Args:
            request: Dictionary containing:
                - agent_id: Agent identifier
                - resource_id: Target resource
                - action: Requested action
                - context: Additional context
        
        Returns:
            Tuple of (decision, confidence, explanation)
        """
        agent_id = request.get('agent_id')
        resource_id = request.get('resource_id')
        action = request.get('action')
        
        explanation = {
            'layer': 'behavioral_analysis',
            'agent_id': agent_id,
            'has_model': agent_id in self.agent_models,
            'behavioral_score': 0.0,
            'expected_probability': 0.0,
            'decision_reason': None
        }
        
        # Check if we have a model for this agent
        if agent_id not in self.agent_models:
            explanation['decision_reason'] = 'no_behavioral_model'
            # Return neutral decision - defer to other layers
            return ('uncertain', 0.5, explanation)
        
        model = self.agent_models[agent_id]
        current_state = f"{resource_id}:{action}"
        
        # Get recent history for this agent
        history = list(self.agent_histories[agent_id])
        
        if len(history) < self.order:
            explanation['decision_reason'] = 'insufficient_history'
            return ('uncertain', 0.5, explanation)
        
        # Get context (previous states)
        context = tuple(history[-self.order:])
        
        # Calculate probability of current state given context
        probability = model.get_transition_probability(context, current_state)
        
        explanation['expected_probability'] = probability
        explanation['context_states'] = context
        
        # Calculate behavioral score
        # High probability = normal behavior
        # Low probability = anomalous behavior
        behavioral_score = probability
        explanation['behavioral_score'] = behavioral_score
        
        # Decision logic
        if behavioral_score >= 0.7:
            # High probability - consistent with normal behavior
            decision = 'grant'
            confidence = behavioral_score
            explanation['decision_reason'] = 'behavior_matches_pattern'
        
        elif behavioral_score >= 0.3:
            # Medium probability - somewhat unusual but not clearly anomalous
            decision = 'uncertain'
            confidence = 0.5
            explanation['decision_reason'] = 'behavior_moderately_unusual'
        
        else:
            # Low probability - significantly deviates from normal pattern
            decision = 'deny'
            confidence = 1.0 - behavioral_score
            explanation['decision_reason'] = 'behavior_anomalous'
        
        # Update agent history
        self._update_agent_history(agent_id, current_state)
        
        return (decision, confidence, explanation)
    
    def _update_agent_history(self, agent_id: str, state: str):
        """
        Update the recent history for an agent.
        
        Args:
            agent_id: Agent identifier
            state: New state to add to history
        """
        history = self.agent_histories[agent_id]
        history.append(state)
        
        # Limit history length
        if len(history) > self.max_history_length:
            history.popleft()
    
    def get_agent_statistics(self, agent_id: str) -> Dict:
        """
        Get statistics about an agent's behavioral model.
        
        Args:
            agent_id: Agent identifier
        
        Returns:
            Dictionary with statistics
        """
        if agent_id not in self.agent_models:
            return {'error': 'no_model'}
        
        model = self.agent_models[agent_id]
        history = self.agent_histories[agent_id]
        
        return {
            'num_states': model.get_num_states(),
            'num_transitions': model.get_num_transitions(),
            'history_length': len(history),
            'most_common_states': model.get_most_common_states(5),
            'model_order': self.order
        }
    
    def save_models(self, filepath: str) -> bool:
        """
        Save trained models to file.
        
        Args:
            filepath: Path to save models
        
        Returns:
            True if successful
        """
        try:
            models_data = {
                'order': self.order,
                'smoothing': self.smoothing,
                'models': {}
            }
            
            for agent_id, model in self.agent_models.items():
                models_data['models'][agent_id] = model.to_dict()
            
            with open(filepath, 'w') as f:
                json.dump(models_data, f, indent=2)
            
            logger.info(f"Saved {len(self.agent_models)} models to {filepath}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            return False
    
    def load_models(self, filepath: str) -> bool:
        """
        Load trained models from file.
        
        Args:
            filepath: Path to load models from
        
        Returns:
            True if successful
        """
        try:
            with open(filepath, 'r') as f:
                models_data = json.load(f)
            
            self.order = models_data['order']
            self.smoothing = models_data['smoothing']
            
            for agent_id, model_dict in models_data['models'].items():
                model = MarkovModel(order=self.order, smoothing=self.smoothing)
                model.from_dict(model_dict)
                self.agent_models[agent_id] = model
            
            logger.info(f"Loaded {len(self.agent_models)} models from {filepath}")
            return True
        
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False


class MarkovModel:
    """
    N-order Markov Chain model for sequence prediction.
    """
    
    def __init__(self, order: int = 2, smoothing: str = 'laplace'):
        """
        Initialize Markov model.
        
        Args:
            order: Order of the Markov chain
            smoothing: Smoothing method
        """
        self.order = order
        self.smoothing = smoothing
        
        # Transition counts: {context: {next_state: count}}
        self.transitions = defaultdict(lambda: defaultdict(int))
        self.context_totals = defaultdict(int)
        self.unique_states = set()
    
    def fit(self, sequence: List[str]):
        """
        Fit the model to a sequence of states.
        
        Args:
            sequence: List of state strings
        """
        if len(sequence) < self.order + 1:
            return
        
        # Build transition counts
        for i in range(len(sequence) - self.order):
            # Context: previous 'order' states
            context = tuple(sequence[i:i + self.order])
            next_state = sequence[i + self.order]
            
            self.transitions[context][next_state] += 1
            self.context_totals[context] += 1
            self.unique_states.add(next_state)
    
    def get_transition_probability(self, context: Tuple[str, ...], 
                                   next_state: str) -> float:
        """
        Get probability of transitioning to next_state given context.
        
        Args:
            context: Tuple of previous states
            next_state: Next state to predict
        
        Returns:
            Probability (0.0 to 1.0)
        """
        if context not in self.transitions:
            # Unseen context - return uniform probability
            return 1.0 / (len(self.unique_states) + 1)
        
        count = self.transitions[context][next_state]
        total = self.context_totals[context]
        
        if self.smoothing == 'laplace':
            # Laplace (add-1) smoothing
            num_states = len(self.unique_states)
            probability = (count + 1) / (total + num_states)
        else:
            # No smoothing
            probability = count / total if total > 0 else 0.0
        
        return probability
    
    def get_num_states(self) -> int:
        """Get number of unique states."""
        return len(self.unique_states)
    
    def get_num_transitions(self) -> int:
        """Get total number of transitions."""
        return sum(self.context_totals.values())
    
    def get_most_common_states(self, n: int = 5) -> List[Tuple[str, int]]:
        """
        Get most common states.
        
        Args:
            n: Number of top states to return
        
        Returns:
            List of (state, frequency) tuples
        """
        state_counts = defaultdict(int)
        
        for context_transitions in self.transitions.values():
            for state, count in context_transitions.items():
                state_counts[state] += count
        
        return sorted(state_counts.items(), key=lambda x: x[1], reverse=True)[:n]
    
    def to_dict(self) -> Dict:
        """Convert model to dictionary for serialization."""
        return {
            'order': self.order,
            'smoothing': self.smoothing,
            'transitions': {
                str(k): dict(v) for k, v in self.transitions.items()
            },
            'context_totals': dict(self.context_totals),
            'unique_states': list(self.unique_states)
        }
    
    def from_dict(self, data: Dict):
        """Load model from dictionary."""
        self.order = data['order']
        self.smoothing = data['smoothing']
        
        # Reconstruct transitions
        self.transitions = defaultdict(lambda: defaultdict(int))
        for context_str, next_states in data['transitions'].items():
            context = eval(context_str)  # Convert string back to tuple
            for state, count in next_states.items():
                self.transitions[context][state] = count
        
        self.context_totals = defaultdict(int, data['context_totals'])
        self.unique_states = set(data['unique_states'])


if __name__ == "__main__":
    print("BehavioralAnalyzer - Layer 2")
    print("This module requires training data from bbac_ics_dataset.")
    print("Use train_models.py to train the models with real data.")
