#!/usr/bin/env python3
"""
BBAC Framework - Metrics Collector
Comprehensive metrics collection for experiments
"""

import time
from typing import Dict, List
import numpy as np


class MetricsCollector:
    """
    Collects comprehensive metrics during experiments.
    
    Metrics collected:
    - Decision latency (ms)
    - Grant/deny rates
    - Confusion matrix (TP, TN, FP, FN)
    - Layer contributions
    - Throughput (requests/second)
    - Anomaly detection rates
    """
    
    def __init__(self):
        """Initialize metrics collector."""
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.start_time = None
        self.end_time = None
        
        # Basic metrics
        self.total_requests = 0
        self.granted = 0
        self.denied = 0
        
        # Latency tracking
        self.latency_samples = []
        
        # Layer decisions
        self.layer_decisions = {
            'layer1': {'grant': 0, 'deny': 0, 'uncertain': 0},
            'layer2': {'grant': 0, 'deny': 0, 'uncertain': 0},
            'layer3': {'grant': 0, 'deny': 0, 'uncertain': 0}
        }
        
        # Confusion matrix components (if ground truth available)
        self.true_positives = 0  # Correctly granted
        self.true_negatives = 0  # Correctly denied
        self.false_positives = 0  # Incorrectly granted
        self.false_negatives = 0  # Incorrectly denied
        
        # Anomaly detection
        self.anomalies_detected = 0
        self.anomalies_missed = 0
        
        # Emergency handling
        self.emergency_accesses = 0
        self.emergency_overrides = 0
        
        # Throughput
        self.requests_per_second = []
        self.last_request_time = None
        self.request_count_window = 0
        self.window_start_time = None
    
    def start_collection(self, controller=None):
        """
        Start metrics collection.
        
        Args:
            controller: Optional BBAC controller to monitor
        """
        self.reset()
        self.start_time = time.time()
        self.window_start_time = self.start_time
    
    def record_decision(self, request: dict, decision: dict, 
                       ground_truth: str = None):
        """
        Record a single decision.
        
        Args:
            request: Access request
            decision: Decision result
            ground_truth: Optional ground truth label
        """
        self.total_requests += 1
        
        # Basic decision
        if decision['decision'] == 'grant':
            self.granted += 1
        else:
            self.denied += 1
        
        # Latency
        if 'latency_ms' in decision:
            self.latency_samples.append(decision['latency_ms'])
        
        # Layer contributions
        if 'explanation' in decision and 'layers' in decision['explanation']:
            layers = decision['explanation']['layers']
            for layer_name, layer_result in layers.items():
                if layer_name in self.layer_decisions:
                    layer_decision = layer_result.get('decision', 'uncertain')
                    self.layer_decisions[layer_name][layer_decision] += 1
        
        # Confusion matrix (if ground truth available)
        if ground_truth:
            predicted = decision['decision']
            
            if predicted == 'grant' and ground_truth == 'grant':
                self.true_positives += 1
            elif predicted == 'deny' and ground_truth == 'deny':
                self.true_negatives += 1
            elif predicted == 'grant' and ground_truth == 'deny':
                self.false_positives += 1
            elif predicted == 'deny' and ground_truth == 'grant':
                self.false_negatives += 1
        
        # Anomaly detection
        if 'anomaly_detected' in decision and decision['anomaly_detected']:
            self.anomalies_detected += 1
        
        # Emergency handling
        if request.get('context', {}).get('emergency', False):
            self.emergency_accesses += 1
            if decision['decision'] == 'grant':
                self.emergency_overrides += 1
        
        # Throughput calculation
        current_time = time.time()
        self.request_count_window += 1
        
        # Calculate throughput every second
        if current_time - self.window_start_time >= 1.0:
            throughput = self.request_count_window / (current_time - self.window_start_time)
            self.requests_per_second.append(throughput)
            self.request_count_window = 0
            self.window_start_time = current_time
    
    def stop_collection(self, controller=None) -> Dict:
        """
        Stop collection and compute final metrics.
        
        Args:
            controller: Optional BBAC controller for final stats
            
        Returns:
            Dictionary of all metrics
        """
        self.end_time = time.time()
        
        # Extract controller stats if available
        if controller:
            self.total_requests = controller.requests_processed
            self.granted = controller.decisions_granted
            self.denied = controller.decisions_denied
            
            if controller.requests_processed > 0:
                self.latency_samples = [
                    controller.total_latency_ms / controller.requests_processed
                ]
        
        return self.compute_metrics()
    
    def compute_metrics(self) -> Dict:
        """Compute all metrics."""
        duration = (self.end_time - self.start_time) if self.end_time else 0
        
        metrics = {
            # Basic metrics
            'total_requests': self.total_requests,
            'granted': self.granted,
            'denied': self.denied,
            'duration_seconds': duration,
            
            # Rates
            'grant_rate': (self.granted / self.total_requests * 100) if self.total_requests > 0 else 0,
            'deny_rate': (self.denied / self.total_requests * 100) if self.total_requests > 0 else 0,
            
            # Latency statistics
            'latency_samples': self.latency_samples,
            'avg_latency': np.mean(self.latency_samples) if self.latency_samples else 0,
            'min_latency': np.min(self.latency_samples) if self.latency_samples else 0,
            'max_latency': np.max(self.latency_samples) if self.latency_samples else 0,
            'std_latency': np.std(self.latency_samples) if self.latency_samples else 0,
            'p50_latency': np.percentile(self.latency_samples, 50) if self.latency_samples else 0,
            'p95_latency': np.percentile(self.latency_samples, 95) if self.latency_samples else 0,
            'p99_latency': np.percentile(self.latency_samples, 99) if self.latency_samples else 0,
            
            # Decision distribution
            'decisions': {
                'grant': self.granted,
                'deny': self.denied
            },
            
            # Layer contributions
            'layer_decisions': self.layer_decisions,
            
            # Confusion matrix
            'confusion_matrix': {
                'true_positives': self.true_positives,
                'true_negatives': self.true_negatives,
                'false_positives': self.false_positives,
                'false_negatives': self.false_negatives
            },
            
            # Derived metrics
            'accuracy': self._compute_accuracy(),
            'precision': self._compute_precision(),
            'recall': self._compute_recall(),
            'f1_score': self._compute_f1(),
            'false_positive_rate': self._compute_fpr(),
            'false_negative_rate': self._compute_fnr(),
            
            # Anomaly detection
            'anomalies_detected': self.anomalies_detected,
            'anomalies_missed': self.anomalies_missed,
            
            # Emergency handling
            'emergency_accesses': self.emergency_accesses,
            'emergency_overrides': self.emergency_overrides,
            
            # Throughput
            'avg_throughput': np.mean(self.requests_per_second) if self.requests_per_second else 0,
            'max_throughput': np.max(self.requests_per_second) if self.requests_per_second else 0,
            'throughput_samples': self.requests_per_second
        }
        
        return metrics
    
    def _compute_accuracy(self) -> float:
        """Compute accuracy."""
        total = self.true_positives + self.true_negatives + \
                self.false_positives + self.false_negatives
        
        if total == 0:
            return 0.0
        
        return (self.true_positives + self.true_negatives) / total * 100
    
    def _compute_precision(self) -> float:
        """Compute precision."""
        predicted_positive = self.true_positives + self.false_positives
        
        if predicted_positive == 0:
            return 0.0
        
        return self.true_positives / predicted_positive * 100
    
    def _compute_recall(self) -> float:
        """Compute recall (sensitivity)."""
        actual_positive = self.true_positives + self.false_negatives
        
        if actual_positive == 0:
            return 0.0
        
        return self.true_positives / actual_positive * 100
    
    def _compute_f1(self) -> float:
        """Compute F1 score."""
        precision = self._compute_precision()
        recall = self._compute_recall()
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def _compute_fpr(self) -> float:
        """Compute false positive rate."""
        actual_negative = self.true_negatives + self.false_positives
        
        if actual_negative == 0:
            return 0.0
        
        return self.false_positives / actual_negative * 100
    
    def _compute_fnr(self) -> float:
        """Compute false negative rate."""
        actual_positive = self.true_positives + self.false_negatives
        
        if actual_positive == 0:
            return 0.0
        
        return self.false_negatives / actual_positive * 100
    
    def print_summary(self, metrics: Dict = None):
        """Print metrics summary."""
        if metrics is None:
            metrics = self.compute_metrics()
        
        print("\n" + "="*70)
        print("METRICS SUMMARY")
        print("="*70)
        
        print("\nBasic Statistics:")
        print(f"  Total Requests: {metrics['total_requests']}")
        print(f"  Granted: {metrics['granted']} ({metrics['grant_rate']:.1f}%)")
        print(f"  Denied: {metrics['denied']} ({metrics['deny_rate']:.1f}%)")
        print(f"  Duration: {metrics['duration_seconds']:.1f}s")
        
        print("\nLatency Statistics:")
        print(f"  Average: {metrics['avg_latency']:.2f}ms")
        print(f"  Min: {metrics['min_latency']:.2f}ms")
        print(f"  Max: {metrics['max_latency']:.2f}ms")
        print(f"  Std Dev: {metrics['std_latency']:.2f}ms")
        print(f"  P50: {metrics['p50_latency']:.2f}ms")
        print(f"  P95: {metrics['p95_latency']:.2f}ms")
        print(f"  P99: {metrics['p99_latency']:.2f}ms")
        
        if metrics['accuracy'] > 0:
            print("\nClassification Metrics:")
            print(f"  Accuracy: {metrics['accuracy']:.1f}%")
            print(f"  Precision: {metrics['precision']:.1f}%")
            print(f"  Recall: {metrics['recall']:.1f}%")
            print(f"  F1 Score: {metrics['f1_score']:.1f}%")
            print(f"  False Positive Rate: {metrics['false_positive_rate']:.1f}%")
            print(f"  False Negative Rate: {metrics['false_negative_rate']:.1f}%")
        
        if metrics['anomalies_detected'] > 0:
            print("\nAnomaly Detection:")
            print(f"  Detected: {metrics['anomalies_detected']}")
            print(f"  Missed: {metrics['anomalies_missed']}")
        
        if metrics['emergency_accesses'] > 0:
            print("\nEmergency Handling:")
            print(f"  Emergency Accesses: {metrics['emergency_accesses']}")
            print(f"  Emergency Overrides: {metrics['emergency_overrides']}")
        
        print("\nThroughput:")
        print(f"  Average: {metrics['avg_throughput']:.2f} req/s")
        print(f"  Maximum: {metrics['max_throughput']:.2f} req/s")
        
        print("="*70 + "\n")


def main():
    """Demo metrics collector."""
    collector = MetricsCollector()
    collector.start_collection()
    
    # Simulate some decisions
    for i in range(100):
        request = {'agent_id': f'agent_{i%5}', 'action': 'read'}
        decision = {
            'decision': 'grant' if i % 10 != 0 else 'deny',
            'latency_ms': 1.5 + (i % 5) * 0.3,
            'confidence': 0.95
        }
        ground_truth = 'grant' if i % 10 != 0 else 'deny'
        
        collector.record_decision(request, decision, ground_truth)
        time.sleep(0.01)
    
    metrics = collector.stop_collection()
    collector.print_summary(metrics)


if __name__ == '__main__':
    main()
