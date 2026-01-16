"""
BBAC Framework - Ablation Study
Tests individual layer contributions
"""

# 1. Biblioteca padrão
import json
import os
from datetime import datetime
from pathlib import Path

# 2. Bibliotecas de terceiros
from experiments.run import ExperimentRunner
from experiments.scenarios import BASELINE_SCENARIOS

class AblationStudy:
    """
    Runs ablation study to measure individual layer contributions.
    
    Tests:
    1. Layer 1 only (Rules)
    2. Layer 1 + Layer 2 (Rules + Behavioral)
    3. Layer 1 + Layer 3 (Rules + ML)
    4. Full BBAC (All layers)
    """
    
    def __init__(self, results_base_dir='results/ablation'):
        """Initialize ablation study."""
        self.results_base_dir = results_base_dir
        self.results_dir = None
        self.runner = ExperimentRunner(results_base_dir)
        
        # Create results directory
        self._setup_results_dir()
        
        # Configurations to test
        self.configurations = [
            {
                'name': 'L1_only',
                'description': 'Layer 1 only (Rules)',
                'layers': {
                    'rule_engine': True,
                    'behavioral_analysis': False,
                    'ml_detection': False
                }
            },
            {
                'name': 'L1_L2',
                'description': 'Layer 1 + Layer 2 (Rules + Behavioral)',
                'layers': {
                    'rule_engine': True,
                    'behavioral_analysis': True,
                    'ml_detection': False
                }
            },
            {
                'name': 'L1_L3',
                'description': 'Layer 1 + Layer 3 (Rules + ML)',
                'layers': {
                    'rule_engine': True,
                    'behavioral_analysis': False,
                    'ml_detection': True
                }
            },
            {
                'name': 'Full_BBAC',
                'description': 'Full BBAC (All layers)',
                'layers': {
                    'rule_engine': True,
                    'behavioral_analysis': True,
                    'ml_detection': True
                }
            }
        ]
    
    def _setup_results_dir(self):
        """Create timestamped results directory."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir = os.path.join(self.results_base_dir, f'ablation_{timestamp}')
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)
    
    def run_study(self, scenario: str = 'normal_operation', duration: int = 45):
        """
        Run ablation study.
        
        Args:
            scenario: Scenario to test
            duration: Duration per configuration (seconds)
        """
        print("\n" + "="*70)
        print("ABLATION STUDY - Layer Contribution Analysis")
        print("="*70)
        print(f"Scenario: {scenario}")
        print(f"Duration per config: {duration}s")
        print(f"Total duration: ~{duration * len(self.configurations) // 60} minutes")
        print(f"Results: {self.results_dir}")
        print("="*70 + "\n")
        
        all_results = {}
        
        for i, config in enumerate(self.configurations, 1):
            print(f"\n[Configuration {i}/{len(self.configurations)}]")
            print(f"Testing: {config['description']}")
            print("-"*70)
            
            # Create configuration for this test
            test_config = {
                'name': config['name'],
                'description': config['description'],
                'layers_enabled': config['layers'],
                'agents': {'robots': 3, 'humans': 2},
                'modifications': {}
            }
            
            # Run experiment
            self.runner.results_dir = os.path.join(self.results_dir, config['name'])
            Path(self.runner.results_dir).mkdir(parents=True, exist_ok=True)
            
            self.runner.run_scenario(scenario, duration=duration, config=test_config)
            
            # Load results
            results_file = os.path.join(
                self.runner.results_dir,
                'metrics',
                f'{scenario}_results.json'
            )
            
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    results = json.load(f)
                    all_results[config['name']] = results['metrics']
            
            print()
        
        # Compare results
        print("\n" + "="*70)
        print("ABLATION STUDY RESULTS - COMPARISON")
        print("="*70)
        
        self._compare_configurations(all_results)
        self._save_comparison(all_results, scenario)
        self._generate_comparison_plots(all_results)
        
        print(f"\n✓ Ablation study completed!")
        print(f"Results saved in: {self.results_dir}")
        print("="*70 + "\n")
    
    def _compare_configurations(self, results: dict):
        """Print comparison table."""
        print("\nPerformance Comparison:")
        print("-"*70)
        
        # Header
        print(f"{'Configuration':<20} {'Grant%':<10} {'Avg Lat':<12} {'P95 Lat':<12} {'Accuracy':<10}")
        print("-"*70)
        
        # Rows
        for config_name, metrics in results.items():
            grant_rate = metrics.get('grant_rate', 0)
            avg_lat = metrics.get('avg_latency', 0)
            p95_lat = metrics.get('p95_latency', 0)
            accuracy = metrics.get('accuracy', 0)
            
            config_display = config_name.replace('_', ' ')
            print(f"{config_display:<20} {grant_rate:<10.1f} {avg_lat:<12.2f} {p95_lat:<12.2f} {accuracy:<10.1f}")
        
        print("-"*70)
        
        # Layer contributions
        print("\nLayer Contribution Analysis:")
        print("-"*70)
        
        if 'L1_only' in results and 'Full_BBAC' in results:
            baseline = results['L1_only']
            full = results['Full_BBAC']
            
            lat_improvement = ((baseline['avg_latency'] - full['avg_latency']) / 
                             baseline['avg_latency'] * 100) if baseline['avg_latency'] > 0 else 0
            
            acc_improvement = full.get('accuracy', 0) - baseline.get('accuracy', 0)
            
            print(f"Latency improvement (L1 → Full): {lat_improvement:+.1f}%")
            print(f"Accuracy improvement (L1 → Full): {acc_improvement:+.1f}%")
            
            # L2 contribution
            if 'L1_L2' in results:
                l1_l2 = results['L1_L2']
                l2_contrib = l1_l2.get('accuracy', 0) - baseline.get('accuracy', 0)
                print(f"Layer 2 contribution: {l2_contrib:+.1f}% accuracy")
            
            # L3 contribution
            if 'L1_L3' in results:
                l1_l3 = results['L1_L3']
                l3_contrib = l1_l3.get('accuracy', 0) - baseline.get('accuracy', 0)
                print(f"Layer 3 contribution: {l3_contrib:+.1f}% accuracy")
        
        print("-"*70)
    
    def _save_comparison(self, results: dict, scenario: str):
        """Save comparison results."""
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'scenario': scenario,
            'configurations': list(results.keys()),
            'results': results,
            'summary': self._compute_summary(results)
        }
        
        filename = os.path.join(self.results_dir, 'ablation_comparison.json')
        with open(filename, 'w') as f:
            json.dump(comparison, f, indent=2)
    
    def _compute_summary(self, results: dict) -> dict:
        """Compute summary statistics."""
        summary = {}
        
        for config_name, metrics in results.items():
            summary[config_name] = {
                'grant_rate': metrics.get('grant_rate', 0),
                'avg_latency': metrics.get('avg_latency', 0),
                'accuracy': metrics.get('accuracy', 0),
                'f1_score': metrics.get('f1_score', 0)
            }
        
        return summary
    
    def _generate_comparison_plots(self, results: dict):
        """Generate comparison plots."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            configs = list(results.keys())
            config_labels = [c.replace('_', '\n') for c in configs]
            
            # Extract metrics
            grant_rates = [results[c].get('grant_rate', 0) for c in configs]
            avg_latencies = [results[c].get('avg_latency', 0) for c in configs]
            accuracies = [results[c].get('accuracy', 0) for c in configs]
            
            # Create figure with subplots
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Grant rate comparison
            axes[0].bar(config_labels, grant_rates, color='skyblue', edgecolor='black')
            axes[0].set_ylabel('Grant Rate (%)')
            axes[0].set_title('Grant Rate by Configuration')
            axes[0].set_ylim(0, 100)
            axes[0].grid(True, alpha=0.3, axis='y')
            
            # Latency comparison
            axes[1].bar(config_labels, avg_latencies, color='lightcoral', edgecolor='black')
            axes[1].set_ylabel('Average Latency (ms)')
            axes[1].set_title('Latency by Configuration')
            axes[1].axhline(y=100, color='r', linestyle='--', label='Target (100ms)')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3, axis='y')
            
            # Accuracy comparison (if available)
            if any(accuracies):
                axes[2].bar(config_labels, accuracies, color='lightgreen', edgecolor='black')
                axes[2].set_ylabel('Accuracy (%)')
                axes[2].set_title('Accuracy by Configuration')
                axes[2].set_ylim(0, 100)
                axes[2].grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            
            filename = os.path.join(self.results_dir, 'ablation_comparison.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"\n✓ Comparison plot saved: {filename}")
            
        except Exception as e:
            print(f"\n⚠️  Could not generate comparison plots: {e}")


def main():
    """Main function for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='BBAC Ablation Study')
    parser.add_argument('--scenario', type=str, default='normal_operation',
                       help='Scenario to test')
    parser.add_argument('--duration', type=int, default=45,
                       help='Duration per configuration (seconds)')
    
    args = parser.parse_args()
    
    study = AblationStudy()
    study.run_study(scenario=args.scenario, duration=args.duration)


if __name__ == '__main__':
    main()
