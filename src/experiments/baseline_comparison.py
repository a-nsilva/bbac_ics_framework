#!/usr/bin/env python3
"""
BBAC Framework - Baseline Comparison
Compares BBAC against traditional access control methods
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run import ExperimentRunner


class BaselineComparison:
    """
    Compares BBAC framework against baseline methods.
    
    Baselines:
    1. Rule-based only (traditional RBAC)
    2. Behavioral-based only
    3. ML-based only
    4. BBAC Hybrid (proposed)
    """
    
    def __init__(self, results_base_dir='results/baseline'):
        """Initialize baseline comparison."""
        self.results_base_dir = results_base_dir
        self.results_dir = None
        self.runner = ExperimentRunner(results_base_dir)
        
        # Create results directory
        self._setup_results_dir()
        
        # Baseline configurations
        self.baselines = [
            {
                'name': 'Rule_Based',
                'description': 'Traditional Rule-based (RBAC)',
                'layers': {
                    'rule_engine': True,
                    'behavioral_analysis': False,
                    'ml_detection': False
                }
            },
            {
                'name': 'Behavioral_Based',
                'description': 'Behavioral Analysis Only',
                'layers': {
                    'rule_engine': False,
                    'behavioral_analysis': True,
                    'ml_detection': False
                }
            },
            {
                'name': 'ML_Based',
                'description': 'ML Anomaly Detection Only',
                'layers': {
                    'rule_engine': False,
                    'behavioral_analysis': False,
                    'ml_detection': True
                }
            },
            {
                'name': 'BBAC_Hybrid',
                'description': 'BBAC Hybrid (Proposed)',
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
        self.results_dir = os.path.join(self.results_base_dir, f'comparison_{timestamp}')
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)
    
    def run_comparison(self, scenarios: list = None, duration: int = 60):
        """
        Run baseline comparison.
        
        Args:
            scenarios: List of scenarios to test (default: all)
            duration: Duration per test (seconds)
        """
        if scenarios is None:
            scenarios = ['normal_operation', 'anomaly_detection']
        
        print("\n" + "="*70)
        print("BASELINE COMPARISON")
        print("="*70)
        print(f"Scenarios: {scenarios}")
        print(f"Baselines: {len(self.baselines)}")
        print(f"Duration per test: {duration}s")
        print(f"Total duration: ~{duration * len(self.baselines) * len(scenarios) // 60} minutes")
        print(f"Results: {self.results_dir}")
        print("="*70 + "\n")
        
        all_results = {}
        
        for scenario in scenarios:
            print(f"\n{'='*70}")
            print(f"SCENARIO: {scenario.upper()}")
            print(f"{'='*70}\n")
            
            scenario_results = {}
            
            for i, baseline in enumerate(self.baselines, 1):
                print(f"\n[Baseline {i}/{len(self.baselines)}]")
                print(f"Testing: {baseline['description']}")
                print("-"*70)
                
                # Create configuration
                test_config = {
                    'name': baseline['name'],
                    'description': baseline['description'],
                    'layers_enabled': baseline['layers'],
                    'agents': {'robots': 3, 'humans': 2},
                    'modifications': {}
                }
                
                # Run experiment
                self.runner.results_dir = os.path.join(
                    self.results_dir, 
                    scenario,
                    baseline['name']
                )
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
                        scenario_results[baseline['name']] = results['metrics']
                
                print()
            
            all_results[scenario] = scenario_results
        
        # Compare results
        print("\n" + "="*70)
        print("BASELINE COMPARISON RESULTS")
        print("="*70)
        
        for scenario, results in all_results.items():
            print(f"\n[{scenario.upper()}]")
            self._compare_baselines(results)
        
        self._save_comparison(all_results)
        self._generate_comparison_plots(all_results)
        
        print(f"\n✓ Baseline comparison completed!")
        print(f"Results saved in: {self.results_dir}")
        print("="*70 + "\n")
    
    def _compare_baselines(self, results: dict):
        """Print comparison table."""
        print("\nPerformance Comparison:")
        print("-"*70)
        
        # Header
        print(f"{'Method':<25} {'Grant%':<10} {'Lat(ms)':<10} {'Acc%':<10} {'F1':<10}")
        print("-"*70)
        
        # Rows
        for baseline_name, metrics in results.items():
            grant_rate = metrics.get('grant_rate', 0)
            avg_lat = metrics.get('avg_latency', 0)
            accuracy = metrics.get('accuracy', 0)
            f1 = metrics.get('f1_score', 0)
            
            method_display = baseline_name.replace('_', ' ')
            print(f"{method_display:<25} {grant_rate:<10.1f} {avg_lat:<10.2f} {accuracy:<10.1f} {f1:<10.1f}")
        
        print("-"*70)
        
        # Highlight best
        if 'BBAC_Hybrid' in results:
            bbac = results['BBAC_Hybrid']
            print(f"\nBBAC Hybrid Performance:")
            print(f"  Grant Rate: {bbac.get('grant_rate', 0):.1f}%")
            print(f"  Avg Latency: {bbac.get('avg_latency', 0):.2f}ms")
            print(f"  Accuracy: {bbac.get('accuracy', 0):.1f}%")
            print(f"  F1 Score: {bbac.get('f1_score', 0):.1f}%")
            
            # Compare with best baseline
            best_baseline = self._find_best_baseline(results)
            if best_baseline and best_baseline != 'BBAC_Hybrid':
                best = results[best_baseline]
                
                lat_improvement = ((best['avg_latency'] - bbac['avg_latency']) / 
                                 best['avg_latency'] * 100) if best['avg_latency'] > 0 else 0
                acc_improvement = bbac.get('accuracy', 0) - best.get('accuracy', 0)
                
                print(f"\nImprovement over {best_baseline.replace('_', ' ')}:")
                print(f"  Latency: {lat_improvement:+.1f}%")
                print(f"  Accuracy: {acc_improvement:+.1f}%")
        
        print("-"*70)
    
    def _find_best_baseline(self, results: dict) -> str:
        """Find best performing baseline (excluding BBAC)."""
        best_name = None
        best_score = -1
        
        for baseline_name, metrics in results.items():
            if baseline_name == 'BBAC_Hybrid':
                continue
            
            # Composite score: accuracy + (1 / latency)
            accuracy = metrics.get('accuracy', 0)
            latency = metrics.get('avg_latency', 1)
            score = accuracy + (100 / latency if latency > 0 else 0)
            
            if score > best_score:
                best_score = score
                best_name = baseline_name
        
        return best_name
    
    def _save_comparison(self, results: dict):
        """Save comparison results."""
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'baselines': [b['name'] for b in self.baselines],
            'results': results,
            'summary': self._compute_summary(results)
        }
        
        filename = os.path.join(self.results_dir, 'baseline_comparison.json')
        with open(filename, 'w') as f:
            json.dump(comparison, f, indent=2)
    
    def _compute_summary(self, results: dict) -> dict:
        """Compute summary statistics."""
        summary = {}
        
        for scenario, scenario_results in results.items():
            summary[scenario] = {}
            for baseline_name, metrics in scenario_results.items():
                summary[scenario][baseline_name] = {
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
            
            # Create separate plots for each scenario
            for scenario, scenario_results in results.items():
                baselines = list(scenario_results.keys())
                baseline_labels = [b.replace('_', '\n') for b in baselines]
                
                # Extract metrics
                grant_rates = [scenario_results[b].get('grant_rate', 0) for b in baselines]
                avg_latencies = [scenario_results[b].get('avg_latency', 0) for b in baselines]
                accuracies = [scenario_results[b].get('accuracy', 0) for b in baselines]
                f1_scores = [scenario_results[b].get('f1_score', 0) for b in baselines]
                
                # Create figure
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                fig.suptitle(f'Baseline Comparison - {scenario.replace("_", " ").title()}', 
                           fontsize=16, fontweight='bold')
                
                # Colors: BBAC in green, others in blue
                colors = ['green' if 'BBAC' in b else 'skyblue' for b in baselines]
                
                # Grant rate
                axes[0, 0].bar(baseline_labels, grant_rates, color=colors, edgecolor='black')
                axes[0, 0].set_ylabel('Grant Rate (%)')
                axes[0, 0].set_title('Grant Rate')
                axes[0, 0].set_ylim(0, 100)
                axes[0, 0].grid(True, alpha=0.3, axis='y')
                
                # Latency
                axes[0, 1].bar(baseline_labels, avg_latencies, color=colors, edgecolor='black')
                axes[0, 1].set_ylabel('Latency (ms)')
                axes[0, 1].set_title('Average Latency')
                axes[0, 1].axhline(y=100, color='r', linestyle='--', label='Target')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3, axis='y')
                
                # Accuracy
                if any(accuracies):
                    axes[1, 0].bar(baseline_labels, accuracies, color=colors, edgecolor='black')
                    axes[1, 0].set_ylabel('Accuracy (%)')
                    axes[1, 0].set_title('Accuracy')
                    axes[1, 0].set_ylim(0, 100)
                    axes[1, 0].grid(True, alpha=0.3, axis='y')
                
                # F1 Score
                if any(f1_scores):
                    axes[1, 1].bar(baseline_labels, f1_scores, color=colors, edgecolor='black')
                    axes[1, 1].set_ylabel('F1 Score (%)')
                    axes[1, 1].set_title('F1 Score')
                    axes[1, 1].set_ylim(0, 100)
                    axes[1, 1].grid(True, alpha=0.3, axis='y')
                
                plt.tight_layout()
                
                filename = os.path.join(
                    self.results_dir, 
                    f'baseline_comparison_{scenario}.png'
                )
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"✓ Comparison plot saved: {filename}")
            
        except Exception as e:
            print(f"\n⚠️  Could not generate comparison plots: {e}")


def main():
    """Main function for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='BBAC Baseline Comparison')
    parser.add_argument('--scenarios', nargs='+', 
                       default=['normal_operation', 'anomaly_detection'],
                       help='Scenarios to test')
    parser.add_argument('--duration', type=int, default=60,
                       help='Duration per test (seconds)')
    
    args = parser.parse_args()
    
    comparison = BaselineComparison()
    comparison.run_comparison(scenarios=args.scenarios, duration=args.duration)


if __name__ == '__main__':
    main()
