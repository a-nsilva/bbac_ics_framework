#!/usr/bin/env python3
"""
BBAC Framework - Main Entry Point
Interactive CLI for running experiments
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.experiments.run import ExperimentRunner
from src.experiments.ablation import AblationStudy
from src.experiments.baseline_comparison import BaselineComparison


def print_banner():
    """Print framework banner."""
    print("\n" + "="*70)
    print("BBAC FRAMEWORK - Behavior-Based Access Control")
    print("Experiment Suite for Industrial Multi-Agent Systems")
    print("="*70 + "\n")


def print_menu():
    """Print main menu."""
    print("\n" + "-"*70)
    print("MAIN MENU")
    print("-"*70)
    print("[1] Run Full Experiment")
    print("[2] Ablation Study (L1, L1+L2, L1+L3, Full)")
    print("[3] Baseline Comparison")
    print("[4] Custom Scenario")
    print("[0] Exit")
    print("-"*70)


def run_full_experiment():
    """Run complete experiment suite."""
    print("\n" + "="*70)
    print("FULL EXPERIMENT - All Scenarios")
    print("="*70)
    
    runner = ExperimentRunner()
    
    print("\nThis will run:")
    print("  • Normal operation scenario")
    print("  • Anomaly detection scenario")
    print("  • Emergency response scenario")
    print("  • Adversarial attack scenario")
    print("\nDuration: ~5 minutes")
    
    confirm = input("\nProceed? [Y/n]: ").strip().lower()
    if confirm in ['', 'y', 'yes']:
        runner.run_all_scenarios()
        print("\n✓ Full experiment completed!")
        print(f"Results saved in: {runner.results_dir}")
    else:
        print("Cancelled.")


def run_ablation_study():
    """Run ablation study."""
    print("\n" + "="*70)
    print("ABLATION STUDY - Layer Contribution Analysis")
    print("="*70)
    
    ablation = AblationStudy()
    
    print("\nConfigurations to test:")
    print("  [1] Layer 1 only (Rules)")
    print("  [2] Layer 1 + Layer 2 (Rules + Behavioral)")
    print("  [3] Layer 1 + Layer 3 (Rules + ML)")
    print("  [4] Full BBAC (All layers)")
    
    print("\nDuration: ~3 minutes")
    
    confirm = input("\nProceed? [Y/n]: ").strip().lower()
    if confirm in ['', 'y', 'yes']:
        ablation.run_study()
        print("\n✓ Ablation study completed!")
        print(f"Results saved in: {ablation.results_dir}")
    else:
        print("Cancelled.")


def run_baseline_comparison():
    """Run baseline comparison."""
    print("\n" + "="*70)
    print("BASELINE COMPARISON")
    print("="*70)
    
    comparison = BaselineComparison()
    
    print("\nComparing:")
    print("  • Rule-based only")
    print("  • Behavioral analysis only")
    print("  • ML detection only")
    print("  • BBAC hybrid (proposed)")
    
    print("\nDuration: ~4 minutes")
    
    confirm = input("\nProceed? [Y/n]: ").strip().lower()
    if confirm in ['', 'y', 'yes']:
        comparison.run_comparison()
        print("\n✓ Baseline comparison completed!")
        print(f"Results saved in: {comparison.results_dir}")
    else:
        print("Cancelled.")


def run_custom_scenario():
    """Run custom scenario."""
    print("\n" + "="*70)
    print("CUSTOM SCENARIO")
    print("="*70)
    
    runner = ExperimentRunner()
    
    print("\nAvailable scenarios:")
    print("  [1] Normal operation")
    print("  [2] Anomaly detection")
    print("  [3] Emergency response")
    print("  [4] Adversarial attack")
    
    choice = input("\nSelect scenario [1-4]: ").strip()
    
    scenarios = {
        '1': 'normal_operation',
        '2': 'anomaly_detection',
        '3': 'emergency_response',
        '4': 'adversarial_attack'
    }
    
    if choice in scenarios:
        duration = input("Duration in seconds [default: 60]: ").strip()
        duration = int(duration) if duration else 60
        
        print(f"\nRunning {scenarios[choice]} for {duration}s...")
        runner.run_scenario(scenarios[choice], duration=duration)
        print("\n✓ Scenario completed!")
        print(f"Results saved in: {runner.results_dir}")
    else:
        print("Invalid choice.")


def main():
    """Main function."""
    print_banner()
    
    # Check ROS2 environment
    if 'ROS_DISTRO' not in os.environ:
        print("⚠️  WARNING: ROS2 environment not sourced!")
        print("Run: source /opt/ros/humble/setup.bash\n")
        response = input("Continue anyway? [y/N]: ").strip().lower()
        if response not in ['y', 'yes']:
            print("Exiting.")
            return
    
    while True:
        print_menu()
        
        try:
            choice = input("\nSelect option: ").strip()
            
            if choice == '0':
                print("\nExiting BBAC Framework. Goodbye!")
                break
            elif choice == '1':
                run_full_experiment()
            elif choice == '2':
                run_ablation_study()
            elif choice == '3':
                run_baseline_comparison()
            elif choice == '4':
                run_custom_scenario()
            else:
                print("Invalid option. Please select 0-4.")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Exiting.")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            print("\nReturning to menu...")


if __name__ == '__main__':
    main()
